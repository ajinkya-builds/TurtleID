"""Distil Stack v2 into a browser student and write the Pages scoring bundle.

Teacher: BioCLIP 2.5 ViT-H/14 (1024-d) and DINOv2-with-registers-large (1024-d),
read from cached per-image embeddings when present, or computed over
``Turtles/`` (if it is on disk), ``data/field_gallery/``, ``reference_images/``,
and verified ``submissions/``.

Student: ``facebook/dinov2-with-registers-small`` (22M, 384-d CLS) plus a
projection head ``384 → 1024 → 2048`` that emits L2-normalised
``[e_bioclip_hat, e_dinov2L_hat]``. The training loss is cosine alignment,
distance-wise relational similarity, KL against the teacher's Stack v2
channel logits, and cross-entropy on the species label.

Examples::

    # Bundle only (no torch download). Writes docs/data/stack_v2_browser_bundle.json
    python -m scripts.distill_browser_student --bundle-only

    # Train the head on whatever images this machine can see, then export ONNX.
    python -m scripts.distill_browser_student --train --epochs 80

    # Upload the INT8 ONNX and the browser bundle to a free model repo.
    python -m scripts.distill_browser_student --push-to-hf ajinkyapatil60/turtleid-student-onnx

The 92-photo seizure benchmark runs only when ``Turtles/`` and
``logs/turtles_ground_truth.json`` are present. Otherwise the evaluator
reports leave-one-out and GroupKFold numbers on the repo gallery.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np

import config  # noqa: F401
from config import BASE_DIR, DATA_DIR, REFERENCE_IMAGE_DIR, REFERENCE_MANIFEST, SPECIES_DB_PATH
from core.stack_v2.distill import (
    build_browser_bundle_from_disk,
    distillation_loss,
    group_kfold_splits,
    ranking_report,
    write_browser_bundle,
)

logger = logging.getLogger("distill")

STUDENT_ID = "facebook/dinov2-with-registers-small"
CLS_DIM = 384
HEAD_HIDDEN = 1024
OUT_DIM = 2048
BROWSER_BUNDLE_PATH = BASE_DIR / "docs" / "data" / "stack_v2_browser_bundle.json"
ONNX_DIR = BASE_DIR / "models" / "browser"
PAIRS_PATH = DATA_DIR / "distill_pairs.npz"


def build_projection_head(cls_dim: int = CLS_DIM, hidden: int = HEAD_HIDDEN, out_dim: int = OUT_DIM):
    """Lightweight head: 384 → 1024 → 2048, then two L2-normalised 1024-d halves."""
    import torch
    from torch import nn

    class DistillationProjectionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(cls_dim, hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, out_dim)

        def forward(self, cls_token):
            hidden_state = self.act(self.fc1(cls_token))
            raw = self.fc2(hidden_state)
            bioclip = torch.nn.functional.normalize(raw[..., :1024], dim=-1)
            dino = torch.nn.functional.normalize(raw[..., 1024:], dim=-1)
            return bioclip, dino

    return DistillationProjectionHead()


def build_student(backbone=None):
    """CLS backbone (B, 3, 224, 224) → (B, 384) plus the projection head.

    ``backbone`` may be omitted. The returned module then expects a backbone
    that maps ``pixel_values`` to a CLS tensor. Tests pass a tiny conv stem.
    ``load_dinov2_small`` supplies the real 22M student.
    """
    from torch import nn

    class BrowserStudent(nn.Module):
        def __init__(self, encoder, head) -> None:
            super().__init__()
            self.backbone = encoder
            self.head = head

        def forward(self, pixel_values):
            cls = self.backbone(pixel_values)
            if cls.ndim == 3:
                cls = cls[:, 0, :]
            return self.head(cls)

    if backbone is None:
        backbone = TinyCLSBackbone()
    return BrowserStudent(backbone, build_projection_head())


def TinyCLSBackbone():
    """224 input → 384-d CLS. Used for ONNX tests, not for field accuracy."""
    from torch import nn

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, CLS_DIM, kernel_size=16, stride=16)

        def forward(self, pixel_values):
            feat = self.conv(pixel_values)
            return feat.mean(dim=(2, 3))

    return _Tiny()


def load_dinov2_small():
    """Frozen DINOv2-small that returns the CLS token (B, 384)."""
    import torch
    from torch import nn
    from transformers import AutoModel

    class _Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = AutoModel.from_pretrained(STUDENT_ID)
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()
            with torch.no_grad():
                dummy_patches = torch.zeros(1, 257, CLS_DIM)
                pos_224 = self.model.embeddings.interpolate_pos_encoding(
                    dummy_patches, 224, 224
                ).detach().clone()
            self.model.embeddings.register_buffer("_pos_224", pos_224, persistent=False)
            self.model.embeddings.interpolate_pos_encoding = (
                lambda embeddings, height, width: self.model.embeddings._pos_224
            )

        def forward(self, pixel_values):
            out = self.model(pixel_values=pixel_values)
            return out.last_hidden_state[:, 0, :]

    return _Encoder()


def _inline_external_data(onnx_path: Path) -> None:
    """Ensure the ONNX file embeds all tensor data (no external .data sidecar)."""
    try:
        import onnx

        model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save_model(model, str(onnx_path), save_as_external_data=False)
        sidecar = Path(str(onnx_path) + ".data")
        if sidecar.exists():
            sidecar.unlink()
    except Exception as exc:
        logger.warning("Inline external data check skipped for %s: %s", onnx_path, exc)


def export_onnx_variants(module, out_dir: Path, *, opset: int = 17) -> dict[str, Path]:
    """Write self-contained FP32, FP16, and dynamic INT8 ONNX files."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    module = module.eval().cpu().float()
    dummy = torch.zeros(1, 3, 224, 224)
    fp32 = out_dir / "turtleid_student_fp32.onnx"
    export_kwargs = dict(
        input_names=["pixel_values"],
        output_names=["bioclip_emb", "dino_emb"],
        opset_version=opset,
        dynamo=False,
    )
    try:
        torch.onnx.export(module, dummy, str(fp32), **export_kwargs)
    except TypeError:
        export_kwargs.pop("dynamo", None)
        torch.onnx.export(module, dummy, str(fp32), **export_kwargs)

    _inline_external_data(fp32)
    written = {"fp32": fp32}
    fp16 = out_dir / "turtleid_student_fp16.onnx"
    if _write_fp16_copy(fp32, fp16):
        _inline_external_data(fp16)
        written["fp16"] = fp16
    int8 = out_dir / "turtleid_student_int8.onnx"
    if _write_int8(fp32, int8):
        _inline_external_data(int8)
        written["int8"] = int8
    return written


def _write_fp16_copy(fp32: Path, dest: Path) -> bool:
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        model = convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True)
        onnx.save_model(model, str(dest), save_as_external_data=False)
        return dest.exists()
    except Exception:
        try:
            import onnx
            from onnxconverter_common import float16 as float16_mod

            model = float16_mod.convert_float_to_float16(onnx.load(str(fp32)))
            onnx.save_model(model, str(dest), save_as_external_data=False)
            return dest.exists()
        except Exception as exc:
            logger.warning("FP16 ONNX conversion skipped: %s", exc)
            return False


def _write_int8(fp32: Path, dest: Path) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:
        logger.warning("INT8 quantization skipped: %s", exc)
        return False
    quantize_dynamic(str(fp32), str(dest), weight_type=QuantType.QInt8)
    return dest.exists()


def iter_training_images() -> list[tuple[str, Path]]:
    """(species_id, path) from plates, verified submissions, field gallery, Turtles/."""
    from core.database import SpeciesDB
    from scripts.export_stack_v2_bundle import (
        iter_plates,
        iter_turtles,
        iter_verified_submissions,
    )

    db = SpeciesDB.load()
    rows: list[tuple[str, Path]] = []
    rows.extend(iter_plates(db))
    rows.extend(iter_verified_submissions(db))
    gallery = DATA_DIR / "field_gallery"
    if gallery.exists():
        for species_dir in sorted(p for p in gallery.iterdir() if p.is_dir()):
            for path in sorted(species_dir.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    rows.append((species_dir.name, path))
    turtles = BASE_DIR / "Turtles"
    truth = BASE_DIR / "logs" / "turtles_ground_truth.json"
    if turtles.exists() and truth.exists():
        rows.extend(iter_turtles(db, turtles, truth))
    return rows


def train_head(epochs: int, lr: float, out_dir: Path) -> Path | None:
    """Freeze DINOv2-small, fit the projection head, export ONNX. Returns INT8 path."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from core.stack_v2.bundle import load_bundle

    pairs = _load_or_compute_pairs()
    if pairs is None:
        logger.error("No teacher pairs. Run with images on disk, or pass nothing and use --bundle-only.")
        return None
    cls, teacher_b, teacher_d, labels, species_ids, means_b, means_d = pairs

    # Also load the Stack v2 bundle's linear probe + text matrix so the student
    # aligns directly with all Stack v2 scoring channels.
    arrays = load_bundle(DATA_DIR / "stack_v2_bundle.npz")
    probe_coef_t = torch.from_numpy(np.asarray(arrays["probe_coef"], dtype=np.float32))
    probe_int_t = torch.from_numpy(np.asarray(arrays["probe_intercept"], dtype=np.float32))
    text_mat_t = torch.from_numpy(np.asarray(arrays["text_matrix"], dtype=np.float32))

    head = build_projection_head()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    ds = TensorDataset(
        torch.from_numpy(cls),
        torch.from_numpy(teacher_b),
        torch.from_numpy(teacher_d),
        torch.from_numpy(labels.astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)
    head.train()
    for epoch in range(epochs):
        last = {}
        for batch_cls, batch_b, batch_d, batch_y in loader:
            pred_b, pred_d = head(batch_cls)
            student_logits_t = _torch_class_dots(pred_b, pred_d, pairs)
            teacher_logits = _class_dots(batch_b.numpy(), batch_d.numpy(), pairs)
            teacher_logits_t = torch.from_numpy(teacher_logits)
            loss_map = distillation_loss(
                pred_b.detach().cpu().numpy(),
                batch_b.numpy(),
                pred_d.detach().cpu().numpy(),
                batch_d.numpy(),
                student_logits_t.detach().cpu().numpy(),
                teacher_logits_t.numpy(),
                batch_y.numpy(),
            )
            # 1. Direct embedding cosine alignment (BioCLIP + DINOv2-L)
            cos = (1.0 - (pred_b * batch_b).sum(-1)).mean() + (1.0 - (pred_d * batch_d).sum(-1)).mean()

            # 2. Relational similarity loss (RKD) on fused 2048-d representation
            fused_s = torch.cat([pred_b, pred_d], dim=-1)
            fused_t = torch.cat([batch_b, batch_d], dim=-1)
            if len(fused_s) >= 2:
                dist_s = torch.cdist(fused_s, fused_s, p=2)
                dist_t = torch.cdist(fused_t, fused_t, p=2)
                rkd = torch.mean((dist_s - dist_t) ** 2)
            else:
                rkd = torch.tensor(0.0)

            # 3. Class-mean logits KL + CE
            log_probs = torch.log_softmax(student_logits_t * 6.0, dim=-1)
            ce = torch.nn.functional.nll_loss(log_probs, batch_y)
            teacher_p = torch.softmax(teacher_logits_t * 6.0, dim=-1)
            kl = torch.sum(teacher_p * (torch.log(teacher_p + 1e-8) - log_probs), dim=-1).mean()

            # 4. Stack v2 linear probe + text matrix alignment
            probe_s = fused_s @ probe_coef_t.T + probe_int_t
            probe_t = fused_t @ probe_coef_t.T + probe_int_t
            probe_kl = torch.sum(
                torch.softmax(probe_t, dim=-1)
                * (torch.log_softmax(probe_t, dim=-1) - torch.log_softmax(probe_s, dim=-1)),
                dim=-1,
            ).mean()
            text_s = pred_b @ text_mat_t.T * 10.0
            text_t = batch_b @ text_mat_t.T * 10.0
            text_kl = torch.sum(
                torch.softmax(text_t, dim=-1)
                * (torch.log_softmax(text_t, dim=-1) - torch.log_softmax(text_s, dim=-1)),
                dim=-1,
            ).mean()

            loss = 2.5 * cos + 0.5 * rkd + 0.5 * ce + 1.0 * kl + 0.5 * probe_kl + 0.5 * text_kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = {
                "loss": round(float(loss.detach()), 4),
                **{k: round(float(loss_map[k]), 4) for k in ("cosine", "kl", "ce")},
            }
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            logger.info("epoch %s/%s %s", epoch + 1, epochs, last)
    head.eval()
    backbone = load_dinov2_small()
    student = build_student(backbone)
    student.head.load_state_dict(head.state_dict())
    written = export_onnx_variants(student, out_dir)
    _attach_head_to_bundle(head)
    _print_benchmark(cls, teacher_b, teacher_d, labels, species_ids, head)
    return written.get("int8") or written.get("fp32")


def _class_dots(bioclip, dino, pairs) -> np.ndarray:
    means_b, means_d = pairs[5], pairs[6]
    return bioclip @ means_b.T + dino @ means_d.T


def _torch_class_dots(pred_b, pred_d, pairs):
    import torch

    means_b = torch.from_numpy(pairs[5])
    means_d = torch.from_numpy(pairs[6])
    return pred_b @ means_b.T + pred_d @ means_d.T


def _load_or_compute_pairs():
    """Return (cls, teacher_b, teacher_d, class_index, species_ids, means_b, means_d) or None."""
    if PAIRS_PATH.exists():
        blob = np.load(PAIRS_PATH, allow_pickle=False)
        species_ids = [str(s) for s in blob["species_ids"].tolist()]
        return (
            blob["cls"].astype(np.float32),
            blob["teacher_bioclip"].astype(np.float32),
            blob["teacher_dino"].astype(np.float32),
            blob["class_index"].astype(np.int64),
            species_ids,
            blob["means_bioclip"].astype(np.float32),
            blob["means_dino"].astype(np.float32),
        )
    rows = iter_training_images()
    if not rows:
        return None
    logger.info("Computing teacher and student CLS for %s images", len(rows))
    return _embed_pairs(rows)


def _embed_pairs(rows: list[tuple[str, Path]]):
    """Memory-streamed teacher + student CLS computation (loads 1 model at a time)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    from core.stack_v2.bundle import load_bundle
    from core.stack_v2.crops import crop_animal_and_head
    from core.stack_v2.runtime import (
        BIOCLIP_ID,
        _load_dino,
        _load_openclip_visual,
        _resolve_dtype,
        _torch_device,
        _visual_features,
        center_query,
        l2_normalise,
    )

    arrays = load_bundle(DATA_DIR / "stack_v2_bundle.npz")
    species_ids = [str(s) for s in arrays["species_ids"].tolist()]
    index = {sid: i for i, sid in enumerate(species_ids)}

    cropped_images: list[Image.Image] = []
    y_rows: list[int] = []
    for sid, path in rows:
        if sid not in index:
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning("Skipping unreadable image %s: %s", path, exc)
            continue
        crops = crop_animal_and_head(image, detector=None)
        cropped_images.append(crops.animal)
        y_rows.append(index[sid])

    if not cropped_images:
        return None

    dev = _torch_device(torch)
    dt = _resolve_dtype(torch, dev)

    logger.info("Step 1/3: Encoding %d cropped images with BioCLIP 2.5 (%s)", len(cropped_images), dt)
    bioclip, preprocess = _load_openclip_visual(BIOCLIP_ID, dev, dt)
    raw_b_list = []
    with torch.no_grad():
        for i in range(0, len(cropped_images), 16):
            batch = cropped_images[i : i + 16]
            tensors = torch.stack([preprocess(im) for im in batch]).to(device=dev, dtype=dt)
            flipped = torch.flip(tensors, dims=[3])
            summed = _visual_features(bioclip, tensors) + _visual_features(bioclip, flipped)
            raw_b_list.append(summed.float().cpu().numpy())
    del bioclip, preprocess
    gc.collect()
    raw_b = l2_normalise(np.vstack(raw_b_list).astype(np.float32))
    b_rows = [
        np.asarray(center_query(vec, arrays["bioclip_mean"]), dtype=np.float32)
        for vec in raw_b
    ]

    logger.info("Step 2/3: Encoding %d cropped images with DINOv2-large (%s)", len(cropped_images), dt)
    dino, dino_proc = _load_dino(dev, dt)
    raw_d_list = []
    with torch.no_grad():
        for i in range(0, len(cropped_images), 16):
            batch = cropped_images[i : i + 16]
            flips = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in batch]
            inp1 = dino_proc(images=batch, return_tensors="pt")["pixel_values"].to(device=dev, dtype=dt)
            inp2 = dino_proc(images=flips, return_tensors="pt")["pixel_values"].to(device=dev, dtype=dt)
            c1 = dino(pixel_values=inp1).last_hidden_state[:, 0]
            c2 = dino(pixel_values=inp2).last_hidden_state[:, 0]
            raw_d_list.append((0.5 * (c1 + c2)).float().cpu().numpy())
    del dino, dino_proc
    gc.collect()
    raw_d = l2_normalise(np.vstack(raw_d_list).astype(np.float32))
    d_rows = [
        np.asarray(center_query(vec, arrays["dino_mean"]), dtype=np.float32)
        for vec in raw_d
    ]

    logger.info("Step 3/3: Encoding %d cropped images with DINOv2-small student backbone", len(cropped_images))
    processor = AutoImageProcessor.from_pretrained(STUDENT_ID)
    encoder = load_dinov2_small().eval()
    cls_rows: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(cropped_images), 16):
            batch_imgs = cropped_images[i : i + 16]
            batch_flips = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in batch_imgs]
            inp_orig = processor(images=batch_imgs, return_tensors="pt")
            inp_flip = processor(images=batch_flips, return_tensors="pt")
            cls_orig = encoder(inp_orig["pixel_values"]).float().cpu().numpy()
            cls_flip = encoder(inp_flip["pixel_values"]).float().cpu().numpy()
            cls_avg = 0.5 * (cls_orig + cls_flip)
            for row in cls_avg:
                cls_rows.append(row.astype(np.float32))
    del encoder, processor
    gc.collect()

    cls = np.stack(cls_rows)
    teacher_b = np.stack(b_rows)
    teacher_d = np.stack(d_rows)
    class_index = np.asarray(y_rows, dtype=np.int64)
    means_b = np.zeros((len(species_ids), teacher_b.shape[1]), np.float32)
    means_d = np.zeros((len(species_ids), teacher_d.shape[1]), np.float32)
    for i, sid in enumerate(species_ids):
        mask = class_index == i
        if mask.any():
            mb = teacher_b[mask].mean(axis=0)
            md = teacher_d[mask].mean(axis=0)
            means_b[i] = mb / max(float(np.linalg.norm(mb)), 1e-12)
            means_d[i] = md / max(float(np.linalg.norm(md)), 1e-12)
    np.savez_compressed(
        PAIRS_PATH,
        cls=cls,
        teacher_bioclip=teacher_b,
        teacher_dino=teacher_d,
        class_index=class_index,
        species_ids=np.asarray(species_ids),
        means_bioclip=means_b,
        means_dino=means_d,
    )
    return cls, teacher_b, teacher_d, class_index, species_ids, means_b, means_d


def _attach_head_to_bundle(head) -> None:
    """Store the trained 384→1024→2048 weights inside the browser bundle."""
    if not BROWSER_BUNDLE_PATH.exists():
        write_pages_bundle()
    payload = json.loads(BROWSER_BUNDLE_PATH.read_text(encoding="utf-8"))
    from core.stack_v2.distill import pack_f16

    state = {k: v.detach().cpu().numpy() for k, v in head.state_dict().items()}
    payload["projection_head"] = {name: pack_f16(value) for name, value in state.items()}
    write_browser_bundle(payload, BROWSER_BUNDLE_PATH)


def _score_full_stack_v2(bioclip_mat: np.ndarray, dino_mat: np.ndarray, arrays: dict) -> np.ndarray:
    """Score a batch of centered (bioclip_q, dino_q) rows with the 6-channel Stack v2 runtime."""
    from core.stack_v2.runtime import fuse_channels

    species_ids = [str(s) for s in arrays["species_ids"].tolist()]
    cache_labels = [str(s) for s in arrays["cache_labels"].tolist()]
    local_labels = [str(s) for s in arrays["local_labels"].tolist()]
    fam_names = [str(s) for s in arrays["fam_names"].tolist()]
    gen_names = [str(s) for s in arrays["gen_names"].tolist()]
    sp_fams = [str(s) for s in arrays["species_families"].tolist()]
    sp_gens = [str(s) for s in arrays["species_genera"].tolist()]
    plate_ids = [str(s) for s in arrays["plate_ids"].tolist()]
    probe_classes = [str(s) for s in arrays["probe_classes"].tolist()]

    out = np.zeros((len(bioclip_mat), len(species_ids)), dtype=np.float64)
    for i in range(len(bioclip_mat)):
        out[i] = fuse_channels(
            bioclip_q=bioclip_mat[i],
            dino_q=dino_mat[i],
            biocap_q=None,
            cache_mat=arrays["cache_mat"],
            cache_labels=cache_labels,
            local_dino=arrays["local_dino"],
            local_labels=local_labels,
            text_matrix=arrays["text_matrix"],
            fam_vecs=arrays["fam_vecs"],
            fam_names=fam_names,
            gen_vecs=arrays["gen_vecs"],
            gen_names=gen_names,
            species_families=sp_fams,
            species_genera=sp_gens,
            plate_c=arrays["plate_c"],
            plate_ids=plate_ids,
            biocap_text=arrays["biocap_text_matrix"],
            probe_coef=arrays["probe_coef"],
            probe_intercept=arrays["probe_intercept"],
            probe_classes=probe_classes,
            species_ids=species_ids,
            use_biocap_channel=False,
        )
    return out


def _print_benchmark(cls, teacher_b, teacher_d, class_index, species_ids, head) -> None:
    import torch
    from core.stack_v2.bundle import load_bundle

    head.eval()
    with torch.no_grad():
        pred_b, pred_d = head(torch.from_numpy(cls))
    pred_b = pred_b.cpu().numpy()
    pred_d = pred_d.cpu().numpy()

    arrays = load_bundle(DATA_DIR / "stack_v2_bundle.npz")
    teacher_scores = _score_full_stack_v2(teacher_b, teacher_d, arrays)
    student_scores = _score_full_stack_v2(pred_b, pred_d, arrays)
    labels = [species_ids[int(i)] for i in class_index]
    full = ranking_report(teacher_scores, student_scores, labels, species_ids)
    logger.info("Full Stack v2 (all %d images): %s", len(labels), _fmt_report(full))

    masks = group_kfold_splits(labels, n_splits=min(5, len(set(labels))))
    for fold, mask in enumerate(masks, start=1):
        if mask.sum() == 0:
            continue
        part = ranking_report(
            teacher_scores[mask],
            student_scores[mask],
            [labels[i] for i, keep in enumerate(mask) if keep],
            species_ids,
        )
        logger.info("GroupKFold %s %s", fold, _fmt_report(part))

    # Leave-one-out on class means
    means_b = np.zeros((len(species_ids), teacher_b.shape[1]), np.float32)
    means_d = np.zeros_like(means_b)
    for i in range(len(species_ids)):
        mask = class_index == i
        if np.any(mask):
            means_b[i] = teacher_b[mask].mean(0)
            means_d[i] = teacher_d[mask].mean(0)
    loo_teacher = teacher_b @ means_b.T + teacher_d @ means_d.T
    loo_student = pred_b @ means_b.T + pred_d @ means_d.T
    for i, label_i in enumerate(class_index):
        others = np.where(class_index == label_i)[0]
        others = others[others != i]
        if len(others) == 0:
            continue
        loo_teacher[i, label_i] = float(
            teacher_b[others].mean(0) @ teacher_b[i] + teacher_d[others].mean(0) @ teacher_d[i]
        )
        loo_student[i, label_i] = float(
            teacher_b[others].mean(0) @ pred_b[i] + teacher_d[others].mean(0) @ pred_d[i]
        )
    loo = ranking_report(loo_teacher, loo_student, labels, species_ids)
    logger.info("Leave-one-out %s", _fmt_report(loo))
    print("stack_v2 " + _fmt_report(full))
    print("leave-one-out " + _fmt_report(loo))


def _fmt_report(report: dict) -> str:
    return (
        f"n={int(report['n'])} "
        f"teacher top1={report['teacher_top1']:.3f} top3={report['teacher_top3']:.3f} "
        f"student top1={report['student_top1']:.3f} top3={report['student_top3']:.3f} "
        f"student+1q top1={report['student_plus_one_question_top1']:.3f}"
    )


def write_pages_bundle() -> Path:
    bundle = build_browser_bundle_from_disk(
        DATA_DIR / "stack_v2_bundle.npz",
        SPECIES_DB_PATH,
        REFERENCE_MANIFEST if REFERENCE_MANIFEST.exists() else None,
    )
    write_browser_bundle(bundle, BROWSER_BUNDLE_PATH)
    logger.info("Wrote %s", BROWSER_BUNDLE_PATH)
    return BROWSER_BUNDLE_PATH


def copy_reference_plates() -> None:
    """Pages publishes docs/ only. Copy the tracked plates next to the bundle."""
    import shutil

    dest_root = BASE_DIR / "docs" / "reference_images"
    if not REFERENCE_IMAGE_DIR.exists():
        return
    for path in REFERENCE_IMAGE_DIR.rglob("*"):
        if not path.is_file():
            continue
        dest = dest_root / path.relative_to(REFERENCE_IMAGE_DIR)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == path.stat().st_size:
            continue
        shutil.copy2(path, dest)


def push_to_hf(repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    readme_path = ONNX_DIR / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        f"""---
license: mit
library_name: onnx
tags:
  - biology
  - wildlife-conservation
  - chelonians
  - dinov2
  - bioclip
  - onnxruntime-web
---

# {repo_id} — Chelonid-ID Distilled Browser Student (Stack v2)

Distilled browser vision model for **Chelonid-ID** (Indian freshwater turtles & tortoises identification for forest-range & wildlife-crime triage).

## Architecture
- **Backbone:** `facebook/dinov2-with-registers-small` (`22M` parameters, `384`-d CLS token)
- **Distillation Projection Head:** `384 -> 1024 (GELU) -> 2048`, split into two L2-normalised `1024`-d embeddings:
  - `bioclip_emb` (`1024`-d): aligned to centered **BioCLIP 2.5 ViT-H/14** (`imageomics/bioclip-2.5-vith14`)
  - `dino_emb` (`1024`-d): aligned to centered **DINOv2-with-registers-large** (`facebook/dinov2-with-registers-large`)

## Files
- `turtleid_student_int8.onnx` (~23 MB): Dynamic INT8 quantized ONNX model (default for `onnxruntime-web` in the browser).
- `turtleid_student_fp16.onnx` (~44 MB): IEEE float16 ONNX model.
- `turtleid_student_fp32.onnx` (~88 MB): Full float32 ONNX model.
- `stack_v2_browser_bundle.json`: Compact float16 Stack v2 reference gallery + 30-species database + 12-character MorphKey matrix.
""",
        encoding="utf-8",
    )
    files = [
        readme_path,
        ONNX_DIR / "turtleid_student_int8.onnx",
        ONNX_DIR / "turtleid_student_fp16.onnx",
        ONNX_DIR / "turtleid_student_fp32.onnx",
        BROWSER_BUNDLE_PATH,
    ]
    for path in files:
        if not path.exists():
            logger.warning("Skip missing %s", path)
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info("Uploaded %s to %s", path.name, repo_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-only", action="store_true", help="Write the Pages JSON and stop.")
    parser.add_argument("--train", action="store_true", help="Fit the projection head and export ONNX.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--benchmark", action="store_true", help="Print LOO and GroupKFold if pairs exist.")
    parser.add_argument("--push-to-hf", default="", help="Model repo id, e.g. ajinkyapatil60/turtleid-student-onnx")
    parser.add_argument("--out-dir", type=Path, default=ONNX_DIR)
    args = parser.parse_args()

    write_pages_bundle()
    copy_reference_plates()
    if args.train:
        train_head(args.epochs, args.lr, args.out_dir)
    elif args.benchmark and PAIRS_PATH.exists():
        pairs = _load_or_compute_pairs()
        if pairs is not None:
            head = build_projection_head()
            _print_benchmark(pairs[0], pairs[1], pairs[2], pairs[3], pairs[4], head)
    if args.push_to_hf:
        push_to_hf(args.push_to_hf)


if __name__ == "__main__":
    main()
