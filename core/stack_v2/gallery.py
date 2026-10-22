"""External multi-prototype gallery (iNaturalist research-grade, cached)."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

PRIORITY_TAXA: dict[str, str] = {
    "hardella_thurjii": "Hardella thurjii",
    "geoclemys_hamiltonii": "Geoclemys hamiltonii",
    "trachemys_scripta_elegans": "Trachemys scripta elegans",
    "pangshura_tentoria": "Pangshura tentoria",
    "nilssonia_gangetica": "Nilssonia gangetica",
    "lissemys_punctata": "Lissemys punctata",
    "pangshura_tecta": "Pangshura tecta",
    "geochelone_elegans": "Geochelone elegans",
}


def ensure_inat_gallery(
    root: Path,
    *,
    per_species: int = 12,
    timeout: float = 25.0,
) -> list[tuple[str, Path]]:
    """Download a small research-grade iNat gallery if missing."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        items = []
        for species_id, files in payload.get("files", {}).items():
            for name in files:
                path = root / species_id / name
                if path.exists():
                    items.append((species_id, path))
        if items:
            logger.info("Using cached external gallery: %d images", len(items))
            return items

    files: dict[str, list[str]] = {}
    items: list[tuple[str, Path]] = []
    for species_id, scientific in PRIORITY_TAXA.items():
        species_dir = root / species_id
        species_dir.mkdir(parents=True, exist_ok=True)
        urls = _inat_photo_urls(scientific, per_species, timeout=timeout)
        saved: list[str] = []
        for index, url in enumerate(urls):
            dest = species_dir / f"inat_{index:02d}.jpg"
            if not dest.exists():
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as exc:
                    logger.warning("iNat download failed %s: %s", url, exc)
                    continue
            try:
                with Image.open(dest) as check:
                    check.verify()
            except Exception:
                dest.unlink(missing_ok=True)
                continue
            saved.append(dest.name)
            items.append((species_id, dest))
        files[species_id] = saved
        logger.info("iNat %s: %d photos", scientific, len(saved))

    manifest_path.write_text(json.dumps({"files": files}, indent=2))
    return items


def _inat_photo_urls(scientific: str, limit: int, *, timeout: float) -> list[str]:
    query = urllib.parse.quote(scientific)
    url = (
        "https://api.inaturalist.org/v1/observations"
        f"?taxon_name={query}&quality_grade=research&photos=true"
        f"&per_page={limit}&order_by=votes"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("iNat API failed for %s: %s", scientific, exc)
        return []

    urls: list[str] = []
    for obs in payload.get("results", []):
        photos = obs.get("photos") or []
        if not photos:
            continue
        src = photos[0].get("url") or ""
        if not src:
            continue
        src = src.replace("square.", "medium.").replace("/square", "/medium")
        urls.append(src)
        if len(urls) >= limit:
            break
    return urls
