"""Seizure / encounter grouping for GroupKFold evaluation."""

from __future__ import annotations

import re
from collections import defaultdict


def seizure_group(filename: str) -> str:
    """Heuristic group id so same-event photos stay together.

    Examples:
      20260203_173642.jpg -> day_20260203
      IMG_6809.JPG / IMG_6809(1).JPG -> img_6809
      IMG-20260204-WA0010.jpg -> wa_20260204
      uuid.jpg -> uuid prefix
    """
    name = filename
    stem = name.rsplit(".", 1)[0]

    m = re.match(r"^(20\d{6})_", stem)
    if m:
        return f"day_{m.group(1)}"

    m = re.match(r"^IMG-(20\d{6})-WA", stem, re.I)
    if m:
        return f"wa_{m.group(1)}"

    m = re.match(r"^IMG_(\d+)", stem, re.I)
    if m:
        # bucket nearby shot numbers (same seizure often sequential)
        num = int(m.group(1))
        return f"img_{num // 10}"

    m = re.match(r"^([0-9a-f]{8})-", stem, re.I)
    if m:
        return f"uuid_{m.group(1)}"

    return f"file_{stem[:24]}"


def groups_for(filenames: list[str]) -> list[str]:
    return [seizure_group(name) for name in filenames]


def summarize_groups(filenames: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for name in filenames:
        counts[seizure_group(name)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
