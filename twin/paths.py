"""Canonical filesystem locations, resolved relative to the repository root."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MADHUTWIN_ROOT", Path(__file__).resolve().parents[1]))

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SYNTHETIC = DATA / "synthetic"

CGMACROS_RAW = RAW / "cgmacros"
SHANGHAI_RAW = RAW / "shanghai" / "ShanghaiDM"

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figures"


def ensure_dirs() -> None:
    for p in (PROCESSED, SYNTHETIC, ARTIFACTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
