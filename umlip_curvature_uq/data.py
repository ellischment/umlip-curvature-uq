"""Reference DFT phonons: the PBE MDR phonon benchmark of Loew et al.

Source: https://alexandria.icams.rub.de/data/phonon_benchmark/ (CC BY 4.0).
Each material is a phonopy YAML file (bz2) with the relaxed unit cell, the
supercell matrix and the finite displacements with DFT (PBE) forces.

Test split: the 1000 held-out materials used for PFT (Koker et al., 2026),
https://github.com/atomicarchitects/nequix/blob/main/data/pbe_mdr_split.json
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import phonopy

BASE_URL = "https://alexandria.icams.rub.de/data/phonon_benchmark"
SPLIT_URL = (
    "https://raw.githubusercontent.com/atomicarchitects/nequix/main/data/pbe_mdr_split.json"
)


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def load_index(cache_dir: Path) -> list[dict]:
    """Rows of pbe.csv: mp_id, nsites, max_freq, stable, ..."""
    path = _download(f"{BASE_URL}/pbe.csv", cache_dir / "pbe.csv")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(cache_dir: Path) -> dict[str, list[str]]:
    path = _download(SPLIT_URL, cache_dir / "pbe_mdr_split.json")
    return json.loads(path.read_text(encoding="utf-8"))


def select_materials(
    cache_dir: Path,
    n: int,
    split: str = "test",
    max_sites: int | None = 8,
    stable_only: bool = True,
) -> list[str]:
    """Deterministic subset: smallest cells first, for a cheap pilot run."""
    ids = set(load_split(cache_dir)[split])
    rows = [r for r in load_index(cache_dir) if r["mp_id"] in ids]
    if stable_only:
        rows = [r for r in rows if r["stable"] == "True"]
    if max_sites is not None:
        rows = [r for r in rows if int(r["nsites"]) <= max_sites]
    rows.sort(key=lambda r: (int(r["nsites"]), r["mp_id"]))
    return [r["mp_id"] for r in rows[:n]]


def load_reference(mp_id: str, cache_dir: Path) -> phonopy.Phonopy:
    """Phonopy object with DFT displacements and forces (force constants built)."""
    path = _download(f"{BASE_URL}/pbe/{mp_id}.yaml.bz2", cache_dir / "pbe" / f"{mp_id}.yaml.bz2")
    return phonopy.load(str(path), produce_fc=True, log_level=0)
