"""Pilot run: DFT vs uMLIP phonons on a few MDR test materials.

Example:
    python -m umlip_curvature_uq.run --n 25 --models mace-mp-0 chgnet mattersim-v1 orb-v3
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import data, metrics, models, phonons

LOW_BIAS_MODEL = "mattersim-v1"


def evaluate_material(mp_id, calcs, cache_dir, mesh, min_freq):
    ref_ph = data.load_reference(mp_id, cache_dir)
    ref = phonons.mesh_frequencies(ref_ph, mesh)
    preds = {}
    rows = []
    for name, calc in calcs.items():
        t0 = time.perf_counter()
        pred = phonons.mesh_frequencies(phonons.model_phonopy(ref_ph, calc), mesh)
        preds[name] = pred
        b = metrics.model_bias(pred, ref, min_freq)
        rows.append({"mp_id": mp_id, "model": name, "seconds": round(time.perf_counter() - t0, 1), **asdict(b)})
    ens = {"mp_id": mp_id, "n_atoms": len(ref_ph.unitcell), "ref_max_freq_thz": float(ref.max())}
    groups = {"all": list(preds)}
    if LOW_BIAS_MODEL in preds and len(preds) > 2:
        groups["without_" + LOW_BIAS_MODEL] = [m for m in preds if m != LOW_BIAS_MODEL]
    for g, names in groups.items():
        p = [preds[m] for m in names]
        s = metrics.ensemble_stats(p, ref, min_freq)
        ens.update({f"{g}:{k}": v for k, v in asdict(s).items()})
        ens[f"{g}:prediction_spread"] = metrics.prediction_spread(p, min_freq)
    return rows, ens


def summarize(ens_rows, threshold):
    out = {"threshold_slope": threshold, "n_materials": len(ens_rows)}
    groups = sorted({k.split(":")[0] for r in ens_rows for k in r if ":" in k})
    for g in groups:
        slope = np.array([r[f"{g}:ensemble_slope"] for r in ens_rows])
        spread = np.array([r[f"{g}:prediction_spread"] for r in ens_rows])
        bias_sq = np.array([r[f"{g}:bias_sq"] for r in ens_rows])
        var = np.array([r[f"{g}:variance"] for r in ens_rows])
        strong = slope < threshold
        out[g] = {
            "mean_ensemble_slope": float(slope.mean()),
            "bias_sq_over_variance": float(bias_sq.mean() / var.mean()),
            **metrics.detection_scores(spread, strong),
        }
    return out


def plot(ens_rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=200)
    x = [r["all:prediction_spread"] for r in ens_rows]
    y = [1 - r["all:ensemble_slope"] for r in ens_rows]
    ax.scatter(x, y, s=18)
    ax.set_xlabel("ensemble spread (relative)")
    ax.set_ylabel("softening of ensemble mean (1 - slope)")
    ax.axhline(0, lw=0.5, color="grey")
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--max-sites", type=int, default=8)
    p.add_argument("--models", nargs="+", default=list(models.REGISTRY))
    p.add_argument("--mesh", type=int, nargs=3, default=[6, 6, 6])
    p.add_argument("--min-freq", type=float, default=0.5, help="THz; modes below are skipped")
    p.add_argument("--threshold", type=float, default=0.9, help="strong softening: slope below this")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    ids = data.select_materials(a.cache, a.n, max_sites=a.max_sites)
    calcs = {m: models.get_calculator(m, a.device) for m in a.models}

    model_rows, ens_rows = [], []
    for i, mp_id in enumerate(ids, 1):
        rows, ens = evaluate_material(mp_id, calcs, a.cache, a.mesh, a.min_freq)
        model_rows += rows
        ens_rows.append(ens)
        print(f"[{i}/{len(ids)}] {mp_id}: ensemble slope {ens['all:ensemble_slope']:.3f}", flush=True)

    for name, rows in (("per_model.csv", model_rows), ("per_material.csv", ens_rows)):
        with (a.out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary = summarize(ens_rows, a.threshold)
    summary["models"] = a.models
    summary["mesh"] = a.mesh
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(ens_rows, a.out / "spread_vs_softening.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
