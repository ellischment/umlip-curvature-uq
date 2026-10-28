# umlip-curvature-uq

Can uncertainty quantification detect systematic curvature errors in atomistic
foundation models?

Universal machine-learning interatomic potentials (uMLIPs) such as MACE-MP-0
and CHGNet tend to underestimate the curvature of the potential-energy surface,
so their phonon frequencies come out too low (Deng et al., 2025; Loew et al.,
2025). Uncertainty estimates based on disagreement between models only see the
variance of the predictions. If several models share the same bias, the spread
can stay small while the error does not.

This repository is a work in progress for an ICOMP 2026 poster. It contains the
pipeline for testing that hypothesis. **No results are reported yet.**

## Hypothesis

- Curvature is underestimated with the same sign across several uMLIPs, so
  bias² exceeds the variance between models.
- Spread-based UQ therefore separates strong from weak softening poorly
  (AUROC close to 0.5).
- Sub-hypothesis: a heterogeneous ensemble detects systematic softening only if
  it contains at least one model with low bias (e.g. MatterSim-v1).

## What the code does

1. Takes materials from the held-out test split of the PBE MDR phonon benchmark
   (the split used for phonon fine-tuning by Koker et al.).
2. Rebuilds DFT force constants from the stored finite displacements and forces.
3. Recomputes the forces on **the same displaced supercells** with each model,
   at the DFT-relaxed geometry, and builds model force constants.
4. Compares frequencies on a Gamma-centred q-mesh:
   - slope of ω_ML against ω_DFT through the origin (below 1 means softening),
   - signed relative error (ω_ML − ω_DFT)/ω_DFT, acoustic and optical bands separately,
   - number of imaginary modes.
5. Treats the set of models as a heterogeneous ensemble:
   - bias² and variance of the ensemble mean, in relative units,
   - prediction spread (no reference data used) as the uncertainty score,
   - AUROC/AUPRC of that score for flagging strong softening,
   - all of the above with and without MatterSim-v1.

Not implemented yet: homogeneous ensembles and multi-head committees (they need
training), Orb confidence head, latent-distance UQ, molecules (OMOL-1k-MD).

## Usage

```bash
python -m venv .venv
.venv/bin/pip install -e ".[models,dev]"   # .venv\Scripts\pip on Windows
pytest
python -m umlip_curvature_uq.run --n 25 --models mace-mp-0 chgnet mattersim-v1 orb-v3
```

Outputs go to `results/`: `per_model.csv`, `per_material.csv`, `summary.json`
and `spread_vs_softening.png`. Reference files are cached in `data/`.

Main options: `--n` number of materials, `--max-sites` largest unit cell,
`--mesh` q-mesh, `--min-freq` (THz) modes below it are skipped, `--threshold`
slope below which softening counts as strong (0.9 by default, not final).

## Notes on the method

- Bands are compared by index after sorting at each q-point; band crossings are
  not tracked.
- Modes with DFT frequency below 0.5 THz are skipped, which removes acoustic
  modes near Gamma.
- `max_freq` in the benchmark table is in K (1 THz ≈ 48 K). On a 6×6×6 mesh our
  DFT maxima agree with it within a few percent.

## Data and references

- Reference phonons: Loew, Sun, Wang, Botti, Marques, *npj Comput. Mater.* 11, 178
  (2025). Data: https://alexandria.icams.rub.de/data/phonon_benchmark/ (CC BY 4.0).
- Test split: Koker et al., PFT, arXiv:2601.07742;
  https://github.com/atomicarchitects/nequix
- Deng et al., *npj Comput. Mater.* 11, 9 (2025).
- Liu et al., heterogeneous ensemble UQ, arXiv:2507.21297.
- Beck et al., *J. Chem. Phys.* 163, 234103 (2025).

## Authors

T. Teplyakova, V. Popova (Central University).

## License

MIT for the code. The reference data keep their own license (CC BY 4.0).
