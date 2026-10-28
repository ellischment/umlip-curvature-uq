"""Signed curvature bias of one model and spread-based uncertainty of several.

All functions take frequencies on the same q-mesh, shape (n_q, n_bands), in THz.
Modes are compared band by band (bands sorted by frequency at each q), which is
the usual approximation when band crossings are not tracked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

N_ACOUSTIC = 3


def _mask(ref: np.ndarray, min_freq: float) -> np.ndarray:
    """Modes used for comparison: reference frequency above `min_freq`.

    Excludes the acoustic modes near Gamma, where relative errors blow up.
    """
    return ref > min_freq


def slope_through_origin(pred: np.ndarray, ref: np.ndarray, min_freq: float = 0.5) -> float:
    """Least-squares slope of pred = s * ref. s < 1 means softening."""
    m = _mask(ref, min_freq)
    x, y = ref[m], pred[m]
    return float(np.dot(x, y) / np.dot(x, x))


def relative_error(pred: np.ndarray, ref: np.ndarray, min_freq: float = 0.5) -> np.ndarray:
    """(w_ML - w_DFT) / w_DFT for the compared modes, flattened."""
    m = _mask(ref, min_freq)
    return (pred[m] - ref[m]) / ref[m]


@dataclass
class ModelBias:
    slope: float
    mean_rel_error: float
    mean_rel_error_acoustic: float
    mean_rel_error_optical: float
    max_freq_ratio: float
    n_imaginary: int


def model_bias(pred: np.ndarray, ref: np.ndarray, min_freq: float = 0.5) -> ModelBias:
    acoustic = np.zeros_like(ref, dtype=bool)
    acoustic[:, :N_ACOUSTIC] = True
    keep = _mask(ref, min_freq)
    rel = np.where(keep, (pred - ref) / np.where(keep, ref, 1.0), np.nan)
    return ModelBias(
        slope=slope_through_origin(pred, ref, min_freq),
        mean_rel_error=float(np.nanmean(rel)),
        mean_rel_error_acoustic=float(np.nanmean(np.where(acoustic, rel, np.nan))),
        mean_rel_error_optical=float(np.nanmean(np.where(~acoustic, rel, np.nan)))
        if ref.shape[1] > N_ACOUSTIC else float("nan"),
        max_freq_ratio=float(pred.max() / ref.max()),
        n_imaginary=int((pred < -min_freq).sum()),
    )


@dataclass
class EnsembleStats:
    bias_sq: float      # mean over modes of (mean_models - ref)^2, relative units
    variance: float     # mean over modes of variance across models, relative units
    spread: float       # sqrt(variance): the uncertainty an ensemble would report
    ensemble_slope: float


def ensemble_stats(preds: list[np.ndarray], ref: np.ndarray, min_freq: float = 0.5) -> EnsembleStats:
    """Bias-variance split of the ensemble mean, in units of the reference frequency.

    Uses relative frequencies w_ML / w_DFT so materials with different stiffness
    are comparable. Spread uses only predictions, as a real UQ method would.
    """
    m = _mask(ref, min_freq)
    stack = np.stack([p[m] / ref[m] for p in preds])  # (n_models, n_modes)
    mean = stack.mean(axis=0)
    variance = float(stack.var(axis=0).mean())
    mean_freq = np.mean(np.stack(preds), axis=0)
    return EnsembleStats(
        bias_sq=float(((mean - 1.0) ** 2).mean()),
        variance=variance,
        spread=float(np.sqrt(variance)),
        ensemble_slope=slope_through_origin(mean_freq, ref, min_freq),
    )


def prediction_spread(preds: list[np.ndarray], min_freq: float = 0.5) -> float:
    """Relative spread across models without any reference data.

    Mean over modes of std / mean across models; modes where the ensemble mean
    is below `min_freq` are skipped.
    """
    stack = np.stack(preds)
    mean = stack.mean(axis=0)
    m = mean > min_freq
    return float((stack.std(axis=0)[m] / mean[m]).mean())


def detection_scores(scores: np.ndarray, strong: np.ndarray) -> dict[str, float]:
    """AUROC and AUPRC of an uncertainty score for flagging strong softening."""
    strong = np.asarray(strong, dtype=bool)
    if strong.all() or not strong.any():
        return {"auroc": float("nan"), "auprc": float("nan"), "n_positive": int(strong.sum())}
    return {
        "auroc": float(roc_auc_score(strong, scores)),
        "auprc": float(average_precision_score(strong, scores)),
        "n_positive": int(strong.sum()),
    }
