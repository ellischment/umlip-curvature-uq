import numpy as np
import pytest

from umlip_curvature_uq import metrics

REF = np.array([[0.0, 0.0, 0.0, 5.0, 6.0], [1.0, 2.0, 3.0, 5.5, 7.0]])


def test_slope_skips_near_zero_modes():
    pred = REF * 0.8
    pred[0, :3] = [9.0, -9.0, 9.0]  # acoustic Gamma modes, below min_freq in REF
    assert metrics.slope_through_origin(pred, REF) == pytest.approx(0.8)


def test_acoustic_and_optical_errors_are_separated():
    pred = REF.copy()
    pred[:, :3] *= 0.5
    pred[:, 3:] *= 1.1
    b = metrics.model_bias(pred, REF)
    assert b.mean_rel_error_acoustic == pytest.approx(-0.5)
    assert b.mean_rel_error_optical == pytest.approx(0.1)


def test_imaginary_modes_counted():
    pred = REF.copy()
    pred[1, 0] = -2.0
    assert metrics.model_bias(pred, REF).n_imaginary == 1


def test_bias_variance_split():
    preds = [REF * 0.8, REF * 1.0]
    s = metrics.ensemble_stats(preds, REF)
    assert s.bias_sq == pytest.approx(0.01)   # mean 0.9 -> (0.9 - 1)^2
    assert s.variance == pytest.approx(0.01)  # values 0.8 and 1.0 -> var 0.01
    assert s.ensemble_slope == pytest.approx(0.9)


def test_detection_scores():
    strong = np.array([1, 1, 0, 0])
    assert metrics.detection_scores(np.array([0.9, 0.8, 0.1, 0.2]), strong)["auroc"] == 1.0
    assert metrics.detection_scores(np.array([0.1, 0.2, 0.9, 0.8]), strong)["auroc"] == 0.0
    assert np.isnan(metrics.detection_scores(np.ones(4), np.zeros(4))["auroc"])
