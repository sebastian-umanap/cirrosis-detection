"""Tests para métricas, bootstrap y DeLong."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.bootstrap import bootstrap_ci, bootstrap_paired_diff
from src.evaluation.delong import delong_test_two_aucs
from src.evaluation.metrics import binary_metrics_at_youden, multiclass_metrics


def test_binary_metrics_perfect_separator():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    bm = binary_metrics_at_youden(y, p)
    assert bm.auc_roc == pytest.approx(1.0)
    assert bm.f1 == pytest.approx(1.0)
    assert bm.sensitivity == pytest.approx(1.0)
    assert bm.specificity == pytest.approx(1.0)


def test_binary_metrics_random():
    rng = np.random.default_rng(0)
    y = (rng.random(100) < 0.5).astype(int)
    p = rng.random(100)
    bm = binary_metrics_at_youden(y, p)
    assert 0.3 < bm.auc_roc < 0.7  # AUC de random ~0.5


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = (rng.random(100) < 0.4).astype(int)
    p = y * 0.7 + rng.random(100) * 0.3
    r = bootstrap_ci(y, p, lambda yt, ys: roc_auc_score(yt, ys), n_resamples=500, seed=1)
    assert r.ci_low <= r.point <= r.ci_high
    assert 0.0 <= r.ci_low <= 1.0
    assert 0.0 <= r.ci_high <= 1.0


def test_bootstrap_paired_diff_zero_when_identical():
    rng = np.random.default_rng(0)
    y = (rng.random(80) < 0.5).astype(int)
    p = rng.random(80)
    r = bootstrap_paired_diff(y, p, p, lambda yt, ys: roc_auc_score(yt, ys), n_resamples=300, seed=2)
    assert abs(r.diff_point) < 1e-9
    assert r.p_value > 0.5     # no diferencia ⇒ p alta


def test_delong_perfect_vs_random():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
    p_perfect = y.astype(float)
    p_random = rng.random(100)
    res = delong_test_two_aucs(y, p_random, p_perfect)
    assert res.auc_b > res.auc_a
    assert res.p_value < 0.01


def test_multiclass_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.1, 0.2, 0.7],
    ])
    mm = multiclass_metrics(y, probs, n_classes=3)
    assert mm.macro_f1 == pytest.approx(1.0)
    assert mm.confusion.trace() == 6
