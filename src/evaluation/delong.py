"""Test de DeLong para comparar dos AUCs sobre las mismas muestras.

Implementación basada en Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm
for Comparing the Areas under Correlated Receiver Operating Characteristic Curves",
IEEE Signal Processing Letters 21(11). Algoritmo O(n log n) sobre las predicciones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DeLongResult:
    auc_a: float
    auc_b: float
    cov: np.ndarray   # 2x2
    z: float
    p_value: float


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(z)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1  # midrank 1-indexed
        i = k
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def _fast_delong(predictions_sorted: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Calcula AUC y matriz de covarianza para K modelos sobre las mismas muestras.

    predictions_sorted: (K, N) — primeras `label_1_count` columnas son positivos.
    Devuelve (aucs: (K,), cov: (K, K)).
    """
    m = label_1_count
    n = predictions_sorted.shape[1] - m
    if m == 0 or n == 0:
        raise ValueError("DeLong requiere al menos un positivo y un negativo.")
    pos = predictions_sorted[:, :m]
    neg = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_sorted[r])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    cov = np.atleast_2d(cov)
    return aucs, cov


def delong_test_two_aucs(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> DeLongResult:
    """Compara AUC(B) - AUC(A) y devuelve z-score + p-value bilateral."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="stable")
    y_sorted = y_true[order]
    a = np.asarray(y_score_a)[order]
    b = np.asarray(y_score_b)[order]
    label_1_count = int((y_sorted == 1).sum())
    preds = np.stack([a, b], axis=0)
    aucs, cov = _fast_delong(preds, label_1_count)
    diff = aucs[1] - aucs[0]
    var_diff = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var_diff <= 0:
        return DeLongResult(auc_a=float(aucs[0]), auc_b=float(aucs[1]), cov=cov, z=float("nan"), p_value=float("nan"))
    z = diff / np.sqrt(var_diff)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return DeLongResult(auc_a=float(aucs[0]), auc_b=float(aucs[1]), cov=cov, z=float(z), p_value=float(p))
