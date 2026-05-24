"""Bootstrap para intervalos de confianza y comparaciones pareadas.

Regla de calidad: TODA métrica reportada debe ir con IC 95% (Reglas de calidad §5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    samples: np.ndarray


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """IC por percentile bootstrap (estratificado por etiqueta para mantener clase positiva)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        # Resample preservando proporción de clase (sin esto, algunos resamples no tienen positivos).
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            sp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            sn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([sp, sn])
        else:
            idx = rng.integers(0, n, size=n)
        try:
            samples[i] = metric_fn(y_true[idx], y_score[idx])
        except ValueError:
            samples[i] = np.nan
    samples = samples[~np.isnan(samples)]
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    point = float(metric_fn(y_true, y_score))
    return BootstrapResult(point=point, ci_low=float(low), ci_high=float(high), samples=samples)


@dataclass(frozen=True)
class PairedDiffResult:
    diff_point: float
    diff_ci_low: float
    diff_ci_high: float
    p_value: float
    samples: np.ndarray


def bootstrap_paired_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> PairedDiffResult:
    """Bootstrap pareado: diferencia metric(B) - metric(A), resampleando ÍNDICES (mismo set en ambos modelos).

    p-value: proporción de resamples donde la diferencia cambia de signo respecto al observado.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(y_score_a)
    b = np.asarray(y_score_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            samples[i] = metric_fn(y_true[idx], b[idx]) - metric_fn(y_true[idx], a[idx])
        except ValueError:
            samples[i] = np.nan
    samples = samples[~np.isnan(samples)]
    point = metric_fn(y_true, b) - metric_fn(y_true, a)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    # p-value bilateral: 2 * min(P(diff<=0), P(diff>=0))
    p_left = float((samples <= 0).mean())
    p_right = float((samples >= 0).mean())
    p_value = 2.0 * min(p_left, p_right)
    return PairedDiffResult(
        diff_point=float(point),
        diff_ci_low=float(low),
        diff_ci_high=float(high),
        p_value=float(p_value),
        samples=samples,
    )


def format_with_ci(point: float, ci_low: float, ci_high: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} (95% CI {ci_low:.{digits}f}–{ci_high:.{digits}f})"
