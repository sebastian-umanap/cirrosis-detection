"""Métricas binarias y multiclase al punto operativo de Youden + utilitarias.

Diseñadas para evaluación por paciente (los samples por slice ya están agregados aguas arriba).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class BinaryMetrics:
    auc_roc: float
    auc_pr: float
    f1: float
    mcc: float
    sensitivity: float
    specificity: float
    threshold_youden: float
    n_pos: int
    n_neg: int


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def binary_metrics_at_youden(y_true: np.ndarray, y_prob: np.ndarray) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        nan = float("nan")
        return BinaryMetrics(nan, nan, nan, nan, nan, nan, 0.5, n_pos, n_neg)
    thr = _youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return BinaryMetrics(
        auc_roc=float(roc_auc_score(y_true, y_prob)),
        auc_pr=float(average_precision_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        sensitivity=float(sens),
        specificity=float(spec),
        threshold_youden=thr,
        n_pos=n_pos,
        n_neg=n_neg,
    )


@dataclass(frozen=True)
class MulticlassMetrics:
    macro_f1: float
    macro_auc_ovr: float
    confusion: np.ndarray
    per_class_f1: np.ndarray


def multiclass_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> MulticlassMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)  # shape (N, K)
    if y_prob.ndim == 1:
        raise ValueError("Para multiclase, y_prob debe ser (N, K).")
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    per_class = f1_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0))
    # AUC one-vs-rest macro, manejando ausencia de clase.
    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")
    return MulticlassMetrics(
        macro_f1=macro_f1,
        macro_auc_ovr=macro_auc,
        confusion=cm,
        per_class_f1=np.asarray(per_class, dtype=float),
    )
