"""Splits estratificados por paciente, sin leakage (regla feedback F4)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.data.inventory import PatientRecord, Task


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def _labels_and_groups(records: Sequence[PatientRecord], task: Task) -> tuple[np.ndarray, np.ndarray]:
    if task == "binary":
        y = np.array([int(r.is_cirrhotic) for r in records])
    elif task == "severity":
        y = np.array([(r.severity or 0) - 1 for r in records])
        if (y < 0).any():
            raise ValueError("Records sin severidad presentes con task='severity'; filtrar antes.")
    else:
        raise ValueError(f"Task desconocida: {task}")
    groups = np.array([r.patient_id for r in records])
    return y, groups


def make_stratified_group_kfold(
    records: Sequence[PatientRecord],
    task: Task = "binary",
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Fold]:
    """Genera folds estratificados por etiqueta y agrupados por `patient_id`.

    Garantiza que ningún paciente esté en train y val del mismo fold (regla anti-leakage).
    """
    y, groups = _labels_and_groups(records, task)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=groups)):
        # Sanidad: intersección de patient_ids debe ser vacía.
        tr_pids = set(groups[tr])
        va_pids = set(groups[va])
        if tr_pids & va_pids:
            raise RuntimeError(
                f"Fold {fold_id}: leakage de paciente — {len(tr_pids & va_pids)} pacientes "
                "están en train y val. Esto es un bug en make_stratified_group_kfold."
            )
        yield Fold(fold_id=fold_id, train_idx=np.asarray(tr), val_idx=np.asarray(va))


def class_pos_weight(y_train: np.ndarray) -> float:
    """`pos_weight` para BCEWithLogitsLoss en tarea binaria: n_neg / n_pos."""
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """Pesos inversos a frecuencia, normalizados para que sumen `num_classes`."""
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = (1.0 / counts)
    w *= num_classes / w.sum()
    return w
