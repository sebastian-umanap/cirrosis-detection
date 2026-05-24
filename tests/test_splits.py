"""Verifica el requisito anti-leakage de los splits por paciente.

Test crítico: si un paciente aparece en train y val de un mismo fold, todo el
proyecto se invalida (regla feedback F4 + reglas de calidad §4).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.data.inventory import Inventory, PatientRecord
from src.data.splits import (
    Fold,
    class_pos_weight,
    class_weights,
    make_stratified_group_kfold,
)


def _mock_records(n_pos: int, n_neg: int, slices_per_patient: int = 3) -> list[PatientRecord]:
    out: list[PatientRecord] = []
    rng = np.random.default_rng(0)
    for i in range(n_pos + n_neg):
        is_pos = i < n_pos
        for k in range(slices_per_patient):
            out.append(
                PatientRecord(
                    patient_id=f"P{i:03d}",
                    modality=("T1w" if k % 2 == 0 else "T2w"),
                    image_path=Path(f"/fake/P{i:03d}_{k}.nii.gz"),
                    mask_path=Path(f"/fake/P{i:03d}_{k}_mask.nii.gz"),
                    is_cirrhotic=is_pos,
                    severity=int(rng.integers(1, 4)) if is_pos else None,
                    age=float(rng.integers(30, 80)),
                    sex="M" if i % 2 else "F",
                    ascites=is_pos and bool(rng.integers(0, 2)),
                    splenomegaly=is_pos and bool(rng.integers(0, 2)),
                    varices=is_pos and bool(rng.integers(0, 2)),
                    hcc=False,
                )
            )
    return out


def test_no_patient_leakage_binary():
    recs = _mock_records(n_pos=80, n_neg=20)
    folds = list(make_stratified_group_kfold(recs, task="binary", n_splits=5, seed=42))
    assert len(folds) == 5
    for f in folds:
        train_pids = {recs[i].patient_id for i in f.train_idx}
        val_pids = {recs[i].patient_id for i in f.val_idx}
        assert train_pids.isdisjoint(val_pids), f"fold {f.fold_id}: leakage"


def test_each_fold_has_both_classes_binary():
    recs = _mock_records(n_pos=50, n_neg=10)
    for f in make_stratified_group_kfold(recs, task="binary", n_splits=5, seed=42):
        y_val = [int(recs[i].is_cirrhotic) for i in f.val_idx]
        assert sum(y_val) >= 1, "fold sin positivos en val"
        assert (len(y_val) - sum(y_val)) >= 1, "fold sin negativos en val"


def test_class_pos_weight():
    y = np.array([1, 1, 1, 0])
    assert class_pos_weight(y) == pytest.approx(1 / 3, rel=1e-6)
    assert class_pos_weight(np.array([1, 1, 1])) == 1.0


def test_class_weights_normalisation():
    y = np.array([0, 0, 1, 2])
    w = class_weights(y, num_classes=3)
    assert w.shape == (3,)
    assert w.sum() == pytest.approx(3.0, rel=1e-6)
