"""5-fold CV del modelo DL principal — invocable desde la CLI:

    python -m src.training train-dl --config configs/dl_2_5d.yaml

Mixed precision + gradient accumulation + early stopping ya están en `train_one_fold`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import CirrhosisDataModule
from src.data.inventory import build_inventory
from src.data.splits import class_pos_weight, make_stratified_group_kfold
from src.models.cnn_2_5d import build_resnet25d
from src.training.train_dl import _evaluate, train_one_fold
from src.training.utils import count_parameters, set_seed

log = logging.getLogger(__name__)


def run_dl_cv(cfg: dict) -> int:
    set_seed(cfg.get("seed", 42))
    data_cfg = cfg["data"]["dataset"]
    root = Path(data_cfg["root"]).resolve()
    inventory = build_inventory(root)
    records = inventory.filter(task="binary")

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(cfg["output"]["logs_dir"]); logs_dir.mkdir(parents=True, exist_ok=True)

    fold_states = []
    oof_rows: list[dict] = []

    for fold in make_stratified_group_kfold(
        records,
        task="binary",
        n_splits=cfg["data"]["splits"]["n_folds"],
        seed=cfg["data"]["splits"]["seed"],
    ):
        cache_dir = cfg["data"]["dataset"].get("cache_dir")
        cache_dir = Path(cache_dir) if cache_dir else None
        dm = CirrhosisDataModule(
            inventory, fold, task="binary",
            batch_size=cfg["data"]["dataloader"]["batch_size"],
            num_workers=cfg["data"]["dataloader"]["num_workers"],
            slices_per_patient_train=cfg["data"]["stack_2_5d"]["slices_per_patient_train"],
            slices_per_patient_eval=cfg["data"]["stack_2_5d"]["slices_per_patient_eval"],
            cache_dir=cache_dir,
        )
        y_tr = np.array([int(r.is_cirrhotic) for r in dm.train_recs])
        pw = class_pos_weight(y_tr)
        model = build_resnet25d(
            num_classes=cfg["model"]["num_classes"],
            in_channels=cfg["model"]["in_channels"],
            pretrained=(cfg["model"]["pretrained_weights"] == "imagenet"),
            dropout=cfg["model"]["dropout"],
            attention_pool=cfg["model"]["attention_pool"],
        )
        total, train_p = count_parameters(model)
        log.info("fold %d: params total=%d train=%d pos_weight=%.2f", fold.fold_id, total, train_p, pw)

        state = train_one_fold(
            model=model,
            train_loader=dm.train_loader(),
            val_loader=dm.val_loader(),
            fold_id=fold.fold_id,
            epochs=cfg["train"]["epochs"],
            lr_backbone=cfg["train"]["optimizer"]["lr_backbone"],
            lr_head=cfg["train"]["optimizer"]["lr_head"],
            weight_decay=cfg["train"]["optimizer"]["weight_decay"],
            pos_weight=pw,
            grad_accum_steps=cfg["train"]["grad_accum_steps"],
            grad_clip_norm=cfg["train"]["grad_clip_norm"],
            patience=cfg["train"]["early_stopping_patience"],
            output_dir=ckpt_dir,
            seed=cfg.get("seed", 42),
        )
        fold_states.append(state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best = torch.load(state.checkpoint_path, map_location=device)
        model.load_state_dict(best["model"])
        _, val_auc, probs, labels, pids = _evaluate(model, dm.val_loader(), device)
        log.info("fold %d best val AUC=%.4f", fold.fold_id, val_auc)
        # Agregación por paciente (mean).
        per_patient: dict[str, list[float]] = {}
        per_label: dict[str, int] = {}
        for pid, p, y in zip(pids, probs, labels, strict=True):
            per_patient.setdefault(pid, []).append(float(p))
            per_label[pid] = int(y)
        for pid, ps in per_patient.items():
            oof_rows.append({"patient_id": pid, "label": per_label[pid],
                             "proba": float(np.mean(ps)), "fold": fold.fold_id})

    out_dir = Path(cfg["output"]["results_csv"]).parent; out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(oof_rows).to_csv(out_dir / "dl_2_5d_preds.csv", index=False)

    timing = pd.DataFrame([{
        "fold": s.fold_id, "epochs_run": s.epochs_run, "best_epoch": s.best_epoch,
        "best_val_auc": s.best_val_auc, "wallclock_h": s.wallclock_sec / 3600,
    } for s in fold_states])
    timing.to_csv(cfg["output"]["timing_csv"], index=False)
    log.info("OOF predictions y timing guardados.")
    print(timing.to_string(index=False))
    return 0
