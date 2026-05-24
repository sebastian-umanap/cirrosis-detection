"""Entrena un solo fold para medir tiempo real y validar la configuración.

Si va bien (val_auc razonable, < ~30 min/epoch), procedemos a 5-fold pleno.
Si no, ajustamos batch_size, slices_per_patient o num_workers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.inventory import build_inventory  # noqa: E402
from src.data.dataset import CirrhosisDataModule  # noqa: E402
from src.data.splits import class_pos_weight, make_stratified_group_kfold  # noqa: E402
from src.models.cnn_2_5d import build_resnet25d  # noqa: E402
from src.training.train_dl import _evaluate, train_one_fold  # noqa: E402
from src.training.utils import count_parameters, set_seed  # noqa: E402


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        defs = cfg.pop("defaults", {})
        for key, fname in (defs or {}).items():
            child = path.parent / fname
            with open(child, encoding="utf-8") as f:
                cfg[key] = yaml.safe_load(f)
    return cfg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = _load_config(PROJECT / "configs" / "dl_2_5d_pilot.yaml")

    set_seed(cfg.get("seed", 42))

    inv = build_inventory(PROJECT / cfg["data"]["dataset"]["root"])
    recs = inv.filter(task="binary")
    cache_dir = PROJECT / cfg["data"]["dataset"]["cache_dir"]
    print(f"records={len(recs)}  cache_dir={cache_dir}  exists={cache_dir.exists()}")

    fold = next(make_stratified_group_kfold(recs, task="binary", n_splits=5, seed=cfg["data"]["splits"]["seed"]))
    print(f"fold {fold.fold_id}: train={len(fold.train_idx)} val={len(fold.val_idx)}")

    dm = CirrhosisDataModule(
        inv, fold, task="binary",
        batch_size=cfg.get("override_batch_size", cfg["data"]["dataloader"]["batch_size"]),
        num_workers=cfg["data"]["dataloader"]["num_workers"],
        slices_per_patient_train=cfg.get("override_slices_train", cfg["data"]["stack_2_5d"]["slices_per_patient_train"]),
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
    print(f"params total={total:,} trainables={train_p:,} pos_weight={pw:.3f}")

    ckpt_dir = PROJECT / cfg["output"]["checkpoint_dir"]
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
    print(f"\nFOLD {state.fold_id} done in {state.wallclock_sec/60:.1f} min "
          f"({state.epochs_run} epochs, best={state.best_val_auc:.4f} @epoch {state.best_epoch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
