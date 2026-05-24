"""Loop de entrenamiento para el modelo DL 2.5D ResNet-50.

Características:
    - Mixed precision torch.cuda.amp (obligatorio por restricción de cómputo).
    - Gradient accumulation para alcanzar batch efectivo deseado en RTX 3060 12 GB.
    - Discriminative LR backbone vs head.
    - Early stopping por AUC de validación.
    - Logs por epoch en CSV.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.splits import class_pos_weight
from src.training.utils import EarlyStopper, discriminative_param_groups, set_seed

log = logging.getLogger(__name__)


@dataclass
class TrainState:
    fold_id: int
    best_val_auc: float
    best_epoch: int
    epochs_run: int
    wallclock_sec: float
    checkpoint_path: Path
    log_csv_path: Path


def _step_train(
    model: nn.Module,
    batch: dict,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accum_steps: int,
    grad_clip_norm: float,
    accum_counter: list[int],
) -> float:
    image = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
    label = batch["label"].to(device, non_blocking=True).float()
    with autocast(dtype=torch.float16):
        logits = model(image).squeeze(-1)
        loss = loss_fn(logits, label) / accum_steps
    scaler.scale(loss).backward()
    accum_counter[0] += 1
    if accum_counter[0] >= accum_steps:
        if grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        accum_counter[0] = 0
    return float(loss.item() * accum_steps)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, list[str]]:
    """Devuelve (loss, auc, probs_per_slice, labels_per_slice, patient_ids_per_slice)."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_pids: list[str] = []
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).float()
        with autocast(dtype=torch.float16):
            logits = model(image).squeeze(-1)
            loss = bce(logits, label)
        total_loss += float(loss.item())
        n_batches += 1
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(label.cpu().numpy().astype(int))
        all_pids.extend(batch["patient_id"])
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits)))
    labels = np.concatenate(all_labels)
    pids = all_pids
    # AUC por paciente (promedio de probas dentro del paciente).
    per_patient: dict[str, list[float]] = {}
    per_patient_label: dict[str, int] = {}
    for pid, p, y in zip(pids, probs, labels, strict=True):
        per_patient.setdefault(pid, []).append(float(p))
        per_patient_label[pid] = int(y)
    pat_probs = np.array([np.mean(per_patient[pid]) for pid in per_patient])
    pat_labels = np.array([per_patient_label[pid] for pid in per_patient])
    auc = roc_auc_score(pat_labels, pat_probs) if len(np.unique(pat_labels)) > 1 else float("nan")
    return total_loss / max(n_batches, 1), float(auc), probs, labels, pids


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_id: int,
    epochs: int,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    pos_weight: float,
    grad_accum_steps: int,
    grad_clip_norm: float,
    patience: int,
    output_dir: Path,
    seed: int = 42,
) -> TrainState:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / f"resnet25d_fold{fold_id}.pt"
    log_csv = output_dir / f"log_fold{fold_id}.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, memory_format=torch.channels_last)

    param_groups = discriminative_param_groups(model, lr_backbone=lr_backbone, lr_head=lr_head)
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    scaler = GradScaler()
    stopper = EarlyStopper(patience=patience, mode="max")

    best_val_auc = -1.0
    best_epoch = -1
    start = time.time()
    accum_counter = [0]

    with open(log_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_auc_patient", "lr_head", "secs"])

        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                tl = _step_train(
                    model, batch, loss_fn, optimizer, scaler, device,
                    accum_steps=grad_accum_steps, grad_clip_norm=grad_clip_norm,
                    accum_counter=accum_counter,
                )
                train_losses.append(tl)
            scheduler.step()
            val_loss, val_auc, _, _, _ = _evaluate(model, val_loader, device)
            epoch_secs = time.time() - epoch_start
            lr_head_now = optimizer.param_groups[1]["lr"]
            writer.writerow([epoch, np.mean(train_losses), val_loss, val_auc, lr_head_now, epoch_secs])
            fh.flush()
            log.info(
                "fold=%d epoch=%d train_loss=%.4f val_loss=%.4f val_auc=%.4f lr_h=%.2e (%.1fs)",
                fold_id, epoch, float(np.mean(train_losses)), val_loss, val_auc, lr_head_now, epoch_secs,
            )
            improved = stopper.step(val_auc)
            if improved:
                best_val_auc = val_auc
                best_epoch = epoch
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_auc": val_auc, "fold": fold_id},
                    ckpt,
                )
            if stopper.should_stop:
                log.info("Early stopping en epoch %d (mejor=%.4f en epoch %d)", epoch, best_val_auc, best_epoch)
                break

    wall = time.time() - start
    return TrainState(
        fold_id=fold_id,
        best_val_auc=best_val_auc,
        best_epoch=best_epoch,
        epochs_run=epoch + 1,
        wallclock_sec=wall,
        checkpoint_path=ckpt,
        log_csv_path=log_csv,
    )


def auto_pos_weight_from_loader(loader: DataLoader) -> float:
    """Calcula pos_weight escaneando el loader una vez."""
    labels: list[int] = []
    for batch in loader:
        labels.extend(int(x) for x in batch["label"].tolist())
    return class_pos_weight(np.array(labels))
