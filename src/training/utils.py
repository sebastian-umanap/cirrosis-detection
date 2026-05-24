"""Utilidades de reproducibilidad y diagnóstico de entrenamiento."""

from __future__ import annotations

import logging
import os
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fija seeds en torch, numpy, random y PYTHONHASHSEED — regla de reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info("Seed=%d deterministic=%s", seed, deterministic)


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Devuelve (total, entrenables) parámetros."""
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train


class EarlyStopper:
    """Early stopping basado en métrica de validación (modo max o min)."""

    def __init__(self, patience: int = 6, mode: str = "max", min_delta: float = 1e-4) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode debe ser 'max' o 'min', no {mode}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        improved = (
            self.best is None
            or (self.mode == "max" and metric > self.best + self.min_delta)
            or (self.mode == "min" and metric < self.best - self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def discriminative_param_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    backbone_attr: str = "backbone",
    head_attrs: Iterable[str] = ("head", "attn_pool"),
) -> list[dict]:
    """LR distintos para backbone (más bajo) vs head (más alto). Justificado en reports/02 §Fase 3."""
    backbone_params = list(getattr(model, backbone_attr).parameters())
    head_params: list = []
    for attr in head_attrs:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            head_params.extend(getattr(model, attr).parameters())
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
