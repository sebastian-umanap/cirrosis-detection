"""Pipelines de preprocessing y augmentation con MONAI/TorchIO.

Justificación en reports/02_methodology_crispmlq.md §Fase 2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from monai.transforms import (  # type: ignore[import-not-found]
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)


def build_preprocess(
    out_spacing: tuple[float, float, float] = (1.0, 1.0, 3.0),
    clip_percentiles: tuple[float, float] = (0.5, 99.5),
    has_mask: bool = True,
) -> Compose:
    """Pipeline determinístico (NO aleatorio) aplicado tanto en train como en eval.

    Pasos: load → ensure-channel-first → reorient RAS → spacing → clip por percentil → z-score.
    Devuelve dict con keys "image" y opcionalmente "mask".
    """
    keys = ["image", "mask"] if has_mask else ["image"]
    mode = ("bilinear", "nearest") if has_mask else ("bilinear",)
    transforms: list[Any] = [
        LoadImaged(keys=keys, ensure_channel_first=False, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=out_spacing, mode=mode),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=clip_percentiles[0],
            upper=clip_percentiles[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    return Compose(transforms)


def build_train_augment(
    rotation_deg: float = 10.0,
    prob_flip_h: float = 0.5,
    gaussian_noise_std: float = 0.05,
    out_size: tuple[int, int, int] | None = None,
    has_mask: bool = True,
) -> Compose:
    """Augmentation estocástica solo en train. No aplica intensity clipping (ya está en preprocess)."""
    keys = ["image", "mask"] if has_mask else ["image"]
    mode = ("bilinear", "nearest") if has_mask else ("bilinear",)
    augs: list[Any] = [
        RandAffined(
            keys=keys,
            prob=0.7,
            rotate_range=(0.0, 0.0, np.deg2rad(rotation_deg)),
            scale_range=(0.05, 0.05, 0.0),
            mode=mode,
            padding_mode="zeros",
        ),
        RandGaussianNoised(keys=["image"], prob=0.3, std=gaussian_noise_std),
    ]
    if out_size is not None:
        augs.insert(0, RandSpatialCropd(keys=keys, roi_size=out_size, random_size=False))
    # Flip horizontal: en RAS+ el eje sagital (L-R) es el primero (índice 0); flip H = flip eje 0.
    if prob_flip_h > 0:
        from monai.transforms import RandFlipd  # type: ignore[import-not-found]
        augs.append(RandFlipd(keys=keys, prob=prob_flip_h, spatial_axis=0))
    return Compose(augs)
