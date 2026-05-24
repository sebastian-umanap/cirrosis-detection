"""Dataset y DataModule sobre CirrMRI600+.

Soporta:
    - Tarea binaria (cirrótico vs sano) o de severidad (1/2/3).
    - Modo "2.5D" (stacks de cortes axiales) y "volumen completo" (para baseline radiómico).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.cache import cache_path_for, load_cached
from src.data.inventory import Inventory, PatientRecord, Task
from src.data.splits import Fold

log = logging.getLogger(__name__)

Mode = Literal["2_5d", "volume"]


@dataclass
class Sample:
    image: torch.Tensor       # (C, H, W) en modo 2.5D | (1, D, H, W) en modo volumen.
    label: int
    patient_id: str
    modality: str
    slice_indices: tuple[int, ...] = ()


class CirrhosisDataset(Dataset[Sample]):
    """Acepta lista de PatientRecord + tarea + modo y emite tensores listos para el modelo."""

    def __init__(
        self,
        records: Sequence[PatientRecord],
        inventory: Inventory,
        task: Task = "binary",
        mode: Mode = "2_5d",
        n_slices_stack: int = 3,
        slices_per_patient: int = 16,
        out_hw: tuple[int, int] = (224, 224),
        out_spacing: tuple[float, float, float] = (1.0, 1.0, 5.0),
        clip_percentiles: tuple[float, float] = (0.5, 99.5),
        augment: bool = False,
        crop_around_mask: bool = True,
        crop_margin: int = 16,
        seed: int = 42,
        cache_dir: Path | None = None,
    ) -> None:
        self.records = list(records)
        self.inventory = inventory
        self.task = task
        self.mode = mode
        self.n_slices_stack = n_slices_stack
        self.slices_per_patient = slices_per_patient
        self.out_hw = out_hw
        self.crop_around_mask = crop_around_mask
        self.crop_margin = crop_margin
        self.augment = augment
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.out_spacing = out_spacing
        self.clip_percentiles = clip_percentiles
        # Build_preprocess/augment se cargan lazy solo si NO hay cache.
        self._preprocess = None
        self._augment_fn = None

    def __len__(self) -> int:
        if self.mode == "volume":
            return len(self.records)
        return len(self.records) * self.slices_per_patient

    def _ensure_monai_pipelines(self) -> None:
        if self._preprocess is None:
            from src.data.transforms import build_preprocess
            self._preprocess = build_preprocess(out_spacing=self.out_spacing, clip_percentiles=self.clip_percentiles)
        if self.augment and self._augment_fn is None:
            from src.data.transforms import build_train_augment
            self._augment_fn = build_train_augment()

    def _load(self, rec: PatientRecord) -> dict[str, np.ndarray]:
        # Camino rápido: cache pre-procesado en disco.
        if self.cache_dir is not None:
            cached = load_cached(self.cache_dir, rec.patient_id, rec.modality)
            if cached is not None:
                img = cached.image[None].astype(np.float32)   # (1, X, Y, Z)
                mask = cached.mask[None].astype(np.uint8)     # (1, X, Y, Z)
                return {"image": img, "mask": mask}
        # Camino lento: re-preprocesar con MONAI cada vez.
        self._ensure_monai_pipelines()
        item: dict[str, str] = {"image": str(rec.image_path)}
        if rec.mask_path is not None:
            item["mask"] = str(rec.mask_path)
        out = self._preprocess(item)
        if self._augment_fn is not None:
            out = self._augment_fn(out)
        np_out: dict[str, np.ndarray] = {}
        for k, v in out.items():
            if hasattr(v, "numpy"):
                np_out[k] = v.numpy()  # type: ignore[assignment]
            elif isinstance(v, np.ndarray):
                np_out[k] = v
        return np_out

    def _liver_bbox(self, mask: np.ndarray) -> tuple[slice, slice, slice] | None:
        if mask.sum() == 0:
            return None
        m = mask[0] if mask.ndim == 4 else mask
        coords = np.argwhere(m > 0)
        mn = coords.min(axis=0)
        mx = coords.max(axis=0) + 1
        margin = self.crop_margin
        sl = []
        for i in range(3):
            sl.append(slice(max(0, mn[i] - margin), min(m.shape[i], mx[i] + margin)))
        return sl[0], sl[1], sl[2]

    @staticmethod
    def _resize_2d(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        import torch.nn.functional as F
        t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
        return t.squeeze().numpy()

    def _pick_center_z(self, z_with_liver: np.ndarray, index: int) -> int:
        """Selecciona el corte central del stack 2.5D.

        - En train (`augment=True`): muestreo uniforme aleatorio dentro del rango con hígado;
          esto hace que cada `index` produzca un stack distinto (anti-overfit y mejor uso del
          presupuesto de 16 stacks/patiente).
        - En eval (`augment=False`): muestreo determinístico — cubre uniformemente el rango con
          hígado a lo largo de `self.slices_per_patient` posiciones, lo que reduce varianza
          entre corridas y permite agregar la probabilidad por paciente con media estable.
        """
        if len(z_with_liver) == 0:
            return 0
        if self.augment:
            rng = np.random.default_rng((self.seed + index * 977) & 0xFFFFFFFF)
            return int(rng.choice(z_with_liver))
        n = max(1, self.slices_per_patient)
        k = index % n
        # Posición uniforme entre quantiles (k+0.5)/n para que cubra el rango.
        q = (k + 0.5) / n
        return int(np.quantile(z_with_liver, q))

    def _augment_stack_2d(self, stack: np.ndarray, index: int) -> np.ndarray:
        """Augmentación 2D ligera aplicada al stack 2.5D (en cache no aplicamos MONAI 3D).

        Operaciones rápidas en numpy/torch:
          - Random horizontal flip (eje x).
          - Random rotación ±rotation_deg (vía torch grid_sample).
          - Gaussian noise σ≤gaussian_noise_std.
          - Gamma jitter γ∈[0.8, 1.2] sobre intensidades centradas en 0.
        """
        if not self.augment:
            return stack
        rng = np.random.default_rng((self.seed + index * 9871) & 0xFFFFFFFF)

        # Flip horizontal.
        if rng.random() < 0.5:
            stack = stack[..., ::-1].copy()

        # Rotación pequeña.
        rot_deg = float(rng.uniform(-10.0, 10.0))
        if abs(rot_deg) > 0.1:
            import torch.nn.functional as F
            theta_rad = np.deg2rad(rot_deg)
            cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
            theta = torch.tensor([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]], dtype=torch.float32).unsqueeze(0)
            t = torch.from_numpy(stack).float().unsqueeze(0)
            grid = F.affine_grid(theta, t.shape, align_corners=False)
            t = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            stack = t.squeeze(0).numpy()

        # Ruido gaussiano.
        sigma = float(rng.uniform(0.0, 0.05))
        if sigma > 0.005:
            stack = stack + rng.normal(0, sigma, size=stack.shape).astype(np.float32)

        # Gamma jitter.
        gamma = float(rng.uniform(0.85, 1.15))
        if abs(gamma - 1.0) > 0.01:
            # Aplica gamma manteniendo signo (intensidades z-score pueden ser negativas).
            sign = np.sign(stack); mag = np.abs(stack)
            stack = sign * np.power(mag + 1e-6, gamma)
        return stack.astype(np.float32)

    def _build_2_5d(self, rec: PatientRecord, index: int) -> Sample:
        loaded = self._load(rec)
        vol = loaded["image"][0]  # (X, Y, Z) tras Spacingd; el axial es plano XY recorrido en Z.
        mask = loaded.get("mask")
        if mask is not None and mask.ndim == 4:
            mask = mask[0]
        # Crop alrededor del hígado.
        if self.crop_around_mask and mask is not None:
            bbox = self._liver_bbox(mask)
            if bbox is not None:
                vol = vol[bbox]
                mask = mask[bbox]
        depth = vol.shape[-1]
        if mask is not None and mask.sum() > 0:
            z_with_liver = np.unique(np.argwhere(mask > 0)[:, -1])
        else:
            z_with_liver = np.arange(depth)
        center_z = self._pick_center_z(z_with_liver, index)
        half = self.n_slices_stack // 2
        z_start = max(0, center_z - half)
        z_end = min(depth, z_start + self.n_slices_stack)
        z_start = max(0, z_end - self.n_slices_stack)
        stack = []
        for z in range(z_start, z_end):
            slc = vol[:, :, z]
            slc = self._resize_2d(slc, self.out_hw)
            stack.append(slc)
        while len(stack) < self.n_slices_stack:
            stack.append(stack[-1])
        arr = np.stack(stack, axis=0).astype(np.float32)
        # Augmentación de slice-level si entrenamiento + cache. (En path MONAI, augment ya viene aplicado en _load.)
        if self.augment and self.cache_dir is not None:
            arr = self._augment_stack_2d(arr, index=index)
        label = self.inventory.label_for(rec, self.task)
        return Sample(
            image=torch.from_numpy(arr),
            label=label,
            patient_id=rec.patient_id,
            modality=rec.modality,
            slice_indices=tuple(range(z_start, z_end)),
        )

    def __getitem__(self, index: int) -> Sample:
        if self.mode == "volume":
            rec = self.records[index]
            loaded = self._load(rec)
            vol = loaded["image"]
            label = self.inventory.label_for(rec, self.task)
            return Sample(
                image=torch.from_numpy(vol).float(),
                label=label,
                patient_id=rec.patient_id,
                modality=rec.modality,
            )
        rec_idx = index // self.slices_per_patient
        rec = self.records[rec_idx]
        return self._build_2_5d(rec, index=index)


def make_loader(ds: CirrhosisDataset, batch_size: int = 16, shuffle: bool = False, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
    )


def _collate(batch: list[Sample]) -> dict[str, torch.Tensor | list[str]]:
    images = torch.stack([b.image for b in batch], dim=0)
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    return {
        "image": images,
        "label": labels,
        "patient_id": [b.patient_id for b in batch],
        "modality": [b.modality for b in batch],
    }


@dataclass
class CirrhosisDataModule:
    """Empaqueta inventario + fold seleccionado en train/val DataLoaders."""

    inventory: Inventory
    fold: Fold
    task: Task = "binary"
    mode: Mode = "2_5d"
    batch_size: int = 16
    num_workers: int = 4
    slices_per_patient_train: int = 16
    slices_per_patient_eval: int = 32
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        all_recs = self.inventory.filter(task=self.task)
        self.train_recs = [all_recs[i] for i in self.fold.train_idx]
        self.val_recs = [all_recs[i] for i in self.fold.val_idx]

    def train_loader(self) -> DataLoader:
        ds = CirrhosisDataset(
            self.train_recs, self.inventory, task=self.task, mode=self.mode,
            slices_per_patient=self.slices_per_patient_train, augment=True,
            cache_dir=self.cache_dir,
        )
        return make_loader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_loader(self) -> DataLoader:
        ds = CirrhosisDataset(
            self.val_recs, self.inventory, task=self.task, mode=self.mode,
            slices_per_patient=self.slices_per_patient_eval, augment=False,
            cache_dir=self.cache_dir,
        )
        return make_loader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
