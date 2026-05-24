"""Pipeline end-to-end CPU para inferencia desde un NIfTI o ZIP de DICOM.

Diseño orientado a O6: latencia < 5 s por estudio en CPU.
"""

from __future__ import annotations

import logging
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib  # type: ignore[import-not-found]
import numpy as np
import torch
from torch import nn

from src.evaluation.gradcam import GradCAM, mass_inside_mask, pointing_game, upsample_to
from src.models.cnn_2_5d import ResNet25D, build_resnet25d

log = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    proba_cirrhosis: float
    severity_probs: np.ndarray | None     # (3,) si tarea severidad disponible.
    liver_mask: np.ndarray                # 3D
    gradcam: np.ndarray                   # 2D (corte central)
    central_slice: np.ndarray             # 2D normalizado (corte central)
    liver_volume_ml: float
    inference_seconds: float
    pointing_inside_mask: bool
    cam_mass_inside_mask: float
    warnings: list[str]


def _read_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Lee un NIfTI con reorientación canónica RAS+ (idéntico al preprocessing de entrenamiento)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    arr = np.asarray(img.get_fdata(), dtype=np.float32)
    affine = img.affine
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])  # type: ignore[attr-defined]
    return arr, affine, zooms


def _resample_to(arr: np.ndarray, spacing: tuple[float, float, float],
                 target: tuple[float, float, float] = (1.0, 1.0, 5.0),
                 order: int = 1) -> np.ndarray:
    """Resampling a spacing objetivo vía scipy zoom — replica src/data/cache.py."""
    from scipy.ndimage import zoom
    factors = tuple(s / t for s, t in zip(spacing, target, strict=True))
    if all(abs(f - 1.0) <= 0.02 for f in factors):
        return arr
    return zoom(arr, factors, order=order, prefilter=False)


def _from_dicom_zip(zip_path: Path) -> Path:
    """Extrae un ZIP de DICOM, lo convierte a NIfTI, y devuelve el path al NIfTI temporal."""
    import dicom2nifti  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory() as td_in, tempfile.TemporaryDirectory(delete=False) as td_out:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td_in)
        dicom2nifti.convert_directory(td_in, td_out, compression=True)
        niftis = list(Path(td_out).glob("*.nii.gz")) + list(Path(td_out).glob("*.nii"))
        if not niftis:
            raise RuntimeError("ZIP de DICOM no produjo NIfTI; revisar contenido.")
        return niftis[0]


def _segment_liver_fallback(vol: np.ndarray) -> np.ndarray:
    """Segmentación trivial como fallback si no hay U-Net entrenado: ROI abdominal central.

    OJO: este fallback solo se usa si no se carga un segmentador real. Reporta como warning en la app.
    """
    d, h, w = vol.shape if vol.ndim == 3 else vol.shape[-3:]
    mask = np.zeros((d, h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    ry, rx = h // 4, w // 4
    yy, xx = np.ogrid[:h, :w]
    disk = ((yy - cy) ** 2 / (ry ** 2) + (xx - cx) ** 2 / (rx ** 2)) <= 1.0
    z_lo, z_hi = d // 4, 3 * d // 4
    mask[z_lo:z_hi, disk] = 1
    return mask


def _preprocess_volume(
    vol: np.ndarray,
    mask: np.ndarray | None,
    spacing: tuple[float, float, float],
    clip_pct: tuple[float, float] = (0.5, 99.5),
) -> tuple[np.ndarray, np.ndarray | None]:
    lo, hi = np.percentile(vol, clip_pct)
    v = np.clip(vol, lo, hi)
    mean = v[v > 0].mean() if (v > 0).any() else 0.0
    std = v[v > 0].std() if (v > 0).any() else 1.0
    v = (v - mean) / max(std, 1e-6)
    return v.astype(np.float32), mask


def _liver_bbox(mask: np.ndarray, margin: int = 16) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return slice(None), slice(None), slice(None)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0) + 1
    sl = []
    for i, dim in enumerate(mask.shape):
        sl.append(slice(max(0, mn[i] - margin), min(dim, mx[i] + margin)))
    return sl[0], sl[1], sl[2]


def _resize_2d(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    import torch.nn.functional as F
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


@dataclass
class InferencePipeline:
    classifier: ResNet25D
    severity_classifier: ResNet25D | None = None
    out_hw: tuple[int, int] = (224, 224)
    n_slices_stack: int = 3
    device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.eval().to(self.device)
        if self.severity_classifier is not None:
            self.severity_classifier.eval().to(self.device)
        self._cam = GradCAM(self.classifier, self.classifier.backbone.layer4)

    def __call__(self, nifti_path: Path, mask_path: Path | None = None) -> InferenceResult:
        t0 = time.time()
        warnings: list[str] = []
        path = Path(nifti_path)
        if path.suffix.lower() == ".zip":
            path = _from_dicom_zip(path)
        vol, _, spacing = _read_nifti(path)
        if vol.ndim != 3:
            raise ValueError(f"Volumen con ndim={vol.ndim}; se esperaba 3.")

        # Si se provee máscara hepática (recomendado: coincide con el entrenamiento),
        # se usa directamente. Si no, se cae al ROI abdominal central (out-of-distribution,
        # predicciones poco fiables — solo para demostrar el pipeline end-to-end).
        if mask_path is not None:
            try:
                m_vol, _, _ = _read_nifti(Path(mask_path))
                if m_vol.shape != vol.shape:
                    raise ValueError(f"shape de máscara {m_vol.shape} != imagen {vol.shape}")
                liver_mask = (m_vol > 0).astype(np.uint8)
                if liver_mask.sum() == 0:
                    raise ValueError("máscara vacía")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"No se pudo usar la máscara provista ({exc}); usando ROI fallback.")
                liver_mask = _segment_liver_fallback(vol)
        else:
            liver_mask = _segment_liver_fallback(vol)
            warnings.append(
                "Sin máscara hepática: usando ROI abdominal central (fallback). "
                "El modelo fue entrenado con recortes centrados en la máscara real, "
                "por lo que sin ella las predicciones son poco fiables (out-of-distribution). "
                "Para resultados válidos, suba también la máscara del hígado."
            )
        liver_volume_ml = float(liver_mask.sum() * np.prod(spacing) / 1000.0)

        # Resampling a 1x1x5 mm IDÉNTICO al preprocessing de entrenamiento (src/data/cache.py).
        # Sin esto, el modelo recibe slices con extensión física distinta -> out-of-distribution.
        vol = _resample_to(vol, spacing, target=(1.0, 1.0, 5.0), order=1)
        liver_mask = _resample_to(liver_mask.astype(np.float32), spacing, target=(1.0, 1.0, 5.0), order=0)
        liver_mask = (liver_mask > 0.5).astype(np.uint8)

        vol_n, _ = _preprocess_volume(vol, liver_mask, spacing)
        bbox = _liver_bbox(liver_mask)
        v_cropped = vol_n[bbox]
        m_cropped = liver_mask[bbox]
        # Selección de cortes axiales con hígado.
        depth = v_cropped.shape[-1] if v_cropped.ndim == 3 else 1
        if m_cropped.ndim == 3 and m_cropped.sum() > 0:
            z_with = np.unique(np.argwhere(m_cropped > 0)[:, -1])
            center_z = int(np.median(z_with))
        else:
            center_z = depth // 2
        half = self.n_slices_stack // 2
        z_start = max(0, center_z - half)
        z_end = min(depth, z_start + self.n_slices_stack)
        z_start = max(0, z_end - self.n_slices_stack)
        stack = []
        for z in range(z_start, z_end):
            slc = v_cropped[..., z]
            slc = _resize_2d(slc, self.out_hw)
            stack.append(slc)
        while len(stack) < self.n_slices_stack:
            stack.append(stack[-1])
        x = torch.from_numpy(np.stack(stack, axis=0)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(x)
        proba_cirr = float(torch.sigmoid(logits.squeeze()).cpu().item())

        cam = self._cam(x).cam
        central = stack[len(stack) // 2]
        cam_full = upsample_to(cam, central.shape)
        # Pointing game contra la máscara hepática (proyectada al corte central).
        if m_cropped.ndim == 3 and m_cropped.sum() > 0:
            mask_central = m_cropped[..., center_z]
            mask_central_r = _resize_2d(mask_central.astype(np.float32), central.shape) > 0.5
        else:
            mask_central_r = np.ones_like(central, dtype=bool)
        pi = pointing_game(cam_full, mask_central_r)
        cm = mass_inside_mask(cam_full, mask_central_r)

        # Severidad opcional.
        sev_probs: np.ndarray | None = None
        if self.severity_classifier is not None:
            with torch.no_grad():
                sev_logits = self.severity_classifier(x)
                sev_probs = torch.softmax(sev_logits, dim=-1).squeeze().cpu().numpy()

        return InferenceResult(
            proba_cirrhosis=proba_cirr,
            severity_probs=sev_probs,
            liver_mask=liver_mask,
            gradcam=cam_full,
            central_slice=central,
            liver_volume_ml=liver_volume_ml,
            inference_seconds=time.time() - t0,
            pointing_inside_mask=pi,
            cam_mass_inside_mask=cm,
            warnings=warnings,
        )


def load_pipeline(
    binary_ckpt: Path,
    severity_ckpt: Path | None = None,
    device: torch.device | None = None,
) -> InferencePipeline:
    """Carga checkpoints y devuelve el pipeline listo para inferencia."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = build_resnet25d(num_classes=1)
    state = torch.load(binary_ckpt, map_location=device)
    clf.load_state_dict(state["model"] if "model" in state else state)
    sev = None
    if severity_ckpt is not None and severity_ckpt.exists():
        sev = build_resnet25d(num_classes=3)
        state_s = torch.load(severity_ckpt, map_location=device)
        sev.load_state_dict(state_s["model"] if "model" in state_s else state_s)
    return InferencePipeline(classifier=clf, severity_classifier=sev, device=device)
