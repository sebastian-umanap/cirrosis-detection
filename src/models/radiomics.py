"""Baseline radiómico: features clásicos + clasificador shallow.

Versión sin PyRadiomics (que falla al compilar en Windows sin MSVC Build Tools).
Implementamos las clases de features más informativas con scikit-image y numpy:

    * First-order (intensidades dentro de la máscara): mean, std, mediana, p10/p90,
      skewness, kurtosis, energy, entropy.
    * GLCM (Gray-Level Co-occurrence Matrix) sobre el corte axial central enmascarado,
      con `skimage.feature.graycomatrix` y `graycoprops` (contraste, disimilaridad,
      homogeneidad, energía, correlación, ASM) en 4 ángulos.
    * LBP (Local Binary Patterns): histograma de patrones uniformes.
    * Shape (a partir de la máscara): volumen estimado, área superficial proxy,
      compactness, # cortes con hígado, ratio del bounding box.

Devuelve ~50 features por estudio. No tan rica como PyRadiomics (~1000 con wavelet)
pero suficiente como baseline interpretable para comparar contra el DL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib  # type: ignore[import-not-found]
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.inventory import PatientRecord

log = logging.getLogger(__name__)


def _quantize_for_glcm(img: np.ndarray, levels: int = 32) -> np.ndarray:
    """Lleva las intensidades a [0, levels-1] uint8 para `graycomatrix`."""
    arr = img.astype(np.float32)
    arr = arr - np.percentile(arr, 0.5)
    arr = np.clip(arr / max(np.percentile(arr, 99.5) - np.percentile(arr, 0.5), 1e-8), 0, 1)
    return (arr * (levels - 1)).astype(np.uint8)


def _first_order(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {k: 0.0 for k in ("mean", "std", "median", "p10", "p90",
                                  "skew", "kurt", "energy", "entropy", "min", "max")}
    v = values.astype(np.float64)
    hist, _ = np.histogram(v, bins=64, density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "skew": float(skew(v)),
        "kurt": float(kurtosis(v)),
        "energy": float(np.sum(v ** 2) / v.size),
        "entropy": entropy,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def _glcm_features(img2d: np.ndarray, mask2d: np.ndarray, levels: int = 32) -> dict[str, float]:
    """GLCM en el corte central. La máscara se usa para acotar la región."""
    img_q = _quantize_for_glcm(img2d, levels=levels)
    # Aplicar máscara: pixeles fuera = 0, ignorados en distancia gracias a `symmetric=True`.
    img_q = np.where(mask2d > 0, img_q, 0)
    distances = [1, 3]
    angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(img_q, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    out: dict[str, float] = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        vals = graycoprops(glcm, prop=prop)
        out[f"glcm_{prop}_mean"] = float(vals.mean())
        out[f"glcm_{prop}_std"] = float(vals.std())
    return out


def _lbp_features(img2d: np.ndarray, mask2d: np.ndarray, P: int = 8, R: int = 1) -> dict[str, float]:
    img_q = _quantize_for_glcm(img2d, levels=256)
    lbp = local_binary_pattern(img_q, P=P, R=R, method="uniform")
    lbp = lbp[mask2d > 0]
    n_bins = P + 2
    if lbp.size == 0:
        return {f"lbp_{i}": 0.0 for i in range(n_bins)}
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return {f"lbp_{i}": float(hist[i]) for i in range(n_bins)}


def _shape_features(mask3d: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    if mask3d.sum() == 0:
        return {k: 0.0 for k in ("liver_volume_ml", "n_slices_with_liver",
                                  "bbox_volume_ml", "compactness",
                                  "axial_bbox_aspect_ratio")}
    voxel_vol_ml = float(np.prod(spacing) / 1000.0)
    n_vox = float(mask3d.sum())
    liver_vol = n_vox * voxel_vol_ml
    z_with = int((mask3d.sum(axis=(0, 1)) > 0).sum())
    coords = np.argwhere(mask3d > 0)
    bb_lo = coords.min(axis=0); bb_hi = coords.max(axis=0) + 1
    bb_size = bb_hi - bb_lo
    bb_vol = float(np.prod(bb_size) * voxel_vol_ml)
    compactness = float(n_vox / max(np.prod(bb_size), 1.0))
    aspect = float(bb_size[0]) / max(float(bb_size[1]), 1.0)
    return {
        "liver_volume_ml": liver_vol,
        "n_slices_with_liver": float(z_with),
        "bbox_volume_ml": bb_vol,
        "compactness": compactness,
        "axial_bbox_aspect_ratio": aspect,
    }


def extract_features_for_record(rec: PatientRecord) -> dict[str, Any]:
    """Extrae features radiómicos clásicos de un volumen + máscara."""
    if rec.mask_path is None:
        raise ValueError(f"{rec.patient_id}: sin máscara")

    img = nib.load(str(rec.image_path)).get_fdata().astype(np.float32)
    mask = nib.load(str(rec.mask_path)).get_fdata().astype(np.uint8)
    spacing = tuple(float(s) for s in nib.load(str(rec.image_path)).header.get_zooms()[:3])  # type: ignore[attr-defined]

    feats: dict[str, Any] = {
        "patient_id": rec.patient_id,
        "modality": rec.modality,
        "label": int(rec.is_cirrhotic),
    }
    if rec.severity is not None:
        feats["severity"] = int(rec.severity)

    # First-order sobre todos los voxeles dentro de la máscara.
    masked_values = img[mask > 0]
    for k, v in _first_order(masked_values).items():
        feats[f"fo_{k}"] = v

    # Corte axial central (con más hígado).
    if mask.sum() == 0:
        for k in ("glcm_contrast_mean", "glcm_contrast_std", "glcm_dissimilarity_mean",
                  "glcm_dissimilarity_std", "glcm_homogeneity_mean", "glcm_homogeneity_std",
                  "glcm_energy_mean", "glcm_energy_std", "glcm_correlation_mean",
                  "glcm_correlation_std", "glcm_ASM_mean", "glcm_ASM_std"):
            feats[k] = 0.0
        for i in range(10):
            feats[f"lbp_{i}"] = 0.0
    else:
        # Selección del corte con mayor área hepática.
        z_areas = mask.sum(axis=(0, 1))
        z_center = int(np.argmax(z_areas))
        img2d = img[..., z_center]
        mask2d = mask[..., z_center]
        for k, v in _glcm_features(img2d, mask2d).items():
            feats[k] = v
        for k, v in _lbp_features(img2d, mask2d).items():
            feats[k] = v

    # Shape (3D).
    for k, v in _shape_features(mask, spacing).items():
        feats[f"shape_{k}"] = v

    return feats


# ---------------------------------------------------------------------------
# Pipeline de selección + clasificador, compatible con sklearn.
# ---------------------------------------------------------------------------


@dataclass
class RadiomicsPipeline:
    k_features: int = 30
    selection: str = "lasso"   # lasso | univariate_f
    classifier_name: str = "logreg"
    classifier_kwargs: dict[str, Any] = field(default_factory=lambda: {"C": 1.0, "max_iter": 2000})
    random_state: int = 42
    _scaler: StandardScaler | None = None
    _clf: Any = None
    _selected_feature_names: list[str] = field(default_factory=list)

    def _build_classifier(self) -> Any:
        # Los hyperparámetros incluido class_weight/balanced vienen del YAML vía
        # `classifier_kwargs`; aquí solo añadimos random_state como default.
        kwargs = {"random_state": self.random_state, **self.classifier_kwargs}
        if self.classifier_name == "logreg":
            return LogisticRegression(**kwargs)
        if self.classifier_name == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**kwargs)
        if self.classifier_name == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(eval_metric="logloss", **kwargs)
        raise ValueError(f"Classifier desconocido: {self.classifier_name}")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RadiomicsPipeline":
        feat_cols = [c for c in X.columns if c not in {"patient_id", "modality", "label", "severity"}]
        Xv = X[feat_cols].to_numpy(dtype=float)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)

        if self.selection == "lasso":
            scaler_pre = StandardScaler()
            Xs = scaler_pre.fit_transform(Xv)
            lasso = LassoCV(cv=3, random_state=self.random_state, n_jobs=-1, max_iter=5000)
            lasso.fit(Xs, y)
            coefs = np.abs(lasso.coef_)
            top_idx = np.argsort(coefs)[::-1]
            nonzero = top_idx[coefs[top_idx] > 0]
            k = min(self.k_features, max(1, len(nonzero)))
            chosen = nonzero[:k] if len(nonzero) >= 1 else top_idx[: self.k_features]
        elif self.selection == "univariate_f":
            kbest = SelectKBest(score_func=f_classif, k=min(self.k_features, Xv.shape[1]))
            kbest.fit(Xv, y)
            chosen = np.where(kbest.get_support())[0]
        else:
            raise ValueError(f"Selección desconocida: {self.selection}")

        self._selected_feature_names = [feat_cols[i] for i in chosen]
        log.info("Radiomics: %d features seleccionados (de %d)", len(chosen), len(feat_cols))

        Xc = Xv[:, chosen]
        self._scaler = StandardScaler().fit(Xc)
        Xcs = self._scaler.transform(Xc)
        self._clf = self._build_classifier()
        self._clf.fit(Xcs, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        feat_cols = [c for c in X.columns if c not in {"patient_id", "modality", "label", "severity"}]
        Xv = X[feat_cols].to_numpy(dtype=float)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        idx = [feat_cols.index(n) for n in self._selected_feature_names]
        Xc = Xv[:, idx]
        assert self._scaler is not None and self._clf is not None
        Xcs = self._scaler.transform(Xc)
        return self._clf.predict_proba(Xcs)

    def feature_importance(self) -> pd.DataFrame:
        assert self._clf is not None
        names = self._selected_feature_names
        if hasattr(self._clf, "feature_importances_"):
            imp = self._clf.feature_importances_
        elif hasattr(self._clf, "coef_"):
            imp = np.abs(self._clf.coef_).ravel()
        else:
            imp = np.zeros(len(names))
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)


def features_csv_path(cache_dir: Path, modality: str, task: str) -> Path:
    return cache_dir / f"radiomics_features_{modality}_{task}.csv"
