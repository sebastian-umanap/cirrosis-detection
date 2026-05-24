"""Inventario del dataset CirrMRI600+.

Layout real del dataset (verificado tras descargar y extraer desde OSF):

    data/CirrMRI600plus_raw/
    ├── CirrMRI600+_CompleteData_age_gender_evaluation.csv
    ├── T1_age_gender_evaluation.csv
    ├── T2_age_gender_evaluation.csv
    ├── T1&T2_Paired_age_gender_evaluation.csv
    ├── Healthy_demographics.csv
    ├── labels.txt
    ├── LICENSE.txt
    ├── Cirrhosis_T1_3D/
    │   ├── train_images/{patient_id}.nii.gz   (248)
    │   ├── train_masks/{patient_id}.nii.gz
    │   ├── valid_images/{patient_id}.nii.gz   (31)
    │   ├── valid_masks/{patient_id}.nii.gz
    │   ├── test_images/{patient_id}.nii.gz    (31)
    │   └── test_masks/{patient_id}.nii.gz
    ├── Cirrhosis_T2_3D/
    │   └── (mismo layout, 256/31/31)
    └── Healthy_subjects/
        ├── T1_W_Healthy/{T1_images,T1_masks}/{id}.nii.gz  (55)
        └── T2_W_Healthy/{T2_images,T2_masks}/{id}.nii.gz  (55)

El CSV de cirróticos tiene columnas:
    Patient ID, Age, Gender, Radiological Evaluation (1=Mild | 2=Moderate | 3=Severe)
El CSV de sanos:
    ID, Age, Gender (sin severidad — son controles)

Nuestra estrategia: consolidamos los splits oficiales (train/valid/test) en una sola
lista de PatientRecord, y hacemos nuestra propia 5-fold CV por paciente. El split
oficial 80:10:10 queda disponible vía la columna `official_split` para reproducir
el benchmark del paper si se quiere.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

log = logging.getLogger(__name__)

Modality = Literal["T1w", "T2w"]
Task = Literal["binary", "severity"]
OfficialSplit = Literal["train", "valid", "test", "healthy"]


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    modality: Modality
    image_path: Path
    mask_path: Path | None
    is_cirrhotic: bool
    severity: int | None      # 1=Mild, 2=Moderate, 3=Severe; None si sano.
    age: float | None
    sex: str | None           # 'M' o 'F'.
    official_split: OfficialSplit
    # Estos campos los reservamos por compatibilidad con el resto del código,
    # aunque el dataset publicado no los provee como columnas separadas:
    ascites: bool | None = None
    splenomegaly: bool | None = None
    varices: bool | None = None
    hcc: bool | None = None


@dataclass
class Inventory:
    records: list[PatientRecord]
    root: Path
    discovered_csvs: dict[str, Path] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])

    def filter(self, modality: Modality | None = None, task: Task = "binary") -> list[PatientRecord]:
        recs = [r for r in self.records if modality is None or r.modality == modality]
        if task == "binary":
            return recs
        return [r for r in recs if r.is_cirrhotic and r.severity is not None]

    def label_for(self, rec: PatientRecord, task: Task) -> int:
        if task == "binary":
            return int(rec.is_cirrhotic)
        if rec.severity is None:
            raise ValueError(f"{rec.patient_id}: severity is None para task='severity'")
        return rec.severity - 1   # 0/1/2 para clasificadores 3-clase.


_SEX_MAP = {"1": "F", "2": "M", 1: "F", 2: "M"}


def _norm(s: str) -> str:
    """Normaliza un nombre quitando puntuación, mayúsculas y separadores."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


_CSV_TARGETS = {
    "complete": "completedataagegenderevaluation",
    "t1": "t1agegenderevaluation",
    "t2": "t2agegenderevaluation",
    "paired": "pairedagegenderevaluation",
    "healthy": "healthydemographics",
    "labels": "labels",
}


def _find_csvs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    candidates = list(root.glob("*.csv")) + list(root.glob("*.txt"))
    for key, target in _CSV_TARGETS.items():
        for cand in candidates:
            if target in _norm(cand.stem):
                if key == "t1" and "paired" in _norm(cand.stem):
                    continue   # evitar match cruzado paired vs t1
                if key == "t2" and "paired" in _norm(cand.stem):
                    continue
                found[key] = cand
                break
    return found


def _read_csv_bom_safe(path: Path) -> pd.DataFrame:
    """Algunos CSV vienen con BOM UTF-8 que ensucia el primer nombre de columna."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def _scan_modality_folder(
    base: Path,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    img_suffix: str = "_images",
    msk_suffix: str = "_masks",
) -> list[tuple[str, str, Path, Path | None]]:
    """Devuelve [(split, patient_id, image_path, mask_path), ...]."""
    out: list[tuple[str, str, Path, Path | None]] = []
    for split in splits:
        img_dir = base / f"{split}{img_suffix}"
        msk_dir = base / f"{split}{msk_suffix}"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.nii*")):
            pid = img.name.replace(".nii.gz", "").replace(".nii", "")
            mask = msk_dir / img.name
            out.append((split, pid, img, mask if mask.exists() else None))
    return out


def _scan_healthy_folder(base: Path, modality_key: str) -> list[tuple[str, Path, Path | None]]:
    """Layout sanos: Healthy_subjects/T{N}_W_Healthy/T{N}_{images,masks}/{id}.nii.gz."""
    img_dir = base / f"{modality_key}_W_Healthy" / f"{modality_key}_images"
    msk_dir = base / f"{modality_key}_W_Healthy" / f"{modality_key}_masks"
    if not img_dir.exists():
        return []
    out = []
    for img in sorted(img_dir.glob("*.nii*")):
        pid = img.name.replace(".nii.gz", "").replace(".nii", "")
        mask = msk_dir / img.name
        out.append((pid, img, mask if mask.exists() else None))
    return out


def build_inventory(root: Path | str) -> Inventory:
    """Construye el inventario completo del dataset.

    Lanza FileNotFoundError si root no existe o si faltan los CSV esenciales.
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"No existe {root}")

    csvs = _find_csvs(root)
    if "complete" not in csvs and ("t1" not in csvs or "t2" not in csvs):
        raise FileNotFoundError(
            f"No se encontró CSV de cirróticos en {root}. "
            "Esperado: CirrMRI600+_CompleteData_age_gender_evaluation.csv "
            "o T1_age_gender_evaluation.csv + T2_age_gender_evaluation.csv."
        )
    cirr_meta = _read_csv_bom_safe(csvs.get("complete") or csvs["t1"])
    healthy_meta = _read_csv_bom_safe(csvs["healthy"]) if "healthy" in csvs else pd.DataFrame()
    log.info("CSVs detectados: %s", {k: v.name for k, v in csvs.items()})

    # Mapas patient_id -> metadata.
    def _build_meta_map(df: pd.DataFrame, id_col_candidates: tuple[str, ...]) -> dict[str, dict]:
        id_col = next((c for c in id_col_candidates if c in df.columns), None)
        if id_col is None:
            return {}
        out: dict[str, dict] = {}
        for _, row in df.iterrows():
            pid = str(int(row[id_col])) if pd.notna(row[id_col]) else None
            if pid is None:
                continue
            out[pid] = {
                "age": float(row.get("Age")) if pd.notna(row.get("Age")) else None,
                "sex": _SEX_MAP.get(int(row["Gender"]) if pd.notna(row.get("Gender")) else 0, None),
                "severity": int(row["Radiological Evaluation"])
                            if "Radiological Evaluation" in df.columns
                            and pd.notna(row["Radiological Evaluation"])
                            else None,
            }
        return out

    cirr_meta_map = _build_meta_map(cirr_meta, ("Patient ID", "PatientID", "ID"))
    healthy_meta_map = _build_meta_map(healthy_meta, ("ID", "Patient ID", "PatientID"))

    records: list[PatientRecord] = []

    # Cirróticos T1w y T2w.
    for modality, folder in (("T1w", "Cirrhosis_T1_3D"), ("T2w", "Cirrhosis_T2_3D")):
        base = root / folder
        if not base.exists():
            log.warning("No existe %s; saltando modalidad %s.", base, modality)
            continue
        for split, pid, img, mask in _scan_modality_folder(base):
            meta = cirr_meta_map.get(pid, {})
            records.append(
                PatientRecord(
                    patient_id=pid,
                    modality=modality,  # type: ignore[arg-type]
                    image_path=img,
                    mask_path=mask,
                    is_cirrhotic=True,
                    severity=meta.get("severity"),
                    age=meta.get("age"),
                    sex=meta.get("sex"),
                    official_split=split,  # type: ignore[arg-type]
                )
            )

    # Sanos.
    healthy_root = root / "Healthy_subjects"
    if healthy_root.exists():
        for modality, key in (("T1w", "T1"), ("T2w", "T2")):
            for pid, img, mask in _scan_healthy_folder(healthy_root, key):
                meta = healthy_meta_map.get(pid, {})
                records.append(
                    PatientRecord(
                        patient_id=f"healthy_{pid}",   # prefijo para no colisionar con cirróticos
                        modality=modality,  # type: ignore[arg-type]
                        image_path=img,
                        mask_path=mask,
                        is_cirrhotic=False,
                        severity=None,
                        age=meta.get("age"),
                        sex=meta.get("sex"),
                        official_split="healthy",
                    )
                )

    if not records:
        raise RuntimeError(f"Inventario vacío bajo {root}; revisar layout.")

    n_cirr = sum(r.is_cirrhotic for r in records)
    n_healthy = sum(not r.is_cirrhotic for r in records)
    n_t1 = sum(r.modality == "T1w" for r in records)
    n_t2 = sum(r.modality == "T2w" for r in records)
    log.info(
        "Inventario: %d records | cirrhotic=%d healthy=%d | T1w=%d T2w=%d",
        len(records), n_cirr, n_healthy, n_t1, n_t2,
    )
    return Inventory(records=records, root=root, discovered_csvs=csvs)


def sanity_check_volume(rec: PatientRecord) -> dict[str, object]:
    """Lee mínimamente el NIfTI y reporta shape, spacing y dtype — usado en EDA."""
    import nibabel as nib  # import lazy

    img = nib.load(str(rec.image_path))
    spacing = img.header.get_zooms()[:3]  # type: ignore[attr-defined]
    return {
        "patient_id": rec.patient_id,
        "modality": rec.modality,
        "shape": tuple(img.shape),
        "spacing": tuple(float(s) for s in spacing),
        "dtype": str(img.get_data_dtype()),
        "is_cirrhotic": rec.is_cirrhotic,
        "severity": rec.severity,
        "official_split": rec.official_split,
    }
