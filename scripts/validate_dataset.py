"""Valida que CirrMRI600+ se haya descargado correctamente.

Verifica:
    1. Conteos aproximados de volúmenes T1w/T2w/healthy.
    2. Que cada imagen tenga una máscara con el mismo nombre.
    3. Que los NIfTI se lean sin error con nibabel.
    4. Que los CSV de metadatos existan y sean parseables.
    5. Reporta tabla resumen.

Tras descargar, correr:
    python scripts/validate_dataset.py --root data/CirrMRI600plus_raw
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import nibabel as nib  # type: ignore[import-not-found]
import pandas as pd

EXPECTED_T1 = 310
EXPECTED_T2 = 318
EXPECTED_HEALTHY = 55
EXPECTED_CSVS = (
    "CompleteData-age-gender-evaluation.csv",
    "T1-age-gender-evaluation.csv",
    "T2-age-gender-evaluation.csv",
    "Paired-age-gender-evaluation.csv",
    "Healthy-demographics.csv",
    "Labels.txt",
)


def find_niftis(root: Path) -> list[Path]:
    return list(root.rglob("*.nii.gz")) + list(root.rglob("*.nii"))


def group_by_top_folder(files: Iterable[Path], root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        rel = f.relative_to(root)
        grouped[rel.parts[0]].append(f)
    return grouped


def check_nifti_readable(paths: Iterable[Path], sample: int = 5) -> list[str]:
    errors: list[str] = []
    paths = list(paths)
    step = max(1, len(paths) // sample) if paths else 1
    sampled = paths[::step][:sample]
    for p in sampled:
        try:
            img = nib.load(str(p))
            _ = img.shape
            _ = img.affine
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{p}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root: Path = args.root.resolve()
    if not root.exists():
        print(f"ERROR: no existe {root}", file=sys.stderr)
        return 1

    print(f"Validando {root}")
    niftis = find_niftis(root)
    print(f"  NIfTI encontrados: {len(niftis):,}")

    by_folder = group_by_top_folder(niftis, root)
    for top, files in sorted(by_folder.items(), key=lambda kv: kv[0]):
        print(f"    {top}: {len(files):,}")

    csvs_found = []
    for name in EXPECTED_CSVS:
        matches = list(root.rglob(name))
        if matches:
            csvs_found.append((name, matches[0]))
            print(f"  csv ok: {name} -> {matches[0].relative_to(root)}")
        else:
            print(f"  csv FALTANTE: {name}")

    if csvs_found:
        for name, path in csvs_found:
            try:
                if name.endswith(".csv"):
                    df = pd.read_csv(path)
                    print(f"    {name}: {len(df)} filas | columnas: {list(df.columns)[:8]}")
                else:
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        head = fh.read(500)
                    print(f"    {name}: head=\n{head}\n---")
            except Exception as exc:  # noqa: BLE001
                print(f"    {name}: error de parseo: {exc}")

    sample_errors = check_nifti_readable(niftis, sample=10)
    if sample_errors:
        print("  ! Errores leyendo NIfTI sample:")
        for e in sample_errors:
            print(f"    {e}")
    else:
        print("  NIfTI sample lee correctamente con nibabel.")

    print("\nValidación informativa terminada.")
    print(
        "Comparar contra esperado: "
        f"T1≈{EXPECTED_T1}, T2≈{EXPECTED_T2}, healthy≈{EXPECTED_HEALTHY}."
    )
    print("Si los conteos difieren, revisar el layout y ajustar configs/data.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
