"""Cache de volúmenes pre-procesados a `.npz` para acelerar el entrenamiento.

Sin cache, cada `__getitem__` del Dataset releería el NIfTI + correría MONAI Spacing/Clip/Norm
(~2-3 s por volumen). Con `slices_per_patient_train=16` esto multiplica los tiempos de epoch
hasta hacerlo impráctico (~10x más lento) en una GPU de consumo.

Solución: pre-procesar TODOS los volúmenes una vez (script-CLI), guardarlos como `.npz`
comprimidos en `data/cache/`. El Dataset lee del cache cuando existe.

Uso:
    python -m src.data.cache build --root data/CirrMRI600plus_raw --cache data/cache --workers 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedVolume:
    image: np.ndarray   # (X, Y, Z) float16 post-preprocess (z-score).
    mask: np.ndarray    # (X, Y, Z) uint8.
    spacing: tuple[float, float, float]


def cache_path_for(cache_dir: Path, patient_id: str, modality: str) -> Path:
    return cache_dir / f"{patient_id}__{modality}.npz"


def _preprocess_one(args: tuple[Path, Path, Path, tuple[float, float, float], tuple[float, float]]) -> tuple[str, bool, str]:
    """Worker: lee NIfTI, aplica preprocess, guarda npz."""
    image_path, mask_path, out_path, target_spacing, clip_pct = args
    try:
        import nibabel as nib  # local imports en worker para que pickling no rompa
        import numpy as np
        from scipy.ndimage import zoom

        img_n = nib.load(str(image_path))
        img = img_n.get_fdata().astype(np.float32)
        # Re-orient a RAS+
        img_n_canon = nib.as_closest_canonical(img_n)
        img = img_n_canon.get_fdata().astype(np.float32)
        spacing = tuple(float(s) for s in img_n_canon.header.get_zooms()[:3])  # type: ignore[attr-defined]

        if mask_path is not None and Path(mask_path).exists():
            msk_n = nib.as_closest_canonical(nib.load(str(mask_path)))
            mask = msk_n.get_fdata().astype(np.uint8)
        else:
            mask = np.zeros(img.shape, dtype=np.uint8)

        # Resampling a target_spacing (mantiene shape proporcional).
        zoom_factors = tuple(s / t for s, t in zip(spacing, target_spacing, strict=True))
        if any(abs(z - 1.0) > 0.02 for z in zoom_factors):
            img = zoom(img, zoom_factors, order=1, prefilter=False)
            mask = zoom(mask, zoom_factors, order=0, prefilter=False).astype(np.uint8)

        # Clipping percentile + z-score por volumen.
        lo, hi = np.percentile(img, clip_pct)
        img = np.clip(img, lo, hi)
        nz = img > 0
        if nz.any():
            mean = img[nz].mean(); std = img[nz].std() or 1.0
            img = (img - mean) / std
        img = img.astype(np.float16)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, image=img, mask=mask,
                            spacing=np.array(target_spacing, dtype=np.float32))
        return (image_path.name, True, "")
    except Exception as exc:  # noqa: BLE001
        return (image_path.name, False, str(exc))


def build_cache(
    inventory,
    cache_dir: Path,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 5.0),
    clip_pct: tuple[float, float] = (0.5, 99.5),
    workers: int = 2,
    skip_existing: bool = True,
) -> dict[str, int]:
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    skipped = 0
    for rec in inventory.records:
        out_path = cache_path_for(cache_dir, rec.patient_id, rec.modality)
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        jobs.append((rec.image_path, rec.mask_path, out_path, target_spacing, clip_pct))
    log.info("cache: %d a generar, %d ya existen", len(jobs), skipped)

    ok = 0; fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_preprocess_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            name, success, err = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                log.warning("fail %s: %s", name, err)
            if (i + 1) % 25 == 0 or i == len(jobs) - 1:
                log.info("  %d/%d (ok=%d fail=%d)", i + 1, len(jobs), ok, fail)
    return {"new": ok, "skipped": skipped, "failed": fail}


def load_cached(cache_dir: Path, patient_id: str, modality: str) -> CachedVolume | None:
    p = cache_path_for(cache_dir, patient_id, modality)
    if not p.exists():
        return None
    data = np.load(p)
    return CachedVolume(
        image=data["image"],
        mask=data["mask"],
        spacing=tuple(float(s) for s in data["spacing"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", type=Path, default=Path("data/CirrMRI600plus_raw"))
    build.add_argument("--cache", type=Path, default=Path("data/cache"))
    build.add_argument("--workers", type=int, default=2)
    build.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 5.0))
    build.add_argument("--force", action="store_true", help="reconstruir aunque exista cache")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.inventory import build_inventory  # noqa: E402

    inv = build_inventory(args.root)
    stats = build_cache(
        inv,
        cache_dir=args.cache,
        target_spacing=tuple(args.spacing),
        workers=args.workers,
        skip_existing=not args.force,
    )
    log.info("listo: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
