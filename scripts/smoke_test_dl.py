"""Smoke test del pipeline DL antes de lanzar la corrida completa.

Verifica:
    1. Importes (torch, monai, modelos, dataset).
    2. CUDA disponible y RTX 3060 con VRAM suficiente.
    3. Inventario carga.
    4. 1 batch de train_loader fluye y forward+backward funciona.
    5. Reporte de tiempos por batch.

Si todo pasa, listo para correr `python -m src.training train-dl --config configs/dl_2_5d.yaml`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main() -> int:
    print("\n=== SMOKE TEST DL ===\n")

    print("[1] Importes…")
    import torch
    from torch import nn
    from torch.cuda.amp import GradScaler, autocast
    print(f"    torch={torch.__version__}  CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    device={torch.cuda.get_device_name(0)}  vram_total={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("    !! CUDA no disponible — el smoke test correrá en CPU (será lento).")

    from src.data.inventory import build_inventory
    from src.data.dataset import CirrhosisDataModule
    from src.data.splits import make_stratified_group_kfold, class_pos_weight
    from src.models.cnn_2_5d import build_resnet25d
    from src.training.utils import set_seed
    import numpy as np

    set_seed(42)

    print("\n[2] Inventario…")
    root = PROJECT / "data" / "CirrMRI600plus_raw"
    cache = PROJECT / "data" / "cache"
    inv = build_inventory(root)
    recs = inv.filter(task="binary")
    print(f"    records={len(recs)}  cirr={sum(r.is_cirrhotic for r in recs)}  healthy={sum(not r.is_cirrhotic for r in recs)}")

    print("\n[3] Fold 0 + DataModule (cache opcional)…")
    fold = next(make_stratified_group_kfold(recs, task="binary", n_splits=5, seed=42))
    dm = CirrhosisDataModule(
        inv, fold, task="binary",
        batch_size=8, num_workers=0,           # 0 workers para smoke test sin overhead
        slices_per_patient_train=2,            # mínimo para test
        slices_per_patient_eval=2,
        cache_dir=cache if cache.exists() else None,
    )
    print(f"    train_recs={len(dm.train_recs)}  val_recs={len(dm.val_recs)}  cache={'sí' if dm.cache_dir else 'NO'}")

    print("\n[4] Modelo…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet25d(num_classes=1, in_channels=3, pretrained=True).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"    ResNet-50 2.5D | params={n:,}  device={device}")

    print("\n[5] Loop de 2 batches…")
    loader = dm.train_loader()
    y_train = np.array([int(r.is_cirrhotic) for r in dm.train_recs])
    pw = class_pos_weight(y_train)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    n_batches = 2
    t0 = time.time()
    it = iter(loader)
    for i in range(n_batches):
        t_b = time.time()
        batch = next(it)
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).float()
        with autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
            logits = model(image).squeeze(-1)
            loss = loss_fn(logits, label)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        print(f"    batch {i}: shape={tuple(image.shape)}  loss={loss.item():.4f}  ({time.time()-t_b:.2f}s)")
    print(f"    total elapsed: {time.time()-t0:.2f}s")

    if torch.cuda.is_available():
        print(f"\n[6] VRAM max usada: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n=== SMOKE TEST OK ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
