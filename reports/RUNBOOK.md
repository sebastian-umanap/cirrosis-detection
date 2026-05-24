# RUNBOOK — pasos para reproducir los resultados

Orden estricto. Cada paso reporta su propio tiempo aproximado.

## 0. Entorno (una sola vez)

```powershell
# Desde la raíz del proyecto cirrosis-detection
python -m pip install --upgrade pip

# Torch + CUDA 12.1 (~2.5 GB de download)
python -m pip install torch==2.3.1 torchvision==0.18.1 `
    --index-url https://download.pytorch.org/whl/cu121

# El resto de deps de training
python -m pip install -r requirements.txt
```

Verificación: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` debe imprimir `True NVIDIA GeForce RTX 3060`.

## 1. Pre-construir cache de volúmenes (~5–8 min, una sola vez)

Pre-procesa los 738 NIfTI (resampling 1×1×5 mm → clip percentiles → z-score) y los guarda como `.npz` comprimidos en `data/cache/`. Sin esto, cada batch de training releería NIfTI desde disco (~2-3 s/volumen) → inviable.

```powershell
python -m src.data.cache build --root data/CirrMRI600plus_raw --cache data/cache --workers 4
```

Espacio en disco esperado: ~2–3 GB de cache.

## 2. Smoke test del pipeline DL (~30 s)

```powershell
python scripts/smoke_test_dl.py
```

Debe imprimir `=== SMOKE TEST OK ===` y reportar:
- `CUDA=True`, device `RTX 3060`
- 2 batches procesados en pocos segundos
- VRAM usada (< 6 GB para batch=8 mínimo)

Si falla, **NO continuar** con el entrenamiento — revisar el traceback.

## 3. Baseline radiómico (CPU, ~30–60 min)

```powershell
python -m src.training train-radiomics --config configs/radiomics.yaml
```

Salida:
- `reports/tables/radiomics_results.csv` (AUC con IC 95% por modelo)
- `reports/tables/radiomics_{logreg,rf,xgb}_preds.csv`

## 4. Entrenamiento DL 2.5D ResNet-50 (~3–5 h con cache, 5-fold)

```powershell
python -m src.training train-dl --config configs/dl_2_5d.yaml
```

Salida:
- `reports/checkpoints/resnet25d_fold{0..4}.pt`
- `reports/checkpoints/log_fold{0..4}.csv` (loss/AUC por epoch)
- `reports/tables/dl_2_5d_preds.csv` (OOF predictions agregadas por paciente)
- `reports/timing.csv`

Si el entrenamiento se interrumpe a la mitad: borrar el checkpoint del fold incompleto y re-lanzar — los folds completos quedan guardados.

## 5. Evaluación final (CPU, < 5 min)

```powershell
python -m src.training evaluate --config configs/eval.yaml
```

Salida:
- `reports/tables/main_results.csv`  ← **va al paper como Tabla I**
- `reports/figures/roc_pr_curves.png` ← **va al paper como Figura 2**
- `reports/tables/delong.csv` ← DeLong test pareado entre modelos

## 6. App Streamlit (CPU, instantáneo si checkpoints listos)

```powershell
streamlit run app/streamlit_app.py
```

Abre <http://localhost:8501>. Sube un NIfTI del dataset (`data/CirrMRI600plus_raw/Cirrhosis_T2_3D/test_images/*.nii.gz`) y verifica que produce probabilidad + Grad-CAM en < 5 s.

## 7. Compilar el paper

```powershell
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Reemplazar los `\PEND{...}` rojos en `main.tex` con los valores de `reports/tables/main_results.csv` y `reports/tables/delong.csv` antes de la compilación final.

---

## Estado de ejecución (referencia)

El pipeline completo ya se ejecutó con éxito. Tiempos reales observados en RTX 3060 12 GB:
- Cache de 738 volúmenes: ~6 min (una sola vez).
- Radiomics 5-fold: ~5 min (CPU).
- DL 2.5D ResNet-50 5-fold: 2.71 h wall-clock (folds: 0.47/0.56/0.39/0.39/0.88 h).
- Evaluación + DeLong + Grad-CAM: < 2 min.

## Notas de reproducción

Si se quiere re-ejecutar más rápido a costa de rigor estadístico:
- `n_folds: 3` en `configs/data.yaml` reduce el tiempo de DL a ~1.6 h.
- `epochs` ya está en 15 con early-stopping (paciencia 4); rara vez llega a 15.

Si CUDA no está disponible en la máquina destino:
- La app y la evaluación corren en CPU sin problema (inferencia ~1-2 s/estudio).
- El re-entrenamiento DL en CPU es impráctico; usar los checkpoints provistos en `reports/checkpoints/`.
