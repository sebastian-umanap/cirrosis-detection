---
title: Detección de cirrosis · RM abdominal
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cirrosis avanzada en RM con ResNet-50 2.5D sobre CirrMRI600+.
---

# Detección de cirrosis hepática avanzada en RM abdominal

Proyecto final · **ISIS-4825 Imágenes y Visión** · Universidad de los Andes (202610)
Autores: **Sebastian Umaña Peinado**, **Jerónimo Franco Castañeda**

> ⚠️ **Prototipo académico.** No apto para uso clínico ni diagnóstico real.

---

## Resumen

Pipeline reproducible de deep learning para clasificación binaria cirrótico vs sano sobre el dataset público **CirrMRI600+** (Choi/Jha et al., *Sci Data* 2025; DOI [10.1038/s41597-025-05201-7](https://doi.org/10.1038/s41597-025-05201-7)). Incluye:

- Preprocessing reproducible (RAS → resample 1×1×5 mm → clip percentil → z-score → crop hepático) con cache.
- Baseline radiómico (scikit-image: first-order + GLCM + LBP + shape, 39 features) con LASSO + LogReg/RF/XGB.
- Modelo principal: **2.5D ResNet-50** con attention pooling y transfer learning desde ImageNet.
- 5-fold CV estratificada **por paciente**, bootstrap CIs y test de DeLong.
- Auditoría de interpretabilidad a gran escala (**n=738**) con Grad-CAM pointing-game.
- Despliegue Streamlit con inferencia en CPU.

Metodología bajo **CRISP-ML(Q)** (Studer et al., 2021).

**Resultado headline (5-fold CV out-of-fold por paciente):** DL alcanza AUC 0.986 (IC 95 % 0.975–0.994) vs XGB+radiomics 0.809 (0.751–0.864); ΔAUC = +0.176, DeLong p = 7.87 × 10⁻¹⁰. La auditoría muestra que el modelo concentra atención sobre el hígado ~1.8× sobre el azar para casos cirróticos (CAM mass 0.49 vs baseline 0.27), pero solo parcialmente (pointing accuracy 0.50). Detalles en `reports/`.

---

## Estructura del repositorio

```
cirrosis-detection/
├── src/                       # Código fuente
│   ├── data/                  # inventory, splits, transforms, dataset, cache
│   ├── models/                # cnn_2_5d (ResNet 2.5D), radiomics
│   ├── training/              # train_dl, train_dl_cv, train_radiomics
│   ├── evaluation/            # metrics, bootstrap, delong, gradcam, evaluate_all
│   └── inference/             # pipeline end-to-end para la app
├── app/                       # Streamlit app
├── notebooks/                 # 6 notebooks Jupyter (EDA → evaluación)
├── configs/                   # YAML configs (data, radiomics, dl_2_5d, eval)
├── scripts/                   # download_cirrmri, validate_dataset, run_eda,
│                              # pointing_game_full, make_gradcam_panels, ...
├── tests/                     # pytest (splits sin leakage, métricas, DeLong)
├── reports/
│   ├── 00_context_summary.md       # resumen ejecutivo y alcance
│   ├── 01_objectives_revised.md    # objetivos SMART
│   ├── 02_methodology_crispmlq.md  # CRISP-ML(Q) las 6 fases
│   ├── RUNBOOK.md                  # cómo correr el pipeline
│   ├── figures/               # 9 figuras (EDA, ROC/PR, Grad-CAM)
│   ├── tables/                # 15 CSVs con resultados
│   ├── checkpoints/.gitkeep   # se llena al entrenar (ver más abajo)
│   └── timing.csv             # tiempos de entrenamiento por fold
├── data/.gitkeep              # se llena al descargar (ver más abajo)
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── run_app.bat                # lanzador de la app en Windows
└── .gitignore
```

## Qué NO está en el repo (por tamaño) y cómo obtenerlo

| Artefacto | Tamaño | Cómo obtener |
|---|---|---|
| Dataset CirrMRI600+ (`data/CirrMRI600plus_raw/`) | ~5.5 GB | `python scripts/download_cirrmri.py --dest data/CirrMRI600plus_raw` (descarga desde [OSF](https://osf.io/cuk24/)) |
| Cache pre-procesado (`data/cache/`) | ~6.6 GB | Se genera al correr el pipeline (paso 2 abajo). No se sube al repo. |
| Checkpoints del DL (`reports/checkpoints/resnet25d_fold{0..4}.pt`, `resnet25d_binary_best.pt`) | ~550 MB | `python scripts/download_checkpoints.py` descarga los 6 desde el [release v1.0](https://github.com/sebastian-umanap/cirrosis-detection/releases/tag/v1.0). Para correr solo la app basta con `--only resnet25d_binary_best.pt` (92 MB). También se regeneran entrenando (paso 4). |

---

## Setup

### Requisitos

- Python 3.10 o 3.11
- (Opcional pero recomendado) GPU NVIDIA con ≥ 8 GB VRAM. El entrenamiento se probó en **RTX 3060 12 GB**. La inferencia funciona en CPU.
- ~25 GB de disco libre (dataset + cache + checkpoints).

### Instalación

```powershell
git clone <repo-url> cirrosis-detection
cd cirrosis-detection

python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows. En Linux/Mac: source .venv/bin/activate

pip install --upgrade pip
pip install "numpy<2.0"
pip install -r requirements.txt           # runtime: app + inferencia (CPU)
pip install -r requirements-dev.txt       # opcional: stack completo de entrenamiento
```

PyTorch con CUDA (solo si vas a entrenar; ajusta a tu CUDA):

```powershell
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

Para CPU solamente:

```powershell
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
```

---

## Reproducir resultados

### 1. Descargar el dataset

```powershell
python scripts/download_cirrmri.py --dest data/CirrMRI600plus_raw
python scripts/validate_dataset.py --root data/CirrMRI600plus_raw
```

Detalles en `scripts/download_cirrmri.md`. El dataset son ~5 ZIPs (~10 GB) que se extraen automáticamente.

### 2. Construir cache de volúmenes pre-procesados (una sola vez, ~6 min)

```powershell
python -m src.data.cache build --root data/CirrMRI600plus_raw --cache data/cache --workers 4
```

### 3. Baseline radiómico (CPU, ~5 min)

```powershell
python -m src.training train-radiomics --config configs/radiomics.yaml
```

### 4. Entrenamiento DL 2.5D ResNet-50 (5-fold, ~2.7 h en RTX 3060)

```powershell
python -m src.training train-dl --config configs/dl_2_5d.yaml
```

### 5. Evaluación final (DeLong + ROC/PR + tablas)

```powershell
python -m src.training evaluate --config configs/eval.yaml
```

### 6. Auditoría de interpretabilidad a gran escala (n=738)

```powershell
python scripts/pointing_game_full.py --folds 0 1 2 3 4
python scripts/make_gradcam_panels.py --ckpt reports/checkpoints/resnet25d_fold0.pt --fold-id 0
```

Más detalles paso a paso en [reports/RUNBOOK.md](reports/RUNBOOK.md).

---

## App de demostración

**Camino rápido (sin entrenar, solo correr la demo):** instala dependencias (sección _Setup_), descarga el checkpoint y lanza la app.

```powershell
python scripts/download_checkpoints.py --only resnet25d_binary_best.pt
.\run_app.bat
```

Cualquier estudio NIfTI del dataset + su máscara sirven como ejemplo. Si aún no tienes el dataset, ver paso 1 de _Reproducir resultados_.

```powershell
# Windows: doble-click en run_app.bat (puerto 8400)
.\run_app.bat

# O manualmente:
python -m streamlit run app/streamlit_app.py --server.port 8400 --server.address localhost
```

Abre `http://localhost:8400`. La app acepta un NIfTI (`.nii.gz`) y, opcionalmente, su máscara hepática. Devuelve:

- Probabilidad de cirrosis.
- Máscara hepática superpuesta.
- Heatmap Grad-CAM.
- Volumen hepático estimado.
- Disclaimer médico.

> **Importante.** Si no subes la máscara hepática, la app usa un ROI abdominal central como fallback y las predicciones quedan fuera de distribución (poco confiables). Para resultados válidos, sube imagen + máscara provistas por el dataset.

**¿Por qué puerto 8400 y no 8501?** En Windows, los rangos 8501–8907 suelen estar reservados por Hyper-V/WSL/Docker, lo que causa `WinError 10013`. El puerto 8400 está fuera del rango reservado.

### Con Docker

```powershell
docker build -t cirrosis-app .
docker run -p 8400:8400 cirrosis-app
```

---

## Tests

```powershell
pytest tests/
```

Verifica los puntos críticos: cero leakage paciente-a-fold, métricas correctas (DeLong, bootstrap CI), splits estratificados.

---

## Documentación de soporte

| Documento | Contenido |
|---|---|
| [reports/00_context_summary.md](reports/00_context_summary.md) | Resumen ejecutivo, alcance, feedback abordado |
| [reports/01_objectives_revised.md](reports/01_objectives_revised.md) | Objetivos SMART con baselines cuantitativos |
| [reports/02_methodology_crispmlq.md](reports/02_methodology_crispmlq.md) | Metodología CRISP-ML(Q), las 6 fases con riesgos |
| [reports/RUNBOOK.md](reports/RUNBOOK.md) | Pasos de ejecución paso a paso |
| [scripts/download_cirrmri.md](scripts/download_cirrmri.md) | Cómo obtener CirrMRI600+ |

---

## Reproducibilidad

- Seed fijada en 42 en `torch`, `numpy`, `random`, `PYTHONHASHSEED`.
- Versiones pineadas en `requirements.txt`.
- Tiempos reales de entrenamiento por fold en `reports/timing.csv`.
- Las predicciones out-of-fold y métricas finales están en `reports/tables/`.

