# 02 — Metodología bajo CRISP-ML(Q)

> **Respuesta al feedback F2:** La primera entrega describía la metodología como "iterativa ágil", lo cual es vago. Se adopta **CRISP-ML(Q)** (Studer et al., *Mach. Learn. Knowl. Extr.*, 2021), un proceso de referencia diseñado específicamente para machine learning con etapas de aseguramiento de calidad embebidas. Cada fase documenta: **objetivos, entradas, salidas, métricas de calidad, riesgos y mitigaciones**.

Cita base: Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., & Müller, K. R. (2021). Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology. *Machine Learning and Knowledge Extraction*, 3(2), 392-413.

---

## Fase 1 — Business & Data Understanding

**Objetivos**
- Establecer el caso de uso clínico: **(T1)** detección binaria cirrótico vs sano en RM abdominal y **(T2)** graduación de severidad radiológica (Mild/Moderate/Severe) entre cirróticos.
- Definir métricas de éxito alineadas con uso clínico: AUC ≥ 0.90 (T1) con sensibilidad ≥ 0.85 al punto Youden (la sensibilidad pesa más que la especificidad porque un falso negativo retrasa diagnóstico); macro-F1 ≥ 0.60 (T2) con errores predominantemente de ±1 grado.
- Identificar restricciones legales/éticas: solo datos públicos con licencia CC-BY-SA 4.0; consentimientos cubiertos por el origen del dataset.
- Comprender el dataset CirrMRI600+ a nivel de inventario y distribución (339 cirróticos + 55 sanos; 310 T1w + 318 T2w).

**Entradas**
- Literatura clínica: ACR Appropriateness Criteria *Chronic Liver Disease* (2023); EASL Clinical Practice Guidelines *Decompensated Cirrhosis* (EASL 2018); Baveno VII (de Franchis et al., *J Hepatol* 2022).
- Literatura técnica: Choi et al. (CirrMRI600+, 2025), Kutaiba et al. (2023), Luetkens et al. (2022), Altınkaya et al. (2026), Liu et al. (arXiv 2502.18225, 2025).
- Acceso al dataset (descarga oficial).

**Salidas**
- Definición formal de las tareas: **T1** binaria, *cirrótico vs sano*, clase positiva = cirrótico; **T2** multiclase, severidad radiológica 1/2/3 (Mild/Moderate/Severe) entre cirróticos.
- Reporte de inventario del dataset (notebook 01 §1): conteos por paciente, estudio, modalidad, severidad (1/2/3), demografía (edad, sexo), complicaciones clínicas binarias (ascitis, esplenomegalia, varices, HCC).
- Documento de objetivos SMART → [01_objectives_revised.md](01_objectives_revised.md).

**Métricas de calidad de la fase**
- Cobertura de literatura: ≥ 20 referencias revisadas, registradas en `paper/refs.bib`.
- Trazabilidad: cada decisión de scope (binaria/multiclase, modalidad, splits) tiene línea explícita en el paper §Materiales o §Métodos.

**Riesgos y mitigaciones**

| Riesgo | Mitigación |
|---|---|
| Cancelación de datos institucionales (Fundación Santa Fe) ya materializada | Pivote completo a datos públicos; documentado como limitación en §Discusión |
| Etiquetas de severidad del dataset pueden no ser equivalentes a Child-Pugh estándar | Validar definición exacta de la etiqueta al cargar metadatos; reportar la convención usada |
| Conflicto entre tarea propuesta (binaria sano/cirrótico) y diseño del dataset | Reformulado a estratificación de severidad; justificado en O1 y §Métodos |

---

## Fase 2 — Data Preparation

**Objetivos**
- Convertir CirrMRI600+ crudo a tensores normalizados, listos para `DataLoader`.
- Garantizar splits sin *leakage* (estratificación por paciente).
- Generar augmentations clínicamente plausibles.

**Entradas**
- Volúmenes NIfTI T1w y T2w + máscaras hepáticas + CSV de etiquetas (CirrMRI600+ tras descarga).

**Pipeline (justificado en orden):**
1. **Lectura** con `nibabel`/`SimpleITK`, manteniendo affine para conservar orientación clínica.
2. **Re-orientación canónica** a RAS+ (estándar NIfTI).
3. **Resampling a spacing isotrópico 1×1×3 mm** (3 mm en Z conserva resolución clínica típica de RM y reduce carga; isotropía in-plane facilita transferencia desde encoders pre-entrenados).
4. **Clipping de intensidades a percentiles [0.5, 99.5] por volumen** (RM no tiene unidades estandarizadas como HU; clipping local controla outliers y artefactos).
5. **Z-score normalization por volumen** (media 0, desviación 1).
6. **Crop alrededor de la máscara hepática** + padding a tamaño fijo (224×224 in-plane para el modelo 2.5D; mantener el contexto inmediato del hígado, no recortar exacto al borde, para preservar señales adyacentes como esplenomegalia/ascitis que correlacionan con descompensación).
7. **Estratificación 5-fold por paciente** con `StratifiedGroupKFold` de scikit-learn (`groups=patient_id`, `y=severity`).
8. **Augmentation (solo en train)** con MONAI o TorchIO:
   - Rotación ±10° (axial), flip horizontal (sagital, médicamente plausible).
   - Elastic deformation suave (probabilidad 0.3).
   - Gamma random [0.8, 1.2].
   - Ruido gaussiano σ ≤ 0.05.
   - Motion artifact (TorchIO) probabilidad 0.1 (refleja artefactos de respiración).
9. **Class weighting** inversamente proporcional a frecuencia si el desbalance excede 1.5:1.

**Salidas**
- `src/data/dataset.py` con `Dataset` y `DataLoader` por split.
- `configs/data.yaml` con todos los hiperparámetros del pipeline.
- Cache de tensores preprocesados en `data/cache/` (`.pt`) para acelerar reentrenamientos.

**Métricas de calidad**
- 0 fugas paciente-a-fold (verificado con assert en `tests/test_splits.py`).
- < 1 % de volúmenes con error de carga (volúmenes corruptos van a `data/quarantine.csv`).
- Histogramas pre- y post-normalización guardados en `reports/figures/`.

**Riesgos y mitigaciones**

| Riesgo | Mitigación |
|---|---|
| Volúmenes con orientación atípica (no LPS/RAS) | Re-orientación canónica obligatoria; assert en loader |
| Spacing extremo en alguna serie | Filtro: rechazar volúmenes con spacing > 5 mm in-plane o > 10 mm en Z |
| Desbalance severo Child-Pugh | Weighted loss + focal loss + estratificación |
| Leakage por slice (mismo paciente en train y val) | `StratifiedGroupKFold` con `groups=patient_id` (no slice_id) |

---

## Fase 3 — Model Engineering

**Objetivos**
- Entrenar y seleccionar el modelo que cumpla O3.

**Entradas**
- DataLoaders de Fase 2, configs por modelo.

**Arquitecturas (justificación de selección):**

| Candidato | Pros | Contras | Decisión |
|---|---|---|---|
| **2.5D ResNet-50 + attention pooling** | Encaja en 12 GB con batch grande; pesos RadImageNet/ImageNet maduros; rápido | Pierde contexto inter-slice fuera del stack | **ELEGIDO** como modelo principal |
| 3D ResNet-18 sobre patches 128×128×32 | Captura morfología 3D | batch=2-4 obliga gradient accumulation; muy lento; menos pesos pre-entrenados disponibles | Trabajo futuro |
| MIL con attention pooling | Maneja volúmenes de profundidad variable de forma elegante | Menos baseline maduro en literatura para esta tarea exacta | Trabajo futuro |

**Hiperparámetros base (a refinar con un fold de validación):**
- Optimizer: AdamW, weight decay 1e-4.
- LR: 1e-4 backbone, 1e-3 head; cosine schedule con warmup 1 epoch.
- Batch effective: 16 (físico 8, accumulation 2 si la VRAM aprieta).
- Epochs: 30 con early stopping por AUC val (paciencia 6).
- Loss: BCE con `pos_weight` derivado del split de entrenamiento (focal loss si BCE no converge).
- **Mixed precision (`torch.cuda.amp`) activo por defecto.**
- Seed 42 fijado en `torch`, `numpy`, `random`, `python hash seed`.

**Salidas**
- Checkpoints `reports/checkpoints/{model}_fold{k}.pt`.
- Logs de entrenamiento (loss/AUC por epoch) en `reports/logs/`.

**Métricas de calidad**
- Reproducibilidad: dos corridas con misma seed difieren < 1 punto de AUC.
- Convergencia: loss val no aumenta más allá de paciencia × 1.5.
- Tiempos reales documentados (s/epoch, wall-clock total) — exigido por la rúbrica para reproducibilidad.

**Riesgos y mitigaciones**

| Riesgo | Mitigación |
|---|---|
| Sobreajuste (dataset pequeño) | Augmentation agresivo, regularización, early stopping, dropout 0.3 en cabezal |
| OOM en RTX 3060 | Gradient accumulation, mixed precision, reducir batch, channels-last |
| Inestabilidad por LR alto en backbone | Discriminative LR (backbone < head) + warmup |
| RadImageNet weights no compatibles con torchvision API | Fallback documentado a ImageNet |

---

## Fase 4 — Model Evaluation

**Objetivos**
- Reportar performance con incertidumbre cuantificada y comparación estadísticamente significativa entre modelos.

**Entradas**
- Predicciones por paciente de O2 (radiomics) y O3 (DL).

**Procedimiento:**
1. **5-fold CV estratificada por paciente.** Cada fold reporta AUC-ROC, AUC-PR, F1, MCC, sensibilidad/especificidad al umbral Youden.
2. **Bootstrap 1000 resamples** sobre las predicciones out-of-fold concatenadas para IC 95 % de cada métrica.
3. **Curvas ROC y PR** superpuestas (radiomics vs DL) con sombra de IC.
4. **Matriz de confusión** al umbral Youden global.
5. **Test de DeLong** (DeLong et al. 1988) para comparar AUCs entre baseline y DL, con corrección si se hacen múltiples comparaciones.
6. **Análisis de errores:** visualización de los 5 FP y 5 FN con mayor probabilidad — buscar patrones (artefactos, artefactos de movimiento, severidad límite).
7. **Análisis de subgrupos (fairness)** si el dataset incluye demografía (edad, sexo): reportar AUC por subgrupo y discutir disparidad.

**Salidas**
- `reports/tables/main_results.csv`, `reports/figures/roc_pr_curves.png`, `reports/figures/confusion_matrix.png`, `reports/tables/delong.csv`, `reports/figures/error_analysis/*.png`.

**Métricas de calidad de la fase**
- IC 95 % reportados para TODA métrica puntual (regla anti-feedback F1).
- Ningún número en el paper sin su CI o desviación.

**Riesgos y mitigaciones**

| Riesgo | Mitigación |
|---|---|
| Métricas optimistas por test set pequeño | Bootstrap y reporte de CIs amplios honestamente |
| DeLong inapropiado si predicciones no son independientes (mismo paciente, múltiples cortes) | Agregamos por paciente antes de DeLong |
| Demografía no disponible | Reportar explícitamente y plantear validación externa como trabajo futuro |

---

## Fase 5 — Deployment

**Objetivos**
- Entregar app local funcional (Streamlit) que cumpla O6.

**Pipeline de inferencia:**
1. Upload NIfTI (`.nii.gz`) o ZIP de DICOM.
2. Conversión DICOM → NIfTI si aplica (`dicom2nifti`).
3. Re-orientación a RAS+, resampling a spacing del entrenamiento, clipping, z-score.
4. Segmentación hepática: **modelo provisto del dataset si está en train-time**; en inferencia sobre datos nuevos sin máscara, **U-Net 2D** entrenada como respaldo (o usar nnU-Net pretrained). Si no se logra entrenar a tiempo, fallback: ROI rectangular abdominal y reportarlo como limitación.
5. Crop hepático.
6. Forward pass 2.5D ResNet-50 → probabilidad de descompensación.
7. Grad-CAM sobre el corte central.
8. UI: render del corte central con máscara overlay (alpha 0.3), heatmap Grad-CAM (alpha 0.5), probabilidad, volumen hepático estimado, disclaimer.

**Salidas**
- `app/streamlit_app.py`, `src/inference/pipeline.py`, `Dockerfile`, `README.md` con instrucciones.

**Métricas de calidad**
- Tiempo total inferencia < 5 s en CPU para volumen 256×256×40.
- Pipeline reproducible: `docker build && docker run` levanta la app en < 60 s sin tocar configuración.

**Riesgos y mitigaciones**

| Riesgo | Mitigación |
|---|---|
| Volúmenes de usuarios fuera de distribución | Disclaimer + sanity checks (rango de intensidades, dimensiones) que devuelven mensaje claro si no aplica |
| Inferencia CPU lenta | Export a TorchScript / ONNX, half-precision dinámico, fusiones de op |
| Faltan máscaras en producción | U-Net 2D liviana como segmentador; o ROI abdominal con caveat |

---

## Fase 6 — Monitoring & Maintenance

**Objetivos** (qué se monitorearía en producción clínica — sección explícita exigida por CRISP-ML(Q))

**Métricas a monitorear:**
- **Distribución de probabilidades:** drift en la mediana o varianza → posible *covariate shift* del scanner/protocolo.
- **Distribución de intensidades de entrada:** percentiles 1/50/99 comparados con la distribución de entrenamiento.
- **Volumen hepático estimado:** sanity check anatómico (rangos plausibles 800–2500 mL).
- **Latencia de inferencia:** p50, p95.
- **Calidad de la máscara** (si U-Net es fallback): proporción de áreas conectadas, plausibilidad.

**Triggers de reentrenamiento:**
- AUC en revisión retrospectiva trimestral cae > 5 puntos vs baseline.
- > 10 % de inputs caen fuera de los rangos de intensidad de entrenamiento.
- Cambio mayor de protocolo en el centro clínico.

**Auditoría:**
- Logs anonimizados de inputs (sin PHI), predicciones, latencia.
- Capacidad de overriding por radiólogo, con feedback loop para active learning.

> Como este es un **prototipo académico**, esta fase no se implementa operacionalmente, pero se describe en el paper §Discusión como roadmap de producción para responder al criterio CRISP-ML(Q).

---

## Diagrama global del flujo CRISP-ML(Q)

```
   ┌──────────────────────────────┐
   │ 1. Business & Data           │ ◄─── Literatura, dataset, restricciones éticas
   │    Understanding             │
   └────────┬─────────────────────┘
            ▼
   ┌──────────────────────────────┐
   │ 2. Data Preparation          │ ◄─── NIfTI → tensores normalizados, splits por paciente
   └────────┬─────────────────────┘
            ▼
   ┌──────────────────────────────┐
   │ 3. Model Engineering         │ ◄─── Radiomics baseline + 2.5D ResNet-50
   └────────┬─────────────────────┘
            ▼
   ┌──────────────────────────────┐
   │ 4. Model Evaluation          │ ◄─── 5-fold CV, bootstrap CIs, DeLong, Grad-CAM
   └────────┬─────────────────────┘
            ▼
   ┌──────────────────────────────┐
   │ 5. Deployment                │ ◄─── Streamlit + Docker, < 5 s CPU
   └────────┬─────────────────────┘
            ▼
   ┌──────────────────────────────┐
   │ 6. Monitoring & Maintenance  │ ◄─── Roadmap (no operacional en prototipo)
   └──────────────────────────────┘
```

Cada flecha hacia abajo es un *quality gate* (entradas validadas antes de pasar a la siguiente fase). Cualquier hallazgo crítico en fases 4 o 5 puede gatillar regreso a fase 2 (re-preparación de datos) o 3 (re-engineering del modelo), siguiendo el principio iterativo de CRISP-ML(Q).
