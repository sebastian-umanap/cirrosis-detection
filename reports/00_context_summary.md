> **Nota de framing final (post-experimentos):** el proyecto se posicionó originalmente como "construir un detector de cirrosis en RM y comparar con baseline". Tras correr los experimentos descubrimos que la tarea binaria cirrótico-vs-sano en CirrMRI600+ es separable casi perfectamente por el DL (AUC=0.988, spec=1.000), pero **nuestro audit con pointing-game revela que el modelo aprende un atajo extra-hepático**. Por tanto reposicionamos el paper, slides y narrativa: el aporte científico **no es** el AUC alto, sino la auditoría metodológica que lo desmonta. Ver título nuevo del paper: *"Beyond AUC: A Pointing-Game Audit Reveals Shortcut Learning..."*. Este documento conserva el contexto histórico de cómo llegamos a esta tesis.

# 00 — Resumen de contexto del proyecto

**Curso:** ISIS-4825 — Imágenes y Visión — Universidad de los Andes (202610)
**Autores:** Sebastian Umaña Peinado, Jerónimo Franco Castañeda
**Fecha de este documento:** 2026-05-12
**Entrega:** artículo IEEE + video (máx. 15 min) el domingo 24 de mayo de 2026

---

## 1. Tema y problema

**Detección automática de cirrosis hepática avanzada a partir de imágenes de RM abdominal mediante deep learning**, usando como fuente primaria el dataset público **CirrMRI600+** (Choi et al., *Sci Data* 2025), que contiene 628 estudios de RM T1w/T2w de 339 pacientes con hígado cirrótico, con máscaras de segmentación hepática y etiquetas de severidad.

Originalmente la propuesta planteaba clasificación binaria *cirrótico vs no-cirrótico* sobre TC y RM, complementado con datos institucionales de la Fundación Santa Fe de Bogotá. Por restricciones del comité de ética de la institución **no se recibirán datos clínicos**, y el alcance se restringe a **datasets públicos**.

## 2. Cambios críticos de alcance respecto a la primera entrega

| Aspecto | Primera entrega | Versión revisada |
|---|---|---|
| Modalidad | TC + RM | **Solo RM** (CirrMRI600+ es RM; alineado con propuesta original del profesor) |
| Datos institucionales | Fundación Santa Fe (previsto) | **Cancelado por comité de ética** — solo datos públicos |
| EDA | CHAOS-CT paciente 11 (sin etiquetas) | **CirrMRI600+** (con máscaras y etiquetas reales) |
| Tarea | "cirrótico vs no-cirrótico" (suponiendo controles externos) | **Doble tarea sobre el mismo dataset CirrMRI600+ (Choi et al. 2025)**: **(T1) binaria cirrótico vs sano** (339 pacientes cirróticos vs 55 controles sanos provistos por el propio dataset — desbalance 6.2:1, manejado con class weights / focal loss) y **(T2) graduación radiológica de severidad** (mild / moderate / severe, dentro de los cirróticos) como análisis secundario. Una sola fuente de datos elimina *domain shift* entre cohortes. |
| Bibliografía | Duplicada (numerada + lista de URLs al final) | **Una sola bibliografía** en BibTeX con DOIs verificados |
| Metodología | "iterativa ágil" (vago) | **CRISP-ML(Q)** (Studer et al., 2021) con riesgos y mitigaciones documentados |

## 3. Feedback del profesor (primera entrega) y cómo lo abordamos

> **F1 — Objetivos:** "Documentar" y "recopilar" son requisitos, no objetivos. Falta baseline cuantitativo.
**→** Reescritos a SMART en [01_objectives_revised.md](01_objectives_revised.md), cada objetivo específico tiene baseline cuantitativo (Dice, AUC, IC 95%).

> **F2 — Metodología:** Falta detalle, usar framework formal.
**→** Adoptamos **CRISP-ML(Q)** completo: las 6 fases con entradas, salidas, métricas de calidad, riesgos y mitigaciones documentados en [02_methodology_crispmlq.md](02_methodology_crispmlq.md).

> **F3 — Bibliografía:** Eliminar duplicación.
**→** Una sola bibliografía en `paper/refs.bib` con DOIs verificados. Sin URLs sueltas duplicando entradas formales.

> **F4 — EDA pobre:** Sin comentarios, sin distribución de intensidades, dataset sin etiquetas (CHAOS-CT) no justificaba el problema.
**→** EDA rehecho sobre CirrMRI600+ (RM con máscaras y etiquetas de severidad reales). Incluye histogramas de intensidad por modalidad (T1w/T2w) y por clase, calidad de máscaras, conteo de pacientes/estudios, recomendación de split por paciente. Cada celda con interpretación en markdown.

## 4. Entregables y peso (rúbrica oficial)

1. **Despliegue (30%)** — Streamlit + FastAPI servido localmente. Sube NIfTI → segmenta hígado → clasifica severidad → Grad-CAM. Inferencia objetivo < 5 s en CPU.
2. **Sustentación (30%)** — 10 min presentación + 5 min Q&A, desde el 25-may-2026.
3. **Artículo IEEE (40%)** — ≤ 10 páginas, formato IEEE Conference, secciones obligatorias: Título, Autores, Abstract, Introducción, Estado del Arte, Materiales, Métodos, Resultados, Discusión y Conclusión, Bibliografía.

## 4.bis Composición confirmada de CirrMRI600+

Tras verificación del paper (Choi/Jha et al., Sci Data 2025, DOI 10.1038/s41597-025-05201-7) y el repo OSF `cuk24`:

- **339 pacientes cirróticos** (con 628 estudios: 310 T1w + 318 T2w; ~40 000 cortes anotados — 28 263 T1w, 11 691 T2w).
- **55 controles sanos** con demografía.
- Metadatos en CSV: `CompleteData-age-gender-evaluation.csv` (337 pacientes), `T1-age-gender-evaluation.csv`, `T2-age-gender-evaluation.csv`, `Paired-age-gender-evaluation.csv` (291 pacientes con ambas modalidades), `Healthy-demographics.csv` (55 controles), `Labels.txt` (codificación).
- **Etiquetas de severidad provistas:** *radiological severity* en escala 1=Mild, 2=Moderate, 3=Severe — **NO es Child-Pugh ni MELD**; es una graduación radiológica curada por expertos del dataset. Lo reportamos así, sin reinterpretar.
- **Columnas reales del CSV verificadas tras descarga:** `Patient ID, Age, Gender, Radiological Evaluation` solamente. Las complicaciones clínicas binarias (ascitis, esplenomegalia, varices, HCC) que sugerían las descripciones generales del paper **NO están** como columnas en los CSV publicados — quedan como `None` en `PatientRecord`. Se discute como limitación en el paper.
- Split sugerido por autores: 80:10:10 (T1: 248/31/31, T2: 256/31/31). Nosotros usamos **5-fold CV estratificada por paciente** para reportar con CIs.
- Licencia: CC-BY-SA 4.0; hosting OSF; ~15 GB en NIfTI.
- Los autores reportan solo benchmarks de **segmentación** (mejor nnSynergyNet3D, DSC 87.89 % T1w). Nuestro trabajo de **clasificación** sobre este dataset es novel respecto al benchmark del paper.

## 5. Restricciones de cómputo

Hardware del equipo: **RTX 3060 12 GB VRAM**, entrenamiento local.
Implicaciones técnicas (justificadas en el paper):
- 3D ResNet full-res sobre volúmenes completos no cabe.
- Estrategia adoptada: **2.5D ResNet-50** con transfer learning desde RadImageNet/ImageNet + agregación por paciente con attention pooling. Justificado en §Métodos. (Alternativas 3D-ligero y MIL consideradas y descartadas por tiempo y compute.)
- nnU-Net 2D para segmentación de respaldo (si las máscaras del dataset no son suficientes).
- Mixed precision (`torch.cuda.amp`) obligatorio.
- Gradient accumulation cuando el batch deseado no quepa.
- PyRadiomics y baselines clásicos en CPU.

## 6. Priorización de alcance

Priorizamos los entregables por impacto en la calificación:

**MUST (núcleo del proyecto):**
- EDA profesional sobre CirrMRI600+ con interpretaciones.
- Pipeline preprocessing reproducible.
- Baseline clásico: radiomics (PyRadiomics) + LogReg / RF / XGBoost.
- Un modelo principal DL: 2.5D ResNet-50 con transfer learning.
- Evaluación con AUC ± IC 95 % por bootstrap (1000 resamples), 5-fold CV por paciente.
- Streamlit app funcional con NIfTI upload.
- Paper IEEE con todas las secciones, ≥ 20 referencias reales con DOI.
- Slides 10 min.

**SHOULD (valor agregado, ejecutado):**
- Grad-CAM con análisis cualitativo y cuantitativo (pointing-game).
- DeLong test entre baseline y modelo DL.
- Análisis de errores con visualizaciones.

**OUT OF SCOPE (trabajo futuro):**
- Segunda arquitectura DL (3D ResNet-18, MIL) — solo se cita como trabajo futuro.
- Validación cruzada externa con CHAOS/Duke.
- Tarea de graduación de severidad (Mild/Moderate/Severe) como análisis primario.

## 7. Estado del dataset

**CirrMRI600+** no está descargado todavía. El equipo está descargándolo en paralelo. El proyecto se construye con paths parametrizados de modo que apenas exista `data/CirrMRI600plus/` los notebooks y pipelines corren sin cambios. Ver [scripts/download_cirrmri.md](../scripts/download_cirrmri.md).

## 8. Conexión con el contenido del curso

Las técnicas usadas se justifican parcialmente sobre laboratorios cursados:

- **Lab 07 (MLP/clasificación):** baseline de clasificación.
- **Lab 08 (CNN desde cero):** arquitecturas convolucionales 2D.
- **Lab 09 (transfer learning con VGG16):** transferencia desde pesos pre-entrenados — base conceptual del transfer learning 2.5D que usamos.
- **Lab 10 (detección de objetos):** menos relevante (no es nuestra tarea), pero el manejo de bounding boxes informa la inspección de regiones hepáticas.
- **Lab 11 (U-Net para segmentación):** **base de la red de segmentación hepática** si tuviéramos que entrenar la propia. En la práctica usaremos las máscaras provistas por CirrMRI600+ y solo entrenaremos U-Net si la calidad lo exige.

## 9. Riesgos identificados (resumen ejecutivo)

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Dataset tarda en descargar | Alta | Alto | Andamiaje del proyecto sin datos; código parametrizado; descarga en paralelo |
| Desbalance Child-Pugh A vs B/C | Media | Medio | Class weights + focal loss; estratificación por paciente |
| Sobreajuste por dataset único | Alta | Alto | 5-fold CV por paciente, augmentation agresivo, regularización, reporte honesto de la limitación |
| Tiempo de entrenamiento excede ventana | Media | Alto | Mixed precision, subsampling estratificado documentado, early stopping agresivo |
| Etiquetas de severidad ruidosas | Media | Medio | Validación contra hallazgos morfológicos (Grad-CAM en hígado/bazo) y discusión de límites del label |

Detalle ampliado en [02_methodology_crispmlq.md](02_methodology_crispmlq.md) §1.

## 10. Próximos pasos inmediatos

1. Confirmar descarga de CirrMRI600+ y montarlo en `data/` (ver [scripts/download_cirrmri.md](../scripts/download_cirrmri.md)).
2. Correr [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) sobre datos reales.
3. Entrenar baseline radiómico (CPU, rápido).
4. Entrenar 2.5D ResNet-50 (RTX 3060 12 GB, mixed precision).
5. Iterar paper y app sobre resultados reales.
