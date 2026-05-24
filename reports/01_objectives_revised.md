# 01 — Objetivos revisados (SMART) y baselines cuantitativos

> **Respuesta al feedback F1:** "Documentar" y "recopilar" son requisitos, no objetivos. A continuación se redefinen los objetivos como SMART (Specific, Measurable, Achievable, Relevant, Time-bound) con un baseline cuantitativo asociado a cada uno.

---

## Objetivo general

Desarrollar y validar un sistema de aprendizaje profundo, sobre el dataset público CirrMRI600+ (Choi/Jha et al., *Sci Data* 2025), capaz de:

- **(T1, tarea primaria)** **clasificar de forma binaria** un estudio de RM abdominal (T1w o T2w) como proveniente de un paciente cirrótico vs un control sano, alcanzando **AUC-ROC ≥ 0.90** y **sensibilidad ≥ 0.85 al umbral de Youden** en validación cruzada 5-fold estratificada **por paciente**, manejando el desbalance natural del dataset (≈ 6.2:1) con class weighting / focal loss y reportando IC 95 % por bootstrap.
- **(T2, análisis secundario)** **graduar la severidad radiológica** (1=Mild, 2=Moderate, 3=Severe; etiqueta tal cual la define el dataset) entre los pacientes cirróticos, alcanzando **macro-F1 ≥ 0.60** y matriz de confusión sin saltos no-monotónicos (i.e., los errores típicos son de ±1 grado, no de 2).

Ambas tareas deben **superar a un baseline radiómico clásico** (PyRadiomics + clasificador shallow) por al menos **5 puntos de AUC (T1)** o **5 puntos de macro-F1 (T2)** con significancia estadística (test de DeLong sobre AUC; bootstrap pareado para F1, *p* < 0.05).

> **Nota sobre el dataset (verificado tras lectura del paper y la API de OSF):** Contrario a lo asumido en la propuesta inicial, CirrMRI600+ **sí incluye 55 controles sanos** (con demografía) junto a los 339 pacientes cirróticos. Por tanto la tarea binaria *cirrótico vs sano* es viable dentro del mismo dataset, sin necesidad de suplementar con CHAOS-MRI o Duke Liver y sin incurrir en *domain shift*. La etiqueta de severidad provista es una **graduación radiológica 1/2/3 curada por expertos del dataset** — **no** es Child-Pugh ni MELD; la utilizamos tal cual la define el paper original, sin reinterpretación clínica.

---

## Objetivos específicos

Cada objetivo declara: **(S)** lo que entrega, **(M)** la métrica, **(A)** por qué es alcanzable con nuestro compute (RTX 3060 12 GB), **(R)** por qué importa, **(T)** horizonte temporal del objetivo.

### O1 — Segmentación hepática

**(S)** Producir máscaras hepáticas reutilizables para todos los volúmenes de CirrMRI600+ T1w y T2w.
**(M)** Dice ≥ 0.90 vs máscaras de referencia del dataset.
**Baseline 1 (cuantitativo, clásico):** thresholding por percentil + region growing + morfología → Dice esperado **0.55–0.70** (basado en Ben-Cohen et al. 2016).
**Baseline 2 (sanity check):** identidad — usar las máscaras que provee el dataset directamente (Dice = 1.00 por definición; sirve para descargar el costo de entrenar U-Net si las máscaras del dataset son de alta calidad).
**Decisión:** dado que CirrMRI600+ ya provee máscaras curadas, **se usarán directamente** y el esfuerzo de segmentación se reserva para *trabajo futuro* o despliegue (donde no tendremos máscaras en imagen nueva — ver O6).
**(A)** Las máscaras existen ya; el clásico corre en CPU; nnU-Net 2D sólo si falla la calidad.
**(R)** Sin ROI hepática, el clasificador miraría órganos adyacentes y la interpretabilidad se contamina.
**(T)** Validación de calidad: día 1.

### O2 — Baseline radiómico clásico

**(S)** Pipeline reproducible PyRadiomics → selección de features → clasificador shallow.
**(M)** AUC-ROC por paciente, media ± IC 95 % bootstrap (1000 resamples), 5-fold CV estratificada por paciente.
**Baseline cuantitativo esperado:** AUC **0.70–0.78** (estimación basada en Yasaka et al. 2018 sobre TC y Park et al. 2019 sobre RM para fibrosis avanzada).
**(A)** PyRadiomics corre en CPU. RF / XGBoost / LogReg en sklearn — rápido.
**(R)** Es el baseline interpretable y barato; los radiomics features son trazables a propiedades físicas (textura GLCM, GLRLM, forma) y permiten justificar la utilidad del DL si supera al clásico.
**(T)** Día 3 (después de EDA y preprocessing).

### O3 — Modelo deep principal: 2.5D ResNet-50

**(S)** ResNet-50 2D inicializado con pesos de RadImageNet (fallback ImageNet) operando sobre stacks de 3 cortes axiales consecutivos centrados en hígado; agregación por paciente con *attention pooling* (Ilse et al. 2018).
**(M)** AUC-ROC por paciente con IC 95 % bootstrap. **Meta: AUC ≥ 0.85**, **al menos +5 puntos sobre O2**.
**(A)** Encaja en 12 GB VRAM con batch ≥ 16 a resolución 224×224 + mixed precision. Tiempo estimado: ≤ 4 h por fold (≈ 20 h totales 5-fold), dentro del presupuesto.
**(R)** Es la elección de mejor relación costo/beneficio para el hardware: el transfer learning desde RadImageNet bate consistentemente entrenamiento *from scratch* en tareas médicas pequeñas (Mei et al. 2022).
**(T)** Día 5–8.

### O4 — Evaluación calibrada

**(S)** Conjunto completo de métricas calibradas con incertidumbre cuantificada.
**(M)** AUC-ROC, AUC-PR, sensibilidad/especificidad al punto operativo de Youden, F1, MCC, **todas con IC 95 % bootstrap (1000 resamples)**. Comparación AUCs con **test de DeLong** (significancia *p* < 0.05). Matriz de confusión a umbral Youden. Curvas ROC y PR superpuestas (baseline vs DL).
**Baseline cuantitativo:** las propias métricas del O2 funcionan como benchmark contra el cual O3 debe ganar.
**(A)** Implementación en `src/evaluation/` con scikit-learn + scipy. Liviano.
**(R)** Sin IC y test estadístico, una "mejora" en AUC podría ser ruido — esto es lo que se ataca el feedback F1.
**(T)** Día 8–9, en paralelo a entrenamientos finales.

### O5 — Interpretabilidad

**(S)** Mapas de saliencia Grad-CAM (Selvaraju 2017) sobre la última capa convolucional del modelo final, superpuestos sobre el corte axial central, validados cualitativamente contra criterios morfológicos clínicos.
**(M)** **Métrica cualitativa principal:** ≥ 70 % de los casos verdaderos positivos muestran activación dentro del hígado segmentado (no en vísceras vecinas, ni en ascitis fuera del hígado, ni en artefactos de borde). **Métrica cuantitativa secundaria:** *Pointing Game accuracy* del mapa Grad-CAM contra la máscara hepática (porcentaje de casos cuyo máximo cae dentro de la máscara).
**Baseline:** mapa aleatorio — Pointing Game esperado ≈ proporción del corte ocupada por hígado (~15-25 %).
**(A)** Grad-CAM es 1-2 forward passes adicionales en inferencia; no requiere entrenamiento.
**(R)** Sin interpretabilidad el modelo no es defendible clínicamente y el comité del curso lo reprochará. Adicionalmente revela *shortcut learning* (e.g., el modelo aprendiendo a detectar ascitis en vez de morfología hepática).
**(T)** Día 9–10.

### O6 — Despliegue funcional

**(S)** App Streamlit que recibe un NIfTI (`.nii.gz`) o un ZIP de DICOM, ejecuta pipeline completo (load → preprocess → liver segmentation → classification → Grad-CAM), y devuelve: probabilidad de descompensación, máscara hepática overlay, mapa de atención, métricas auxiliares (volumen hepático estimado en mL). Disclaimer médico visible. Inferencia en **CPU**.
**(M)** Tiempo total inferencia (load + preprocess + seg + clf + cam) **< 5 s** por estudio en CPU para volumen típico (~256×256×40). Funcional end-to-end con NIfTI de prueba.
**Baseline:** un script CLI que hace lo mismo sin UI (sirve como sanity check y métrica de latencia).
**(A)** Streamlit corre todo en Python; modelos exportados con `torch.jit` o ONNX para velocidad CPU.
**(R)** Es 30 % de la nota.
**(T)** Día 10–12.

---

## Matriz objetivo → entregable → evidencia

| Objetivo | Entregable | Evidencia |
|---|---|---|
| O1 | Pipeline de máscaras | `src/data/masks.py`, validación en `notebooks/03_segmentation_baseline.ipynb` |
| O2 | Baseline radiómico | `src/models/radiomics.py`, resultados en `notebooks/04_classification_baselines.ipynb`, tabla en paper §Resultados |
| O3 | Modelo 2.5D ResNet-50 | `src/models/cnn_2_5d.py`, training en `notebooks/05_deep_learning_models.ipynb`, checkpoint `reports/checkpoints/` |
| O4 | Suite de evaluación | `src/evaluation/metrics.py`, `src/evaluation/bootstrap.py`, `src/evaluation/delong.py`, tablas+ROC+PR en paper §Resultados |
| O5 | Grad-CAM | `src/evaluation/gradcam.py`, figuras cualitativas en paper §Resultados |
| O6 | App Streamlit | `app/streamlit_app.py`, Dockerfile, README de uso |

## Criterios de éxito agregados

El proyecto se considerará exitoso si **al menos 4 de los 6 objetivos** alcanzan su métrica meta. O1 está garantizado por el uso de máscaras del dataset; O4 y O6 son implementaciones; el riesgo principal es que O3 no alcance AUC ≥ 0.85 — en ese caso reportamos lo obtenido con honestidad, discutimos causas (single-dataset, tamaño efectivo, dificultad intrínseca de Child-Pugh por imagen), y la entrega sigue siendo defendible por la calidad metodológica y la app funcional.
