# Descarga de CirrMRI600+

Dataset: **CirrMRI600+** (Choi/Jha et al., *Sci Data* 2025).
DOI: [10.17605/OSF.IO/CUK24](https://doi.org/10.17605/OSF.IO/CUK24)
Paper: [10.1038/s41597-025-05201-7](https://doi.org/10.1038/s41597-025-05201-7)
Hosting oficial: <https://osf.io/cuk24/>
Licencia: **CC-BY-SA 4.0**
Tamaño aproximado: **~15 GB** en NIfTI (`.nii.gz`).

> Si vas a publicar resultados, **cita el paper original** y respeta la licencia (atribución + share-alike).

---

## Opción 1 (recomendada) — Cliente `osfclient`

`osfclient` es la herramienta oficial para descargar nodos OSF en lote.

```bash
# Una sola vez por máquina
pip install osfclient

# Desde la raíz del proyecto cirrosis-detection
mkdir -p data
cd data
osf -p cuk24 clone CirrMRI600plus_raw
```

Esto crea `data/CirrMRI600plus_raw/cuk24/osfstorage/...` con la estructura original del repositorio.

Una vez termine la descarga, se renombra/aplana al layout esperado por el proyecto:

```bash
python ../scripts/normalize_layout.py \
  --src data/CirrMRI600plus_raw \
  --dst data/CirrMRI600plus
```

(El script `normalize_layout.py` se incluye en este proyecto.)

## Opción 2 — Descarga manual desde el navegador

1. Abrir <https://osf.io/cuk24/>.
2. En "Files" → seleccionar la carpeta raíz → botón "Download as zip".
3. Descomprimir en `cirrosis-detection/data/CirrMRI600plus/`.

Esta opción es la más simple si solo se va a descargar una vez, pero el zip puede ser grande (>15 GB) y el navegador a veces aborta.

## Opción 3 — `wget` con URL canónica de OSF (un archivo a la vez)

Para volúmenes individuales (ej. durante pruebas iniciales) se puede usar:

```bash
# Reemplazar {file_guid} con el GUID que aparece en la URL de un archivo en OSF
wget "https://osf.io/{file_guid}/download" -O archivo.nii.gz
```

No recomendado para descarga completa por la cantidad de archivos.

---

## Layout esperado tras la descarga

Después de descargar y normalizar, `data/CirrMRI600plus/` debe verse así (basado en la documentación del paper):

```
data/CirrMRI600plus/
├── T1w/
│   ├── images/
│   │   ├── patient_001.nii.gz
│   │   ├── patient_002.nii.gz
│   │   └── ...                            # 310 volúmenes
│   └── labels/
│       ├── patient_001.nii.gz             # máscara hepática
│       └── ...
├── T2w/
│   ├── images/
│   │   └── ...                            # 318 volúmenes
│   └── labels/
│       └── ...
├── healthy/                               # 55 controles sanos (revisar nombre real tras descarga)
│   ├── T1w/
│   └── T2w/
└── metadata/
    ├── CompleteData-age-gender-evaluation.csv     # 337 pacientes
    ├── T1-age-gender-evaluation.csv
    ├── T2-age-gender-evaluation.csv
    ├── Paired-age-gender-evaluation.csv           # 291 pacientes con ambas modalidades
    ├── Healthy-demographics.csv                   # 55 controles
    └── Labels.txt                                 # codificación de columnas
```

> **Nota:** El layout exacto puede variar ligeramente respecto a lo descrito en el paper; tras descargar, validar con `tree -L 3 data/CirrMRI600plus/` y, si la estructura difiere, ajustar `configs/data.yaml` (paths) — el código del proyecto está parametrizado.

---

## Validación post-descarga

Tras descargar, correr:

```bash
python scripts/validate_dataset.py --root data/CirrMRI600plus
```

Que verifica:

1. Conteos: 310 T1w + 318 T2w + 55 healthy (warning si difiere).
2. Cada imagen tiene su máscara correspondiente.
3. Los NIfTI se leen sin error con `nibabel`.
4. Los CSV existen y tienen columnas esperadas.
5. No hay PHI evidente en los headers DICOM-derivados.

Si todo pasa, queda listo para correr `notebooks/01_eda.ipynb`.

---

## .gitignore

`data/CirrMRI600plus/` **no se versiona** (está en `.gitignore` del proyecto). Solo se versiona la documentación de cómo obtenerlo.
