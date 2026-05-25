"""App Streamlit — Detección de cirrosis hepática en RM abdominal.

Prototipo académico. NO apto para uso clínico.

Para correr:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Garantiza que `src/` sea importable cuando la app corre desde la raíz (HF Spaces).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.pipeline import InferenceResult, load_pipeline  # noqa: E402

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------

CHECKPOINT_RELEASE_URL = (
    "https://github.com/sebastian-umanap/cirrosis-detection/releases/download/v1.0"
)


def _ensure_checkpoint(path: Path) -> Path:
    """Descarga el checkpoint desde el release de GitHub si no existe localmente.

    Pensado para despliegues (HF Spaces / Streamlit Cloud) donde el .pt no está en el repo.
    La descarga ocurre una vez por contenedor; queda cacheada en `reports/checkpoints/`.
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{CHECKPOINT_RELEASE_URL}/{path.name}"
    with st.spinner(f"Descargando checkpoint {path.name} (~92 MB, una sola vez por sesión)…"):
        urllib.request.urlretrieve(url, path)  # noqa: S310
    return path


def _default_binary_ckpt() -> Path:
    """Devuelve el primer checkpoint disponible: 'best' si existe, si no fold0..4."""
    candidates = [
        Path("reports/checkpoints/resnet25d_binary_best.pt"),
        *[Path(f"reports/checkpoints/resnet25d_fold{k}.pt") for k in range(5)],
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


DEFAULT_BINARY_CKPT = _default_binary_ckpt()
DEFAULT_SEVERITY_CKPT = Path("reports/checkpoints/resnet25d_severity_best.pt")

st.set_page_config(
    page_title="Detección de cirrosis · RM abdominal",
    page_icon="🧪",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Cache del modelo
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo…")
def _load_model(binary_ckpt: str, severity_ckpt: str | None):
    bp = Path(binary_ckpt)
    sp = Path(severity_ckpt) if severity_ckpt else None
    return load_pipeline(bp, sp)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("Detección de cirrosis hepática en RM abdominal")
st.caption("Proyecto final · ISIS-4825 Imágenes y Visión · Universidad de los Andes")

st.error(
    "⚠️ **Prototipo académico**. Este sistema **no** está validado para uso clínico, "
    "**no** reemplaza la lectura radiológica y **no** debe usarse para tomar decisiones diagnósticas.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Configuración")
    binary_ckpt = st.text_input("Checkpoint clasificación binaria", value=str(DEFAULT_BINARY_CKPT))
    use_severity = st.checkbox("Incluir clasificación de severidad (3 clases)", value=False)
    severity_ckpt = st.text_input("Checkpoint severidad", value=str(DEFAULT_SEVERITY_CKPT)) if use_severity else None
    st.markdown("---")
    st.caption("**Dataset de entrenamiento:** CirrMRI600+ (Choi/Jha et al., *Sci Data* 2025).")
    st.caption("**Hardware entrenamiento:** RTX 3060 12GB. **Inferencia:** CPU.")

col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded = st.file_uploader(
        "1. Estudio de RM (NIfTI `.nii.gz` o ZIP de DICOM)",
        type=["nii", "gz", "zip"],
        accept_multiple_files=False,
    )
with col_up2:
    uploaded_mask = st.file_uploader(
        "2. Máscara hepática (NIfTI, opcional pero recomendado)",
        type=["nii", "gz"],
        accept_multiple_files=False,
        help="Sin máscara, el sistema usa un ROI abdominal central (fallback) y las predicciones "
        "son poco fiables porque el modelo se entrenó con recortes centrados en el hígado real. "
        "Para volúmenes del dataset, la máscara está en la carpeta labels/masks correspondiente.",
    )

if uploaded is None:
    st.info("Sube un estudio para empezar. Para resultados válidos, sube también la máscara hepática del hígado.")
    st.stop()

# Garantiza el checkpoint: si no está local, lo descarga del release v1.0.
try:
    _ensure_checkpoint(Path(binary_ckpt))
except Exception as exc:  # noqa: BLE001
    st.error(
        f"No se pudo asegurar el checkpoint `{binary_ckpt}`: {exc}. "
        "Verifica conectividad o ajusta el path en la barra lateral."
    )
    st.stop()

# Guardar a temp y correr.
def _suffix_for(name: str) -> str:
    """Preserva la extensión COMPLETA (.nii.gz, no solo .gz) para que nibabel reconozca el formato."""
    n = name.lower()
    if n.endswith(".nii.gz"):
        return ".nii.gz"
    if n.endswith(".nii"):
        return ".nii"
    if n.endswith(".zip"):
        return ".zip"
    return "".join(Path(name).suffixes) or Path(name).suffix


def _save_temp(uploaded_file) -> Path:
    with tempfile.NamedTemporaryFile(suffix=_suffix_for(uploaded_file.name), delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


tmp_path = _save_temp(uploaded)
mask_tmp_path = _save_temp(uploaded_mask) if uploaded_mask is not None else None

try:
    pipeline = _load_model(binary_ckpt, severity_ckpt)
    with st.spinner("Procesando estudio…"):
        result: InferenceResult = pipeline(tmp_path, mask_path=mask_tmp_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Falló la inferencia: {exc}")
    st.stop()

# ----------------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------------

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("Probabilidad de cirrosis", f"{result.proba_cirrhosis * 100:.1f} %")
with col2:
    st.metric("Volumen hepático estimado", f"{result.liver_volume_ml:.0f} mL")
with col3:
    st.metric("Tiempo de inferencia", f"{result.inference_seconds:.2f} s")

if result.severity_probs is not None:
    st.subheader("Severidad radiológica (Mild / Moderate / Severe)")
    sev_labels = ["Mild", "Moderate", "Severe"]
    cols = st.columns(3)
    for c, lbl, p in zip(cols, sev_labels, result.severity_probs, strict=True):
        c.metric(lbl, f"{float(p) * 100:.1f} %")

st.subheader("Corte axial central + Grad-CAM")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(result.central_slice, cmap="gray")
    ax.set_title("Corte axial central")
    ax.axis("off")
    st.pyplot(fig)
with right:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(result.central_slice, cmap="gray")
    ax.imshow(result.gradcam, cmap="jet", alpha=0.5)
    ax.set_title("Grad-CAM (rojo = activación)")
    ax.axis("off")
    st.pyplot(fig)

st.subheader("Validación de interpretabilidad")
c1, c2 = st.columns(2)
c1.metric("Máximo del CAM dentro del hígado", "Sí" if result.pointing_inside_mask else "No")
c2.metric(
    "Masa del CAM dentro del hígado",
    f"{(result.cam_mass_inside_mask * 100):.0f} %" if np.isfinite(result.cam_mass_inside_mask) else "n/d",
)

if result.warnings:
    with st.expander("Advertencias del pipeline", expanded=True):
        for w in result.warnings:
            st.warning(w)

with st.expander("Detalles técnicos"):
    st.markdown(
        f"""
**Modelo binario:** ResNet-50 2.5D + attention pooling. Entrenado sobre **CirrMRI600+**
(339 pacientes cirróticos vs 55 controles sanos) con 5-fold CV estratificada **por paciente**,
mixed precision, transfer learning desde ImageNet.

**Checkpoint cargado:** `{binary_ckpt}`
**Severidad:** {"activa con " + str(severity_ckpt) if use_severity else "desactivada"}

**Citación dataset:** Choi/Jha et al., *Sci Data* 2025 — DOI 10.1038/s41597-025-05201-7.
""".strip()
    )

# Permite descargar reporte simple en PNG.
buf = io.BytesIO()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(result.central_slice, cmap="gray"); axes[0].axis("off"); axes[0].set_title("Corte central")
axes[1].imshow(result.central_slice, cmap="gray")
axes[1].imshow(result.gradcam, cmap="jet", alpha=0.5)
axes[1].set_title(f"Grad-CAM | p(cirrosis)={result.proba_cirrhosis:.2f}")
axes[1].axis("off")
fig.tight_layout()
fig.savefig(buf, format="png", dpi=150)
st.download_button("Descargar reporte PNG", data=buf.getvalue(), file_name="reporte_cirrosis.png", mime="image/png")
