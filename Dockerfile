# Imagen para Hugging Face Spaces (CPU) — sirve la app Streamlit.
# Para entrenamiento usar entorno con GPU local.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias de sistema mínimas para nibabel/dicom2nifti.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Usuario no-root: requisito de Hugging Face Spaces (UID 1000).
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Instala dependencias como `user`. requirements.txt ya pinea torch CPU vía extra-index.
COPY --chown=user requirements.txt ./
RUN pip install --user -r requirements.txt

# Copia el código. Los checkpoints se descargan en runtime desde el release v1.0
# (ver streamlit_app.py::_ensure_checkpoint), así que no se incluyen en la imagen.
COPY --chown=user src/ ./src/
COPY --chown=user app/ ./app/
COPY --chown=user configs/ ./configs/
COPY --chown=user reports/ ./reports/
COPY --chown=user scripts/ ./scripts/

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false"]
