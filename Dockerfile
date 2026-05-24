# Imagen mínima para servir la app Streamlit (inferencia CPU).
# Para entrenamiento usar entorno con GPU local (RTX 3060 12 GB).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias de sistema mínimas para nibabel/SimpleITK/dicom2nifti
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar torch CPU primero (más liviano que la versión CUDA)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 torchvision==0.18.1

COPY requirements.txt ./
# Re-instala todo respetando requirements.txt; torch ya está en cache
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY reports/checkpoints/ ./reports/checkpoints/

EXPOSE 8501
HEALTHCHECK CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
