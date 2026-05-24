"""Pipeline end-to-end de inferencia (CPU) usado por la app Streamlit."""

from src.inference.pipeline import InferencePipeline, InferenceResult, load_pipeline

__all__ = ["InferencePipeline", "InferenceResult", "load_pipeline"]
