"""Compatibilidad temporal: la configuración vive en rag_lab.core.inference."""

from rag_lab.core.inference import (
    CLASSIFIER_PROFILE,
    DEEP_ANSWER_PROFILE,
    EVIDENCE_PROFILE,
    INFERENCE_LAYER_VERSION,
    RAG_ANSWER_PROFILE,
    InferenceProfile,
    ModelCapabilities,
    ReasoningMode,
    validate_profile,
)

__all__ = [
    "CLASSIFIER_PROFILE",
    "DEEP_ANSWER_PROFILE",
    "EVIDENCE_PROFILE",
    "INFERENCE_LAYER_VERSION",
    "RAG_ANSWER_PROFILE",
    "InferenceProfile",
    "ModelCapabilities",
    "ReasoningMode",
    "validate_profile",
]
