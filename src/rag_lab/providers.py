from __future__ import annotations

from typing import Protocol

from rag_lab.core.contracts import (
    EmbeddingProvider,
    RetrieverProvider,
    EvidenceEvaluatorProvider,
)

from rag_lab.core.models import GenerationResult
from rag_lab.inference import InferenceProfile


class ChatGenerator(Protocol):
    """Contrato para generar una respuesta."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        profile: InferenceProfile,
    ) -> GenerationResult:
        ...