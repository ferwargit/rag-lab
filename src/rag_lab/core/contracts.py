from typing import Protocol

from collections.abc import Sequence

from rag_lab.core.inference import InferenceProfile
from rag_lab.core.models import (
    EmbeddedChunk,
    SearchResult,
    EvidenceDecision,
    GenerationResult,
)


class EmbeddingProvider(Protocol):
    """Contrato para generar embeddings."""

    def embed(
        self,
        text: str,
    ) -> list[float]:
        ...


class VectorStore(Protocol):
    """Contrato para un almacenamiento de chunks vectorizados."""

    @property
    def items(self) -> tuple[EmbeddedChunk, ...]:
        ...


class RetrieverProvider(Protocol):
    """Contrato para recuperar documentos."""

    def search(
        self,
        query_embedding: tuple[float, ...],
        *,
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        ...


class EvidenceEvaluatorProvider(Protocol):
    """Contrato para evaluar la evidencia recuperada."""

    def evaluate(
        self,
        query: str,
        results: Sequence[SearchResult],
    ) -> EvidenceDecision:
        ...


class ChatGenerator(Protocol):
    """Contrato para generar una respuesta."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        profile: InferenceProfile,
    ) -> GenerationResult:
        ...