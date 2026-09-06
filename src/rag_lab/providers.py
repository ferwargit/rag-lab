from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from rag_lab.core.contracts import EmbeddingProvider

from rag_lab.generation import GenerationResult
from rag_lab.inference import InferenceProfile
from rag_lab.retrieval import SearchResult
from rag_lab.metrics import ExecutionMetrics

if TYPE_CHECKING:
    from rag_lab.evidence_evaluator import EvidenceDecision


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

    @property
    def last_metrics(self) -> ExecutionMetrics | None:
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
