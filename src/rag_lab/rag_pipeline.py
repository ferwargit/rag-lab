from collections.abc import Sequence

from rag_lab.models import RAGResult
from rag_lab.pipeline_utils import select_results
from rag_lab.prompting import build_rag_messages
from rag_lab.providers import (
    ChatGenerator,
    EmbeddingProvider,
    EvidenceEvaluatorProvider,
    RetrieverProvider,
)
from rag_lab.evidence_evaluator import EvidenceDecision


ABSTENTION_MESSAGE = (
    "No tengo información suficiente en el contexto "
    "disponible para responder esta pregunta."
)


class RAGPipeline:
    """Orquestador principal del sistema RAG."""

    def __init__(
        self,
        embedding_client: EmbeddingProvider,
        retriever: RetrieverProvider,
        evidence_evaluator: EvidenceEvaluatorProvider,
        chat_client: ChatGenerator,
        *,
        top_k: int = 3,
) -> None:
        if top_k <= 0:
            raise ValueError(
                "top_k debe ser mayor que cero."
            )

        self.embedding_client = embedding_client
        self.retriever = retriever
        self.evidence_evaluator = evidence_evaluator
        self.chat_client = chat_client
        self.top_k = top_k

    def ask(self, query: str) -> RAGResult:
        """Ejecuta una consulta RAG completa."""

        query_embedding = tuple(
            self.embedding_client.embed(query)
        )

        results = self.retriever.search(
            query_embedding,
            top_k=self.top_k,
            score_threshold=None,
        )

        decision = self.evidence_evaluator.evaluate(
            query,
            results,
        )

        if not decision.sufficient:
            return RAGResult(
                answer=ABSTENTION_MESSAGE,
                sufficient=False,
                retrieved_chunk_ids=tuple(
                    result.chunk.id
                    for result in results
                ),
                selected_chunk_ids=(),
            )

        selected_results = select_results(
            results,
            decision.selected_chunk_ids,
        )

        return RAGResult(
            answer="",
            sufficient=True,
            retrieved_chunk_ids=tuple(
                result.chunk.id
                for result in results
            ),
            selected_chunk_ids=tuple(
                result.chunk.id
                for result in selected_results
            ),
        )