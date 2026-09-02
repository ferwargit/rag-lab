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

        raise NotImplementedError