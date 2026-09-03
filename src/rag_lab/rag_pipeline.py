from rag_lab.generation import GenerationResult
from rag_lab.inference import RAG_ANSWER_PROFILE
from rag_lab.metrics import (
    ExecutionMetrics,
    metrics_from_generation,
)
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
        self.last_generation: GenerationResult | None = None

    @property
    def last_metrics(self) -> ExecutionMetrics | None:
        """Devuelve las métricas de la última generación de respuesta."""

        if self.last_generation is None:
            return None

        return metrics_from_generation(
            self.last_generation,
        )

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

        if not results:
            return RAGResult(
                answer=ABSTENTION_MESSAGE,
                sufficient=False,
                retrieved_chunk_ids=(),
                selected_chunk_ids=(),
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

        if not selected_results:
            return RAGResult(
                answer=ABSTENTION_MESSAGE,
                sufficient=False,
                retrieved_chunk_ids=tuple(
                    result.chunk.id
                    for result in results
                ),
                selected_chunk_ids=(),
            )

        messages = build_rag_messages(
            query,
            selected_results,
        )

        generation = self.chat_client.generate(
            messages,
            profile=RAG_ANSWER_PROFILE,
        )

        self.last_generation = generation

        if not generation.content.strip():
            return RAGResult(
                answer=ABSTENTION_MESSAGE,
                sufficient=False,
                retrieved_chunk_ids=tuple(
                    result.chunk.id
                    for result in results
                ),
                selected_chunk_ids=(),
            )

        return RAGResult(
            answer=generation.content,
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