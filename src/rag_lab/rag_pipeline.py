import time

from rag_lab.core.models import GenerationResult
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
from typing import Literal
from rag_lab.single_call import SingleCallRAG


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
        chat_client: ChatGenerator,
        evidence_evaluator: EvidenceEvaluatorProvider | None = None,
        *,
        top_k: int = 3,
        mode: Literal["two_call", "single_call"] = "single_call",
        single_call_rag: SingleCallRAG | None = None,
) -> None:
        if top_k <= 0:
            raise ValueError(
                "top_k debe ser mayor que cero."
            )

        if mode not in ("two_call", "single_call"):
            raise ValueError(
                "mode debe ser 'two_call' o 'single_call'."
            )

        if mode == "two_call" and evidence_evaluator is None:
            raise ValueError(
                "evidence_evaluator es obligatorio cuando mode='two_call'."
            )

        if mode == "single_call" and single_call_rag is None:
            single_call_rag = SingleCallRAG(chat_client)

        self.embedding_client = embedding_client
        self.retriever = retriever
        self.evidence_evaluator = evidence_evaluator
        self.chat_client = chat_client
        self.top_k = top_k
        self.mode = mode
        self.single_call_rag = single_call_rag
        self.last_generation: GenerationResult | None = None
        self.last_execution_time_seconds: float | None = None
        self.last_stage_execution_times_seconds: dict[str, float] = {}

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

        start_time = time.perf_counter()

        self.last_generation = None
        self.last_execution_time_seconds = None
        self.last_stage_execution_times_seconds = {}

        try:
            stage_start = time.perf_counter()

            query_embedding = tuple(
                self.embedding_client.embed(query)
            )

            self.last_stage_execution_times_seconds["embedding"] = (
                time.perf_counter() - stage_start
            )

            stage_start = time.perf_counter()
            
            results = self.retriever.search(
                query_embedding,
                top_k=self.top_k,
                score_threshold=None,
            )

            self.last_stage_execution_times_seconds["retrieval"] = (
                time.perf_counter() - stage_start
            )
            
            if not results:
                return RAGResult(
                    answer=ABSTENTION_MESSAGE,
                    sufficient=False,
                    retrieved_chunk_ids=(),
                    selected_chunk_ids=(),
                )

            if self.mode == "single_call":
                assert self.single_call_rag is not None

                stage_start = time.perf_counter()

                result = self.single_call_rag.run(
                    query,
                    results,
                )

                self.last_stage_execution_times_seconds[
                    "single_call"
                ] = time.perf_counter() - stage_start

                self.last_generation = result.generation

                if not result.sufficient:
                    return RAGResult(
                        answer=ABSTENTION_MESSAGE,
                        sufficient=False,
                        retrieved_chunk_ids=tuple(
                            result.chunk.id
                            for result in results
                        ),
                        selected_chunk_ids=(),
                    )

                if not result.answer.strip():
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
                    answer=result.answer,
                    sufficient=True,
                    retrieved_chunk_ids=tuple(
                        result.chunk.id
                        for result in results
                    ),
                    selected_chunk_ids=result.selected_chunk_ids,
                )
            
            stage_start = time.perf_counter()

            assert self.evidence_evaluator is not None

            decision = self.evidence_evaluator.evaluate(query, results)

            self.last_stage_execution_times_seconds["evidence"] = (
                time.perf_counter() - stage_start
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

            stage_start = time.perf_counter()
            
            generation = self.chat_client.generate(
                messages,
                profile=RAG_ANSWER_PROFILE,
            )

            self.last_stage_execution_times_seconds["answer_generation"] = (
                time.perf_counter() - stage_start
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


        finally:
            self.last_execution_time_seconds = (
                time.perf_counter() - start_time
            )