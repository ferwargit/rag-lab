import pytest

from rag_lab.models import RAGResult
from rag_lab.retrieval import SearchResult
from rag_lab.rag_pipeline import (
    ABSTENTION_MESSAGE,
    RAGPipeline,
)
from rag_lab.generation import GenerationResult


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0]


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[tuple[float, ...], int, float | None]] = []

    def search(
        self,
        query_embedding: tuple[float, ...],
        *,
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        self.calls.append(
            (
                query_embedding,
                top_k,
                score_threshold,
            )
        )
        return list(self.results)


class FakeEvidenceEvaluator:
    def __init__(self, decision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, list[SearchResult]]] = []

    def evaluate(
        self,
        query: str,
        results: list[SearchResult],
    ):
        self.calls.append((query, list(results)))
        return self.decision


class FakeChatClient:
    def __init__(self, generation_result) -> None:
        self.generation_result = generation_result
        self.calls = []

    def generate(self, messages, *, profile):
        self.calls.append(
            {
                "messages": messages,
                "profile": profile,
            }
        )
        return self.generation_result


def make_search_result(
    chunk_id: str,
    text: str,
    score: float,
) -> SearchResult:
    from rag_lab.models import EmbeddedChunk

    chunk = EmbeddedChunk(
        id=chunk_id,
        text=text,
        source="test.txt",
        index=0,
        embedding=(1.0, 0.0),
    )

    return SearchResult(
        chunk=chunk,
        score=score,
    )


class DummyEmbeddingClient:
    pass


class DummyRetriever:
    pass


class DummyEvidenceEvaluator:
    pass


class DummyChatClient:
    pass


def test_rag_pipeline_stores_dependencies() -> None:
    embedding_client = DummyEmbeddingClient()
    retriever = DummyRetriever()
    evidence_evaluator = DummyEvidenceEvaluator()
    chat_client = DummyChatClient()

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
        top_k=3,
    )

    assert pipeline.embedding_client is embedding_client
    assert pipeline.retriever is retriever
    assert (
        pipeline.evidence_evaluator
        is evidence_evaluator
    )
    assert pipeline.chat_client is chat_client
    assert pipeline.top_k == 3


def test_rag_pipeline_rejects_invalid_top_k() -> None:
    with pytest.raises(
        ValueError,
        match="top_k debe ser mayor que cero",
    ):
        RAGPipeline(
            embedding_client=DummyEmbeddingClient(),
            retriever=DummyRetriever(),
            evidence_evaluator=DummyEvidenceEvaluator(),
            chat_client=DummyChatClient(),
            top_k=0,
        )



def test_abstention_message_is_defined() -> None:
    assert ABSTENTION_MESSAGE.startswith(
        "No tengo información suficiente"
    )


def test_rag_pipeline_abstains_when_evidence_is_insufficient() -> None:
    from rag_lab.evidence_evaluator import EvidenceDecision
    from rag_lab.generation import GenerationResult

    retrieved = [
        make_search_result(
            "knowledge-000",
            "MIDI Laboratory es una aplicación personal desarrollada con Electron.",
            0.80,
        )
    ]

    embedding_client = FakeEmbeddingClient()
    retriever = FakeRetriever(retrieved)

    evidence_evaluator = FakeEvidenceEvaluator(
        EvidenceDecision(
            sufficient=False,
            selected_chunk_ids=(),
        )
    )

    chat_client = FakeChatClient(
        GenerationResult(
            content="NO DEBERÍA GENERARSE",
            reasoning=None,
            input_tokens=0,
            total_output_tokens=0,
            reasoning_output_tokens=0,
            tokens_per_second=None,
            time_to_first_token_seconds=None,
        )
    )

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
        top_k=3,
    )

    result = pipeline.ask(
        "¿Qué sistema operativo utiliza MIDI Laboratory?"
    )

    assert isinstance(result, RAGResult)
    assert result.sufficient is False
    assert result.answer == ABSTENTION_MESSAGE
    assert result.retrieved_chunk_ids == ("knowledge-000",)
    assert result.selected_chunk_ids == ()

    assert embedding_client.calls == [
        "¿Qué sistema operativo utiliza MIDI Laboratory?"
    ]

    assert len(retriever.calls) == 1
    assert retriever.calls[0][1] == 3

    assert len(evidence_evaluator.calls) == 1

    assert chat_client.calls == []


def test_rag_pipeline_selects_evidence_before_generation() -> None:
    from rag_lab.evidence_evaluator import EvidenceDecision

    retrieved = [
        make_search_result(
            "knowledge-001",
            "El dispositivo MIDI se conecta al ordenador mediante una interfaz USB MIDI.",
            0.80,
        ),
        make_search_result(
            "knowledge-000",
            "MIDI Laboratory es una aplicación personal desarrollada con Electron.",
            0.70,
        ),
    ]

    embedding_client = FakeEmbeddingClient()
    retriever = FakeRetriever(retrieved)

    evidence_evaluator = FakeEvidenceEvaluator(
        EvidenceDecision(
            sufficient=True,
            selected_chunk_ids=("knowledge-001",),
        )
    )

    chat_client = FakeChatClient(
        generation_result=GenerationResult(
            content="El piano se conecta al ordenador mediante una interfaz USB MIDI.",
            reasoning=None,
            input_tokens=100,
            total_output_tokens=15,
            reasoning_output_tokens=0,
            tokens_per_second=40.0,
            time_to_first_token_seconds=0.2,
        ),
    )

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
    )

    result = pipeline.ask(
        "¿Cómo se conecta el piano al ordenador?"
    )

    assert result.sufficient is True
    assert result.retrieved_chunk_ids == (
        "knowledge-001",
        "knowledge-000",
    )
    assert result.selected_chunk_ids == (
        "knowledge-001",
    )

    assert len(evidence_evaluator.calls) == 1

    assert evidence_evaluator.calls[0][0] == (
        "¿Cómo se conecta el piano al ordenador?"
    )

    assert len(chat_client.calls) == 1
    assert chat_client.calls[0]["profile"].name == "rag-answer"

    messages = chat_client.calls[0]["messages"]

    assert len(messages) == 2

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_content = messages[1]["content"]

    assert "knowledge-001" in user_content
    assert (
        "El dispositivo MIDI se conecta al ordenador mediante "
        "una interfaz USB MIDI."
    ) in user_content

    assert "knowledge-000" not in user_content
    assert (
        "MIDI Laboratory es una aplicación personal desarrollada "
        "con Electron."
    ) not in user_content


def test_rag_pipeline_abstains_when_sufficient_evidence_has_no_selection() -> None:
    from rag_lab.evidence_evaluator import EvidenceDecision

    retrieved = [
        make_search_result(
            "knowledge-001",
            "El dispositivo MIDI se conecta al ordenador mediante una interfaz USB MIDI.",
            0.80,
        )
    ]

    embedding_client = FakeEmbeddingClient()
    retriever = FakeRetriever(retrieved)

    evidence_evaluator = FakeEvidenceEvaluator(
        EvidenceDecision(
            sufficient=True,
            selected_chunk_ids=(),
        )
    )

    chat_client = FakeChatClient(
        generation_result=None,
    )

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
    )

    result = pipeline.ask(
        "¿Cómo se conecta el piano al ordenador?"
    )

    assert result.sufficient is False
    assert result.answer == ABSTENTION_MESSAGE

    assert result.retrieved_chunk_ids == (
        "knowledge-001",
    )

    assert result.selected_chunk_ids == ()

    assert chat_client.calls == []


def test_rag_pipeline_abstains_when_retrieval_returns_no_results() -> None:
    embedding_client = FakeEmbeddingClient()

    retriever = FakeRetriever(
        results=[],
    )

    evidence_evaluator = FakeEvidenceEvaluator(
        decision=None,
    )

    chat_client = FakeChatClient(
        generation_result=None,
    )

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
    )

    result = pipeline.ask(
        "¿Qué información desconocida existe?"
    )

    assert result.sufficient is False
    assert result.answer == ABSTENTION_MESSAGE
    assert result.retrieved_chunk_ids == ()
    assert result.selected_chunk_ids == ()

    assert len(embedding_client.calls) == 1
    assert len(retriever.calls) == 1

    assert evidence_evaluator.calls == []
    assert chat_client.calls == []


def test_rag_pipeline_abstains_when_generation_returns_empty_content() -> None:
    from rag_lab.evidence_evaluator import EvidenceDecision
    from rag_lab.generation import GenerationResult

    retrieved = [
        make_search_result(
            "knowledge-001",
            "El dispositivo MIDI se conecta al ordenador mediante una interfaz USB MIDI.",
            0.80,
        )
    ]

    embedding_client = FakeEmbeddingClient()
    retriever = FakeRetriever(retrieved)

    evidence_evaluator = FakeEvidenceEvaluator(
        EvidenceDecision(
            sufficient=True,
            selected_chunk_ids=("knowledge-001",),
        )
    )

    chat_client = FakeChatClient(
        generation_result=GenerationResult(
            content="",
            reasoning=None,
            input_tokens=100,
            total_output_tokens=0,
            reasoning_output_tokens=0,
            tokens_per_second=None,
            time_to_first_token_seconds=None,
        ),
    )

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
    )

    result = pipeline.ask(
        "¿Cómo se conecta el piano al ordenador?"
    )

    assert result.sufficient is False
    assert result.answer == ABSTENTION_MESSAGE
    assert result.retrieved_chunk_ids == ("knowledge-001",)
    assert result.selected_chunk_ids == ()


def test_rag_pipeline_exposes_last_metrics() -> None:
    from rag_lab.evidence_evaluator import EvidenceDecision
    from rag_lab.generation import GenerationResult

    retrieved = [
        make_search_result(
            "knowledge-001",
            (
                "El dispositivo MIDI se conecta al ordenador "
                "mediante una interfaz USB MIDI."
            ),
            0.80,
        )
    ]

    embedding_client = FakeEmbeddingClient()
    retriever = FakeRetriever(retrieved)

    evidence_evaluator = FakeEvidenceEvaluator(
        EvidenceDecision(
            sufficient=True,
            selected_chunk_ids=("knowledge-001",),
        )
    )

    generation = GenerationResult(
        content="Se conecta mediante una interfaz USB MIDI.",
        reasoning=None,
        input_tokens=100,
        total_output_tokens=12,
        reasoning_output_tokens=0,
        tokens_per_second=40.0,
        time_to_first_token_seconds=0.2,
    )

    chat_client = FakeChatClient(
        generation_result=generation,
    )

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
    )

    assert pipeline.last_metrics is None

    result = pipeline.ask(
        "¿Cómo se conecta el piano al ordenador?"
    )

    assert result.sufficient is True
    assert pipeline.last_metrics is not None
    assert pipeline.last_metrics.input_tokens == 100
    assert pipeline.last_metrics.total_output_tokens == 12
    assert pipeline.last_metrics.reasoning_output_tokens == 0
    assert pipeline.last_metrics.tokens_per_second == 40.0
    assert pipeline.last_metrics.time_to_first_token_seconds == 0.2


def test_rag_pipeline_clears_last_generation_before_each_ask() -> None:
    from rag_lab.evidence_evaluator import EvidenceDecision
    from rag_lab.generation import GenerationResult

    retrieved = [
        make_search_result(
            "knowledge-001",
            (
                "El dispositivo MIDI se conecta al ordenador "
                "mediante una interfaz USB MIDI."
            ),
            0.80,
        )
    ]

    embedding_client = FakeEmbeddingClient()
    retriever = FakeRetriever(retrieved)

    evidence_evaluator = FakeEvidenceEvaluator(
        EvidenceDecision(
            sufficient=True,
            selected_chunk_ids=("knowledge-001",),
        )
    )

    chat_client = FakeChatClient(
        generation_result=GenerationResult(
            content="Respuesta válida.",
            reasoning=None,
            input_tokens=100,
            total_output_tokens=10,
            reasoning_output_tokens=0,
            tokens_per_second=40.0,
            time_to_first_token_seconds=0.2,
        )
    )

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
    )

    first_result = pipeline.ask("Primera pregunta")

    assert first_result.sufficient is True
    assert pipeline.last_metrics is not None

    pipeline.evidence_evaluator = FakeEvidenceEvaluator(
        EvidenceDecision(
            sufficient=False,
            selected_chunk_ids=(),
        )
    )

    second_result = pipeline.ask("Segunda pregunta")

    assert second_result.sufficient is False
    assert pipeline.last_generation is None
    assert pipeline.last_metrics is None