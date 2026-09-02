import pytest

from rag_lab.evidence_evaluator import EvidenceEvaluator
from rag_lab.retrieval import SearchResult
from rag_lab.models import EmbeddedChunk


class FakeChatClient:
    """Cliente falso para pruebas unitarias."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        return self.response


def make_result() -> SearchResult:
    chunk = EmbeddedChunk(
        id="chunk-001",
        text="Texto de prueba.",
        source="test.txt",
        index=0,
        embedding=(1.0, 0.0, 0.0),
        metadata={},
    )

    return SearchResult(
        chunk=chunk,
        score=0.8,
    )


def test_evaluator_parses_true() -> None:
    client = FakeChatClient(
        '{"sufficient": true}'
    )

    evaluator = EvidenceEvaluator(client)

    decision = evaluator.evaluate(
        "Pregunta de prueba",
        [make_result()],
    )

    assert decision.sufficient is True


def test_evaluator_parses_false() -> None:
    client = FakeChatClient(
        '{"sufficient": false}'
    )

    evaluator = EvidenceEvaluator(client)

    decision = evaluator.evaluate(
        "Pregunta de prueba",
        [make_result()],
    )

    assert decision.sufficient is False


def test_evaluator_rejects_invalid_json() -> None:
    client = FakeChatClient(
        "NO ES JSON"
    )

    evaluator = EvidenceEvaluator(client)

    with pytest.raises(
        ValueError,
        match="JSON inválido",
    ):
        evaluator.evaluate(
            "Pregunta de prueba",
            [make_result()],
        )


def test_evaluator_rejects_invalid_sufficient_value() -> None:
    client = FakeChatClient(
        '{"sufficient": "yes"}'
    )

    evaluator = EvidenceEvaluator(client)

    with pytest.raises(
        ValueError,
        match="debe ser booleano",
    ):
        evaluator.evaluate(
            "Pregunta de prueba",
            [make_result()],
        )