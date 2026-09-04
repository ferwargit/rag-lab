import pytest

from rag_lab.generation import GenerationResult
from rag_lab.inference import InferenceProfile
from rag_lab.models import EmbeddedChunk
from rag_lab.retrieval import SearchResult


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        profile: InferenceProfile,
    ) -> GenerationResult:
        self.calls.append(
            {
                "messages": messages,
                "profile": profile,
            }
        )

        return GenerationResult(
            content=self.response,
            reasoning=None,
            input_tokens=10,
            total_output_tokens=5,
            reasoning_output_tokens=0,
            tokens_per_second=40.0,
            time_to_first_token_seconds=0.2,
            generation_time_seconds=0.1,
        )


def make_result(
    chunk_id: str = "chunk-001",
    text: str = "Contenido de prueba.",
) -> SearchResult:
    chunk = EmbeddedChunk(
        id=chunk_id,
        text=text,
        source="test.txt",
        index=0,
        embedding=(1.0, 0.0),
    )

    return SearchResult(
        chunk=chunk,
        score=0.9,
    )


def test_single_call_parses_sufficient_true() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient(
        """
        {
          "sufficient": true,
          "answer": "Respuesta correcta."
        }
        """
    )

    single_call = SingleCallRAG(client)

    result = single_call.run(
        "Pregunta de prueba",
        [make_result()],
    )

    assert result.sufficient is True
    assert result.answer == "Respuesta correcta."


def test_single_call_parses_sufficient_false() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient(
        """
        {
          "sufficient": false,
          "answer": "No hay información suficiente."
        }
        """
    )

    single_call = SingleCallRAG(client)

    result = single_call.run(
        "Pregunta de prueba",
        [make_result()],
    )

    assert result.sufficient is False
    assert result.answer == "No hay información suficiente."


def test_single_call_rejects_invalid_json() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient("NO ES JSON")

    single_call = SingleCallRAG(client)

    with pytest.raises(
        ValueError,
        match="JSON inválido",
    ):
        single_call.run(
            "Pregunta de prueba",
            [make_result()],
        )


def test_single_call_calls_llm_exactly_once() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient(
        """
        {
          "sufficient": true,
          "answer": "Respuesta correcta."
        }
        """
    )

    single_call = SingleCallRAG(client)

    single_call.run(
        "Pregunta de prueba",
        [make_result()],
    )

    assert len(client.calls) == 1


def test_single_call_rejects_non_object_json() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient(
        '["sufficient", true]'
    )

    single_call = SingleCallRAG(client)

    with pytest.raises(
        ValueError,
        match="debe ser un objeto JSON",
    ):
        single_call.run(
            "Pregunta de prueba",
            [make_result()],
        )


def test_single_call_rejects_invalid_sufficient_type() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient(
        """
        {
          "sufficient": "yes",
          "answer": "Respuesta."
        }
        """
    )

    single_call = SingleCallRAG(client)

    with pytest.raises(
        ValueError,
        match="sufficient.*booleano",
    ):
        single_call.run(
            "Pregunta de prueba",
            [make_result()],
        )


def test_single_call_rejects_invalid_answer_type() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient(
        """
        {
          "sufficient": true,
          "answer": 123
        }
        """
    )

    single_call = SingleCallRAG(client)

    with pytest.raises(
        ValueError,
        match="answer.*string",
    ):
        single_call.run(
            "Pregunta de prueba",
            [make_result()],
        )


def test_single_call_rejects_empty_answer() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient(
        """
        {
          "sufficient": true,
          "answer": "   "
        }
        """
    )

    single_call = SingleCallRAG(client)

    with pytest.raises(
        ValueError,
        match="answer.*vacío",
    ):
        single_call.run(
            "Pregunta de prueba",
            [make_result()],
        )


def test_single_call_builds_expected_messages() -> None:
    from rag_lab.single_call import (
        SINGLE_CALL_SYSTEM_INSTRUCTION,
        SingleCallRAG,
    )

    client = FakeChatClient(
        """
        {
          "sufficient": true,
          "answer": "Respuesta correcta."
        }
        """
    )

    single_call = SingleCallRAG(client)

    single_call.run(
        "¿Cómo se conecta el piano?",
        [
            make_result(
                chunk_id="knowledge-001",
                text="El piano se conecta mediante una interfaz USB MIDI.",
            )
        ],
    )

    assert len(client.calls) == 1

    messages = client.calls[0]["messages"]

    assert len(messages) == 2

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SINGLE_CALL_SYSTEM_INSTRUCTION

    assert messages[1]["role"] == "user"

    user_content = messages[1]["content"]

    assert "¿Cómo se conecta el piano?" in user_content
    assert (
        "El piano se conecta mediante una interfaz USB MIDI."
        in user_content
    )

    assert "Devuelve únicamente el JSON solicitado." in user_content


def test_single_call_exposes_last_metrics() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient(
        """
        {
          "sufficient": true,
          "answer": "Respuesta correcta."
        }
        """
    )

    single_call = SingleCallRAG(client)

    assert single_call.last_metrics is None

    single_call.run(
        "Pregunta de prueba",
        [make_result()],
    )

    assert single_call.last_metrics is not None
    assert single_call.last_metrics.input_tokens == 10
    assert single_call.last_metrics.total_output_tokens == 5
    assert single_call.last_metrics.reasoning_output_tokens == 0
    assert single_call.last_metrics.tokens_per_second == 40.0
    assert single_call.last_metrics.time_to_first_token_seconds == 0.2
    assert single_call.last_metrics.generation_time_seconds == 0.1


def test_single_call_false_does_not_select_chunks() -> None:
    from rag_lab.single_call import SingleCallRAG

    client = FakeChatClient(
        """
        {
          "sufficient": false,
          "answer": "No hay información suficiente."
        }
        """
    )

    single_call = SingleCallRAG(client)

    result = single_call.run(
        "Pregunta de prueba",
        [make_result()],
    )

    assert result.sufficient is False
    assert result.answer == "No hay información suficiente."

    # La decisión insuficiente no debe permitir selección.
    assert result.selected_chunk_ids == ()