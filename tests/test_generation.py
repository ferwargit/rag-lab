from rag_lab.generation import LocalChatClient
from rag_lab.inference import ModelCapabilities


def test_local_chat_client_starts_without_cached_capabilities() -> None:
    client = LocalChatClient()

    assert client._model_capabilities is None


def test_local_chat_client_can_store_model_capabilities() -> None:
    client = LocalChatClient()

    capabilities = ModelCapabilities(
        model_id="qwen/qwen3.5-9b",
        reasoning_options=("off", "on"),
        default_reasoning="on",
    )

    client._model_capabilities = capabilities

    assert client._model_capabilities == capabilities


def test_local_chat_client_fetches_model_capabilities_only_once(
    monkeypatch,
) -> None:
    client = LocalChatClient()

    capabilities = ModelCapabilities(
        model_id="qwen/qwen3.5-9b",
        reasoning_options=("off", "on"),
        default_reasoning="on",
    )

    calls = 0

    def fake_get_model_capabilities() -> ModelCapabilities:
        nonlocal calls
        calls += 1
        return capabilities

    monkeypatch.setattr(
        client,
        "get_model_capabilities",
        fake_get_model_capabilities,
    )

    first = client._model_capabilities

    if first is None:
        client._model_capabilities = client.get_model_capabilities()

    second = client._model_capabilities

    if second is None:
        client._model_capabilities = client.get_model_capabilities()

    assert calls == 1
    assert client._model_capabilities == capabilities


def test_generate_uses_cached_model_capabilities(
    monkeypatch,
) -> None:
    client = LocalChatClient()

    capabilities = ModelCapabilities(
        model_id="qwen/qwen3.5-9b",
        reasoning_options=("off", "on"),
        default_reasoning="on",
    )

    calls = 0

    def fake_get_model_capabilities() -> ModelCapabilities:
        nonlocal calls
        calls += 1
        return capabilities

    monkeypatch.setattr(
        client,
        "get_model_capabilities",
        fake_get_model_capabilities,
    )

    response = {
        "output": [
            {
                "type": "message",
                "content": "Respuesta de prueba.",
            }
        ],
        "stats": {
            "input_tokens": 10,
            "total_output_tokens": 5,
            "reasoning_output_tokens": 0,
            "tokens_per_second": 40.0,
            "time_to_first_token_seconds": 0.2,
        },
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            import json

            return json.dumps(response).encode("utf-8")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    messages = [
        {
            "role": "user",
            "content": "Pregunta de prueba.",
        }
    ]

    client.generate(
        messages,
        profile=__import__(
            "rag_lab.inference",
            fromlist=["EVIDENCE_PROFILE"],
        ).EVIDENCE_PROFILE,
    )

    client.generate(
        messages,
        profile=__import__(
            "rag_lab.inference",
            fromlist=["EVIDENCE_PROFILE"],
        ).EVIDENCE_PROFILE,
    )

    assert calls == 1