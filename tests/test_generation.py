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