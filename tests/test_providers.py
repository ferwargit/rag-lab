from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.retrieval import Retriever
from rag_lab.providers import (
    EmbeddingProvider,
    RetrieverProvider,
)


def test_local_embedding_client_matches_provider_shape() -> None:
    client: EmbeddingProvider = LocalEmbeddingClient()

    assert callable(client.embed)


def test_retriever_matches_provider_shape() -> None:
    assert hasattr(Retriever, "search")