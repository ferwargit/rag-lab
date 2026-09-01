from pathlib import Path

from rag_lab.context import build_context
from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


def main() -> None:
    store = JsonVectorStore(
        Path("storage/index.json")
    )

    store.load()

    embedding_client = LocalEmbeddingClient()

    query = "¿Cómo se conecta el piano al ordenador?"

    query_embedding = embedding_client.embed(query)

    retriever = Retriever(store)

    results = retriever.search(
        tuple(query_embedding),
        top_k=3,
    )

    context = build_context(results)

    print("PREGUNTA")
    print("=" * 70)
    print(query)
    print()

    print("CONTEXTO RECUPERADO")
    print("=" * 70)
    print(context)


if __name__ == "__main__":
    main()