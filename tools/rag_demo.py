from pathlib import Path

from rag_lab.context import build_context
from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.generation import LocalChatClient
from rag_lab.prompting import build_rag_messages
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


def main() -> None:
    store = JsonVectorStore(
        Path("storage/index.json")
    )

    store.load()

    query = "¿Cómo se conecta el piano al ordenador?"

    embedding_client = LocalEmbeddingClient()

    query_embedding = embedding_client.embed(query)

    retriever = Retriever(store)

    results = retriever.search(
        tuple(query_embedding),
        top_k=3,
    )

    messages = build_rag_messages(
        query,
        results,
    )

    chat_client = LocalChatClient()

    answer = chat_client.generate(messages)

    print("PREGUNTA")
    print("=" * 70)
    print(query)
    print()

    print("RESPUESTA RAG")
    print("=" * 70)
    print(answer)
    print()

    print("FUENTES RECUPERADAS")
    print("=" * 70)

    for position, result in enumerate(results, start=1):
        print(
            f"{position}. "
            f"{result.chunk.id} "
            f"(score={result.score:.6f})"
        )


if __name__ == "__main__":
    main()