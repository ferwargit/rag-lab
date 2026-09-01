from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


def main() -> None:
    index_path = "storage/index.json"

    store = JsonVectorStore(
        path=__import__("pathlib").Path(index_path)
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

    print(f"Pregunta: {query}")
    print()

    for position, result in enumerate(results, start=1):
        chunk = result.chunk

        print("=" * 70)
        print(f"RANK: {position}")
        print(f"SCORE: {result.score:.6f}")
        print(f"ID: {chunk.id}")
        print(f"SOURCE: {chunk.source}")
        print("=" * 70)
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()