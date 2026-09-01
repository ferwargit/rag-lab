from pathlib import Path

from rag_lab.chunking import chunk_text
from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.indexing import embed_chunks
from rag_lab.vector_store import JsonVectorStore


def main() -> None:
    document_path = Path("data/knowledge.txt")
    index_path = Path("storage/index.json")

    if not document_path.exists():
        raise FileNotFoundError(
            f"No existe el documento: {document_path}"
        )

    text = document_path.read_text(encoding="utf-8")

    chunks = chunk_text(
        text,
        source=str(document_path),
        document_id="knowledge",
        max_chars=200,
        overlap=40,
    )

    embedding_client = LocalEmbeddingClient()

    embedded_chunks = embed_chunks(
        chunks,
        embedding_client,
    )

    store = JsonVectorStore(index_path)

    store.add_many(embedded_chunks)
    store.save()

    print(f"Documento: {document_path}")
    print(f"Chunks indexados: {store.count()}")
    print(f"Dimensiones: {store.dimension}")
    print(f"Índice: {index_path}")


if __name__ == "__main__":
    main()