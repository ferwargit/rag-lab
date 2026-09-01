from pathlib import Path

from rag_lab.chunking import chunk_text
from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.indexing import embed_chunks


def main() -> None:
    path = Path("data/knowledge.txt")

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    text = path.read_text(encoding="utf-8")

    chunks = chunk_text(
        text,
        source=str(path),
        document_id="knowledge",
        max_chars=200,
        overlap=40,
    )

    client = LocalEmbeddingClient()

    embedded_chunks = embed_chunks(
        chunks,
        client,
    )

    print(f"Documento: {path}")
    print(f"Chunks: {len(embedded_chunks)}")
    print()

    for chunk in embedded_chunks:
        print("=" * 70)
        print(f"ID: {chunk.id}")
        print(f"SOURCE: {chunk.source}")
        print(f"DIMENSIONS: {len(chunk.embedding)}")
        print(f"FIRST 5 VALUES: {chunk.embedding[:5]}")
        print("=" * 70)
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()