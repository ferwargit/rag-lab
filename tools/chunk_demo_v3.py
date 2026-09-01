from pathlib import Path

from rag_lab.models import DocumentChunk


def chunk_text(
    text: str,
    *,
    source: str,
    document_id: str,
    max_chars: int = 200,
    overlap: int = 40,
) -> list[DocumentChunk]:
    """
    Divide un documento en chunks respetando límites de párrafo.

    Cada chunk conserva su identidad y procedencia.
    """

    if max_chars <= 0:
        raise ValueError("max_chars debe ser mayor que 0.")

    if overlap < 0:
        raise ValueError("overlap no puede ser negativo.")

    if overlap >= max_chars:
        raise ValueError("overlap debe ser menor que max_chars.")

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks: list[DocumentChunk] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)

        if current and current_length + 1 + paragraph_length > max_chars:
            chunk_index = len(chunks)

            chunks.append(
                DocumentChunk(
                    id=f"{document_id}-{chunk_index:03d}",
                    text="\n".join(current),
                    source=source,
                    index=chunk_index,
                    metadata={
                        "type": "text",
                        "language": "es",
                    },
                )
            )

            overlap_text = current[-1]

            if len(overlap_text) <= overlap:
                current = [overlap_text]
                current_length = len(overlap_text)
            else:
                current = []
                current_length = 0

        current.append(paragraph)
        current_length += (
            paragraph_length
            if len(current) == 1
            else paragraph_length + 1
        )

    if current:
        chunk_index = len(chunks)

        chunks.append(
            DocumentChunk(
                id=f"{document_id}-{chunk_index:03d}",
                text="\n".join(current),
                source=source,
                index=chunk_index,
                metadata={
                    "type": "text",
                    "language": "es",
                },
            )
        )

    return chunks


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

    print(f"Documento: {path}")
    print(f"Caracteres: {len(text)}")
    print(f"Chunks: {len(chunks)}")
    print()

    for chunk in chunks:
        print("=" * 70)
        print(f"ID: {chunk.id}")
        print(f"INDEX: {chunk.index}")
        print(f"SOURCE: {chunk.source}")
        print(f"METADATA: {chunk.metadata}")
        print("=" * 70)
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()