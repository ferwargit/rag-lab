from pathlib import Path


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[str]:
    """Divide un texto en fragmentos con solapamiento."""

    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0.")

    if overlap < 0:
        raise ValueError("overlap no puede ser negativo.")

    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size.")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def main() -> None:
    path = Path("data/knowledge.txt")

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    text = path.read_text(encoding="utf-8")

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap=40,
    )

    print(f"Documento: {path}")
    print(f"Caracteres: {len(text)}")
    print(f"Chunks: {len(chunks)}")
    print()

    for index, chunk in enumerate(chunks):
        print("=" * 70)
        print(f"CHUNK {index}")
        print("=" * 70)
        print(chunk)
        print()


if __name__ == "__main__":
    main()