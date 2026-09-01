from pathlib import Path


def chunk_text(
    text: str,
    max_chars: int = 200,
    overlap: int = 40,
) -> list[str]:
    """
    Divide un texto respetando límites de párrafo.

    Los párrafos se agrupan hasta aproximarse a max_chars.
    Se utiliza overlap para conservar contexto entre chunks.
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

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)

        if (
            current
            and current_length + 1 + paragraph_length > max_chars
        ):
            chunks.append("\n".join(current))

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
        chunks.append("\n".join(current))

    return chunks


def main() -> None:
    path = Path("data/knowledge.txt")

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    text = path.read_text(encoding="utf-8")

    chunks = chunk_text(
        text,
        max_chars=200,
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