from rag_lab.ingestion.chunking import chunk_text
from rag_lab.ingestion.indexing import embed_chunk, embed_chunks

def test_chunking_returns_document_chunks() -> None:
    text = (
        "Primer párrafo.\n"
        "Segundo párrafo.\n"
        "Tercer párrafo."
    )

    chunks = chunk_text(
        text,
        source="test.txt",
        document_id="test",
        max_chars=100,
        overlap=20,
    )

    assert chunks
    assert all(chunk.source == "test.txt" for chunk in chunks)
    assert all(
        chunk.id.startswith("test-")
        for chunk in chunks
    )


def test_chunk_ids_are_sequential() -> None:
    text = (
        "Primer párrafo.\n"
        "Segundo párrafo.\n"
        "Tercer párrafo.\n"
        "Cuarto párrafo."
    )

    chunks = chunk_text(
        text,
        source="test.txt",
        document_id="test",
        max_chars=30,
        overlap=10,
    )

    ids = [chunk.id for chunk in chunks]

    expected_ids = [
        f"test-{index:03d}"
        for index in range(len(chunks))
    ]

    assert ids == expected_ids


def test_chunking_rejects_invalid_configuration() -> None:
    text = "Texto."

    try:
        chunk_text(
            text,
            source="test.txt",
            document_id="test",
            max_chars=10,
            overlap=10,
        )

        raise AssertionError(
            "Se esperaba ValueError."
        )

    except ValueError as exc:
        assert "overlap" in str(exc)