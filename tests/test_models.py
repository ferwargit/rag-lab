from rag_lab.models import DocumentChunk


def test_document_chunk_stores_its_data() -> None:
    chunk = DocumentChunk(
        id="test-001",
        text="Texto de prueba.",
        source="test.txt",
        index=0,
        metadata={"type": "text"},
    )

    assert chunk.id == "test-001"
    assert chunk.text == "Texto de prueba."
    assert chunk.source == "test.txt"
    assert chunk.index == 0
    assert chunk.metadata["type"] == "text"