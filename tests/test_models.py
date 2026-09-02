from dataclasses import FrozenInstanceError
from rag_lab.models import (
    DocumentChunk,
    EmbeddedChunk,
    RAGResult,
)

import pytest


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


def test_rag_result_stores_its_data() -> None:
    result = RAGResult(
        answer="Respuesta de prueba.",
        sufficient=True,
        retrieved_chunk_ids=(
            "chunk-001",
            "chunk-002",
        ),
        selected_chunk_ids=(
            "chunk-001",
        ),
    )

    assert result.answer == "Respuesta de prueba."
    assert result.sufficient is True
    assert result.retrieved_chunk_ids == (
        "chunk-001",
        "chunk-002",
    )
    assert result.selected_chunk_ids == (
        "chunk-001",
    )


def test_rag_result_is_immutable() -> None:
    result = RAGResult(
        answer="Respuesta de prueba.",
        sufficient=True,
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.answer = "Otra respuesta."