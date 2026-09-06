from pathlib import Path

import pytest

from rag_lab.core.models import EmbeddedChunk
from rag_lab.vector_store import JsonVectorStore, VectorStoreError


def make_chunk(
    chunk_id: str = "chunk-001",
    embedding: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> EmbeddedChunk:
    return EmbeddedChunk(
        id=chunk_id,
        text="Texto de prueba",
        source="test.txt",
        index=0,
        embedding=embedding,
    )


def test_vector_store_add_and_count(tmp_path: Path) -> None:
    store = JsonVectorStore(tmp_path / "index.json")

    store.add(make_chunk())

    assert store.count() == 1
    assert store.items[0].id == "chunk-001"


def test_vector_store_rejects_incompatible_dimensions(
    tmp_path: Path,
) -> None:
    store = JsonVectorStore(tmp_path / "index.json")

    store.add(make_chunk())
    incompatible = make_chunk(
        chunk_id="chunk-002",
        embedding=(1.0, 2.0),
    )

    with pytest.raises(VectorStoreError, match="Dimensionalidad incompatible"):
        store.add(incompatible)


def test_vector_store_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "index.json"

    store = JsonVectorStore(path)
    store.add(make_chunk())
    store.save()

    loaded = JsonVectorStore(path)
    loaded.load()

    assert loaded.count() == 1
    assert loaded.items[0] == make_chunk()