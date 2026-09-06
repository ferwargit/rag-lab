from collections.abc import Sequence

from rag_lab.core.models import DocumentChunk, EmbeddedChunk
from rag_lab.core.contracts import EmbeddingProvider


def embed_chunk(
    chunk: DocumentChunk,
    client: EmbeddingProvider,
) -> EmbeddedChunk:
    """Genera un embedding para un DocumentChunk."""

    embedding = client.embed(chunk.text)

    return EmbeddedChunk(
        id=chunk.id,
        text=chunk.text,
        source=chunk.source,
        index=chunk.index,
        embedding=tuple(embedding),
        metadata=chunk.metadata.copy(),
    )


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    client: EmbeddingProvider,
) -> list[EmbeddedChunk]:
    """Genera embeddings para una colección de chunks."""

    return [
        embed_chunk(chunk, client)
        for chunk in chunks
    ]
