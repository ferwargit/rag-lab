from collections.abc import Sequence

from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.models import DocumentChunk, EmbeddedChunk


def embed_chunk(
    chunk: DocumentChunk,
    client: LocalEmbeddingClient,
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
    client: LocalEmbeddingClient,
) -> list[EmbeddedChunk]:
    """Genera embeddings para una colección de chunks."""

    return [
        embed_chunk(chunk, client)
        for chunk in chunks
    ]