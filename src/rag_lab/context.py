from collections.abc import Sequence

from rag_lab.retrieval import SearchResult


def build_context(
    results: Sequence[SearchResult],
) -> str:
    """Construye el contexto textual que recibirá el LLM."""

    sections: list[str] = []

    for result in results:
        chunk = result.chunk

        sections.append(
            "\n".join(
                [
                    (
                        f"[Fuente: {chunk.source} | "
                        f"Chunk: {chunk.id} | "
                        f"Score: {result.score:.6f}]"
                    ),
                    chunk.text,
                ]
            )
        )

    return "\n\n".join(sections)