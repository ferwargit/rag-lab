from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    """Representa un fragmento de un documento."""

    id: str
    text: str
    source: str
    index: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    """Representa un chunk junto con su embedding."""

    id: str
    text: str
    source: str
    index: int
    embedding: tuple[float, ...]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RAGResult:
    """Resultado final de una consulta al sistema RAG."""

    answer: str
    sufficient: bool
    retrieved_chunk_ids: tuple[str, ...] = ()
    selected_chunk_ids: tuple[str, ...] = ()
