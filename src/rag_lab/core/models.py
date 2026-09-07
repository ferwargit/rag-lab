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


@dataclass(frozen=True)
class SearchResult:
    """Resultado de una búsqueda semántica."""

    chunk: EmbeddedChunk
    score: float


@dataclass(frozen=True)
class EvidenceDecision:
    """Representa la decisión semántica sobre la evidencia."""

    sufficient: bool
    selected_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    content: str
    reasoning: str | None
    input_tokens: int
    total_output_tokens: int
    reasoning_output_tokens: int
    tokens_per_second: float | None
    time_to_first_token_seconds: float | None
    generation_time_seconds: float
    response_id: str | None = None
    model_instance_id: str | None = None