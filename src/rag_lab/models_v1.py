from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    """Representa un fragmento de un documento."""

    id: str
    text: str
    source: str
    index: int
    metadata: dict[str, str] = field(default_factory=dict)