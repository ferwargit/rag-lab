import json
from pathlib import Path
from typing import Iterable

from rag_lab.models import EmbeddedChunk


class VectorStoreError(RuntimeError):
    """Error producido por el almacenamiento vectorial."""


class JsonVectorStore:
    """Almacenamiento vectorial mínimo basado en JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._items: list[EmbeddedChunk] = []

    @property
    def dimension(self) -> int | None:
        """Devuelve la dimensionalidad del índice."""
        if not self._items:
            return None

        return len(self._items[0].embedding)

    @property
    def items(self) -> tuple[EmbeddedChunk, ...]:
        """Devuelve los elementos almacenados de forma inmutable."""
        return tuple(self._items)

    def add(self, item: EmbeddedChunk) -> None:
        """Agrega un chunk vectorizado al índice."""

        item_dimension = len(item.embedding)

        if self.dimension is not None:
            if item_dimension != self.dimension:
                raise VectorStoreError(
                    "Dimensionalidad incompatible: "
                    f"índice={self.dimension}, item={item_dimension}"
                )

        self._items.append(item)

    def add_many(self, items: Iterable[EmbeddedChunk]) -> None:
        """Agrega múltiples chunks."""

        for item in items:
            self.add(item)

    def count(self) -> int:
        """Cantidad de elementos almacenados."""
        return len(self._items)

    def save(self) -> None:
        """Persiste el índice en disco."""

        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "id": item.id,
                "text": item.text,
                "source": item.source,
                "index": item.index,
                "embedding": list(item.embedding),
                "metadata": item.metadata,
            }
            for item in self._items
        ]

        self.path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> None:
        """Carga el índice desde disco."""

        if not self.path.exists():
            raise FileNotFoundError(
                f"No existe el índice: {self.path}"
            )

        raw = json.loads(
            self.path.read_text(encoding="utf-8")
        )

        if not isinstance(raw, list):
            raise VectorStoreError(
                "El índice JSON debe contener una lista."
            )

        self._items.clear()

        for entry in raw:
            item = EmbeddedChunk(
                id=entry["id"],
                text=entry["text"],
                source=entry["source"],
                index=entry["index"],
                embedding=tuple(
                    float(value)
                    for value in entry["embedding"]
                ),
                metadata=dict(entry.get("metadata", {})),
            )

            self.add(item)