from typing import Sequence

import urllib.error
import urllib.request
import json


class EmbeddingError(RuntimeError):
    """Error producido al generar embeddings."""


class LocalEmbeddingClient:
    """Cliente mínimo para el endpoint de embeddings de LM Studio."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:1234",
        model: str = "text-embedding-nomic-embed-text-v1.5",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, text: str) -> list[float]:
        """Genera un embedding para un único texto."""

        if not text.strip():
            raise ValueError("No se puede generar un embedding para texto vacío.")

        payload = json.dumps(
            {
                "model": self.model,
                "input": [text],
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.base_url}/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw_data = response.read()

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingError(
                f"LM Studio respondió HTTP {exc.code}: {detail}"
            ) from exc

        except urllib.error.URLError as exc:
            raise EmbeddingError(
                f"No se pudo conectar con LM Studio: {exc.reason}"
            ) from exc

        try:
            result = json.loads(raw_data)

        except json.JSONDecodeError as exc:
            raise EmbeddingError(
                "LM Studio devolvió una respuesta que no es JSON válido."
            ) from exc

        try:
            embedding = result["data"][0]["embedding"]

        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(
                "La respuesta de LM Studio no contiene el embedding esperado."
            ) from exc

        if not isinstance(embedding, list):
            raise EmbeddingError("El embedding recibido no es una lista.")

        return [float(value) for value in embedding]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Genera embeddings para varios textos."""

        return [self.embed(text) for text in texts]