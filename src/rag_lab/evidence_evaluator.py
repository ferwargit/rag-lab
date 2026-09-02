import json
from collections.abc import Sequence
from typing import Protocol

from rag_lab.evidence import EvidenceDecision
from rag_lab.generation import LocalChatClient, GenerationResult
from rag_lab.retrieval import SearchResult
from rag_lab.inference import EVIDENCE_PROFILE, InferenceProfile


class ChatGenerator(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        profile: InferenceProfile,
    ) -> GenerationResult:
        ...


SYSTEM_INSTRUCTION = """Evalúa las evidencias proporcionadas para determinar
si contienen información suficiente para responder la pregunta.

Puedes combinar información de varias evidencias.

Selecciona únicamente los chunks que aporten información necesaria
para responder la pregunta.

Si las evidencias no permiten responder, sufficient debe ser false.

Devuelve exclusivamente JSON válido con esta forma:

{
  "sufficient": true,
  "selected_chunk_ids": ["chunk-id-1", "chunk-id-2"]
}

o:

{
  "sufficient": false,
  "selected_chunk_ids": []
}

No incluyas explicaciones.

No incluyas Markdown.

No incluyas texto antes o después del JSON.
"""


class EvidenceEvaluator:
    """Evalúa suficiencia y selecciona evidencias relevantes."""

    def __init__(
        self,
        client: ChatGenerator,
    ) -> None:
        self.client = client

    def evaluate(
        self,
        query: str,
        results: Sequence[SearchResult],
    ) -> EvidenceDecision:
        """Evalúa un conjunto de evidencias recuperadas."""

        evidence_sections: list[str] = []

        valid_chunk_ids: set[str] = set()

        for result in results:
            chunk = result.chunk

            valid_chunk_ids.add(chunk.id)

            evidence_sections.append(
                "\n".join(
                    [
                        (
                            f"[Chunk: {chunk.id} | "
                            f"Score: {result.score:.6f}]"
                        ),
                        chunk.text,
                    ]
                )
            )

        evidence = "\n\n".join(evidence_sections)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": (
                    f"Pregunta:\n{query}\n\n"
                    f"Evidencias:\n{evidence}"
                ),
            },
        ]

        generation = self.client.generate(
            messages,
            profile=EVIDENCE_PROFILE,
        )

        raw_response = generation.content

        try:
            parsed = json.loads(raw_response)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "El evaluador de evidencia devolvió JSON inválido."
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "La respuesta del evaluador debe ser un objeto JSON."
            )

        sufficient = parsed.get("sufficient")

        if not isinstance(sufficient, bool):
            raise ValueError(
                "El campo 'sufficient' debe ser booleano."
            )

        selected_ids = parsed.get(
            "selected_chunk_ids",
            [],
        )

        if not isinstance(selected_ids, list):
            raise ValueError(
                "selected_chunk_ids debe ser una lista."
            )

        if not all(
            isinstance(chunk_id, str)
            for chunk_id in selected_ids
        ):
            raise ValueError(
                "Todos los selected_chunk_ids deben ser strings."
            )

        unknown_ids = set(selected_ids) - valid_chunk_ids

        if unknown_ids:
            raise ValueError(
                "El evaluador seleccionó chunks que no fueron "
                f"proporcionados: {sorted(unknown_ids)}"
            )

        if not sufficient and selected_ids:
            raise ValueError(
                "Una decisión insuficiente no puede seleccionar chunks."
            )

        return EvidenceDecision(
            sufficient=sufficient,
            selected_chunk_ids=tuple(selected_ids),
        )


def create_local_evidence_evaluator() -> EvidenceEvaluator:
    """Crea un evaluador utilizando Qwen en LM Studio."""

    return EvidenceEvaluator(
        LocalChatClient()
    )