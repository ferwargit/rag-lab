import json
from collections.abc import Sequence
from typing import Protocol

from rag_lab.generation import LocalChatClient
from rag_lab.retrieval import SearchResult
from rag_lab.evidence import EvidenceDecision


class ChatGenerator(Protocol):
    """Interfaz mínima que necesita el evaluador."""

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        ...


SYSTEM_INSTRUCTION = """Evalúa si el conjunto de evidencias contiene
información suficiente para responder la pregunta.

Puedes combinar información de varias evidencias.

No inventes información que no esté respaldada por las evidencias.

Devuelve exclusivamente JSON válido con esta forma:

{"sufficient": true}

o:

{"sufficient": false}
"""


class EvidenceEvaluator:
    """Evalúa si las evidencias recuperadas son suficientes."""

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
        """Evalúa un conjunto de evidencias."""

        evidence_sections: list[str] = []

        for result in results:
            chunk = result.chunk

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

        raw_response = self.client.generate(messages)

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "El evaluador de evidencia devolvió "
                "JSON inválido."
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

        return EvidenceDecision(
            sufficient=sufficient,
        )


def create_local_evidence_evaluator() -> EvidenceEvaluator:
    """Crea un evaluador utilizando Qwen en LM Studio."""

    return EvidenceEvaluator(
        client=LocalChatClient(),
    )