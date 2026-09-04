import json
from collections.abc import Sequence
from dataclasses import dataclass

from rag_lab.generation import GenerationResult
from rag_lab.inference import RAG_ANSWER_PROFILE
from rag_lab.prompting import build_rag_messages
from rag_lab.retrieval import SearchResult
from rag_lab.providers import ChatGenerator


SINGLE_CALL_SYSTEM_INSTRUCTION = """Eres un asistente RAG.

Debes decidir si el contexto recuperado contiene información
suficiente para responder la pregunta y, si la contiene,
responderla.

Debes devolver exclusivamente un objeto JSON válido.

Formato obligatorio:

{
  "sufficient": true,
  "answer": "respuesta"
}

Reglas:

- "sufficient" debe ser un booleano.
- "answer" debe ser un string.
- Utiliza exclusivamente la información del contexto.
- No inventes información.
- Si el contexto NO contiene información suficiente para
  responder, usa "sufficient": false.
- Cuando "sufficient" sea false, explica brevemente que
  no hay información suficiente.
- Cuando "sufficient" sea true, responde de forma precisa
  y concisa.
- No incluyas markdown.
- No incluyas texto fuera del JSON.
- Responde en español.
"""


@dataclass(frozen=True)
class SingleCallResult:
    """Resultado de una ejecución single-call."""

    answer: str
    sufficient: bool
    generation: GenerationResult


class SingleCallRAG:
    """Evalúa suficiencia y genera la respuesta en una sola llamada."""

    def __init__(self, client: ChatGenerator) -> None:
        self.client = client
        self.last_generation: GenerationResult | None = None

    def run(
        self,
        query: str,
        results: Sequence[SearchResult],
    ) -> SingleCallResult:
        """Ejecuta una única generación para evaluar y responder."""

        self.last_generation = None

        base_messages = build_rag_messages(
            query,
            list(results),
        )

        user_content = base_messages[1]["content"]

        messages = [
            {
                "role": "system",
                "content": SINGLE_CALL_SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": (
                    f"{user_content}\n\n"
                    "Devuelve únicamente el JSON solicitado."
                ),
            },
        ]

        generation = self.client.generate(
            messages,
            profile=RAG_ANSWER_PROFILE,
        )

        self.last_generation = generation

        try:
            parsed = json.loads(generation.content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "La respuesta del single-call contiene JSON inválido. "
                f"Respuesta recibida: {generation.content!r}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "La respuesta del single-call debe ser un objeto JSON."
            )

        sufficient = parsed.get("sufficient")

        if not isinstance(sufficient, bool):
            raise ValueError(
                "El campo 'sufficient' debe ser booleano."
            )

        answer = parsed.get("answer")

        if not isinstance(answer, str):
            raise ValueError(
                "El campo 'answer' debe ser un string."
            )

        if not answer.strip():
            raise ValueError(
                "El campo 'answer' no puede estar vacío."
            )

        return SingleCallResult(
            answer=answer,
            sufficient=sufficient,
            generation=generation,
        )