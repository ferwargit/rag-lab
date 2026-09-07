from rag_lab.core.models import SearchResult
from rag_lab.generation.context import build_context


SYSTEM_INSTRUCTION = """Eres un asistente que responde preguntas utilizando
únicamente la información proporcionada en el contexto recuperado.

Reglas:
- Utiliza exclusivamente el contexto.
- Si la información necesaria no aparece en el contexto, dilo claramente.
- No inventes datos.
- Responde en español.
- Sé preciso y conciso.
"""


def build_rag_messages(
    query: str,
    results: list[SearchResult],
) -> list[dict[str, str]]:
    """Construye los mensajes que recibirá el LLM."""

    context = build_context(results)

    user_message = f"""Contexto recuperado:

{context}

Pregunta:
{query}

Responde utilizando únicamente el contexto recuperado."""

    return [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]