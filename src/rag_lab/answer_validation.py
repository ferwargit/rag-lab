from collections.abc import Sequence


def validate_answer_terms(
    answer: str,
    expected_terms: Sequence[str],
) -> bool:
    """Comprueba si una respuesta contiene todos los términos esperados."""

    normalized_answer = answer.casefold()

    return all(
        term.casefold() in normalized_answer
        for term in expected_terms
    )