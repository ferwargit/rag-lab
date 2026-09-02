from rag_lab.answer_validation import validate_answer_terms


def test_validate_answer_terms_accepts_expected_term() -> None:
    answer = (
        "El piano se conecta al ordenador mediante "
        "una interfaz USB MIDI."
    )

    assert validate_answer_terms(
        answer,
        ["interfaz USB MIDI"],
    ) is True


def test_validate_answer_terms_is_case_insensitive() -> None:
    answer = "La conexión utiliza una INTERFAZ USB MIDI."

    assert validate_answer_terms(
        answer,
        ["interfaz usb midi"],
    ) is True


def test_validate_answer_terms_rejects_missing_term() -> None:
    answer = "El piano se conecta mediante MIDI."

    assert validate_answer_terms(
        answer,
        ["interfaz USB MIDI"],
    ) is False


def test_validate_answer_terms_requires_all_terms() -> None:
    answer = (
        "La interfaz de usuario se ejecuta "
        "en el proceso renderer."
    )

    assert validate_answer_terms(
        answer,
        ["proceso renderer", "Electron"],
    ) is False


def test_validate_answer_terms_accepts_empty_expected_terms() -> None:
    answer = "Cualquier respuesta."

    assert validate_answer_terms(
        answer,
        [],
    ) is True