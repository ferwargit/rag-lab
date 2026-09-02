from pathlib import Path

from rag_lab.benchmark import BenchmarkCase, load_benchmark, evaluate_benchmark_case
from rag_lab.models import RAGResult

BENCHMARK_PATH = Path("data/benchmark.json")


def test_load_benchmark_returns_four_cases() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

    assert len(cases) == 4


def test_load_benchmark_preserves_answerability() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

    assert [case.answerable for case in cases] == [
        True,
        True,
        True,
        False,
    ]


def test_load_benchmark_preserves_expected_chunks() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

    assert cases[0].expected_chunk_ids == (
        "knowledge-001",
    )

    assert cases[1].expected_chunk_ids == (
        "knowledge-002",
    )

    assert cases[2].expected_chunk_ids == (
        "knowledge-003",
    )

    assert cases[3].expected_chunk_ids == ()


def test_load_benchmark_preserves_expected_answer_terms() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

    assert cases[0].expected_answer_terms == (
        "interfaz USB MIDI",
    )

    assert cases[1].expected_answer_terms == (
        "proceso renderer",
    )

    assert cases[2].expected_answer_terms == (
        "adaptar progresivamente los ejercicios",
    )

    assert cases[3].expected_answer_terms == ()


def test_benchmark_case_is_immutable() -> None:
    case = BenchmarkCase(
        id="q001",
        query="Pregunta",
        answerable=True,
        expected_chunk_ids=("knowledge-001",),
        expected_answer_terms=("interfaz USB MIDI",),
    )

    try:
        case.answerable = False
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "BenchmarkCase debe ser inmutable."
        )


def test_evaluate_benchmark_case_accepts_correct_answer() -> None:
    case = BenchmarkCase(
        id="q001",
        query="¿Cómo se conecta el piano al ordenador?",
        answerable=True,
        expected_chunk_ids=("knowledge-001",),
        expected_answer_terms=("interfaz USB MIDI",),
    )

    result = RAGResult(
        answer="Se conecta mediante una interfaz USB MIDI.",
        sufficient=True,
        retrieved_chunk_ids=("knowledge-001",),
        selected_chunk_ids=("knowledge-001",),
    )

    assert evaluate_benchmark_case(case, result) is True


def test_evaluate_benchmark_case_rejects_wrong_answerability() -> None:
    case = BenchmarkCase(
        id="q001",
        query="Pregunta",
        answerable=True,
        expected_chunk_ids=("knowledge-001",),
        expected_answer_terms=("interfaz USB MIDI",),
    )

    result = RAGResult(
        answer="No tengo información suficiente.",
        sufficient=False,
        retrieved_chunk_ids=(),
        selected_chunk_ids=(),
    )

    assert evaluate_benchmark_case(case, result) is False


def test_evaluate_benchmark_case_rejects_missing_expected_chunk() -> None:
    case = BenchmarkCase(
        id="q001",
        query="Pregunta",
        answerable=True,
        expected_chunk_ids=("knowledge-001",),
        expected_answer_terms=("interfaz USB MIDI",),
    )

    result = RAGResult(
        answer="Se conecta mediante una interfaz USB MIDI.",
        sufficient=True,
        retrieved_chunk_ids=("knowledge-002",),
        selected_chunk_ids=("knowledge-002",),
    )

    assert evaluate_benchmark_case(case, result) is False


def test_evaluate_benchmark_case_rejects_missing_answer_term() -> None:
    case = BenchmarkCase(
        id="q001",
        query="Pregunta",
        answerable=True,
        expected_chunk_ids=("knowledge-001",),
        expected_answer_terms=("interfaz USB MIDI",),
    )

    result = RAGResult(
        answer="Se conecta mediante MIDI.",
        sufficient=True,
        retrieved_chunk_ids=("knowledge-001",),
        selected_chunk_ids=("knowledge-001",),
    )

    assert evaluate_benchmark_case(case, result) is False


def test_evaluate_benchmark_case_accepts_abstention() -> None:
    case = BenchmarkCase(
        id="q004",
        query="¿Qué sistema operativo utiliza MIDI Laboratory?",
        answerable=False,
        expected_chunk_ids=(),
        expected_answer_terms=(),
    )

    result = RAGResult(
        answer=(
            "No tengo información suficiente en el contexto "
            "disponible para responder esta pregunta."
        ),
        sufficient=False,
        retrieved_chunk_ids=("knowledge-000", "knowledge-001"),
        selected_chunk_ids=(),
    )

    assert evaluate_benchmark_case(case, result) is True