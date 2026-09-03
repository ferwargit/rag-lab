from pathlib import Path

from rag_lab.benchmark import (
    BenchmarkCase,
    BenchmarkExecution,
    BenchmarkResult,
    build_benchmark_execution,
    build_benchmark_result,
    evaluate_benchmark_case,
    load_benchmark,
    run_and_evaluate_benchmark,
    run_benchmark,
)
from rag_lab.metrics import ExecutionMetrics
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


def test_benchmark_result_stores_rag_result_data() -> None:
    result = BenchmarkResult(
        case_id="q001",
        answer="Se conecta mediante una interfaz USB MIDI.",
        sufficient=True,
        retrieved_chunk_ids=(
            "knowledge-001",
            "knowledge-000",
        ),
        selected_chunk_ids=(
            "knowledge-001",
        ),
    )

    assert result.case_id == "q001"
    assert result.answer == (
        "Se conecta mediante una interfaz USB MIDI."
    )
    assert result.sufficient is True
    assert result.retrieved_chunk_ids == (
        "knowledge-001",
        "knowledge-000",
    )
    assert result.selected_chunk_ids == (
        "knowledge-001",
    )


def test_benchmark_result_is_immutable() -> None:
    result = BenchmarkResult(
        case_id="q001",
        answer="Respuesta",
        sufficient=True,
        retrieved_chunk_ids=("knowledge-001",),
        selected_chunk_ids=("knowledge-001",),
    )

    try:
        result.answer = "Otra respuesta"
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "BenchmarkResult debe ser inmutable."
        )


def test_build_benchmark_result_copies_rag_result_data() -> None:
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
        retrieved_chunk_ids=(
            "knowledge-001",
            "knowledge-000",
        ),
        selected_chunk_ids=(
            "knowledge-001",
        ),
    )

    benchmark_result = build_benchmark_result(
        case,
        result,
    )

    assert benchmark_result == BenchmarkResult(
        case_id="q001",
        answer="Se conecta mediante una interfaz USB MIDI.",
        sufficient=True,
        retrieved_chunk_ids=(
            "knowledge-001",
            "knowledge-000",
        ),
        selected_chunk_ids=(
            "knowledge-001",
        ),
    )


def test_build_benchmark_result_uses_case_id() -> None:
    case = BenchmarkCase(
        id="q004",
        query="Pregunta",
        answerable=False,
        expected_chunk_ids=(),
        expected_answer_terms=(),
    )

    result = RAGResult(
        answer="No tengo información suficiente.",
        sufficient=False,
        retrieved_chunk_ids=("knowledge-000",),
        selected_chunk_ids=(),
    )

    benchmark_result = build_benchmark_result(
        case,
        result,
    )

    assert benchmark_result.case_id == "q004"


def test_run_benchmark_executes_all_cases() -> None:
    cases = [
        BenchmarkCase(
            id="q001",
            query="Pregunta 1",
            answerable=True,
            expected_chunk_ids=("knowledge-001",),
            expected_answer_terms=("USB MIDI",),
        ),
        BenchmarkCase(
            id="q002",
            query="Pregunta 2",
            answerable=True,
            expected_chunk_ids=("knowledge-002",),
            expected_answer_terms=("renderer",),
        ),
    ]

    received_queries: list[str] = []

    def fake_ask(query: str) -> RAGResult:
        received_queries.append(query)

        return RAGResult(
            answer="Respuesta de prueba.",
            sufficient=True,
            retrieved_chunk_ids=("knowledge-001",),
            selected_chunk_ids=("knowledge-001",),
        )

    results = run_benchmark(
        cases,
        fake_ask,
    )

    assert received_queries == [
        "Pregunta 1",
        "Pregunta 2",
    ]

    assert len(results) == 2
    assert results[0].case_id == "q001"
    assert results[1].case_id == "q002"


def test_run_benchmark_preserves_case_order() -> None:
    cases = [
        BenchmarkCase(
            id="q003",
            query="Pregunta 3",
            answerable=True,
            expected_chunk_ids=("knowledge-003",),
            expected_answer_terms=("objetivo",),
        ),
        BenchmarkCase(
            id="q001",
            query="Pregunta 1",
            answerable=True,
            expected_chunk_ids=("knowledge-001",),
            expected_answer_terms=("USB MIDI",),
        ),
    ]

    def fake_ask(query: str) -> RAGResult:
        return RAGResult(
            answer=f"Respuesta para {query}",
            sufficient=True,
            retrieved_chunk_ids=(),
            selected_chunk_ids=(),
        )

    results = run_benchmark(
        cases,
        fake_ask,
    )

    assert [result.case_id for result in results] == [
        "q003",
        "q001",
    ]


def test_run_and_evaluate_benchmark_returns_results_and_evaluations() -> None:
    cases = [
        BenchmarkCase(
            id="q001",
            query="Pregunta 1",
            answerable=True,
            expected_chunk_ids=("knowledge-001",),
            expected_answer_terms=("USB MIDI",),
        ),
        BenchmarkCase(
            id="q004",
            query="Pregunta 4",
            answerable=False,
            expected_chunk_ids=(),
            expected_answer_terms=(),
        ),
    ]

    def fake_ask(query: str) -> RAGResult:
        if query == "Pregunta 1":
            return RAGResult(
                answer="Se conecta mediante USB MIDI.",
                sufficient=True,
                retrieved_chunk_ids=("knowledge-001",),
                selected_chunk_ids=("knowledge-001",),
            )

        return RAGResult(
            answer="No tengo información suficiente.",
            sufficient=False,
            retrieved_chunk_ids=("knowledge-000",),
            selected_chunk_ids=(),
        )

    results, evaluations = run_and_evaluate_benchmark(
        cases,
        fake_ask,
    )

    assert len(results) == 2
    assert evaluations == [True, True]

    assert [result.case_id for result in results] == [
        "q001",
        "q004",
    ]


def test_run_and_evaluate_benchmark_preserves_failed_evaluation() -> None:
    cases = [
        BenchmarkCase(
            id="q001",
            query="Pregunta",
            answerable=True,
            expected_chunk_ids=("knowledge-001",),
            expected_answer_terms=("USB MIDI",),
        )
    ]

    def fake_ask(query: str) -> RAGResult:
        return RAGResult(
            answer="Respuesta incorrecta.",
            sufficient=True,
            retrieved_chunk_ids=("knowledge-002",),
            selected_chunk_ids=("knowledge-002",),
        )

    results, evaluations = run_and_evaluate_benchmark(
        cases,
        fake_ask,
    )

    assert len(results) == 1
    assert results[0].case_id == "q001"
    assert evaluations == [False]


def test_benchmark_execution_stores_component_metrics() -> None:
    result = BenchmarkResult(
        case_id="q001",
        answer="Respuesta",
        sufficient=True,
        retrieved_chunk_ids=("knowledge-001",),
        selected_chunk_ids=("knowledge-001",),
    )

    evidence_metrics = ExecutionMetrics(
        input_tokens=409,
        total_output_tokens=26,
        reasoning_output_tokens=0,
        tokens_per_second=41.5,
        time_to_first_token_seconds=0.28,
    )

    answer_metrics = ExecutionMetrics(
        input_tokens=179,
        total_output_tokens=28,
        reasoning_output_tokens=0,
        tokens_per_second=42.1,
        time_to_first_token_seconds=0.32,
    )

    execution = BenchmarkExecution(
        case_id="q001",
        result=result,
        evidence_metrics=evidence_metrics,
        answer_metrics=answer_metrics,
    )

    assert execution.case_id == "q001"
    assert execution.result == result
    assert execution.evidence_metrics == evidence_metrics
    assert execution.answer_metrics == answer_metrics


def test_benchmark_execution_allows_missing_answer_metrics() -> None:
    result = BenchmarkResult(
        case_id="q004",
        answer="No tengo información suficiente.",
        sufficient=False,
        retrieved_chunk_ids=("knowledge-000",),
        selected_chunk_ids=(),
    )

    evidence_metrics = ExecutionMetrics(
        input_tokens=415,
        total_output_tokens=20,
        reasoning_output_tokens=0,
        tokens_per_second=41.2,
        time_to_first_token_seconds=0.27,
    )

    execution = BenchmarkExecution(
        case_id="q004",
        result=result,
        evidence_metrics=evidence_metrics,
        answer_metrics=None,
    )

    assert execution.evidence_metrics == evidence_metrics
    assert execution.answer_metrics is None


def test_benchmark_execution_is_immutable() -> None:
    execution = BenchmarkExecution(
        case_id="q001",
        result=BenchmarkResult(
            case_id="q001",
            answer="Respuesta",
            sufficient=True,
            retrieved_chunk_ids=("knowledge-001",),
            selected_chunk_ids=("knowledge-001",),
        ),
        evidence_metrics=None,
        answer_metrics=None,
    )

    try:
        execution.case_id = "q002"
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "BenchmarkExecution debe ser inmutable."
        )


def test_build_benchmark_execution_combines_result_and_metrics() -> None:
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
        retrieved_chunk_ids=(
            "knowledge-001",
            "knowledge-000",
        ),
        selected_chunk_ids=(
            "knowledge-001",
        ),
    )

    evidence_metrics = ExecutionMetrics(
        input_tokens=409,
        total_output_tokens=26,
        reasoning_output_tokens=0,
        tokens_per_second=41.5,
        time_to_first_token_seconds=0.28,
    )

    answer_metrics = ExecutionMetrics(
        input_tokens=179,
        total_output_tokens=28,
        reasoning_output_tokens=0,
        tokens_per_second=42.1,
        time_to_first_token_seconds=0.32,
    )

    execution = build_benchmark_execution(
        case,
        result,
        evidence_metrics=evidence_metrics,
        answer_metrics=answer_metrics,
    )

    assert execution.case_id == "q001"

    assert execution.result.answer == (
        "Se conecta mediante una interfaz USB MIDI."
    )

    assert execution.result.sufficient is True

    assert execution.evidence_metrics == evidence_metrics
    assert execution.answer_metrics == answer_metrics


def test_build_benchmark_execution_allows_missing_answer_metrics() -> None:
    case = BenchmarkCase(
        id="q004",
        query="¿Qué sistema operativo utiliza MIDI Laboratory?",
        answerable=False,
        expected_chunk_ids=(),
        expected_answer_terms=(),
    )

    result = RAGResult(
        answer="No tengo información suficiente.",
        sufficient=False,
        retrieved_chunk_ids=("knowledge-000",),
        selected_chunk_ids=(),
    )

    evidence_metrics = ExecutionMetrics(
        input_tokens=415,
        total_output_tokens=20,
        reasoning_output_tokens=0,
        tokens_per_second=41.2,
        time_to_first_token_seconds=0.27,
    )

    execution = build_benchmark_execution(
        case,
        result,
        evidence_metrics=evidence_metrics,
        answer_metrics=None,
    )

    assert execution.case_id == "q004"
    assert execution.result.sufficient is False
    assert execution.evidence_metrics == evidence_metrics
    assert execution.answer_metrics is None