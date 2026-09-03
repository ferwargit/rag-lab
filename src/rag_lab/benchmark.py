from dataclasses import dataclass

import json
from pathlib import Path
from rag_lab.answer_validation import validate_answer_terms
from rag_lab.models import RAGResult
from rag_lab.metrics import ExecutionMetrics

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_lab.rag_pipeline import RAGPipeline

@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    query: str
    answerable: bool
    expected_chunk_ids: tuple[str, ...]
    expected_answer_terms: tuple[str, ...]


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    """Carga los casos del benchmark desde un archivo JSON."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [
        BenchmarkCase(
            id=item["id"],
            query=item["query"],
            answerable=item["answerable"],
            expected_chunk_ids=tuple(
                item["expected_chunk_ids"]
            ),
            expected_answer_terms=tuple(
                item["expected_answer_terms"]
            ),
        )
        for item in data
    ]


def evaluate_benchmark_case(
    case: BenchmarkCase,
    result: RAGResult,
) -> bool:
    """Evalúa un resultado RAG contra un caso del benchmark."""

    if result.sufficient is not case.answerable:
        return False

    if not case.answerable:
        return (
            result.selected_chunk_ids == ()
            and result.answer.strip() != ""
        )

    expected_chunk_ids = set(case.expected_chunk_ids)
    selected_chunk_ids = set(result.selected_chunk_ids)

    if not expected_chunk_ids.issubset(selected_chunk_ids):
        return False

    return validate_answer_terms(
        result.answer,
        case.expected_answer_terms,
    )


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    answer: str
    sufficient: bool
    retrieved_chunk_ids: tuple[str, ...]
    selected_chunk_ids: tuple[str, ...]


def build_benchmark_result(
    case: BenchmarkCase,
    result: RAGResult,
) -> BenchmarkResult:
    """Convierte un resultado RAG en un resultado de benchmark."""

    return BenchmarkResult(
        case_id=case.id,
        answer=result.answer,
        sufficient=result.sufficient,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        selected_chunk_ids=result.selected_chunk_ids,
    )


from collections.abc import Callable, Sequence


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    ask: Callable[[str], RAGResult],
) -> list[BenchmarkResult]:
    """Ejecuta los casos del benchmark usando una función de consulta."""

    return [
        build_benchmark_result(
            case,
            ask(case.query),
        )
        for case in cases
    ]


def run_and_evaluate_benchmark(
    cases: Sequence[BenchmarkCase],
    ask: Callable[[str], RAGResult],
) -> tuple[list[BenchmarkResult], list[bool]]:
    """Ejecuta el benchmark y evalúa cada resultado."""

    results = run_benchmark(
        cases,
        ask,
    )

    evaluations = [
        evaluate_benchmark_case(
            case,
            result,
        )
        for case, result in zip(cases, results, strict=True)
    ]

    return results, evaluations


@dataclass(frozen=True)
class BenchmarkExecution:
    """Resultado completo de una ejecución de benchmark."""

    result: BenchmarkResult
    evidence_metrics: ExecutionMetrics | None
    answer_metrics: ExecutionMetrics | None


def build_benchmark_execution(
    case: BenchmarkCase,
    result: RAGResult,
    *,
    evidence_metrics: ExecutionMetrics | None,
    answer_metrics: ExecutionMetrics | None,
) -> BenchmarkExecution:
    """Construye el registro completo de una ejecución."""

    benchmark_result = build_benchmark_result(
        case,
        result,
    )

    return BenchmarkExecution(
        result=benchmark_result,
        evidence_metrics=evidence_metrics,
        answer_metrics=answer_metrics,
    )


def run_benchmark_with_metrics(
    cases: Sequence[BenchmarkCase],
    pipeline: "RAGPipeline",
) -> list[BenchmarkExecution]:
    """Ejecuta el benchmark y conserva las métricas de cada componente."""

    executions: list[BenchmarkExecution] = []

    for case in cases:
        result = pipeline.ask(case.query)

        execution = build_benchmark_execution(
            case,
            result,
            evidence_metrics=(
                pipeline.evidence_evaluator.last_metrics
            ),
            answer_metrics=pipeline.last_metrics,
        )

        executions.append(execution)

    return executions


def summarize_benchmark_execution(
    execution: BenchmarkExecution,
    passed: bool,
) -> str:
    """Genera un resumen textual de una ejecución."""

    status = "PASS" if passed else "FAIL"

    return (
        f"{execution.result.case_id} | "
        f"{status} | "
        f"sufficient={execution.result.sufficient} | "
        f"retrieved={len(execution.result.retrieved_chunk_ids)} | "
        f"selected={len(execution.result.selected_chunk_ids)}"
    )


@dataclass(frozen=True)
class BenchmarkReport:
    """Resultado global de una ejecución de benchmark."""

    executions: tuple[BenchmarkExecution, ...]
    evaluations: tuple[bool, ...]

    @property
    def total_cases(self) -> int:
        return len(self.executions)

    @property
    def passed_cases(self) -> int:
        return sum(self.evaluations)

    @property
    def accuracy(self) -> float:
        if not self.executions:
            return 0.0

        return self.passed_cases / self.total_cases


def build_benchmark_report(
    executions: Sequence[BenchmarkExecution],
    evaluations: Sequence[bool],
) -> BenchmarkReport:
    """Construye un reporte global a partir de ejecuciones y evaluaciones."""

    if len(executions) != len(evaluations):
        raise ValueError(
            "La cantidad de ejecuciones y evaluaciones debe coincidir."
        )

    return BenchmarkReport(
        executions=tuple(executions),
        evaluations=tuple(evaluations),
    )


def run_full_benchmark(
    cases: Sequence[BenchmarkCase],
    ask: Callable[[str], RAGResult],
) -> BenchmarkReport:
    """Ejecuta, evalúa y resume un benchmark completo."""

    results, evaluations = run_and_evaluate_benchmark(
        cases,
        ask,
    )

    executions = tuple(
        build_benchmark_execution(
            case,
            result,
            evidence_metrics=None,
            answer_metrics=None,
        )
        for case, result in zip(
            cases,
            results,
            strict=True,
        )
    )

    return build_benchmark_report(
        executions,
        evaluations,
    )