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
    execution_time_seconds: float | None
    stage_execution_times_seconds: tuple[tuple[str, float], ...] = ()

    @property
    def case_id(self) -> str:
        """Devuelve el ID del caso ejecutado."""
        return self.result.case_id


def build_benchmark_execution(
    case: BenchmarkCase,
    result: RAGResult,
    *,
    evidence_metrics: ExecutionMetrics | None,
    answer_metrics: ExecutionMetrics | None,
    execution_time_seconds: float | None,
    stage_execution_times_seconds: tuple[tuple[str, float], ...] = (),
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
        execution_time_seconds=execution_time_seconds,
        stage_execution_times_seconds=stage_execution_times_seconds,
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
            execution_time_seconds=(
                pipeline.last_execution_time_seconds
            ),
            stage_execution_times_seconds=tuple(
                pipeline.last_stage_execution_times_seconds.items()
            ),
        )

        executions.append(execution)

    return executions


def summarize_benchmark_execution(
    execution: BenchmarkExecution,
    passed: bool,
) -> str:
    """Genera un resumen textual de una ejecución."""

    status = "PASS" if passed else "FAIL"

    total_time = (
        f"{execution.execution_time_seconds:.2f} s"
        if execution.execution_time_seconds is not None
        else "N/A"
    )

    lines = [
        (
            f"{execution.result.case_id} | "
            f"{status} | "
            f"sufficient={execution.result.sufficient} | "
            f"retrieved={len(execution.result.retrieved_chunk_ids)} | "
            f"selected={len(execution.result.selected_chunk_ids)} | "
            f"total={total_time}"
        ),
    ]

    if execution.evidence_metrics is not None:
        metrics = execution.evidence_metrics

        speed = (
            f"{metrics.tokens_per_second:.2f} tok/s"
            if metrics.tokens_per_second is not None
            else "N/A"
        )

        ttft = (
            f"{metrics.time_to_first_token_seconds:.2f} s"
            if metrics.time_to_first_token_seconds is not None
            else "N/A"
        )

        lines.append(
            (
                f"     Evidence | "
                f"input={metrics.input_tokens} | "
                f"output={metrics.total_output_tokens} | "
                f"reasoning={metrics.reasoning_output_tokens} | "
                f"speed={speed} | "
                f"TTFT={ttft} | "
                f"generation={metrics.generation_time_seconds:.2f} s"
            )
        )

    if execution.answer_metrics is not None:
        metrics = execution.answer_metrics

        speed = (
            f"{metrics.tokens_per_second:.2f} tok/s"
            if metrics.tokens_per_second is not None
            else "N/A"
        )

        ttft = (
            f"{metrics.time_to_first_token_seconds:.2f} s"
            if metrics.time_to_first_token_seconds is not None
            else "N/A"
        )

        lines.append(
            (
                f"     Answer   | "
                f"input={metrics.input_tokens} | "
                f"output={metrics.total_output_tokens} | "
                f"reasoning={metrics.reasoning_output_tokens} | "
                f"speed={speed} | "
                f"TTFT={ttft} | "
                f"generation={metrics.generation_time_seconds:.2f} s"
            )
        )

    for stage_name, stage_time in execution.stage_execution_times_seconds:
        display_name = {
            "embedding": "Embedding",
            "retrieval": "Retrieval",
            "evidence": "Evidence",
            "answer_generation": "Answer Generation",
        }.get(stage_name, stage_name)

        lines.append(
            f"     {display_name:<18} | {stage_time:.2f} s"
        )

    return "\n".join(lines)


@dataclass(frozen=True)
class MetricsSummary:
    """Resumen agregado de métricas de ejecución."""

    sample_count: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_reasoning_output_tokens: float
    avg_tokens_per_second: float | None
    avg_time_to_first_token_seconds: float | None


def build_metrics_summary(
    metrics: Sequence[ExecutionMetrics | None],
) -> MetricsSummary:
    """Calcula promedios a partir de métricas de ejecución."""

    valid_metrics = [
        metric
        for metric in metrics
        if metric is not None
    ]

    if not valid_metrics:
        return MetricsSummary(
            sample_count=0,
            avg_input_tokens=0.0,
            avg_output_tokens=0.0,
            avg_reasoning_output_tokens=0.0,
            avg_tokens_per_second=None,
            avg_time_to_first_token_seconds=None,
        )

    return MetricsSummary(
        sample_count=len(valid_metrics),
        avg_input_tokens=(
            sum(metric.input_tokens for metric in valid_metrics)
            / len(valid_metrics)
        ),
        avg_output_tokens=(
            sum(metric.total_output_tokens for metric in valid_metrics)
            / len(valid_metrics)
        ),
        avg_reasoning_output_tokens=(
            sum(
                metric.reasoning_output_tokens
                for metric in valid_metrics
            )
            / len(valid_metrics)
        ),
        avg_tokens_per_second=_average_optional_metric(
            metric.tokens_per_second
            for metric in valid_metrics
        ),
        avg_time_to_first_token_seconds=_average_optional_metric(
            metric.time_to_first_token_seconds
            for metric in valid_metrics
        ),
    )


def _average_optional_metric(
    values: Sequence[float | None],
) -> float | None:
    """Calcula el promedio ignorando valores None."""

    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if not valid_values:
        return None

    return round(
        sum(valid_values) / len(valid_values),
        6,
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

    @property
    def evidence_metrics_summary(self) -> MetricsSummary:
        return build_metrics_summary(
            execution.evidence_metrics
            for execution in self.executions
        )

    @property
    def answer_metrics_summary(self) -> MetricsSummary:
        return build_metrics_summary(
            execution.answer_metrics
            for execution in self.executions
        )


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
            execution_time_seconds=None,
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


def run_full_benchmark_with_metrics(
    cases: Sequence[BenchmarkCase],
    pipeline: "RAGPipeline",
) -> BenchmarkReport:
    """Ejecuta, evalúa y reporta un benchmark con métricas."""

    executions = run_benchmark_with_metrics(
        cases,
        pipeline,
    )

    evaluations = tuple(
        evaluate_benchmark_case(
            case,
            execution.result,
        )
        for case, execution in zip(
            cases,
            executions,
            strict=True,
        )
    )

    return build_benchmark_report(
        executions,
        evaluations,
    )


def format_metrics_summary(
    title: str,
    summary: MetricsSummary,
) -> list[str]:
    """Formatea un resumen de métricas para el reporte."""

    speed = (
        f"{summary.avg_tokens_per_second:.2f} tok/s"
        if summary.avg_tokens_per_second is not None
        else "N/A"
    )

    ttft = (
        f"{summary.avg_time_to_first_token_seconds:.2f} s"
        if summary.avg_time_to_first_token_seconds is not None
        else "N/A"
    )

    return [
        title,
        "-" * len(title),
        f"Samples: {summary.sample_count}",
        f"Avg input tokens: {summary.avg_input_tokens:.2f}",
        f"Avg output tokens: {summary.avg_output_tokens:.2f}",
        (
            "Avg reasoning tokens: "
            f"{summary.avg_reasoning_output_tokens:.2f}"
        ),
        f"Avg speed: {speed}",
        f"Avg TTFT: {ttft}",
    ]


def format_benchmark_report(
    report: BenchmarkReport,
) -> str:
    """Formatea un reporte de benchmark como texto legible."""

    lines = [
        "Benchmark Report",
        "================",
        f"Cases: {report.total_cases}",
        f"Passed: {report.passed_cases}",
        f"Accuracy: {report.accuracy:.2%}",
        "",
    ]

    lines.extend(
        summarize_benchmark_execution(
            execution,
            passed,
        )
        for execution, passed in zip(
            report.executions,
            report.evaluations,
            strict=True,
        )
    )

    lines.append("")

    lines.extend(
        format_metrics_summary(
            "Evidence Evaluator",
            report.evidence_metrics_summary,
        )
    )

    lines.append("")

    lines.extend(
        format_metrics_summary(
            "Answer Generation",
            report.answer_metrics_summary,
        )
    )

    return "\n".join(lines)