from rag_lab.metrics import (
    ComponentExecutionMetrics,
    ExecutionMetrics,
    metrics_from_generation,
)
from rag_lab.generation import GenerationResult


def test_execution_metrics_stores_generation_metrics() -> None:
    metrics = ExecutionMetrics(
        input_tokens=179,
        total_output_tokens=28,
        reasoning_output_tokens=0,
        tokens_per_second=42.08,
        time_to_first_token_seconds=0.322,
    )

    assert metrics.input_tokens == 179
    assert metrics.total_output_tokens == 28
    assert metrics.reasoning_output_tokens == 0
    assert metrics.tokens_per_second == 42.08
    assert metrics.time_to_first_token_seconds == 0.322


def test_execution_metrics_accepts_missing_optional_timing() -> None:
    metrics = ExecutionMetrics(
        input_tokens=100,
        total_output_tokens=20,
        reasoning_output_tokens=0,
        tokens_per_second=None,
        time_to_first_token_seconds=None,
    )

    assert metrics.tokens_per_second is None
    assert metrics.time_to_first_token_seconds is None


def test_execution_metrics_is_immutable() -> None:
    metrics = ExecutionMetrics(
        input_tokens=100,
        total_output_tokens=20,
        reasoning_output_tokens=0,
        tokens_per_second=40.0,
        time_to_first_token_seconds=0.2,
    )

    try:
        metrics.input_tokens = 200
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "ExecutionMetrics debe ser inmutable."
        )


def test_metrics_from_generation_extracts_metrics() -> None:
    generation = GenerationResult(
        content="Respuesta",
        reasoning=None,
        input_tokens=179,
        total_output_tokens=28,
        reasoning_output_tokens=0,
        tokens_per_second=42.08,
        time_to_first_token_seconds=0.322,
    )

    metrics = metrics_from_generation(generation)

    assert metrics.input_tokens == 179
    assert metrics.total_output_tokens == 28
    assert metrics.reasoning_output_tokens == 0
    assert metrics.tokens_per_second == 42.08
    assert metrics.time_to_first_token_seconds == 0.322


def test_metrics_from_generation_preserves_missing_timing() -> None:
    generation = GenerationResult(
        content="Respuesta",
        reasoning=None,
        input_tokens=100,
        total_output_tokens=20,
        reasoning_output_tokens=0,
        tokens_per_second=None,
        time_to_first_token_seconds=None,
    )

    metrics = metrics_from_generation(generation)

    assert metrics.tokens_per_second is None
    assert metrics.time_to_first_token_seconds is None


def test_component_execution_metrics_stores_component_name() -> None:
    execution = ComponentExecutionMetrics(
        component="evidence-evaluator",
        metrics=ExecutionMetrics(
            input_tokens=409,
            total_output_tokens=26,
            reasoning_output_tokens=0,
            tokens_per_second=41.52,
            time_to_first_token_seconds=0.28,
        ),
    )

    assert execution.component == "evidence-evaluator"
    assert execution.metrics.input_tokens == 409


def test_component_execution_metrics_is_immutable() -> None:
    execution = ComponentExecutionMetrics(
        component="rag-answer",
        metrics=ExecutionMetrics(
            input_tokens=179,
            total_output_tokens=28,
            reasoning_output_tokens=0,
            tokens_per_second=42.08,
            time_to_first_token_seconds=0.322,
        ),
    )

    try:
        execution.component = "otro-componente"
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "ComponentExecutionMetrics debe ser inmutable."
        )