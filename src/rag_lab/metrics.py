from dataclasses import dataclass

from rag_lab.generation import GenerationResult


@dataclass(frozen=True)
class ExecutionMetrics:
    """Métricas de ejecución de una generación."""

    input_tokens: int
    total_output_tokens: int
    reasoning_output_tokens: int
    tokens_per_second: float | None
    time_to_first_token_seconds: float | None


def metrics_from_generation(
    generation: GenerationResult,
) -> ExecutionMetrics:
    """Extrae las métricas de ejecución de una generación."""

    return ExecutionMetrics(
        input_tokens=generation.input_tokens,
        total_output_tokens=generation.total_output_tokens,
        reasoning_output_tokens=generation.reasoning_output_tokens,
        tokens_per_second=generation.tokens_per_second,
        time_to_first_token_seconds=(
            generation.time_to_first_token_seconds
        ),
    )


@dataclass(frozen=True)
class ComponentExecutionMetrics:
    """Métricas de ejecución de un componente del pipeline."""

    component: str
    metrics: ExecutionMetrics