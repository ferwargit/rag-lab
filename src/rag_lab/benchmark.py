from dataclasses import dataclass

import json
from pathlib import Path
from rag_lab.answer_validation import validate_answer_terms
from rag_lab.models import RAGResult

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