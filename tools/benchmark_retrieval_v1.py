import json
from pathlib import Path

from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


def load_benchmark(path: Path) -> list[dict]:
    """Carga y valida el dataset de evaluación."""

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el benchmark: {path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError(
            "El benchmark debe contener una lista."
        )

    return data


def main() -> None:
    benchmark_path = Path("data/benchmark.json")
    index_path = Path("storage/index.json")

    benchmark = load_benchmark(benchmark_path)

    store = JsonVectorStore(index_path)
    store.load()

    embedding_client = LocalEmbeddingClient()
    retriever = Retriever(store)

    top_k = 3

    total_with_expected = 0
    successful_retrievals = 0

    print("RETRIEVAL BENCHMARK")
    print("=" * 70)
    print()

    for case in benchmark:
        case_id = case["id"]
        query = case["query"]
        expected_ids = set(
            case.get("expected_chunk_ids", [])
        )

        query_embedding = embedding_client.embed(query)

        results = retriever.search(
            tuple(query_embedding),
            top_k=top_k,
        )

        retrieved_ids = [
            result.chunk.id
            for result in results
        ]

        if expected_ids:
            total_with_expected += 1

            hit = bool(
                expected_ids.intersection(retrieved_ids)
            )

            if hit:
                successful_retrievals += 1
        else:
            hit = None

        print(f"[{case_id}]")
        print(f"Query: {query}")
        print(f"Expected: {sorted(expected_ids)}")
        print(f"Retrieved: {retrieved_ids}")

        for position, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"  {position}. "
                f"{result.chunk.id} "
                f"score={result.score:.6f}"
            )

        if expected_ids:
            print(
                f"Result: "
                f"{'PASS' if hit else 'FAIL'}"
            )
        else:
            print(
                "Result: NO_EXPECTED_CHUNK "
                "(evaluación manual de evidencia)"
            )

        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if total_with_expected:
        recall_at_k = (
            successful_retrievals /
            total_with_expected
        )

        print(
            f"Recall@{top_k}: "
            f"{successful_retrievals}/"
            f"{total_with_expected} "
            f"({recall_at_k:.1%})"
        )

    print(
        f"Casos evaluados: {len(benchmark)}"
    )


if __name__ == "__main__":
    main()