from pathlib import Path
import time

from rag_lab.benchmark import (
    evaluate_benchmark_case,
    load_benchmark,
)
from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.generation import LocalChatClient
from rag_lab.models import RAGResult
from rag_lab.prompting import build_rag_messages
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


BENCHMARK_PATH = Path("data/benchmark.json")
INDEX_PATH = Path("storage/index.json")


def main() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

    embedding_client = LocalEmbeddingClient()

    store = JsonVectorStore(INDEX_PATH)
    store.load()

    retriever = Retriever(store)

    chat_client = LocalChatClient()

    # Evitamos que el experimento mida el GET /api/v1/models.
    chat_client._model_capabilities = (
        chat_client.get_model_capabilities()
    )

    print("Benchmark sin Evidence Evaluator")
    print("================================")
    print()

    passed_count = 0

    for case in cases:
        start_time = time.perf_counter()

        query_embedding = tuple(
            embedding_client.embed(case.query)
        )

        results = retriever.search(
            query_embedding,
            top_k=3,
            score_threshold=None,
        )

        if not results:
            result = RAGResult(
                answer=(
                    "No tengo información suficiente en el "
                    "contexto disponible para responder esta pregunta."
                ),
                sufficient=False,
                retrieved_chunk_ids=(),
                selected_chunk_ids=(),
            )
        else:
            messages = build_rag_messages(
                case.query,
                results,
            )

            generation = chat_client.generate(
                messages,
                profile=__import__(
                    "rag_lab.inference",
                    fromlist=["RAG_ANSWER_PROFILE"],
                ).RAG_ANSWER_PROFILE,
            )

            result = RAGResult(
                answer=generation.content,
                sufficient=True,
                retrieved_chunk_ids=tuple(
                    result.chunk.id
                    for result in results
                ),
                selected_chunk_ids=tuple(
                    result.chunk.id
                    for result in results
                ),
            )

        elapsed = time.perf_counter() - start_time

        passed = evaluate_benchmark_case(
            case,
            result,
        )

        if passed:
            passed_count += 1

        print(
            f"{case.id} | "
            f"{'PASS' if passed else 'FAIL'} | "
            f"sufficient={result.sufficient} | "
            f"retrieved={len(result.retrieved_chunk_ids)} | "
            f"selected={len(result.selected_chunk_ids)} | "
            f"total={elapsed:.2f} s"
        )

        print(
            f"     Answer: {result.answer.strip()}"
        )
        print()

    accuracy = passed_count / len(cases)

    print("================================")
    print(f"Cases: {len(cases)}")
    print(f"Passed: {passed_count}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()