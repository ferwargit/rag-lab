from pathlib import Path
import json
import time

from rag_lab.benchmark import (
    evaluate_benchmark_case,
    load_benchmark,
)
from rag_lab.context import build_context
from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.generation import LocalChatClient
from rag_lab.inference import RAG_ANSWER_PROFILE
from rag_lab.models import RAGResult
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


BENCHMARK_PATH = Path("data/benchmark.json")
INDEX_PATH = Path("storage/index.json")


SYSTEM_INSTRUCTION = """Eres un asistente RAG que debe decidir si el
contexto recuperado contiene información suficiente para responder la
pregunta y, si la contiene, responderla.

Debes devolver exclusivamente un objeto JSON válido.

Formato obligatorio:

{
  "sufficient": true,
  "answer": "respuesta"
}

Reglas:

- "sufficient" debe ser un booleano.
- "answer" debe ser un string.
- Utiliza exclusivamente la información del contexto.
- No inventes información.
- Si el contexto NO contiene información suficiente para responder,
  usa "sufficient": false.
- Cuando "sufficient" sea false, "answer" debe explicar brevemente
  que no hay información suficiente.
- Cuando "sufficient" sea true, responde la pregunta de forma precisa
  y concisa.
- No incluyas markdown.
- No incluyas texto fuera del JSON.
- Responde en español.
"""


def main() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

    embedding_client = LocalEmbeddingClient()

    store = JsonVectorStore(INDEX_PATH)
    store.load()

    retriever = Retriever(store)

    chat_client = LocalChatClient()

    # Evitamos incluir el coste del GET /api/v1/models
    # dentro de la medición de cada caso.
    chat_client._model_capabilities = (
        chat_client.get_model_capabilities()
    )

    print("Benchmark single-call")
    print("=====================")
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

        context = build_context(results)

        user_message = (
            f"Contexto recuperado:\n\n"
            f"{context}\n\n"
            f"Pregunta:\n{case.query}\n\n"
            f"Devuelve únicamente el JSON solicitado."
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        generation = chat_client.generate(
            messages,
            profile=RAG_ANSWER_PROFILE,
        )

        try:
            parsed = json.loads(generation.content)
        except json.JSONDecodeError:
            result = RAGResult(
                answer=generation.content,
                sufficient=False,
                retrieved_chunk_ids=tuple(
                    item.chunk.id for item in results
                ),
                selected_chunk_ids=(),
            )
        else:
            sufficient = parsed.get("sufficient")
            answer = parsed.get("answer")

            if (
                not isinstance(sufficient, bool)
                or not isinstance(answer, str)
            ):
                result = RAGResult(
                    answer=generation.content,
                    sufficient=False,
                    retrieved_chunk_ids=tuple(
                        item.chunk.id for item in results
                    ),
                    selected_chunk_ids=(),
                )
            elif sufficient:
                result = RAGResult(
                    answer=answer,
                    sufficient=True,
                    retrieved_chunk_ids=tuple(
                        item.chunk.id for item in results
                    ),
                    selected_chunk_ids=tuple(
                        item.chunk.id for item in results
                    ),
                )
            else:
                result = RAGResult(
                    answer=answer,
                    sufficient=False,
                    retrieved_chunk_ids=tuple(
                        item.chunk.id for item in results
                    ),
                    selected_chunk_ids=(),
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

        print(
            f"     Raw: {generation.content.strip()}"
        )

        print()

    accuracy = passed_count / len(cases)

    print("=====================")
    print(f"Cases: {len(cases)}")
    print(f"Passed: {passed_count}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()