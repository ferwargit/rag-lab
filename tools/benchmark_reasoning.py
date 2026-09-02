from pathlib import Path
import time

from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.generation import LocalChatClient
from rag_lab.inference import (
    RAG_ANSWER_OFF_PROFILE,
    RAG_ANSWER_ON_PROFILE,
)
from rag_lab.prompting import build_rag_messages
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


INDEX_PATH = Path("storage/index.json")


def run_case(
    label: str,
    client: LocalChatClient,
    profile,
    query: str,
    results,
) -> None:
    messages = build_rag_messages(
        query,
        results,
    )

    started = time.perf_counter()

    result = client.generate(
        messages,
        profile=profile,
    )

    elapsed = time.perf_counter() - started

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)
    print(f"Pregunta: {query}")
    print()
    print("Respuesta:")
    print(result.content)
    print()
    print(f"Tiempo total: {elapsed:.3f} s")
    print(f"Input tokens: {result.input_tokens}")
    print(f"Output tokens: {result.total_output_tokens}")
    print(
        f"Reasoning tokens: "
        f"{result.reasoning_output_tokens}"
    )
    print(f"Tokens/s: {result.tokens_per_second}")
    print(
        "TTFT: "
        f"{result.time_to_first_token_seconds}"
    )


def main() -> None:
    embedding_client = LocalEmbeddingClient()

    store = JsonVectorStore(INDEX_PATH)
    store.load()

    retriever = Retriever(store)

    client = LocalChatClient(
        timeout=240.0,
    )

    query = "¿Cómo se conecta el piano al ordenador?"

    query_embedding = embedding_client.embed(query)

    results = retriever.search(
        tuple(query_embedding),
        top_k=3,
        score_threshold=None,
    )


    print("CANDIDATOS RECUPERADOS")
    print("=" * 70)

    for result in results:
        print(
            f"{result.chunk.id}: "
            f"{result.score:.6f}"
        )

    run_case(
        "RAG ANSWER — REASONING OFF",
        client,
        RAG_ANSWER_OFF_PROFILE,
        query,
        results,
    )

    run_case(
        "RAG ANSWER — REASONING ON",
        client,
        RAG_ANSWER_ON_PROFILE,
        query,
        results,
    )


if __name__ == "__main__":
    main()