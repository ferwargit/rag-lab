from pathlib import Path

from rag_lab.benchmark import (
    format_benchmark_report,
    load_benchmark,
    run_full_benchmark_with_metrics,
)
from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.evidence_evaluator import EvidenceEvaluator
from rag_lab.generation import LocalChatClient
from rag_lab.rag_pipeline import RAGPipeline
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

    evidence_evaluator = EvidenceEvaluator(
        LocalChatClient()
    )

    chat_client = LocalChatClient()

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        evidence_evaluator=evidence_evaluator,
        chat_client=chat_client,
        top_k=3,
    )

    report = run_full_benchmark_with_metrics(
        cases,
        pipeline,
    )

    print(
        format_benchmark_report(report)
    )


if __name__ == "__main__":
    main()