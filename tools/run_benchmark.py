from pathlib import Path
import itertools
import sys
import threading
import time

from rag_lab.benchmark import (
    format_benchmark_report,
    load_benchmark,
    run_full_benchmark_with_metrics,
)
from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.generation import LocalChatClient
from rag_lab.rag_pipeline import RAGPipeline
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


BENCHMARK_PATH = Path("data/benchmark.json")
INDEX_PATH = Path("storage/index.json")


def show_spinner(stop_event: threading.Event) -> None:
    """Muestra un indicador simple mientras se procesa el benchmark."""
    for symbol in itertools.cycle(["|", "/", "-", "\\"]):
        if stop_event.is_set():
            break

        sys.stdout.write(f"\rProcesando benchmark... {symbol}")
        sys.stdout.flush()
        time.sleep(0.2)

    sys.stdout.write("\rProcesando benchmark... listo.\n")
    sys.stdout.flush()


def main() -> None:
    cases = load_benchmark(BENCHMARK_PATH)

    embedding_client = LocalEmbeddingClient()

    store = JsonVectorStore(INDEX_PATH)
    store.load()

    retriever = Retriever(store)

    chat_client = LocalChatClient()

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        retriever=retriever,
        chat_client=chat_client,
        top_k=3,
    )

    stop_event = threading.Event()

    spinner_thread = threading.Thread(
        target=show_spinner,
        args=(stop_event,),
        daemon=True,
    )

    spinner_thread.start()

    try:
        report = run_full_benchmark_with_metrics(cases, pipeline)
    finally:
        stop_event.set()
        spinner_thread.join()

    print(format_benchmark_report(report))


if __name__ == "__main__":
    main()