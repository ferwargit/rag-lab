import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_lab.embeddings import LocalEmbeddingClient
from rag_lab.generation import LocalChatClient
from rag_lab.inference import (
    INFERENCE_LAYER_VERSION,
    RAG_ANSWER_PROFILE,
    DEEP_ANSWER_PROFILE,
)
from rag_lab.prompting import build_rag_messages
from rag_lab.retrieval import Retriever
from rag_lab.vector_store import JsonVectorStore


BENCHMARK_PATH = Path("data/benchmark.json")
INDEX_PATH = Path("storage/index.json")
OUTPUT_DIR = Path("benchmarks")

MODELS = {
    "off": RAG_ANSWER_PROFILE,
    "on": DEEP_ANSWER_PROFILE,
}


def load_answerable_cases() -> list[dict]:
    raw = json.loads(
        BENCHMARK_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(raw, list):
        raise ValueError(
            "El benchmark debe contener una lista."
        )

    cases = [
        case
        for case in raw
        if isinstance(case, dict)
        and case.get("answerable") is True
    ]

    if not cases:
        raise ValueError(
            "No se encontraron casos answerable."
        )

    return cases


def build_context_for_case(
    query: str,
    embedding_client: LocalEmbeddingClient,
    retriever: Retriever,
):
    query_embedding = embedding_client.embed(query)

    return retriever.search(
        tuple(query_embedding),
        top_k=3,
        score_threshold=None,
    )


def run_generation(
    client: LocalChatClient,
    profile,
    query: str,
    results,
) -> dict:
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

    return {
        "response": result.content,
        "elapsed_seconds": elapsed,
        "input_tokens": result.input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "reasoning_output_tokens": result.reasoning_output_tokens,
        "tokens_per_second": result.tokens_per_second,
        "time_to_first_token_seconds": (
            result.time_to_first_token_seconds
        ),
        "reasoning_used": (
            result.reasoning_output_tokens > 0
        ),
    }


def build_markdown_report(
    timestamp: str,
    model_name: str,
    cases: list[dict],
    results: list[dict],
) -> str:
    lines: list[str] = []

    lines.append("# Benchmark OFF vs ON — RAG Answer")
    lines.append("")
    lines.append(f"- Fecha: {timestamp}")
    lines.append(f"- Modelo: `{model_name}`")
    lines.append(
        f"- Inference Layer: `{INFERENCE_LAYER_VERSION}`"
    )
    lines.append("- Objetivo: comparar reasoning OFF vs ON en generación RAG.")
    lines.append("- Retrieval: idéntico para ambas variantes.")
    lines.append("- Temperatura: idéntica dentro de los perfiles.")
    lines.append("")
    lines.append("## Resultados")
    lines.append("")
    lines.append(
        "| Query | Modo | Tiempo (s) | Output | Reasoning | tok/s | TTFT |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|"
    )

    for item in results:
        lines.append(
            "| "
            f"{item['query_id']} | "
            f"{item['mode'].upper()} | "
            f"{item['elapsed_seconds']:.3f} | "
            f"{item['total_output_tokens']} | "
            f"{item['reasoning_output_tokens']} | "
            f"{item['tokens_per_second']:.2f} | "
            f"{item['time_to_first_token_seconds']:.3f} |"
        )

    lines.append("")
    lines.append("## Respuestas")
    lines.append("")

    current_query = None

    for item in results:
        if item["query_id"] != current_query:
            current_query = item["query_id"]
            lines.append(
                f"### {item['query_id']}: "
                f"{item['query']}"
            )
            lines.append("")
            lines.append(
                "**Chunks recuperados:** "
                + ", ".join(item["retrieved_chunk_ids"])
            )
            lines.append("")

        lines.append(
            f"#### Reasoning {item['mode'].upper()}"
        )
        lines.append("")
        lines.append(item["response"])
        lines.append("")

    lines.append("## Evaluación")
    lines.append("")
    lines.append(
        "La evaluación de calidad debe considerar principalmente:"
    )
    lines.append("")
    lines.append(
        "1. Corrección respecto del contexto recuperado."
    )
    lines.append(
        "2. Ausencia de información inventada."
    )
    lines.append(
        "3. Claridad y precisión."
    )
    lines.append(
        "4. Coste adicional de reasoning."
    )
    lines.append("")
    lines.append(
        "## Decisión"
    )
    lines.append("")
    lines.append(
        "Pendiente de decisión final tras revisar las respuestas "
        "OFF y ON y sus métricas."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = load_answerable_cases()

    embedding_client = LocalEmbeddingClient()

    store = JsonVectorStore(INDEX_PATH)
    store.load()

    retriever = Retriever(store)

    client = LocalChatClient(
        timeout=240.0,
    )

    timestamp = datetime.now(
        timezone.utc,
    ).strftime("%Y-%m-%dT%H-%M-%SZ")

    all_results: list[dict] = []

    for case in cases:
        query_id = case["id"]
        query = case["query"]

        retrieved = build_context_for_case(
            query,
            embedding_client,
            retriever,
        )

        retrieved_chunk_ids = [
            result.chunk.id
            for result in retrieved
        ]

        print()
        print("=" * 70)
        print(f"{query_id}: {query}")
        print("=" * 70)
        print(
            "Chunks: "
            + ", ".join(retrieved_chunk_ids)
        )

        for mode, profile in MODELS.items():
            print()
            print(
                f"Running reasoning {mode.upper()}..."
            )

            generation = run_generation(
                client,
                profile,
                query,
                retrieved,
            )

            item = {
                "query_id": query_id,
                "query": query,
                "mode": mode,
                "model": client.model,
                "profile": profile.name,
                "reasoning": profile.reasoning,
                "max_output_tokens": profile.max_output_tokens,
                "temperature": profile.temperature,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                **generation,
            }

            all_results.append(item)

            print(
                f"Tiempo: "
                f"{generation['elapsed_seconds']:.3f} s"
            )
            print(
                f"Output tokens: "
                f"{generation['total_output_tokens']}"
            )
            print(
                f"Reasoning tokens: "
                f"{generation['reasoning_output_tokens']}"
            )
            print()
            print("Respuesta:")
            print(generation["response"])

    report_data = {
        "benchmark": "rag_answer_reasoning_off_vs_on",
        "timestamp_utc": timestamp,
        "model": client.model,
        "inference_layer_version": INFERENCE_LAYER_VERSION,
        "cases": len(cases),
        "results": all_results,
    }

    json_path = (
        OUTPUT_DIR
        / f"reasoning_off_on_{timestamp}.json"
    )

    markdown_path = (
        OUTPUT_DIR
        / f"reasoning_off_on_{timestamp}.md"
    )

    json_path.write_text(
        json.dumps(
            report_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown_path.write_text(
        build_markdown_report(
            timestamp,
            client.model,
            cases,
            all_results,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("BENCHMARK FINALIZADO")
    print("=" * 70)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()