from rag_lab.generation import LocalChatClient
from rag_lab.inference import EVIDENCE_PROFILE, ANSWER_PROFILE


def print_result(result) -> None:
    print(f"Respuesta: {result.content}")
    print(
        f"Tokens entrada: {result.input_tokens}"
    )
    print(
        f"Tokens salida: {result.total_output_tokens}"
    )
    print(
        f"Tokens razonamiento: "
        f"{result.reasoning_output_tokens}"
    )
    print(
        f"Tokens/s: {result.tokens_per_second}"
    )
    print(
        f"TTFT: {result.time_to_first_token_seconds}"
    )

    if result.reasoning:
        preview = result.reasoning[:300].replace("\n", " ")
        print(f"Reasoning preview: {preview}...")


def main() -> None:
    client = LocalChatClient()

    messages = [
        {
            "role": "system",
            "content": (
                "Responde de forma extremadamente breve."
            ),
        },
        {
            "role": "user",
            "content": (
                "¿Qué método utiliza MIDI Laboratory "
                "para conectar el dispositivo MIDI al ordenador?"
            ),
        },
    ]

    print("=== REASONING OFF ===")

    off_result = client.generate(
        messages,
        profile=EVIDENCE_PROFILE,
    )

    print_result(off_result)

    print()
    print("=== REASONING LOW ===")

    low_result = client.generate(
        messages,
        profile=ANSWER_PROFILE,
    )

    print_result(low_result)


if __name__ == "__main__":
    main()