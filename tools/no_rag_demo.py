from rag_lab.generation import LocalChatClient


def main() -> None:
    query = "¿Cómo se conecta el piano al ordenador?"

    messages = [
        {
            "role": "system",
            "content": (
                "Responde en español de forma precisa. "
                "No tienes acceso a documentación externa."
            ),
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    client = LocalChatClient()

    answer = client.generate(messages)

    print("PREGUNTA")
    print("=" * 70)
    print(query)
    print()

    print("RESPUESTA SIN RAG")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()