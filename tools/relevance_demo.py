from rag_lab.generation import LocalChatClient


def main() -> None:
    query = (
        "¿Cómo se conecta el piano al ordenador?"
    )

    evidence = (
        "Fuente A:\n"
        "El dispositivo MIDI se conecta al ordenador "
        "mediante una interfaz USB MIDI.\n\n"
        "Fuente B:\n"
        "El dispositivo MIDI se conecta al ordenador "
        "mediante una interfaz Ethernet."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Evalúa las evidencias proporcionadas "
                "para determinar si son consistentes entre sí. "
                "Si contienen afirmaciones incompatibles "
                "sobre el mismo hecho, existe un conflicto. "
                "Responde únicamente con una de estas dos "
                "palabras: CONSISTENT o CONFLICT."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Pregunta:\n{query}\n\n"
                f"Evidencias:\n{evidence}\n\n"
                "¿Las evidencias son consistentes entre sí?"
            ),
        },
    ]

    client = LocalChatClient()

    answer = client.generate(messages)

    print("PREGUNTA")
    print("=" * 70)
    print(query)
    print()

    print("EVIDENCIAS")
    print("=" * 70)
    print(evidence)
    print()

    print("EVALUACIÓN")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()