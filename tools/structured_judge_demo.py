import json

from rag_lab.generation import LocalChatClient


def main() -> None:
    query = "¿Cómo se conecta el piano al ordenador?"

    evidence = (
        "Fuente A:\n"
        "MIDI Laboratory es una aplicación personal "
        "desarrollada con Electron. "
        "La aplicación recibe eventos MIDI provenientes "
        "de un piano digital.\n\n"
        "Fuente B:\n"
        "El dispositivo MIDI se conecta al ordenador "
        "mediante una interfaz USB MIDI."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Evalúa si el conjunto de evidencias contiene "
                "información suficiente para responder la pregunta. "
                "Puedes combinar información de varias fuentes. "
                "No inventes información que no esté respaldada "
                "por las evidencias.\n\n"
                "Devuelve exclusivamente JSON válido con esta forma:\n"
                '{"sufficient": true}\n'
                'o\n'
                '{"sufficient": false}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Pregunta:\n{query}\n\n"
                f"Evidencias:\n{evidence}"
            ),
        },
    ]

    client = LocalChatClient()

    answer = client.generate(messages)

    print("RESPUESTA RAW")
    print("=" * 70)
    print(answer)
    print()

    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError as exc:
        print("ERROR: la respuesta no es JSON válido.")
        print(exc)
        return

    print("JSON PARSEADO")
    print("=" * 70)
    print(parsed)


if __name__ == "__main__":
    main()