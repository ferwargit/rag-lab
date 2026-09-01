from rag_lab.embeddings import LocalEmbeddingClient


def main() -> None:
    client = LocalEmbeddingClient()

    text = "El piano envía eventos MIDI al ordenador."

    embedding = client.embed(text)

    print(f"Texto: {text}")
    print(f"Dimensiones: {len(embedding)}")
    print()
    print("Primeros 10 valores:")

    for value in embedding[:10]:
        print(value)


if __name__ == "__main__":
    main()