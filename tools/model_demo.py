from rag_lab.models import DocumentChunk


def main() -> None:
    chunk = DocumentChunk(
        id="knowledge-000",
        text="El piano envía eventos MIDI al ordenador.",
        source="data/knowledge.txt",
        index=0,
        metadata={
            "type": "text",
            "language": "es",
        },
    )

    print(chunk)


if __name__ == "__main__":
    main()