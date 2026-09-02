from rag_lab.generation import LocalChatClient


def main() -> None:
    client = LocalChatClient()

    capabilities = client.get_model_capabilities()

    print("MODELO")
    print("=" * 70)
    print(capabilities.model_id)

    print()
    print("REASONING OPTIONS")
    print("=" * 70)

    for option in capabilities.reasoning_options:
        print(f"- {option}")

    print()
    print("DEFAULT")
    print("=" * 70)
    print(capabilities.default_reasoning)


if __name__ == "__main__":
    main()