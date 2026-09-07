"""Compatibilidad temporal: el cliente de chat vive en infraestructura."""

from rag_lab.infrastructure.chat.lm_studio import (
    GenerationError,
    LocalChatClient,
)

__all__ = [
    "GenerationError",
    "LocalChatClient",
]