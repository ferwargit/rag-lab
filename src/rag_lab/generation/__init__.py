"""API pública del paquete de generación."""

from rag_lab.infrastructure.chat.lm_studio import (
    GenerationError,
    LocalChatClient,
)

__all__ = [
    "GenerationError",
    "LocalChatClient",
]