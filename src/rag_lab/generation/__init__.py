"""API pública del paquete de generación."""

from rag_lab.generation.prompting import (
    SYSTEM_INSTRUCTION,
    build_rag_messages,
)
from rag_lab.infrastructure.chat.lm_studio import (
    GenerationError,
    LocalChatClient,
)

__all__ = [
    "GenerationError",
    "LocalChatClient",
    "SYSTEM_INSTRUCTION",
    "build_rag_messages",
]