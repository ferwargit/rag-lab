from dataclasses import dataclass
from typing import Literal

INFERENCE_LAYER_VERSION = "1.0.0"

ReasoningMode = Literal[
    "off",
    "low",
    "medium",
    "high",
    "on",
]


@dataclass(frozen=True)
class InferenceProfile:
    name: str
    reasoning: ReasoningMode
    max_output_tokens: int
    temperature: float


@dataclass(frozen=True)
class ModelCapabilities:
    model_id: str
    reasoning_options: tuple[ReasoningMode, ...]
    default_reasoning: ReasoningMode | None = None

    @property
    def supports_reasoning(self) -> bool:
        return bool(self.reasoning_options)

    def supports_reasoning_mode(
        self,
        mode: ReasoningMode,
    ) -> bool:
        return mode in self.reasoning_options


EVIDENCE_PROFILE = InferenceProfile(
    name="evidence",
    reasoning="off",
    max_output_tokens=256,
    temperature=0.0,
)


CLASSIFIER_PROFILE = InferenceProfile(
    name="classifier",
    reasoning="off",
    max_output_tokens=128,
    temperature=0.0,
)


RAG_ANSWER_PROFILE = InferenceProfile(
    name="rag-answer",
    reasoning="off",
    max_output_tokens=1024,
    temperature=0.2,
)


DEEP_ANSWER_PROFILE = InferenceProfile(
    name="deep-answer",
    reasoning="on",
    max_output_tokens=8192,
    temperature=0.2,
)


def validate_profile(
    profile: InferenceProfile,
    capabilities: ModelCapabilities,
) -> None:
    if profile.max_output_tokens <= 0:
        raise ValueError(
            "max_output_tokens debe ser mayor que cero."
        )

    if not 0.0 <= profile.temperature <= 1.0:
        raise ValueError(
            "temperature debe estar entre 0.0 y 1.0."
        )

    if not capabilities.supports_reasoning_mode(
        profile.reasoning
    ):
        supported = ", ".join(
            capabilities.reasoning_options
        ) or "ninguno"

        raise ValueError(
            f"El perfil '{profile.name}' solicita "
            f"reasoning='{profile.reasoning}', pero el modelo "
            f"'{capabilities.model_id}' soporta: {supported}."
        )
