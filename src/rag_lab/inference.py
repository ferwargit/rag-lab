from dataclasses import dataclass
from typing import Literal


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


ANSWER_PROFILE = InferenceProfile(
    name="answer",
    reasoning="low",
    max_output_tokens=2048,
    temperature=0.2,
)