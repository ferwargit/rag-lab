from rag_lab.inference import (
    ANSWER_PROFILE,
    CLASSIFIER_PROFILE,
    EVIDENCE_PROFILE,
)


def test_evidence_profile_disables_reasoning() -> None:
    assert EVIDENCE_PROFILE.reasoning == "off"
    assert EVIDENCE_PROFILE.max_output_tokens == 256


def test_classifier_profile_disables_reasoning() -> None:
    assert CLASSIFIER_PROFILE.reasoning == "off"
    assert CLASSIFIER_PROFILE.max_output_tokens == 128


def test_answer_profile_uses_low_reasoning() -> None:
    assert ANSWER_PROFILE.reasoning == "low"
    assert ANSWER_PROFILE.max_output_tokens == 2048