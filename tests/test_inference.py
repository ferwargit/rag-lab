import pytest

from rag_lab.inference import (
    CLASSIFIER_PROFILE,
    DEEP_ANSWER_PROFILE,
    EVIDENCE_PROFILE,
    INFERENCE_LAYER_VERSION,
    InferenceProfile,
    ModelCapabilities,
    RAG_ANSWER_PROFILE,
    validate_profile,
)

def test_inference_layer_version_is_1_0_0() -> None:
    assert INFERENCE_LAYER_VERSION == "1.0.0"


def test_evidence_profile_disables_reasoning() -> None:
    assert EVIDENCE_PROFILE.reasoning == "off"
    assert EVIDENCE_PROFILE.max_output_tokens == 256


def test_classifier_profile_disables_reasoning() -> None:
    assert CLASSIFIER_PROFILE.reasoning == "off"
    assert CLASSIFIER_PROFILE.max_output_tokens == 128


def test_rag_answer_profile_disables_reasoning() -> None:
    assert RAG_ANSWER_PROFILE.reasoning == "off"
    assert RAG_ANSWER_PROFILE.max_output_tokens == 1024


def test_deep_answer_profile_enables_reasoning() -> None:
    assert DEEP_ANSWER_PROFILE.reasoning == "on"
    assert DEEP_ANSWER_PROFILE.max_output_tokens == 8192


def test_qwen_style_capabilities_support_off_and_on() -> None:
    capabilities = ModelCapabilities(
        model_id="qwen/qwen3.5-9b",
        reasoning_options=("off", "on"),
        default_reasoning="on",
    )

    assert capabilities.supports_reasoning_mode("off")
    assert capabilities.supports_reasoning_mode("on")
    assert not capabilities.supports_reasoning_mode("low")


def test_validate_profile_accepts_supported_reasoning() -> None:
    capabilities = ModelCapabilities(
        model_id="qwen/qwen3.5-9b",
        reasoning_options=("off", "on"),
    )

    validate_profile(
        EVIDENCE_PROFILE,
        capabilities,
    )

    validate_profile(
        RAG_ANSWER_PROFILE,
        capabilities,
    )

    validate_profile(
        DEEP_ANSWER_PROFILE,
        capabilities,
    )


def test_validate_profile_rejects_unsupported_reasoning() -> None:
    capabilities = ModelCapabilities(
        model_id="qwen/qwen3.5-9b",
        reasoning_options=("off", "on"),
    )

    unsupported = InferenceProfile(
        name="unsupported",
        reasoning="low",
        max_output_tokens=256,
        temperature=0.2,
)

    with pytest.raises(
        ValueError,
        match="soporta: off, on",
    ):
        validate_profile(
            unsupported,
            capabilities,
        )


def test_validate_profile_rejects_invalid_token_budget() -> None:
    capabilities = ModelCapabilities(
        model_id="qwen/qwen3.5-9b",
        reasoning_options=("off", "on"),
    )

    invalid = InferenceProfile(
        name="invalid",
        reasoning="off",
        max_output_tokens=0,
        temperature=0.2,
    )

    with pytest.raises(
        ValueError,
        match="mayor que cero",
    ):
        validate_profile(
            invalid,
            capabilities,
        )


def test_validate_profile_rejects_invalid_temperature() -> None:
    capabilities = ModelCapabilities(
        model_id="qwen/qwen3.5-9b",
        reasoning_options=("off", "on"),
    )

    invalid = InferenceProfile(
        name="invalid",
        reasoning="off",
        max_output_tokens=256,
        temperature=1.5,
    )

    with pytest.raises(
        ValueError,
        match="entre 0.0 y 1.0",
    ):
        validate_profile(
            invalid,
            capabilities,
        )