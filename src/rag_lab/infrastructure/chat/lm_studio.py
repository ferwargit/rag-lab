import time
import json
import urllib.error
import urllib.request

from rag_lab.core.inference import (
    DEEP_ANSWER_PROFILE,
    InferenceProfile,
    ModelCapabilities,
    ReasoningMode,
    validate_profile,
)
from rag_lab.core.models import GenerationResult

class GenerationError(RuntimeError):
    """Error durante una generación del modelo local."""


class LocalChatClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "qwen/qwen3.5-9b",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._model_capabilities: ModelCapabilities | None = None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        profile: InferenceProfile = DEEP_ANSWER_PROFILE,
    ) -> GenerationResult:
        """Genera una respuesta usando un perfil de inferencia."""

        system_messages = [
            message["content"]
            for message in messages
            if message["role"] == "system"
        ]

        non_system_messages = [
            message
            for message in messages
            if message["role"] != "system"
        ]

        if len(non_system_messages) != 1:
            raise GenerationError(
                "LocalChatClient actualmente espera exactamente "
                "un mensaje no-system."
            )

        user_message = non_system_messages[0]

        if user_message["role"] != "user":
            raise GenerationError(
                "El único mensaje no-system debe tener role='user'."
            )

        if self._model_capabilities is None:
            self._model_capabilities = self.get_model_capabilities()

        capabilities = self._model_capabilities

        try:
            validate_profile(
                profile,
                capabilities,
            )
        except ValueError as exc:
            raise GenerationError(str(exc)) from exc

        payload = {
            "model": self.model,
            "input": user_message["content"],
            "store": False,
            "temperature": profile.temperature,
            "max_output_tokens": profile.max_output_tokens,
            "reasoning": profile.reasoning,
        }

        if system_messages:
            payload["system_prompt"] = "\n\n".join(system_messages)

        payload_bytes = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.base_url}/api/v1/chat",
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        generation_start = time.perf_counter()

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_body = response.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise GenerationError(
                f"LM Studio devolvió HTTP {exc.code}: {detail}"
            ) from exc

        except urllib.error.URLError as exc:
            raise GenerationError(
                f"No se pudo conectar con LM Studio: {exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise GenerationError(
                "La generación en LM Studio agotó el timeout."
            ) from exc

        finally:
            generation_time_seconds = (
                time.perf_counter() - generation_start
            )

        try:
            response_json = json.loads(response_body)

        except json.JSONDecodeError as exc:
            raise GenerationError(
                "LM Studio devolvió una respuesta JSON inválida."
            ) from exc

        output = response_json.get("output")

        if not isinstance(output, list):
            raise GenerationError(
                "La respuesta de LM Studio no contiene un campo "
                "'output' válido."
            )

        message_parts: list[str] = []
        reasoning_parts: list[str] = []

        for item in output:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            content = item.get("content")

            if not isinstance(content, str):
                continue

            if item_type == "message":
                message_parts.append(content)

            elif item_type == "reasoning":
                reasoning_parts.append(content)

        content = "".join(message_parts)
        reasoning = "".join(reasoning_parts) or None

        stats = response_json.get("stats")

        if not isinstance(stats, dict):
            raise GenerationError(
                "La respuesta de LM Studio no contiene estadísticas válidas."
            )

        input_tokens = stats.get("input_tokens", 0)
        total_output_tokens = stats.get("total_output_tokens", 0)
        reasoning_output_tokens = stats.get("reasoning_output_tokens", 0)

        tokens_per_second = stats.get("tokens_per_second")
        time_to_first_token_seconds = stats.get(
            "time_to_first_token_seconds"
        )

        response_id = response_json.get("response_id")
        model_instance_id = response_json.get("model_instance_id")

        if not isinstance(input_tokens, int):
            raise GenerationError("input_tokens inválido.")

        if not isinstance(total_output_tokens, int):
            raise GenerationError("total_output_tokens inválido.")

        if not isinstance(reasoning_output_tokens, int):
            raise GenerationError("reasoning_output_tokens inválido.")

        if tokens_per_second is not None and not isinstance(
            tokens_per_second,
            (int, float),
        ):
            raise GenerationError("tokens_per_second inválido.")

        if time_to_first_token_seconds is not None and not isinstance(
            time_to_first_token_seconds,
            (int, float),
        ):
            raise GenerationError(
                "time_to_first_token_seconds inválido."
            )

        if response_id is not None and not isinstance(response_id, str):
            raise GenerationError("response_id inválido.")

        if model_instance_id is not None and not isinstance(
            model_instance_id,
            str,
        ):
            raise GenerationError("model_instance_id inválido.")

        if not content:
            raise GenerationError(
                "LM Studio no devolvió un mensaje textual."
            )

        return GenerationResult(
            content=content,
            reasoning=reasoning,
            input_tokens=input_tokens,
            total_output_tokens=total_output_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            tokens_per_second=(
                float(tokens_per_second)
                if tokens_per_second is not None
                else None
            ),
            time_to_first_token_seconds=(
                float(time_to_first_token_seconds)
                if time_to_first_token_seconds is not None
                else None
            ),
            generation_time_seconds=generation_time_seconds,
            response_id=response_id,
            model_instance_id=model_instance_id,
        )

    def get_model_capabilities(
        self,
    ) -> ModelCapabilities:
        """Obtiene las capacidades declaradas por LM Studio."""

        request = urllib.request.Request(
            url=f"{self.base_url}/api/v1/models",
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_body = response.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise GenerationError(
                f"LM Studio devolvió HTTP {exc.code}: {detail}"
            ) from exc

        except urllib.error.URLError as exc:
            raise GenerationError(
                f"No se pudo consultar LM Studio: {exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise GenerationError(
                "La consulta de capacidades de LM Studio agotó "
                "el timeout."
            ) from exc

        try:
            response_json = json.loads(response_body)

        except json.JSONDecodeError as exc:
            raise GenerationError(
                "LM Studio devolvió JSON inválido al consultar "
                "las capacidades."
            ) from exc

        models = response_json.get("models")

        if not isinstance(models, list):
            raise GenerationError(
                "La respuesta de LM Studio no contiene una lista "
                "de modelos válida."
            )

        model_data = next(
            (
                model
                for model in models
                if isinstance(model, dict)
                and model.get("key") == self.model
            ),
            None,
        )

        if model_data is None:
            raise GenerationError(
                f"El modelo '{self.model}' no aparece en "
                "GET /api/v1/models."
            )

        capabilities = model_data.get("capabilities", {})

        if not isinstance(capabilities, dict):
            capabilities = {}

        reasoning = capabilities.get("reasoning", {})

        if not isinstance(reasoning, dict):
            reasoning = {}

        allowed_options = reasoning.get(
            "allowed_options",
            [],
        )

        if not isinstance(allowed_options, list):
            allowed_options = []

        valid_modes: list[ReasoningMode] = []

        for option in allowed_options:
            if option in {
                "off",
                "low",
                "medium",
                "high",
                "on",
            }:
                valid_modes.append(option)

        default_reasoning = reasoning.get("default")

        if default_reasoning not in {
            None,
            "off",
            "low",
            "medium",
            "high",
            "on",
        }:
            default_reasoning = None

        return ModelCapabilities(
            model_id=self.model,
            reasoning_options=tuple(valid_modes),
            default_reasoning=default_reasoning,
        )
