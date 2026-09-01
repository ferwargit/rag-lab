import json
import urllib.error
import urllib.request


class GenerationError(RuntimeError):
    """Error producido durante la generación."""


class LocalChatClient:
    """Cliente mínimo para chat completions de LM Studio."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:1234",
        model: str = "qwen/qwen3.5-9b",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Genera una respuesta usando el LLM local."""

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:
                raw_data = response.read()

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise GenerationError(
                f"LM Studio respondió HTTP {exc.code}: {detail}"
            ) from exc

        except urllib.error.URLError as exc:
            raise GenerationError(
                f"No se pudo conectar con LM Studio: {exc.reason}"
            ) from exc

        try:
            result = json.loads(raw_data)

        except json.JSONDecodeError as exc:
            raise GenerationError(
                "LM Studio devolvió JSON inválido."
            ) from exc

        try:
            content = result["choices"][0]["message"]["content"]

        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(
                "La respuesta de LM Studio no contiene "
                "el formato esperado."
            ) from exc

        if not isinstance(content, str):
            raise GenerationError(
                "La respuesta del modelo no es texto."
            )

        return content.strip()