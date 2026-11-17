"""Cliente de Gemini (Google AI Studio) con salida estructurada validada con pydantic.

Credenciales: variable GEMINI_API_KEY (o GOOGLE_API_KEY), normalmente en el fichero .env del proyecto.
Las peticiones concurrentes se limitan con un semáforo global para no chocar con el límite de peticiones
por minuto de la cuenta; los 429/5xx se reintentan con espera exponencial.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import threading
import time
from types import SimpleNamespace
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .config import LlmConfig

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
# effort de config.yaml -> thinking_level (Gemini 3.x) / thinking_budget (Gemini 2.x)
THINKING_LEVEL = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "XHIGH", "max": "XHIGH"}
CHECK_TIMEOUT_S = 60
THINKING_BUDGET = {"low": 1024, "medium": 4096, "high": 16384, "xhigh": 24576, "max": 32768}


class LlmError(RuntimeError):
    pass


def _transient_errors() -> tuple[type[BaseException], ...]:
    """Timeouts y fallos de conexión (httpx/httpx2 dentro del SDK) además de los de red de Python."""
    try:
        from google.genai import _api_client
        extra = tuple(getattr(_api_client, "_HTTPX_TRANSIENT_EXC", ()))
    except Exception:
        extra = ()
    return (OSError, TimeoutError, *extra)


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def gemini_schema(model: type[BaseModel]) -> dict:
    """JSON Schema de pydantic con las referencias ($ref/$defs) resueltas en línea."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return resolve({**target, **extra})
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def _text_of(content: list[dict]) -> list[str]:
    return [block["text"] for block in content if block.get("type", "text") == "text" and block.get("text")]


class Llm:
    def __init__(self, cfg: LlmConfig, client=None):
        self.cfg = cfg
        if client is None:
            from google import genai
            from google.genai import types

            key = api_key()
            if not key:
                raise LlmError("Falta GEMINI_API_KEY: añádela al fichero .env del proyecto (ver README)")
            client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=cfg.timeout_s * 1000))
        self.client = client
        self._slots = threading.BoundedSemaphore(max(1, cfg.max_concurrency))
        self._model_lock = threading.Lock()
        self._active = 0  # índice en self.models del modelo en uso

    @property
    def models(self) -> list[str]:
        return list(dict.fromkeys([self.cfg.model, *self.cfg.fallback_models]))

    def _config(self, model: str, system: str, output: type[BaseModel], max_tokens: int):
        from google.genai import types

        if model.startswith("gemini-2"):
            thinking = types.ThinkingConfig(thinking_budget=THINKING_BUDGET[self.cfg.effort])
        else:
            thinking = types.ThinkingConfig(thinking_level=THINKING_LEVEL[self.cfg.effort])
        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=gemini_schema(output),
            max_output_tokens=max_tokens,
            thinking_config=thinking,
        )

    def _switch(self, failed: str, reason: str) -> bool:
        """Pasa al siguiente modelo de reserva (para todos los hilos). False si no quedan."""
        with self._model_lock:
            models = self.models
            if models[self._active] != failed:  # otro hilo ya cambió de modelo
                return True
            if self._active + 1 >= len(models):
                return False
            self._active += 1
            log.warning("Gemini: %s con %s; se pasa a %s", reason, failed, models[self._active])
            return True

    def _call(self, *, system: str, contents: list[str], output: type[BaseModel], max_tokens: int):
        """generate_content con reintentos (cuota, red) y cambio a un modelo de reserva si el actual está
        saturado (503 repetidos) o no existe (404)."""
        from google.genai import errors

        delay, attempt, overloaded = 5.0, 0, 0
        while True:
            model = self.models[self._active]
            attempt += 1
            try:
                with self._slots:
                    return model, self.client.models.generate_content(
                        model=model, contents=contents, config=self._config(model, system, output, max_tokens))
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                detail = (getattr(exc, "message", None) or str(exc)).splitlines()[0][:160]
                if code == 404 and self._switch(model, "modelo no disponible (404)"):
                    attempt, overloaded, delay = 0, 0, 5.0
                    continue
                if code in (500, 503):
                    overloaded += 1
                    if overloaded >= self.cfg.overload_retries and self._switch(model, f"{code} repetidos ({detail})"):
                        attempt, overloaded, delay = 0, 0, 5.0
                        continue
                if code not in RETRY_STATUS or attempt > self.cfg.max_retries:
                    raise LlmError(f"Gemini {code} con {model}: {detail}") from exc
                wait = min(120.0, delay) * (1 + random.random() / 2)
                log.warning("Gemini %s con %s (intento %d/%d): %s. Reintento en %.0f s",
                            code, model, attempt, self.cfg.max_retries, detail, wait)
            except _transient_errors() as exc:  # sin respuesta en timeout_s, red caída...
                kind = "sin respuesta" if "timeout" in type(exc).__name__.lower() else "error de red"
                overloaded += 1
                if overloaded >= self.cfg.overload_retries and self._switch(model, f"{kind} repetido"):
                    attempt, overloaded, delay = 0, 0, 5.0
                    continue
                if attempt > self.cfg.max_retries:
                    raise LlmError(f"Gemini {kind} con {model}: {type(exc).__name__} {exc}") from exc
                wait = min(120.0, delay)
                log.warning("Gemini %s con %s (%s); reintento en %.0f s", kind, model, type(exc).__name__, wait)
            time.sleep(wait)
            delay *= 2

    def structured(
        self,
        *,
        system: str,
        content: list[dict],
        output: type[T],
        max_tokens: int = 64000,
    ) -> tuple[T, object]:
        """Devuelve (objeto validado, metadatos con model/usage/stop_reason) o lanza LlmError."""
        used, resp = self._call(system=system, contents=_text_of(content), output=output, max_tokens=max_tokens)

        feedback = getattr(resp, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LlmError(f"Gemini bloqueó la petición ({feedback.block_reason})")
        cand = (resp.candidates or [None])[0]
        finish = str(getattr(getattr(cand, "finish_reason", None), "value", getattr(cand, "finish_reason", None)))
        if finish == "MAX_TOKENS":
            raise LlmError("Respuesta truncada por max_output_tokens")
        if finish in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
            raise LlmError(f"Gemini cortó la respuesta (finish_reason={finish})")
        text = resp.text or ""
        try:
            parsed = output.model_validate_json(text)
        except ValidationError as exc:
            raise LlmError(f"Salida no válida para {output.__name__} (finish_reason={finish}): {exc}") from exc

        um = getattr(resp, "usage_metadata", None)
        usage = SimpleNamespace(
            input_tokens=getattr(um, "prompt_token_count", None),
            # los tokens de razonamiento se facturan como salida
            output_tokens=(getattr(um, "candidates_token_count", None) or 0) + (getattr(um, "thoughts_token_count", None) or 0),
            cache_read_input_tokens=getattr(um, "cached_content_token_count", None),
            cache_creation_input_tokens=None,
        )
        model = getattr(resp, "model_version", None) or used
        return parsed, SimpleNamespace(model=model, usage=usage, stop_reason=finish)


def check(cfg: LlmConfig) -> None:
    """Comprueba la clave, lista los modelos disponibles y prueba el modelo principal y los de reserva."""
    from google import genai
    from google.genai import errors

    key = api_key()
    if not key:
        raise LlmError("No hay GEMINI_API_KEY en el entorno ni en .env")
    from google.genai import types
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=CHECK_TIMEOUT_S * 1000))
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "gemini" in m.name and (not actions or "generateContent" in actions):
            names.append(m.name.removeprefix("models/"))
    candidates = list(dict.fromkeys([cfg.model, *cfg.fallback_models]))
    print("Modelos Gemini disponibles con tu clave (* = principal, r = reserva):")
    for n in sorted(names):
        mark = "*" if n == cfg.model else ("r" if n in candidates else " ")
        print(f"  {mark} {n}")

    class Ping(BaseModel):
        ok: bool
        saludo: str

    print(f"\nPrueba de cada modelo configurado (una llamada, máximo {CHECK_TIMEOUT_S} s cada una):")
    working = []
    for name in candidates:
        if names and name not in names:
            print(f"  ✗ {name}: no aparece en la lista de tu clave")
            continue
        one = Llm(cfg.model_copy(update={"model": name, "fallback_models": [], "max_retries": 0, "overload_retries": 99,
                                         "effort": "low"}), client)
        print(f"  … {name}", end="", flush=True)
        t0 = time.monotonic()
        try:
            res, meta = one.structured(system="Responde en JSON.", content=[{"type": "text", "text": "Di hola."}],
                                       output=Ping, max_tokens=2048)
            print(f"\r  ✓ {name}: responde en {time.monotonic() - t0:.1f} s")
            working.append(name)
        except (LlmError, errors.APIError, *_transient_errors()) as exc:
            msg = str(exc).replace("\n", " ")[:150]
            if "timeout" in type(exc).__name__.lower() or "sin respuesta" in msg:
                msg = f"sin respuesta en {CHECK_TIMEOUT_S} s"
            print(f"\r  ✗ {name}: {msg}")
    print()
    if not working:
        raise LlmError("Ningún modelo configurado responde ahora mismo. Si son 503, Google está saturado: "
                       "prueba más tarde o añade otro modelo a llm.fallback_models en config.yaml.")
    if working[0] != cfg.model:
        print(f"El principal no responde ahora; el pipeline usará {working[0]} (reserva) hasta que se recupere.")
    else:
        print("Todo correcto.")
