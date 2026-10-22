import asyncio
import base64
import contextlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from fastapi import WebSocket, WebSocketDisconnect
from pathlib import Path
from dotenv import load_dotenv

from core.file_reader import (
    FileReadError,
    FileReadResult,
    format_files_for_prompt,
    read_file_from_bytes,
)
from core.tool_manager import (
    SYSTEM_PROMPT_TOOLS,
    TOOLS_SCHEMA,
    build_tool_result_messages,
    handle_tool_calls,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "https://foundation-models.api.cloud.ru/v1/chat/completions"
API_KEY = os.getenv("API_KEY")
DEFAULT_MODEL = os.getenv("CLOUD_MODEL", "anthropic/claude-opus-4.6")
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
MAX_TOOL_ROUNDS = 5
MAX_HISTORY_MESSAGES = 40
MAX_MESSAGE_CHARS = 100_000
MAX_WS_FILE_BYTES = 15 * 1024 * 1024
WS_HEARTBEAT_INTERVAL = 15
API_CONNECT_TIMEOUT = 30.0
API_READ_TIMEOUT = 300.0
API_WRITE_TIMEOUT = 60.0
API_POOL_TIMEOUT = 60.0
SUPPORTED_WS_CLIENT_TYPES = {"ping", "message", None}

AVAILABLE_MODELS = [
    {
        "id": "anthropic/claude-opus-4.8",
        "name": "Claude Opus 4.8",
        "provider": "Anthropic",
        "description": "Самая мощная модель Claude — глубокий анализ, сложные задачи",
    },
    {
        "id": "anthropic/claude-sonnet-4.6",
        "name": "Claude Sonnet 4.6",
        "provider": "Anthropic",
        "description": "Быстрая и умная — баланс скорости и качества",
    },
    {
        "id": "openai/gpt-5.5",
        "name": "GPT-5.5",
        "provider": "OpenAI",
        "description": "Флагманская модель OpenAI",
    },
    {
        "id": "openai/gpt-5.3-codex",
        "name": "GPT-5.3 Codex",
        "provider": "OpenAI",
        "description": "Специализирована на коде и разработке",
    },
    {
        "id": "zai-org/GLM-5.3",
        "name": "GLM-5.3",
        "provider": "ZAI",
        "description": "Мощная open-source модель",
    },
    {
        "id": "google/gemini-3.1-pro-preview",
        "name": "Gemini 3.1 Pro",
        "provider": "Google",
        "description": "Продвинутая модель Google с большим контекстом",
    },
]
AVAILABLE_MODEL_IDS = {model["id"] for model in AVAILABLE_MODELS}

_http_client: httpx.AsyncClient | None = None
_streaming_supported_cache: dict[str, bool] = {}


@dataclass(slots=True)
class WsChatRequest:
    message: str
    history: list[dict[str, Any]]
    files: list[dict[str, Any]]
    temperature: float
    max_tokens: int
    model: str


@dataclass(slots=True)
class WsPayloadValidation:
    kind: Literal["ping", "message", "invalid"]
    request: WsChatRequest | None = None
    error_message: str | None = None


class AgentInputError(ValueError):
    pass


def get_available_models() -> list[dict[str, str]]:
    return AVAILABLE_MODELS


def get_default_model() -> str:
    return DEFAULT_MODEL if DEFAULT_MODEL in AVAILABLE_MODEL_IDS else AVAILABLE_MODELS[0]["id"]


def get_api_config() -> dict[str, Any]:
    return {
        "model": get_default_model(),
        "models": get_available_models(),
    }


async def startup_http_client() -> None:
    global _http_client
    if _http_client is not None:
        return
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=API_CONNECT_TIMEOUT,
            read=API_READ_TIMEOUT,
            write=API_WRITE_TIMEOUT,
            pool=API_POOL_TIMEOUT,
        ),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )


async def shutdown_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        await startup_http_client()
    return _http_client


def clamp_temperature(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE
    return max(0.0, min(2.0, value))


def clamp_max_tokens(value: Any) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOKENS
    return max(1, min(32768, value))


def normalize_model(value: Any) -> str:
    if isinstance(value, str) and value in AVAILABLE_MODEL_IDS:
        return value
    return get_default_model()


def build_cloudru_headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError("Не найден API_KEY. Добавь его в файл .env")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def build_cloudru_body(
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    model: str,
    *,
    stream: bool,
    use_tools: bool = True,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "stream": stream,
    }
    if use_tools:
        body["tools"] = TOOLS_SCHEMA
        body["tool_choice"] = "auto"
    return body


def clean_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in {"user", "assistant", "system"} and isinstance(content, str) and content:
            cleaned.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    return cleaned


def inject_system_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = [msg.copy() for msg in messages]
    if copied and copied[0].get("role") == "system":
        copied[0]["content"] = f"{copied[0].get('content', '')}\n\n{SYSTEM_PROMPT_TOOLS}"
    else:
        copied.insert(0, {"role": "system", "content": SYSTEM_PROMPT_TOOLS})
    return copied


def _read_ws_file(file_item: dict[str, Any]) -> FileReadResult:
    if not isinstance(file_item, dict):
        raise AgentInputError("Элемент files должен быть объектом")

    name = file_item.get("name")
    file_data = file_item.get("data")
    mime_type = file_item.get("type", "application/octet-stream")

    if not isinstance(name, str) or not name.strip():
        raise AgentInputError("У файла отсутствует имя")
    if not isinstance(file_data, str) or not file_data:
        raise AgentInputError(f"Файл {name}: отсутствуют данные")

    try:
        raw_bytes = base64.b64decode(file_data, validate=True)
    except Exception as exc:
        raise AgentInputError(f"Файл {name}: некорректный base64") from exc

    if len(raw_bytes) > MAX_WS_FILE_BYTES:
        raise AgentInputError(f"Файл {name}: превышен лимит {MAX_WS_FILE_BYTES // (1024 * 1024)} МБ")

    try:
        return read_file_from_bytes(raw_bytes, name)
    except FileReadError as exc:
        raise AgentInputError(str(exc)) from exc


def build_user_content(user_text: str, files: list[dict[str, Any]]) -> str:
    text = (user_text or "")[:MAX_MESSAGE_CHARS]
    if not files:
        return text
    file_results = [_read_ws_file(item) for item in files]
    return format_files_for_prompt(file_results, text)


def prepare_messages(*, history: list[dict[str, Any]], user_text: str, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = clean_messages(history)
    user_content = build_user_content(user_text, files)
    messages.append({"role": "user", "content": user_content})
    return inject_system_prompt(messages)


def validate_ws_payload(raw: str) -> WsPayloadValidation:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return WsPayloadValidation(kind="invalid", error_message="Invalid JSON")

    event_type = data.get("type")
    if event_type == "ping":
        return WsPayloadValidation(kind="ping")
    if event_type not in SUPPORTED_WS_CLIENT_TYPES:
        return WsPayloadValidation(kind="invalid", error_message=f"Неподдерживаемый тип события: {event_type}")

    message = data.get("message", "")
    history = data.get("history", [])
    files = data.get("files", [])

    if not isinstance(message, str):
        return WsPayloadValidation(kind="invalid", error_message="Поле message должно быть строкой")
    if not isinstance(history, list):
        return WsPayloadValidation(kind="invalid", error_message="Поле history должно быть списком")
    if not isinstance(files, list):
        return WsPayloadValidation(kind="invalid", error_message="Поле files должно быть списком")
    if not message and not files:
        return WsPayloadValidation(kind="invalid", error_message="Пустое сообщение")

    request = WsChatRequest(
        message=message,
        history=history,
        files=files,
        temperature=clamp_temperature(data.get("temperature")),
        max_tokens=clamp_max_tokens(data.get("max_tokens")),
        model=normalize_model(data.get("model")),
    )
    return WsPayloadValidation(kind="message", request=request)


def parse_sse_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        logger.debug("Не удалось распарсить SSE JSON: %s", payload[:200])
        return None


async def safe_send_json(ws: WebSocket, payload: dict[str, Any]) -> bool:
    try:
        await ws.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError:
        return False
    except Exception:
        logger.exception("Не удалось отправить WS payload")
        return False


@contextlib.asynccontextmanager
async def status_keepalive(ws: WebSocket, content: str):
    stop_event = asyncio.Event()

    async def _keepalive() -> None:
        while not stop_event.is_set():
            await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
            if stop_event.is_set():
                break
            ok = await safe_send_json(ws, build_status_event(content))
            if not ok:
                break

    task = asyncio.create_task(_keepalive())
    try:
        yield
    finally:
        stop_event.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def build_status_event(content: str) -> dict[str, Any]:
    return {"type": "status", "content": content}


def build_done_event(*, answer: str, model: str, files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "done",
        "answer": answer,
        "model": model,
        "usage": None,
        "files": files,
    }


def map_exception_to_message(error: Exception | str) -> str:
    if isinstance(error, str):
        return error
    if isinstance(error, AgentInputError):
        return str(error)
    if isinstance(error, httpx.HTTPStatusError):
        status = getattr(error.response, "status_code", "?")
        detail = error.response.text[:500] if error.response is not None else str(error)
        return f"API ошибка ({status}): {detail}"
    if isinstance(error, httpx.RequestError):
        return f"Ошибка соединения с Cloud.ru: {error}"
    return f"Ошибка: {error}"


def build_error_event(error: Exception | str) -> dict[str, str]:
    return {"type": "error", "message": map_exception_to_message(error)}


async def run_tool_calls(message_dict: dict[str, Any], tool_calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assistant_message = {
        "role": "assistant",
        "content": message_dict.get("content") or "",
        "tool_calls": tool_calls,
    }
    tool_result = await asyncio.to_thread(handle_tool_calls, assistant_message)
    return build_tool_result_messages(tool_result, tool_result), tool_result.get("saved_files", [])

async def check_streaming_support(model: str) -> bool:
    cached = _streaming_supported_cache.get(model)
    if cached is not None:
        return cached

    client = await get_http_client()
    headers = build_cloudru_headers()
    body = build_cloudru_body(
        messages=inject_system_prompt([{"role": "user", "content": "ping"}]),
        temperature=0.0,
        max_tokens=8,
        model=model,
        stream=True,
    )

    try:
        async with client.stream("POST", API_URL, headers=headers, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    _streaming_supported_cache[model] = True
                    return True
        _streaming_supported_cache[model] = False
        return False
    except Exception as exc:
        logger.warning("Streaming probe failed for %s: %s", model, exc)
        _streaming_supported_cache[model] = False
        return False


async def stream_response(ws: WebSocket, messages: list[dict[str, Any]], temperature: float, max_tokens: int, model: str) -> tuple[str, list[dict[str, Any]]]:
    full_text = ""
    all_saved_files: list[dict[str, Any]] = []
    current_messages = list(messages)
    client = await get_http_client()

    for _ in range(MAX_TOOL_ROUNDS):
        headers = build_cloudru_headers()
        body = build_cloudru_body(current_messages, temperature, max_tokens, model, stream=True)
        assistant_tool_calls: list[dict[str, Any]] = []
        assistant_text = ""

        async with status_keepalive(ws, "Жду потоковый ответ модели…"):
            async with client.stream("POST", API_URL, headers=headers, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    payload = parse_sse_line(line)
                    if payload is None:
                        continue
                    for choice in payload.get("choices", []):
                        delta = choice.get("delta", {}) or {}
                        text_delta = delta.get("content", "") or ""
                        if text_delta:
                            assistant_text += text_delta
                            full_text += text_delta
                            if not await safe_send_json(ws, {"type": "chunk", "content": text_delta}):
                                raise WebSocketDisconnect()

                        for tool_delta in delta.get("tool_calls", []) or []:
                            index = tool_delta.get("index", 0)
                            while len(assistant_tool_calls) <= index:
                                assistant_tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                })
                            current = assistant_tool_calls[index]
                            if tool_delta.get("id"):
                                current["id"] = tool_delta["id"]
                            if tool_delta.get("function", {}).get("name"):
                                current["function"]["name"] = tool_delta["function"]["name"]
                            if tool_delta.get("function", {}).get("arguments"):
                                current["function"]["arguments"] += tool_delta["function"]["arguments"]

        if not assistant_tool_calls:
            break

        assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": assistant_tool_calls}
        if assistant_text:
            assistant_msg["content"] = assistant_text
        current_messages.append(assistant_msg)

        async with status_keepalive(ws, "Инструменты ещё выполняются…"):
            tool_result_msgs, files_saved = await run_tool_calls(assistant_msg, assistant_tool_calls)

        all_saved_files.extend(files_saved)
        for file_info in files_saved:
            if file_info.get("success"):
                if not await safe_send_json(ws, build_status_event(f"✅ Файл сохранён: {file_info['file_name']}")):
                    raise WebSocketDisconnect()
        current_messages.extend(tool_result_msgs)
    else:
        warning = "\n\n⚠️ Достигнут лимит раундов вызова инструментов."
        full_text += warning
        if not await safe_send_json(ws, {"type": "chunk", "content": warning}):
            raise WebSocketDisconnect()

    return full_text, all_saved_files


async def sync_response(ws: WebSocket, messages: list[dict[str, Any]], temperature: float, max_tokens: int, model: str) -> tuple[str, list[dict[str, Any]]]:
    full_text = ""
    all_saved_files: list[dict[str, Any]] = []
    current_messages = list(messages)
    client = await get_http_client()

    for round_num in range(MAX_TOOL_ROUNDS):
        headers = build_cloudru_headers()
        body = build_cloudru_body(current_messages, temperature, max_tokens, model, stream=False)
        logger.info("[API REQUEST] round=%s, model=%s, messages_count=%s", round_num, model, len(body["messages"]))

        async with status_keepalive(ws, "Жду ответ модели…"):
            response = await client.post(API_URL, headers=headers, json=body)
        response.raise_for_status()

        result = response.json()
        choices = result.get("choices", [])
        if not choices:
            break

        message = choices[0].get("message", {})
        content_text = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", [])

        if content_text:
            full_text += content_text
            paragraphs = content_text.split("\n")
            for index, paragraph in enumerate(paragraphs):
                chunk = paragraph + ("\n" if index < len(paragraphs) - 1 else "")
                if chunk and not await safe_send_json(ws, {"type": "chunk", "content": chunk}):
                    raise WebSocketDisconnect()

        if not tool_calls:
            break

        current_messages.append(message)
        if not await safe_send_json(ws, build_status_event("💾 Сохраняю файл…")):
            raise WebSocketDisconnect()

        async with status_keepalive(ws, "Инструменты ещё выполняются…"):
            tool_result_msgs, files_saved = await run_tool_calls(message, tool_calls)

        all_saved_files.extend(files_saved)
        for file_info in files_saved:
            if file_info.get("success"):
                if not await safe_send_json(ws, build_status_event(f"✅ Файл сохранён: {file_info['file_name']}")):
                    raise WebSocketDisconnect()

        current_messages.extend(tool_result_msgs)
    else:
        warning = "\n\n⚠️ Достигнут лимит раундов вызова инструментов."
        full_text += warning
        if not await safe_send_json(ws, {"type": "chunk", "content": warning}):
            raise WebSocketDisconnect()

    return full_text, all_saved_files
