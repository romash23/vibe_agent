import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.agent import (
    WsChatRequest,
    build_done_event,
    build_error_event,
    build_status_event,
    check_streaming_support,
    get_api_config,
    get_available_models,
    prepare_messages,
    safe_send_json,
    shutdown_http_client,
    startup_http_client,
    stream_response,
    sync_response,
    validate_ws_payload,
)
from core.file_reader import get_supported_extensions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UI_PATH = STATIC_DIR / "v0_ws_ui.html"

app = FastAPI(title="Local Claude API Server", version="3.0.0")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def _startup() -> None:
    await startup_http_client()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await shutdown_http_client()


@app.get("/", response_class=HTMLResponse)
async def chat_page() -> HTMLResponse:
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.get("/supported-formats")
async def supported_formats() -> dict:
    return {"extensions": get_supported_extensions()}


@app.get("/api/config")
async def api_config() -> dict:
    return get_api_config()


@app.get("/models")
async def models() -> list[dict]:
    return get_available_models()


async def process_ws_message(ws: WebSocket, payload: WsChatRequest) -> bool:
    if not await safe_send_json(ws, build_status_event("Обрабатываю запрос…")):
        return False

    try:
        messages = prepare_messages(
            history=payload.history,
            user_text=payload.message,
            files=payload.files,
        )
    except Exception as error:
        return await safe_send_json(ws, build_error_event(error))

    if not await safe_send_json(ws, build_status_event("Агент думает…")):
        return False

    try:
        use_streaming = await check_streaming_support(payload.model)
        if use_streaming:
            try:
                full_text, saved_files = await stream_response(
                    ws=ws,
                    messages=messages,
                    temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                    model=payload.model,
                )
            except Exception as stream_error:
                logger.warning("Стриминг не удался (%s), фоллбэк на обычный запрос", stream_error)
                if not await safe_send_json(ws, build_status_event("Переключаюсь на обычный режим…")):
                    return False
                full_text, saved_files = await sync_response(
                    ws=ws,
                    messages=messages,
                    temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                    model=payload.model,
                )
        else:
            full_text, saved_files = await sync_response(
                ws=ws,
                messages=messages,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
                model=payload.model,
            )
    except Exception as error:
        return await safe_send_json(ws, build_error_event(error))

    return await safe_send_json(
        ws,
        build_done_event(
            answer=full_text,
            model=payload.model,
            files=saved_files,
        ),
    )


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("WS connection open")

    try:
        while True:
            raw = await ws.receive_text()
            payload = validate_ws_payload(raw)

            if payload.kind == "ping":
                if not await safe_send_json(ws, {"type": "pong"}):
                    break
                continue

            if payload.error_message:
                if not await safe_send_json(ws, build_error_event(payload.error_message)):
                    break
                continue

            if payload.request is None:
                if not await safe_send_json(ws, build_error_event("Некорректный запрос")):
                    break
                continue

            if not await process_ws_message(ws, payload.request):
                break

    except WebSocketDisconnect:
        logger.info("WS connection closed")
    except Exception:
        logger.exception("WS fatal error")
