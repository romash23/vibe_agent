import json
import logging
import os
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from file_reader import (
    FileReadError,
    FileReadResult,
    format_files_for_prompt,
    get_supported_extensions,
    read_file_from_bytes,
)

from tool_manager import (
    TOOLS_SCHEMA,
    SYSTEM_PROMPT_TOOLS,
    handle_tool_calls,
    build_tool_result_messages,
)

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

API_URL = "https://foundation-models.api.cloud.ru/v1/chat/completions"
API_KEY = os.getenv("API_KEY")
DEFAULT_MODEL = os.getenv(
    "CLOUD_MODEL",
    "anthropic/claude-opus-4.6",
)

if not API_KEY:
    raise RuntimeError(
        "Не найден CLOUD_API_KEY. Добавь его в файл .env"
    )


app = FastAPI(
    title="Local Claude API Server",
    version="1.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def chat_page():
    return """
<!doctype html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Chat</title>

    <style>
        :root {
            color-scheme: dark;
            font-family: Arial, sans-serif;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            background: #171717;
            color: #f2f2f2;
        }

        .app {
            display: flex;
            flex-direction: column;
            height: 100vh;
            max-width: 1000px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 22px;
            border-bottom: 1px solid #333;
        }

        header h1 {
            margin: 0;
            font-size: 20px;
        }

        .model {
            color: #aaa;
            font-size: 13px;
        }

        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

        .message {
            display: flex;
            margin-bottom: 18px;
        }

        .message.user {
            justify-content: flex-end;
        }

        .bubble {
            max-width: 80%;
            padding: 12px 15px;
            border-radius: 14px;
            white-space: pre-wrap;
            line-height: 1.5;
            overflow-wrap: anywhere;
        }

        .user .bubble {
            background: #315f9d;
        }

        .assistant .bubble {
            background: #292929;
        }

        .error .bubble {
            background: #722d2d;
        }

        .empty {
            max-width: 600px;
            margin: 80px auto;
            text-align: center;
            color: #999;
            line-height: 1.6;
        }

        form {
            display: flex;
            gap: 10px;
            padding: 16px 22px 22px;
            border-top: 1px solid #333;
        }

        textarea {
            flex: 1;
            min-height: 52px;
            max-height: 180px;
            padding: 13px;
            resize: vertical;
            border: 1px solid #444;
            border-radius: 10px;
            background: #242424;
            color: #fff;
            font: inherit;
        }

        button {
            align-self: stretch;
            min-width: 100px;
            border: 0;
            border-radius: 10px;
            background: #4e82c2;
            color: #fff;
            cursor: pointer;
            font: inherit;
        }

        button:hover {
            background: #6196d6;
        }

        button:disabled,
        textarea:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .typing {
            color: #aaa;
            font-size: 14px;
            margin: 0 0 14px 4px;
        }

        @media (max-width: 600px) {
            .bubble {
                max-width: 92%;
            }

            form {
                padding: 12px;
            }

            #messages {
                padding: 16px 12px;
            }

            header {
                padding: 14px;
            }
        }
    </style>
</head>

<body>
    <main class="app">
        <header>
            <h1>Локальный чат</h1>
            <div class="model" id="modelName">Модель из .env</div>
        </header>

        <section id="messages">
            <div class="empty" id="emptyMessage">
                Напишите сообщение, чтобы начать диалог.
            </div>
        </section>

        <form id="chatForm">
            <textarea
                id="messageInput"
                placeholder="Введите сообщение..."
                rows="2"
                autofocus
            ></textarea>

            <button id="sendButton" type="submit">
                Отправить
            </button>
        </form>
    </main>

    <script>
        const form = document.getElementById("chatForm");
        const input = document.getElementById("messageInput");
        const sendButton = document.getElementById("sendButton");
        const messagesElement = document.getElementById("messages");
        const emptyMessage = document.getElementById("emptyMessage");
        const modelName = document.getElementById("modelName");

        const messages = [];

        function addMessage(role, content, type = role) {
            emptyMessage.style.display = "none";

            const messageElement = document.createElement("div");
            messageElement.className = `message ${type}`;

            const bubble = document.createElement("div");
            bubble.className = "bubble";
            bubble.textContent = content;

            messageElement.appendChild(bubble);
            messagesElement.appendChild(messageElement);
            messagesElement.scrollTop = messagesElement.scrollHeight;

            return messageElement;
        }

        function setLoading(isLoading) {
            input.disabled = isLoading;
            sendButton.disabled = isLoading;
            sendButton.textContent = isLoading ? "Ждите..." : "Отправить";
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            const text = input.value.trim();

            if (!text || sendButton.disabled) {
                return;
            }

            addMessage("user", text);
            messages.push({
                role: "user",
                content: text
            });

            input.value = "";
            setLoading(true);

            const typingElement = addMessage(
                "assistant",
                "Модель печатает...",
                "assistant"
            );
            typingElement.classList.add("typing");

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        messages: messages,
                        temperature: 0.7,
                        max_tokens: 2048
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(
                        data.detail || "Сервер вернул ошибку"
                    );
                }

                typingElement.remove();

                addMessage("assistant", data.answer);

                messages.push({
                    role: "assistant",
                    content: data.answer
                });

                if (data.model) {
                    modelName.textContent = data.model;
                }
            } catch (error) {
                typingElement.remove();
                addMessage(
                    "error",
                    `Ошибка: ${error.message}`,
                    "error"
                );
            } finally {
                setLoading(false);
                input.focus();
            }
        });

        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                form.requestSubmit();
            }
        });
    </script>
</body>
</html>
"""


# --------------- Pydantic-модели ---------------

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, gt=0, le=100000)


class ChatResponse(BaseModel):
    answer: str
    model: str | None = None
    usage: dict | None = None
    files_saved: list[dict] | None = None  # НОВОЕ: инфо о сохранённых файлах


# --------------- Вспомогательная функция ---------------

async def _call_cloud_api(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    use_tools: bool = True,
) -> dict:
    """
    Отправляет messages в Cloud.ru API и возвращает сырой JSON-ответ.
    Вынесено, чтобы оба эндпоинта использовали одну логику.

    Args:
        messages: список сообщений
        model: модель (или DEFAULT_MODEL)
        temperature: температура генерации
        max_tokens: максимум токенов
        use_tools: передавать ли tools в запрос
    """
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }

    # Добавляем тулзы в запрос
    if use_tools:
        payload["tools"] = TOOLS_SCHEMA
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                API_URL,
                headers=headers,
                json=payload,
            )

        if response.is_error:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text,
            )

        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise HTTPException(
                status_code=502,
                detail=f"Cloud.ru вернул неожиданный ответ: {data}",
            )

        return data

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка соединения с Cloud.ru: {error}",
        )


async def _process_response(
    data: dict,
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict:
    """
    Обрабатывает ответ Cloud.ru API.
    Если модель вернула tool_calls — выполняет их и делает follow-up запрос.
    Если обычный текст — возвращает как есть.

    Returns:
        dict: {"answer": str, "model": str, "usage": dict, "files_saved": list}
    """
    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls")

    # --- Обычный текстовый ответ (без tool_calls) ---
    if not tool_calls:
        answer = message.get("content", "")
        if not answer:
            raise HTTPException(
                status_code=502,
                detail=f"В ответе отсутствует текст: {data}",
            )
        return {
            "answer": answer,
            "model": data.get("model"),
            "usage": data.get("usage"),
            "files_saved": [],
        }

    # --- Модель вызвала тулзу ---
    logger.info(f"Модель вызвала {len(tool_calls)} тулз(ы)")

    # 1. Выполняем тулзы локально (сохраняем файлы)
    tool_result = handle_tool_calls(message)

    for f in tool_result["files_saved"]:
        if f["success"]:
            logger.info(f"✅ Файл сохранён: {f['file_name']} → {f['file_path']}")
        else:
            logger.error(f"❌ Ошибка сохранения: {f['error']}")

    # 2. Формируем follow-up: отправляем результаты тулз обратно модели,
    #    чтобы она сгенерировала финальный текстовый ответ для пользователя
    followup_messages = messages.copy()

    # Добавляем ответ ассистента с tool_calls (как его вернул API)
    followup_messages.append(message)

    # Добавляем результаты выполнения тулз
    tool_result_msgs = build_tool_result_messages(tool_calls, tool_result)
    followup_messages.extend(tool_result_msgs)

    # 3. Делаем повторный запрос БЕЗ тулз, чтобы получить текстовый ответ
    followup_data = await _call_cloud_api(
        messages=followup_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        use_tools=False,  # Без тулз — нужен только текст
    )

    followup_message = followup_data["choices"][0]["message"]
    answer = followup_message.get("content", "")

    # Если модель вернула текст вместе с tool_calls (некоторые модели так делают)
    # и follow-up пустой — используем оригинальный текст
    if not answer and tool_result.get("text_content"):
        answer = tool_result["text_content"]

    if not answer:
        # Крайний случай: собираем ответ из информации о файлах
        saved_names = [
            f["file_name"] for f in tool_result["files_saved"] if f["success"]
        ]
        if saved_names:
            answer = f"Файлы сохранены: {', '.join(saved_names)}"
        else:
            answer = "Модель вызвала функцию, но не вернула текстовый ответ."

    return {
        "answer": answer,
        "model": followup_data.get("model") or data.get("model"),
        "usage": followup_data.get("usage"),
        "files_saved": tool_result["files_saved"],
    }


def _inject_system_prompt(messages: list[dict]) -> list[dict]:
    """
    Добавляет системный промпт с правилами тулз в начало messages.
    Если системный промпт уже есть — дополняет его.
    Не мутирует оригинальный список.
    """
    messages = [msg.copy() for msg in messages]

    if messages and messages[0].get("role") == "system":
        # Дополняем существующий системный промпт
        messages[0]["content"] = (
            messages[0]["content"] + "\n\n" + SYSTEM_PROMPT_TOOLS
        )
    else:
        # Вставляем новый системный промпт в начало
        messages.insert(0, {
            "role": "system",
            "content": SYSTEM_PROMPT_TOOLS,
        })

    return messages


# --------------- Эндпоинты ---------------

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/models")
async def models():
    """Показывает модели, доступные через Cloud.ru."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://foundation-models.api.cloud.ru/v1/models",
                headers=headers,
            )

        if response.is_error:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text,
            )

        return response.json()

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка соединения с Cloud.ru: {error}",
        )


@app.get("/supported-formats")
async def supported_formats():
    """Возвращает список поддерживаемых расширений файлов."""
    return {"extensions": get_supported_extensions()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    JSON-эндпоинт (без файлов).
    Используется встроенным UI на главной странице.
    """
    messages = [
        msg.model_dump()
        for msg in request.messages
    ]

    # Внедряем системный промпт с правилами тулз
    messages = _inject_system_prompt(messages)

    # Отправляем в Cloud.ru (с тулзами)
    data = await _call_cloud_api(
        messages=messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        use_tools=True,
    )

    # Обрабатываем ответ (текст или tool_calls)
    result = await _process_response(
        data=data,
        messages=messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    return ChatResponse(
        answer=result["answer"],
        model=result["model"],
        usage=result["usage"],
        files_saved=result["files_saved"] or None,
    )


@app.post("/chat-with-files")
async def chat_with_files(
    message: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    history: str = Form(default="[]"),
    temperature: float = Form(default=0.7),
    max_tokens: int = Form(default=2048),
):
    """
    Multipart-эндпоинт (с файлами).
    Используется новым UI с поддержкой вложений.
    """
    # 1. Читаем вложенные файлы
    read_files: list[FileReadResult] = []

    for upload_file in files:
        try:
            raw = await upload_file.read()
            result = read_file_from_bytes(
                raw,
                upload_file.filename or "unknown",
            )
            read_files.append(result)
        except FileReadError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )
        finally:
            await upload_file.close()

    # 2. Формируем промпт с файлами
    user_content = format_files_for_prompt(read_files, message)

    # 3. Парсим историю диалога
    try:
        conversation_history = json.loads(history)
    except json.JSONDecodeError:
        conversation_history = []

    # 4. Собираем messages + системный промпт с тулзами
    messages = conversation_history + [
        {"role": "user", "content": user_content}
    ]
    messages = _inject_system_prompt(messages)

    # 5. Отправляем в Cloud.ru (с тулзами)
    data = await _call_cloud_api(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        use_tools=True,
    )

    # 6. Обрабатываем ответ (текст или tool_calls)
    result = await _process_response(
        data=data,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 7. Возвращаем ответ
    return {
        "answer": result["answer"],
        "model": result["model"],
        "usage": result["usage"],
        "files_saved": result["files_saved"] or [],
        "attached_files": [
            {
                "filename": f.filename,
                "category": f.category,
                "size_bytes": f.size_bytes,
            }
            for f in read_files
        ],
    }