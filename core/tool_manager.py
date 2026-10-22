"""
tool_manager.py — модуль управления тулзами для LLM-агента.

Отвечает за:
- Определение схем тулз (tools schema) для API
- Системный промпт с правилами использования тулз
- Обработку tool_calls из ответа модели
- Локальное сохранение сгенерированных файлов
"""

import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

# Папка, куда сохраняются сгенерированные файлы
GENERATED_FILES_DIR = "generated_files"

# Разрешённые расширения (защита от записи куда попало)
ALLOWED_EXTENSIONS = {
    ".html", ".css", ".js", ".json",
    ".py", ".ts", ".tsx", ".jsx",
    ".md", ".txt", ".yaml", ".yml",
    ".xml", ".svg", ".sh", ".bat",
    ".sql", ".env", ".cfg", ".ini",
    ".csv", ".log", ".jsonl"
}

# Максимальный размер файла (в символах) — защита от случайного мусора
MAX_FILE_SIZE = 500_000  # 500K символов

# ============================================================
# СИСТЕМНЫЙ ПРОМПТ ДЛЯ ТУЛЗ
# ============================================================

SYSTEM_PROMPT_TOOLS = """
## Правила работы с файлами

Ты — эксперт по разработке (вайбкодингу). У тебя есть инструмент save_generated_code.

### Когда ОБЯЗАТЕЛЬНО вызывать save_generated_code:
- Пользователь просит создать новый файл
- Пользователь просит переписать/обновить существующий файл
- Пользователь просит внести правки в код, и результат — это полный файл
- Объём кода в ответе превышает 30-40 строк

### Когда НЕ вызывать (отвечать текстом):
- Пользователь задаёт вопрос о коде (объяснение, ревью)
- Нужно показать маленький фрагмент/пример (до 30 строк)
- Пользователь явно просит "покажи в чате"

### Важные правила:
1. Если вызываешь save_generated_code — передавай ВЕСЬ файл целиком, а не фрагмент.
2. В текстовом ответе (content) кратко опиши, что было сделано и какие изменения внесены.
3. Не дублируй весь код в текстовом ответе, если уже вызвал функцию.
4. Имя файла (file_name) — только имя с расширением, без путей (например: "new_ui.html").
""".strip()

# ============================================================
# СХЕМА ТУЛЗ ДЛЯ API
# ============================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "save_generated_code",
            "description": (
                "Сохраняет сгенерированный или обновлённый код в файл на компьютере пользователя. "
                "Вызывай эту функцию, когда пользователь просит создать, обновить или переписать файл с кодом, "
                "или когда объём генерируемого кода превышает 30-40 строк."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": (
                            "Имя файла с расширением. Только имя, без пути. "
                            "Примеры: 'new_ui.html', 'styles.css', 'app.py'"
                        )
                    },
                    "file_content": {
                        "type": "string",
                        "description": "Полное содержимое файла. Весь код целиком, готовый к запуску."
                    },
                    "description": {
                        "type": "string",
                        "description": "Краткое описание: что это за файл и какие изменения были внесены."
                    }
                },
                "required": ["file_name", "file_content"]
            }
        }
    }
]

# ============================================================
# ОБРАБОТКА TOOL CALLS
# ============================================================

def handle_tool_calls(message: dict) -> dict:
    """
    Обрабатывает ответ модели, содержащий tool_calls.
    
    Args:
        message: объект message из response["choices"][0]["message"]
        
    Returns:
        dict с результатами:
        {
            "has_tool_calls": bool,
            "text_content": str | None,      # текстовый ответ модели (если есть)
            "files_saved": [                   # список сохранённых файлов
                {
                    "tool_call_id": str,
                    "file_name": str,
                    "file_path": str,
                    "description": str,
                    "success": bool,
                    "error": str | None
                }
            ]
        }
    """
    result = {
        "has_tool_calls": False,
        "text_content": message.get("content"),
        "files_saved": []
    }

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return result

    result["has_tool_calls"] = True

    for tool_call in tool_calls:
        func_name = tool_call.get("function", {}).get("name", "")
        tool_call_id = tool_call.get("id", "unknown")

        if func_name == "save_generated_code":
            file_result = _execute_save_generated_code(tool_call)
            file_result["tool_call_id"] = tool_call_id
            result["files_saved"].append(file_result)
        else:
            logger.warning(f"Неизвестная тулза: {func_name}")
            result["files_saved"].append({
                "tool_call_id": tool_call_id,
                "file_name": None,
                "file_path": None,
                "description": None,
                "success": False,
                "error": f"Неизвестная функция: {func_name}"
            })

    return result


def _execute_save_generated_code(tool_call: dict) -> dict:
    """
    Выполняет сохранение файла на диск.
    
    Возвращает dict с информацией о результате.
    """
    try:
        # Парсим аргументы
        raw_args = tool_call.get("function", {}).get("arguments", "{}")
        args = json.loads(raw_args)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга аргументов тулзы: {e}")
        return {
            "file_name": None,
            "file_path": None,
            "description": None,
            "success": False,
            "error": f"Невалидный JSON в аргументах: {e}"
        }

    file_name = args.get("file_name", "").strip()
    file_content = args.get("file_content", "")
    description = args.get("description", "Без описания")

    # --- Валидация ---

    # Проверка имени файла
    if not file_name:
        return _error_result("Пустое имя файла")

    # Защита от path traversal
    if "/" in file_name or "\\" in file_name or ".." in file_name:
        return _error_result(f"Недопустимое имя файла: {file_name}")

    # Проверка расширения
    _, ext = os.path.splitext(file_name)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return _error_result(
            f"Расширение '{ext}' не разрешено. "
            f"Допустимые: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Проверка размера
    if len(file_content) > MAX_FILE_SIZE:
        return _error_result(
            f"Файл слишком большой: {len(file_content)} символов "
            f"(максимум {MAX_FILE_SIZE})"
        )

    if not file_content.strip():
        return _error_result("Пустое содержимое файла")

    # --- Сохранение ---

    try:
        # Создаём папку если нет
        os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

        file_path = os.path.join(GENERATED_FILES_DIR, file_name)

        # Если файл существует — делаем бэкап
        if os.path.exists(file_path):
            _backup_file(file_path)

        # Пишем файл
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        abs_path = os.path.abspath(file_path)
        logger.info(f"Файл сохранён: {abs_path} ({len(file_content)} символов)")

        return {
            "file_name": file_name,
            "file_path": abs_path,
            "description": description,
            "success": True,
            "error": None
        }

    except OSError as e:
        logger.error(f"Ошибка записи файла {file_name}: {e}")
        return _error_result(f"Ошибка записи: {e}")


def _backup_file(file_path: str):
    """
    Создаёт резервную копию существующего файла перед перезаписью.
    Формат: filename.ext → filename.ext.bak.20261015_142300
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.bak.{timestamp}"
        os.rename(file_path, backup_path)
        logger.info(f"Бэкап создан: {backup_path}")
    except OSError as e:
        logger.warning(f"Не удалось создать бэкап {file_path}: {e}")


def _error_result(error_msg: str) -> dict:
    """Возвращает стандартный dict ошибки."""
    return {
        "file_name": None,
        "file_path": None,
        "description": None,
        "success": False,
        "error": error_msg
    }


# ============================================================
# ФОРМИРОВАНИЕ TOOL RESULTS ДЛЯ FOLLOW-UP ЗАПРОСА
# ============================================================

def build_tool_result_messages(tool_calls: list, handle_result: dict) -> list:
    """
    Формирует сообщения с результатами выполнения тулз
    для отправки обратно в API (если нужен follow-up).
    
    Это нужно, если модель вызвала тулзу и ждёт результат,
    чтобы сформировать финальный текстовый ответ.
    
    Args:
        tool_calls: список tool_calls из ответа модели
        handle_result: результат handle_tool_calls()
        
    Returns:
        list сообщений role=tool для добавления в messages
    """
    tool_messages = []

    for i, tool_call in enumerate(tool_calls):
        tool_call_id = tool_call.get("id", f"call_{i}")

        if i < len(handle_result["files_saved"]):
            file_info = handle_result["files_saved"][i]
            if file_info["success"]:
                content = json.dumps({
                    "status": "success",
                    "file_name": file_info["file_name"],
                    "file_path": file_info["file_path"],
                    "message": f"Файл {file_info['file_name']} успешно сохранён"
                }, ensure_ascii=False)
            else:
                content = json.dumps({
                    "status": "error",
                    "error": file_info["error"]
                }, ensure_ascii=False)
        else:
            content = json.dumps({
                "status": "error",
                "error": "Результат обработки не найден"
            }, ensure_ascii=False)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

    return tool_messages