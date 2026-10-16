# file_reader.py
"""
Утилита для чтения файлов различных текстовых форматов.
Извлекает текстовое содержимое для последующей передачи в LLM.
"""

import os
from pathlib import Path
from typing import Optional


# Маппинг расширений -> MIME-категория для отображения
SUPPORTED_EXTENSIONS: dict[str, str] = {
    # Текстовые документы
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
    ".rst": "text",
    ".log": "text",
    ".csv": "text",
    ".tsv": "text",
    ".json": "text",
    ".xml": "text",
    ".yaml": "text",
    ".yml": "text",
    ".toml": "text",
    ".ini": "text",
    ".cfg": "text",
    ".conf": "text",
    ".env": "text",
    ".properties": "text",

    # Код
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".jsx": "code",
    ".tsx": "code",
    ".html": "code",
    ".htm": "code",
    ".css": "code",
    ".scss": "code",
    ".sass": "code",
    ".less": "code",
    ".java": "code",
    ".c": "code",
    ".cpp": "code",
    ".cc": "code",
    ".h": "code",
    ".hpp": "code",
    ".cs": "code",
    ".go": "code",
    ".rs": "code",
    ".rb": "code",
    ".php": "code",
    ".swift": "code",
    ".kt": "code",
    ".kts": "code",
    ".scala": "code",
    ".r": "code",
    ".R": "code",
    ".lua": "code",
    ".perl": "code",
    ".pl": "code",
    ".sh": "code",
    ".bash": "code",
    ".zsh": "code",
    ".fish": "code",
    ".bat": "code",
    ".cmd": "code",
    ".ps1": "code",
    ".sql": "code",
    ".graphql": "code",
    ".proto": "code",
    ".dart": "code",
    ".vue": "code",
    ".svelte": "code",

    # Разметка / конфиги проектов
    ".dockerfile": "code",
    ".dockerignore": "text",
    ".gitignore": "text",
    ".editorconfig": "text",
    ".eslintrc": "text",
    ".prettierrc": "text",
    ".makefile": "code",
}

# Файлы без расширения, которые мы тоже умеем читать
SUPPORTED_FILENAMES: dict[str, str] = {
    "Dockerfile": "code",
    "Makefile": "code",
    "Vagrantfile": "code",
    "Jenkinsfile": "code",
    "Procfile": "text",
    ".gitignore": "text",
    ".dockerignore": "text",
    ".env": "text",
    ".env.local": "text",
    ".env.example": "text",
}

# Лимит размера файла (в байтах) — 1 МБ
MAX_FILE_SIZE: int = 1 * 1024 * 1024

# Кодировки для попытки декодирования (по приоритету)
ENCODINGS_TO_TRY: list[str] = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]


class FileReadError(Exception):
    """Ошибка при чтении файла."""
    def __init__(self, message: str, filename: str = ""):
        self.filename = filename
        super().__init__(message)


class FileReadResult:
    """Результат чтения файла."""

    def __init__(
        self,
        filename: str,
        content: str,
        extension: str,
        category: str,
        size_bytes: int,
    ):
        self.filename = filename
        self.content = content
        self.extension = extension
        self.category = category  # "text" | "code"
        self.size_bytes = size_bytes

    def to_prompt_block(self) -> str:
        """
        Форматирует содержимое файла в блок для вставки в промпт LLM.
        """
        tick = chr(96)
        fence = tick * 3
        lang_hint = self.extension.lstrip(".") if self.category == "code" else ""
        fence_open = fence + lang_hint

        header = (
            "\U0001f4ce Файл: "
            + tick + self.filename + tick
            + " ("
            + self.category
            + ", "
            + self._human_size()
            + ")"
        )

        lines = [
            header,
            fence_open,
            self.content,
            fence,
        ]

        return "\n".join(lines)
    
    def _human_size(self) -> str:
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        elif self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        else:
            return f"{self.size_bytes / (1024 * 1024):.1f} MB"

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "extension": self.extension,
            "category": self.category,
            "size_bytes": self.size_bytes,
            "content": self.content,
        }
        
def is_supported(filename: str) -> bool:
    """Проверяет, поддерживается ли файл для чтения."""
    name = Path(filename).name
    ext = Path(filename).suffix.lower()

    if name in SUPPORTED_FILENAMES:
        return True
    if ext in SUPPORTED_EXTENSIONS:
        return True
    return False


def get_supported_extensions() -> list[str]:
    """Возвращает список поддерживаемых расширений (для фронтенда)."""
    exts = sorted(set(SUPPORTED_EXTENSIONS.keys()))
    return exts


def _detect_category(filename: str) -> str:
    """Определяет категорию файла."""
    name = Path(filename).name
    ext = Path(filename).suffix.lower()

    if name in SUPPORTED_FILENAMES:
        return SUPPORTED_FILENAMES[name]
    return SUPPORTED_EXTENSIONS.get(ext, "text")


def _decode_bytes(raw: bytes, filename: str) -> str:
    """Пытается декодировать байты в строку, перебирая кодировки."""
    for encoding in ENCODINGS_TO_TRY:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue

    raise FileReadError(
        f"Не удалось декодировать файл '{filename}'. "
        f"Возможно, это бинарный файл.",
        filename=filename,
    )


def read_file_from_bytes(
    raw: bytes,
    filename: str,
    max_size: Optional[int] = None,
) -> FileReadResult:
    """
    Читает файл из байтов (полученных через upload).
    """
    limit = max_size or MAX_FILE_SIZE

    if not is_supported(filename):
        ext = Path(filename).suffix
        raise FileReadError(
            f"Формат '{ext or 'без расширения'}' не поддерживается. "
            f"Поддерживаемые: текстовые файлы и файлы с кодом.",
            filename=filename,
        )

    size = len(raw)
    if size == 0:
        raise FileReadError(
            f"Файл '{filename}' пуст.",
            filename=filename,
        )

    if size > limit:
        human_limit = f"{limit / (1024 * 1024):.1f} MB"
        raise FileReadError(
            f"Файл '{filename}' слишком большой ({size} байт). "
            f"Максимум: {human_limit}.",
            filename=filename,
        )

    content = _decode_bytes(raw, filename)
    ext = Path(filename).suffix.lower()
    category = _detect_category(filename)

    return FileReadResult(
        filename=filename,
        content=content,
        extension=ext,
        category=category,
        size_bytes=size,
    )


def read_file_from_path(
    filepath: str | Path,
    max_size: Optional[int] = None,
) -> FileReadResult:
    """
    Читает файл с диска.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileReadError(
            f"Файл не найден: {path}",
            filename=path.name,
        )

    if not path.is_file():
        raise FileReadError(
            f"Не является файлом: {path}",
            filename=path.name,
        )

    raw = path.read_bytes()
    return read_file_from_bytes(raw, path.name, max_size)


def format_files_for_prompt(
    files: list[FileReadResult],
    user_message: str,
) -> str:
    """
    Собирает финальный промпт: текст пользователя + вложенные файлы.
    """
    if not files:
        return user_message

    parts = []
    parts.append("Пользователь приложил файлы:\n")

    for f in files:
        parts.append(f.to_prompt_block())

    parts.append("")
    parts.append(
        f"Сообщение пользователя:\n{user_message}"
    )

    return "\n\n".join(parts)