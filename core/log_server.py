from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import json
import os

app = FastAPI(title="Log Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


class LogEntry(BaseModel):
    timestamp: str
    user_message: str
    assistant_response: str
    model: str = "unknown"
    response_time_ms: int = 0
    files: list = []
    tokens: Optional[dict] = None


def get_log_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"chat_{today}.jsonl")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/log")
async def save_log(entry: LogEntry):
    log_line = entry.dict()
    log_line["saved_at"] = datetime.now().isoformat()

    filepath = get_log_filename()
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

    return {"status": "saved"}


@app.get("/logs")
async def get_logs(date: Optional[str] = None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    filepath = os.path.join(LOG_DIR, f"chat_{date}.jsonl")

    if not os.path.exists(filepath):
        return {"date": date, "entries": [], "count": 0}

    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    return {"date": date, "entries": entries, "count": len(entries)}


@app.get("/logs/dates")
async def get_available_dates():
    dates = []
    if os.path.exists(LOG_DIR):
        for filename in sorted(os.listdir(LOG_DIR)):
            if filename.startswith("chat_") and filename.endswith(".jsonl"):
                date = filename.replace("chat_", "").replace(".jsonl", "")
                dates.append(date)
    return {"dates": dates}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)