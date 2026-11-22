"""Клиент Ollama с дисковым кешем ответов.

Ключ кеша - модель, сообщения, схема ответа и параметры генерации: повторный запуск
эксперимента не вызывает модель, а изменение промпта даёт новый ключ.
"""

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"


@dataclass
class LLMResult:
    content: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    done_reason: str
    cached: bool


def chat(
    model: str,
    system: str,
    user: str,
    options: dict,
    schema: dict | str | None = None,
    cache_dir: Path | None = None,
    timeout: int = 3600,
    offline: bool = False,
) -> LLMResult | None:
    """Вызов модели; при offline=True только чтение кеша, без ответа в кеше - None."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": options,
        "keep_alive": "30m",
    }
    if schema is not None:
        body["format"] = schema
    key = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"{key}.json" if cache_dir else None
    if path and path.exists():
        return LLMResult(**json.loads(path.read_text(encoding="utf-8")) | {"cached": True})
    if offline:
        return None

    request = urllib.request.Request(  # noqa: S310 - только локальный адрес Ollama
        f"{OLLAMA_URL}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - локальный адрес
        data = json.load(response)
    result = LLMResult(
        content=data["message"]["content"],
        prompt_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        seconds=round(time.monotonic() - started, 1),
        done_reason=data.get("done_reason", ""),
        cached=False,
    )
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {k: v for k, v in result.__dict__.items() if k != "cached"}
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return result


def embed(texts: list[str], model: str = "bge-m3") -> list[list[float]]:
    body = json.dumps({"model": model, "input": texts}).encode()
    request = urllib.request.Request(  # noqa: S310 - только локальный адрес Ollama
        f"{OLLAMA_URL}/api/embed", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310 - локальный адрес
        return json.load(response)["embeddings"]
