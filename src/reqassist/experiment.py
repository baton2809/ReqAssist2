"""Корпус ДЗ 3 и прогон LLM-конфигураций на локальной модели.

Корпус: 5 согласованных БТ после нормализации и по 4 испорченные копии каждого.
Тексты, цитаты и ответы модели лежат в data/ и в git не попадают.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from reqassist.data.checklist import ITEM_SOURCE, load_item_texts
from reqassist.data.corrupt import corrupt
from reqassist.data.normalize import NormDoc, load_normalized
from reqassist.evidence import quote_supported
from reqassist.llm import validators
from reqassist.llm.ollama import chat

ITEMS = [i for i, src in ITEM_SOURCE.items() if src == "bt_text"]
MODEL = "gemma3:12b"


@dataclass
class Paths:
    product: Path  # рабочий репозиторий продукта (REQASSIST_DIR)
    data: Path  # локальная папка data/ этого репозитория

    @property
    def documents(self) -> Path:
        return self.product / "test-fixtures/br-documents"

    @property
    def checklist(self) -> Path:
        return self.product / "public/checklists/v1.md"


def build_corpus(paths: Paths, gold: dict) -> list[dict]:
    """Нормализует оригиналы, строит испорченные копии и сохраняет их тексты локально."""
    out = paths.data / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for doc in load_normalized(paths.documents):
        entries = [(doc, "original", "", None)]
        entries += [
            (c.doc, c.operation, c.section_title, c.removed) for c in corrupt(doc, gold[doc.doc_id])
        ]
        for d, operation, title, removed in entries:
            (out / f"{d.doc_id}.md").write_text(d.text, encoding="utf-8")
            manifest.append(
                {
                    "doc_id": d.doc_id,
                    "source_id": d.source_id,
                    "operation": operation,
                    "section": title,
                    "removed": removed,
                    "sha256": d.sha256,
                }
            )
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest


def load_corpus(paths: Paths) -> list[NormDoc]:
    manifest = json.loads((paths.data / "corpus/manifest.json").read_text(encoding="utf-8"))
    return [
        NormDoc(
            m["doc_id"],
            m["source_id"],
            (paths.data / f"corpus/{m['doc_id']}.md").read_text(encoding="utf-8"),
        )
        for m in manifest
    ]


def raw_html(paths: Paths, doc_id: str) -> str:
    """HTML контента страницы, как его получает продукт (для абляции нормализации)."""
    files = sorted(paths.documents.glob("*.html"))
    soup = BeautifulSoup(files[int(doc_id.split("_")[1]) - 1].read_text(encoding="utf-8"), "lxml")
    return str(soup.select_one("#main-content"))


def _product_note(paths: Paths) -> str:
    source = (paths.product / "back/requirements/validation.js").read_text(encoding="utf-8")
    return re.search(r": '(Используется стандартный чек-лист[^']*)'", source).group(1)


def _prompt(method: str, paths: Paths) -> tuple[str, dict | str, tuple[str, str]]:
    """Системный промпт, схема ответа и (чеклист, пометка) для промпта продукта."""
    item_texts = load_item_texts(paths.checklist, ITEMS)
    if method == "mvp":
        return (*validators.mvp_prompt(item_texts), ("", ""))
    if method == "lite":
        return (*validators.lite_prompt(item_texts), ("", ""))
    system = (paths.product / "public/prompts/validation/v1.txt").read_text(encoding="utf-8")
    return system, "json", (paths.checklist.read_text(encoding="utf-8"), _product_note(paths))


def _request(method: str, doc: NormDoc, run: int, paths: Paths, product: tuple[str, str]):
    """Сообщение пользователя и параметры генерации для одного вызова."""
    options = {"temperature": 0.1, "seed": run, "num_ctx": 16384, "num_predict": 3000}
    if method in {"mvp", "lite"}:
        return validators.document_message(doc.text), options
    document = raw_html(paths, doc.doc_id) if method == "product_html" else doc.text
    options |= {"num_ctx": 32768, "num_predict": 6000}
    return validators.product_message(document, *product), options


def _record(method: str, doc: NormDoc, run: int, result) -> dict:
    parse = validators.parse_structured if method in {"mvp", "lite"} else validators.parse_product
    try:
        items, error = parse(result.content, ITEMS), ""
    except (json.JSONDecodeError, AttributeError) as exc:
        items, error = {}, type(exc).__name__
    for value in items.values():
        value["quote_ok"] = quote_supported(value["quote"], doc.text) if value["quote"] else None
    return {
        "method": method,
        "doc_id": doc.doc_id,
        "source_id": doc.source_id,
        "run": run,
        "items": items,
        "parse_error": error,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "seconds": result.seconds,
        "done_reason": result.done_reason,
        "cached": result.cached,
    }


def run_method(method: str, docs: list[NormDoc], paths: Paths, runs: int, log) -> None:
    """Прогон одной конфигурации; результаты дописываются в data/runs/<method>.jsonl."""
    cache = paths.data / "cache/llm"
    out = paths.data / f"runs/{method}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {
            (r["doc_id"], r["run"])
            for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())
        }
    system, schema, product = _prompt(method, paths)
    for doc in docs:
        for run in range(1, runs + 1):
            if (doc.doc_id, run) in done:
                continue
            user, options = _request(method, doc, run, paths, product)
            result = chat(MODEL, system, user, options, schema, cache)
            record = _record(method, doc, run, result)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            tokens = f"in={result.prompt_tokens} out={result.output_tokens}"
            error = record["parse_error"]
            log(f"{method} {doc.doc_id} run={run} {result.seconds}s {tokens} {error}")


def reparse(method: str, docs: list[NormDoc], paths: Paths) -> int:
    """Пересобирает data/runs/<method>.jsonl из кеша ответов после правки разбора.

    Модель не вызывается: пара документ-запуск без ответа в кеше пропускается.
    """
    cache = paths.data / "cache/llm"
    out = paths.data / f"runs/{method}.jsonl"
    old = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    by_id = {d.doc_id: d for d in docs}
    system, schema, product = _prompt(method, paths)
    records = []
    for r in old:
        doc = by_id[r["doc_id"]]
        user, options = _request(method, doc, r["run"], paths, product)
        result = chat(MODEL, system, user, options, schema, cache, offline=True)
        if result is None:
            records.append(r)
            continue
        records.append(_record(method, doc, r["run"], result) | {"cached": r["cached"]})
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return len(records)
