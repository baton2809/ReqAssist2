"""Сборка корпуса и прогон LLM-конфигураций ДЗ 3 на локальной модели.

Примеры:
    uv run --group eda --group ml python scripts/hw3_run.py corpus
    uv run --group eda --group ml python scripts/hw3_run.py run --method mvp --runs 3
    uv run --group eda --group ml python scripts/hw3_run.py run --method lite --docs doc_04
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from reqassist.experiment import Paths, build_corpus, load_corpus, reparse, run_method

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("corpus")
    run = sub.add_parser("run")
    run.add_argument("--method", required=True, choices=["mvp", "lite", "product", "product_html"])
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--docs", nargs="*", help="id документов; по умолчанию весь корпус")
    again = sub.add_parser("reparse", help="разобрать ответы из кеша заново, без вызова модели")
    again.add_argument(
        "--method", required=True, choices=["mvp", "lite", "product", "product_html"]
    )
    args = parser.parse_args()

    paths = Paths(Path(os.environ.get("REQASSIST_DIR", ROOT.parent / "ReqAssist")), ROOT / "data")
    if args.command == "corpus":
        gold = json.loads((paths.data / "labels/gold.json").read_text(encoding="utf-8"))["labels"]
        manifest = build_corpus(paths, gold)
        print(f"документов в корпусе: {len(manifest)}")
        return

    docs = load_corpus(paths)
    if args.command == "reparse":
        print(f"записей пересобрано: {reparse(args.method, docs, paths)}")
        return
    if args.method == "product_html":
        docs = [d for d in docs if d.doc_id == d.source_id]
    if args.docs:
        docs = [d for d in docs if d.doc_id in args.docs]

    def log(message: str) -> None:
        print(f"{datetime.now():%H:%M:%S} {message}", flush=True)

    run_method(args.method, docs, paths, args.runs, log)


if __name__ == "__main__":
    sys.exit(main())
