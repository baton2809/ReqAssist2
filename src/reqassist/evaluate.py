"""Оценка бейзлайнов и основной модели ДЗ 3.

Эталон: разметка оригиналов (data/labels/gold.json) и метки испорченных копий, выведенные
из тех же цитат. Срезы:

- original - пункты оригиналов; original_dev и original_heldout - отдельно БТ, на которых
  подбирался промпт MVP, и остальные
- affected - пункты копий, изменённые порчей
- eval - original и affected вместе: уникальные пары без повторов нетронутых пунктов
- нетронутые пункты копий в метрики качества не входят: по ним считается, меняет ли модель
  ответ без причины (unaffected_consistency)
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from reqassist.baselines import b05_majority, b1_headings, chunks, features, normalize_rows
from reqassist.data.checklist import load_item_texts
from reqassist.data.corrupt import _relabel
from reqassist.data.normalize import NormDoc
from reqassist.experiment import ITEMS, Paths, load_corpus
from reqassist.metrics import (
    ORDINAL,
    bootstrap_ci,
    completeness_score,
    false_completed_rate,
    macro_f1,
    missing_recall,
    paired_bootstrap_diff,
    qwk,
)

LLM_METHODS = {"mvp": "MVP", "lite": "B3-lite", "product": "B3", "product_html": "B3 на HTML"}
# окно контекста, с которым запускалась конфигурация (experiment.run_method)
NUM_CTX = {"mvp": 16384, "lite": 16384, "product": 32768, "product_html": 32768}
# исходные БТ, на которых подбирался промпт MVP
DEV_SOURCES = ("doc_02", "doc_04")
SLICES = ("original", "original_dev", "original_heldout", "affected", "eval")


def build_gold(paths: Paths, docs: list[NormDoc]) -> dict[str, dict[str, dict]]:
    """doc_id -> пункт -> {status, slice}."""
    labels = json.loads((paths.data / "labels/gold.json").read_text(encoding="utf-8"))["labels"]
    manifest = {
        m["doc_id"]: m
        for m in json.loads((paths.data / "corpus/manifest.json").read_text(encoding="utf-8"))
    }
    originals = {d.doc_id: d for d in docs if d.doc_id == d.source_id}
    gold: dict[str, dict[str, dict]] = {}
    for doc in docs:
        src = labels[doc.source_id]
        if doc.doc_id == doc.source_id:
            gold[doc.doc_id] = {i: {"status": src[i]["status"], "slice": "original"} for i in ITEMS}
            continue
        m = manifest[doc.doc_id]
        new, _ = _relabel(
            originals[doc.source_id],
            src,
            tuple(m["removed"]),
            m["operation"].startswith("placeholder"),
        )
        gold[doc.doc_id] = {
            i: {
                "status": new[i],
                "slice": "affected" if new[i] != src[i]["status"] else "unaffected",
            }
            for i in ITEMS
        }
    return gold


def load_llm_runs(paths: Paths, method: str) -> dict[int, dict[str, dict]]:
    """run -> doc_id -> запись прогона."""
    path = paths.data / f"runs/{method}.jsonl"
    runs: dict[int, dict[str, dict]] = {}
    if path.exists():
        for record in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            runs.setdefault(record["run"], {})[record["doc_id"]] = record
    return runs


def quote_verified(value: dict) -> bool:
    """Цитата для completed и partial есть и найдена в тексте документа."""
    return bool(value.get("quote")) and value.get("quote_ok") is True


def with_quote_check(statuses: dict[str, dict]) -> dict[str, str]:
    """Пост-правило: completed и partial без подтверждённой цитаты понижаются до missing."""
    out = {}
    for item, v in statuses.items():
        status = v["status"]
        if status in {"completed", "partial"} and not quote_verified(v):
            status = "missing"
        out[item] = status
    return out


def _embed_cached(cache: Path, texts: list[str]) -> np.ndarray:
    """Эмбеддинги bge-m3 с кешем по хешу набора текстов: оценка повторяется без Ollama."""
    key = hashlib.sha256("\n\x00".join(texts).encode()).hexdigest()
    path = cache / f"{key}.npy"
    if path.exists():
        return np.load(path)
    from reqassist.llm.ollama import embed

    vecs = normalize_rows(np.array(embed(texts)))
    np.save(path, vecs)
    return vecs


def b2_predict(paths: Paths, docs: list[NormDoc], gold: dict) -> dict[str, dict[str, str]]:
    """B2: признаки сходства пункта с фрагментами документа (bge-m3) и логрегрессия.

    Кросс-валидация по исходному документу: оригинал и его копии целиком в одном фолде.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cache = paths.data / "cache/emb"
    cache.mkdir(parents=True, exist_ok=True)
    item_texts = load_item_texts(paths.checklist, ITEMS)
    item_vecs = _embed_cached(cache, [item_texts[i] for i in ITEMS])
    rows, meta = [], []
    for doc in docs:
        parts = chunks(doc)
        cached = cache / f"{doc.sha256}.npy"
        if cached.exists():
            vecs = np.load(cached)
        else:
            vecs = _embed_cached(cache, [t for t, _ in parts])
            np.save(cached, vecs)
        b1 = b1_headings(doc, ITEMS)
        for k, item in enumerate(ITEMS):
            rows.append(features(item_vecs[k], vecs, parts, b1[item] != "missing", len(doc.text)))
            meta.append((doc.doc_id, doc.source_id, item))
    x = np.array(rows)
    predictions: dict[str, dict[str, str]] = {d.doc_id: {} for d in docs}
    sources = sorted({d.source_id for d in docs})
    for held_out in sources:
        train = [
            k
            for k, (doc_id, src, item) in enumerate(meta)
            if src != held_out and gold[doc_id][item]["status"] != "not_applicable"
        ]
        y = [gold[meta[k][0]][meta[k][2]]["status"] for k in train]
        y = ["partial" if s == "partial_placeholder" else s for s in y]
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        )
        model.fit(x[train], y)
        test = [k for k, (_, src, _) in enumerate(meta) if src == held_out]
        for k, label in zip(test, model.predict(x[test]), strict=True):
            doc_id, _, item = meta[k]
            predictions[doc_id][item] = (
                "partial_placeholder" if x[k][-1] and label != "completed" else label
            )
    return predictions


def _in_slice(slice_name: str, g: dict, source_id: str) -> bool:
    if slice_name == "eval":
        return g["slice"] in {"original", "affected"}
    if slice_name == "original_dev":
        return g["slice"] == "original" and source_id in DEV_SOURCES
    if slice_name == "original_heldout":
        return g["slice"] == "original" and source_id not in DEV_SOURCES
    return g["slice"] == slice_name


def _pairs(run: dict, gold: dict, by_id: dict, slice_name: str) -> list[tuple[str, str, str]]:
    """(source_id, предсказание, эталон) для пунктов среза; n/a по разметке исключены."""
    return [
        (by_id[doc_id].source_id, run[doc_id][i], g["status"])
        for doc_id, items in gold.items()
        if doc_id in run
        for i, g in items.items()
        if g["status"] in ORDINAL and _in_slice(slice_name, g, by_id[doc_id].source_id)
    ]


def _groups(pairs: list[tuple[str, str, str]]) -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list] = {}
    for src, p, e in pairs:
        groups.setdefault(src, []).append((p, e))
    return groups


def _metric_block(pairs: list[tuple[str, str, str]]) -> dict:
    """Метрики одного запуска на одном срезе."""
    if not pairs:
        return {"n": 0}
    pred, exp = [p[1] for p in pairs], [p[2] for p in pairs]
    block = {
        "n": len(pairs),
        "qwk": qwk(pred, exp),
        "macro_f1": macro_f1(pred, exp),
        # not_applicable на пункте, который по разметке применим
        "na_on_applicable": sum(p == "not_applicable" for p in pred) / len(pred),
    }
    if "missing" in exp:
        block["missing_recall"] = missing_recall(pred, exp)
        block["n_missing"] = exp.count("missing")
    if any(e != "completed" for e in exp):
        block["false_completed"] = false_completed_rate(pred, exp)
        block["n_gaps"] = sum(e != "completed" for e in exp)
    groups = _groups(pairs)
    if len(groups) > 1:
        block["qwk_by_source"] = {
            s: qwk([p for p, _ in g], [e for _, e in g]) for s, g in groups.items()
        }
    return block


def _average_runs(blocks: list[dict]) -> dict:
    """Среднее метрик по запускам; значения каждого запуска - в ключе <метрика>_runs."""
    blocks = [b for b in blocks if b.get("n")] or [{"n": 0}]
    if len(blocks) == 1:
        return blocks[0]
    out: dict = {}
    for key in blocks[0]:
        values = [b[key] for b in blocks if key in b]
        if key in {"n", "n_missing", "n_gaps"}:
            out[key] = values[0]
        elif isinstance(values[0], dict):
            out[key] = {}
            for k in values[0]:
                arr = [v[k] for v in values if v.get(k) == v.get(k)]
                out[key][k] = mean(arr) if arr else float("nan")
        else:
            arr = [v for v in values if v == v]
            out[key] = mean(arr) if arr else float("nan")
            out[key + "_runs"] = values
    return out


def _llm_meta(key: str, runs: dict[int, dict[str, dict]], n_docs: int) -> dict:
    records = [r for run in runs.values() for r in run.values()]
    seconds = sorted(r["seconds"] for r in records)
    meta = {
        "calls": len(records),
        # документов с разобранным ответом в каждом запуске, из n_docs
        "coverage": [sum(bool(r["items"]) for r in run.values()) for run in runs.values()],
        "n_docs": n_docs,
        "parse_errors": sum(bool(r["parse_error"]) for r in records),
        "truncated": sum(r["done_reason"] == "length" for r in records),
        "context_overflow": sum(
            r["prompt_tokens"] + r["output_tokens"] >= NUM_CTX[key] for r in records
        ),
        "seconds_mean": round(mean(seconds), 1),
        "seconds_p95": seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))],
        "prompt_tokens_mean": round(mean(r["prompt_tokens"] for r in records)),
        "prompt_tokens_max": max(r["prompt_tokens"] for r in records),
        "output_tokens_mean": round(mean(r["output_tokens"] for r in records)),
        "conflicts": sum(v.get("conflict", False) for r in records for v in r["items"].values()),
        # пункты, которых нет ни в одном списке ответа продукта: считаются missing
        "unlisted_items": sum(
            v.get("unlisted", False) for r in records for v in r["items"].values()
        ),
    }
    if key == "mvp":
        quoted = [
            v
            for r in records
            for v in r["items"].values()
            if v["status"] in {"completed", "partial"}
        ]
        meta["unverified_quotes"] = (
            sum(not quote_verified(v) for v in quoted) / len(quoted) if quoted else None
        )
    return meta


def evaluate(paths: Paths) -> dict:
    docs = load_corpus(paths)
    gold = build_gold(paths, docs)
    by_id = {d.doc_id: d for d in docs}
    methods: dict[str, list[dict[str, dict[str, str]]]] = {}

    # B0.5: самый частый статус пункта на других исходных документах
    sources = sorted({d.source_id for d in docs})
    b05: dict[str, dict[str, str]] = {}
    for held_out in sources:
        train = [
            (i, g["status"])
            for doc_id, items in gold.items()
            if by_id[doc_id].source_id != held_out
            for i, g in items.items()
        ]
        majority = b05_majority(train)
        for d in docs:
            if d.source_id == held_out:
                b05[d.doc_id] = {i: majority.get(i, "missing") for i in ITEMS}
    methods["B0.5"] = [b05]
    methods["B1"] = [{d.doc_id: b1_headings(d, ITEMS) for d in docs}]
    methods["B2"] = [b2_predict(paths, docs, gold)]

    llm_meta: dict[str, dict] = {}
    for key, name in LLM_METHODS.items():
        runs = load_llm_runs(paths, key)
        if not runs:
            continue
        methods[name] = [
            {
                doc_id: {i: v["status"] for i, v in r["items"].items()}
                for doc_id, r in run.items()
                if r["items"]
            }
            for _, run in sorted(runs.items())
        ]
        if key == "mvp":
            methods["MVP + проверка цитат"] = [
                {doc_id: with_quote_check(r["items"]) for doc_id, r in run.items() if r["items"]}
                for _, run in sorted(runs.items())
            ]
        n_docs = len(docs) if key != "product_html" else len(sources)
        llm_meta[name] = _llm_meta(key, runs, n_docs)

    gold_path = paths.data / "labels/gold.json"
    report: dict = {
        "methods": {},
        "llm": llm_meta,
        "gold": {},
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "dev_sources": list(DEV_SOURCES),
    }
    for slice_name in ("original", "affected"):
        report["gold"][slice_name] = dict(
            Counter(
                g["status"]
                for items in gold.values()
                for g in items.values()
                if g["slice"] == slice_name
            )
        )
    for name, runs in methods.items():
        entry: dict = {}
        for slice_name in SLICES:
            per_run = [_pairs(run, gold, by_id, slice_name) for run in runs]
            entry[slice_name] = _average_runs([_metric_block(p) for p in per_run])
            # интервал: бутстреп по исходным БТ, в группе - пары всех запусков
            pooled = _groups([pair for p in per_run for pair in p])
            if len(pooled) > 1:
                entry[slice_name]["qwk_ci"] = bootstrap_ci(pooled, qwk)
        # скор документа: ошибка каждого запуска и размах между запусками
        errors, ranges = [], []
        for doc_id in gold:
            if not all(doc_id in run for run in runs):
                continue
            expert = completeness_score(g["status"] for g in gold[doc_id].values())
            scores = [completeness_score(run[doc_id].values()) for run in runs]
            errors.append(mean(abs(s - expert) for s in scores))
            ranges.append(max(scores) - min(scores))
        entry["score_mae"] = mean(errors) if errors else None
        entry["score_range_max"] = max(ranges) if len(runs) > 1 and ranges else None
        entry["score_range_mean"] = mean(ranges) if len(runs) > 1 and ranges else None
        # нетронутые пункты копий: совпадает ли ответ с ответом на оригинале
        same, total = 0, 0
        for run in runs:
            for doc_id, items in gold.items():
                src = by_id[doc_id].source_id
                if doc_id == src or doc_id not in run or src not in run:
                    continue
                for i, g in items.items():
                    if g["slice"] == "unaffected":
                        total += 1
                        same += run[doc_id][i] == run[src][i]
        entry["unaffected_consistency"] = same / total if total else None
        # уровень шума для сравнения: совпадение ответа двух запусков на одном документе
        if len(runs) > 1:
            agree = [
                runs[0][doc_id][i] == other[doc_id][i]
                for other in runs[1:]
                for doc_id in gold
                if doc_id in runs[0] and doc_id in other
                for i in ITEMS
            ]
            entry["run_agreement"] = mean(agree) if agree else None
        entry["runs"] = len(runs)
        report["methods"][name] = entry
    report["paired"] = _paired(methods, gold, by_id)
    return report


def _paired(methods: dict, gold: dict, by_id: dict) -> dict:
    """MVP против бейзлайнов: разность QWK на срезе eval по первому запуску каждой конфигурации.

    При 5 исходных БТ интервал ориентировочный; значимость различий не утверждается.
    """
    out = {}
    if "MVP" not in methods:
        return out
    for other in ("B3", "B3-lite", "B2", "B1"):
        if other not in methods:
            continue
        a, b = methods["MVP"][0], methods[other][0]
        common = {d: a[d] for d in a if d in b}
        ga = _groups(_pairs(common, gold, by_id, "eval"))
        gb = _groups(_pairs({d: b[d] for d in common}, gold, by_id, "eval"))
        groups = {s: (ga[s], gb[s]) for s in ga if s in gb}
        if len(groups) < 2:
            continue
        diffs = {
            s: qwk([p for p, _ in x], [e for _, e in x]) - qwk([p for p, _ in y], [e for _, e in y])
            for s, (x, y) in groups.items()
        }
        out[f"MVP - {other}"] = {
            "qwk_diff_by_source": diffs,
            "better_sources": sum(d > 0 for d in diffs.values()),
            "n_sources": len(diffs),
            "qwk_diff_ci": paired_bootstrap_diff(groups, qwk),
        }
    return out


def save_report(report: dict, path: Path) -> None:
    def clean(obj):
        if isinstance(obj, float):
            return None if obj != obj else round(obj, 4)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [clean(v) for v in obj]
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(report), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
