"""Искусственная порча согласованных БТ с выводом меток из цитат-доказательств.

Одна операция на копию. Метка пункта пересчитывается по тому, какие его доказательства
попали в изменённый раздел; все изменённые метки потом проверяет аналитик.
"""

import re
from dataclasses import dataclass, field

from reqassist.data.normalize import NormDoc
from reqassist.evidence import find_span

PLACEHOLDER = "[уточнить: заполнить раздел]"
OPERATIONS = ("delete_section", "empty_body", "placeholder", "placeholder_2")


@dataclass
class Corruption:
    doc: NormDoc
    operation: str
    section_title: str
    labels: dict[str, str]
    removed: tuple[int, int] = (0, 0)  # удалённый полуинтервал исходного текста
    changed: dict[str, tuple[str, str]] = field(default_factory=dict)  # пункт -> (было, стало)


def _section_bounds(text: str) -> list[tuple[int, int, int, str]]:
    """(начало заголовка, начало тела, конец раздела, заголовок) для каждого раздела."""
    heads = [(m.start(), m.end(), m.group(2)) for m in re.finditer(r"^(#{1,6}) (.*)$", text, re.M)]
    bounds = []
    for i, (start, head_end, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        bounds.append((start, head_end, end, title))
    return bounds


def _evidence_sections(doc: NormDoc, gold: dict) -> dict[int, set[str]]:
    """Номер раздела -> пункты со статусом completed или partial, чьи цитаты в нём лежат."""
    bounds = _section_bounds(doc.text)
    result: dict[int, set[str]] = {}
    for item, label in gold.items():
        if label["status"] not in {"completed", "partial"}:
            continue
        for quote in label["evidence"]:
            span = find_span(quote, doc.text)
            if span is None:
                continue
            for i, (start, _, end, _) in enumerate(bounds):
                if start <= span[0] < end:
                    result.setdefault(i, set()).add(item)
    return result


def _relabel(doc: NormDoc, gold: dict, removed: tuple[int, int], placeholder: bool) -> dict:
    labels, changed = {}, {}
    for item, label in gold.items():
        status = label["status"]
        spans = [find_span(q, doc.text) for q in label["evidence"]]
        spans = [s for s in spans if s is not None]
        hit = [s for s in spans if removed[0] <= s[0] < removed[1]]
        new = status
        if status in {"completed", "partial"} and hit:
            if len(hit) == len(spans):
                new = "partial_placeholder" if placeholder else "missing"
            elif status == "completed":
                new = "partial"
        labels[item] = new
        if new != status:
            changed[item] = (status, new)
    return labels, changed


def corrupt(doc: NormDoc, gold: dict) -> list[Corruption]:
    """Четыре копии: разделы выбираются по числу затронутых пунктов, детерминированно.

    Если разделов с доказательствами меньше четырёх, они используются по кругу.
    """
    bounds = _section_bounds(doc.text)
    ranked = sorted(_evidence_sections(doc, gold).items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if not ranked:
        return []
    targets = [ranked[i % len(ranked)][0] for i in range(len(OPERATIONS))]
    copies = []
    for k, (operation, sec) in enumerate(zip(OPERATIONS, targets, strict=True), 1):
        start, body_start, end, title = bounds[sec]
        if operation == "delete_section":
            new_text, removed = doc.text[:start] + doc.text[end:], (start, end)
        elif operation == "empty_body":
            new_text, removed = doc.text[:body_start] + "\n\n" + doc.text[end:], (body_start, end)
        else:
            new_text = doc.text[:body_start] + f"\n\n{PLACEHOLDER}\n\n" + doc.text[end:]
            removed = (body_start, end)
        placeholder = operation.startswith("placeholder")
        labels, changed = _relabel(doc, gold, removed, placeholder=placeholder)
        new_doc = NormDoc(f"{doc.doc_id}_c{k}", doc.doc_id, new_text)
        copies.append(Corruption(new_doc, operation, title, labels, removed, changed))
    return copies
