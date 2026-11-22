"""Бейзлайны ДЗ 3 без LLM.

B0.5 - самый частый статус, B1 - заголовки, B2 - эмбеддинги и логистическая регрессия.

Все работают на нормализованном тексте (NormDoc), как и LLM-конфигурации.
"""

import re
from collections import Counter

import numpy as np

from reqassist.baseline import ITEM_TO_SECTION
from reqassist.data.confluence import PLACEHOLDER_RE
from reqassist.data.normalize import NormDoc
from reqassist.data.sections import canonical_section

# заглушка внутри текста раздела, а не только раздел целиком из неё
_PLACEHOLDER_IN_TEXT = re.compile(r"\[уточнить[^\]]*\]", re.IGNORECASE)


def b05_majority(train: list[tuple[str, str]]) -> dict[str, str]:
    """Самый частый статус каждого пункта на обучающих документах: (пункт, статус) -> статус."""
    by_item: dict[str, Counter] = {}
    for item, status in train:
        by_item.setdefault(item, Counter())[status] += 1
    return {item: counts.most_common(1)[0][0] for item, counts in by_item.items()}


def b1_headings(doc: NormDoc, items: list[str]) -> dict[str, str]:
    """Пустой раздел - missing, раздел из одной заглушки - partial_placeholder."""
    result = {}
    for item in items:
        section_type = ITEM_TO_SECTION.get(item)
        sections = [
            s for s in doc.sections if section_type and canonical_section(s.title) == section_type
        ]
        if not sections or all(s.is_empty for s in sections):
            result[item] = "missing"
        elif all(
            PLACEHOLDER_RE.match(s.body) or _PLACEHOLDER_IN_TEXT.fullmatch(s.body.strip())
            for s in sections
        ):
            result[item] = "partial_placeholder"
        else:
            result[item] = "completed"
    return result


def chunks(doc: NormDoc, size: int = 500) -> list[tuple[str, bool]]:
    """Фрагменты для эмбеддингов: строки таблиц с шапкой, абзацы склеиваются до size символов.

    Возвращает (текст, это строка таблицы).
    """
    out: list[tuple[str, bool]] = []
    for section in doc.sections:
        prefix = f"{section.title}: "
        buffer, header = "", ""
        for line in section.body.splitlines():
            if line.startswith("|"):
                if buffer:
                    out.append((prefix + buffer, False))
                    buffer = ""
                if not header:
                    header = line
                elif not set(line) <= set("|- "):
                    out.append((prefix + header + "\n" + line, True))
                continue
            header = ""
            if line.strip():
                buffer = f"{buffer} {line.strip()}".strip()
                if len(buffer) >= size:
                    out.append((prefix + buffer, False))
                    buffer = ""
        if buffer:
            out.append((prefix + buffer, False))
        if not section.body.strip():
            out.append((section.title, False))
    return out


def features(
    item_vec: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_meta: list[tuple[str, bool]],
    heading_found: bool,
    doc_len: int,
) -> np.ndarray:
    """Признаки пары (пункт, документ) для B2."""
    sims = chunk_vecs @ item_vec
    order = np.argsort(-sims)
    top = sims[order]
    best_text, best_is_table = chunk_meta[order[0]]
    return np.array(
        [
            top[0],
            top[: min(3, len(top))].mean(),
            top[1] if len(top) > 1 else top[0],
            float((sims > 0.5).sum()),
            float(heading_found),
            float(best_is_table),
            np.log1p(len(best_text)),
            np.log1p(doc_len),
            float(bool(_PLACEHOLDER_IN_TEXT.search(best_text))),
        ]
    )


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
