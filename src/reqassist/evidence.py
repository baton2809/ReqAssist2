"""Поиск цитат-доказательств в тексте документа.

Сравнение идёт без учёта пробелов, регистра и разницы ё/е: нормализация HTML
меняет пробелы вокруг знаков препинания, а смысл цитаты от этого не меняется.
"""

import re

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 90


def _key_with_map(text: str) -> tuple[str, list[int]]:
    chars, index = [], []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        chars.append("е" if ch in "ёЁ" else ch.lower())
        index.append(i)
    return "".join(chars), index


def find_span(quote: str, text: str) -> tuple[int, int] | None:
    """Полуинтервал [start, end) цитаты в исходном тексте или None."""
    key_q, _ = _key_with_map(quote)
    if not key_q:
        return None
    key_t, index = _key_with_map(text)
    pos = key_t.find(key_q)
    if pos < 0:
        return None
    return index[pos], index[pos + len(key_q) - 1] + 1


def quote_supported(quote: str, text: str) -> bool:
    """Точное совпадение без учёта пробелов, иначе нечёткое (для цитат модели)."""
    if not quote or not quote.strip():
        return False
    if find_span(quote, text) is not None:
        return True
    norm = re.sub(r"\s+", " ", quote).strip().lower()
    return fuzz.partial_ratio(norm, re.sub(r"\s+", " ", text).lower()) >= FUZZY_THRESHOLD
