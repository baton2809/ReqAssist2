"""Метрики задачи A (валидация БТ по чеклисту) из раздела 2.1 дизайн-документа."""

from collections import Counter
from collections.abc import Iterable, Sequence
from statistics import mean
from typing import Literal

Status = Literal["completed", "partial", "partial_placeholder", "missing", "not_applicable"]

STATUS_WEIGHTS: dict[str, float] = {
    "completed": 1.0,
    "partial": 0.5,
    "partial_placeholder": 0.1,
    "missing": 0.0,
}


def completeness_score(statuses: Iterable[Status]) -> int:
    """Скор полноты БТ от 0 до 100; пункты not_applicable не входят в знаменатель."""
    counts = Counter(statuses)
    applicable = sum(counts.values()) - counts["not_applicable"]
    if applicable == 0:
        return 0
    weighted = sum(STATUS_WEIGHTS[status] * counts[status] for status in STATUS_WEIGHTS)
    return round(weighted / applicable * 100)


def score_mae(predicted: Sequence[float], expert: Sequence[float]) -> float:
    """Средняя абсолютная разница скора модели и эксперта по документам."""
    if len(predicted) != len(expert):
        raise ValueError("длины predicted и expert не совпадают")
    if not predicted:
        raise ValueError("пустая выборка")
    return mean(abs(p - e) for p, e in zip(predicted, expert, strict=True))


def score_stability(runs: Sequence[float]) -> float:
    """Разница между максимальным и минимальным скором одного документа на повторных запусках."""
    if len(runs) < 2:
        raise ValueError("нужно минимум два запуска")
    return max(runs) - min(runs)


def missing_recall(predicted: Sequence[Status], expert: Sequence[Status]) -> float:
    """Доля пунктов missing по разметке эксперта, которые модель тоже отметила как missing."""
    pairs = [(p, e) for p, e in zip(predicted, expert, strict=True) if e == "missing"]
    if not pairs:
        raise ValueError("в разметке нет пунктов missing")
    return sum(p == "missing" for p, _ in pairs) / len(pairs)


def false_completed_rate(predicted: Sequence[Status], expert: Sequence[Status]) -> float:
    """Доля незаполненных по разметке эксперта пунктов, которые модель отметила как completed."""
    gaps = {"missing", "partial", "partial_placeholder"}
    pairs = [(p, e) for p, e in zip(predicted, expert, strict=True) if e in gaps]
    if not pairs:
        raise ValueError("в разметке нет незаполненных пунктов")
    return sum(p == "completed" for p, _ in pairs) / len(pairs)
