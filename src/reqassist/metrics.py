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
    """Средняя абсолютная разница скора модели и разметки по документам."""
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
    """Доля пунктов missing по разметке, которые модель тоже отметила как missing."""
    pairs = [(p, e) for p, e in zip(predicted, expert, strict=True) if e == "missing"]
    if not pairs:
        raise ValueError("в разметке нет пунктов missing")
    return sum(p == "missing" for p, _ in pairs) / len(pairs)


def false_completed_rate(predicted: Sequence[Status], expert: Sequence[Status]) -> float:
    """Доля незаполненных по разметке пунктов, которые модель отметила как completed."""
    gaps = {"missing", "partial", "partial_placeholder"}
    pairs = [(p, e) for p, e in zip(predicted, expert, strict=True) if e in gaps]
    if not pairs:
        raise ValueError("в разметке нет незаполненных пунктов")
    return sum(p == "completed" for p, _ in pairs) / len(pairs)


# порядок статусов для QWK: от худшего к лучшему
ORDINAL: dict[str, int] = {"missing": 0, "partial_placeholder": 1, "partial": 2, "completed": 3}


def qwk(predicted: Sequence[Status], expert: Sequence[Status]) -> float:
    """Quadratic weighted kappa по рангам статусов.

    Пары, где по разметке пункт not_applicable, не учитываются. Если модель поставила
    not_applicable применимому пункту, это ошибка: пункт не проверен, ранг как у missing.
    Для пункта с разметкой missing такой ответ QWK не штрафует - он учитывается в recall
    missing и в доле not_applicable на применимых пунктах.
    """
    from sklearn.metrics import cohen_kappa_score

    pairs = [
        (ORDINAL.get(p, 0), ORDINAL[e])
        for p, e in zip(predicted, expert, strict=True)
        if e in ORDINAL
    ]
    if len({e for _, e in pairs} | {p for p, _ in pairs}) < 2:
        return float("nan")
    return float(
        cohen_kappa_score(
            [e for _, e in pairs], [p for p, _ in pairs], labels=list(range(4)), weights="quadratic"
        )
    )


def macro_f1(predicted: Sequence[Status], expert: Sequence[Status]) -> float:
    """Macro-F1 по статусам, которые встречаются в разметке."""
    from sklearn.metrics import f1_score

    pairs = [(p, e) for p, e in zip(predicted, expert, strict=True) if e in ORDINAL]
    labels = sorted({e for _, e in pairs}, key=ORDINAL.__getitem__)
    return float(
        f1_score(
            [e for _, e in pairs],
            [p for p, _ in pairs],
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def bootstrap_ci(
    groups: dict[str, list], metric, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% интервал метрики бутстрепом по группам (исходным документам).

    groups: id исходного документа -> список пар (предсказание, эталон).
    При 5 группах различных выборок мало, интервал ориентировочный.
    """
    import random

    rng = random.Random(seed)  # noqa: S311 - бутстреп, не криптография
    keys = list(groups)
    values = []
    for _ in range(n):
        pairs = [pair for k in (rng.choice(keys) for _ in keys) for pair in groups[k]]
        value = metric([p for p, _ in pairs], [e for _, e in pairs])
        if value == value:  # пропуск nan
            values.append(value)
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def paired_bootstrap_diff(
    groups: dict[str, tuple[list, list]], metric, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% интервал разности метрик двух конфигураций на одних и тех же группах.

    groups: id исходного документа -> (пары конфигурации A, пары конфигурации B).
    """
    import random

    rng = random.Random(seed)  # noqa: S311 - бутстреп, не криптография
    keys = list(groups)
    values = []
    for _ in range(n):
        sample = [rng.choice(keys) for _ in keys]
        a = [pair for k in sample for pair in groups[k][0]]
        b = [pair for k in sample for pair in groups[k][1]]
        diff = metric([p for p, _ in a], [e for _, e in a]) - metric(
            [p for p, _ in b], [e for _, e in b]
        )
        if diff == diff:
            values.append(diff)
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]
