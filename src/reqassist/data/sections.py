"""Приведение заголовков разделов БТ к каноническим типам."""

import re

# порядок важен: первое совпадение побеждает
CANONICAL_SECTIONS: list[tuple[str, re.Pattern[str]]] = [
    ("jira_links", re.compile(r"jira|ссылк[иа] в")),
    ("stakeholders", re.compile(r"заинтересованн")),
    ("product", re.compile(r"^продукт")),
    ("goal", re.compile(r"цел[ьи]")),
    ("constraints", re.compile(r"ограничени|допущени")),
    ("as_is", re.compile(r"as[\s-]?is")),
    ("to_be", re.compile(r"to[\s-]?be")),
    ("architecture", re.compile(r"архитектур")),
    ("algorithm", re.compile(r"алгоритм|расч[её]т|формул")),
    ("data_ki", re.compile(r"к/и|критичност|целостност")),
    ("ui", re.compile(r"ui|ux|интерфейс")),
    ("integration", re.compile(r"интеграц|api|выгрузк")),
    ("glossary", re.compile(r"глоссари|термин")),
    ("acceptance", re.compile(r"критери[ий] при[её]мки")),
    ("tests", re.compile(r"тест")),
    ("requirements", re.compile(r"требовани")),
]


def normalize_title(title: str) -> str:
    title = title.lower().replace("ё", "е")
    title = re.sub(r"^[\d.\s]+", "", title)  # нумерация вида "6.2 "
    return re.sub(r"\s+", " ", title).strip()


def canonical_section(title: str) -> str:
    normalized = normalize_title(title)
    for name, pattern in CANONICAL_SECTIONS:
        if pattern.search(normalized):
            return name
    return "other"
