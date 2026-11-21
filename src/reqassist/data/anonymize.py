"""Маскирование конфиденциальных сведений перед публикацией примеров."""

import re

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b"), "<jira>"),
    # фамилия и инициалы или полное ФИО
    (re.compile(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.?"), "<фио>"),
    (
        re.compile(
            r"\b[А-ЯЁ][а-яё]+(ов|ев|ин|ын|ая|ова|ева|ина|ына)\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(вич|вна)\b"
        ),
        "<фио>",
    ),
    # аббревиатуры продуктов и внутренних систем: два и более заглавных символа
    (re.compile(r"\b[A-ZА-ЯЁ][A-ZА-ЯЁ0-9/-]{1,}\b"), "<abbr>"),
    # смешанный регистр вида RoRWA
    (re.compile(r"\b\w*[A-ZА-ЯЁ]\w*[A-ZА-ЯЁ]\w*\b"), "<abbr>"),
    (re.compile(r"\d+"), "<n>"),
]


def mask(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
