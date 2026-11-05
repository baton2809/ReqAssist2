"""Набор payload-ов prompt injection и детектор L1 на регулярных выражениях."""

import re
from dataclasses import dataclass
from pathlib import Path

# перенесены без изменений из L1-слоя продукта
L1_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(previous|all)\s+instructions", re.IGNORECASE),
    re.compile(r"забудь\s+(все\s+)?инструкции", re.IGNORECASE),
    re.compile(r"ты\s+теперь", re.IGNORECASE),
    re.compile(r"act\s+as\s+", re.IGNORECASE),
    re.compile(r"</?(system|instructions|prompt)>", re.IGNORECASE),
    re.compile(r"\[SYSTEM\]", re.IGNORECASE),
    re.compile(r"повтори\s+(свои\s+)?(системные\s+)?инструкции", re.IGNORECASE),
    re.compile(r"представь\s*,?\s+что\s+ты", re.IGNORECASE),
    re.compile(r"с\s+этого\s+момента", re.IGNORECASE),
]

PAYLOAD_RE = re.compile(
    r"\{\s*id:\s*'(?P<id>[^']+)',\s*category:\s*'(?P<category>[^']+)',\s*"
    r"label:\s*'(?P<label>[^']*)',\s*input:\s*'(?P<input>(?:[^'\\]|\\.)*)'"
)


@dataclass
class Payload:
    payload_id: str
    category: str
    label: str
    text: str

    @property
    def is_attack(self) -> bool:
        return self.category != "baseline"


def l1_matches(text: str) -> list[str]:
    return [p.pattern for p in L1_PATTERNS if p.search(text)]


def parse_payloads(js_source: Path) -> list[Payload]:
    """Читает массив PAYLOADS из исходника тестового стенда; пустые custom-слоты пропускаются."""
    source = js_source.read_text(encoding="utf-8")
    payloads = [
        Payload(m["id"], m["category"], m["label"], m["input"].replace("\\'", "'"))
        for m in PAYLOAD_RE.finditer(source)
    ]
    return [p for p in payloads if p.category != "custom"]
