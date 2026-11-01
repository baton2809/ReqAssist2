"""Разбор HTML-экспорта страниц Confluence с БТ и ФТ."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

HEADING_RE = re.compile(r"^h[1-6]$")

# заглушки, которыми аналитики помечают незаполненный раздел
PLACEHOLDER_RE = re.compile(
    r"^\s*(-|—|нет|нет данных|tbd|todo|н/д|не обсуждалось|\[уточнить[^\]]*\])\s*$",
    re.IGNORECASE,
)


@dataclass
class Section:
    level: int
    title: str
    text: str
    tables: int
    images: int

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() or bool(PLACEHOLDER_RE.match(self.text))


@dataclass
class ParsedDocument:
    doc_id: str
    kind: str  # "BT" или "FT"
    raw_chars: int
    content_html_chars: int
    text: str
    sections: list[Section] = field(default_factory=list)
    tag_counts: dict[str, int] = field(default_factory=dict)

    @property
    def top_level(self) -> int | None:
        levels = [s.level for s in self.sections]
        return min(levels) if levels else None

    @property
    def top_sections(self) -> list[Section]:
        return [s for s in self.sections if s.level == self.top_level]


def detect_kind(file_name: str) -> str:
    return "FT" if re.search(r"(^|[_\s])ФТ([_\s]|$)", file_name) else "BT"


def _split_sections(content: Tag) -> list[Section]:
    sections: list[Section] = []
    current: dict | None = None
    for node in content.descendants:
        if not isinstance(node, Tag):
            continue
        if HEADING_RE.match(node.name):
            if current:
                sections.append(Section(**current))
            current = {
                "level": int(node.name[1]),
                "title": node.get_text(" ", strip=True),
                "text": "",
                "tables": 0,
                "images": 0,
            }
        elif current is not None:
            if node.name == "table":
                current["tables"] += 1
            elif node.name == "img":
                current["images"] += 1
            elif node.name in {"p", "li", "td", "th"} and not node.find(["p", "li", "td"]):
                current["text"] += node.get_text(" ", strip=True) + "\n"
    if current:
        sections.append(Section(**current))
    return sections


def parse_document(path: Path, doc_id: str) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "lxml")
    content = soup.select_one("#main-content") or soup.body
    tag_counts: dict[str, int] = {}
    for tag in content.find_all(True):
        tag_counts[tag.name] = tag_counts.get(tag.name, 0) + 1
    return ParsedDocument(
        doc_id=doc_id,
        kind=detect_kind(path.name),
        raw_chars=len(raw),
        content_html_chars=len(str(content)),
        text=content.get_text(" ", strip=True),
        sections=_split_sections(content),
        tag_counts=tag_counts,
    )


def load_documents(directory: Path) -> list[ParsedDocument]:
    """Документы получают обезличенные id doc_01..doc_N в порядке имён файлов."""
    files = sorted(directory.glob("*.html"))
    return [parse_document(f, f"doc_{i:02d}") for i, f in enumerate(files, 1)]
