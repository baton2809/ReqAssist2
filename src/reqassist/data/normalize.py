"""Нормализация HTML страницы Confluence в markdown для модели.

Сохраняются заголовки, абзацы, списки и таблицы (как markdown-таблицы), изображения
заменяются маркером. Стили, span, служебные элементы страницы удаляются.
Результат фиксируется хешем: разметка и цитаты привязаны к нему.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

IMAGE_MARK = "[изображение]"
_BLOCK = {"p", "div", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "pre"}
_SKIP = {"script", "style", "noscript", "colgroup", "col"}


@dataclass
class Section:
    level: int
    title: str
    body: str

    @property
    def is_empty(self) -> bool:
        return not self.body.strip()


@dataclass
class NormDoc:
    doc_id: str
    source_id: str  # id исходного документа; у испорченных копий совпадает с оригиналом
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def sections(self) -> list[Section]:
        return split_sections(self.text)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _inline(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in _SKIP:
        return ""
    if node.name == "img":
        return f" {IMAGE_MARK} "
    if node.name == "br":
        return " "
    return "".join(_inline(c) for c in node.children)


def _table(table: Tag) -> str:
    """Таблица в markdown; colspan и rowspan разворачиваются повтором значения."""
    grid: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}  # колонка -> (текст, сколько строк ещё занять)
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        row: list[str] = []
        cells = iter(tr.find_all(["td", "th"], recursive=False))
        col = 0
        while True:
            if col in pending:
                value, left = pending.pop(col)
                row.append(value)
                if left > 1:
                    pending[col] = (value, left - 1)
                col += 1
                continue
            cell = next(cells, None)
            if cell is None:
                break
            value = _clean(_inline(cell)).replace("|", "/")
            span = int(cell.get("colspan", 1) or 1)
            rows = int(cell.get("rowspan", 1) or 1)
            for _ in range(span):
                row.append(value)
                if rows > 1:
                    pending[col] = (value, rows - 1)
                col += 1
        while col in pending:
            value, left = pending.pop(col)
            row.append(value)
            if left > 1:
                pending[col] = (value, left - 1)
            col += 1
        if any(row):
            grid.append(row)
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    lines = ["| " + " | ".join(grid[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(lines)


def _blocks(node: Tag, out: list[str], list_depth: int = 0) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            text = _clean(str(child))
            if text:
                out.append(text)
            continue
        name = child.name
        if name in _SKIP:
            continue
        if re.fullmatch(r"h[1-6]", name):
            out.append(f"{'#' * int(name[1])} {_clean(_inline(child))}")
        elif name == "table":
            out.append(_table(child))
        elif name in {"ul", "ol"}:
            _blocks_list(child, out, list_depth)
        elif name == "img":
            out.append(IMAGE_MARK)
        elif any(isinstance(c, Tag) and c.name in _BLOCK for c in child.children):
            _blocks(child, out, list_depth)
        else:
            text = _clean(_inline(child))
            if text:
                out.append(text)


def _blocks_list(lst: Tag, out: list[str], depth: int) -> None:
    for li in lst.find_all("li", recursive=False):
        nested = [c for c in li.children if isinstance(c, Tag) and c.name in {"ul", "ol"}]
        head = "".join(_inline(c) for c in li.children if c not in nested)
        if _clean(head):
            out.append(f"{'  ' * depth}- {_clean(head)}")
        for sub in nested:
            _blocks_list(sub, out, depth + 1)


def normalize_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("#main-content") or soup.body or soup
    out: list[str] = []
    _blocks(content, out)
    text = "\n\n".join(b for b in out if b.strip())
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current: Section | None = None
    body: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6}) (.*)$", line)
        if m:
            if current:
                current.body = "\n".join(body).strip()
                sections.append(current)
            current, body = Section(len(m.group(1)), m.group(2).strip(), ""), []
        elif current is not None:
            body.append(line)
    if current:
        current.body = "\n".join(body).strip()
        sections.append(current)
    return sections


def load_normalized(directory: Path, kind: str = "BT") -> list[NormDoc]:
    """Документы получают id doc_01..doc_N в порядке имён файлов, как в EDA."""
    from reqassist.data.confluence import detect_kind

    docs = []
    for i, f in enumerate(sorted(directory.glob("*.html")), 1):
        if kind and detect_kind(f.name) != kind:
            continue
        doc_id = f"doc_{i:02d}"
        docs.append(NormDoc(doc_id, doc_id, normalize_html(f.read_text(encoding="utf-8"))))
    return docs
