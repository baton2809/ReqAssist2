"""Разбор чеклиста БТ и разметка пунктов по источнику проверки."""

import re
from dataclasses import dataclass
from pathlib import Path

SECTION_RE = re.compile(r"^###\s+(\d+)\.\s+(.+)$")
ITEM_RE = re.compile(r"^-\s+\*\*(\d+\.\d+)\s+(.+?)\*\*")

# откуда можно проверить пункт; разметка по описанию пункта, подтверждена аналитиком:
# bt_text - по тексту БТ, ft_text - по тексту ФТ, linked_doc - нужен связанный документ (CR),
# process - действие в процессе (Jira, совещание, согласование), в тексте документа не видно
ITEM_SOURCE: dict[str, str] = {
    "1.1": "process", "1.2": "process", "1.3": "linked_doc", "1.4": "process", "1.5": "process",
    "2.1": "process", "2.2": "bt_text", "2.3": "bt_text", "2.4": "bt_text", "2.5": "process",
    "2.6": "bt_text",
    "3.1": "linked_doc", "3.2": "process", "3.3": "bt_text", "3.4": "bt_text", "3.5": "bt_text",
    "3.6": "bt_text", "3.7": "bt_text", "3.8": "bt_text", "3.9": "bt_text", "3.10": "bt_text",
    "3.11": "process", "3.12": "process",
    "4.1": "bt_text", "4.2": "process", "4.3": "process",
    "5.1": "ft_text", "5.2": "ft_text", "5.3": "ft_text", "5.4": "ft_text", "5.5": "ft_text",
    "5.6": "ft_text", "5.7": "ft_text", "5.8": "ft_text",
    "6.1": "bt_text", "6.2": "bt_text", "6.3": "bt_text", "6.4": "bt_text",
}  # fmt: skip


@dataclass
class ChecklistItem:
    item_id: str
    section_no: int
    section: str

    @property
    def source(self) -> str:
        return ITEM_SOURCE.get(self.item_id, "unknown")


def parse_checklist(path: Path) -> list[ChecklistItem]:
    """Названия пунктов не сохраняются: в описаниях встречаются ФИО и внутренние системы."""
    items: list[ChecklistItem] = []
    section_no, section = 0, ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := SECTION_RE.match(line):
            section_no, section = int(m.group(1)), m.group(2).strip()
        elif m := ITEM_RE.match(line):
            items.append(ChecklistItem(m.group(1), section_no, section))
    return items
