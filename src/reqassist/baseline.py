"""Бейзлайн B1: проверка пунктов чеклиста по заголовкам разделов документа."""

from dataclasses import dataclass

from reqassist.data.checklist import ChecklistItem
from reqassist.data.confluence import ParsedDocument
from reqassist.data.sections import canonical_section
from reqassist.metrics import Status, completeness_score

# пункты, которые можно привязать к отдельному разделу документа
ITEM_TO_SECTION: dict[str, str] = {
    "3.3": "stakeholders",
    "3.4": "data_ki",
    "3.6": "product",
    "3.10": "tests",
    "4.1": "architecture",
    "6.1": "integration",
}


@dataclass
class BaselineResult:
    doc_id: str
    statuses: dict[str, Status]

    def score(self, item_ids: list[str] | None = None) -> int:
        ids = item_ids if item_ids is not None else list(self.statuses)
        return completeness_score(self.statuses[i] for i in ids)


def _section_status(doc: ParsedDocument, section_type: str) -> Status:
    sections = [s for s in doc.sections if canonical_section(s.title) == section_type]
    if not sections:
        return "missing"
    if all(s.is_empty for s in sections):
        return "partial_placeholder"
    return "completed"


def run_b1(
    doc: ParsedDocument, items: list[ChecklistItem], source: str = "bt_text"
) -> BaselineResult:
    """Оценивает пункты, проверяемые по тексту документа; без привязки к разделу - missing."""
    statuses: dict[str, Status] = {}
    for item in items:
        if item.source != source:
            continue
        section_type = ITEM_TO_SECTION.get(item.item_id)
        statuses[item.item_id] = _section_status(doc, section_type) if section_type else "missing"
    return BaselineResult(doc.doc_id, statuses)
