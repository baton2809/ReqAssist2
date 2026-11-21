"""Промпты и разбор ответов для трёх LLM-конфигураций валидации БТ.

- mvp: правила, определения статусов, цитата-доказательство, JSON-схема, синтетический пример
- lite: короткая инструкция и та же схема без цитат (наивный бейзлайн)
- product: системный промпт продукта как есть, ответ в формате продукта
"""

import json
import re

STATUSES = ["completed", "partial", "partial_placeholder", "missing", "not_applicable"]

_MVP_RULES = """Ты проверяешь бизнес-требования (БТ) банка по пунктам чеклиста качества.
Для каждого пункта определи статус по тексту документа.

Статусы:
- completed - в тексте есть сведения, которые полностью закрывают пункт
- partial - сведения есть, но закрывают пункт не полностью: например, раскрыт только один из перечисленных аспектов
- partial_placeholder - на месте нужных сведений стоит заглушка вроде [уточнить]
- missing - сведений нет; заголовок раздела без содержания - тоже missing
- not_applicable - пункт условный ("при необходимости", "при изменении", "для новых интеграций"), а условие в этой задаче не выполняется

Правила:
1. Оценивай только по тексту документа. Не додумывай содержимое задач Jira, CR, ссылок и картинок (они помечены [изображение]).
2. Если в документе есть хотя бы часть сведений, о которых спрашивает пункт (назван процесс, продукт, смежная система, дата выхода, сценарий), ставь partial, а не missing. completed - только когда раскрыто всё, что перечислено в пункте.
3. not_applicable допустим только для условных пунктов и только если условие в задаче не выполняется:
   - 3.7 - задача не затрагивает гарантии, ЕДБГ, аккредитивы и облигации;
   - 6.2 - в задаче нет новых расчётов, формул или алгоритмов;
   - 6.3 - эффективная ставка не меняется;
   - 6.4 - новых интеграций нет.
   Остальные пункты никогда не бывают not_applicable.
4. Таблица сценариев или пронумерованные критерии с условиями и ожидаемым результатом в разделе критериев приёмки - это тестовые сценарии (3.10).
5. Для completed и partial поле quote - дословная цитата из текста внутри <документ> до 200 символов. Никогда не цитируй описание пункта чеклиста. Для остальных статусов quote - пустая строка.
6. Порядок мысли: сначала найди цитату, затем кратко обоснуй (reason, до 150 символов), затем поставь статус.
7. Если цитату из документа найти не удаётся, статус не может быть completed или partial.

Пример на вымышленном документе.
Документ: "# Продукт\\nКредиты для малого бизнеса, плавающая ставка.\\n# Критерии приёмки\\n[уточнить: сценарии]"
Ответ для трёх пунктов:
{"3.6": {"quote": "Кредиты для малого бизнеса, плавающая ставка.", "reason": "назван продукт и тип ставки, остальные продукты не рассмотрены", "status": "partial"},
 "3.10": {"quote": "", "reason": "вместо сценариев заглушка", "status": "partial_placeholder"},
 "3.3": {"quote": "", "reason": "заказчик не указан", "status": "missing"}}
"""


def _item_list(item_texts: dict[str, str]) -> str:
    return "\n".join(f"{item}: {text}" for item, text in item_texts.items())


def _schema(item_ids: list[str], with_quote: bool) -> dict:
    props = {"status": {"type": "string", "enum": STATUSES}}
    if with_quote:
        props = {"quote": {"type": "string"}, "reason": {"type": "string"}} | props
    item = {"type": "object", "properties": props, "required": list(props)}
    return {
        "type": "object",
        "properties": dict.fromkeys(item_ids, item),
        "required": item_ids,
    }


def mvp_prompt(item_texts: dict[str, str]) -> tuple[str, dict]:
    system = _MVP_RULES + "\nПункты чеклиста:\n" + _item_list(item_texts)
    return system, _schema(list(item_texts), with_quote=True)


def lite_prompt(item_texts: dict[str, str]) -> tuple[str, dict]:
    system = (
        "Проверь бизнес-требования по пунктам чеклиста. Для каждого пункта верни статус: "
        + ", ".join(STATUSES)
        + ".\n\nПункты чеклиста:\n"
        + _item_list(item_texts)
    )
    return system, _schema(list(item_texts), with_quote=False)


def document_message(text: str) -> str:
    return f"<документ>\n{text}\n</документ>\n\nВерни JSON по схеме для всех пунктов."


def product_message(document: str, checklist: str, standard_note: str) -> str:
    """Сообщение пользователя в том же виде, в каком его собирает продукт."""
    return (
        "Проанализируй предоставленное бизнес-требование:\n\n"
        f"<business_requirements>\n{document}\n</business_requirements>\n\n"
        f'{standard_note}\n<checklist source="default">\n{checklist}\n</checklist>\n\n'
        "Выполни анализ согласно инструкциям."
    )


def parse_structured(content: str, item_ids: list[str]) -> dict[str, dict]:
    data = json.loads(content)
    result = {}
    for item in item_ids:
        value = data.get(item) or {}
        status = value.get("status")
        result[item] = {
            "status": status if status in STATUSES else "missing",
            "quote": value.get("quote", ""),
            "reason": value.get("reason", ""),
        }
    return result


def _entry_id(entry) -> str:
    """item_id записи; число не приводится к строке пункта: 3.10 в JSON читается как 3.1."""
    if not isinstance(entry, dict):
        return ""
    value = entry.get("item_id", "")
    return value.strip() if isinstance(value, str) else ""


def parse_product(content: str, item_ids: list[str]) -> dict[str, dict]:
    """Ответ продукта: checklist_coverage со списками completed, missing, not_applicable.

    Пункт, которого нет ни в одном списке, считается missing - так же скор считает продукт;
    такие пункты помечаются флагом unlisted. Если модель поместила пункт в несколько
    списков, побеждает статус с меньшим вкладом в скор (missing, partial_placeholder,
    partial, not_applicable, completed), а пункт помечается флагом conflict.
    Записи не-объекты и item_id числом пропускаются.
    """
    text = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M)
    coverage = json.loads(text).get("checklist_coverage", {})
    if not isinstance(coverage, dict):
        coverage = {}
    result = {
        item: {"status": "missing", "quote": "", "reason": "", "conflict": False}
        for item in item_ids
    }
    mapping = {
        "full": "completed",
        "completed": "completed",
        "partial": "partial",
        "partial_placeholder": "partial_placeholder",
        "missing": "missing",
        "not_applicable": "not_applicable",
    }
    seen: dict[str, set[str]] = {item: set() for item in item_ids}
    for entry in coverage.get("completed") or []:
        item = _entry_id(entry)
        if item in result:
            seen[item].add(mapping.get(str(entry.get("status", "full")), "completed"))
            result[item]["quote"] = str(entry.get("found_in_section", ""))
    for key in ("not_applicable", "missing"):
        for entry in coverage.get(key) or []:
            item = _entry_id(entry)
            if item in result:
                seen[item].add(key)
    order = ("missing", "partial_placeholder", "partial", "not_applicable", "completed")
    for item, statuses in seen.items():
        for status in order:
            if status in statuses:
                result[item]["status"] = status
                break
        result[item]["conflict"] = len(statuses) > 1
        result[item]["unlisted"] = not statuses
    return result
