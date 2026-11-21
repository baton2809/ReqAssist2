# Данные

Данные ReqAssist - внутренние документы, поэтому в репозиторий не попадают. Здесь описано, что это за данные, откуда они берутся и как воспроизвести EDA локально.

## Наборы

| Набор | Путь в рабочем репозитории продукта | Формат | Объём | Конфиденциальность |
| --- | --- | --- | --- | --- |
| Документы БТ и ФТ | `test-fixtures/br-documents/` | HTML-экспорт страниц Confluence | 5 БТ и 1 ФТ | внутренние: номера задач, продукты, ФИО |
| Стандартный чеклист | `public/checklists/v1.md` | markdown | 39 пунктов в 6 разделах | описания содержат ФИО сотрудников |
| Набор атак prompt injection | `back/routes/securityTest.js`, массив `PAYLOADS` | JS-литерал | 25 атак и 5 легитимных текстов | синтетический |
| Результаты прогона атак | `scripts/security/results-*.json` | JSON | 18 запусков | синтетический вход, метаданные ответа |
| Eval-кейсы | `test-fixtures/eval/` | JSON | 5 кейсов | синтетический |

## Как появляются данные

- **БТ и ФТ** пишут аналитики в Confluence по мере поступления задач (CR): новая страница на каждую доработку, затем правки до согласования. Поток регулярный, но неравномерный; у каждой страницы есть история версий. Сервис получает страницу через REST API Confluence в двух форматах: storage format (для сохранения) и view HTML (для модели). Экспорт в `test-fixtures` - сохранённые view-страницы.
- **Чеклист** ведёт подразделение; меняется редко, вручную.
- **Набор атак** собран командой проекта по категориям: точные шаблоны, семантические перефразировки, обфускация, обходы через социальную инженерию, легитимные тексты.

## Воспроизведение EDA

```bash
uv sync --group eda
uv run --group eda python -m ipykernel install --sys-prefix --name reqassist --display-name "Python (reqassist)"
REQASSIST_DIR=/путь/к/ReqAssist uv run --group eda jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

Вторая команда регистрирует ядро Jupyter `reqassist` внутри `.venv`: ноутбук запускается на нём, чтобы использовать пакеты проекта, а не системный Python. Без доступа к внутренним документам ноутбук не выполнится, но все выводы и графики сохранены в нём и видны на GitHub.

Ноутбук печатает только агрегаты, документы обезличены как `doc_01`..`doc_06`. Графики сохраняются в `reports/figures/`.

## Данные эксперимента ДЗ 3

Всё ниже создаётся локально в `data/` и в git не попадает:

| Путь | Что это |
| --- | --- |
| `labels/gold.json` | разметка 17 проверяемых пунктов для 5 БТ: статус и цитата-доказательство из документа |
| `corpus/` | нормализованные документы `doc_01`..`doc_05`, 20 испорченных копий `doc_0N_c1`..`c4` и `manifest.json` (операция порчи, удалённый фрагмент, SHA-256 текста) |
| `runs/<конфигурация>.jsonl` | ответы модели по документам и запускам: статусы, цитаты, токены, время |
| `cache/` | кеш ответов модели и эмбеддингов: повторный запуск оценки не вызывает модель |

Разметка копий не хранится отдельно: она выводится из разметки оригинала и удалённых фрагментов из `manifest.json`.

## Воспроизведение эксперимента ДЗ 3

Нужен [Ollama](https://ollama.com) с моделями `gemma3:12b` и `bge-m3`; на ноутбуке с 16 ГБ памяти полный прогон занимает около 8 часов.

```bash
ollama pull gemma3:12b
ollama pull bge-m3
uv sync --group eda --group ml
REQASSIST_DIR=/путь/к/ReqAssist uv run --group eda --group ml python scripts/hw3_run.py corpus
REQASSIST_DIR=/путь/к/ReqAssist uv run --group eda --group ml python scripts/hw3_run.py run --method mvp --runs 3
REQASSIST_DIR=/путь/к/ReqAssist uv run --group eda --group ml python scripts/hw3_run.py run --method lite --runs 3
REQASSIST_DIR=/путь/к/ReqAssist uv run --group eda --group ml python scripts/hw3_run.py run --method product
REQASSIST_DIR=/путь/к/ReqAssist uv run --group eda --group ml python scripts/hw3_run.py run --method product_html
REQASSIST_DIR=/путь/к/ReqAssist uv run --group eda --group ml jupyter nbconvert --to notebook --execute --inplace notebooks/02_models.ipynb
```

Прогон можно прерывать: уже выполненные пары документ-запуск пропускаются. Ноутбук пересчитывает метрики из `data/runs/`, сохраняет `reports/hw3/metrics.json` и графики в `reports/figures/`.
