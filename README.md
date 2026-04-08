# Eho Dialogue Extractor

Инструмент для подготовки датасета по книгам Макса Фрая:
- извлекает голос Сэра Макса;
- собирает знания о мире Ехо;
- строит per-book и глобальную базу знаний;
- генерирует синтетические пары для fine-tune.

Скрипт работает с локальной LLM через OpenAI-совместимый API: `ollama`, `vllm`, `llama.cpp server`, `LM Studio` и похожие сервисы.

## Что умеет

- Загружает книги из `.fb2`, `.fb2.zip` и `.txt`.
- Извлекает голос Макса через быстрый `regex`-extractor или через LLM.
- Извлекает знания о персонажах, местах, магии, истории, событиях, существах и быте.
- Делает semantic chunking: `section -> абзацы -> fallback по длине`.
- Может делать несколько extraction-проходов по одному чанку и использовать соседние чанки как supporting context.
- Канонизирует сущности внутри книги и между книгами.
- Линкует новые факты к уже накопленной базе знаний через retrieval.
- Строит отдельные per-book артефакты и отдельную общую базу знаний:
  - `knowledge_base_raw.*`
  - `knowledge_base.*`
- Поддерживает resume после остановки.
- Корректно реагирует на `Ctrl+C`: сохраняет прогресс и завершает пайплайн мягко.
- Пишет человекочитаемые `.txt`-версии результатов.

## Структура проекта

- `extract_dialogues.py` — основной конвейер.
- `dataset_studio.py` — веб-студия для ревью и правки датасета.
- `dataset_studio_assets/` — статические файлы интерфейса студии.
- `extract_regex.py` — быстрый локальный extractor голоса без LLM.
- `diagnose_llm.py` — вспомогательная диагностика модели.
- `tests/` — unit- и smoke-тесты ключевых частей пайплайна.

## Требования

- Python 3.11+
- Локальная модель с OpenAI-совместимым API
- Для дефолтного сценария: установленный `ollama`

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Быстрый старт

1. Создай и активируй виртуальное окружение.
2. Установи зависимости:

```bash
pip install -r requirements.txt
```

3. Положи книги в папку `books/`.
4. Запусти локальную модель, например:

```bash
ollama run gemma4:e4b
```

5. Запусти пайплайн:

```bash
python extract_dialogues.py
```

Полезный пример:

```bash
python extract_dialogues.py --workers 2 --voice-extractor regex --chunk-size 2500
```

Если не нужно автозапускать `ollama serve`:

```bash
python extract_dialogues.py --no-auto-serve
```

Если нужно начать заново без resume:

```bash
python extract_dialogues.py --no-resume
```

## Dataset Studio

После того как в `output/` появились `knowledge_*.jsonl`, `voice_*.jsonl`, `chunks_*.jsonl` и другие артефакты, можно открыть веб-редактор:

```bash
python dataset_studio.py --output-dir ./output
```

По умолчанию студия поднимается на `http://127.0.0.1:8766`.

Что умеет студия:
- просматривать, оценивать, редактировать, создавать и удалять факты;
- смотреть исходный текст чанка, из которого поднят факт или пример;
- работать со вкладкой `Chunks`: искать по исходным кускам книги, смотреть связанные факты и примеры, создавать новый факт прямо из выбранного чанка;
- редактировать `time_scope`, заметки ревью и оценку;
- создавать и править связи между фактами;
- создавать, редактировать и объединять темы;
- редактировать `voice`- и `synth`-примеры;
- генерировать новые `voice`/`synth`-примеры по выбранным фактам;
- отправлять один или несколько фактов на повторный LLM-анализ;
- просматривать старые `llm_traces`, открывать конкретный trace, редактировать prompt и повторно запускать его с новым trace;
- делать полностью новый ручной LLM-запуск прямо из интерфейса;
- смотреть вкладку `Pipeline`: текущий `metadata.json`, историю событий из `metadata_history.jsonl` и журнал `llm_jobs`;
- смотреть вкладку `Timeline`, если pipeline уже построил `timeline_graph.json` и `timeline_resolution_raw.json`;
- откатывать изменения и экспортировать финальный датасет.

Для headless UI smoke-тестов внутри WSL есть отдельный скрипт, который поднимает изолированный Chromium for Testing и не трогает Windows-браузер:

```bash
tests/run_ui_headless_wsl.sh
```

Он сам:
- создаёт локальный WSL-venv `.wsl-ui-venv`;
- ставит `selenium`;
- скачивает portable `Chrome for Testing` и `chromedriver` в Linux home;
- докачивает нужные runtime-библиотеки без системной установки;
- запускает `tests.test_dataset_studio_ui` в headless-режиме.

Все пользовательские изменения пишутся отдельно и не затирают исходные pipeline-файлы:
- `output/editor_workspace/facts_ops.jsonl`
- `output/editor_workspace/samples_ops.jsonl`
- `output/editor_workspace/themes_ops.jsonl`
- `output/editor_workspace/relations_ops.jsonl`

Финальный экспорт студии сохраняется в:

```text
output/editor_workspace/exports/<timestamp>/
```

Там появятся:
- `knowledge_base_final.json`
- `knowledge_base_final.txt`
- `dataset_final.jsonl`
- `dataset_final.txt`
- `voice_final.jsonl`
- `synth_final.jsonl`
- `themes_final.json`
- `relations_final.json`

## Основные параметры

- `--books-dir` — папка с книгами.
- `--output-dir` — папка результатов.
- `--api-base` — URL API модели.
- `--model` — имя модели.
- `--chunk-size` — целевой размер чанка.
- `--workers` — количество воркеров.
- `--voice-extractor regex|llm` — чем извлекать голос Макса.
- `--skip-synth` — пропустить синтетические пары.
- `--synth-count` — лимит синтетических пар на книгу.
- `--request-timeout` — таймаут одного запроса.
- `--seed` — фиксированный seed для воспроизводимости.

## Что появляется в `output/`

### По каждой книге

- `voice_<book>.jsonl` — пары для обучения голоса.
- `voice_<book>.txt` — человекочитаемая версия голоса.
- `knowledge_<book>.jsonl` — потоковые факты по мере обработки.
- `knowledge_<book>.json` — финальная база знаний книги.
- `knowledge_<book>.txt` — человекочитаемая база знаний книги.
- `synth_<book>.jsonl` — синтетические пары.
- `synth_<book>.txt` — человекочитаемая синтетика.
- `chunks_<book>.jsonl` — checkpoint по чанкам.
- `synth_progress_<book>.jsonl` — checkpoint синтетики.
- `done_<book>.marker` — маркер завершённой книги.

### Глобальные артефакты

- `dataset.jsonl` — итоговый объединённый датасет.
- `dataset.txt` — человекочитаемая версия датасета.
- `knowledge_base_raw.json` — сырая глобальная база знаний, собранная из всех книг.
- `knowledge_base_raw.txt` — человекочитаемая сырая глобальная база.
- `knowledge_base.json` — глобальная канонизированная и дедуплицированная база знаний.
- `knowledge_base.txt` — человекочитаемая финальная глобальная база.
- `metadata.json` — параметры запуска и итоговая статистика.

## Как устроен пайплайн

1. Книга разбирается и режется на semantic chunks.
2. Из каждого чанка извлекается голос Макса.
3. Из каждого чанка извлекаются факты о мире и событиях.
4. Факты нормализуются и привязываются к уже накопленным сущностям.
5. Для каждой книги строится отдельная база знаний.
6. После обработки всех книг запускается отдельный этап сборки общей базы знаний из per-book артефактов.
7. При необходимости генерируются синтетические пары в голосе Макса.

## Resume и остановка

Повторный запуск продолжает уже начатую обработку:
- по чанкам через `chunks_<book>.jsonl`;
- по синтетике через `synth_progress_<book>.jsonl`.

Если нажать `Ctrl+C`, скрипт:
- мягко завершит текущий этап;
- сохранит уже полученный прогресс;
- позволит продолжить следующим запуском.

Для полного пересчёта используй `--no-resume`.

## Тесты

Быстрый прогон основных тестов:

```bash
python -m unittest discover -s tests -v
```

Headless UI smoke-тесты в WSL:

```bash
tests/run_ui_headless_wsl.sh
```
