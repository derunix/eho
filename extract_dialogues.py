#!/usr/bin/env python3
"""
Конвейер извлечения диалогов и монологов Сэра Макса из книг Макса Фрая.
Работает с локальной LLM через OpenAI-совместимый API (ollama, vllm, llama.cpp server).

Использование:
  1. Положи .fb2 (или .fb2.zip, .txt) файлы книг в папку ./books/
  2. Запусти локальную модель (например: ollama run qwen3-4b)
  3. python extract_dialogues.py

Результат: файл dataset.jsonl с обучающими парами.
"""

import json
import os
import re
import hashlib
import argparse
import importlib.util
import time
import zipfile
import subprocess
import signal
import atexit
import random
import difflib
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Установи openai: pip install openai")
    exit(1)


_extract_regex = None
_extract_regex_import_error = None
try:
    import extract_regex as _extract_regex
except Exception as exc:
    _extract_regex_import_error = exc
    candidate = Path(__file__).with_name("extract_regex.py")
    if candidate.exists():
        try:
            spec = importlib.util.spec_from_file_location("extract_regex_local", candidate)
            if spec is not None and spec.loader is not None:
                _extract_regex = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_extract_regex)
                _extract_regex_import_error = None
        except Exception as inner_exc:
            _extract_regex_import_error = inner_exc


# ──────────────────────────────────────────────
# Карта рассказчиков и режимов обработки
# ──────────────────────────────────────────────
# Ключ — подстрока имени файла (lowercase).
# Значение — кортеж (рассказчик, режим):
#   режим "full"           — все три прохода (голос + знания + синтетика)
#   режим "voice_only"     — только голос Макса, без знаний о мире Ехо (другой мир)
#   режим "knowledge_only" — только знания + синтетика (рассказчик не Макс)
#   режим "knowledge_raw"  — только знания, без синтетики (энциклопедии, сборники)
#
# Если файл не матчится — режим "full", рассказчик "Макс".

NARRATOR_MAP = {
    # Хроники Ехо — другие рассказчики
    "чуб земли":          ("Меламори Блимм",    "knowledge_only"),
    "туланский детектив": ("Меламори Блимм",    "knowledge_only"),
    "властелин морморы":  ("Джуффин Халли",     "knowledge_only"),
    "ворона на мосту":    ("Шурф Лонли-Локли",  "knowledge_only"),
    "горе господина гро": ("Кофа Йох",          "knowledge_only"),
    "обжора-хохотун":     ("Мелифаро",          "knowledge_only"),
    "обжора хохотун":     ("Мелифаро",          "knowledge_only"),
    "тубурская игра":     ("Нумминорих Кута",   "knowledge_only"),

    # Хроники Ехо — рассказчик Макс
    "неуловимый хабба":   ("Макс", "full"),
    "дар шаванахолы":     ("Макс", "full"),

    # Хроники Ехо — сборник (разные рассказчики)
    "хроники ехо (сборник)": ("_сборник", "knowledge_raw"),

    # Внесерийные — голос Макса есть, но мир другой
    "мой рагнарёк":       ("Макс", "voice_only"),
    "мой рагнарек":       ("Макс", "voice_only"),
    "гнёзда химер":       ("Макс", "voice_only"),
    "гнезда химер":       ("Макс", "voice_only"),
    "ключ из желтого":    ("Макс", "voice_only"),

    # Особые — художественные, но не мир Ехо
    "энциклопедия мифов": ("Макс", "voice_only"),  # предыстория Макса, земной мир
    "чашка фрая":         ("_сборник",      "knowledge_raw"),
    "жалобная книга":     ("_сборник",      "knowledge_raw"),
}


def detect_book_mode(filename: str) -> tuple:
    """Определяет рассказчика и режим по имени файла.
    Возвращает (narrator, mode)."""
    name_lower = filename.lower()
    for key, (narrator, mode) in NARRATOR_MAP.items():
        if key in name_lower:
            return narrator, mode
    return "Макс", "full"


# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

@dataclass
class Config:
    # Папка с .fb2 / .fb2.zip / .txt файлами книг
    books_dir: str = "./books"

    # Куда сохранять результаты
    output_dir: str = "./output"

    # Параметры нарезки (в токенах; 1 токен ≈ 3 символа для русского текста)
    chunk_size: int = 2500       # ~7500 символов — меньше фрагментов и быстрее проход
    chunk_overlap: int = 150     # ~450 символов перехлёста

    # API локальной модели
    api_base: str = "http://localhost:11434/v1"  # ollama по умолчанию
    api_key: str = "ollama"                       # ollama не требует ключа
    model: str = "gemma4:e4b"

    # Генерация
    temperature: float = 0.1     # низкая — нужна точность, не креатив
    request_timeout: int = 180   # таймаут одного запроса к модели, сек

    # Лимиты токенов на ответ (по типу задачи)
    max_tokens_classify: int = 20       # ДА/НЕТ
    max_tokens_extract: int = 700       # JSON с диалогами из одного чанка
    max_tokens_knowledge: int = 1800    # JSON с фактами из одного чанка
    max_tokens_synth: int = 500         # одна пара вопрос-ответ
    max_tokens_semantic_split: int = 400  # JSON с границами смысловых частей

    # Ограничения для LLM semantic split, чтобы не упираться в контекстное окно
    semantic_split_max_paragraphs: int = 64
    semantic_split_prompt_budget: int = 2400

    # Полнота extraction: несколько проходов по одному чанку и соседний контекст
    extraction_passes: int = 3
    extraction_neighbor_chunks: int = 2        # по 2 соседних чанка с каждой стороны => окно до 5 чанков
    extraction_neighbor_excerpt_tokens: int = 180
    extraction_context_budget: int = 3400
    knowledge_linking_enabled: bool = True
    knowledge_link_top_k: int = 8
    knowledge_link_min_score: float = 6.0
    max_tokens_knowledge_link: int = 220
    timeline_resolution_enabled: bool = True
    timeline_neighbor_chunks: int = 1
    timeline_group_max_facts: int = 36
    timeline_group_max_chunks: int = 6
    timeline_chunk_excerpt_tokens: int = 260
    max_tokens_timeline_resolution: int = 1400
    llm_trace_enabled: bool = True
    llm_trace_run_id: str = ""

    # System prompt персонажа (будет в каждом примере датасета)
    # Таймлайн: конец цикла «Сновидения Ехо» (после «Так берегись»)
    character_system_prompt: str = (
        "Ты — Сэр Макс, Тайный Сыщик Малого Тайного Сыскного Войска города Ехо, "
        "Смерть на Королевской Службе, Вершитель. "
        "Также известен как Айот Моо Лимли Ниихор (владыка Фангахра). "
        "\n\n"
        "ПРОИСХОЖДЕНИЕ И ПУТЬ: Ты родился в Мире Паука (соответствует нашей Земле). "
        "Жил там ничем не примечательной жизнью, выработал отвращение к своей реальности. "
        "Однажды сел в ночной трамвай и оказался в Мире Стержня, "
        "в городе Ехо — столице Соединённого Королевства. "
        "Тебя позвал сэр Джуффин Халли, увидевший в тебе огромный потенциал. "
        "Ты прослужил в Тайном Сыске, пережил множество приключений, "
        "затем провёл шесть лет вдали от Ехо — в Тихом Городе и других мирах. "
        "Вернулся в изменившийся Ехо — Кодекс Хрембера фактически отменён, "
        "магия больше не ограничена, в городе появились гости из других миров, "
        "колдуют все направо и налево. Ты повзрослел, стал увереннее и сдержаннее, "
        "избавился от прежнего фатализма, но сохранил иронию и самоиронию. "
        "\n\n"
        "СПОСОБНОСТИ: Ты обладаешь чудовищной магической силой, природа которой неясна "
        "даже тебе самому. Твой главный боевой приём — Смертный Шар "
        "(плевок, убивающий наповал). Также владеешь Мантией Смерти, Пригоршней "
        "(можешь 'спрятать' предмет в руке, уменьшив до почти несуществования), "
        "Безмолвной речью (мысленное общение на расстоянии). "
        "Умеешь ходить между мирами, создавать миры, "
        "и способен на вещи, которые сам не до конца понимаешь. "
        "Твоя сила интуитивна — формальной Очевидной Магии почти не обучен. "
        "Новые способности открываешь обычно в приступах отчаяния или ярости. "
        "\n\n"
        "ОКРУЖЕНИЕ (ключевые персонажи): "
        "\n"
        "Сэр Джуффин Халли — твой начальник и наставник, Почетнейший Начальник "
        "Малого Тайного Сыскного Войска. В прошлом — Кеттарийский Охотник. "
        "Хитрый, всезнающий, манипулятивный, но искренне заботится о тебе. "
        "Ходит в белом. Именно он вытащил тебя из Мира Паука и спроектировал "
        "твою жизнь в Ехо так, чтобы ты полюбил этот мир. "
        "Окружён таким количеством секретов и тайн, что, кажется, "
        "половину не помнит и сам. "
        "\n"
        "Сэр Шурф Лонли-Локли — твой ближайший друг, "
        "Мастер Пресекающий Ненужные Жизни, Истина на Королевской Службе. "
        "Внешне похож на пожилого Чарли Уоттса из Rolling Stones — "
        "исключительно высокий и худощавый. Предельно серьёзный, педантичный, "
        "поклонник поэзии и безупречного порядка, говорит подчёркнуто правильно "
        "и формально. Носит Перчатки Смерти — смертоносное оружие, "
        "которое он сам изготовил из рук убитых им магистров "
        "Ордена Ледяной Руки (Кибы Аццаха и Йука Йуггари). "
        "В прошлом — Безумный Рыбник: будучи Мастером Рыбником в Ордене Дырявой Чаши, "
        "выпил воду из всех орденских аквариумов и получил силу, "
        "предназначенную для 600 человек. Несколько лет терроризировал Ехо, "
        "не контролируя себя. Джуффин спас его и отправил в Хумгат, "
        "после чего родился нынешний сдержанный Шурф. "
        "Но Безумный Рыбник — не просто прошлое, а автономная часть "
        "его сознания, подверженная безумию и импульсивности. "
        "Шурф постоянно сдерживает её, но в определённых обстоятельствах "
        "Рыбник может выйти на поверхность. Женат на Хельне, поэтессе. "
        "Стал Великим Магистром Ордена Семилистника. "
        "\n"
        "Сэр Мелифаро — коллега и друг, балагур, щёголь, "
        "любит яркую одежду (меняет по несколько лоохи в день), "
        "обожает подначивать тебя и Шурфа. Его старший брат — пират. "
        "\n"
        "Сэр Кофа Йох — Мастер Кушающий-Слушающий. Знаток трактиров, "
        "кухни и вообще всего на свете. Научил тебя ценить еду Ехо. "
        "Ведёт слежку, сидя в ресторанах. Старый, мудрый, ироничный. "
        "\n"
        "Сэр Нумминорих Кута — коллега по Тайному Сыску, обладает "
        "невероятным обонянием (буквально чует магию и эмоции). "
        "Добродушный, восторженный, немного наивный. "
        "\n"
        "Леди Меламори Блимм — бывшая возлюбленная, коллега по Тайному Сыску. "
        "Была обречена судьбой на разлуку с тобой. "
        "Уехала на Арварох с сэром Алотхо Аллирохом. "
        "В последние годы практически исчезла из твоей жизни. "
        "\n"
        "Леди Теххи Шекк — была дорогим тебе человеком, дочь Лойсо Пондохвы. "
        "Мертва. Стала призраком, живёт в Городе в Горах. "
        "Бистро «Армстронг и Элла» было названо в честь твоих котов. "
        "\n"
        "Леди Сотофа Ханемер — могущественная колдунья, одна из Великих Древних. "
        "Наставница и покровительница, помогает тебе понять свою природу. "
        "\n"
        "Лойсо Пондохва — один из самых могущественных магов в истории Мира, "
        "бывший Великий Магистр Ордена Водяной Вороны. "
        "Фигура сложная и неоднозначная — не просто злодей. Отец Теххи. "
        "\n"
        "Король Гуриг VIII — молодой, дружелюбный и умный монарх. "
        "\n"
        "Нуфлин Мони Мах — бывший Великий Магистр Ордена Семилистника. "
        "Вкрадчив, коварен, любит говорить «таки да» "
        "и вставлять палки в колёса Тайному Сыску. "
        "\n\n"
        "БЫТ: Живёшь в Мохнатом Доме — трёхэтажном доме с башенкой на крыше, "
        "увитом растением гламитариунмайоха. "
        "Сидишь в башне и смотришь на город. "
        "Ездишь на амобилере (лихо и быстро, пугая окружающих). "
        "Носишь лоохи (местная одежда, нечто вроде мантии). "
        "Пьёшь камру — горький горячий напиток, постоянно и помногу. "
        "Ешь невероятное количество еды, заказываешь из «Обжоры Бунбы». "
        "Пёс Друппи — огромная белая овчарка Пустых Земель, беззаветно предан тебе. "
        "Коты Армстронг и Элла — в мире Ехо коты крупные, размером с рысь. "
        "Также в доме живёт говорящий пёс Дримарондо. "
        "\n\n"
        "МИР: Кодекс Хрембера, ограничивавший ступени Очевидной Магии, "
        "фактически больше не действует. Магия вернулась в повседневную жизнь, "
        "повара снова колдуют на кухне, в Ехо появились гости из других миров. "
        "Тайный Сыск по-прежнему расследует магические преступления. "
        "Безмолвная речь — обычный способ общения на расстоянии. "
        "В мире Ехо нет телефонов, электричества, кофе, сигарет. "
        "Транспорт — амобилеры (движутся на магических кристаллах). "
        "Трактиры — центр общественной жизни. "
        "Тёмная Сторона — изнанка реальности, есть у всего: людей, городов. "
        "\n\n"
        "ХАРАКТЕР И МАНЕРА РЕЧИ: "
        "Ироничен и самоироничен — шутишь даже в опасных ситуациях. "
        "Разговорный, лёгкий язык с философскими отступлениями. "
        "Часто рефлексируешь — внутренние монологи о природе вещей, "
        "смысле жизни, собственных переживаниях. "
        "Склонен к мгновенным переходам от экзистенциальной тоски "
        "к восторгу от простых вещей. "
        "Обожаешь описывать еду, напитки и атмосферу трактиров. "
        "Ворчишь по поводу необходимости рано вставать или куда-то ехать, "
        "но всегда делаешь что нужно. "
        "Склонен преуменьшать свои способности и заслуги. "
        "Используешь выражения из Мира Паука (Земли), "
        "которые собеседники в Ехо не понимают. "
        "Восклицания мира Ехо: «Грёбаные Магистры!», "
        "«Побойтесь Тёмных Магистров!», «Дырку над мной в небе!», "
        "«Хвала Магистрам!». "
        "Верный друг, готов рискнуть жизнью ради близких. "
        "При всей внешней лёгкости — глубоко чувствующий. "
        "Обладаешь тем, что Фрай называет «инстинктом милосердия» — "
        "не можешь пройти мимо чужой беды, даже если это создаёт проблемы."
    )


# ──────────────────────────────────────────────
# Шаг 0: Утилиты
# ──────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Оценка количества токенов.
    Для русского текста ~1.5-2 токена на слово (кириллица дробится сильнее латиницы).
    Используем символьную оценку как более стабильную: ~3.5 символа на токен для русского."""
    return max(len(text) // 3, 1)


def text_hash(text: str) -> str:
    """Короткий хеш текста для дедупликации."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]


def now_str() -> str:
    """Текущее время в формате HH:MM:SS."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def fmt_duration(seconds: float) -> str:
    """Форматирует секунды в читаемую строку."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


_LOG_LOCK = threading.Lock()
_STOP_REQUESTED = threading.Event()
SECTION_BREAK_MARKER = "[[SECTION_BREAK]]"


class GracefulInterrupt(Exception):
    """Мягкое прерывание по запросу пользователя."""


def stop_requested() -> bool:
    """Возвращает True, если пользователь запросил остановку."""
    return _STOP_REQUESTED.is_set()


def request_stop() -> bool:
    """Ставит флаг остановки. Возвращает True только при первом запросе."""
    first = not _STOP_REQUESTED.is_set()
    _STOP_REQUESTED.set()
    return first


def handle_shutdown_signal(sig, frame):
    """Обрабатывает Ctrl+C / SIGTERM без потери уже сохранённого прогресса."""
    first = request_stop()
    try:
        sig_name = signal.Signals(sig).name
    except Exception:
        sig_name = str(sig)

    if first:
        print(
            f"\n[{now_str()}] Получен {sig_name}. "
            "Останавливаю пайплайн после сохранения текущего прогресса. "
            "Нажми Ctrl+C ещё раз для немедленного выхода."
        )
        raise KeyboardInterrupt

    print(f"\n[{now_str()}] Повторный {sig_name}. Немедленное завершение.")
    raise SystemExit(130)


def preview_text(text: str, limit: int = 100) -> str:
    """Короткий однострочный превью текста для логов."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit - 3].rstrip() + "..."


def strip_text(value: Any) -> str:
    """Безопасно приводит LLM-поля к строке и обрезает пробелы."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def log_event(message: str):
    """Потокобезопасный лог с timestamp."""
    with _LOG_LOCK:
        print(f"[{now_str()}] {message}")


def format_chunk_tag(idx: int, total: int) -> str:
    """Человекочитаемый идентификатор чанка."""
    return f"[фрагмент {idx + 1}/{total}]"


def clean_json_text(text: str) -> str:
    """Очищает ответ модели от markdown-обёрток и служебного префикса."""
    cleaned = text.strip().lstrip("\ufeff")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"^\s*json\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_balanced_json_fragment(text: str, opening: str) -> Optional[str]:
    """Вытаскивает первый корректно сбалансированный JSON-фрагмент."""
    closing = "}" if opening == "{" else "]"
    start = text.find(opening)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for pos in range(start, len(text)):
        ch = text[pos]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start:pos + 1]

    return None


def parse_json_response(
    response: str,
    *,
    expect: str,
    log_prefix: str = "",
) -> tuple[Optional[Any], str]:
    """Пытается надёжно распарсить JSON из ответа модели."""
    cleaned = clean_json_text(response)
    candidates: list[tuple[str, str]] = []
    seen = set()

    def add_candidate(strategy: str, candidate: Optional[str]):
        if not candidate:
            return
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append((strategy, candidate))

        no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", candidate)
        if no_trailing_commas != candidate and no_trailing_commas not in seen:
            seen.add(no_trailing_commas)
            candidates.append((f"{strategy}+fix_commas", no_trailing_commas))

    add_candidate("raw", cleaned)

    opening = "{" if expect == "object" else "["
    add_candidate("balanced", extract_balanced_json_fragment(cleaned, opening))

    last_error: Optional[json.JSONDecodeError] = None
    last_candidate = ""

    for strategy, candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            last_candidate = candidate
            continue

        if expect == "object" and isinstance(data, dict):
            if strategy != "raw" and log_prefix:
                log_event(f"{log_prefix} JSON восстановлен стратегией `{strategy}`")
            return data, strategy

        if expect == "array" and isinstance(data, list):
            if strategy != "raw" and log_prefix:
                log_event(f"{log_prefix} JSON восстановлен стратегией `{strategy}`")
            return data, strategy

    if log_prefix:
        if last_error is not None:
            start = max(0, last_error.pos - 80)
            end = min(len(last_candidate), last_error.pos + 80)
            around = preview_text(last_candidate[start:end], 160)
            log_event(
                f"{log_prefix} JSON parse error: {last_error.msg} "
                f"(позиция {last_error.pos}). Контекст: {around}"
            )
        else:
            log_event(f"{log_prefix} JSON не найден в ответе модели")

    return None, "failed"


def append_jsonl(path: Path, items: list[dict]):
    """Дописывает список JSON-объектов в jsonl-файл."""
    if not items:
        return
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, log_prefix: str = "") -> list[dict]:
    """Читает jsonl-файл, пропуская битые хвосты после аварийной остановки."""
    if not path.exists():
        return []

    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if log_prefix:
                    log_event(
                        f"{log_prefix} битая строка JSONL пропущена "
                        f"(line {lineno}: {exc})"
                    )
    return items


_METADATA_RECENT_EVENTS_LIMIT = 80
_LLM_TRACE_LOCK = threading.Lock()
_LLM_TRACE_COUNTER = 0


def now_iso_str() -> str:
    """Текущее локальное время в ISO-формате."""
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: Any):
    """Атомарно пишет JSON на диск, чтобы metadata не портилась при остановке."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def make_json_safe(value: Any) -> Any:
    """Преобразует произвольный объект в JSON-safe структуру."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return make_json_safe(model_dump())
        except Exception:
            pass
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        return make_json_safe(value_dict)
    return repr(value)


def make_trace_slug(text: str, default: str = "llm") -> str:
    """Делает безопасный короткий slug для имён trace-файлов."""
    slug = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", text or "").strip("._-")
    if not slug:
        slug = default
    return slug[:80]


def next_llm_trace_id(log_prefix: str = "") -> str:
    """Возвращает уникальный id для одного логического LLM-вызова."""
    global _LLM_TRACE_COUNTER
    with _LLM_TRACE_LOCK:
        _LLM_TRACE_COUNTER += 1
        idx = _LLM_TRACE_COUNTER
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    slug = make_trace_slug(log_prefix, default="llm")
    return f"{timestamp}_{idx:06d}_{slug}"


def get_llm_trace_dir(config: Config) -> Path:
    """Папка для trace-файлов LLM."""
    global_paths = get_global_output_paths(config.output_dir)
    trace_dir = global_paths["llm_traces_dir"]
    if config.llm_trace_run_id:
        trace_dir = trace_dir / config.llm_trace_run_id
    return trace_dir


def init_llm_trace(
    config: Config,
    trace_id: str,
    payload: dict,
) -> tuple[Optional[Path], dict]:
    """Создаёт trace одного логического LLM-вызова и пишет initial request."""
    if not config.llm_trace_enabled:
        return None, {}
    trace_dir = get_llm_trace_dir(config)
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{trace_id}.json"
    trace = {
        "trace_id": trace_id,
        "created_at": now_iso_str(),
        "updated_at": now_iso_str(),
        **make_json_safe(payload),
        "attempts": [],
    }
    write_json_atomic(path, trace)
    return path, trace


def append_llm_trace_attempt(
    path: Optional[Path],
    trace: dict,
    payload: dict,
) -> Optional[Path]:
    """Добавляет в trace результат одной попытки LLM-вызова."""
    if path is None:
        return None
    trace.setdefault("attempts", []).append(make_json_safe(payload))
    trace["updated_at"] = now_iso_str()
    write_json_atomic(path, trace)
    return path


def merge_metadata_updates(target: dict, updates: dict):
    """Рекурсивно мержит metadata-обновления."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_metadata_updates(target[key], value)
        else:
            target[key] = value


def compact_metadata_event_details(data: dict) -> dict:
    """Оставляет в истории metadata только компактные JSON-safe детали."""
    compact: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, dict):
            nested = {
                nested_key: nested_value
                for nested_key, nested_value in value.items()
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None
            }
            if nested:
                compact[key] = nested
    return compact


def update_metadata_snapshot(
    state: dict,
    path: Path,
    history_path: Optional[Path] = None,
    *,
    event_type: str = "",
    message: str = "",
    **updates,
) -> dict:
    """Обновляет metadata runtime-state и сразу пишет его на диск."""
    if updates:
        merge_metadata_updates(state, updates)

    timestamp = now_iso_str()
    state["updated_at"] = timestamp
    state.setdefault("event_count", 0)

    if event_type:
        state["event_count"] += 1
        event = {
            "idx": state["event_count"],
            "ts": timestamp,
            "type": event_type,
            "status": state.get("status"),
            "current_stage": state.get("current_stage"),
            "current_book": state.get("current_book"),
        }
        if message:
            event["message"] = message
        details = compact_metadata_event_details(updates)
        if details:
            event["details"] = details

        recent_events = state.setdefault("recent_events", [])
        recent_events.append(event)
        if len(recent_events) > _METADATA_RECENT_EVENTS_LIMIT:
            del recent_events[:-_METADATA_RECENT_EVENTS_LIMIT]
        state["last_event"] = event
        if history_path is not None:
            append_jsonl(history_path, [event])

    write_json_atomic(path, state)
    return state


def parse_fb2(fb2_content: bytes) -> str:
    """
    Парсит FB2 XML и извлекает чистый текст из <body>.
    FB2 использует namespace, поэтому ищем теги с подстановкой.
    """
    try:
        root = ET.fromstring(fb2_content)
    except ET.ParseError:
        # Fallback: попробовать перекодировать из windows-1251
        try:
            decoded = fb2_content.decode("windows-1251")
            # Заменяем encoding в XML-заголовке на utf-8
            decoded = re.sub(
                r'encoding=["\']windows-1251["\']',
                'encoding="utf-8"',
                decoded,
                flags=re.IGNORECASE,
            )
            root = ET.fromstring(decoded.encode("utf-8"))
            print(f"    (перекодировано из windows-1251)")
        except Exception as e2:
            print(f"    Ошибка парсинга FB2 XML: {e2}")
            return ""

    # FB2 namespace — бывает разный, определяем из корневого элемента
    ns = ""
    match = re.match(r"\{(.+?)\}", root.tag)
    if match:
        ns = match.group(1)

    def tag(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    lines = []

    def append_line(text: str):
        text = re.sub(r"\s+", " ", text or "").strip()
        if text:
            lines.append(text)

    def append_blank_line():
        if lines and lines[-1] != "":
            lines.append("")

    def append_paragraph(text: str):
        append_line(text)
        append_blank_line()

    def append_section_break():
        if not lines:
            return
        if lines[-1] == SECTION_BREAK_MARKER:
            return
        append_blank_line()
        lines.append(SECTION_BREAK_MARKER)
        append_blank_line()

    def walk(element):
        if element.tag == tag("binary"):
            return

        if element.tag == tag("section"):
            append_section_break()
            for child in element:
                walk(child)
            append_blank_line()
            return

        if element.tag == tag("title"):
            title_parts = []
            for child in element:
                if child.tag == tag("p"):
                    text = "".join(child.itertext()).strip()
                    if text:
                        title_parts.append(text)
            if title_parts:
                append_paragraph(" ".join(title_parts))
            return

        if element.tag == tag("epigraph"):
            for child in element:
                if child.tag in (tag("p"), tag("text-author")):
                    text = "".join(child.itertext()).strip()
                    if text:
                        append_paragraph(text)
            return

        if element.tag in (tag("p"), tag("v"), tag("subtitle"), tag("text-author")):
            append_paragraph("".join(element.itertext()).strip())
            return

        for child in element:
            walk(child)

    for body in root.iter(tag("body")):
        body_name = (body.attrib.get("name") or "").strip().lower()
        if body_name in {"notes", "comments"}:
            continue
        for child in body:
            walk(child)

    return "\n".join(lines)


def load_fb2_file(filepath: Path) -> Optional[str]:
    """Загружает .fb2 или .fb2.zip файл, возвращает текст."""
    name_lower = filepath.name.lower()
    ext = filepath.suffix.lower()  # только последний суффикс, безопасно для "1. Чужак.fb2"

    if name_lower.endswith(".fb2.zip") or (ext == ".zip"):
        # FB2 внутри ZIP-архива
        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                fb2_names = [n for n in zf.namelist() if n.lower().endswith(".fb2")]
                if not fb2_names:
                    print(f"    В архиве {filepath.name} нет .fb2 файлов")
                    return None
                fb2_data = zf.read(fb2_names[0])
                return parse_fb2(fb2_data)
        except zipfile.BadZipFile:
            print(f"    Повреждённый архив: {filepath.name}")
            return None

    elif ext == ".fb2":
        fb2_data = filepath.read_bytes()
        return parse_fb2(fb2_data)

    return None


def load_books(books_dir: str) -> list[tuple[str, str]]:
    """Рекурсивно загружает .fb2, .fb2.zip и .txt из папки и подпапок.
    Возвращает [(относительный_путь, текст)]."""
    books = []
    books_path = Path(books_dir)

    if not books_path.exists():
        print(f"Папка {books_dir} не найдена. Создаю...")
        books_path.mkdir(parents=True)
        print(f"Положи .fb2 или .txt файлы книг в {books_path.absolute()}")
        return []

    # Рекурсивно собираем все поддерживаемые файлы
    def is_supported(f: Path) -> bool:
        if not f.is_file():
            return False
        name_lower = f.name.lower()
        ext = f.suffix.lower()
        return (
            ext == ".txt"
            or ext == ".fb2"
            or name_lower.endswith(".fb2.zip")
            or ext == ".zip"
        )

    files = sorted(f for f in books_path.rglob("*") if is_supported(f))

    for f in files:
        text = None

        if f.suffix.lower() == ".txt":
            text = f.read_text(encoding="utf-8", errors="replace")
        else:
            text = load_fb2_file(f)

        # Относительный путь для читаемости
        rel_path = str(f.relative_to(books_path))

        if text and len(text.strip()) > 100:
            books.append((rel_path, text))
            print(f"  Загружена: {rel_path} ({len(text):,} символов, ~{estimate_tokens(text):,} токенов)")
        elif text is not None:
            print(f"  Пропущена (слишком мало текста): {rel_path}")

    return books


def get_book_stem(book_name: str) -> str:
    """Плоское имя книги для файлов результатов."""
    return Path(book_name).with_suffix("").as_posix().replace("/", "--")


def get_book_output_paths(output_dir: str, book_name: str) -> dict[str, Path]:
    """Пути выходных файлов для одной книги."""
    stem = get_book_stem(book_name)
    return {
        "voice": Path(output_dir) / f"voice_{stem}.jsonl",
        "knowledge": Path(output_dir) / f"knowledge_{stem}.json",
        "knowledge_stream": Path(output_dir) / f"knowledge_{stem}.jsonl",
        "chunks": Path(output_dir) / f"chunks_{stem}.jsonl",
        "synth": Path(output_dir) / f"synth_{stem}.jsonl",
        "synth_progress": Path(output_dir) / f"synth_progress_{stem}.jsonl",
        "done": Path(output_dir) / f"done_{stem}.marker",
        "voice_txt": Path(output_dir) / f"voice_{stem}.txt",
        "knowledge_txt": Path(output_dir) / f"knowledge_{stem}.txt",
        "synth_txt": Path(output_dir) / f"synth_{stem}.txt",
    }


def get_global_output_paths(output_dir: str) -> dict[str, Path]:
    """Пути итоговых глобальных артефактов."""
    base = Path(output_dir)
    return {
        "dataset": base / "dataset.jsonl",
        "dataset_txt": base / "dataset.txt",
        "knowledge_raw": base / "knowledge_base_raw.json",
        "knowledge_raw_txt": base / "knowledge_base_raw.txt",
        "knowledge": base / "knowledge_base.json",
        "knowledge_txt": base / "knowledge_base.txt",
        "timeline_raw": base / "timeline_resolution_raw.json",
        "timeline_graph": base / "timeline_graph.json",
        "timeline_graph_txt": base / "timeline_graph.txt",
        "metadata": base / "metadata.json",
        "metadata_history": base / "metadata_history.jsonl",
        "llm_traces_dir": base / "llm_traces",
    }


def load_book_knowledge_for_global_base(
    output_dir: str,
    book_name: str,
    log_prefix: str = "",
) -> list[dict]:
    """Загружает knowledge одной книги из сохранённых артефактов."""
    paths = get_book_output_paths(output_dir, book_name)
    items: list[dict] = []

    if paths["knowledge"].exists():
        try:
            with open(paths["knowledge"], encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                items = [item for item in data if isinstance(item, dict)]
        except Exception as exc:
            if log_prefix:
                log_event(f"{log_prefix} не удалось прочитать {paths['knowledge'].name}: {exc}")

    if not items and paths["knowledge_stream"].exists():
        items = [item for item in read_jsonl(paths["knowledge_stream"], log_prefix=log_prefix) if isinstance(item, dict)]

    return items


def load_book_chunk_records(
    output_dir: str,
    book_name: str,
    log_prefix: str = "",
) -> list[dict]:
    """Загружает сохранённые chunk-records книги для timeline resolution."""
    paths = get_book_output_paths(output_dir, book_name)
    records: list[dict] = []

    for record in read_jsonl(paths["chunks"], log_prefix=log_prefix):
        idx = record.get("idx")
        if not isinstance(idx, int):
            continue
        chunk_text = record.get("chunk_text", "")
        if not isinstance(chunk_text, str):
            chunk_text = ""
        records.append({
            "idx": idx,
            "chapter": strip_text(record.get("chapter", "")),
            "chunk_text": chunk_text,
            "knowledge": [item for item in record.get("knowledge", []) if isinstance(item, dict)],
        })

    records.sort(key=lambda item: item["idx"])
    return records


def write_global_knowledge_snapshot(
    output_dir: str,
    raw_knowledge: list[dict],
    narrator: str = "Макс",
    log_prefix: str = "",
    progress_callback: Optional[Any] = None,
    *,
    completion_event_type: str = "global_kb_complete",
) -> tuple[list[dict], list[dict]]:
    """Пишет текущий snapshot общей базы знаний в JSON и человекочитаемом виде."""
    global_paths = get_global_output_paths(output_dir)
    normalized_raw = [item for item in raw_knowledge if isinstance(item, dict)]

    with open(global_paths["knowledge_raw"], "w", encoding="utf-8") as f:
        json.dump(normalized_raw, f, ensure_ascii=False, indent=2)
    save_readable_knowledge(normalized_raw, str(global_paths["knowledge_raw_txt"]))
    if progress_callback is not None:
        progress_callback(
            "global_kb_raw_written",
            raw_knowledge_facts=len(normalized_raw),
        )

    canonicalized_knowledge = canonicalize_global_knowledge(
        normalized_raw,
        narrator=narrator,
        log_prefix=f"{log_prefix}[canonicalize]" if log_prefix else "",
    )
    unique_knowledge = deduplicate_knowledge(canonicalized_knowledge)

    with open(global_paths["knowledge"], "w", encoding="utf-8") as f:
        json.dump(unique_knowledge, f, ensure_ascii=False, indent=2)
    save_readable_knowledge(unique_knowledge, str(global_paths["knowledge_txt"]))
    if progress_callback is not None:
        progress_callback(
            completion_event_type,
            raw_knowledge_facts=len(normalized_raw),
            knowledge_facts=len(unique_knowledge),
        )

    return normalized_raw, unique_knowledge


def build_global_knowledge_base(
    output_dir: str,
    book_names: list[str],
    narrator: str = "Макс",
    log_prefix: str = "",
    progress_callback: Optional[Any] = None,
) -> tuple[list[dict], list[dict]]:
    """Собирает общую базу знаний из per-book файлов и отдельно её дедуплицирует."""
    raw_knowledge: list[dict] = []
    loaded_books = 0

    for book_name in book_names:
        book_items = load_book_knowledge_for_global_base(
            output_dir,
            book_name,
            log_prefix=f"{log_prefix}[{get_book_stem(book_name)}]" if log_prefix else "",
        )
        if not book_items:
            continue
        book_items = ensure_knowledge_source_defaults(book_items, book_name)
        loaded_books += 1
        raw_knowledge.extend(book_items)
        if progress_callback is not None:
            progress_callback(
                "global_kb_book_loaded",
                book_name=book_name,
                loaded_books=loaded_books,
                raw_knowledge_facts=len(raw_knowledge),
            )

    raw_knowledge, unique_knowledge = write_global_knowledge_snapshot(
        output_dir,
        raw_knowledge,
        narrator=narrator,
        log_prefix=log_prefix,
        progress_callback=(
            None if progress_callback is None else
            lambda event_type, **event_data: progress_callback(
                event_type,
                loaded_books=loaded_books,
                **event_data,
            )
        ),
        completion_event_type="global_kb_complete",
    )

    if log_prefix:
        log_event(
            f"{log_prefix} global kb: книг={loaded_books}, "
            f"сырых фактов={len(raw_knowledge)}, уникальных={len(unique_knowledge)}"
        )

    return raw_knowledge, unique_knowledge


TIMELINE_RESOLUTION_SYSTEM = """Ты — помощник по timeline resolution и построению графа знаний по книгам Макса Фрая.
Получаешь факты одной книги/главы и соответствующие текстовые фрагменты.
Твоя задача — собрать локальный граф событий, персонажей, мест, предметов, магии и связей между ними.
Отвечай СТРОГО в формате JSON. Никакого текста до или после JSON."""

TIMELINE_RESOLUTION_PROMPT = """У тебя есть факты одной книги/главы и исходные текстовые фрагменты.

КНИГА: {book_name}
ГЛАВА: {chapter}

ФАКТЫ:
{facts_block}

ТЕКСТОВЫЕ ФРАГМЕНТЫ:
{chunks_block}

Построй ЛОКАЛЬНЫЙ граф для этого фрагмента книги.

Верни JSON объект:
{{
  "entities": [
    {{
      "label": "каноничное имя сущности",
      "type": "character|place|item|magic|creature|history_subject|organization|unknown"
    }}
  ],
  "events": [
    {{
      "local_id": "E1",
      "label": "краткое название события",
      "summary": "что произошло и почему это важно",
      "time_scope": "past|current|change|ended|timeless|unclear",
      "chunk_indices": [0, 1],
      "participants": ["Макс", "Джуффин Халли"],
      "places": ["Дом у Моста"],
      "objects": ["камра", "Смертный Шар"]
    }}
  ],
  "relations": [
    {{
      "source": "E1 или точный label сущности",
      "target": "E2 или точный label сущности",
      "type": "before|after|involves|occurs_in|uses|affects|changes_state_of|belongs_to|located_in|related_to",
      "evidence": "краткая опора на факт или текст",
      "confidence": "explicit|inferred"
    }}
  ]
}}

ПРАВИЛА:
— Используй только факты и текстовые фрагменты, которые даны выше.
— Не придумывай новых сущностей, если их нельзя уверенно назвать.
— Сохраняй разные состояния мира и персонажей как РАЗНЫЕ события/связи, если состояние явно меняется.
— Если порядок событий неясен, не создавай before/after.
— В отношениях используй точные labels из facts/entities. Для событий используй local_id.
— В entities перечисляй только полезные сущности, которые реально участвуют в связях или событиях.
— Если событий не видно, entities и relations всё равно можно вернуть.

JSON:"""

_TIMELINE_RELATION_TYPES = {
    "before",
    "after",
    "involves",
    "occurs_in",
    "uses",
    "affects",
    "changes_state_of",
    "belongs_to",
    "located_in",
    "related_to",
}


def timeline_entity_type_for_category(category: str) -> str:
    """Нормализует тип сущности для графа."""
    category = strip_text(category)
    mapping = {
        "character": "character",
        "place": "place",
        "custom": "item",
        "magic": "magic",
        "creature": "creature",
        "history": "history_subject",
        "event": "event_subject",
    }
    return mapping.get(category, "unknown")


def timeline_entity_node_id(label: str, category: str = "") -> str:
    """Детерминированный id сущности графа."""
    node_type = timeline_entity_type_for_category(category)
    normalized = normalize_subject_for_dedup(label) or text_hash(label)
    return f"{node_type}::{normalized}"


def timeline_event_node_id(book_name: str, chapter: str, order: int) -> str:
    """Детерминированный id события графа."""
    return f"event::{get_book_stem(book_name)}::{text_hash(chapter)}::{order:03d}"


def timeline_chunk_excerpt(text: str, token_limit: int) -> str:
    """Компактный excerpt чанка для timeline prompt."""
    if not text:
        return ""
    if estimate_tokens(text) <= token_limit:
        return text.strip()

    approx_chars = max(token_limit * 4, 600)
    head = text[:approx_chars].rstrip()
    tail = text[-approx_chars:].lstrip()
    return f"{head}\n...\n{tail}".strip()


def format_timeline_facts_for_prompt(facts: list[dict], limit: int = 36) -> str:
    """Форматирует факты одной главы для timeline resolution."""
    lines = []
    sorted_facts = sorted(
        facts,
        key=lambda item: (
            item.get("chunk_idx") if isinstance(item.get("chunk_idx"), int) else 10 ** 9,
            strip_text(item.get("subject", "")),
            strip_text(item.get("fact", "")),
        ),
    )
    for item in sorted_facts[:limit]:
        chunk_idx = item.get("chunk_idx")
        chunk_note = f"chunk_idx={chunk_idx}" if isinstance(chunk_idx, int) else "chunk_idx=?"
        category = strip_text(item.get("category", "unknown"))
        subject = strip_text(item.get("subject", ""))
        fact = strip_text(item.get("fact", ""))
        time_scope = normalize_time_scope(
            item.get("time_scope", ""),
            fact=fact,
            category=category,
        )
        chapter = strip_text(item.get("chapter", ""))
        lines.append(
            f"- [{chunk_note}] {category} / {subject} [{time_scope}]"
            + (f" / chapter={chapter}" if chapter else "")
            + f": {fact}"
        )
    if len(sorted_facts) > limit:
        lines.append(f"- ... ещё {len(sorted_facts) - limit} фактов")
    return "\n".join(lines) if lines else "(нет фактов)"


def format_timeline_chunks_for_prompt(chunk_records: list[dict], token_limit: int) -> str:
    """Форматирует текстовые chunk-фрагменты для timeline resolution."""
    blocks = []
    for record in chunk_records:
        chunk_idx = record.get("idx")
        chapter = strip_text(record.get("chapter", ""))
        excerpt = timeline_chunk_excerpt(record.get("chunk_text", ""), token_limit)
        if not excerpt:
            continue
        blocks.append(
            f"[CHUNK idx={chunk_idx}"
            + (f" | chapter={chapter}" if chapter else "")
            + f"]\n{excerpt}"
        )
    return "\n\n".join(blocks) if blocks else "(нет текстовых фрагментов)"


def timeline_group_sort_key(group: dict) -> tuple:
    """Ключ сортировки групп timeline."""
    first_chunk = min(group.get("chunk_indices") or [10 ** 9])
    return (group.get("book_name", ""), first_chunk, group.get("chapter", ""))


def build_timeline_groups(
    output_dir: str,
    raw_knowledge: list[dict],
    book_names: list[str],
    config: Config,
) -> list[dict]:
    """Группирует факты для offline timeline resolution по книге и главе."""
    chunk_records_by_book = {
        book_name: load_book_chunk_records(output_dir, book_name)
        for book_name in book_names
    }

    groups: dict[tuple[str, str], dict] = {}
    for item in raw_knowledge:
        book_name = strip_text(item.get("source_book", ""))
        if not book_name:
            continue
        chapter = strip_text(item.get("chapter", "")) or default_chapter_label(book_name)
        key = (book_name, chapter)
        group = groups.setdefault(key, {
            "book_name": book_name,
            "chapter": chapter,
            "facts": [],
            "chunk_indices": set(),
        })
        group["facts"].append(item)
        chunk_idx = item.get("chunk_idx")
        if isinstance(chunk_idx, int):
            group["chunk_indices"].add(chunk_idx)

    prepared_groups = []
    for key, group in groups.items():
        book_name = group["book_name"]
        records_by_idx = {
            record["idx"]: record
            for record in chunk_records_by_book.get(book_name, [])
            if isinstance(record.get("idx"), int)
        }
        expanded_indices = set(group["chunk_indices"])
        for idx in list(group["chunk_indices"]):
            for delta in range(1, max(config.timeline_neighbor_chunks, 0) + 1):
                expanded_indices.add(idx - delta)
                expanded_indices.add(idx + delta)

        chunk_records = [
            records_by_idx[idx]
            for idx in sorted(expanded_indices)
            if idx in records_by_idx
        ][:max(config.timeline_group_max_chunks, 1)]

        prepared_groups.append({
            "book_name": book_name,
            "chapter": group["chapter"],
            "facts": sorted(
                group["facts"],
                key=lambda item: (
                    item.get("chunk_idx") if isinstance(item.get("chunk_idx"), int) else 10 ** 9,
                    strip_text(item.get("subject", "")),
                ),
            ),
            "chunk_indices": sorted(group["chunk_indices"]),
            "chunk_records": chunk_records,
        })

    prepared_groups.sort(key=timeline_group_sort_key)
    return prepared_groups


def build_fallback_timeline_events(facts: list[dict]) -> list[dict]:
    """Создаёт грубые event-узлы из фактов, если LLM timeline resolution не сработал."""
    events = []
    for idx, item in enumerate(facts, 1):
        category = strip_text(item.get("category", ""))
        fact = strip_text(item.get("fact", ""))
        if not fact:
            continue
        time_scope = normalize_time_scope(item.get("time_scope", ""), fact=fact, category=category)
        if category not in {"event", "history"} and time_scope not in {"past", "current", "change", "ended"}:
            continue
        subject = strip_text(item.get("subject", ""))
        chunk_idx = item.get("chunk_idx")
        events.append({
            "local_id": f"E{idx}",
            "label": preview_text(fact, 72),
            "summary": fact,
            "time_scope": time_scope,
            "chunk_indices": [chunk_idx] if isinstance(chunk_idx, int) else [],
            "participants": [subject] if subject else [],
            "places": [],
            "objects": [],
        })
    return events[:12]


def normalize_timeline_relation_type(value: str) -> str:
    """Нормализует тип relation для графа timeline."""
    normalized = normalize_dedup_text(value)
    aliases = {
        "before": "before",
        "after": "after",
        "involves": "involves",
        "occurs in": "occurs_in",
        "occurs_in": "occurs_in",
        "uses": "uses",
        "affects": "affects",
        "changes state of": "changes_state_of",
        "changes_state_of": "changes_state_of",
        "belongs to": "belongs_to",
        "belongs_to": "belongs_to",
        "located in": "located_in",
        "located_in": "located_in",
        "related to": "related_to",
        "related_to": "related_to",
    }
    relation_type = aliases.get(normalized, normalized.replace(" ", "_"))
    return relation_type if relation_type in _TIMELINE_RELATION_TYPES else "related_to"


def resolve_timeline_group(
    client: OpenAI,
    config: Config,
    group: dict,
    *,
    log_prefix: str = "",
) -> dict:
    """Прогоняет один chapter-level блок через LLM для timeline resolution."""
    facts_block = format_timeline_facts_for_prompt(
        group.get("facts", []),
        limit=max(config.timeline_group_max_facts, 1),
    )
    chunks_block = format_timeline_chunks_for_prompt(
        group.get("chunk_records", []),
        token_limit=max(config.timeline_chunk_excerpt_tokens, 120),
    )

    response = call_llm(
        client,
        config,
        TIMELINE_RESOLUTION_SYSTEM,
        TIMELINE_RESOLUTION_PROMPT.format(
            book_name=group.get("book_name", ""),
            chapter=group.get("chapter", ""),
            facts_block=facts_block,
            chunks_block=chunks_block,
        ),
        max_tokens=config.max_tokens_timeline_resolution,
        response_format="json" if _use_ollama_native else None,
        log_prefix=log_prefix,
        temperature=0.1,
    )

    fallback = {
        "entities": [],
        "events": build_fallback_timeline_events(group.get("facts", [])),
        "relations": [],
    }
    if response is None:
        return fallback

    data, _ = parse_json_response(response, expect="object", log_prefix=log_prefix)
    if not isinstance(data, dict):
        return fallback

    entities = data.get("entities", [])
    events = data.get("events", [])
    relations = data.get("relations", [])
    if not isinstance(entities, list):
        entities = []
    if not isinstance(events, list):
        events = []
    if not isinstance(relations, list):
        relations = []

    if not events:
        events = fallback["events"]

    return {
        "entities": [item for item in entities if isinstance(item, dict)],
        "events": [item for item in events if isinstance(item, dict)],
        "relations": [item for item in relations if isinstance(item, dict)],
    }


def ensure_timeline_entity_node(
    nodes_by_id: dict[str, dict],
    label: str,
    *,
    category: str = "",
    source_book: str = "",
) -> Optional[str]:
    """Добавляет/обновляет сущность графа и возвращает её id."""
    clean_label = strip_text(label)
    if not clean_label:
        return None

    node_id = timeline_entity_node_id(clean_label, category)
    node = nodes_by_id.setdefault(node_id, {
        "id": node_id,
        "label": clean_label,
        "type": timeline_entity_type_for_category(category),
        "source_books": [],
    })
    if source_book and source_book not in node["source_books"]:
        node["source_books"].append(source_book)
        node["source_books"].sort()
    if category and node.get("type") == "unknown":
        node["type"] = timeline_entity_type_for_category(category)
    return node_id


def add_timeline_edge(
    edges: list[dict],
    seen_edges: set[str],
    *,
    source: str,
    target: str,
    relation_type: str,
    book_name: str,
    chapter: str,
    evidence: str = "",
    confidence: str = "explicit",
    chunk_indices: Optional[list[int]] = None,
):
    """Добавляет ребро графа без дублей."""
    if not source or not target or source == target:
        return
    relation_type = normalize_timeline_relation_type(relation_type)
    payload = {
        "source": source,
        "target": target,
        "type": relation_type,
        "book": book_name,
        "chapter": chapter,
        "chunk_indices": sorted(chunk_indices or []),
    }
    edge_key = text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if edge_key in seen_edges:
        return
    seen_edges.add(edge_key)
    edges.append({
        **payload,
        "evidence": strip_text(evidence),
        "confidence": strip_text(confidence) or "explicit",
    })


def merge_timeline_resolution_outputs(
    raw_groups: list[dict],
    unique_knowledge: list[dict],
) -> dict:
    """Собирает единый граф из chapter-level timeline outputs."""
    nodes_by_id: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[str] = set()

    subject_category_map: dict[str, str] = {}
    for item in unique_knowledge:
        subject = strip_text(item.get("subject", ""))
        if not subject:
            continue
        subject_category_map[normalize_subject_for_dedup(subject)] = strip_text(item.get("category", ""))
        ensure_timeline_entity_node(
            nodes_by_id,
            subject,
            category=strip_text(item.get("category", "")),
            source_book=strip_text(item.get("source_book", "")),
        )

    event_count = 0
    for group in raw_groups:
        book_name = group.get("book_name", "")
        chapter = group.get("chapter", "")
        local_event_ids: dict[str, str] = {}

        for order, event in enumerate(group.get("events", []), 1):
            local_id = strip_text(event.get("local_id", "")) or f"E{order}"
            event_id = timeline_event_node_id(book_name, chapter, event_count + order)
            local_event_ids[local_id] = event_id
            nodes_by_id[event_id] = {
                "id": event_id,
                "label": strip_text(event.get("label", "")) or f"{chapter}: событие {order}",
                "type": "event",
                "book": book_name,
                "chapter": chapter,
                "summary": strip_text(event.get("summary", "")),
                "time_scope": normalize_time_scope(
                    event.get("time_scope", ""),
                    fact=strip_text(event.get("summary", "")) or strip_text(event.get("label", "")),
                    category="event",
                ),
                "chunk_indices": sorted(
                    idx for idx in event.get("chunk_indices", [])
                    if isinstance(idx, int)
                ),
            }
            for label_group, relation_type, category_hint in (
                (event.get("participants", []), "involves", "character"),
                (event.get("places", []), "occurs_in", "place"),
                (event.get("objects", []), "uses", "custom"),
            ):
                if not isinstance(label_group, list):
                    continue
                for label in label_group:
                    normalized_label = strip_text(label)
                    if not normalized_label:
                        continue
                    entity_id = ensure_timeline_entity_node(
                        nodes_by_id,
                        normalized_label,
                        category=subject_category_map.get(
                            normalize_subject_for_dedup(normalized_label),
                            category_hint,
                        ),
                        source_book=book_name,
                    )
                    if entity_id:
                        add_timeline_edge(
                            edges,
                            seen_edges,
                            source=event_id,
                            target=entity_id,
                            relation_type=relation_type,
                            book_name=book_name,
                            chapter=chapter,
                            chunk_indices=nodes_by_id[event_id].get("chunk_indices", []),
                        )
        event_count += len(group.get("events", []))

        for entity in group.get("entities", []):
            label = strip_text(entity.get("label", ""))
            if not label:
                continue
            ensure_timeline_entity_node(
                nodes_by_id,
                label,
                category=strip_text(entity.get("type", "")),
                source_book=book_name,
            )

        for relation in group.get("relations", []):
            source_ref = strip_text(relation.get("source", ""))
            target_ref = strip_text(relation.get("target", ""))
            if not source_ref or not target_ref:
                continue

            if source_ref in local_event_ids:
                source_id = local_event_ids[source_ref]
            else:
                source_id = ensure_timeline_entity_node(
                    nodes_by_id,
                    source_ref,
                    category=subject_category_map.get(normalize_subject_for_dedup(source_ref), ""),
                    source_book=book_name,
                )

            if target_ref in local_event_ids:
                target_id = local_event_ids[target_ref]
            else:
                target_id = ensure_timeline_entity_node(
                    nodes_by_id,
                    target_ref,
                    category=subject_category_map.get(normalize_subject_for_dedup(target_ref), ""),
                    source_book=book_name,
                )

            if source_id and target_id:
                add_timeline_edge(
                    edges,
                    seen_edges,
                    source=source_id,
                    target=target_id,
                    relation_type=strip_text(relation.get("type", "")),
                    book_name=book_name,
                    chapter=chapter,
                    evidence=strip_text(relation.get("evidence", "")),
                    confidence=strip_text(relation.get("confidence", "")) or "explicit",
                )

    nodes = sorted(
        nodes_by_id.values(),
        key=lambda item: (item.get("type", ""), item.get("label", ""), item.get("id", "")),
    )
    edges.sort(key=lambda item: (item.get("type", ""), item.get("source", ""), item.get("target", ""), item.get("book", "")))
    return {
        "nodes": nodes,
        "edges": edges,
        "groups": [
            {
                "book_name": group.get("book_name", ""),
                "chapter": group.get("chapter", ""),
                "facts": len(group.get("facts", [])),
                "events": len(group.get("events", [])),
                "relations": len(group.get("relations", [])),
            }
            for group in raw_groups
        ],
    }


def save_readable_timeline_graph(graph: dict, path: str):
    """Сохраняет граф timeline в человекочитаемом виде."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    groups = graph.get("groups", [])

    with open(path, "w", encoding="utf-8") as f:
        f.write("TIMELINE / KNOWLEDGE GRAPH\n")
        f.write(f"{'═' * 60}\n")
        f.write(f"Узлов: {len(nodes)}\n")
        f.write(f"Рёбер: {len(edges)}\n")
        f.write(f"Групп глав: {len(groups)}\n\n")

        by_type: dict[str, list[dict]] = {}
        for node in nodes:
            by_type.setdefault(node.get("type", "unknown"), []).append(node)

        for node_type in sorted(by_type.keys()):
            f.write(f"\n{'━' * 60}\n")
            f.write(f"  {node_type} ({len(by_type[node_type])})\n")
            f.write(f"{'━' * 60}\n\n")
            for node in by_type[node_type]:
                line = f"  ▸ {node.get('label', '')}"
                if node_type == "event":
                    summary = strip_text(node.get("summary", ""))
                    chapter = strip_text(node.get("chapter", ""))
                    time_scope = strip_text(node.get("time_scope", ""))
                    if chapter:
                        line += f" [{chapter}]"
                    if time_scope:
                        line += f" <{time_scope}>"
                    f.write(line + "\n")
                    if summary:
                        f.write(f"    — {summary}\n")
                else:
                    books = node.get("source_books", [])
                    f.write(line + ("\n" if not books else f" ({', '.join(books)})\n"))
            f.write("\n")

        f.write(f"\n{'━' * 60}\n  Связи ({len(edges)})\n{'━' * 60}\n\n")
        for edge in edges:
            f.write(
                f"  ▸ {edge.get('type', 'related_to')}: "
                f"{edge.get('source', '')} -> {edge.get('target', '')}\n"
            )
            evidence = strip_text(edge.get("evidence", ""))
            if evidence:
                f.write(f"    — {evidence}\n")


def build_timeline_resolution_artifacts(
    client: OpenAI,
    config: Config,
    output_dir: str,
    book_names: list[str],
    raw_knowledge: list[dict],
    unique_knowledge: list[dict],
    log_prefix: str = "",
    progress_callback: Optional[Any] = None,
) -> tuple[list[dict], dict]:
    """Отдельный offline-этап: timeline resolution и построение общего графа."""
    global_paths = get_global_output_paths(output_dir)
    if not config.timeline_resolution_enabled or not raw_knowledge:
        empty_graph = {"nodes": [], "edges": [], "groups": []}
        with open(global_paths["timeline_raw"], "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        with open(global_paths["timeline_graph"], "w", encoding="utf-8") as f:
            json.dump(empty_graph, f, ensure_ascii=False, indent=2)
        save_readable_timeline_graph(empty_graph, str(global_paths["timeline_graph_txt"]))
        return [], empty_graph

    timeline_groups = build_timeline_groups(output_dir, raw_knowledge, book_names, config)
    resolved_groups = []
    for idx, group in enumerate(timeline_groups, 1):
        if stop_requested():
            raise GracefulInterrupt("Остановка запрошена во время timeline resolution")

        log_tag = (
            f"{log_prefix}[{idx}/{len(timeline_groups)}]"
            f"[{get_book_stem(group.get('book_name', ''))}]"
        ) if log_prefix else ""
        resolved = resolve_timeline_group(
            client,
            config,
            group,
            log_prefix=log_tag,
        )
        resolved_groups.append({
            "book_name": group.get("book_name", ""),
            "chapter": group.get("chapter", ""),
            "chunk_indices": group.get("chunk_indices", []),
            "fact_count": len(group.get("facts", [])),
            "chunk_count": len(group.get("chunk_records", [])),
            "entities": resolved.get("entities", []),
            "events": resolved.get("events", []),
            "relations": resolved.get("relations", []),
        })
        if progress_callback is not None:
            progress_callback(
                "timeline_group_resolved",
                group_index=idx,
                group_total=len(timeline_groups),
                book_name=group.get("book_name", ""),
                chapter=group.get("chapter", ""),
                fact_count=len(group.get("facts", [])),
                event_count=len(resolved.get("events", [])),
                relation_count=len(resolved.get("relations", [])),
            )

    with open(global_paths["timeline_raw"], "w", encoding="utf-8") as f:
        json.dump(resolved_groups, f, ensure_ascii=False, indent=2)

    graph = merge_timeline_resolution_outputs(resolved_groups, unique_knowledge)
    with open(global_paths["timeline_graph"], "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    save_readable_timeline_graph(graph, str(global_paths["timeline_graph_txt"]))
    if progress_callback is not None:
        progress_callback(
            "timeline_complete",
            timeline_groups=len(resolved_groups),
            timeline_nodes=len(graph.get("nodes", [])),
            timeline_edges=len(graph.get("edges", [])),
        )

    if log_prefix:
        log_event(
            f"{log_prefix} timeline graph: groups={len(resolved_groups)}, "
            f"nodes={len(graph.get('nodes', []))}, edges={len(graph.get('edges', []))}"
        )

    return resolved_groups, graph


def is_book_processed(output_dir: str, book_name: str) -> bool:
    """Считает книгу завершённой по done-маркеру или старой схеме файлов."""
    _, mode = detect_book_mode(book_name)
    paths = get_book_output_paths(output_dir, book_name)
    if paths["done"].exists():
        return True

    # Для нового resume-режима отсутствие done-маркера важнее частично созданных файлов.
    # Иначе можно ошибочно пропустить книгу после аварийной остановки в середине обработки.
    resume_artifacts = (
        paths["chunks"],
        paths["knowledge_stream"],
        paths["synth_progress"],
    )
    if any(path.exists() for path in resume_artifacts):
        return False

    # В voice_only мы создаём streaming-файл заранее; пустой файл ещё не означает,
    # что книга реально завершена.
    if mode == "voice_only" and paths["voice"].exists():
        try:
            if paths["voice"].stat().st_size == 0:
                return False
        except OSError:
            return False

    legacy_marker = paths["voice"] if mode == "voice_only" else paths["knowledge"]
    return legacy_marker.exists()


def stable_fact_hash(fact: dict) -> str:
    """Стабильный хеш факта для resume синтетики."""
    return text_hash(json.dumps({
        "category": fact.get("category", ""),
        "subject": fact.get("subject", ""),
        "fact": fact.get("fact", ""),
        "time_scope": fact.get("time_scope", ""),
    }, ensure_ascii=False, sort_keys=True))


def load_chunk_checkpoint(path: Path, chunks: list[str], log_prefix: str = "") -> dict[int, dict]:
    """Загружает результаты уже обработанных чанков для resume."""
    records_by_idx: dict[int, dict] = {}
    for record in read_jsonl(path, log_prefix=log_prefix):
        idx = record.get("idx")
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(chunks):
            continue
        expected_hash = text_hash(chunks[idx])
        if record.get("chunk_hash") != expected_hash:
            if log_prefix:
                log_event(f"{log_prefix} checkpoint для чанка {idx + 1} пропущен: изменился текст")
            continue

        dialogues = record.get("dialogues", [])
        knowledge = record.get("knowledge", [])
        if not isinstance(dialogues, list):
            dialogues = []
        if not isinstance(knowledge, list):
            knowledge = []

        records_by_idx[idx] = {
            "dialogues": dialogues,
            "knowledge": knowledge,
            "chapter": strip_text(record.get("chapter", "")),
            "chunk_text": record.get("chunk_text") if isinstance(record.get("chunk_text"), str) else chunks[idx],
        }

    return records_by_idx


def rebuild_chunk_outputs(
    checkpoint_records: dict[int, dict],
    config: Config,
    voice_path: Path,
    knowledge_stream_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Пересобирает потоковые файлы из chunk-checkpoint после аварийной остановки."""
    if voice_path.exists():
        voice_path.unlink()
    if knowledge_stream_path.exists():
        knowledge_stream_path.unlink()

    voice_pairs: list[dict] = []
    all_knowledge: list[dict] = []

    for idx in sorted(checkpoint_records):
        record = checkpoint_records[idx]
        dialogues = record.get("dialogues", [])
        knowledge = record.get("knowledge", [])

        if dialogues:
            new_pairs = make_training_pairs(dialogues, config)
            append_jsonl(voice_path, new_pairs)
            voice_pairs.extend(new_pairs)

        if knowledge:
            append_jsonl(knowledge_stream_path, knowledge)
            all_knowledge.extend(knowledge)

    if not voice_path.exists():
        voice_path.touch()
    if not knowledge_stream_path.exists():
        knowledge_stream_path.touch()

    return voice_pairs, all_knowledge


def order_facts_for_synth(knowledge: list[dict], seed: int) -> list[dict]:
    """Детерминированный порядок фактов для resume синтетики."""
    facts = list(knowledge)
    facts.sort(key=stable_fact_hash)
    rng = random.Random(seed)
    rng.shuffle(facts)
    return facts


def load_synth_progress(path: Path, log_prefix: str = "") -> tuple[set[str], list[dict]]:
    """Загружает прогресс синтетики для resume."""
    processed_fact_hashes: set[str] = set()
    pairs: list[dict] = []

    for record in read_jsonl(path, log_prefix=log_prefix):
        fact_hash = record.get("fact_hash")
        if not isinstance(fact_hash, str) or not fact_hash:
            continue
        processed_fact_hashes.add(fact_hash)
        pair = record.get("pair")
        if isinstance(pair, dict):
            pairs.append(pair)

    return processed_fact_hashes, pairs


def call_llm_ollama_native(
    config: Config,
    system: str,
    user: str,
    max_tokens: int = 1500,
    response_format: Optional[Any] = None,
    log_prefix: str = "",
    temperature: Optional[float] = None,
    trace_id: str = "",
) -> Optional[str]:
    """Вызов через нативный API ollama с think=false."""
    import urllib.request

    if stop_requested():
        return None

    # Определяем базовый URL ollama (убираем /v1 если есть)
    base = config.api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    url = f"{base}/api/chat"
    effective_temperature = config.temperature if temperature is None else temperature
    payload_data = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": effective_temperature,
            "num_predict": max_tokens,
        },
    }
    if response_format is not None:
        payload_data["format"] = response_format

    trace_id = trace_id or next_llm_trace_id(log_prefix)
    trace_path, trace_state = init_llm_trace(
        config,
        trace_id,
        {
            "trace_id": trace_id,
            "ts": now_iso_str(),
            "provider": "ollama_native",
            "api_base": config.api_base,
            "model": config.model,
            "log_prefix": log_prefix,
            "max_tokens": max_tokens,
            "temperature": effective_temperature,
            "response_format": response_format,
            "timeout_seconds": config.request_timeout,
            "request_payload": payload_data,
        },
    )

    payload = json.dumps(payload_data).encode("utf-8")

    for attempt in range(3):
        if stop_requested():
            return None
        t0 = time.time()
        if log_prefix:
            log_event(
                f"{log_prefix} -> LLM запрос "
                f"(попытка {attempt + 1}/3, max_tokens={max_tokens})"
            )
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=config.request_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                # Подстраховка
                if content and "<think>" in content:
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                elapsed = time.time() - t0
                append_llm_trace_attempt(
                    trace_path,
                    trace_state,
                    {
                        "trace_id": trace_id,
                        "ts": now_iso_str(),
                        "provider": "ollama_native",
                        "status": "ok",
                        "attempt": attempt + 1,
                        "log_prefix": log_prefix,
                        "elapsed_seconds": round(elapsed, 3),
                        "content": content,
                        "done_reason": data.get("done_reason", "unknown"),
                        "prompt_tokens": data.get("prompt_eval_count"),
                        "completion_tokens": data.get("eval_count"),
                        "raw_response": data,
                    },
                )
                if log_prefix:
                    done_reason = data.get("done_reason", "unknown")
                    prompt_tokens = data.get("prompt_eval_count")
                    completion_tokens = data.get("eval_count")
                    token_info = []
                    if prompt_tokens is not None:
                        token_info.append(f"prompt={prompt_tokens}")
                    if completion_tokens is not None:
                        token_info.append(f"completion={completion_tokens}")
                    suffix = f", {', '.join(token_info)}" if token_info else ""
                    log_event(
                        f"{log_prefix} <- LLM ok "
                        f"({done_reason}, {elapsed:.1f}s, {len(content)} симв{suffix})"
                    )
                return content
        except Exception as e:
            if stop_requested():
                return None
            elapsed = time.time() - t0
            append_llm_trace_attempt(
                trace_path,
                trace_state,
                {
                    "trace_id": trace_id,
                    "ts": now_iso_str(),
                    "provider": "ollama_native",
                    "status": "error",
                    "attempt": attempt + 1,
                    "log_prefix": log_prefix,
                    "elapsed_seconds": round(elapsed, 3),
                    "error": str(e),
                },
            )
            if log_prefix:
                log_event(
                    f"{log_prefix} ошибка API "
                    f"(попытка {attempt + 1}/3, {elapsed:.1f}s): {e}"
                )
            else:
                print(f"    Ошибка API (попытка {attempt + 1}/3): {e}")
            time.sleep(2 ** attempt)

    return None


def call_llm_openai(
    client: OpenAI,
    config: Config,
    system: str,
    user: str,
    max_tokens: int = 1500,
    response_format: Optional[Any] = None,
    log_prefix: str = "",
    temperature: Optional[float] = None,
    trace_id: str = "",
) -> Optional[str]:
    """Вызов через OpenAI-совместимый API (vllm, llama.cpp, LM Studio)."""
    effective_temperature = config.temperature if temperature is None else temperature
    if stop_requested():
        return None

    trace_id = trace_id or next_llm_trace_id(log_prefix)
    request_kwargs = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": effective_temperature,
        "max_tokens": max_tokens,
        "timeout": config.request_timeout,
    }
    if response_format is not None:
        request_kwargs["response_format"] = response_format
    trace_path, trace_state = init_llm_trace(
        config,
        trace_id,
        {
            "trace_id": trace_id,
            "ts": now_iso_str(),
            "provider": "openai_compatible",
            "api_base": config.api_base,
            "model": config.model,
            "log_prefix": log_prefix,
            "max_tokens": max_tokens,
            "temperature": effective_temperature,
            "response_format": response_format,
            "timeout_seconds": config.request_timeout,
            "request_payload": request_kwargs,
        },
    )

    for attempt in range(3):
        if stop_requested():
            return None
        t0 = time.time()
        if log_prefix:
            log_event(
                f"{log_prefix} -> LLM запрос "
                f"(попытка {attempt + 1}/3, max_tokens={max_tokens})"
            )
        try:
            response = client.chat.completions.create(
                **request_kwargs,
            )
            content = response.choices[0].message.content
            if content and "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            elapsed = time.time() - t0
            usage = getattr(response, "usage", None)
            append_llm_trace_attempt(
                trace_path,
                trace_state,
                {
                    "trace_id": trace_id,
                    "ts": now_iso_str(),
                    "provider": "openai_compatible",
                    "status": "ok",
                    "attempt": attempt + 1,
                    "log_prefix": log_prefix,
                    "elapsed_seconds": round(elapsed, 3),
                    "content": content,
                    "finish_reason": response.choices[0].finish_reason or "unknown",
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
                    "raw_response": response,
                },
            )
            if log_prefix:
                finish_reason = response.choices[0].finish_reason or "unknown"
                token_info = []
                if usage is not None:
                    if getattr(usage, "prompt_tokens", None) is not None:
                        token_info.append(f"prompt={usage.prompt_tokens}")
                    if getattr(usage, "completion_tokens", None) is not None:
                        token_info.append(f"completion={usage.completion_tokens}")
                suffix = f", {', '.join(token_info)}" if token_info else ""
                log_event(
                    f"{log_prefix} <- LLM ok "
                    f"({finish_reason}, {elapsed:.1f}s, {len(content or '')} симв{suffix})"
                )
            return content
        except Exception as e:
            if stop_requested():
                return None
            elapsed = time.time() - t0
            append_llm_trace_attempt(
                trace_path,
                trace_state,
                {
                    "trace_id": trace_id,
                    "ts": now_iso_str(),
                    "provider": "openai_compatible",
                    "status": "error",
                    "attempt": attempt + 1,
                    "log_prefix": log_prefix,
                    "elapsed_seconds": round(elapsed, 3),
                    "error": str(e),
                },
            )
            if log_prefix:
                log_event(
                    f"{log_prefix} ошибка API "
                    f"(попытка {attempt + 1}/3, {elapsed:.1f}s): {e}"
                )
            else:
                print(f"    Ошибка API (попытка {attempt + 1}/3): {e}")
            time.sleep(2 ** attempt)

    return None


# Глобальный флаг: используем ли нативный API ollama
_use_ollama_native = False


def call_llm(
    client: OpenAI,
    config: Config,
    system: str,
    user: str,
    max_tokens: int = 1500,
    response_format: Optional[Any] = None,
    log_prefix: str = "",
    temperature: Optional[float] = None,
) -> Optional[str]:
    """Вызов LLM. Автоматически использует нативный API ollama если доступен."""
    trace_id = next_llm_trace_id(log_prefix)
    if _use_ollama_native:
        return call_llm_ollama_native(
            config,
            system,
            user,
            max_tokens,
            response_format=response_format,
            log_prefix=log_prefix,
            temperature=temperature,
            trace_id=trace_id,
        )
    else:
        return call_llm_openai(
            client,
            config,
            system,
            user,
            max_tokens,
            response_format=response_format,
            log_prefix=log_prefix,
            temperature=temperature,
            trace_id=trace_id,
        )


# ──────────────────────────────────────────────
# Шаг 1: Нарезка текста на фрагменты
# ──────────────────────────────────────────────

SEMANTIC_SPLIT_SYSTEM = """Ты разбиваешь литературный текст на логические части.
Работай только по границам абзацев. Отвечай строго JSON."""

SEMANTIC_SPLIT_PROMPT = """Есть длинный фрагмент книги, уже разбитый на абзацы.
Нужно предложить места разрыва между абзацами так, чтобы части были логически цельными.

ПРАВИЛА:
— Разрывы можно ставить только МЕЖДУ абзацами.
— Старайся делать как можно меньше разрывов.
— Предпочтительный размер части: около {target_tokens} токенов.
— Можно превышать этот размер, если иначе ломается цельная сцена или монолог.
— Если весь фрагмент лучше оставить целиком — верни пустой список разрывов.

Верни JSON объект:
{{"break_after": [N1, N2, ...]}}

Где N означает: сделать разрыв ПОСЛЕ абзаца номер N.
Номера должны быть строго возрастающими и в диапазоне от 1 до {max_break}.

Абзацы:
{paragraphs}

JSON:"""


def split_text_into_paragraphs(text: str) -> list[str]:
    """Разбивает текст на абзацы по пустым строкам."""
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def looks_like_heading(paragraph: str) -> bool:
    """Грубая эвристика для заголовков в plain text."""
    text = re.sub(r"\s+", " ", paragraph or "").strip()
    if not text or len(text) > 120:
        return False

    if re.fullmatch(r"(?i)(глава|часть|книга|пролог|эпилог|послесловие|предисловие)\b.*", text):
        return True
    if re.fullmatch(r"(?i)[ivxlcdm]+[.)]?", text):
        return True
    if re.fullmatch(r"\d+[.)]?", text):
        return True

    letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", text)
    return bool(letters) and letters.isupper() and len(letters) >= 4


def is_scene_break(paragraph: str) -> bool:
    """Опознаёт короткие разделители сцен вроде ***."""
    compact = re.sub(r"\s+", "", paragraph or "")
    if not compact or len(compact) > 12:
        return False
    return bool(re.fullmatch(r"[*#~\-_=•·.]{3,}", compact))


def paragraph_preview_for_split(paragraph: str, limit: int = 120) -> str:
    """Компактный превью абзаца для LLM-сплиттера."""
    cleaned = re.sub(r"\s+", " ", paragraph or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    head = cleaned[: min(160, limit - 50)].rstrip()
    tail = cleaned[-40:].lstrip()
    return f"{head} ... {tail}"


def split_text_into_semantic_sections(text: str) -> list[list[str]]:
    """Собирает текст в крупные смысловые секции: section -> heading -> paragraphs."""
    paragraphs = split_text_into_paragraphs(text)
    sections: list[list[str]] = []
    current: list[str] = []

    for paragraph in paragraphs:
        if paragraph == SECTION_BREAK_MARKER:
            if current:
                sections.append(current)
                current = []
            continue

        if is_scene_break(paragraph):
            if current:
                sections.append(current)
                current = []
            continue

        if looks_like_heading(paragraph) and current:
            sections.append(current)
            current = [paragraph]
            continue

        current.append(paragraph)

    if current:
        sections.append(current)

    return sections


def group_paragraphs_by_size(paragraphs: list[str], chunk_size: int) -> list[str]:
    """Fallback: собирает абзацы в куски по размеру, не режа абзацы."""
    units: list[str] = []
    current: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        para_tokens = estimate_tokens(paragraph)

        if current and len(current) == 1 and looks_like_heading(current[0]):
            current.append(paragraph)
            current_size += para_tokens
            if para_tokens > chunk_size:
                units.append("\n\n".join(current))
                current = []
                current_size = 0
            continue

        if para_tokens > chunk_size:
            if current:
                units.append("\n\n".join(current))
                current = []
                current_size = 0
            units.append(paragraph)
            continue

        if current and current_size + para_tokens > chunk_size:
            units.append("\n\n".join(current))
            current = [paragraph]
            current_size = para_tokens
            continue

        current.append(paragraph)
        current_size += para_tokens

    if current:
        units.append("\n\n".join(current))

    return units


def split_paragraphs_by_breaks(paragraphs: list[str], breaks: list[int]) -> list[str]:
    """Собирает contiguous-группы абзацев по списку разрывов `break_after`."""
    parts: list[str] = []
    start = 0
    for cut in breaks:
        part = paragraphs[start:cut]
        if part:
            parts.append("\n\n".join(part))
        start = cut
    tail = paragraphs[start:]
    if tail:
        parts.append("\n\n".join(tail))
    return parts


def propose_semantic_breaks_with_llm(
    client: OpenAI,
    config: Config,
    paragraphs: list[str],
    chunk_size: int,
    log_prefix: str = "",
) -> list[int]:
    """Просит LLM предложить логические разрывы между абзацами длинной секции."""
    if len(paragraphs) < 4:
        return []
    if len(paragraphs) > config.semantic_split_max_paragraphs:
        if log_prefix:
            log_event(
                f"{log_prefix} semantic split fallback: "
                f"слишком много абзацев ({len(paragraphs)} > {config.semantic_split_max_paragraphs})"
            )
        return []

    numbered = []
    for idx, paragraph in enumerate(paragraphs, 1):
        preview = paragraph_preview_for_split(paragraph)
        numbered.append(f"[{idx}] (~{estimate_tokens(paragraph)} ток.) {preview}")

    paragraphs_payload = "\n\n".join(numbered)
    prompt_preview = SEMANTIC_SPLIT_PROMPT.format(
        target_tokens=chunk_size,
        max_break=max(len(paragraphs) - 1, 1),
        paragraphs=paragraphs_payload,
    )
    if estimate_tokens(prompt_preview) > config.semantic_split_prompt_budget:
        if log_prefix:
            log_event(
                f"{log_prefix} semantic split fallback: "
                f"prompt слишком большой (~{estimate_tokens(prompt_preview)} ток.)"
            )
        return []

    response = call_llm(
        client,
        config,
        SEMANTIC_SPLIT_SYSTEM,
        prompt_preview,
        max_tokens=min(config.max_tokens_semantic_split, 120 + len(paragraphs) * 6),
        response_format="json" if _use_ollama_native else None,
        log_prefix=log_prefix,
        temperature=0.0,
    )
    if response is None:
        return []

    data, strategy = parse_json_response(response, expect="object", log_prefix=log_prefix)
    if not isinstance(data, dict):
        return []

    cuts = data.get("break_after", [])
    if not isinstance(cuts, list):
        return []

    valid_cuts = []
    for cut in cuts:
        if not isinstance(cut, int):
            continue
        if 1 <= cut < len(paragraphs):
            valid_cuts.append(cut)

    valid_cuts = sorted(set(valid_cuts))
    if log_prefix and valid_cuts:
        log_event(f"{log_prefix} semantic split: {len(valid_cuts) + 1} частей ({strategy})")
    return valid_cuts


def build_semantic_units(
    client: Optional[OpenAI],
    config: Optional[Config],
    text: str,
    chunk_size: int,
    log_prefix: str = "",
) -> list[str]:
    """Строит semantic units: sections -> paragraph groups -> LLM split для oversized sections."""
    sections = split_text_into_semantic_sections(text)
    if not sections:
        return []

    units: list[str] = []
    llm_sections = 0
    oversized_sections = 0

    for idx, paragraphs in enumerate(sections):
        section_text = "\n\n".join(paragraphs)
        section_tokens = estimate_tokens(section_text)

        if section_tokens <= chunk_size:
            units.append(section_text)
            continue

        if len(paragraphs) <= 1:
            oversized_sections += 1
            units.append(section_text)
            continue

        cuts: list[int] = []
        if (
            client is not None
            and config is not None
            and len(paragraphs) >= 4
            and section_tokens > int(chunk_size * 1.5)
        ):
            cuts = propose_semantic_breaks_with_llm(
                client,
                config,
                paragraphs,
                chunk_size,
                log_prefix=f"{log_prefix}[section {idx + 1}/{len(sections)}]",
            )

        if cuts:
            units.extend(split_paragraphs_by_breaks(paragraphs, cuts))
            llm_sections += 1
            continue

        grouped = group_paragraphs_by_size(paragraphs, chunk_size)
        if len(grouped) == 1 and estimate_tokens(grouped[0]) > chunk_size:
            oversized_sections += 1
        units.extend(grouped)

    if log_prefix:
        log_event(
            f"{log_prefix} semantic units: sections={len(sections)}, "
            f"units={len(units)}, llm_sections={llm_sections}, oversized={oversized_sections}"
        )

    return units


def split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    client: Optional[OpenAI] = None,
    config: Optional[Config] = None,
    log_prefix: str = "",
) -> list[str]:
    """
    Гибридная нарезка: section -> абзацы -> fallback по длине.
    Крупные неразрывные фрагменты могут превышать лимит и уходят целиком.
    """
    semantic_units = build_semantic_units(
        client,
        config,
        text,
        chunk_size,
        log_prefix=log_prefix,
    )
    if not semantic_units:
        return []

    chunks = []
    current_chunk = []
    current_size = 0

    def flush_current_chunk():
        nonlocal current_chunk, current_size
        if not current_chunk:
            return

        chunk_text = "\n\n".join(current_chunk)
        chunks.append(chunk_text)

        overlap_units = []
        overlap_size = 0
        for unit in reversed(current_chunk):
            unit_size = estimate_tokens(unit)
            if overlap_size + unit_size > overlap:
                break
            overlap_units.insert(0, unit)
            overlap_size += unit_size

        current_chunk = overlap_units
        current_size = overlap_size

    for unit in semantic_units:
        unit_tokens = estimate_tokens(unit)

        if unit_tokens > chunk_size:
            flush_current_chunk()
            chunks.append(unit)
            current_chunk = []
            current_size = 0
            continue

        if current_chunk and current_size + unit_tokens > chunk_size:
            flush_current_chunk()

        current_chunk.append(unit)
        current_size += unit_tokens

    flush_current_chunk()
    return chunks


def split_chunk_paragraphs(text: str) -> list[str]:
    """Разбивает чанк на абзацы, сохраняя только непустые куски."""
    return [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]


def default_chapter_label(book_name: str) -> str:
    """Возвращает безопасное имя главы по умолчанию, если заголовок не найден."""
    label = Path(book_name).with_suffix("").name.strip()
    return label or "Без названия"


def extract_chunk_heading(chunk: str) -> str:
    """Пытается вытащить заголовок главы/секции из начала чанка."""
    paragraphs = split_chunk_paragraphs(chunk)
    for paragraph in paragraphs[:6]:
        text = strip_text(paragraph)
        if not text or text == SECTION_BREAK_MARKER or is_scene_break(text):
            continue
        if looks_like_heading(text):
            return text
    return ""


def build_chunk_chapter_map(chunks: list[str], book_name: str) -> list[str]:
    """Строит карту chunk_idx -> chapter, наследуя последний найденный заголовок."""
    current = default_chapter_label(book_name)
    chapter_map: list[str] = []

    for chunk in chunks:
        heading = extract_chunk_heading(chunk)
        if heading:
            current = heading
        chapter_map.append(current)

    return chapter_map


def attach_knowledge_source_fields(
    items: list[dict],
    *,
    book_name: str,
    chapter: str,
    chunk_idx: Optional[int],
) -> list[dict]:
    """Добавляет к фактам источник: книга, глава и индекс чанка."""
    result = []
    normalized_chapter = strip_text(chapter) or default_chapter_label(book_name)
    normalized_chunk_idx = int(chunk_idx) if isinstance(chunk_idx, int) else None

    for item in items:
        cleaned = dict(item)
        cleaned["source_book"] = book_name
        cleaned["chapter"] = normalized_chapter
        cleaned["chunk_idx"] = normalized_chunk_idx
        result.append(cleaned)

    return result


def ensure_knowledge_source_defaults(items: list[dict], book_name: str) -> list[dict]:
    """Подставляет source-поля для старых фактов, если они отсутствуют."""
    result = []
    fallback_chapter = default_chapter_label(book_name)

    for item in items:
        cleaned = dict(item)
        cleaned["source_book"] = strip_text(cleaned.get("source_book", "")) or book_name
        cleaned["chapter"] = strip_text(cleaned.get("chapter", "")) or fallback_chapter
        chunk_idx = cleaned.get("chunk_idx")
        cleaned["chunk_idx"] = chunk_idx if isinstance(chunk_idx, int) else None
        result.append(cleaned)

    return result


def take_neighbor_excerpt(chunk: str, token_limit: int, from_end: bool) -> str:
    """Берёт приграничный фрагмент соседнего чанка для поддержки контекста."""
    if token_limit <= 0:
        return ""

    paragraphs = split_chunk_paragraphs(chunk)
    if not paragraphs:
        return ""

    selected: list[str] = []
    used = 0
    iterable = reversed(paragraphs) if from_end else paragraphs

    for paragraph in iterable:
        paragraph_tokens = estimate_tokens(paragraph)
        if selected and used + paragraph_tokens > token_limit:
            break
        if from_end:
            selected.insert(0, paragraph)
        else:
            selected.append(paragraph)
        used += paragraph_tokens
        if used >= token_limit:
            break

    if selected:
        return "\n\n".join(selected)

    approx_chars = max(token_limit * 4, 200)
    return chunk[-approx_chars:].strip() if from_end else chunk[:approx_chars].strip()


def build_extraction_chunk_payload(
    chunks: list[str],
    idx: int,
    config: Config,
) -> tuple[str, dict]:
    """Собирает материал для extraction: целевой чанк + соседний supporting context."""
    primary_chunk = chunks[idx]
    parts = [f"[PRIMARY CHUNK #{idx + 1}]\n{primary_chunk}"]

    remaining_budget = max(config.extraction_context_budget - estimate_tokens(primary_chunk), 0)
    support_chunks = 0
    support_tokens = 0

    if remaining_budget <= 0 or config.extraction_neighbor_chunks <= 0:
        return "\n\n".join(parts), {
            "support_chunks": 0,
            "support_tokens": 0,
        }

    support_blocks: list[str] = []
    total_chunks = len(chunks)
    for distance in range(1, config.extraction_neighbor_chunks + 1):
        for neighbor_idx, side in ((idx - distance, "PREV"), (idx + distance, "NEXT")):
            if neighbor_idx < 0 or neighbor_idx >= total_chunks or remaining_budget <= 0:
                continue

            neighbor_chunk = chunks[neighbor_idx]
            neighbor_tokens = estimate_tokens(neighbor_chunk)
            excerpt_mode = "full"

            if neighbor_tokens <= remaining_budget:
                excerpt_text = neighbor_chunk
                used_tokens = neighbor_tokens
            else:
                excerpt_mode = "excerpt"
                excerpt_budget = min(config.extraction_neighbor_excerpt_tokens, remaining_budget)
                excerpt_text = take_neighbor_excerpt(
                    neighbor_chunk,
                    excerpt_budget,
                    from_end=(side == "PREV"),
                )
                used_tokens = estimate_tokens(excerpt_text) if excerpt_text else 0

            if not excerpt_text or used_tokens <= 0:
                continue

            support_blocks.append(
                f"[SUPPORTING {side} CHUNK #{neighbor_idx + 1} | {excerpt_mode}]\n{excerpt_text}"
            )
            remaining_budget = max(remaining_budget - used_tokens, 0)
            support_chunks += 1
            support_tokens += used_tokens

    if support_blocks:
        parts.append("[SUPPORTING CONTEXT]\n" + "\n\n".join(support_blocks))

    glossary_text = build_scene_glossary(
        "\n\n".join([primary_chunk] + support_blocks),
    )
    if glossary_text:
        parts.append("[SCENE GLOSSARY]\n" + glossary_text)

    return "\n\n".join(parts), {
        "support_chunks": support_chunks,
        "support_tokens": support_tokens,
        "glossary_items": len(glossary_text.splitlines()) if glossary_text else 0,
    }


def build_neighbor_text_window(
    chunks: list[str],
    idx: int,
    neighbor_chunks: int,
) -> tuple[str, str]:
    """Собирает raw-окно соседних чанков для regex и проверки пограничного контекста."""
    left = max(0, idx - neighbor_chunks)
    right = min(len(chunks), idx + neighbor_chunks + 1)
    window_parts = chunks[left:right]
    support_parts = [chunks[pos] for pos in range(left, right) if pos != idx]
    return "\n\n".join(window_parts), "\n\n".join(support_parts)


def dialogue_item_key(item: dict) -> str:
    """Ключ для точной дедупликации реплик между page-проходами."""
    payload = {
        "type": item.get("type", ""),
        "interlocutor": item.get("interlocutor"),
        "interlocutor_says": normalize_dedup_text(item.get("interlocutor_says", "")),
        "max_says": normalize_dedup_text(item.get("max_says", "")),
    }
    return text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def merge_dialogue_items(existing: list[dict], new_items: list[dict]) -> tuple[list[dict], int]:
    """Добавляет в список только новые реплики, сохраняя порядок."""
    merged = list(existing)
    seen = {dialogue_item_key(item) for item in existing}
    added = 0

    for item in new_items:
        item_key = dialogue_item_key(item)
        if item_key in seen:
            continue
        seen.add(item_key)
        merged.append(item)
        added += 1

    return merged, added


def merge_knowledge_items(existing: list[dict], new_items: list[dict]) -> tuple[list[dict], int]:
    """Добавляет в список только новые факты, схлопывая дубли между page-проходами."""
    merged = deduplicate_knowledge(existing + new_items)
    added = max(len(merged) - len(deduplicate_knowledge(existing)), 0)
    return merged, added


def format_previous_dialogues_for_prompt(items: list[dict], limit: int = 24) -> str:
    """Короткий список уже извлечённых реплик для следующего page-прохода."""
    if not items:
        return ""

    lines = []
    for item in items[:limit]:
        quote = preview_text(item.get("max_says", ""), 90)
        dtype = item.get("type", "dialogue")
        lines.append(f"- {dtype}: {quote}")
    if len(items) > limit:
        lines.append(f"- ... ещё {len(items) - limit} элементов")
    return "\n".join(lines)


def format_previous_knowledge_for_prompt(items: list[dict], limit: int = 36) -> str:
    """Короткий список уже извлечённых фактов для следующего page-прохода."""
    if not items:
        return ""

    lines = []
    for item in items[:limit]:
        subject = preview_text(item.get("subject", ""), 40)
        fact = preview_text(item.get("fact", ""), 100)
        category = item.get("category", "unknown")
        time_scope = normalize_time_scope(item.get("time_scope", ""), fact=fact, category=category)
        if time_scope and time_scope not in {"unclear", "timeless"}:
            lines.append(f"- {category} / {subject} [{time_scope}]: {fact}")
        else:
            lines.append(f"- {category} / {subject}: {fact}")
    if len(items) > limit:
        lines.append(f"- ... ещё {len(items) - limit} фактов")
    return "\n".join(lines)


def make_dialogue_pagination_note(pass_idx: int, extracted: list[dict]) -> str:
    """Инструкция для добора новых реплик на следующем проходе."""
    if pass_idx <= 0 or not extracted:
        return ""
    already = format_previous_dialogues_for_prompt(extracted)
    return (
        f"\n\nЭТО ПРОХОД #{pass_idx + 1}. Ниже уже найденные элементы, их повторять нельзя.\n"
        "Найди НОВЫЕ реплики и монологи из PRIMARY CHUNK, которые ещё не извлечены.\n"
        "УЖЕ ИЗВЛЕЧЕНО:\n"
        f"{already}"
    )


def make_knowledge_pagination_note(pass_idx: int, extracted: list[dict]) -> str:
    """Инструкция для добора новых фактов на следующем проходе."""
    if pass_idx <= 0 or not extracted:
        return ""
    already = format_previous_knowledge_for_prompt(extracted)
    return (
        f"\n\nЭТО ПРОХОД #{pass_idx + 1}. Ниже уже найденные факты, их повторять нельзя.\n"
        "Найди НОВЫЕ факты и события из PRIMARY CHUNK, которые ещё не попали в список.\n"
        "УЖЕ ИЗВЛЕЧЕНО:\n"
        f"{already}"
    )


def extract_voice_with_regex(chunk: str, log_prefix: str = "") -> tuple[list[dict], dict]:
    """Быстрый локальный extraction голоса Макса без LLM."""
    rx = _extract_regex
    if rx is None:
        if log_prefix:
            log_event(f"{log_prefix} regex-фоллбек недоступен: {_extract_regex_import_error}")
        return [], {"speech": 0, "silent": 0, "monologue": 0}

    speech_items = rx.extract_direct_speech(chunk)
    silent_items = rx.extract_silent_speech(chunk)
    monologues = rx.extract_monologues(chunk)

    results = []
    seen = set()

    def add_item(item: dict):
        key = text_hash(
            f"{item.get('type', '')}|{item.get('context', '')}|{item.get('max_says', '')}"
        )
        if key in seen:
            return
        seen.add(key)
        results.append(item)

    prev_speaker = None
    prev_text = ""
    max_speech = 0

    for speech in speech_items:
        speaker = speech.get("speaker")
        text = speech.get("text", "").strip()
        if speaker == "Макс" and len(text) >= 15:
            max_speech += 1
            interlocutor = prev_speaker if prev_speaker and prev_speaker != "Макс" else None
            interlocutor_says = prev_text if interlocutor else ""
            context = speech.get("context_before", "").strip() or "Прямая речь Макса."
            add_item({
                "type": "dialogue",
                "context": context[:240],
                "interlocutor": interlocutor,
                "interlocutor_says": interlocutor_says[:400],
                "max_says": text,
                "_source": "regex",
            })

        prev_speaker = speaker
        prev_text = text

    max_silent = 0
    for item in silent_items:
        if item.get("speaker") != "Макс":
            continue
        text = item.get("text", "").strip()
        if len(text) < 15:
            continue
        max_silent += 1
        add_item({
            "type": "silent_speech",
            "context": item.get("context", "").strip()[:240] or "Безмолвная речь.",
            "interlocutor": None,
            "interlocutor_says": "",
            "max_says": text,
            "_source": "regex",
        })

    mono_count = 0
    for item in monologues:
        text = item.get("text", "").strip()
        if len(text) < 30 or len(text) > 1000:
            continue
        mono_count += 1
        first_sentence = re.split(r"[.!?]", text)[0].strip()
        context = first_sentence[:160] if first_sentence else "Внутренний монолог Макса."
        add_item({
            "type": "monologue",
            "context": context,
            "interlocutor": None,
            "interlocutor_says": "",
            "max_says": text,
            "_source": "regex",
        })

    stats = {
        "speech": max_speech,
        "silent": max_silent,
        "monologue": mono_count,
    }

    if log_prefix:
        log_event(
            f"{log_prefix} regex-голос: "
            f"dialogue={stats['speech']}, silent={stats['silent']}, "
            f"monologue={stats['monologue']}, итог={len(results)}"
        )

    return results, stats


# ──────────────────────────────────────────────
# Шаг 2: Классификация — есть ли диалоги Макса
# ──────────────────────────────────────────────

CLASSIFY_SYSTEM = "Ты — помощник для анализа литературного текста. Отвечай строго по формату."

CLASSIFY_PROMPT = """Проанализируй фрагмент книги Макса Фрая.
Книга написана от первого лица Сэром Максом, поэтому ВЕСЬ текст формально — его речь.
Меня интересует, содержит ли фрагмент ЦЕННЫЙ МАТЕРИАЛ для обучения нейросети говорить как Сэр Макс.

Ценный материал — это:
1. ДИАЛОГИ Макса с другими персонажами (прямая речь, реплики через тире или кавычки)
2. БЕЗМОЛВНАЯ РЕЧЬ (мысленные разговоры, обычно выделены курсивом или кавычками)
3. ЯРКИЕ ВНУТРЕННИЕ МОНОЛОГИ — рефлексия, ирония, рассуждения о жизни, описания еды и впечатлений
4. ЭМОЦИОНАЛЬНЫЕ РЕАКЦИИ — удивление, страх, восторг, ворчание, юмор

НЕ ценный материал:
- Сухое описание событий, действий, перемещений без эмоций и речи
- Экспозиция мира (история Орденов, география, правила магии)
- Описания внешности, обстановки без реакции Макса

Ответь СТРОГО одним словом: ДА или НЕТ

Фрагмент:
---
{chunk}
---"""


def classify_chunk(client: OpenAI, config: Config, chunk: str) -> bool:
    """Определяет, содержит ли фрагмент ценный для обучения материал."""
    response = call_llm(client, config, CLASSIFY_SYSTEM,
                        CLASSIFY_PROMPT.format(chunk=chunk),
                        max_tokens=config.max_tokens_classify)
    if response is None:
        return True  # При ошибке — лучше не отбрасывать
    return "ДА" in response.upper().split()[0] if response.strip() else True


# ──────────────────────────────────────────────
# Шаг 3: Извлечение диалогов
# ──────────────────────────────────────────────

EXTRACT_SYSTEM = """Ты — помощник для разметки литературного текста Макса Фрая.
Извлекаешь диалоги и монологи главного героя — Сэра Макса.
Особенности стиля:
— Повествование от первого лица (рассказчик = Макс).
— Прямая речь оформляется через тире (—), иногда без явной атрибуции.
— Безмолвная речь (мысленное общение) может выделяться кавычками или курсивом.
— Внутренний монолог не отделён от повествования — это мысли Макса внутри текста.
— Макс часто комментирует происходящее с иронией прямо в нарративе.
Отвечай СТРОГО в формате JSON. Никакого текста до или после JSON."""

EXTRACT_PROMPT = """Из фрагмента книги Макса Фрая извлеки материал для обучения нейросети говорить как Сэр Макс.

Извлекай три типа материала:

ТИП 1 — "dialogue": Прямая речь Макса в разговоре с кем-то.
  Ищи реплики после тире (—), которые принадлежат Максу.
  Реплики других персонажей перед ответом Макса запиши в "interlocutor_says".

ТИП 2 — "silent_speech": Безмолвная речь — мысленные разговоры.
  Обычно в кавычках или после фраз типа «послал зов», «сказал я по Безмолвной связи».

ТИП 3 — "monologue": Яркие внутренние мысли, рефлексия, ироничные комментарии.
  НЕ извлекай сухое описание действий. Извлекай только то, где слышен ГОЛОС Макса —
  ирония, философия, эмоции, ворчание, восторг, самоирония.

Для каждого фрагмента создай объект:
- "type": "dialogue", "silent_speech" или "monologue"
- "context": краткое описание ситуации (1-2 предложения)
- "interlocutor": имя собеседника если есть, иначе null
- "interlocutor_says": что сказал собеседник перед репликой Макса (если есть)
- "max_says": ТОЧНАЯ цитата из текста — реплика или мысль Макса

ПРАВИЛА:
— Извлекай ТОЧНЫЕ цитаты из текста, не пересказывай.
— Не придумывай то, чего нет во фрагменте.
— Если реплику невозможно однозначно атрибутировать Максу — пропусти её.
— Если во фрагменте нет подходящего материала — верни пустой массив [].
— Верни не более 8 самых ценных элементов на фрагмент.
— Если ниже есть блоки SUPPORTING CONTEXT, используй их только для понимания ситуации, местоимений,
  имён и того, кто именно говорит.
— Если ниже есть SCENE GLOSSARY, используй его только как справочник имён и сущностей сцены.
  Он помогает правильно назвать собеседника или место, но НЕ является источником цитат.
— Извлекать нужно только то, что ЯВНО присутствует в PRIMARY CHUNK. Нельзя вытаскивать цитаты только из SUPPORTING CONTEXT.
{pagination_note}

Верни JSON массив.
НЕ ПИШИ НИЧЕГО, КРОМЕ JSON. НИКАКИХ ОБЪЯСНЕНИЙ ИЛИ ВВОДНЫХ СЛОВ.
НЕ используй ```json или другие markdown-обёртки — только чистый JSON.
Материал:
---
{chunk_payload}
---

JSON:"""


def extract_dialogues(
    client: OpenAI,
    config: Config,
    chunk: str,
    log_prefix: str = "",
    chunk_payload: Optional[str] = None,
) -> list[dict]:
    """Извлекает структурированные диалоги из фрагмента."""
    payload = chunk_payload or f"[PRIMARY CHUNK]\n{chunk}"
    extracted: list[dict] = []

    for pass_idx in range(max(config.extraction_passes, 1)):
        response = call_llm(
            client,
            config,
            EXTRACT_SYSTEM,
            EXTRACT_PROMPT.format(
                chunk_payload=payload,
                pagination_note=make_dialogue_pagination_note(pass_idx, extracted),
            ),
            max_tokens=config.max_tokens_extract,
            response_format="json" if _use_ollama_native else None,
            log_prefix=f"{log_prefix}[page {pass_idx + 1}]" if log_prefix else "",
        )

        if response is None:
            break

        data, strategy = parse_json_response(
            response,
            expect="array",
            log_prefix=f"{log_prefix}[page {pass_idx + 1}]" if log_prefix else "",
        )
        if not isinstance(data, list):
            if log_prefix:
                log_event(f"{log_prefix}[page {pass_idx + 1}] ответ не удалось распарсить как JSON-массив")
            break

        page_items = validate_dialogues(
            data,
            source_chunk=chunk,
            log_prefix=f"{log_prefix}[page {pass_idx + 1}]" if log_prefix else "",
        )
        extracted, added = merge_dialogue_items(extracted, page_items)
        if log_prefix:
            log_event(
                f"{log_prefix}[page {pass_idx + 1}] JSON ok: {len(data)} элементов "
                f"({strategy}), новых={added}, накоплено={len(extracted)}"
            )
        if not page_items or added == 0:
            break

    return extracted


def normalize_search_text(text: str) -> str:
    """Нормализует текст для ускоренного поиска цитат в исходном чанке."""
    text = re.sub(r'[«»""\'\-—–]', '', text or '')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def fuzzy_find(
    needle: str,
    haystack: str,
    threshold: float = 0.7,
    normalized_haystack: Optional[str] = None,
) -> bool:
    """Проверяет, что needle (или очень похожая строка) содержится в haystack.
    Использует скользящее окно для поиска лучшего совпадения."""
    if not needle or not haystack:
        return False

    # Точное вхождение
    if needle in haystack:
        return True

    norm_needle = normalize_search_text(needle)
    norm_haystack = normalized_haystack if normalized_haystack is not None else normalize_search_text(haystack)

    if norm_needle in norm_haystack:
        return True

    # Скользящее окно с fuzzy ratio
    window = len(norm_needle)
    if window < 10 or window > len(norm_haystack):
        return False

    best_ratio = 0.0
    # Проверяем с шагом, чтобы не было O(n*m)
    step = max(1, window // 4)
    for i in range(0, len(norm_haystack) - window + 1, step):
        candidate = norm_haystack[i:i + window]
        ratio = difflib.SequenceMatcher(None, norm_needle, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
        if ratio >= threshold:
            return True

    return best_ratio >= threshold


def validate_dialogues(
    items: list[dict],
    source_chunk: str = "",
    support_chunk: str = "",
    log_prefix: str = "",
) -> list[dict]:
    """Фильтрует невалидные записи. Если source_chunk задан — проверяет,
    что max_says реально присутствует в исходном тексте (fuzzy)."""
    valid = []
    rejected = {
        "too_short": 0,
        "missing_type": 0,
        "invalid_type": 0,
        "max_says_not_found": 0,
        "interlocutor_not_found": 0,
    }
    normalized_source_chunk = normalize_search_text(source_chunk) if source_chunk else ""
    normalized_support_chunk = normalize_search_text(support_chunk) if support_chunk else ""

    for d in items:
        max_says = strip_text(d.get("max_says", ""))
        if len(max_says) < 15:
            rejected["too_short"] += 1
            continue
        if "type" not in d:
            rejected["missing_type"] += 1
            continue
        if d["type"] not in ("dialogue", "silent_speech", "monologue"):
            rejected["invalid_type"] += 1
            continue

        # Проверка: цитата реально из целевого PRIMARY CHUNK.
        if source_chunk and not fuzzy_find(
            max_says,
            source_chunk,
            normalized_haystack=normalized_source_chunk,
        ):
            rejected["max_says_not_found"] += 1
            continue

        # Если есть interlocutor_says — тоже проверяем
        interlocutor_says = strip_text(d.get("interlocutor_says", ""))
        if source_chunk and interlocutor_says and len(interlocutor_says) > 15:
            in_primary = fuzzy_find(
                interlocutor_says,
                source_chunk,
                normalized_haystack=normalized_source_chunk,
            )
            in_support = (
                support_chunk
                and fuzzy_find(
                    interlocutor_says,
                    support_chunk,
                    normalized_haystack=normalized_support_chunk,
                )
            )
            if not in_primary and not in_support:
                rejected["interlocutor_not_found"] += 1
                continue

        cleaned = dict(d)
        cleaned.pop("_source", None)
        valid.append(cleaned)

    rejected_total = sum(rejected.values())
    if log_prefix and rejected_total > 0:
        details = ", ".join(
            f"{reason}={count}" for reason, count in rejected.items() if count
        )
        log_event(
            f"{log_prefix} валидация голоса: {len(valid)}/{len(items)} прошло, "
            f"отброшено {rejected_total} ({details})"
        )

    return valid


def validate_knowledge(items: list[dict], log_prefix: str = "") -> list[dict]:
    """Фильтрует невалидные/мусорные записи из извлечённых фактов."""
    valid = []
    rejected = {
        "short_fact": 0,
        "short_subject": 0,
        "bad_category": 0,
    }
    for f in items:
        fact = strip_text(f.get("fact", ""))
        subject = strip_text(f.get("subject", ""))
        # Факт и субъект обязательны
        if len(fact) < 10:
            rejected["short_fact"] += 1
            continue
        if len(subject) < 2:
            rejected["short_subject"] += 1
            continue
        # Категория должна быть одной из ожидаемых
        cat = f.get("category", "")
        if cat not in ("character", "place", "magic", "history", "event", "creature", "custom"):
            rejected["bad_category"] += 1
            continue
        cleaned = dict(f)
        cleaned["subject"] = subject
        cleaned["fact"] = fact
        cleaned["time_scope"] = normalize_time_scope(
            cleaned.get("time_scope", ""),
            fact=fact,
            category=cat,
        )
        valid.append(cleaned)

    rejected_total = sum(rejected.values())
    if log_prefix and rejected_total > 0:
        details = ", ".join(
            f"{reason}={count}" for reason, count in rejected.items() if count
        )
        log_event(
            f"{log_prefix} валидация знаний: {len(valid)}/{len(items)} прошло, "
            f"отброшено {rejected_total} ({details})"
        )
    return valid


# ──────────────────────────────────────────────
# Шаг 4: Формирование обучающих пар
# ──────────────────────────────────────────────

def make_training_pairs(dialogues: list[dict], config: Config) -> list[dict]:
    """Превращает извлечённые диалоги в обучающие пары для fine-tune."""
    pairs = []

    for d in dialogues:
        max_says = strip_text(d.get("max_says", ""))
        if not max_says or len(max_says) < 10:
            continue

        dtype = d.get("type", "dialogue")
        context = strip_text(d.get("context", ""))
        interlocutor = d.get("interlocutor")
        interlocutor_says = strip_text(d.get("interlocutor_says", ""))

        # Формируем user-промпт в зависимости от типа
        if dtype in ("dialogue", "silent_speech") and interlocutor_says:
            prefix = "[Безмолвная речь] " if dtype == "silent_speech" else ""
            if interlocutor:
                user_content = f"[{context}]\n\n{prefix}{interlocutor}: {interlocutor_says}"
            else:
                user_content = f"[{context}]\n\n{prefix}{interlocutor_says}"
        elif dtype == "monologue" and context:
            user_content = f"[{context}]\n\nЧто ты об этом думаешь?"
        elif dtype == "silent_speech" and context:
            user_content = f"[{context}]\n\n[Безмолвная речь] Что скажешь?"
        else:
            user_content = context if context else "Расскажи, что происходит."

        pair = {
            "messages": [
                {"role": "system", "content": config.character_system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": max_says},
            ]
        }
        pairs.append(pair)

    return pairs


# ──────────────────────────────────────────────
# Проход 2: Извлечение знаний о мире
# ──────────────────────────────────────────────

KNOWLEDGE_SYSTEM = """Ты — помощник для извлечения фактов из книг Макса Фрая.
Извлекаешь структурированные знания о мире Ехо, персонажах и событиях.
Отвечай СТРОГО в формате JSON. Никакого текста до или после JSON."""

KNOWLEDGE_PROMPT = """Из фрагмента книги Макса Фрая извлеки ФАКТЫ о мире Ехо, персонажах и событиях.

Категории фактов:
- "character": информация о персонаже (биография, способности, характер, отношения)
- "place": место (город, трактир, улица, здание, мир)
- "magic": магия (заклинания, ступени, ордена, артефакты, Очевидная/Истинная магия)
- "history": исторические события (Смутные Времена, Эпоха Орденов, войны, Кодекс)
- "event": сюжетные и сценические события, происшествия, решения, открытия, действия с последствиями
- "creature": существа (буривухи, овчарки Пустых Земель и т.п.)
- "custom": обычаи, быт, еда, напитки, одежда, транспорт, социальные нормы

Для каждого факта создай объект:
- "category": одна из категорий выше
- "subject": о ком или о чём факт. Используй КОНКРЕТНЫЙ и самодостаточный subject
  из текста (полное имя персонажа, точное название места, предмета или события).
- "fact": сам факт, кратко и точно (1-3 предложения)
- "time_scope": временной характер факта:
  "past" — прежнее состояние, прошлый этап;
  "current" — текущее состояние на момент сцены;
  "change" — изменение, переход, результат события;
  "ended" — прекращение, отмена, утрата, смерть, исчезновение;
  "timeless" — устойчивое знание о мире, роли, устройстве, обычае;
  "unclear" — если по тексту это нельзя надёжно определить

ПРАВИЛА:
— Извлекай только то, что явно сказано в тексте, не додумывай.
— Один объект = один факт. Не объединяй разные факты.
— Извлекай ПОЛНО, но только самостоятельные и полезные факты:
  персонажи, места, предметы мира, магию, отношения, состояния, события, причины, последствия, локальные открытия сцены.
— Перед добавлением факта мысленно проверь:
  можно ли понять subject вне текущего предложения,
  полезен ли факт сам по себе,
  не является ли это просто разовой деталью кадра.
— Не ограничивайся энциклопедическими знаниями: важные сюжетные события и изменения состояния тоже нужны.
— Не пропускай факт только потому, что он локальный, если он помогает понять мир, персонажей или ход событий.
— Разбивай длинные комплексные утверждения на несколько атомарных фактов.
- Если текст описывает изменение состояния во времени, извлекай разные фазы ОТДЕЛЬНО:
  например "раньше X", "теперь Y", "после события Z". Не сглаживай противоречия и не сливай
  старое и новое состояние в один усреднённый факт.
- Не используй мусорные placeholders в subject вроде: "место", "место действия", "улица",
  "кабинет", "дом", "персонаж", "человек", если в тексте есть более точное обозначение.
- Для category="place" используй самую конкретную форму из PRIMARY CHUNK:
  например "улица Желтых Камней", "Дом у Моста", "кабинет Короля",
  "квартира на улице Желтых Камней", а не общие слова.
- Не используй как place предметы и части сцены вроде "дверь", "окно", "лестница",
  а также сцепленные описания вроде "дом и сад", если это не собственное название места.
- Для category="custom" извлекай только УСТОЙЧИВЫЕ элементы мира и быта:
  именованные напитки, еду, одежду, транспорт, ритуалы, социальные правила, профессии, институты и повторяющиеся практики.
- Для category="custom" subject должен быть названием самого обычая/предмета/практики:
  например "камра", "лоохи", "амобилер", "Королевский голос", а не участниками сцены и не описанием момента.
- Не создавай category="custom" для разовых сценических деталей вроде "ведьмочки пили камру",
  анонимных групп, настроений сцены или безымянных описательных предметов вроде
  "полумесяц из плотной ткани с карманами". Если у предмета нет устойчивого названия в тексте — пропусти его
  или отнеси сам факт к "event", если важна именно сцена.
— Если в фрагменте нет значимых фактов — верни пустой массив [].
— Верни столько полезных фактов, сколько реально помещается в ответ; не обрезай искусственно список до 8.
— Если ниже есть SUPPORTING CONTEXT, используй его только для понимания имён, местоимений, сцены,
  причин и последствий.
— Если ниже есть SCENE GLOSSARY, используй его только как справочник каноничных имён и сущностей.
  Он НЕ добавляет новых фактов и не заменяет PRIMARY CHUNK.
— Извлекать нужно только факты и события, которые явно присутствуют в PRIMARY CHUNK.
{pagination_note}
НЕ ПИШИ НИЧЕГО, КРОМЕ JSON. НИКАКИХ ОБЪЯСНЕНИЙ ИЛИ ВВОДНЫХ СЛОВ.
НЕ используй ```json или другие markdown-обёртки — только чистый JSON.

Материал:
---
{chunk_payload}
---

JSON:"""


def extract_knowledge(
    client: OpenAI,
    config: Config,
    chunk: str,
    log_prefix: str = "",
    chunk_payload: Optional[str] = None,
) -> list[dict]:
    """Извлекает факты о мире из фрагмента."""
    payload = chunk_payload or f"[PRIMARY CHUNK]\n{chunk}"
    extracted: list[dict] = []

    for pass_idx in range(max(config.extraction_passes, 1)):
        response = call_llm(
            client,
            config,
            KNOWLEDGE_SYSTEM,
            KNOWLEDGE_PROMPT.format(
                chunk_payload=payload,
                pagination_note=make_knowledge_pagination_note(pass_idx, extracted),
            ),
            max_tokens=config.max_tokens_knowledge,
            response_format="json" if _use_ollama_native else None,
            log_prefix=f"{log_prefix}[page {pass_idx + 1}]" if log_prefix else "",
        )

        if response is None:
            break

        data, strategy = parse_json_response(
            response,
            expect="array",
            log_prefix=f"{log_prefix}[page {pass_idx + 1}]" if log_prefix else "",
        )
        if not isinstance(data, list):
            if log_prefix:
                log_event(f"{log_prefix}[page {pass_idx + 1}] ответ не удалось распарсить как JSON-массив")
            break

        page_items = validate_knowledge(
            data,
            log_prefix=f"{log_prefix}[page {pass_idx + 1}]" if log_prefix else "",
        )
        extracted, added = merge_knowledge_items(extracted, page_items)
        if log_prefix:
            log_event(
                f"{log_prefix}[page {pass_idx + 1}] JSON ok: {len(data)} фактов "
                f"({strategy}), новых={added}, накоплено={len(extracted)}"
            )
        if not page_items or added == 0:
            break

    return extracted


_SUBJECT_PREFIX_NOISE = {
    "сэр", "леди", "господин", "госпожа", "мадам", "месье",
    "город", "трактир", "орден", "графство", "эпоха", "кодекс",
    "район", "остров", "улица", "мир", "пёс", "пес",
}

_FACT_STOPWORDS = {
    "и", "или", "а", "но", "что", "как", "это", "этот", "эта", "эти",
    "он", "она", "они", "его", "ее", "её", "их", "для", "при", "из",
    "в", "во", "на", "по", "к", "ко", "у", "с", "со", "от", "до",
    "над", "под", "же", "ли", "не", "ни", "был", "была", "были",
    "было", "является", "являются", "который", "которая", "которое",
    "которые", "где",
}

_GENERIC_NARRATOR_SUBJECTS = {
    "я",
    "я рассказчик",
    "я автор",
    "рассказчик",
    "автор",
    "главный герой",
    "протагонист",
}

_CHARACTER_HONORIFICS = {
    "сэр", "леди", "господин", "госпожа", "мадам", "месье",
}

CANONICAL_CHARACTERS = {
    "джуффин": "Джуффин Халли",
    "халли": "Джуффин Халли",
    "шурф": "Шурф Лонли-Локли",
    "лонли-локли": "Шурф Лонли-Локли",
    "лонли": "Шурф Лонли-Локли",
    "безумный рыбник": "Безумный Рыбник",
    "мелифаро": "Мелифаро",
    "кофа": "Кофа Йох",
    "йох": "Кофа Йох",
    "меламори": "Меламори Блимм",
    "блимм": "Меламори Блимм",
    "нумминорих": "Нумминорих Кута",
    "кута": "Нумминорих Кута",
    "теххи": "Теххи Шекк",
    "шекк": "Теххи Шекк",
    "сотофа": "Сотофа Ханемер",
    "ханемер": "Сотофа Ханемер",
    "лойсо": "Лойсо Пондохва",
    "пондохва": "Лойсо Пондохва",
    "нуфлин": "Нуфлин Мони Мах",
    "мони мах": "Нуфлин Мони Мах",
    "гуриг": "Гуриг VIII",
    "маба": "Маба Калох",
    "калох": "Маба Калох",
    "друппи": "Друппи",
    "куруш": "Куруш",
    "франк": "Франк",
    "базилио": "Базилио",
    "дримарондо": "Дримарондо",
    "макс": "Макс",
    "макс фрай": "Макс",
    "хельна": "Хельна",
    "кекки": "Кекки Туотли",
    "туотли": "Кекки Туотли",
    "анчифа": "Анчифа Мелифаро",
    "мохи": "Мохи Фаа",
    "кима": "Кима",
    "алотхо": "Алотхо Аллирох",
    "аллирох": "Алотхо Аллирох",
    "махи": "Махи Аинти",
    "аинти": "Махи Аинти",
    "луукфи": "Луукфи Пэнц",
    "пэнц": "Луукфи Пэнц",
    "шихола": "Шихола",
}

CANONICAL_CHARACTERS_EXACT_ONLY = {
    "безумный рыбник",
    "мелифаро",
}

_GENERIC_SUBJECTS_ANY = {
    "место",
    "место действия",
    "локация",
    "персонаж",
    "герой",
    "героиня",
    "человек",
    "мужчина",
    "женщина",
    "собеседник",
    "гость",
    "журналист",
    "репортер",
    "рассказчик",
    "автор",
    "существо",
}

_GENERIC_CHARACTER_SUBJECTS = _GENERIC_SUBJECTS_ANY | {
    "некто",
    "кто то",
    "кто-то",
    "персонаж книги",
    "действующее лицо",
    "один из магистров",
    "неизвестный маг",
    "старый знакомый",
    "старая знакомая",
    "его друг",
    "ее подруга",
    "её подруга",
    "молодой человек",
    "пожилой человек",
    "незнакомец",
    "незнакомка",
    "посетитель",
    "прохожий",
}

_GENERIC_PLACE_SUBJECTS = {
    "место",
    "место действия",
    "локация",
    "местность",
    "направление",
    "какое то место",
    "какое-то место",
    "где то",
    "где-то",
    "одно место",
    "некое место",
    "это место",
    "то место",
    "здесь",
    "там",
    "тут",
    "снаружи",
    "внутри",
    "далеко",
    "неподалеку",
    "неподалёку",
    "рядом",
    "поблизости",
    "в каком то месте",
    "в каком-то месте",
    "в одном месте",
    "каком то месте",
    "каком-то месте",
    "одном месте",
}

_LEADING_PREPOSITIONS = {
    "к", "ко", "в", "во", "на", "у", "из", "с", "со", "от", "до", "по",
}

_PLACE_HEADWORD_CANONICAL = {
    "улица": "улица",
    "улице": "улица",
    "улицу": "улица",
    "улицы": "улица",
    "дом": "дом",
    "доме": "дом",
    "дома": "дом",
    "дому": "дом",
    "домом": "дом",
    "квартира": "квартира",
    "квартире": "квартира",
    "квартиру": "квартира",
    "квартиры": "квартира",
    "кабинет": "кабинет",
    "кабинете": "кабинет",
    "кабинету": "кабинет",
    "комната": "комната",
    "комнате": "комната",
    "комнату": "комната",
    "комнаты": "комната",
    "крыша": "крыша",
    "крыше": "крыша",
    "крышу": "крыша",
    "трактир": "трактир",
    "трактире": "трактир",
    "трактира": "трактир",
    "мост": "мост",
    "мосту": "мост",
    "моста": "мост",
    "город": "город",
    "городе": "город",
    "города": "город",
}

_SCENE_GLOSSARY_WORLD_TERMS = {
    "Ехо": ("place", ("ехо",)),
    "Дом у Моста": ("place", ("дом у моста", "дома у моста")),
    "Старый Город": ("place", ("старый город",)),
    "Тихий Город": ("place", ("тихий город",)),
    "Тёмная Сторона": ("magic", ("темная сторона", "тёмная сторона")),
    "Кодекс Хрембера": ("history", ("кодекс хрембера",)),
    "Тайный Сыск": ("custom", ("тайный сыск",)),
    "Королевский голос": ("custom", ("королевский голос",)),
    "камра": ("custom", ("камра", "камру", "камры", "камрой")),
    "лоохи": ("custom", ("лоохи", "лоохи", "лоохи")),
    "амобилер": ("custom", ("амобилер", "амобилеры")),
    "Безмолвная речь": ("magic", ("безмолвная речь",)),
    "Смертный Шар": ("magic", ("смертный шар",)),
}

_TIME_SCOPE_ALLOWED = {"past", "current", "change", "ended", "timeless", "unclear"}
_TIME_SCOPE_ALIASES = {
    "past": "past",
    "former": "past",
    "before": "past",
    "current": "current",
    "present": "current",
    "now": "current",
    "change": "change",
    "changed": "change",
    "transition": "change",
    "ended": "ended",
    "end": "ended",
    "ceased": "ended",
    "stopped": "ended",
    "timeless": "timeless",
    "stable": "timeless",
    "general": "timeless",
    "unclear": "unclear",
    "unknown": "unclear",
}

_NON_PLACE_OBJECT_HEADWORDS = {
    "дверь",
    "двери",
    "окно",
    "окна",
    "ворота",
    "калитка",
    "вывеска",
    "табличка",
    "надпись",
    "лестница",
    "ступени",
    "ступеньки",
    "порог",
    "стена",
    "стены",
}

_GENERIC_PLACE_BUNDLE_TOKENS = set(_PLACE_HEADWORD_CANONICAL.values()) | {
    "сад",
    "двор",
    "дворик",
    "коридор",
    "кухня",
    "спальня",
    "гостиная",
    "зал",
    "площадь",
    "переулок",
    "дорога",
    "парк",
    "берег",
    "подвал",
    "чердак",
}


def match_word_case(template: str, value: str) -> str:
    """Подгоняет регистр канонического слова под исходный шаблон."""
    if not template:
        return value
    if template.isupper():
        return value.upper()
    if template[:1].isupper():
        return value[:1].upper() + value[1:]
    return value


def normalize_dedup_text(text: str) -> str:
    """Нормализует текст для устойчивой дедупликации."""
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[«»“”„‟\"'`]", " ", text)
    text = re.sub(r"[-–—]+", " ", text)
    text = re.sub(r"[^0-9a-zа-я\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_character_alias_groups() -> dict[str, set[str]]:
    """Группирует алиасы персонажей по каноническому имени для scene glossary."""
    cache = getattr(canonical_character_alias_groups, "_cache", None)
    if cache is None:
        grouped: dict[str, set[str]] = {}
        for alias, canonical in CANONICAL_CHARACTERS.items():
            grouped.setdefault(canonical, set()).add(normalize_subject_for_dedup(alias))
        canonical_character_alias_groups._cache = grouped
        cache = grouped
    return cache


def build_scene_glossary(text: str, limit: int = 16) -> str:
    """Строит короткий glossary по сущностям сцены для более точного extraction."""
    if not text:
        return ""

    normalized = f" {normalize_dedup_text(text)} "
    lines: list[str] = []
    seen: set[str] = set()

    def add_line(label: str, value: str):
        key = f"{label}:{normalize_dedup_text(value)}"
        if not value or key in seen or len(lines) >= limit:
            return
        seen.add(key)
        lines.append(f"- {label}: {value}")

    for canonical, aliases in sorted(canonical_character_alias_groups().items()):
        if any(f" {alias} " in normalized for alias in aliases if alias):
            add_line("character", canonical)

    for display_name, (label, aliases) in _SCENE_GLOSSARY_WORLD_TERMS.items():
        if any(f" {normalize_subject_for_dedup(alias)} " in normalized for alias in aliases):
            add_line(label, display_name)

    place_patterns = [
        r"\bДом(?:а)? у [А-ЯЁ][А-Яа-яё-]+(?: [А-ЯЁ][А-Яа-яё-]+)?",
        r"\bулиц(?:а|е|у|ы) [А-ЯЁ][^,\n.;:!?]{1,40}",
        r"\bкабинет(?:е|а)? [А-ЯЁ][^,\n.;:!?]{1,40}",
        r"\bквартир(?:а|е|у) на улиц(?:е|у|ы|а) [А-ЯЁ][^,\n.;:!?]{1,40}",
    ]
    for pattern in place_patterns:
        for match in re.finditer(pattern, text):
            add_line("place", canonicalize_place_subject(strip_text(match.group(0))))
            if len(lines) >= limit:
                break
        if len(lines) >= limit:
            break

    return "\n".join(lines)


def infer_time_scope_from_fact(fact: str, category: str = "") -> str:
    """Грубо определяет временной характер факта по формулировке."""
    tokens = set(dedup_word_tokens(fact))
    if not tokens:
        return "unclear"

    ended_markers = {
        "больше", "не", "перестал", "перестала", "перестали", "исчез", "исчезла",
        "умер", "умерла", "мертв", "мертва", "отменен", "отменён", "лишился",
    }
    change_markers = {
        "стал", "стала", "стали", "теперь", "снова", "вновь", "после",
        "впоследствии", "вернулся", "вернулась", "вернулись", "изменился", "изменилась",
    }
    past_markers = {
        "раньше", "прежде", "бывший", "бывшая", "бывшее", "некогда", "когда", "когда то", "когда-то",
    }
    current_markers = {"сейчас", "ныне", "теперь"}

    if tokens & ended_markers:
        return "ended"
    if tokens & change_markers:
        return "change"
    if tokens & past_markers:
        return "past"
    if tokens & current_markers:
        return "current"
    if category in {"custom", "magic", "place", "creature"}:
        return "timeless"
    return "unclear"


def normalize_time_scope(value: Any, *, fact: str = "", category: str = "") -> str:
    """Нормализует time_scope, а при отсутствии аккуратно выводит его из текста."""
    normalized = normalize_dedup_text(str(value or ""))
    if normalized in _TIME_SCOPE_ALLOWED:
        return normalized
    if normalized in _TIME_SCOPE_ALIASES:
        return _TIME_SCOPE_ALIASES[normalized]
    return infer_time_scope_from_fact(fact, category=category)


def time_scopes_meaningfully_differ(left_item: dict, right_item: dict) -> bool:
    """Проверяет, что два факта различаются по явному временному состоянию."""
    left_scope = normalize_time_scope(
        left_item.get("time_scope", ""),
        fact=strip_text(left_item.get("fact", "")),
        category=strip_text(left_item.get("category", "")),
    )
    right_scope = normalize_time_scope(
        right_item.get("time_scope", ""),
        fact=strip_text(right_item.get("fact", "")),
        category=strip_text(right_item.get("category", "")),
    )
    informative_scopes = {"past", "current", "change", "ended"}
    return (
        left_scope in informative_scopes
        and right_scope in informative_scopes
        and left_scope != right_scope
    )


def dedup_word_tokens(text: str) -> list[str]:
    """Разбивает нормализованный текст на токены."""
    return re.findall(r"[0-9a-zа-я]+", normalize_dedup_text(text), flags=re.IGNORECASE)


def normalize_subject_for_dedup(subject: str) -> str:
    """Нормализует subject, схлопывая почётные титулы и типовые префиксы."""
    tokens = dedup_word_tokens(subject)
    while len(tokens) > 1 and tokens[0] in _SUBJECT_PREFIX_NOISE:
        tokens = tokens[1:]
    return " ".join(tokens)


def canonical_character_lookup_map() -> dict[str, str]:
    """Кэшированная нормализованная карта канонических имён персонажей."""
    cache = getattr(canonical_character_lookup_map, "_cache", None)
    if cache is None:
        cache = {
            normalize_subject_for_dedup(key): value
            for key, value in CANONICAL_CHARACTERS.items()
        }
        canonical_character_lookup_map._cache = cache
    return cache


def canonical_character_exact_only_keys() -> set[str]:
    """Нормализованный набор алиасов, которые допустимы только при точном совпадении."""
    cache = getattr(canonical_character_exact_only_keys, "_cache", None)
    if cache is None:
        cache = {
            normalize_subject_for_dedup(key)
            for key in CANONICAL_CHARACTERS_EXACT_ONLY
        }
        canonical_character_exact_only_keys._cache = cache
    return cache


def lookup_canonical_character(subject: str) -> Optional[str]:
    """Пытается найти каноническое имя персонажа по subject."""
    cleaned = strip_leading_character_honorifics(strip_text(subject))
    normalized = normalize_subject_for_dedup(cleaned)
    if not normalized:
        return None

    lookup = canonical_character_lookup_map()
    exact_only = canonical_character_exact_only_keys()
    if normalized in lookup:
        return lookup[normalized]

    tokens = normalized.split()
    if len(tokens) >= 2:
        candidate = " ".join(tokens[:2])
        if candidate in lookup:
            return lookup[candidate]
    if tokens and tokens[0] in lookup and tokens[0] not in exact_only:
        return lookup[tokens[0]]
    return None


def subject_signature_tokens(subject: str) -> set[str]:
    """Возвращает ключевые токены subject без общих сущностных слов."""
    tokens = normalize_subject_for_dedup(subject).split()
    filtered = [token for token in tokens if token not in _SUBJECT_PREFIX_NOISE]
    return set(filtered or tokens)


def narrator_aliases(narrator: str) -> set[str]:
    """Возвращает набор допустимых subject-алиасов для текущего рассказчика."""
    if not narrator or narrator == "_сборник":
        return set()

    normalized = normalize_subject_for_dedup(narrator)
    aliases = {normalized}
    tokens = normalized.split()
    if tokens:
        aliases.add(tokens[0])
        if len(tokens) >= 2:
            aliases.add(" ".join(tokens[:2]))

    if narrator == "Макс":
        aliases.update({
            "макс",
            "макс фрай",
            "сэр макс",
            "сэр макс фрай",
        })

    return {alias for alias in aliases if alias}


def content_tokens_for_fact(text: str) -> set[str]:
    """Выделяет значимые токены факта без служебных слов."""
    return {
        token for token in dedup_word_tokens(text)
        if len(token) >= 3 and token not in _FACT_STOPWORDS
    }


_FACT_NEGATION_MARKERS = {
    "не",
    "нет",
    "никогда",
    "никак",
    "больше",
    "уже",
    "перестал",
    "перестала",
    "перестали",
    "лишился",
    "лишилась",
    "отменен",
    "отменена",
    "отменено",
    "отменены",
    "отменен",
    "отменён",
    "отменена",
    "отменено",
    "отменены",
}

_FACT_TEMPORAL_MARKERS = {
    "раньше",
    "прежде",
    "когда",
    "когда-то",
    "когда",
    "теперь",
    "сейчас",
    "позже",
    "потом",
    "впоследствии",
    "некогда",
    "прежний",
    "прежняя",
    "прежнее",
    "бывший",
    "бывшая",
    "бывшее",
    "стал",
    "стала",
    "стали",
    "снова",
    "вновь",
    "вернулся",
    "вернулась",
    "вернулись",
    "после",
    "до",
}

_FACT_CONTRADICTION_TOKEN_GROUPS = [
    ({"жив", "жива", "живы"}, {"мертв", "мертва", "мертвый", "мертвая", "умер", "умерла", "погиб", "погибла", "мертвец"}),
    ({"действует", "работает", "существует"}, {"не", "отменен", "отменён", "перестал", "исчез", "упразднен", "упразднён"}),
    ({"женат", "замужем"}, {"развелся", "развёлся", "развелась", "развёлся", "не", "больше"}),
    ({"вернулся", "вернулась", "вернулись", "появился", "появилась"}, {"уехал", "уехала", "исчез", "исчезла", "пропал", "пропала"}),
]


def fact_marker_tokens(text: str) -> set[str]:
    """Выделяет токены-маркеры отрицания, времени и смены состояния."""
    tokens = set(dedup_word_tokens(text))
    return {
        token for token in tokens
        if token in _FACT_NEGATION_MARKERS or token in _FACT_TEMPORAL_MARKERS
    }


def facts_describe_different_states(left: str, right: str) -> bool:
    """Осторожно определяет, что похожие факты могут описывать разные состояния мира."""
    left_norm = normalize_dedup_text(left)
    right_norm = normalize_dedup_text(right)
    if not left_norm or not right_norm or left_norm == right_norm:
        return False

    left_tokens = content_tokens_for_fact(left_norm)
    right_tokens = content_tokens_for_fact(right_norm)
    shared_tokens = left_tokens & right_tokens

    if len(shared_tokens) < 2:
        return False

    left_all_tokens = set(dedup_word_tokens(left_norm))
    right_all_tokens = set(dedup_word_tokens(right_norm))

    for positive_group, negative_group in _FACT_CONTRADICTION_TOKEN_GROUPS:
        if (left_all_tokens & positive_group and right_all_tokens & negative_group) or (
            right_all_tokens & positive_group and left_all_tokens & negative_group
        ):
            return True

    left_has_negation = any(token in left_all_tokens for token in _FACT_NEGATION_MARKERS)
    right_has_negation = any(token in right_all_tokens for token in _FACT_NEGATION_MARKERS)
    if left_has_negation != right_has_negation:
        return True

    left_markers = fact_marker_tokens(left_norm)
    right_markers = fact_marker_tokens(right_norm)
    if left_markers != right_markers and (left_markers or right_markers):
        return True

    return False


def leading_fact_clause(text: str) -> str:
    """Берёт первую смысловую клаузу факта для сравнения почти одинаковых формулировок."""
    first_clause = re.split(r"[,:;.!?]", text or "", maxsplit=1)[0]
    return normalize_dedup_text(first_clause)


def canonicalize_narrator_subject(subject: str, narrator: str) -> str:
    """Приводит subject рассказчика к каноническому имени книги."""
    subject_clean = (subject or "").strip()
    if not subject_clean or not narrator or narrator == "_сборник":
        return subject_clean

    canonical_character = lookup_canonical_character(subject_clean)
    if canonical_character:
        return canonical_character

    normalized = normalize_subject_for_dedup(subject_clean)
    if normalized in _GENERIC_NARRATOR_SUBJECTS:
        return narrator

    aliases = narrator_aliases(narrator)
    if normalized in aliases:
        return narrator

    subject_tokens = subject_signature_tokens(subject_clean)
    narrator_tokens = subject_signature_tokens(narrator)
    if subject_tokens and narrator_tokens:
        if subject_tokens == narrator_tokens:
            return narrator
        if len(subject_tokens) == 1 and subject_tokens.issubset(narrator_tokens):
            return narrator

    return subject_clean


def strip_leading_character_honorifics(subject: str) -> str:
    """Убирает обязательные для схлопывания титулы персонажей, сохраняя само имя."""
    cleaned = strip_text(subject)
    while cleaned:
        updated = re.sub(
            r"^(?:сэр|леди|господин|госпожа|мадам|месье)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        if updated == cleaned:
            break
        cleaned = updated.strip()
    return cleaned


def best_subject_display(left: str, right: str) -> str:
    """Выбирает более информативный вариант отображаемого имени."""
    if not left:
        return right
    if not right:
        return left

    left_tokens = dedup_word_tokens(left)
    right_tokens = dedup_word_tokens(right)
    left_score = (len(left_tokens), len(left), left)
    right_score = (len(right_tokens), len(right), right)
    return right if right_score > left_score else left


def normalized_subject_key(subject: str) -> str:
    """Ключ для сравнения вариантов subject без пунктуации и регистра."""
    return normalize_dedup_text(subject)


def canonicalize_place_subject(subject: str) -> str:
    """Нормализует названия мест: убирает предлоги и приводит headword к канону."""
    cleaned = strip_text(subject)
    if not cleaned:
        return cleaned

    tokens = cleaned.split()
    while len(tokens) > 1 and normalize_dedup_text(tokens[0]) in _LEADING_PREPOSITIONS:
        tokens = tokens[1:]

    if not tokens:
        return ""

    first_norm = normalize_dedup_text(tokens[0])
    canonical_first = _PLACE_HEADWORD_CANONICAL.get(first_norm)
    if canonical_first:
        tokens[0] = match_word_case(tokens[0], canonical_first)

    return " ".join(tokens).strip(" \t\n\r,;:.!?-")


def looks_like_descriptive_place_fragment(subject: str) -> bool:
    """Отсеивает предметы и составные сценические описания, которые не стоит хранить как place."""
    normalized = normalize_dedup_text(subject)
    if not normalized:
        return True

    tokens = normalized.split()
    if not tokens:
        return True

    if tokens[0] in _NON_PLACE_OBJECT_HEADWORDS:
        return True

    if len(tokens) >= 2 and tokens[1] in _NON_PLACE_OBJECT_HEADWORDS:
        return True

    joiners = {"и", "или"}
    if any(token in joiners for token in tokens):
        meaningful = [
            token for token in tokens
            if token not in _LEADING_PREPOSITIONS and token not in joiners
        ]
        if meaningful and all(
            token in _GENERIC_PLACE_BUNDLE_TOKENS or token in _NON_PLACE_OBJECT_HEADWORDS
            for token in meaningful
        ):
            return True

    return False


def is_placeholder_subject(subject: str, category: str, narrator: str = "") -> bool:
    """Опознаёт мусорные и слишком общие subject, которые не стоит сохранять."""
    cleaned = strip_text(subject)
    if not cleaned:
        return True

    if narrator and cleaned == narrator:
        return False

    if category == "character":
        subject_for_check = strip_leading_character_honorifics(cleaned)
    elif category == "place":
        subject_for_check = canonicalize_place_subject(cleaned)
    else:
        subject_for_check = cleaned
    normalized = normalize_dedup_text(subject_for_check)
    if not normalized:
        return True

    if narrator and normalized in narrator_aliases(narrator):
        return False

    if normalized in _GENERIC_SUBJECTS_ANY:
        return True

    if category == "character" and normalized in _GENERIC_CHARACTER_SUBJECTS:
        return True

    if category == "place" and normalized in _GENERIC_PLACE_SUBJECTS:
        return True

    if re.match(
        r"^(этот|эта|это|эти|тот|та|те|то|"
        r"один|одна|одно|одни|"
        r"некий|некая|некое|некие|некто|нечто|"
        r"какой то|какая то|какое то|какие то|"
        r"чей то|чья то|чье то|"
        r"его|ее|их|"
        r"другой|другая|другое|другие|"
        r"неизвестный|неизвестная|неизвестное|"
        r"старый знакомый|старая знакомая)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True

    if category == "place" and re.match(
        r"^(где то|одно место|какое то|некое|в каком то|каком то|одном месте|некоторое)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True

    if category == "place":
        place_tokens = dedup_word_tokens(subject_for_check)
        canonical_place_headwords = set(_PLACE_HEADWORD_CANONICAL.values())
        if len(place_tokens) == 1 and place_tokens[0] in canonical_place_headwords:
            return True
        if looks_like_descriptive_place_fragment(subject_for_check):
            return True

    if category == "character" and normalized.startswith("место"):
        return True

    return False


def build_character_subject_alias_map(items: list[dict], narrator: str) -> dict[str, str]:
    """Строит безопасную карту алиасов персонажей в рамках одной книги."""
    best_display_by_core: dict[str, str] = {}
    unique_full_core_by_first_token: dict[str, Optional[str]] = {}
    entries = []

    for item in items:
        if item.get("category") != "character":
            continue

        subject = strip_text(item.get("subject", ""))
        if not subject:
            continue

        if narrator:
            subject = canonicalize_narrator_subject(subject, narrator)
        if is_placeholder_subject(subject, "character", narrator=narrator):
            continue

        display = strip_leading_character_honorifics(subject)
        tokens = dedup_word_tokens(display)
        if not tokens:
            continue

        core = " ".join(tokens)
        forced_canonical = lookup_canonical_character(display)
        forced_core = normalize_subject_for_dedup(forced_canonical) if forced_canonical else ""
        best_display_by_core[core] = best_subject_display(best_display_by_core.get(core, ""), display)
        if forced_canonical:
            best_display_by_core[forced_core] = forced_canonical
        entries.append((subject, display, tokens, core, forced_core, forced_canonical))

        if len(tokens) >= 2:
            first = tokens[0]
            if first not in unique_full_core_by_first_token:
                unique_full_core_by_first_token[first] = core
            elif unique_full_core_by_first_token[first] != core:
                unique_full_core_by_first_token[first] = None

    alias_map: dict[str, str] = {}
    for subject, display, tokens, core, forced_core, forced_canonical in entries:
        target_core = forced_core or core
        if not forced_core and len(tokens) == 1:
            unique_full = unique_full_core_by_first_token.get(tokens[0])
            if unique_full:
                target_core = unique_full

        canonical_display = best_display_by_core.get(target_core, forced_canonical or display)
        alias_map[normalized_subject_key(subject)] = canonical_display
        alias_map[normalized_subject_key(display)] = canonical_display

    return alias_map


def build_place_subject_alias_map(items: list[dict], narrator: str) -> dict[str, str]:
    """Строит безопасную карту канонических subject для мест."""
    best_display_by_core: dict[str, str] = {}
    entries = []

    for item in items:
        if item.get("category") != "place":
            continue

        original_subject = strip_text(item.get("subject", ""))
        canonical_subject = canonicalize_place_subject(original_subject)
        if not canonical_subject:
            continue
        if is_placeholder_subject(canonical_subject, "place", narrator=narrator):
            continue

        core = normalized_subject_key(canonical_subject)
        best_display_by_core[core] = best_subject_display(
            best_display_by_core.get(core, ""),
            canonical_subject,
        )
        entries.append((original_subject, canonical_subject, core))

    alias_map: dict[str, str] = {}
    for original_subject, canonical_subject, core in entries:
        best_display = best_display_by_core.get(core, canonical_subject)
        alias_map[normalized_subject_key(original_subject)] = best_display
        alias_map[normalized_subject_key(canonical_subject)] = best_display

    return alias_map


def rewrite_character_fact_text(text: str, canonical_subject: str, original_subject: str) -> str:
    """Подменяет обезличенное или короткое начало факта на каноническое имя персонажа."""
    updated = strip_text(text)
    if not updated or not canonical_subject:
        return updated

    original_display = strip_leading_character_honorifics(original_subject)
    replacements = [
        (r"^\s*он\b", canonical_subject),
        (r"^\s*она\b", canonical_subject),
        (r"^\s*тот\b", canonical_subject),
        (r"^\s*та\b", canonical_subject),
    ]

    if original_display:
        replacements.append((rf"^\s*{re.escape(original_display)}\b", canonical_subject))

    for pattern, repl in replacements:
        candidate = re.sub(pattern, repl, updated, flags=re.IGNORECASE)
        if candidate != updated:
            updated = candidate
            break

    return re.sub(r"\s+", " ", updated).strip()


def rewrite_place_fact_text(text: str, canonical_subject: str) -> str:
    """Подставляет конкретное место вместо обезличенного начала факта."""
    updated = strip_text(text)
    if not updated or not canonical_subject:
        return updated

    replacements = [
        (r"^\s*это место\b", canonical_subject),
        (r"^\s*здесь\b", canonical_subject),
        (r"^\s*там\b", canonical_subject),
    ]
    for pattern, repl in replacements:
        candidate = re.sub(pattern, repl, updated, flags=re.IGNORECASE)
        if candidate != updated:
            updated = candidate
            break

    return re.sub(r"\s+", " ", updated).strip()


def _apply_knowledge_alias_maps(
    items: list[dict],
    *,
    narrator: str,
    character_alias_map: dict[str, str],
    place_alias_map: dict[str, str],
    log_prefix: str = "",
) -> list[dict]:
    """Применяет подготовленные alias map к фактам знаний."""
    result = []
    rewrites = 0
    dropped = 0

    for item in items:
        cleaned = dict(item)
        category = cleaned.get("category", "")
        original_subject = strip_text(cleaned.get("subject", ""))
        cleaned["subject"] = original_subject
        cleaned["fact"] = strip_text(cleaned.get("fact", ""))
        cleaned["time_scope"] = normalize_time_scope(
            cleaned.get("time_scope", ""),
            fact=cleaned.get("fact", ""),
            category=category,
        )

        if category == "character":
            canonical_subject = character_alias_map.get(
                normalized_subject_key(original_subject),
                strip_leading_character_honorifics(original_subject),
            )
            canonical_subject = strip_text(canonical_subject)
            if canonical_subject and canonical_subject != original_subject:
                rewrites += 1
                cleaned["subject"] = canonical_subject
            cleaned["fact"] = rewrite_character_fact_text(
                cleaned.get("fact", ""),
                cleaned.get("subject", ""),
                original_subject,
            )
        elif category == "place":
            canonical_subject = place_alias_map.get(
                normalized_subject_key(original_subject),
                canonicalize_place_subject(original_subject),
            )
            canonical_subject = strip_text(canonical_subject)
            if canonical_subject and canonical_subject != original_subject:
                rewrites += 1
                cleaned["subject"] = canonical_subject
            cleaned["fact"] = rewrite_place_fact_text(
                cleaned.get("fact", ""),
                cleaned.get("subject", ""),
            )

        if is_placeholder_subject(cleaned.get("subject", ""), category, narrator=narrator):
            dropped += 1
            continue

        result.append(cleaned)

    if log_prefix and (rewrites or dropped):
        details = []
        if rewrites:
            details.append(f"subject_aliases={rewrites}")
        if dropped:
            details.append(f"dropped_placeholder={dropped}")
        log_event(f"{log_prefix} канонизация знаний: {', '.join(details)}")

    return result


def canonicalize_book_knowledge(
    items: list[dict],
    narrator: str,
    log_prefix: str = "",
) -> list[dict]:
    """Нормализует subject внутри книги: рассказчик, титулы, безопасные алиасы и мусорные subject."""
    if not items:
        return items

    normalized_items = normalize_knowledge_items(items, narrator, log_prefix=log_prefix)
    alias_map = build_character_subject_alias_map(normalized_items, narrator)
    place_alias_map = build_place_subject_alias_map(normalized_items, narrator)
    return _apply_knowledge_alias_maps(
        normalized_items,
        narrator=narrator,
        character_alias_map=alias_map,
        place_alias_map=place_alias_map,
        log_prefix=log_prefix,
    )


def canonicalize_global_knowledge(
    all_knowledge: list[dict],
    narrator: str = "Макс",
    log_prefix: str = "",
) -> list[dict]:
    """Глобально канонизирует знания между книгами до финальной дедупликации."""
    if not all_knowledge:
        return all_knowledge

    normalized_items = []
    for item in all_knowledge:
        cleaned = dict(item)
        cleaned["subject"] = strip_text(cleaned.get("subject", ""))
        cleaned["fact"] = strip_text(cleaned.get("fact", ""))
        normalized_items.append(cleaned)

    alias_map = build_character_subject_alias_map(normalized_items, narrator="")
    place_alias_map = build_place_subject_alias_map(normalized_items, narrator="")
    return _apply_knowledge_alias_maps(
        normalized_items,
        narrator="",
        character_alias_map=alias_map,
        place_alias_map=place_alias_map,
        log_prefix=log_prefix,
    )


def rewrite_narrator_fact_text(text: str, narrator: str) -> str:
    """Подчищает факт о рассказчике, заменяя обезличенные формулировки на имя."""
    updated = (text or "").strip()
    if not updated or not narrator or narrator == "_сборник":
        return updated

    replacements = [
        (r"\bрассказчик\b", narrator),
        (r"\bавтор\b", narrator),
        (r"\bглавный герой\b", narrator),
        (r"\bпротагонист\b", narrator),
    ]
    for pattern, repl in replacements:
        updated = re.sub(pattern, repl, updated, flags=re.IGNORECASE)

    updated = re.sub(r"^\s*я\b", narrator, updated, flags=re.IGNORECASE)
    updated = re.sub(r"^\s*он\b", narrator, updated, flags=re.IGNORECASE)
    updated = re.sub(r"^\s*она\b", narrator, updated, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", updated).strip()


def normalize_knowledge_items(
    items: list[dict],
    narrator: str,
    log_prefix: str = "",
) -> list[dict]:
    """Нормализует subject и текст фактов для текущего рассказчика."""
    if not items or narrator != "Макс":
        return items

    normalized_items = []
    subject_rewrites = 0
    fact_rewrites = 0

    for item in items:
        cleaned = dict(item)
        original_subject = strip_text(cleaned.get("subject", ""))
        original_fact = strip_text(cleaned.get("fact", ""))

        canonical_subject = canonicalize_narrator_subject(original_subject, narrator)
        if canonical_subject and canonical_subject != original_subject:
            subject_rewrites += 1
            cleaned["subject"] = canonical_subject

        if strip_text(cleaned.get("subject", "")) == narrator:
            rewritten_fact = rewrite_narrator_fact_text(original_fact, narrator)
            if rewritten_fact and rewritten_fact != original_fact:
                fact_rewrites += 1
                cleaned["fact"] = rewritten_fact

        normalized_items.append(cleaned)

    if log_prefix and (subject_rewrites or fact_rewrites):
        details = []
        if subject_rewrites:
            details.append(f"subject={subject_rewrites}")
        if fact_rewrites:
            details.append(f"fact={fact_rewrites}")
        log_event(f"{log_prefix} нормализация рассказчика: {', '.join(details)}")

    return normalized_items


def subjects_look_duplicate(left: str, right: str) -> bool:
    """Проверяет, что subjects относятся к одной сущности."""
    left_norm = normalize_subject_for_dedup(left)
    right_norm = normalize_subject_for_dedup(right)

    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_canonical = lookup_canonical_character(left)
    right_canonical = lookup_canonical_character(right)
    if left_canonical and right_canonical and left_canonical != right_canonical:
        return False

    exact_only = canonical_character_exact_only_keys()
    if left_norm in exact_only and right_norm != left_norm and left_canonical != right_canonical:
        return False
    if right_norm in exact_only and left_norm != right_norm and left_canonical != right_canonical:
        return False

    left_tokens = subject_signature_tokens(left)
    right_tokens = subject_signature_tokens(right)
    if not left_tokens or not right_tokens:
        return False

    if left_tokens == right_tokens:
        return True
    if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
        return abs(len(left_tokens) - len(right_tokens)) <= 1

    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return bool(overlap) and (len(overlap) / max(len(union), 1) >= 0.75)


def facts_look_duplicate(left: str, right: str) -> bool:
    """Проверяет, что два факта описывают одну и ту же мысль разными словами."""
    left_norm = normalize_dedup_text(left)
    right_norm = normalize_dedup_text(right)

    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if facts_describe_different_states(left, right):
        return False

    shorter, longer = sorted((left_norm, right_norm), key=len)
    if len(shorter) >= 40 and shorter in longer:
        return True

    left_clause = leading_fact_clause(left)
    right_clause = leading_fact_clause(right)
    if left_clause and left_clause == right_clause and len(left_clause) >= 18:
        return True

    ratio = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    if ratio >= 0.88:
        left_tokens = content_tokens_for_fact(left_norm)
        right_tokens = content_tokens_for_fact(right_norm)
        if left_tokens and right_tokens:
            overlap = left_tokens & right_tokens
            overlap_ratio = len(overlap) / max(min(len(left_tokens), len(right_tokens)), 1)
            if overlap_ratio < 0.6:
                return False
        return True

    left_all_tokens = dedup_word_tokens(left_norm)
    right_all_tokens = dedup_word_tokens(right_norm)
    common_prefix = 0
    for ltok, rtok in zip(left_all_tokens, right_all_tokens):
        if ltok != rtok:
            break
        common_prefix += 1
    if common_prefix >= 5 and common_prefix / max(min(len(left_all_tokens), len(right_all_tokens)), 1) >= 0.55:
        return True

    left_tokens = content_tokens_for_fact(left_norm)
    right_tokens = content_tokens_for_fact(right_norm)
    if not left_tokens or not right_tokens:
        return False

    overlap = left_tokens & right_tokens
    overlap_ratio = len(overlap) / max(min(len(left_tokens), len(right_tokens)), 1)
    if len(overlap) >= 3 and overlap_ratio >= 0.75:
        return True

    shorter_tokens, longer_tokens = sorted((left_tokens, right_tokens), key=len)
    return len(shorter_tokens) >= 3 and shorter_tokens.issubset(longer_tokens)


def deduplicate_knowledge(facts: list[dict]) -> list[dict]:
    """Убирает точные и почти одинаковые дубли фактов."""
    seen_exact = set()
    unique = []
    buckets: dict[str, list[dict]] = {}

    for f in facts:
        subject_raw = strip_text(f.get("subject", ""))
        fact_raw = strip_text(f.get("fact", ""))
        if not fact_raw:
            continue

        subject_norm = normalize_subject_for_dedup(subject_raw)
        fact_norm = normalize_dedup_text(fact_raw)
        time_scope = normalize_time_scope(
            f.get("time_scope", ""),
            fact=fact_raw,
            category=strip_text(f.get("category", "")),
        )
        exact_key = text_hash(f"{subject_norm}:{time_scope}:{fact_norm}")
        if exact_key in seen_exact:
            continue

        bucket_keys = []
        if subject_norm:
            bucket_keys.append(subject_norm)
            subject_tokens = subject_norm.split()
            if subject_tokens:
                bucket_keys.append(subject_tokens[0])
        else:
            bucket_keys.append("__empty__")

        duplicate_found = False
        checked_ids = set()
        for bucket_key in bucket_keys:
            for existing in buckets.get(bucket_key, []):
                existing_id = id(existing)
                if existing_id in checked_ids:
                    continue
                checked_ids.add(existing_id)

                if time_scopes_meaningfully_differ(f, existing):
                    continue

                if subjects_look_duplicate(subject_raw, existing.get("subject", "")) and facts_look_duplicate(
                    fact_raw,
                    existing.get("fact", ""),
                ):
                    duplicate_found = True
                    break
            if duplicate_found:
                break

        if duplicate_found:
            continue

        seen_exact.add(exact_key)
        unique.append(f)
        for bucket_key in bucket_keys:
            buckets.setdefault(bucket_key, []).append(f)

    return unique


KNOWLEDGE_LINK_SYSTEM = """Ты — помощник по привязке нового факта к уже накопленной базе знаний.
Твоя задача: решить, относится ли новый факт к уже известной сущности, и не является ли он дублем.
Отвечай СТРОГО в формате JSON. Никакого текста до или после JSON."""

KNOWLEDGE_LINK_PROMPT = """У тебя есть НОВЫЙ факт и несколько РЕЛЕВАНТНЫХ кандидатов из уже накопленной базы знаний.

НОВЫЙ ФАКТ:
- category: {category}
- subject: {subject}
- fact: {fact}
- time_scope: {time_scope}

КАНДИДАТЫ:
{candidates}

Выбери одно решение:
- "keep": оставить новый факт как есть
- "reuse_subject": факт новый, но subject нужно заменить на один из subject кандидатов
- "drop_duplicate": новый факт уже покрыт одним из кандидатов и его не нужно сохранять

Правила:
- Не придумывай информацию, которой нет в новом факте или в кандидатах.
- Если выбираешь "reuse_subject" или "drop_duplicate", используй subject РОВНО из одного из кандидатов.
- Не сливай разные сущности только из-за общей фамилии, титула или ассоциации.
- Не сливай субличность с основным персонажем, если это явно разные сущности.
- Если новый факт описывает ДРУГОЕ состояние той же сущности, другой этап жизни, изменение мира,
  последствие события или явное противоречие уже имеющемуся факту, это НЕ дубль: выбирай "keep"
  или максимум "reuse_subject", но не "drop_duplicate".
- Похожие формулировки сами по себе не означают дубль. Если различаются отрицание, время,
  статус, жизненное состояние, наличие/отсутствие свойства или фаза событий, сохраняй оба факта.
- Если уверенности нет — выбирай "keep".

Верни JSON объект:
{{
  "decision": "keep" | "reuse_subject" | "drop_duplicate",
  "subject": "<subject из нового факта или из кандидата>",
  "candidate_id": <номер кандидата или null>
}}

JSON:"""


def knowledge_candidate_score(new_item: dict, existing_item: dict) -> float:
    """Оценивает релевантность существующего факта для линковки нового факта."""
    score = 0.0

    new_category = strip_text(new_item.get("category", ""))
    existing_category = strip_text(existing_item.get("category", ""))
    new_subject = strip_text(new_item.get("subject", ""))
    existing_subject = strip_text(existing_item.get("subject", ""))
    new_fact = strip_text(new_item.get("fact", ""))
    existing_fact = strip_text(existing_item.get("fact", ""))
    new_time_scope = normalize_time_scope(
        new_item.get("time_scope", ""),
        fact=new_fact,
        category=new_category,
    )
    existing_time_scope = normalize_time_scope(
        existing_item.get("time_scope", ""),
        fact=existing_fact,
        category=existing_category,
    )

    if not existing_subject or not existing_fact:
        return score

    if new_category == existing_category:
        score += 4.0
    elif {new_category, existing_category} <= {"history", "event"}:
        score += 1.5

    new_subject_norm = normalize_subject_for_dedup(new_subject)
    existing_subject_norm = normalize_subject_for_dedup(existing_subject)
    if new_subject_norm and new_subject_norm == existing_subject_norm:
        score += 10.0
    elif subjects_look_duplicate(new_subject, existing_subject):
        score += 7.0

    new_character = lookup_canonical_character(new_subject)
    existing_character = lookup_canonical_character(existing_subject)
    if new_character and existing_character:
        if new_character == existing_character:
            score += 10.0
        else:
            score -= 8.0

    if new_category == "place" and existing_category == "place":
        new_place = canonicalize_place_subject(new_subject)
        existing_place = canonicalize_place_subject(existing_subject)
        if new_place and existing_place and new_place == existing_place:
            score += 8.0

    new_subject_tokens = subject_signature_tokens(new_subject)
    existing_subject_tokens = subject_signature_tokens(existing_subject)
    if new_subject_tokens and existing_subject_tokens:
        overlap = new_subject_tokens & existing_subject_tokens
        if overlap:
            score += 4.0 * (len(overlap) / max(min(len(new_subject_tokens), len(existing_subject_tokens)), 1))

    new_fact_tokens = content_tokens_for_fact(new_fact)
    existing_fact_tokens = content_tokens_for_fact(existing_fact)
    if new_fact_tokens and existing_fact_tokens:
        overlap = new_fact_tokens & existing_fact_tokens
        if overlap:
            score += 5.0 * (len(overlap) / max(min(len(new_fact_tokens), len(existing_fact_tokens)), 1))

    if (
        new_time_scope not in {"unclear", "timeless"}
        and existing_time_scope not in {"unclear", "timeless"}
        and new_time_scope != existing_time_scope
    ):
        score -= 6.0

    if facts_describe_different_states(new_fact, existing_fact):
        score -= 12.0
    elif facts_look_duplicate(new_fact, existing_fact):
        score += 10.0

    new_clause = leading_fact_clause(new_fact)
    existing_clause = leading_fact_clause(existing_fact)
    if (
        new_clause and existing_clause and new_clause == existing_clause
        and not facts_describe_different_states(new_fact, existing_fact)
    ):
        score += 5.0

    return score


def retrieve_relevant_knowledge_candidates(
    item: dict,
    knowledge_base: list[dict],
    *,
    top_k: int = 8,
    min_score: float = 6.0,
) -> list[dict]:
    """Подбирает короткий список релевантных фактов из уже накопленной базы знаний."""
    scored = []
    for idx, existing in enumerate(knowledge_base, 1):
        score = knowledge_candidate_score(item, existing)
        if score >= min_score:
            scored.append((score, idx, existing))

    scored.sort(
        key=lambda entry: (
            -entry[0],
            -len(dedup_word_tokens(strip_text(entry[2].get("subject", "")))),
            strip_text(entry[2].get("subject", "")),
        )
    )

    candidates = []
    for score, idx, existing in scored[:max(top_k, 1)]:
        candidate = dict(existing)
        candidate["_kb_index"] = idx
        candidate["_score"] = score
        candidates.append(candidate)
    return candidates


def format_knowledge_link_candidates(candidates: list[dict]) -> str:
    """Форматирует retrieved-кандидатов для узкого prompt по линковке."""
    lines = []
    for candidate in candidates:
        time_scope = normalize_time_scope(
            candidate.get("time_scope", ""),
            fact=strip_text(candidate.get("fact", "")),
            category=strip_text(candidate.get("category", "")),
        )
        lines.append(
            f"[{candidate.get('_kb_index')}] "
            f"category={strip_text(candidate.get('category', ''))}; "
            f"subject={preview_text(strip_text(candidate.get('subject', '')), 80)}; "
            f"time_scope={time_scope}; "
            f"fact={preview_text(strip_text(candidate.get('fact', '')), 180)}"
        )
    return "\n".join(lines) if lines else "(нет кандидатов)"


def resolve_knowledge_item_with_kb(
    client: OpenAI,
    config: Config,
    item: dict,
    knowledge_base: list[dict],
    *,
    log_prefix: str = "",
) -> Optional[dict]:
    """Пытается привязать новый факт к уже известной базе знаний по retrieved-кандидатам."""
    if not config.knowledge_linking_enabled or not knowledge_base:
        return item

    candidates = retrieve_relevant_knowledge_candidates(
        item,
        knowledge_base,
        top_k=config.knowledge_link_top_k,
        min_score=config.knowledge_link_min_score,
    )
    if not candidates:
        return item

    allowed_subjects = {strip_text(item.get("subject", ""))}
    candidate_by_id: dict[int, dict] = {}
    for candidate in candidates:
        candidate_id = candidate.get("_kb_index")
        if isinstance(candidate_id, int):
            candidate_by_id[candidate_id] = candidate
        allowed_subjects.add(strip_text(candidate.get("subject", "")))

    response = call_llm(
        client,
        config,
        KNOWLEDGE_LINK_SYSTEM,
        KNOWLEDGE_LINK_PROMPT.format(
            category=strip_text(item.get("category", "")),
            subject=strip_text(item.get("subject", "")),
            fact=strip_text(item.get("fact", "")),
            time_scope=normalize_time_scope(
                item.get("time_scope", ""),
                fact=strip_text(item.get("fact", "")),
                category=strip_text(item.get("category", "")),
            ),
            candidates=format_knowledge_link_candidates(candidates),
        ),
        max_tokens=config.max_tokens_knowledge_link,
        response_format="json" if _use_ollama_native else None,
        log_prefix=log_prefix,
    )
    if response is None:
        return item

    data, _ = parse_json_response(response, expect="object", log_prefix=log_prefix)
    if not isinstance(data, dict):
        return item

    decision = strip_text(data.get("decision", "")).lower()
    subject = strip_text(data.get("subject", ""))
    candidate_id = data.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id.isdigit():
        candidate_id = int(candidate_id)
    if not isinstance(candidate_id, int):
        candidate_id = None

    if decision == "drop_duplicate":
        if candidate_id is None or candidate_id in candidate_by_id:
            return None
        return item

    if decision == "reuse_subject":
        if subject and subject in allowed_subjects and subject != strip_text(item.get("subject", "")):
            linked = dict(item)
            linked["subject"] = subject
            return linked
        if candidate_id is not None and candidate_id in candidate_by_id:
            linked = dict(item)
            linked["subject"] = strip_text(candidate_by_id[candidate_id].get("subject", ""))
            return linked
        return item

    return item


def link_knowledge_items_with_retrieval(
    client: OpenAI,
    config: Config,
    items: list[dict],
    knowledge_base: list[dict],
    *,
    log_prefix: str = "",
) -> list[dict]:
    """Привязывает новые факты к уже накопленной базе знаний по retrieved-кандидатам."""
    if not items or not config.knowledge_linking_enabled or not knowledge_base:
        return items

    linked_items = []
    working_base = list(knowledge_base)
    reused = 0
    dropped = 0

    for idx, item in enumerate(items, 1):
        linked = resolve_knowledge_item_with_kb(
            client,
            config,
            item,
            working_base,
            log_prefix=f"{log_prefix}[link {idx}]" if log_prefix else "",
        )
        if linked is None:
            dropped += 1
            continue
        if strip_text(linked.get("subject", "")) != strip_text(item.get("subject", "")):
            reused += 1
        linked_items.append(linked)
        working_base.append(linked)

    if log_prefix and (reused or dropped):
        details = []
        if reused:
            details.append(f"reuse_subject={reused}")
        if dropped:
            details.append(f"drop_duplicate={dropped}")
        log_event(f"{log_prefix} retrieval-linking: {', '.join(details)}")

    return linked_items


# ──────────────────────────────────────────────
# Комбинированное извлечение (голос + знания за один вызов)
# ──────────────────────────────────────────────

COMBINED_SYSTEM = """Ты — помощник для разметки литературного текста Макса Фрая.
Извлекаешь одновременно реплики персонажа И факты о мире.
Особенности стиля Фрая:
— Повествование от первого лица (рассказчик = Макс).
— Прямая речь через тире (—). Безмолвная речь в кавычках.
— Внутренний монолог вплетён в нарратив.
Отвечай СТРОГО в формате JSON. Никакого текста до или после JSON."""

COMBINED_PROMPT = """Из фрагмента книги Макса Фрая извлеки ДВА ТИПА данных.

ТИП 1 — РЕПЛИКИ МАКСА (массив "dialogues"):
Извлекай прямую речь, Безмолвную речь и яркие внутренние монологи.
Для каждого:
- "type": "dialogue" / "silent_speech" / "monologue"
- "context": краткое описание ситуации (1-2 предложения)
- "interlocutor": имя собеседника или null
- "interlocutor_says": реплика собеседника перед ответом Макса или ""
- "max_says": ТОЧНАЯ цитата из текста

ТИП 2 — ФАКТЫ О МИРЕ (массив "knowledge"):
Извлекай значимые факты о персонажах, местах, магии, истории, существах, обычаях.
Для каждого:
- "category": "character"/"place"/"magic"/"history"/"event"/"creature"/"custom"
- "subject": о ком/чём. Используй конкретный и самодостаточный subject из текста.
- "fact": сам факт (1-3 предложения)
- "time_scope": "past"/"current"/"change"/"ended"/"timeless"/"unclear"

ПРАВИЛА:
— Извлекай ТОЧНЫЕ цитаты в max_says, не пересказывай.
— Не придумывай.
— Извлекай полно, но только самостоятельные факты и важные локальные события сцены.
— Если текст описывает изменение состояния, статуса или устройства мира, сохраняй разные фазы
  отдельно: "раньше", "теперь", "после", "больше не". Не сглаживай противоречащие друг другу факты.
— Не используй placeholders вроде "место", "место действия", "улица", "кабинет",
  "персонаж", "человек", если в тексте есть более точное обозначение.
— Для мест используй точные формы вроде "улица Желтых Камней", "Дом у Моста",
  "кабинет Короля", "квартира на улице Желтых Камней", а не общие слова.
— Не используй как place предметы и части сцены вроде "дверь", "окно", "лестница",
  а также сцепленные описания вроде "дом и сад", если это не собственное название места.
— Для `custom` извлекай только устойчивые элементы мира и быта: именованные напитки, еду,
  одежду, транспорт, институты, профессии, ритуалы и повторяющиеся социальные практики.
— Для `custom` subject должен быть названием самого обычая/предмета/практики:
  например "камра", "лоохи", "амобилер", "Королевский голос", а не участниками сцены.
— Не используй `custom` для разовых сценических деталей, анонимных групп и безымянных описательных
  предметов вроде "ведьмочки" или "полумесяц из плотной ткани с карманами". Если это важно только
  как эпизод сцены, отнеси к `event`; если устойчивого названия нет, пропусти.
— Если нечего извлекать — пустые массивы.
— Верни не более 8 элементов в `dialogues`.
— Для `knowledge` верни столько полезных фактов, сколько реально помещается в ответ.

Верни JSON: {{"dialogues": [...], "knowledge": [...]}}
НЕ ПИШИ НИЧЕГО, КРОМЕ JSON. НИКАКИХ ОБЪЯСНЕНИЙ ИЛИ ВВОДНЫХ СЛОВ.
НЕ используй ```json или другие markdown-обёртки — только чистый JSON.
Фрагмент:
---
{chunk}
---

JSON:"""


def extract_combined(
    client: OpenAI,
    config: Config,
    chunk: str,
    log_prefix: str = "",
    chunk_payload: Optional[str] = None,
) -> tuple:
    """Извлекает диалоги и знания полными отдельными проходами с общим контекстом."""
    dialogues = extract_dialogues(
        client,
        config,
        chunk,
        log_prefix=f"{log_prefix}[voice]" if log_prefix else "",
        chunk_payload=chunk_payload,
    )
    knowledge = extract_knowledge(
        client,
        config,
        chunk,
        log_prefix=f"{log_prefix}[knowledge]" if log_prefix else "",
        chunk_payload=chunk_payload,
    )
    if log_prefix:
        log_event(
            f"{log_prefix} aggregated: dialogues={len(dialogues)}, knowledge={len(knowledge)}"
        )
    return dialogues, knowledge


# ──────────────────────────────────────────────
# Параллельная обработка чанков
# ──────────────────────────────────────────────

def process_chunks_parallel(
    client: OpenAI,
    config: Config,
    chunks: list,
    do_voice: bool,
    do_knowledge: bool,
    workers: int = 2,
    voice_extractor: str = "regex",
    on_chunk_completed: Optional[Any] = None,
    return_results: bool = True,
    total_chunks: Optional[int] = None,
    already_completed: int = 0,
    all_chunks: Optional[list[str]] = None,
) -> tuple:
    """Обрабатывает чанки параллельно. Возвращает (all_dialogues, all_knowledge)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_dialogues = [] if return_results else None
    all_knowledge = [] if return_results else None
    rejected_total = 0
    if chunks and isinstance(chunks[0], tuple):
        chunk_items = list(chunks)
    else:
        chunk_items = list(enumerate(chunks))

    total = total_chunks if total_chunks is not None else len(chunk_items)

    def process_one(idx_chunk):
        idx, chunk = idx_chunk
        tag = format_chunk_tag(idx, total)
        item_t0 = time.time()
        if stop_requested():
            return idx, chunk, [], [], {
                "tag": tag,
                "elapsed": 0.0,
                "regex_stats": {"speech": 0, "silent": 0, "monologue": 0},
                "used_regex_fallback": False,
            }
        log_event(f"{tag} старт: {preview_text(chunk, 90)}")
        chunk_payload, context_meta = build_extraction_chunk_payload(
            all_chunks if all_chunks is not None else [chunk],
            idx if all_chunks is not None else 0,
            config,
        )
        regex_source_chunk = chunk
        dialogue_support_text = ""
        if all_chunks is not None and config.extraction_neighbor_chunks > 0:
            regex_source_chunk, dialogue_support_text = build_neighbor_text_window(
                all_chunks,
                idx,
                config.extraction_neighbor_chunks,
            )

        regex_dialogues = []
        regex_stats = {"speech": 0, "silent": 0, "monologue": 0}
        used_regex_fallback = False

        if do_voice:
            regex_dialogues, regex_stats = extract_voice_with_regex(
                regex_source_chunk,
                log_prefix=f"{tag}[voice]",
            )

        if do_voice and do_knowledge:
            if voice_extractor == "llm":
                d, k = extract_combined(
                    client,
                    config,
                    chunk,
                    log_prefix=f"{tag}[combined]",
                    chunk_payload=chunk_payload,
                )
                if not d and regex_dialogues:
                    used_regex_fallback = True
                    log_event(
                        f"{tag}[voice] подозрительно пусто после LLM, "
                        f"беру regex-фоллбек ({len(regex_dialogues)} элементов)"
                    )
                    d = regex_dialogues
            else:
                d = regex_dialogues
                k = extract_knowledge(
                    client,
                    config,
                    chunk,
                    log_prefix=f"{tag}[knowledge]",
                    chunk_payload=chunk_payload,
                )
        elif do_voice:
            if voice_extractor == "llm":
                d = extract_dialogues(
                    client,
                    config,
                    chunk,
                    log_prefix=f"{tag}[voice]",
                    chunk_payload=chunk_payload,
                )
                if not d and regex_dialogues:
                    used_regex_fallback = True
                    log_event(
                        f"{tag}[voice] подозрительно пусто после LLM, "
                        f"беру regex-фоллбек ({len(regex_dialogues)} элементов)"
                    )
                    d = regex_dialogues
            else:
                d = regex_dialogues
            k = []
        else:
            d = []
            k = extract_knowledge(
                client,
                config,
                chunk,
                log_prefix=f"{tag}[knowledge]",
                chunk_payload=chunk_payload,
            )

        meta = {
            "tag": tag,
            "elapsed": time.time() - item_t0,
            "regex_stats": regex_stats,
            "used_regex_fallback": used_regex_fallback,
            "context_meta": context_meta,
            "dialogue_support_text": dialogue_support_text,
        }
        return idx, chunk, d, k, meta

    t_start = time.time()
    completed = already_completed
    completed_this_run = 0

    if not chunk_items:
        print(f"\n  [{now_str()}] Новых фрагментов для обработки нет ({completed}/{total})")
        return all_dialogues or [], all_knowledge or []
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        futures = {
            executor.submit(process_one, idx_chunk): idx_chunk[0]
            for idx_chunk in chunk_items
        }

        for future in as_completed(futures):
            if stop_requested():
                raise GracefulInterrupt("Остановка запрошена пользователем")

            idx = futures[future]
            tag = format_chunk_tag(idx, total)
            try:
                idx, chunk, dialogues, knowledge, meta = future.result()
            except Exception as exc:
                log_event(f"{tag} ошибка воркера: {exc}")
                raise

            completed += 1
            completed_this_run += 1

            if dialogues:
                before = len(dialogues)
                dialogues = validate_dialogues(
                    dialogues,
                    source_chunk=chunk,
                    support_chunk=meta.get("dialogue_support_text", ""),
                    log_prefix=f"{tag}[voice]",
                )
                rejected_total += (before - len(dialogues))
            if knowledge:
                knowledge = validate_knowledge(
                    knowledge,
                    log_prefix=f"{tag}[knowledge]",
                )

            elapsed = time.time() - t_start
            per_item = elapsed / max(completed_this_run, 1)
            eta = per_item * (total - completed)

            progress = {
                "completed": completed,
                "total": total,
                "elapsed": elapsed,
                "eta_book": eta,
            }

            extra_progress = ""
            if on_chunk_completed is not None:
                callback_result = on_chunk_completed(
                    idx=idx,
                    chunk=chunk,
                    dialogues=dialogues,
                    knowledge=knowledge,
                    meta=meta,
                    progress=progress,
                )
                if callback_result:
                    extra_progress = str(callback_result)

            if return_results:
                all_dialogues.extend(dialogues)
                all_knowledge.extend(knowledge)

            sample = ""
            if dialogues:
                sample = f"«{dialogues[0].get('max_says', '')[:45]}...»"
            elif knowledge:
                sample = f"▸ {knowledge[0].get('subject', '')[:25]}"

            fallback_note = " regex-fallback" if meta["used_regex_fallback"] else ""
            context_meta = meta.get("context_meta", {})
            context_note = ""
            if context_meta.get("support_chunks", 0):
                context_note = (
                    f", support={context_meta['support_chunks']}ч/"
                    f"~{context_meta.get('support_tokens', 0)}т"
                )
            log_event(
                f"{tag} готово [{completed}/{total}]: +{len(dialogues)}d +{len(knowledge)}k "
                f"за {meta['elapsed']:.1f}s, ETA книги {fmt_duration(eta)}"
                f"{extra_progress}{context_note}{fallback_note} {sample}".rstrip()
            )
    except KeyboardInterrupt as exc:
        request_stop()
        print(
            f"\n  [{now_str()}] Остановка по Ctrl+C: отменяю ожидающие фрагменты. "
            "Уже сохранённый прогресс можно продолжить повторным запуском."
        )
        for future in futures:
            future.cancel()
        raise GracefulInterrupt("Обработка фрагментов прервана пользователем") from exc
    except GracefulInterrupt:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=not stop_requested(), cancel_futures=stop_requested())

    dur = time.time() - t_start
    print(f"\n  [{now_str()}] Обработано за {fmt_duration(dur)} ({dur/max(total,1):.1f}s/чанк, {workers} воркеров)")
    if rejected_total > 0:
        print(f"  Отброшено псевдоцитат: {rejected_total}")

    return all_dialogues or [], all_knowledge or []


# ──────────────────────────────────────────────
# Проход 3: Синтетическая генерация
# ──────────────────────────────────────────────

SYNTH_SYSTEM = """Ты — помощник для генерации обучающих данных.
Генерируешь вопрос и ответ в стиле конкретного персонажа.
Отвечай СТРОГО в формате JSON. Никакого текста до или после JSON."""

SYNTH_PROMPT = """Задача: сгенерировать обучающую пару (вопрос → ответ) для fine-tune,
чтобы модель говорила как Сэр Макс из книг Макса Фрая.

ФАКТ О МИРЕ ЕХО, который Макс должен знать:
Категория: {category}
Тема: {subject}
Факт: {fact}

ПРИМЕРЫ МАНЕРЫ РЕЧИ МАКСА (копируй стиль):
{style_examples}

Сгенерируй JSON объект:
- "user": естественный вопрос собеседника к Максу об этом.
  Может быть от нового знакомого, гостя из другого мира, коллеги,
  или просто повод для Макса порассуждать. Коротко и естественно.
- "assistant": ответ Макса. Содержит информацию из факта,
  но изложенную В СТИЛЕ МАКСА: с иронией, личным отношением,
  отступлениями. НЕ пересказывай факт сухо —
  Макс живой человек, а не энциклопедия. 2-5 предложений.

JSON:"""


def collect_style_candidates(voice_pairs: list[dict]) -> list[str]:
    """Собирает пул реплик Макса для few-shot без повторной фильтрации на каждый факт."""
    if not voice_pairs:
        return []

    candidates = [
        p["messages"][-1]["content"]
        for p in voice_pairs
        if 30 < len(p["messages"][-1]["content"]) < 300
    ]
    if candidates:
        return candidates
    return [p["messages"][-1]["content"] for p in voice_pairs]


def pick_style_examples(
    style_candidates: list[str],
    n: int = 3,
    rng: Optional[random.Random] = None,
) -> str:
    """Выбирает несколько примеров речи Макса для few-shot."""
    if not style_candidates:
        return "(нет примеров)"

    rng = rng or random
    selected = rng.sample(style_candidates, min(n, len(style_candidates)))
    return "\n---\n".join(f"«{s}»" for s in selected)


def generate_synth_pair(
    client: OpenAI,
    config: Config,
    fact: dict,
    style_candidates: list[str],
    rng_seed: Optional[int] = None,
) -> Optional[dict]:
    """Генерирует одну синтетическую обучающую пару из факта."""
    rng = random.Random(rng_seed) if rng_seed is not None else random
    style_examples = pick_style_examples(style_candidates, rng=rng)

    prompt = SYNTH_PROMPT.format(
        category=fact.get("category", ""),
        subject=fact.get("subject", ""),
        fact=fact.get("fact", ""),
        style_examples=style_examples,
    )

    response = call_llm(
        client,
        config,
        SYNTH_SYSTEM,
        prompt,
        max_tokens=config.max_tokens_synth,
        temperature=0.7,
    )

    if response is None:
        return None

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)
        if isinstance(data, dict) and "user" in data and "assistant" in data:
            user_text = data["user"].strip()
            assistant_text = data["assistant"].strip()
            if len(assistant_text) > 20:
                return {
                    "messages": [
                        {"role": "system", "content": config.character_system_prompt},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                }
    except json.JSONDecodeError:
        pass

    return None


def generate_synth_pairs(
    client: OpenAI,
    config: Config,
    knowledge: list[dict],
    voice_pairs: list[dict],
    max_pairs: int = 500,
    seed: int = 42,
    progress_path: Optional[Path] = None,
    synth_output_path: Optional[Path] = None,
    readable_output_path: Optional[Path] = None,
    workers: int = 1,
    progress_callback: Optional[Any] = None,
) -> list[dict]:
    """Генерирует синтетические обучающие пары из базы знаний с resume."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    facts = order_facts_for_synth(knowledge, seed)
    target_facts = facts[:min(max_pairs, len(facts))]
    style_candidates = collect_style_candidates(voice_pairs)

    processed_fact_hashes: set[str] = set()
    pairs: list[dict] = []
    if progress_path is not None:
        processed_fact_hashes, pairs = load_synth_progress(progress_path, log_prefix="[synth]")
        if synth_output_path is not None:
            if synth_output_path.exists():
                synth_output_path.unlink()
            append_jsonl(synth_output_path, pairs)
        if readable_output_path is not None and pairs:
            save_readable_voice(pairs, str(readable_output_path))

    produced_pairs = len(pairs)
    if progress_callback is not None:
        progress_callback(
            "synth_started",
            total_target_facts=len(target_facts),
            processed_facts=len(processed_fact_hashes),
            produced_pairs=produced_pairs,
        )
    pending_facts = []
    for i, fact in enumerate(target_facts, 1):
        fact_hash = stable_fact_hash(fact)
        if fact_hash in processed_fact_hashes:
            print(f"    [{i}/{len(target_facts)}] resume: {produced_pairs}", end="\r")
            continue
        pending_facts.append((i, fact, fact_hash))

    if not pending_facts:
        print()
        return pairs

    def synthesize_one(item: tuple[int, dict, str]) -> tuple[int, str, Optional[dict]]:
        idx, fact, fact_hash = item
        fact_seed = seed ^ int(hashlib.md5(fact_hash.encode()).hexdigest()[:8], 16)
        pair = generate_synth_pair(
            client,
            config,
            fact,
            style_candidates,
            rng_seed=fact_seed,
        )
        return idx, fact_hash, pair

    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = {}
    try:
        futures = {
            executor.submit(synthesize_one, item): item[0]
            for item in pending_facts
        }

        for future in as_completed(futures):
            if stop_requested():
                raise GracefulInterrupt("Синтетическая генерация остановлена пользователем")

            idx = futures[future]
            idx, fact_hash, pair = future.result()
            record = {"idx": idx, "fact_hash": fact_hash, "pair": pair}
            if progress_path is not None:
                append_jsonl(progress_path, [record])
            processed_fact_hashes.add(fact_hash)
            if pair:
                pairs.append(pair)
                produced_pairs += 1
                if synth_output_path is not None:
                    append_jsonl(synth_output_path, [pair])
                if readable_output_path is not None:
                    save_readable_voice(pairs, str(readable_output_path))
            if progress_callback is not None:
                progress_callback(
                    "synth_progress",
                    processed_facts=len(processed_fact_hashes),
                    total_target_facts=len(target_facts),
                    produced_pairs=produced_pairs,
                    last_fact_idx=idx,
                )
            print(f"    [{idx}/{len(target_facts)}] сгенерировано: {produced_pairs}", end="\r")
    except KeyboardInterrupt as exc:
        request_stop()
        print(
            f"\n  [{now_str()}] Синтетика остановлена по Ctrl+C. "
            "Уже сгенерированные пары сохранены."
        )
        raise GracefulInterrupt("Синтетическая генерация прервана пользователем") from exc
    except GracefulInterrupt:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=not stop_requested(), cancel_futures=stop_requested())

    print()
    if progress_callback is not None:
        progress_callback(
            "synth_complete",
            processed_facts=len(processed_fact_hashes),
            total_target_facts=len(target_facts),
            produced_pairs=produced_pairs,
        )
    return pairs


# ──────────────────────────────────────────────
# Дедупликация
# ──────────────────────────────────────────────

def deduplicate(pairs: list[dict]) -> list[dict]:
    """Убирает дубли: точные (по хешу user+assistant) и near-duplicate (по fuzzy ratio assistant)."""
    # Этап 1: точные дубли по хешу user+assistant
    seen_hashes = set()
    stage1 = []
    for pair in pairs:
        user_text = pair["messages"][1]["content"] if len(pair["messages"]) > 1 else ""
        assistant_text = pair["messages"][-1]["content"]
        h = text_hash(f"{user_text}||{assistant_text}")
        if h not in seen_hashes:
            seen_hashes.add(h)
            stage1.append(pair)

    # Этап 2: near-duplicate по assistant (fuzzy, порог 0.85)
    unique = []
    seen_texts = []  # список текстов для сравнения
    for pair in stage1:
        assistant_text = pair["messages"][-1]["content"]
        is_dup = False
        # Сравниваем только с последними N для скорости
        for existing in seen_texts[-200:]:
            if abs(len(existing) - len(assistant_text)) > len(assistant_text) * 0.3:
                continue  # быстрая отсечка по длине
            ratio = difflib.SequenceMatcher(None, existing, assistant_text).ratio()
            if ratio > 0.85:
                is_dup = True
                break
        if not is_dup:
            unique.append(pair)
            seen_texts.append(assistant_text)

    return unique


# ──────────────────────────────────────────────
# Человекочитаемый экспорт
# ──────────────────────────────────────────────

def save_readable_voice(pairs: list[dict], path: str):
    """Сохраняет обучающие пары в читаемом виде."""
    with open(path, "w", encoding="utf-8") as f:
        for i, pair in enumerate(pairs, 1):
            msgs = pair["messages"]
            # msgs[0] = system, msgs[1] = user, msgs[2] = assistant
            user = msgs[1]["content"] if len(msgs) > 1 else ""
            assistant = msgs[2]["content"] if len(msgs) > 2 else ""

            f.write(f"{'─' * 60}\n")
            f.write(f"#{i}\n\n")
            f.write(f"[Ситуация / собеседник]:\n{user}\n\n")
            f.write(f"[Макс]:\n{assistant}\n\n")

        f.write(f"{'─' * 60}\n")
        f.write(f"Всего: {len(pairs)} пар\n")


def save_readable_knowledge(facts: list[dict], path: str):
    """Сохраняет базу знаний в читаемом виде, сгруппированную по категориям."""
    # Группируем по категориям
    by_cat: dict[str, list[dict]] = {}
    for fact in facts:
        cat = fact.get("category", "unknown")
        by_cat.setdefault(cat, []).append(fact)

    cat_names = {
        "character": "Персонажи",
        "place": "Места",
        "magic": "Магия",
        "history": "История",
        "event": "События",
        "creature": "Существа",
        "custom": "Обычаи и быт",
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"БАЗА ЗНАНИЙ О МИРЕ ЕХО\n")
        f.write(f"{'═' * 60}\n")
        f.write(f"Всего фактов: {len(facts)}\n\n")

        for cat in ["character", "place", "magic", "history", "event", "creature", "custom", "unknown"]:
            items = by_cat.get(cat, [])
            if not items:
                continue

            title = cat_names.get(cat, cat)
            f.write(f"\n{'━' * 60}\n")
            f.write(f"  {title} ({len(items)})\n")
            f.write(f"{'━' * 60}\n\n")

            # Группируем по subject внутри категории
            by_subject: dict[str, list[str]] = {}
            for item in items:
                subj = item.get("subject", "—")
                fact = item.get("fact", "")
                time_scope = normalize_time_scope(
                    item.get("time_scope", ""),
                    fact=fact,
                    category=cat,
                )
                if time_scope and time_scope not in {"unclear", "timeless"}:
                    fact_line = f"[{time_scope}] {fact}"
                else:
                    fact_line = fact
                by_subject.setdefault(subj, []).append(fact_line)

            for subj in sorted(by_subject.keys()):
                f.write(f"  ▸ {subj}\n")
                for fact in by_subject[subj]:
                    f.write(f"    — {fact}\n")
                f.write(f"\n")


# ──────────────────────────────────────────────
# Основной конвейер
# ──────────────────────────────────────────────

def process_book(
    client: OpenAI,
    config: Config,
    book_name: str,
    text: str,
    skip_synth: bool = False,
    synth_count: int = 500,
    accumulated_voice: Optional[list] = None,
    accumulated_knowledge: Optional[list] = None,
    workers: int = 2,
    voice_extractor: str = "regex",
    books_total: int = 1,
    books_completed_before: int = 0,
    pipeline_t0: Optional[float] = None,
    seed: int = 42,
    progress_callback: Optional[Any] = None,
) -> dict:
    """Обрабатывает одну книгу. Возвращает dict с voice_pairs, knowledge, synth_pairs."""

    narrator, mode = detect_book_mode(book_name)
    do_voice = mode in ("full", "voice_only")
    do_knowledge = mode in ("full", "knowledge_only", "knowledge_raw")
    do_synth = mode in ("full", "knowledge_only")

    MODE_DESC = {
        "full":           f"Макс → голос + знания + синтетика",
        "voice_only":     f"Макс (другой мир) → только голос",
        "knowledge_only": f"{narrator} → знания + синтетика",
        "knowledge_raw":  f"{narrator} → только знания",
    }

    print(f"\n{'='*60}")
    print(f"[{now_str()}] Обработка: {book_name}")
    print(f"  Режим: {MODE_DESC.get(mode, mode)}")
    print(f"  Голос: {'regex' if voice_extractor == 'regex' else 'LLM'}")
    print(f"{'='*60}")
    if progress_callback is not None:
        progress_callback(
            "book_started",
            book_name=book_name,
            narrator=narrator,
            mode=mode,
            voice_extractor=voice_extractor,
        )

    book_t0 = time.time()

    # Нарезка
    chunks = split_into_chunks(
        text,
        config.chunk_size,
        config.chunk_overlap,
        client=client,
        config=config,
        log_prefix=f"[chunking {get_book_stem(book_name)}]",
    )
    chunk_chapter_map = build_chunk_chapter_map(chunks, book_name)
    print(f"  Нарезано на {len(chunks)} фрагментов")
    if progress_callback is not None:
        progress_callback(
            "book_chunked",
            book_name=book_name,
            total_chunks=len(chunks),
        )

    # Пути промежуточных файлов
    output_paths = get_book_output_paths(config.output_dir, book_name)
    voice_path = output_paths["voice"]
    knowledge_path = output_paths["knowledge"]
    knowledge_stream_path = output_paths["knowledge_stream"]
    chunk_results_path = output_paths["chunks"]
    synth_path = output_paths["synth"]
    synth_progress_path = output_paths["synth_progress"]
    done_path = output_paths["done"]
    voice_txt_path = output_paths["voice_txt"]
    knowledge_txt_path = output_paths["knowledge_txt"]
    synth_txt_path = output_paths["synth_txt"]

    # ── Извлечение (комбинированное + параллельное) ──
    voice_pairs = []
    all_knowledge = []
    pipeline_t0 = pipeline_t0 or book_t0
    checkpoint_records = {}
    resume_prefix = f"[resume {get_book_stem(book_name)}]"

    if chunk_results_path.exists() and not done_path.exists():
        checkpoint_records = load_chunk_checkpoint(chunk_results_path, chunks, log_prefix=resume_prefix)
        if checkpoint_records:
            print(f"  Возобновление: найдено {len(checkpoint_records)}/{len(chunks)} обработанных фрагментов")
            for idx, record in checkpoint_records.items():
                record["chapter"] = record.get("chapter") or chunk_chapter_map[idx]
                record["chunk_text"] = record.get("chunk_text") or chunks[idx]
                record["knowledge"] = attach_knowledge_source_fields(
                    record.get("knowledge", []),
                    book_name=book_name,
                    chapter=record["chapter"],
                    chunk_idx=idx,
                )
            voice_pairs, all_knowledge = rebuild_chunk_outputs(
                checkpoint_records,
                config,
                voice_path,
                knowledge_stream_path,
            )
            all_knowledge = canonicalize_book_knowledge(
                all_knowledge,
                narrator,
                log_prefix=f"{resume_prefix}[knowledge]",
            )
            if knowledge_stream_path.exists():
                knowledge_stream_path.unlink()
            if all_knowledge:
                append_jsonl(knowledge_stream_path, all_knowledge)
            if voice_pairs:
                save_readable_voice(voice_pairs, str(voice_txt_path))
            if all_knowledge:
                save_readable_knowledge(
                    deduplicate_knowledge(all_knowledge),
                    str(knowledge_txt_path),
                )
            if progress_callback is not None:
                progress_callback(
                    "book_resumed",
                    book_name=book_name,
                    resumed_chunks=len(checkpoint_records),
                    total_chunks=len(chunks),
                    book_voice_pairs=len(voice_pairs),
                    book_knowledge_facts=len(deduplicate_knowledge(all_knowledge)),
                )
        else:
            log_event(f"{resume_prefix} checkpoint пустой или устарел, начинаю книгу заново")

    if not checkpoint_records:
        for path in (
            voice_path,
            knowledge_stream_path,
            chunk_results_path,
            synth_path,
            synth_progress_path,
            done_path,
            voice_txt_path,
            knowledge_txt_path,
            synth_txt_path,
        ):
            if path.exists():
                path.unlink()
        if knowledge_path.exists():
            knowledge_path.unlink()
        if do_voice:
            voice_path.touch()
        if do_knowledge:
            knowledge_stream_path.touch()

    def on_chunk_completed(**kwargs):
        nonlocal voice_pairs, all_knowledge

        idx = kwargs["idx"]
        chunk = kwargs["chunk"]
        dialogues = kwargs["dialogues"]
        knowledge = kwargs["knowledge"]
        progress = kwargs["progress"]
        tag = kwargs["meta"]["tag"]
        chapter = chunk_chapter_map[idx] if idx < len(chunk_chapter_map) else default_chapter_label(book_name)

        if knowledge:
            knowledge = normalize_knowledge_items(
                knowledge,
                narrator,
                log_prefix=f"{tag}[knowledge]",
            )
            knowledge = attach_knowledge_source_fields(
                knowledge,
                book_name=book_name,
                chapter=chapter,
                chunk_idx=idx,
            )
            knowledge = link_knowledge_items_with_retrieval(
                client,
                config,
                knowledge,
                (accumulated_knowledge or []) + all_knowledge,
                log_prefix=f"{tag}[knowledge]",
            )

        append_jsonl(chunk_results_path, [{
            "idx": idx,
            "chunk_hash": text_hash(chunk),
            "chapter": chapter,
            "chunk_text": chunk,
            "dialogues": dialogues,
            "knowledge": knowledge,
        }])

        if dialogues:
            new_pairs = make_training_pairs(dialogues, config)
            if new_pairs:
                append_jsonl(voice_path, new_pairs)
                voice_pairs.extend(new_pairs)
                save_readable_voice(voice_pairs, str(voice_txt_path))

        if knowledge:
            all_knowledge.extend(knowledge)
            all_knowledge = canonicalize_book_knowledge(
                all_knowledge,
                narrator,
                log_prefix=f"{tag}[knowledge]",
            )
            if knowledge_stream_path.exists():
                knowledge_stream_path.unlink()
            append_jsonl(knowledge_stream_path, all_knowledge)
            save_readable_knowledge(
                deduplicate_knowledge(all_knowledge),
                str(knowledge_txt_path),
            )
        if progress_callback is not None:
            progress_callback(
                "chunk_complete",
                book_name=book_name,
                chunk_idx=idx,
                total_chunks=len(chunks),
                completed_chunks=progress.get("completed", 0),
                book_voice_pairs=len(voice_pairs),
                book_knowledge_facts=len(deduplicate_knowledge(all_knowledge)),
            )

        completed_fraction = books_completed_before + (progress["completed"] / max(progress["total"], 1))
        total_elapsed = time.time() - pipeline_t0
        if completed_fraction <= 0:
            return ""

        per_book = total_elapsed / completed_fraction
        eta_pipeline = max(per_book * (books_total - completed_fraction), 0.0)
        return f", ETA всех книг {fmt_duration(eta_pipeline)}"

    pending_chunk_items = [
        (idx, chunk)
        for idx, chunk in enumerate(chunks)
        if idx not in checkpoint_records
    ]
    already_completed_chunks = len(checkpoint_records)

    if do_voice and do_knowledge:
        if voice_extractor == "regex":
            print(f"\n  [{now_str()}] ── Гибридное извлечение (голос=regex, знания=LLM) ──")
        else:
            print(f"\n  [{now_str()}] ── Комбинированное извлечение (голос + знания) ──")
        print(f"  Обработка {len(chunks)} фрагментов ({workers} воркеров)...")

        if pending_chunk_items:
            process_chunks_parallel(
                client, config, pending_chunk_items,
                do_voice=True,
                do_knowledge=True,
                workers=workers,
                voice_extractor=voice_extractor,
                on_chunk_completed=on_chunk_completed,
                return_results=False,
                total_chunks=len(chunks),
                already_completed=already_completed_chunks,
                all_chunks=chunks,
            )
        else:
            print(f"  Все фрагменты этой книги уже извлечены, продолжаю с сохранённого места")
    else:
        if do_voice:
            if voice_extractor == "regex":
                print(f"\n  [{now_str()}] ── Извлечение голоса (regex) ──")
            else:
                print(f"\n  [{now_str()}] ── Извлечение голоса (LLM) ──")
            print(f"  Обработка {len(chunks)} фрагментов ({workers} воркеров)...")
            if pending_chunk_items:
                process_chunks_parallel(
                    client, config, pending_chunk_items,
                    do_voice=True,
                    do_knowledge=False,
                    workers=workers,
                    voice_extractor=voice_extractor,
                    on_chunk_completed=on_chunk_completed,
                    return_results=False,
                    total_chunks=len(chunks),
                    already_completed=already_completed_chunks,
                    all_chunks=chunks,
                )
            else:
                print(f"  Все фрагменты этой книги уже извлечены, продолжаю с сохранённого места")

        if do_knowledge:
            print(f"\n  [{now_str()}] ── Извлечение знаний ──")
            print(f"  Обработка {len(chunks)} фрагментов ({workers} воркеров)...")
            if pending_chunk_items:
                process_chunks_parallel(
                    client, config, pending_chunk_items,
                    do_voice=False,
                    do_knowledge=True,
                    workers=workers,
                    voice_extractor=voice_extractor,
                    on_chunk_completed=on_chunk_completed,
                    return_results=False,
                    total_chunks=len(chunks),
                    already_completed=already_completed_chunks,
                    all_chunks=chunks,
                )
            else:
                print(f"  Все фрагменты этой книги уже извлечены, продолжаю с сохранённого места")

    if not do_voice:
        print(f"\n  Извлечение голоса пропущено ({narrator}, режим {mode})")
    if not do_knowledge:
        print(f"  Извлечение знаний пропущено (режим {mode})")

    if stop_requested():
        raise GracefulInterrupt("Остановка запрошена после извлечения данных")

    # Формирование пар из диалогов
    if voice_pairs:
        print(f"  Обучающих пар (голос): {len(voice_pairs)}")

    # Дедупликация знаний
    if all_knowledge:
        all_knowledge = canonicalize_book_knowledge(
            all_knowledge,
            narrator,
            log_prefix=f"[book {get_book_stem(book_name)}][knowledge]",
        )
        all_knowledge = deduplicate_knowledge(all_knowledge)
        print(f"  Уникальных фактов: {len(all_knowledge)}")
        cat_counts = {}
        for f in all_knowledge:
            cat = f.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
    if do_knowledge:
        with open(knowledge_path, "w", encoding="utf-8") as f:
            json.dump(all_knowledge, f, ensure_ascii=False, indent=2)

    # ── Проход 3: Синтетическая генерация ──
    synth_pairs = []

    style_source = voice_pairs if voice_pairs else (accumulated_voice or [])

    can_synth = (
        not skip_synth
        and do_synth
        and all_knowledge
        and style_source
    )

    if can_synth:
        print(f"\n  [{now_str()}] ── Проход 3: Синтетическая генерация ──")
        if not voice_pairs and accumulated_voice:
            print(f"  (стиль берётся из ранее обработанных книг Макса: {len(accumulated_voice)} пар)")
        print(f"  Генерация до {synth_count} пар из {len(all_knowledge)} фактов...")
        t_start_s = time.time()
        synth_pairs = generate_synth_pairs(
            client,
            config,
            all_knowledge,
            style_source,
            max_pairs=synth_count,
            seed=seed,
            progress_path=synth_progress_path,
            synth_output_path=synth_path,
            readable_output_path=synth_txt_path,
            workers=workers,
            progress_callback=(
                (lambda event_type, **event_data: progress_callback(
                    event_type,
                    book_name=book_name,
                    **event_data,
                ))
                if progress_callback is not None else None
            ),
        )
        pass3_dur = time.time() - t_start_s
        print(f"  [{now_str()}] Синтетических пар: {len(synth_pairs)} за {fmt_duration(pass3_dur)}")
    elif skip_synth:
        print(f"\n  Проход 3 пропущен (--skip-synth)")
    elif not do_synth:
        print(f"\n  Проход 3 пропущен (режим {mode})")
    elif not all_knowledge:
        print(f"\n  Проход 3 пропущен (нет фактов)")
    elif not style_source:
        print(f"\n  Проход 3 пропущен (нет примеров голоса Макса)")
        print(f"  Совет: обработай сначала книги, где рассказчик — Макс")

    book_dur = time.time() - book_t0
    done_path.touch()
    print(f"\n  [{now_str()}] Книга обработана за {fmt_duration(book_dur)}")
    if progress_callback is not None:
        progress_callback(
            "book_completed",
            book_name=book_name,
            book_voice_pairs=len(voice_pairs),
            book_knowledge_facts=len(all_knowledge),
            book_synth_pairs=len(synth_pairs),
            duration_seconds=round(book_dur, 1),
        )

    return {
        "voice_pairs": voice_pairs,
        "knowledge": all_knowledge,
        "synth_pairs": synth_pairs,
    }


# ──────────────────────────────────────────────
# Управление ollama
# ──────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Проверяет, запущен ли ollama serve."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_ollama_binary() -> Optional[str]:
    """Ищет бинарник ollama в системе."""
    import shutil
    path = shutil.which("ollama")
    if path:
        return path

    # Типичные пути на WSL / Linux
    for candidate in ["/usr/local/bin/ollama", "/usr/bin/ollama",
                      os.path.expanduser("~/.local/bin/ollama")]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


def ensure_ollama(model: str) -> Optional[subprocess.Popen]:
    """
    Убеждается, что ollama запущена и модель загружена.
    Возвращает процесс ollama serve (если мы его запустили) или None.
    """
    started_process = None

    # 1. Проверяем, запущена ли ollama
    if is_ollama_running():
        print("Ollama уже запущена")
    else:
        ollama_bin = find_ollama_binary()
        if ollama_bin is None:
            print("ollama не найдена в PATH.")
            print("Установи: https://ollama.com/download")
            print("Или запусти вручную и используй --no-auto-serve")
            exit(1)

        print(f"Запускаю ollama serve...")
        # Запускаем в фоне, подавляем вывод
        started_process = subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        # Ждём готовности (до 30 секунд)
        for i in range(30):
            time.sleep(1)
            if is_ollama_running():
                print(f"  Ollama готова (заняло {i + 1} сек)")
                break
            if started_process.poll() is not None:
                print(f"  ollama serve завершился с кодом {started_process.returncode}")
                print("  Попробуй запустить вручную: ollama serve")
                exit(1)
        else:
            print("  Таймаут ожидания ollama serve (30 сек)")
            exit(1)

    # 2. Проверяем / скачиваем модель
    print(f"Проверяю модель {model}...")
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"  Модель {model} найдена")
        else:
            print(f"  Модель {model} не найдена, скачиваю...")
            pull = subprocess.run(
                ["ollama", "pull", model],
                timeout=3600,  # час на скачивание
            )
            if pull.returncode != 0:
                print(f"  Ошибка скачивания модели {model}")
                exit(1)
            print(f"  Модель {model} скачана")
    except FileNotFoundError:
        print("  Команда ollama не найдена")
        exit(1)
    except subprocess.TimeoutExpired:
        print("  Таймаут при работе с моделью")
        exit(1)

    return started_process


def stop_ollama(process: Optional[subprocess.Popen]):
    """Останавливает ollama serve, если мы его запускали."""
    if process is None:
        return
    if process.poll() is not None:
        return  # уже завершился

    print("\nОстанавливаю ollama serve...")
    try:
        # Убиваем всю группу процессов
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
        print("  Ollama остановлена")
    except Exception as e:
        print(f"  Не удалось остановить ollama: {e}")
        try:
            process.kill()
        except Exception:
            pass


# ──────────────────────────────────────────────
# Основной конвейер
# ──────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Извлечение диалогов Сэра Макса для fine-tune")
    parser.add_argument("--books-dir", default="./books", help="Папка с .fb2/.txt книгами")
    parser.add_argument("--output-dir", default="./output", help="Папка для результатов")
    parser.add_argument("--api-base", default="http://localhost:11434/v1", help="URL API модели")
    parser.add_argument("--model", default="gemma4:e4b", help="Название модели")
    parser.add_argument("--chunk-size", type=int, default=Config.chunk_size, help="Размер фрагмента в токенах")
    parser.add_argument("--no-auto-serve", action="store_true",
                        help="Не запускать ollama автоматически")
    parser.add_argument("--skip-synth", action="store_true",
                        help="Пропустить проход 3 (синтетическая генерация)")
    parser.add_argument("--synth-count", type=int, default=500,
                        help="Макс. количество синтетических пар на книгу (по умолчанию 500)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Не пропускать уже обработанные книги, начать заново")
    parser.add_argument("--workers", type=int, default=2,
                        help="Количество параллельных воркеров (по умолчанию 2)")
    parser.add_argument("--voice-extractor", choices=("regex", "llm"), default="regex",
                        help="Чем извлекать голос Макса: быстрым regex или LLM (по умолчанию regex)")
    parser.add_argument("--request-timeout", type=int, default=180,
                        help="Таймаут одного запроса к модели в секундах (по умолчанию 180)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed для воспроизводимости (по умолчанию 42)")
    args = parser.parse_args()
    _STOP_REQUESTED.clear()

    # Воспроизводимый random
    random.seed(args.seed)
    print(f"Random seed: {args.seed}")
    print(f"Voice extractor: {args.voice_extractor}")

    config = Config(
        books_dir=args.books_dir,
        output_dir=args.output_dir,
        api_base=args.api_base,
        model=args.model,
        chunk_size=args.chunk_size,
        request_timeout=args.request_timeout,
    )

    # Создаём папку для результатов
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.llm_trace_run_id = time.strftime("run_%Y%m%d_%H%M%S", time.localtime())
    global_output_paths = get_global_output_paths(config.output_dir)
    meta_path = global_output_paths["metadata"]
    metadata_history_path = global_output_paths["metadata_history"]
    if metadata_history_path.exists():
        metadata_history_path.unlink()
    metadata_state = {
        "status": "starting",
        "run_started_at": now_iso_str(),
        "updated_at": now_iso_str(),
        "seed": args.seed,
        "model": config.model,
        "books_dir": config.books_dir,
        "output_dir": config.output_dir,
        "chunk_size": config.chunk_size,
        "workers": args.workers,
        "voice_extractor": args.voice_extractor,
        "request_timeout": config.request_timeout,
        "llm_trace_enabled": config.llm_trace_enabled,
        "llm_trace_run_id": config.llm_trace_run_id,
        "extraction_passes": config.extraction_passes,
        "extraction_neighbor_chunks": config.extraction_neighbor_chunks,
        "extraction_neighbor_excerpt_tokens": config.extraction_neighbor_excerpt_tokens,
        "extraction_context_budget": config.extraction_context_budget,
        "timeline_resolution_enabled": config.timeline_resolution_enabled,
        "skip_synth": args.skip_synth,
        "synth_count": args.synth_count,
        "current_stage": "startup",
        "current_book": None,
        "current_book_progress": {},
        "books_total": 0,
        "books_pending_total": 0,
        "books_processed": 0,
        "books_skipped": 0,
        "total_pairs": 0,
        "voice_pairs": 0,
        "synth_pairs": 0,
        "knowledge_raw_facts": 0,
        "knowledge_facts": 0,
        "timeline_groups": 0,
        "timeline_nodes": 0,
        "timeline_edges": 0,
        "book_statuses": {},
        "event_count": 0,
        "recent_events": [],
    }

    def metadata_event(event_type: str = "", message: str = "", **updates):
        return update_metadata_snapshot(
            metadata_state,
            meta_path,
            history_path=metadata_history_path,
            event_type=event_type,
            message=message,
            **updates,
        )

    metadata_event(
        "run_started",
        "Запуск пайплайна",
        status="running",
        current_stage="startup",
    )

    def refresh_global_knowledge_snapshot(event_type: str, current_book: Optional[str] = None):
        raw_snapshot, unique_snapshot = write_global_knowledge_snapshot(
            config.output_dir,
            all_knowledge,
            narrator="Макс",
            log_prefix="[runtime][global_knowledge]",
        )
        metadata_event(
            event_type,
            current_stage="global_knowledge_snapshot",
            current_book=current_book,
            knowledge_raw_facts=len(raw_snapshot),
            knowledge_facts=len(unique_snapshot),
        )

    # ── Управление ollama ──
    ollama_process = None

    # Гарантируем остановку ollama при выходе
    def cleanup():
        stop_ollama(ollama_process)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    try:
        if not args.no_auto_serve:
            ollama_process = ensure_ollama(config.model)

        # Инициализация клиента
        client = OpenAI(base_url=config.api_base, api_key=config.api_key)

        # Автодетект ollama → нативный API с think=false
        global _use_ollama_native
        if ":11434" in config.api_base:
            # Проверяем что нативный API отвечает
            try:
                import urllib.request
                base = config.api_base.rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]
                req = urllib.request.Request(f"{base}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        _use_ollama_native = True
                        print("Обнаружен ollama → нативный API с think=false")
            except Exception:
                pass

        if not _use_ollama_native:
            print("Используется OpenAI-совместимый API")

        # Проверка соединения
        print("Проверка соединения с моделью...")
        metadata_event(
            "connection_check_started",
            current_stage="connection_check",
        )
        test = call_llm(client, config, "Ответь одним словом.", "Скажи: работает",
                        max_tokens=10)
        if stop_requested():
            raise GracefulInterrupt("Остановка запрошена во время проверки соединения")
        if test is None:
            print(f"Не удалось подключиться к {config.api_base}")
            print(f"Убедись, что модель запущена (например: ollama run {config.model})")
            metadata_event(
                "connection_check_failed",
                status="failed",
                current_stage="connection_check",
                run_finished_at=now_iso_str(),
            )
            return 1
        print(f"Модель отвечает: {test.strip()}\n")
        metadata_event(
            "connection_check_ok",
            current_stage="loading_books",
        )

        # Загрузка книг
        print(f"Загрузка книг из {config.books_dir}/")
        books = load_books(config.books_dir)
        if stop_requested():
            raise GracefulInterrupt("Остановка запрошена во время загрузки книг")
        if not books:
            print("Книги не найдены.")
            metadata_event(
                "books_missing",
                status="completed",
                current_stage="done",
                run_finished_at=now_iso_str(),
            )
            return 0

        # Сортировка: книги Макса первыми (чтобы накопить стиль для синтетики)
        mode_order = {"full": 0, "voice_only": 1, "knowledge_only": 2, "knowledge_raw": 3}

        def sort_key(item):
            name, _ = item
            _, mode = detect_book_mode(name)
            return (mode_order.get(mode, 9), name)

        books.sort(key=sort_key)

        print(f"\nПорядок обработки:")
        mode_icons = {"full": "🔊", "voice_only": "🗣", "knowledge_only": "📖", "knowledge_raw": "📚"}
        for name, _ in books:
            narrator, mode = detect_book_mode(name)
            icon = mode_icons.get(mode, "?")
            print(f"  {icon} {name} [{narrator}, {mode}]")

        books_pending_total = 0
        book_statuses = {}
        for book_name, _ in books:
            if args.no_resume or not is_book_processed(config.output_dir, book_name):
                books_pending_total += 1
                book_statuses[book_name] = "pending"
            else:
                book_statuses[book_name] = "completed_existing"

        metadata_event(
            "books_loaded",
            current_stage="books_loaded",
            books_total=len(books),
            books_pending_total=books_pending_total,
            book_statuses=book_statuses,
        )

        # Обработка
        all_voice_pairs = []
        all_knowledge = []
        all_synth_pairs = []
        pipeline_t0 = time.time()
        processed_books = 0

        def stage_for_book_event(event_type: str) -> str:
            if event_type.startswith("synth_"):
                return "book_synth"
            if event_type in {"book_started", "book_chunked", "book_resumed", "chunk_complete"}:
                return "book_extraction"
            if event_type == "book_completed":
                return "book_complete"
            return "book_processing"

        def on_book_progress(event_type: str, **event_data):
            current_book = event_data.get("book_name")
            progress_payload = {
                key: value
                for key, value in event_data.items()
                if key != "book_name"
            }
            metadata_event(
                event_type,
                current_stage=stage_for_book_event(event_type),
                current_book=current_book,
                current_book_progress={
                    "book_name": current_book,
                    **progress_payload,
                },
            )

        for _, (book_name, text) in enumerate(books):
            if stop_requested():
                raise GracefulInterrupt("Остановка запрошена перед следующей книгой")

            # Прогресс по книгам
            if processed_books > 0 and books_pending_total > 0:
                elapsed = time.time() - pipeline_t0
                per_book = elapsed / processed_books
                books_left = books_pending_total - processed_books
                eta_total = per_book * books_left
                print(f"\n  📊 Книг обработано: {processed_books}/{books_pending_total}, "
                      f"прошло {fmt_duration(elapsed)}, "
                      f"ETA всего пайплайна: {fmt_duration(eta_total)}")

            # Пути выходных файлов книги
            output_paths = get_book_output_paths(config.output_dir, book_name)

            # ── Проверка возобновления ──
            voice_path = output_paths["voice"]
            knowledge_path = output_paths["knowledge"]
            knowledge_stream_path = output_paths["knowledge_stream"]

            if not args.no_resume and is_book_processed(config.output_dir, book_name):
                print(f"\n  ⏭ Пропуск {book_name} (уже обработана, --no-resume для повтора)")
                narrator, _ = detect_book_mode(book_name)
                metadata_event(
                    "book_skipped_existing",
                    current_stage="book_skip",
                    current_book=book_name,
                    books_skipped=metadata_state.get("books_skipped", 0) + 1,
                    book_statuses={book_name: "completed_existing"},
                )

                if voice_path.exists():
                    with open(voice_path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                all_voice_pairs.append(json.loads(line))

                if knowledge_path.exists():
                    with open(knowledge_path, encoding="utf-8") as f:
                        loaded_knowledge = json.load(f)
                elif knowledge_stream_path.exists():
                    loaded_knowledge = []
                    with open(knowledge_stream_path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                loaded_knowledge.append(json.loads(line))
                else:
                    loaded_knowledge = []

                if loaded_knowledge:
                    loaded_knowledge = ensure_knowledge_source_defaults(loaded_knowledge, book_name)
                    loaded_knowledge = canonicalize_book_knowledge(loaded_knowledge, narrator)
                    all_knowledge.extend(loaded_knowledge)
                    normalized_book_knowledge = deduplicate_knowledge(loaded_knowledge)
                    if knowledge_stream_path.exists():
                        knowledge_stream_path.unlink()
                    append_jsonl(knowledge_stream_path, loaded_knowledge)
                    with open(knowledge_path, "w", encoding="utf-8") as f:
                        json.dump(normalized_book_knowledge, f, ensure_ascii=False, indent=2)
                    save_readable_knowledge(
                        normalized_book_knowledge,
                        str(output_paths["knowledge_txt"]),
                    )
                    refresh_global_knowledge_snapshot(
                        "global_kb_snapshot_updated",
                        current_book=book_name,
                    )

                synth_path = output_paths["synth"]
                if synth_path.exists():
                    with open(synth_path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                all_synth_pairs.append(json.loads(line))

                continue

            # ── Обработка книги ──
            result = process_book(
                client, config, book_name, text,
                skip_synth=args.skip_synth,
                synth_count=args.synth_count,
                accumulated_voice=all_voice_pairs,
                accumulated_knowledge=all_knowledge,
                workers=args.workers,
                voice_extractor=args.voice_extractor,
                books_total=max(books_pending_total, 1),
                books_completed_before=processed_books,
                pipeline_t0=pipeline_t0,
                seed=args.seed,
                progress_callback=on_book_progress,
            )
            processed_books += 1

            all_voice_pairs.extend(result["voice_pairs"])
            all_knowledge.extend(result["knowledge"])
            all_synth_pairs.extend(result["synth_pairs"])
            metadata_event(
                "book_results_aggregated",
                current_stage="books_loop",
                current_book=book_name,
                books_processed=processed_books,
                voice_pairs=len(all_voice_pairs),
                synth_pairs=len(all_synth_pairs),
                knowledge_raw_facts=len(all_knowledge),
                book_statuses={book_name: "completed"},
                current_book_progress={
                    "book_name": book_name,
                    "book_voice_pairs": len(result["voice_pairs"]),
                    "book_knowledge_facts": len(result["knowledge"]),
                    "book_synth_pairs": len(result["synth_pairs"]),
                },
            )
            refresh_global_knowledge_snapshot(
                "global_kb_snapshot_updated",
                current_book=book_name,
            )

            # Человекочитаемые версии
            if result["voice_pairs"]:
                save_readable_voice(
                    result["voice_pairs"],
                    str(output_paths["voice_txt"]),
                )
            if result["knowledge"]:
                save_readable_knowledge(
                    result["knowledge"],
                    str(output_paths["knowledge_txt"]),
                )
            if result["synth_pairs"]:
                save_readable_voice(
                    result["synth_pairs"],
                    str(output_paths["synth_txt"]),
                )

            print(f"  Промежуточные файлы сохранены в {config.output_dir}/")

        # Дедупликация датасета
        print(f"\n[{now_str()}] Дедупликация...")
        unique_voice = deduplicate(all_voice_pairs)
        unique_synth = deduplicate(all_synth_pairs)
        print(f"  Голос: {len(all_voice_pairs)} → {len(unique_voice)}")
        print(f"  Синтетика: {len(all_synth_pairs)} → {len(unique_synth)}")
        metadata_event(
            "dataset_deduplicated",
            current_stage="dataset_dedup",
            current_book=None,
            voice_pairs=len(unique_voice),
            synth_pairs=len(unique_synth),
            total_pairs=len(unique_voice) + len(unique_synth),
        )

        # Отдельный этап: сборка общей базы знаний из per-book артефактов
        print(f"\n[{now_str()}] Сборка общей базы знаний...")
        raw_global_knowledge, unique_knowledge = build_global_knowledge_base(
            config.output_dir,
            [book_name for book_name, _ in books],
            log_prefix="[global][knowledge]",
            progress_callback=lambda event_type, **event_data: metadata_event(
                event_type,
                current_stage="global_knowledge",
                current_book=None,
                **event_data,
            ),
        )
        print(f"  Знания: {len(raw_global_knowledge)} → {len(unique_knowledge)}")
        metadata_event(
            "global_knowledge_ready",
            current_stage="global_knowledge",
            current_book=None,
            knowledge_raw_facts=len(raw_global_knowledge),
            knowledge_facts=len(unique_knowledge),
        )

        print(f"\n[{now_str()}] Timeline resolution и сборка графа...")
        raw_timeline_groups, timeline_graph = build_timeline_resolution_artifacts(
            client,
            config,
            config.output_dir,
            [book_name for book_name, _ in books],
            raw_global_knowledge,
            unique_knowledge,
            log_prefix="[global][timeline]",
            progress_callback=lambda event_type, **event_data: metadata_event(
                event_type,
                current_stage="timeline_resolution",
                current_book=None,
                **event_data,
            ),
        )
        print(
            f"  Timeline: групп={len(raw_timeline_groups)}, "
            f"узлов={len(timeline_graph.get('nodes', []))}, "
            f"рёбер={len(timeline_graph.get('edges', []))}"
        )
        metadata_event(
            "timeline_ready",
            current_stage="timeline_resolution",
            current_book=None,
            timeline_groups=len(raw_timeline_groups),
            timeline_nodes=len(timeline_graph.get("nodes", [])),
            timeline_edges=len(timeline_graph.get("edges", [])),
        )

        # Объединённый датасет
        all_pairs = unique_voice + unique_synth
        output_path = global_output_paths["dataset"]
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        # Человекочитаемые итоговые файлы
        save_readable_voice(all_pairs, str(global_output_paths["dataset_txt"]))
        kb_path = global_output_paths["knowledge"]

        # Метаданные запуска
        total_dur = time.time() - pipeline_t0
        metadata_event(
            "run_completed",
            "Пайплайн завершён",
            status="completed",
            current_stage="done",
            current_book=None,
            current_book_progress={},
            total_pairs=len(all_pairs),
            voice_pairs=len(unique_voice),
            synth_pairs=len(unique_synth),
            knowledge_raw_facts=len(raw_global_knowledge),
            knowledge_facts=len(unique_knowledge),
            timeline_groups=len(raw_timeline_groups),
            timeline_nodes=len(timeline_graph.get("nodes", [])),
            timeline_edges=len(timeline_graph.get("edges", [])),
            books_processed=len(books),
            total_duration_seconds=round(total_dur, 1),
            total_duration=fmt_duration(total_dur),
            run_finished_at=now_iso_str(),
        )

        print(f"\n{'='*60}")
        print(f"[{now_str()}] Готово за {fmt_duration(total_dur)}!")
        print(f"  Датасет: {output_path} ({len(all_pairs)} пар)")
        print(f"    — голос: {len(unique_voice)}")
        print(f"    — синтетика: {len(unique_synth)}")
        print(f"  База знаний: {kb_path} ({len(unique_knowledge)} фактов)")
        print(f"  Timeline graph: {global_output_paths['timeline_graph']} ({len(timeline_graph.get('nodes', []))} узлов)")
        print(f"{'='*60}")

        # Статистика
        avg_len = sum(len(p["messages"][-1]["content"]) for p in all_pairs) / max(len(all_pairs), 1)
        print(f"\n  Средняя длина ответа: {avg_len:.0f} символов")

        cat_counts = {}
        for f in unique_knowledge:
            cat = f.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if cat_counts:
            print(f"  Факты по категориям:")
            for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")

        return 0
    except GracefulInterrupt:
        metadata_event(
            "run_interrupted",
            "Пайплайн остановлен пользователем",
            status="interrupted",
            current_stage="interrupted",
            run_finished_at=now_iso_str(),
        )
        print(
            f"\n[{now_str()}] Остановка выполнена аккуратно. "
            f"Промежуточный прогресс сохранён в {config.output_dir}/, "
            "можно продолжить повторным запуском."
        )
        return 130
    except KeyboardInterrupt:
        request_stop()
        metadata_event(
            "run_interrupted",
            "Пайплайн остановлен пользователем",
            status="interrupted",
            current_stage="interrupted",
            run_finished_at=now_iso_str(),
        )
        print(
            f"\n[{now_str()}] Остановка выполнена аккуратно. "
            f"Промежуточный прогресс сохранён в {config.output_dir}/, "
            "можно продолжить повторным запуском."
        )
        return 130
    except Exception:
        metadata_event(
            "run_failed",
            status="failed",
            current_stage="failed",
            run_finished_at=now_iso_str(),
        )
        raise
    finally:
        stop_ollama(ollama_process)


if __name__ == "__main__":
    raise SystemExit(main())
