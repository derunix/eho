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
    max_tokens_knowledge: int = 700     # JSON с фактами из одного чанка
    max_tokens_synth: int = 500         # одна пара вопрос-ответ

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


def preview_text(text: str, limit: int = 100) -> str:
    """Короткий однострочный превью текста для логов."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit - 3].rstrip() + "..."


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

    for body in root.iter(tag("body")):
        for element in body.iter():
            # Пропускаем binary и metadata
            if element.tag == tag("binary"):
                continue

            # Абзацы, заголовки, стихи, цитаты
            if element.tag in (tag("p"), tag("v"), tag("subtitle"), tag("text-author")):
                text = "".join(element.itertext()).strip()
                if text:
                    lines.append(text)

            # Пустая строка между секциями для читаемости
            elif element.tag == tag("section"):
                if lines and lines[-1] != "":
                    lines.append("")

            # Заголовки секций
            elif element.tag == tag("title"):
                title_text = " ".join(
                    "".join(p.itertext()).strip()
                    for p in element.iter(tag("p"))
                )
                if title_text:
                    lines.append(title_text)
                    lines.append("")

            # Эпиграфы
            elif element.tag == tag("epigraph"):
                for p in element.iter(tag("p")):
                    text = "".join(p.itertext()).strip()
                    if text:
                        lines.append(text)
                lines.append("")

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
) -> Optional[str]:
    """Вызов через нативный API ollama с think=false."""
    import urllib.request

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

    payload = json.dumps(payload_data).encode("utf-8")

    for attempt in range(3):
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
                if log_prefix:
                    elapsed = time.time() - t0
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
            if log_prefix:
                elapsed = time.time() - t0
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
) -> Optional[str]:
    """Вызов через OpenAI-совместимый API (vllm, llama.cpp, LM Studio)."""
    effective_temperature = config.temperature if temperature is None else temperature
    for attempt in range(3):
        t0 = time.time()
        if log_prefix:
            log_event(
                f"{log_prefix} -> LLM запрос "
                f"(попытка {attempt + 1}/3, max_tokens={max_tokens})"
            )
        try:
            kwargs = {
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
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(
                **kwargs,
            )
            content = response.choices[0].message.content
            if content and "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if log_prefix:
                elapsed = time.time() - t0
                finish_reason = response.choices[0].finish_reason or "unknown"
                usage = getattr(response, "usage", None)
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
            if log_prefix:
                elapsed = time.time() - t0
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
    if _use_ollama_native:
        return call_llm_ollama_native(
            config,
            system,
            user,
            max_tokens,
            response_format=response_format,
            log_prefix=log_prefix,
            temperature=temperature,
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
        )


# ──────────────────────────────────────────────
# Шаг 1: Нарезка текста на фрагменты
# ──────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Разбивает текст на куски по ~chunk_size токенов с перехлёстом.
    Режет по абзацам, чтобы не разрывать предложения.
    """
    paragraphs = re.split(r"\n\s*\n|\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_size + para_tokens > chunk_size and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(chunk_text)

            # Перехлёст: берём последние абзацы на ~overlap токенов
            overlap_paras = []
            overlap_size = 0
            for p in reversed(current_chunk):
                p_size = estimate_tokens(p)
                if overlap_size + p_size > overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_size += p_size

            current_chunk = overlap_paras
            current_size = overlap_size

        current_chunk.append(para)
        current_size += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


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

Верни JSON массив.
НЕ ПИШИ НИЧЕГО, КРОМЕ JSON. НИКАКИХ ОБЪЯСНЕНИЙ ИЛИ ВВОДНЫХ СЛОВ.
НЕ используй ```json или другие markdown-обёртки — только чистый JSON.
Фрагмент:
---
{chunk}
---

JSON:"""


def extract_dialogues(
    client: OpenAI,
    config: Config,
    chunk: str,
    log_prefix: str = "",
) -> list[dict]:
    """Извлекает структурированные диалоги из фрагмента."""
    response = call_llm(
        client,
        config,
        EXTRACT_SYSTEM,
        EXTRACT_PROMPT.format(chunk=chunk),
        max_tokens=config.max_tokens_extract,
        response_format="json" if _use_ollama_native else None,
        log_prefix=log_prefix,
    )

    if response is None:
        return []

    data, strategy = parse_json_response(response, expect="array", log_prefix=log_prefix)
    if isinstance(data, list):
        if log_prefix:
            log_event(f"{log_prefix} JSON ok: {len(data)} элементов ({strategy})")
        return data

    if log_prefix:
        log_event(f"{log_prefix} ответ не удалось распарсить как JSON-массив")

    return []


def fuzzy_find(needle: str, haystack: str, threshold: float = 0.7) -> bool:
    """Проверяет, что needle (или очень похожая строка) содержится в haystack.
    Использует скользящее окно для поиска лучшего совпадения."""
    if not needle or not haystack:
        return False

    # Точное вхождение
    if needle in haystack:
        return True

    # Нормализованное сравнение (убираем лишние пробелы, тире, кавычки)
    def normalize(s):
        s = re.sub(r'[«»""\'\-—–]', '', s)
        s = re.sub(r'\s+', ' ', s).strip().lower()
        return s

    norm_needle = normalize(needle)
    norm_haystack = normalize(haystack)

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

    for d in items:
        source_kind = d.get("_source", "")
        max_says = d.get("max_says", "").strip()
        if len(max_says) < 15:
            rejected["too_short"] += 1
            continue
        if "type" not in d:
            rejected["missing_type"] += 1
            continue
        if d["type"] not in ("dialogue", "silent_speech", "monologue"):
            rejected["invalid_type"] += 1
            continue

        # Проверка: цитата реально из текста, а не галлюцинация LLM
        if source_kind != "regex" and source_chunk and not fuzzy_find(max_says, source_chunk):
            rejected["max_says_not_found"] += 1
            continue

        # Если есть interlocutor_says — тоже проверяем
        interlocutor_says = d.get("interlocutor_says", "").strip()
        if source_kind != "regex" and source_chunk and interlocutor_says and len(interlocutor_says) > 15:
            if not fuzzy_find(interlocutor_says, source_chunk):
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
        fact = f.get("fact", "").strip()
        subject = f.get("subject", "").strip()
        # Факт и субъект обязательны
        if len(fact) < 10:
            rejected["short_fact"] += 1
            continue
        if len(subject) < 2:
            rejected["short_subject"] += 1
            continue
        # Категория должна быть одной из ожидаемых
        cat = f.get("category", "")
        if cat not in ("character", "place", "magic", "history", "creature", "custom"):
            rejected["bad_category"] += 1
            continue
        valid.append(f)

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
        max_says = d.get("max_says", "").strip()
        if not max_says or len(max_says) < 10:
            continue

        dtype = d.get("type", "dialogue")
        context = d.get("context", "").strip()
        interlocutor = d.get("interlocutor")
        interlocutor_says = d.get("interlocutor_says", "").strip()

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
Извлекаешь структурированные знания о мире Ехо.
Отвечай СТРОГО в формате JSON. Никакого текста до или после JSON."""

KNOWLEDGE_PROMPT = """Из фрагмента книги Макса Фрая извлеки ФАКТЫ о мире Ехо.

Категории фактов:
- "character": информация о персонаже (биография, способности, характер, отношения)
- "place": место (город, трактир, улица, здание, мир)
- "magic": магия (заклинания, ступени, ордена, артефакты, Очевидная/Истинная магия)
- "history": исторические события (Смутные Времена, Эпоха Орденов, войны, Кодекс)
- "creature": существа (буривухи, овчарки Пустых Земель и т.п.)
- "custom": обычаи, быт, еда, напитки, одежда, транспорт, социальные нормы

Для каждого факта создай объект:
- "category": одна из категорий выше
- "subject": о ком или о чём факт (имя персонажа, название места и т.д.)
- "fact": сам факт, кратко и точно (1-3 предложения)

ПРАВИЛА:
— Извлекай только то, что явно сказано в тексте, не додумывай.
— Один объект = один факт. Не объединяй разные факты.
— Пропускай тривиальные факты (кто-то куда-то пошёл).
— Если в фрагменте нет значимых фактов — верни пустой массив [].
— Верни не более 8 самых полезных фактов на фрагмент.
НЕ ПИШИ НИЧЕГО, КРОМЕ JSON. НИКАКИХ ОБЪЯСНЕНИЙ ИЛИ ВВОДНЫХ СЛОВ.
НЕ используй ```json или другие markdown-обёртки — только чистый JSON.

Фрагмент:
---
{chunk}
---

JSON:"""


def extract_knowledge(
    client: OpenAI,
    config: Config,
    chunk: str,
    log_prefix: str = "",
) -> list[dict]:
    """Извлекает факты о мире из фрагмента."""
    response = call_llm(
        client,
        config,
        KNOWLEDGE_SYSTEM,
        KNOWLEDGE_PROMPT.format(chunk=chunk),
        max_tokens=config.max_tokens_knowledge,
        response_format="json" if _use_ollama_native else None,
        log_prefix=log_prefix,
    )

    if response is None:
        return []

    data, strategy = parse_json_response(response, expect="array", log_prefix=log_prefix)
    if isinstance(data, list):
        if log_prefix:
            log_event(f"{log_prefix} JSON ok: {len(data)} фактов ({strategy})")
        return data

    if log_prefix:
        log_event(f"{log_prefix} ответ не удалось распарсить как JSON-массив")

    return []


def deduplicate_knowledge(facts: list[dict]) -> list[dict]:
    """Убирает дублирующиеся факты по хешу subject+fact."""
    seen = set()
    unique = []
    for f in facts:
        subject = f.get("subject", "").strip().lower()
        fact = f.get("fact", "").strip().lower()
        if not fact:
            continue
        h = text_hash(f"{subject}:{fact}")
        if h not in seen:
            seen.add(h)
            unique.append(f)
    return unique


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
- "category": "character"/"place"/"magic"/"history"/"creature"/"custom"
- "subject": о ком/чём
- "fact": сам факт (1-3 предложения)

ПРАВИЛА:
— Извлекай ТОЧНЫЕ цитаты в max_says, не пересказывай.
— Не придумывай. Пропускай тривиальное.
— Если нечего извлекать — пустые массивы.
— Верни не более 8 элементов в `dialogues` и не более 8 элементов в `knowledge`.

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
) -> tuple:
    """Извлекает диалоги И знания за один вызов LLM."""
    response = call_llm(
        client,
        config,
        COMBINED_SYSTEM,
        COMBINED_PROMPT.format(chunk=chunk),
        max_tokens=config.max_tokens_extract + config.max_tokens_knowledge,
        response_format="json" if _use_ollama_native else None,
        log_prefix=log_prefix,
    )

    dialogues, knowledge = [], []

    if response is None:
        return dialogues, knowledge

    data, strategy = parse_json_response(response, expect="object", log_prefix=log_prefix)
    if isinstance(data, dict):
        dialogues = data.get("dialogues", [])
        knowledge = data.get("knowledge", [])
        if not isinstance(dialogues, list):
            dialogues = []
        if not isinstance(knowledge, list):
            knowledge = []
        if log_prefix:
            log_event(
                f"{log_prefix} JSON ok: dialogues={len(dialogues)}, "
                f"knowledge={len(knowledge)} ({strategy})"
            )
    elif log_prefix:
        log_event(f"{log_prefix} ответ не удалось распарсить как JSON-объект")

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
        log_event(f"{tag} старт: {preview_text(chunk, 90)}")

        regex_dialogues = []
        regex_stats = {"speech": 0, "silent": 0, "monologue": 0}
        used_regex_fallback = False

        if do_voice:
            regex_dialogues, regex_stats = extract_voice_with_regex(
                chunk,
                log_prefix=f"{tag}[voice]",
            )

        if do_voice and do_knowledge:
            if voice_extractor == "llm":
                d, k = extract_combined(
                    client,
                    config,
                    chunk,
                    log_prefix=f"{tag}[combined]",
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
                )
        elif do_voice:
            if voice_extractor == "llm":
                d = extract_dialogues(
                    client,
                    config,
                    chunk,
                    log_prefix=f"{tag}[voice]",
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
            )

        meta = {
            "tag": tag,
            "elapsed": time.time() - item_t0,
            "regex_stats": regex_stats,
            "used_regex_fallback": used_regex_fallback,
        }
        return idx, chunk, d, k, meta

    t_start = time.time()
    completed = already_completed
    completed_this_run = 0

    if not chunk_items:
        print(f"\n  [{now_str()}] Новых фрагментов для обработки нет ({completed}/{total})")
        return all_dialogues or [], all_knowledge or []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one, idx_chunk): idx_chunk[0]
            for idx_chunk in chunk_items
        }

        for future in as_completed(futures):
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
            log_event(
                f"{tag} готово [{completed}/{total}]: +{len(dialogues)}d +{len(knowledge)}k "
                f"за {meta['elapsed']:.1f}s, ETA книги {fmt_duration(eta)}"
                f"{extra_progress}{fallback_note} {sample}".rstrip()
            )

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


def pick_style_examples(voice_pairs: list[dict], n: int = 3) -> str:
    """Выбирает несколько примеров речи Макса для few-shot."""
    if not voice_pairs:
        return "(нет примеров)"

    candidates = [
        p["messages"][-1]["content"]
        for p in voice_pairs
        if 30 < len(p["messages"][-1]["content"]) < 300
    ]
    if not candidates:
        candidates = [p["messages"][-1]["content"] for p in voice_pairs]

    selected = random.sample(candidates, min(n, len(candidates)))
    return "\n---\n".join(f"«{s}»" for s in selected)


def generate_synth_pair(
    client: OpenAI,
    config: Config,
    fact: dict,
    voice_pairs: list[dict],
) -> Optional[dict]:
    """Генерирует одну синтетическую обучающую пару из факта."""
    style_examples = pick_style_examples(voice_pairs)

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
) -> list[dict]:
    """Генерирует синтетические обучающие пары из базы знаний с resume."""
    facts = order_facts_for_synth(knowledge, seed)
    target_facts = facts[:min(max_pairs, len(facts))]

    processed_fact_hashes: set[str] = set()
    pairs: list[dict] = []
    if progress_path is not None:
        processed_fact_hashes, pairs = load_synth_progress(progress_path, log_prefix="[synth]")
        if synth_output_path is not None:
            if synth_output_path.exists():
                synth_output_path.unlink()
            append_jsonl(synth_output_path, pairs)

    produced_pairs = len(pairs)

    for i, fact in enumerate(target_facts, 1):
        fact_hash = stable_fact_hash(fact)
        if fact_hash in processed_fact_hashes:
            print(f"    [{i}/{len(target_facts)}] resume: {produced_pairs}", end="\r")
            continue

        pair = generate_synth_pair(client, config, fact, voice_pairs)
        record = {"fact_hash": fact_hash, "pair": pair}
        if progress_path is not None:
            append_jsonl(progress_path, [record])
        processed_fact_hashes.add(fact_hash)
        if pair:
            pairs.append(pair)
            produced_pairs += 1
            if synth_output_path is not None:
                append_jsonl(synth_output_path, [pair])
        print(f"    [{i}/{len(target_facts)}] сгенерировано: {produced_pairs}", end="\r")

    print()
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
        "creature": "Существа",
        "custom": "Обычаи и быт",
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"БАЗА ЗНАНИЙ О МИРЕ ЕХО\n")
        f.write(f"{'═' * 60}\n")
        f.write(f"Всего фактов: {len(facts)}\n\n")

        for cat in ["character", "place", "magic", "history", "creature", "custom", "unknown"]:
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
                by_subject.setdefault(subj, []).append(fact)

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
    workers: int = 2,
    voice_extractor: str = "regex",
    books_total: int = 1,
    books_completed_before: int = 0,
    pipeline_t0: Optional[float] = None,
    seed: int = 42,
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

    book_t0 = time.time()

    # Нарезка
    chunks = split_into_chunks(text, config.chunk_size, config.chunk_overlap)
    print(f"  Нарезано на {len(chunks)} фрагментов")

    # Пути промежуточных файлов
    output_paths = get_book_output_paths(config.output_dir, book_name)
    voice_path = output_paths["voice"]
    knowledge_path = output_paths["knowledge"]
    knowledge_stream_path = output_paths["knowledge_stream"]
    chunk_results_path = output_paths["chunks"]
    synth_path = output_paths["synth"]
    synth_progress_path = output_paths["synth_progress"]
    done_path = output_paths["done"]

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
            voice_pairs, all_knowledge = rebuild_chunk_outputs(
                checkpoint_records,
                config,
                voice_path,
                knowledge_stream_path,
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

        append_jsonl(chunk_results_path, [{
            "idx": idx,
            "chunk_hash": text_hash(chunk),
            "dialogues": dialogues,
            "knowledge": knowledge,
        }])

        if dialogues:
            new_pairs = make_training_pairs(dialogues, config)
            if new_pairs:
                append_jsonl(voice_path, new_pairs)
                voice_pairs.extend(new_pairs)

        if knowledge:
            append_jsonl(knowledge_stream_path, knowledge)
            all_knowledge.extend(knowledge)

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
                )
            else:
                print(f"  Все фрагменты этой книги уже извлечены, продолжаю с сохранённого места")

    if not do_voice:
        print(f"\n  Извлечение голоса пропущено ({narrator}, режим {mode})")
    if not do_knowledge:
        print(f"  Извлечение знаний пропущено (режим {mode})")

    # Формирование пар из диалогов
    if voice_pairs:
        print(f"  Обучающих пар (голос): {len(voice_pairs)}")

    # Дедупликация знаний
    if all_knowledge:
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

def main():
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

    # ── Управление ollama ──
    ollama_process = None

    if not args.no_auto_serve:
        ollama_process = ensure_ollama(config.model)

    # Гарантируем остановку ollama при выходе
    def cleanup():
        stop_ollama(ollama_process)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda sig, frame: (cleanup(), exit(1)))
    signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup(), exit(1)))

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
    test = call_llm(client, config, "Ответь одним словом.", "Скажи: работает",
                    max_tokens=10)
    if test is None:
        print(f"Не удалось подключиться к {config.api_base}")
        print(f"Убедись, что модель запущена (например: ollama run {config.model})")
        return
    print(f"Модель отвечает: {test.strip()}\n")

    # Загрузка книг
    print(f"Загрузка книг из {config.books_dir}/")
    books = load_books(config.books_dir)
    if not books:
        print("Книги не найдены.")
        return

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
    for book_name, _ in books:
        if args.no_resume or not is_book_processed(config.output_dir, book_name):
            books_pending_total += 1

    # Обработка
    all_voice_pairs = []
    all_knowledge = []
    all_synth_pairs = []
    pipeline_t0 = time.time()
    processed_books = 0

    for book_idx, (book_name, text) in enumerate(books):
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

            # Загружаем ранее сохранённые результаты
            if voice_path.exists():
                with open(voice_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_voice_pairs.append(json.loads(line))

            if knowledge_path.exists():
                with open(knowledge_path, encoding="utf-8") as f:
                    all_knowledge.extend(json.load(f))
            elif knowledge_stream_path.exists():
                with open(knowledge_stream_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_knowledge.append(json.loads(line))

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
            workers=args.workers,
            voice_extractor=args.voice_extractor,
            books_total=max(books_pending_total, 1),
            books_completed_before=processed_books,
            pipeline_t0=pipeline_t0,
            seed=args.seed,
        )
        processed_books += 1

        all_voice_pairs.extend(result["voice_pairs"])
        all_knowledge.extend(result["knowledge"])
        all_synth_pairs.extend(result["synth_pairs"])

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

    # Дедупликация
    print(f"\n[{now_str()}] Дедупликация...")
    unique_voice = deduplicate(all_voice_pairs)
    unique_synth = deduplicate(all_synth_pairs)
    unique_knowledge = deduplicate_knowledge(all_knowledge)
    print(f"  Голос: {len(all_voice_pairs)} → {len(unique_voice)}")
    print(f"  Синтетика: {len(all_synth_pairs)} → {len(unique_synth)}")
    print(f"  Знания: {len(all_knowledge)} → {len(unique_knowledge)}")

    # Объединённый датасет
    all_pairs = unique_voice + unique_synth
    output_path = Path(config.output_dir) / "dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Отдельно база знаний
    kb_path = Path(config.output_dir) / "knowledge_base.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(unique_knowledge, f, ensure_ascii=False, indent=2)

    # Человекочитаемые итоговые файлы
    save_readable_voice(all_pairs, str(Path(config.output_dir) / "dataset.txt"))
    save_readable_knowledge(unique_knowledge, str(Path(config.output_dir) / "knowledge_base.txt"))

    # Метаданные запуска
    total_dur = time.time() - pipeline_t0
    metadata = {
        "seed": args.seed,
        "model": config.model,
        "chunk_size": config.chunk_size,
        "workers": args.workers,
        "voice_extractor": args.voice_extractor,
        "request_timeout": config.request_timeout,
        "synth_count": args.synth_count,
        "skip_synth": args.skip_synth,
        "total_pairs": len(all_pairs),
        "voice_pairs": len(unique_voice),
        "synth_pairs": len(unique_synth),
        "knowledge_facts": len(unique_knowledge),
        "books_processed": len(books),
        "total_duration_seconds": round(total_dur, 1),
        "total_duration": fmt_duration(total_dur),
    }
    meta_path = Path(config.output_dir) / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[{now_str()}] Готово за {fmt_duration(total_dur)}!")
    print(f"  Датасет: {output_path} ({len(all_pairs)} пар)")
    print(f"    — голос: {len(unique_voice)}")
    print(f"    — синтетика: {len(unique_synth)}")
    print(f"  База знаний: {kb_path} ({len(unique_knowledge)} фактов)")
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

    # Останавливаем ollama если мы её запускали
    stop_ollama(ollama_process)


if __name__ == "__main__":
    main()
