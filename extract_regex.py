#!/usr/bin/env python3
"""
Извлечение диалогов Сэра Макса — regex-подход.
НЕ использует LLM для извлечения. Работает мгновенно.

LLM нужна только для:
- Проход 2: извлечение знаний (опционально, --with-knowledge)
- Проход 3: синтетическая генерация (опционально)

Использование:
  python extract_regex.py --books-dir ./books
  python extract_regex.py --books-dir ./books --with-knowledge
"""
import json
import os
import re
import hashlib
import argparse
import random
import difflib
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# Карта рассказчиков
# ──────────────────────────────────────────────

NARRATOR_MAP = {
    "чуб земли":          ("Меламори Блимм",    "knowledge_only"),
    "туланский детектив": ("Меламори Блимм",    "knowledge_only"),
    "властелин морморы":  ("Джуффин Халли",     "knowledge_only"),
    "ворона на мосту":    ("Шурф Лонли-Локли",  "knowledge_only"),
    "горе господина гро": ("Кофа Йох",          "knowledge_only"),
    "обжора-хохотун":     ("Мелифаро",          "knowledge_only"),
    "обжора хохотун":     ("Мелифаро",          "knowledge_only"),
    "тубурская игра":     ("Нумминорих Кута",   "knowledge_only"),
    "неуловимый хабба":   ("Макс", "full"),
    "дар шаванахолы":     ("Макс", "full"),
    "хроники ехо (сборник)": ("_сборник", "knowledge_raw"),
    "мой рагнарёк":       ("Макс", "voice_only"),
    "мой рагнарек":       ("Макс", "voice_only"),
    "гнёзда химер":       ("Макс", "voice_only"),
    "гнезда химер":       ("Макс", "voice_only"),
    "ключ из желтого":    ("Макс", "voice_only"),
    "энциклопедия мифов": ("Макс", "voice_only"),
    "чашка фрая":         ("_сборник",      "knowledge_raw"),
    "жалобная книга":     ("_сборник",      "knowledge_raw"),
}

def detect_book_mode(filename):
    name_lower = filename.lower()
    for key, val in NARRATOR_MAP.items():
        if key in name_lower:
            return val
    return "Макс", "full"


# ──────────────────────────────────────────────
# FB2 парсер
# ──────────────────────────────────────────────

def parse_fb2(fb2_content):
    try:
        root = ET.fromstring(fb2_content)
    except ET.ParseError:
        try:
            decoded = fb2_content.decode("windows-1251")
            decoded = re.sub(r'encoding=["\']windows-1251["\']', 'encoding="utf-8"',
                             decoded, flags=re.IGNORECASE)
            root = ET.fromstring(decoded.encode("utf-8"))
        except Exception:
            return ""

    ns = ""
    match = re.match(r"\{(.+?)\}", root.tag)
    if match:
        ns = match.group(1)

    def tag(name):
        return f"{{{ns}}}{name}" if ns else name

    lines = []
    for body in root.iter(tag("body")):
        for element in body.iter():
            if element.tag == tag("binary"):
                continue
            if element.tag in (tag("p"), tag("v"), tag("subtitle"), tag("text-author")):
                # Собираем текст с маркерами emphasis для Безмолвной речи
                parts = []
                for node in element.iter():
                    if node.tag == tag("emphasis") and node.text:
                        # Оборачиваем emphasis в маркер ‹›
                        parts.append(f"‹{node.text.strip()}›")
                        if node.tail:
                            parts.append(node.tail)
                    elif node.text and node.tag == element.tag:
                        parts.append(node.text)
                    elif node.tail and node.tag != tag("emphasis"):
                        parts.append(node.tail)

                text = "".join(parts).strip()
                if not text:
                    # Fallback — просто весь текст
                    text = "".join(element.itertext()).strip()
                if text:
                    lines.append(text)
            elif element.tag == tag("section"):
                if lines and lines[-1] != "":
                    lines.append("")
            elif element.tag == tag("title"):
                title_text = " ".join("".join(p.itertext()).strip()
                                      for p in element.iter(tag("p")))
                if title_text:
                    lines.append(title_text)
                    lines.append("")
    return "\n".join(lines)


def load_fb2_file(filepath):
    name_lower = filepath.name.lower()
    ext = filepath.suffix.lower()
    if name_lower.endswith(".fb2.zip") or ext == ".zip":
        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                fb2_names = [n for n in zf.namelist() if n.lower().endswith(".fb2")]
                if not fb2_names:
                    return None
                return parse_fb2(zf.read(fb2_names[0]))
        except zipfile.BadZipFile:
            return None
    elif ext == ".fb2":
        return parse_fb2(filepath.read_bytes())
    return None


def load_books(books_dir):
    books = []
    books_path = Path(books_dir)
    if not books_path.exists():
        print(f"Папка {books_dir} не найдена.")
        return []

    def is_supported(f):
        return f.is_file() and (f.suffix.lower() in (".txt", ".fb2")
                                 or f.name.lower().endswith(".fb2.zip")
                                 or f.suffix.lower() == ".zip")

    for f in sorted(books_path.rglob("*")):
        if not is_supported(f):
            continue
        text = None
        if f.suffix.lower() == ".txt":
            text = f.read_text(encoding="utf-8", errors="replace")
        else:
            text = load_fb2_file(f)
        rel_path = str(f.relative_to(books_path))
        if text and len(text.strip()) > 100:
            books.append((rel_path, text))
            print(f"  {rel_path} ({len(text):,} симв.)")
    return books


# ──────────────────────────────────────────────
# Regex-извлечение прямой речи
# ──────────────────────────────────────────────

# Паттерны атрибуции говорящего
SPEAKER_PATTERNS = [
    # "— текст, — сказал Джуффин" → speaker=Джуффин
    r'(?:сказал[аи]?\s+|спросил[аи]?\s+|ответил[аи]?\s+|'
    r'проворчал[аи]?\s+|пробормотал[аи]?\s+|буркнул[аи]?\s+|'
    r'вздохнул[аи]?\s+|усмехнулся\s+|усмехнулась\s+|'
    r'рассмеялся\s+|рассмеялась\s+|'
    r'кивнул[аи]?\s+|покачал[аи]?\s+головой\s+|'
    r'объяснил[аи]?\s+|заметил[аи]?\s+|добавил[аи]?\s+|'
    r'согласился\s+|согласилась\s+|'
    r'обиделся\s+|обиделась\s+|'
    r'подтвердил[аи]?\s+|возразил[аи]?\s+|'
    r'воскликнул[аи]?\s+|прошептал[аи]?\s+|'
    r'предложил[аи]?\s+|попросил[аи]?\s+|'
    r'крикнул[аи]?\s+|произнёс\s+|произнесла\s+)'
    r'((?:сэр\s+|леди\s+|господин\s+)?[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\-]+)?)',

    # "— текст, — он/она/я"
    r'(?:сказал[аи]?\s+|спросил[аи]?\s+|ответил[аи]?\s+)(я|он|она)\b',
]

# Ключевые персонажи Ехо для поиска по имени
KNOWN_SPEAKERS = {
    "Джуффин", "Шурф", "Мелифаро", "Кофа", "Меламори", "Теххи",
    "Нумминорих", "Лонли-Локли", "Халли", "Гуриг", "Нуфлин",
    "Сотофа", "Лойсо", "Дримарондо", "Друппи", "Франк",
    "Кекки", "Базилио", "Кима", "Анчифа", "Мохи",
    "Джуффин Халли", "Шурф Лонли-Локли", "Кофа Йох",
    "Меламори Блимм",
}


def detect_speaker(line, context_after=""):
    """Определяет говорящего из строки с репликой."""
    combined = line + " " + context_after

    # "спросил я", "сказал я" → Макс
    if re.search(r'\b(сказал[аи]?\s+я|спросил[аи]?\s+я|ответил\s+я|'
                 r'вздохнул\s+я|буркнул\s+я|пробормотал\s+я|'
                 r'согласился\s+я|подтвердил\s+я|возразил\s+я|'
                 r'воскликнул\s+я|прошептал\s+я|предложил\s+я)\b', combined):
        return "Макс"

    # Ищем имя после глагола речи
    for pattern in SPEAKER_PATTERNS:
        m = re.search(pattern, combined)
        if m:
            speaker = m.group(1).strip()
            if speaker in ("я",):
                return "Макс"
            if speaker in ("он", "она"):
                return speaker
            return speaker

    return None


def extract_direct_speech(text):
    """
    Извлекает прямую речь из русского литературного текста.
    Возвращает список словарей.
    """
    lines = text.split("\n")
    results = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Прямая речь начинается с тире
        if not line.startswith("—") and not line.startswith("–") and not line.startswith("-\u00a0"):
            continue

        # Убираем начальное тире
        speech_line = re.sub(r'^[—–]\s*', '', line).strip()
        if len(speech_line) < 5:
            continue

        # Контекст: следующие 2 строки (для атрибуции)
        context_after = " ".join(
            lines[j].strip() for j in range(i + 1, min(i + 3, len(lines)))
            if lines[j].strip()
        )

        # Контекст: предыдущие 2 строки
        context_before_lines = []
        for j in range(max(0, i - 2), i):
            if lines[j].strip():
                context_before_lines.append(lines[j].strip())
        context_before = " ".join(context_before_lines)

        speaker = detect_speaker(speech_line, context_after)

        # Разделяем реплику и авторскую речь
        # "Какое дельце? — спросил я. — Опять убивать?"
        # Части реплики разделены авторскими вставками
        parts = re.split(r'\s*—\s*(?=[а-яё])', speech_line)

        actual_speech = ""
        if len(parts) == 1:
            actual_speech = parts[0]
        else:
            # Первая часть — точно речь
            # Остальные — речь или авторская вставка
            speech_parts = [parts[0]]
            for p in parts[1:]:
                # Если начинается с глагола речи — это авторская вставка
                if re.match(r'(сказал|спросил|ответил|вздохнул|проворчал|'
                           r'буркнул|пробормотал|усмехнул|рассмеял|'
                           r'кивнул|покачал|объяснил|заметил|добавил|'
                           r'согласил|обидел|подтвердил|возразил|'
                           r'воскликнул|прошептал|предложил|попросил|'
                           r'крикнул|произнёс|произнесла)', p, re.IGNORECASE):
                    # Может содержать продолжение речи после точки
                    # "спросил я. — Опять убивать?"
                    continuation = re.search(r'[.!?]\s*—?\s*(.+)', p)
                    if continuation:
                        speech_parts.append(continuation.group(1))
                    # Ищем говорящего в этой части
                    if not speaker:
                        speaker = detect_speaker(p, "")
                else:
                    speech_parts.append(p)

            actual_speech = " ".join(speech_parts).strip()

        # Очистка
        actual_speech = re.sub(r'\s+', ' ', actual_speech).strip()
        actual_speech = actual_speech.rstrip('.')

        if len(actual_speech) < 5:
            continue

        results.append({
            "speaker": speaker,
            "text": actual_speech,
            "line_num": i,
            "context_before": context_before[:200],
            "raw_line": line,
        })

    return results


def extract_silent_speech(text):
    """
    Извлекает Безмолвную речь — мысленное общение между персонажами.
    В книгах Фрая оформляется:
    1. В «кавычках-ёлочках» внутри повествования
    2. Через маркеры emphasis в FB2 (‹маркер›)
    3. С контекстными словами: зов, Безмолвная речь, послал зов, сказал я мысленно
    """
    results = []

    # Паттерн 1: «текст» с контекстом Безмолвной речи рядом
    # Ищем «кавычки» в параграфах, содержащих маркеры мысленной речи
    silent_markers = (
        r'[Бб]езмолвн|[Зз]ов\b|послал[аи]?\s+зов|'
        r'мысленно|[Бб]езмолвн\w+\s+реч|'
        r'подумал[аи]?\s+я|сказал[аи]?\s+я\s+(?:ему|ей|им)\s+мысленно|'
        r'ответил[аи]?\s+я\s+мысленно|'
        r'услышал\s+я\s+(?:его|её|их)\s+голос|'
        r'прозвучал\s+(?:в|у)\s+(?:моей|моём)\s+голов'
    )

    lines = text.split("\n")
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 10:
            continue

        # Контекст: текущая строка + соседние
        window = line_stripped
        for j in range(max(0, i-1), min(len(lines), i+2)):
            if j != i:
                window += " " + lines[j].strip()

        has_silent_marker = bool(re.search(silent_markers, window))
        if not has_silent_marker:
            continue

        # Извлекаем текст в «кавычках»
        quotes = re.findall(r'«([^»]{5,300})»', line_stripped)
        for q in quotes:
            # Определяем, кто говорит
            speaker = None
            if re.search(r'(подумал\s+я|сказал\s+я|ответил\s+я|послал\s+я)', window):
                speaker = "Макс"
            elif re.search(r'(подумал\s+я|мой\s+зов|я\s+послал)', window):
                speaker = "Макс"
            else:
                # Ищем имя в контексте
                for name in KNOWN_SPEAKERS:
                    if name in window:
                        speaker = name
                        break

            results.append({
                "type": "silent_speech",
                "speaker": speaker,
                "text": q,
                "context": line_stripped[:200],
                "line_num": i,
            })

    # Паттерн 2: emphasis-маркеры из FB2 (‹текст›)
    emphasis_in_context = re.finditer(r'‹([^›]{5,300})›', text)
    for m in emphasis_in_context:
        emph_text = m.group(1)
        # Берём контекст вокруг
        start = max(0, m.start() - 200)
        end = min(len(text), m.end() + 200)
        ctx = text[start:end]

        if re.search(silent_markers, ctx):
            speaker = None
            if re.search(r'(подумал\s+я|сказал\s+я|послал\s+я|мой\s+зов)', ctx):
                speaker = "Макс"
            results.append({
                "type": "silent_speech",
                "speaker": speaker,
                "text": emph_text,
                "context": ctx[:200].replace("\n", " "),
                "line_num": -1,
            })

    # Дедупликация по тексту
    seen = set()
    unique = []
    for r in results:
        h = hashlib.md5(r["text"].lower().encode()).hexdigest()[:12]
        if h not in seen:
            seen.add(h)
            unique.append(r)

    return unique


def extract_monologues(text, min_length=80):
    """
    Извлекает внутренние монологи Макса — фрагменты повествования от первого лица
    с характерными маркерами (ирония, рефлексия, эмоции).
    """
    # Маркеры внутреннего голоса Макса
    markers = [
        r'\bя\s+(подумал|решил|вздохнул|усмехнул|обрадовал|'
        r'ужаснул|растерял|понял|вспомнил|представил|почувствовал)\b',
        r'\bмне\s+(казалось|хотелось|пришлось|стало|показалось|'
        r'нравилось|было\s+страшно|было\s+стыдно|было\s+лень)\b',
        r'\bмоя\s+(жизнь|судьба|голова)\b',
        r'\bк\s+счастью\b',
        r'\bк\s+сожалению\b',
        r'\bвпрочем\b',
        r'\bстрого\s+говоря\b',
        r'\bна\s+самом\s+деле\b',
        r'\bчестно\s+говоря\b',
        r'\bнадо\s+же\b',
        r'\bбоже\s+мой\b',
        r'\bгрёбаные\s+магистры\b',
        r'\bхвала\s+магистрам\b',
        r'\bдырку\b.*\bв\s+небе\b',
        r'\bкамр[уыа]\b',
    ]

    paragraphs = re.split(r'\n\s*\n|\n', text)
    results = []

    for para in paragraphs:
        para = para.strip()
        if len(para) < min_length:
            continue
        # Пропускаем прямую речь
        if para.startswith("—") or para.startswith("–"):
            continue

        # Проверяем маркеры
        marker_count = sum(1 for m in markers if re.search(m, para, re.IGNORECASE))
        # Также: местоимения первого лица
        first_person = len(re.findall(r'\b(я|мне|мой|моя|моё|мою|моей|моих|меня|мной)\b',
                                      para, re.IGNORECASE))

        # Нужно хотя бы 1 маркер или много первого лица
        if marker_count >= 1 or first_person >= 3:
            results.append({
                "text": para,
                "marker_count": marker_count,
                "first_person_count": first_person,
            })

    return results


# ──────────────────────────────────────────────
# Формирование обучающих пар
# ──────────────────────────────────────────────

# System prompt
CHARACTER_SYSTEM_PROMPT = (
    "Ты — Сэр Макс, Тайный Сыщик Малого Тайного Сыскного Войска города Ехо, "
    "Смерть на Королевской Службе, Вершитель."
    # Укороченная версия — полный промпт из extract_dialogues.py можно подставить
)


def make_pairs_from_speech(speech_items, system_prompt):
    """Формирует обучающие пары из извлечённой прямой речи."""
    pairs = []
    prev_speaker = None
    prev_text = None

    for item in speech_items:
        speaker = item["speaker"]
        text = item["text"]

        if speaker == "Макс" and len(text) >= 15:
            # Контекст = предыдущая реплика другого персонажа
            if prev_speaker and prev_speaker != "Макс" and prev_text:
                user_content = f"[{prev_speaker}]: {prev_text}"
            elif item.get("context_before"):
                user_content = f"[Ситуация]: {item['context_before'][:200]}"
            else:
                user_content = "[Продолжай разговор]"

            pairs.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": text},
                ]
            })

        prev_speaker = speaker
        prev_text = text

    return pairs


def make_pairs_from_monologues(monologues, system_prompt, max_items=500):
    """Формирует обучающие пары из внутренних монологов."""
    pairs = []
    for mono in monologues[:max_items]:
        text = mono["text"]
        if len(text) < 30 or len(text) > 1000:
            continue

        # Берём первое предложение как «тему»
        first_sent = re.split(r'[.!?]', text)[0].strip()
        if len(first_sent) > 100:
            first_sent = first_sent[:100]

        pairs.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[Расскажи, что ты думаешь о: {first_sent}]"},
                {"role": "assistant", "content": text},
            ]
        })

    return pairs


def make_pairs_from_silent(silent_items, system_prompt):
    """Формирует обучающие пары из Безмолвной речи."""
    pairs = []
    for item in silent_items:
        if item["speaker"] != "Макс":
            continue
        text = item["text"]
        if len(text) < 15:
            continue

        context = item.get("context", "")[:200]
        pairs.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[Безмолвная речь] {context}"},
                {"role": "assistant", "content": text},
            ]
        })

    return pairs


# ──────────────────────────────────────────────
# Дедупликация
# ──────────────────────────────────────────────

def text_hash(text):
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]

def deduplicate(pairs):
    seen = set()
    unique = []
    for pair in pairs:
        user = pair["messages"][1]["content"]
        assistant = pair["messages"][-1]["content"]
        h = text_hash(f"{user}||{assistant}")
        if h not in seen:
            seen.add(h)
            unique.append(pair)
    return unique


# ──────────────────────────────────────────────
# Человекочитаемый экспорт
# ──────────────────────────────────────────────

def save_readable(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, pair in enumerate(pairs, 1):
            user = pair["messages"][1]["content"]
            assistant = pair["messages"][-1]["content"]
            f.write(f"{'─' * 60}\n#{i}\n\n")
            f.write(f"[Ситуация]: {user}\n\n")
            f.write(f"[Макс]: {assistant}\n\n")
        f.write(f"{'─' * 60}\nВсего: {len(pairs)} пар\n")


def save_stats(speech_items, path):
    """Сохраняет статистику по говорящим."""
    speakers = {}
    for item in speech_items:
        s = item["speaker"] or "неизвестно"
        speakers[s] = speakers.get(s, 0) + 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("СТАТИСТИКА РЕПЛИК\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Всего реплик: {len(speech_items)}\n\n")
        for speaker, count in sorted(speakers.items(), key=lambda x: -x[1]):
            pct = count / len(speech_items) * 100
            f.write(f"  {speaker}: {count} ({pct:.1f}%)\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regex-извлечение диалогов Сэра Макса")
    parser.add_argument("--books-dir", default="./books")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Загрузка книг...")
    books = load_books(args.books_dir)
    if not books:
        print("Книги не найдены.")
        return

    # Сортировка
    mode_order = {"full": 0, "voice_only": 1, "knowledge_only": 2, "knowledge_raw": 3}
    books.sort(key=lambda x: (mode_order.get(detect_book_mode(x[0])[1], 9), x[0]))

    all_pairs = []
    total_speech = 0
    total_monologues = 0

    for book_name, text in books:
        narrator, mode = detect_book_mode(book_name)
        do_voice = mode in ("full", "voice_only")

        icon = {"full": "🔊", "voice_only": "🗣", "knowledge_only": "📖", "knowledge_raw": "📚"}
        print(f"\n{icon.get(mode, '?')} {book_name} [{narrator}, {mode}]")

        if not do_voice:
            print(f"  Пропуск (нет голоса Макса)")
            continue

        # Извлечение прямой речи
        speech = extract_direct_speech(text)
        max_speech = [s for s in speech if s["speaker"] == "Макс"]
        print(f"  Реплик всего: {len(speech)}, из них Макс: {len(max_speech)}")

        # Извлечение Безмолвной речи
        silent = extract_silent_speech(text)
        max_silent = [s for s in silent if s["speaker"] == "Макс"]
        print(f"  Безмолвная речь: {len(silent)}, из них Макс: {len(max_silent)}")

        # Извлечение монологов
        monologues = extract_monologues(text)
        print(f"  Монологов: {len(monologues)}")

        # Формирование пар
        voice_pairs = make_pairs_from_speech(speech, CHARACTER_SYSTEM_PROMPT)
        silent_pairs = make_pairs_from_silent(silent, CHARACTER_SYSTEM_PROMPT)
        mono_pairs = make_pairs_from_monologues(monologues, CHARACTER_SYSTEM_PROMPT)
        book_pairs = voice_pairs + silent_pairs + mono_pairs
        print(f"  Обучающих пар: {len(book_pairs)} "
              f"({len(voice_pairs)} диалогов + {len(silent_pairs)} безмолвных + {len(mono_pairs)} монологов)")

        all_pairs.extend(book_pairs)
        total_speech += len(max_speech)
        total_monologues += len(monologues)

        # Промежуточное сохранение
        book_stem = Path(book_name).with_suffix("").as_posix().replace("/", "--")
        save_readable(book_pairs, str(Path(args.output_dir) / f"voice_{book_stem}.txt"))

        # Статистика по говорящим
        save_stats(speech, str(Path(args.output_dir) / f"stats_{book_stem}.txt"))

    # Дедупликация
    unique = deduplicate(all_pairs)
    print(f"\nДедупликация: {len(all_pairs)} → {len(unique)}")

    # Сохранение
    output_path = Path(args.output_dir) / "dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in unique:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    save_readable(unique, str(Path(args.output_dir) / "dataset.txt"))

    print(f"\n{'=' * 60}")
    print(f"Готово!")
    print(f"  Датасет: {output_path} ({len(unique)} пар)")
    print(f"  dataset.txt — читаемая версия")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
