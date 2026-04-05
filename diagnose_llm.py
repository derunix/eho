#!/usr/bin/env python3
"""Диагностика: что реально отвечает модель на наши промпты."""
import json
import urllib.request

MODEL = "gemma4:e4b"
BASE = "http://localhost:11434"

CHUNK = """— Послушай, Макс, — сказал Джуффин, — у меня для тебя есть одно дельце.
— Какое ещё дельце? — спросил я, с подозрением глядя на шефа. — Опять кого-нибудь убивать?
— Ну что ты сразу — убивать! — обиделся Джуффин. — Просто нужно заглянуть в одно место и разобраться, что там творится.
Я вздохнул. Когда Джуффин говорит «просто заглянуть», это обычно означает, что мне предстоит провести ночь в обществе какого-нибудь обезумевшего мага, который пытается уничтожить мир. Впрочем, камру мне всё равно допить не дали."""

def call(system, user):
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 1500},
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
        return data.get("message", {}).get("content", "")


# Тест 1: Комбинированный промпт (как в скрипте)
print("=" * 60)
print("ТЕСТ 1: Комбинированный промпт")
print("=" * 60)

COMBINED_SYSTEM = """Ты — помощник для разметки литературного текста Макса Фрая.
Извлекаешь одновременно реплики персонажа И факты о мире.
Отвечай СТРОГО в формате JSON. Никакого текста до или после JSON."""

COMBINED_PROMPT = f"""Из фрагмента книги Макса Фрая извлеки ДВА ТИПА данных.

ТИП 1 — РЕПЛИКИ МАКСА (массив "dialogues"):
Для каждого:
- "type": "dialogue" / "silent_speech" / "monologue"
- "context": краткое описание ситуации
- "interlocutor": имя собеседника или null
- "interlocutor_says": реплика собеседника или ""
- "max_says": ТОЧНАЯ цитата из текста

ТИП 2 — ФАКТЫ О МИРЕ (массив "knowledge"):
Для каждого:
- "category": "character"/"place"/"magic"/"history"/"creature"/"custom"
- "subject": о ком/чём
- "fact": сам факт

Верни JSON: {{"dialogues": [...], "knowledge": [...]}}
НЕ ПИШИ НИЧЕГО, КРОМЕ JSON. НИКАКИХ ОБЪЯСНЕНИЙ ИЛИ ВВОДНЫХ СЛОВ.
НЕ используй ```json


Фрагмент:
---
{CHUNK}
---

JSON:"""

resp1 = call(COMBINED_SYSTEM, COMBINED_PROMPT)
print(f"Длина ответа: {len(resp1)} символов")
print(f"Ответ:")
print(resp1[:2000])
print()

# Попробуем распарсить
try:
    data = json.loads(resp1)
    print(f"JSON парсится: ДА")
    print(f"  dialogues: {len(data.get('dialogues', []))}")
    print(f"  knowledge: {len(data.get('knowledge', []))}")
except json.JSONDecodeError as e:
    print(f"JSON парсится: НЕТ — {e}")

print()

# Тест 2: Простой промпт (только диалоги)
print("=" * 60)
print("ТЕСТ 2: Простой промпт (только диалоги)")
print("=" * 60)

SIMPLE_SYSTEM = """Извлеки реплики Макса из текста. Ответь JSON массивом."""

SIMPLE_PROMPT = f"""Найди все реплики Сэра Макса в этом фрагменте.
Для каждой реплики:
- "context": ситуация (кратко)
- "speaker_before": что сказал собеседник перед Максом
- "max_says": точная цитата Макса из текста

Ответь ТОЛЬКО JSON массивом, без пояснений.
НЕ ПИШИ НИЧЕГО, КРОМЕ JSON. НИКАКИХ ОБЪЯСНЕНИЙ ИЛИ ВВОДНЫХ СЛОВ.
НЕ используй ```json
Текст:
---
{CHUNK}
---"""

resp2 = call(SIMPLE_SYSTEM, SIMPLE_PROMPT)
print(f"Длина ответа: {len(resp2)} символов")
print(f"Ответ:")
print(resp2[:2000])
print()

try:
    data = json.loads(resp2)
    print(f"JSON парсится: ДА, {len(data)} элементов")
except json.JSONDecodeError as e:
    print(f"JSON парсится: НЕТ — {e}")

print()

# Тест 3: Ещё проще — просто найди прямую речь
print("=" * 60)
print("ТЕСТ 3: Минимальный промпт")
print("=" * 60)

resp3 = call(
    "Ты извлекаешь прямую речь из текста. Отвечай JSON.",
    f'Найди все реплики после тире (—) в тексте. Для каждой: {{"speaker":"кто","text":"цитата"}}. Ответь JSON массивом.НЕ ПИШИ НИЧЕГО, КРОМЕ JSON. НИКАКИХ ОБЪЯСНЕНИЙ ИЛИ ВВОДНЫХ СЛОВ. НЕ используй ```json\n\n{CHUNK}'
)
print(f"Длина ответа: {len(resp3)} символов")
print(f"Ответ:")
print(resp3[:2000])

try:
    data = json.loads(resp3)
    print(f"JSON парсится: ДА, {len(data)} элементов")
except json.JSONDecodeError as e:
    print(f"JSON парсится: НЕТ — {e}")
