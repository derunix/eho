#!/usr/bin/env python3
"""
Бенчмарк LLM-моделей для пайплайна извлечения знаний Eho Dataset.

Сравнивает модели по:
  - Качеству extraction (hit rate, precision, noise, parse errors)
  - Внутренним метрикам LLM из ollama native API (prefill/decode tok/s, latency)
  - Аппаратным метрикам GPU (VRAM, util, temp, power, throttling)
  - Offload-диагностике (ollama ps: VRAM/CPU split, offload ratio)

Использование:
  python diagnose_llm.py                          # все модели группы A+B
  python diagnose_llm.py --models gemma4:e4b qwen3:8b   # конкретные модели
  python diagnose_llm.py --group A                # только группа A
  python diagnose_llm.py --group C                # CPU-offload (отдельное ранжирование)
  python diagnose_llm.py --group 4090             # RTX 4090 server suite (10 моделей, Tier S/1/2/3)
  python diagnose_llm.py --group 4090 --cost-per-hour 0.656  # с учётом стоимости Vast.ai (48 GB)
  python diagnose_llm.py --include-failed         # перетестировать ранее провалившиеся
  python diagnose_llm.py --ensemble               # ensemble top-3 после single-model
  python diagnose_llm.py --kv-cache-test          # тест KV cache / flash attention
  python diagnose_llm.py --context-test           # тест num_ctx (2048/4096/8192/16384)
  python diagnose_llm.py --loq-test               # тест Баланс vs Производительность
  python diagnose_llm.py --all                    # все тесты

Результаты: benchmark_results/ (JSON + таблицы)
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
import argparse
import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

BASE_URL = "http://localhost:11434"
RESULTS_DIR = Path("benchmark_results")

# Сколько чанков в среднем на книгу (для экстраполяции ETA пайплайна)
EST_CHUNKS_PER_BOOK = 120

# RTX 4050 Laptop theoretical decode ceiling для 8B Q4 (для decode_efficiency)
THEORETICAL_DECODE_CEILING_TOKS = 55.0

# Стоимость аренды сервера ($/час). Устанавливается через --cost-per-hour.
COST_PER_HOUR_USD = 0.0


# ──────────────────────────────────────────────
# Модели
# ──────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    group: str          # A, B, C, D
    params_total: str   # human-readable
    params_active: str  # for MoE
    vram_est_gb: float  # estimated min VRAM
    notes: str = ""
    previously_failed: bool = False


MODELS: list[ModelSpec] = [
    # Группа A — влезают в 6 GB VRAM целиком
    ModelSpec("gemma4:e4b",    "A", "12B MoE", "4.5B", 5.5, "MoE 128K ctx, Google"),
    ModelSpec("gemma4:e2b",    "A", "5B MoE",  "2.3B", 3.0, "min quality bound"),
    ModelSpec("qwen3:8b",      "A", "8B",      "8B",   5.5, "dense, secondary model"),
    ModelSpec("llama3.1:8b",   "A", "8B",      "8B",   5.5, "classic stable"),
    ModelSpec("mistral:7b",    "A", "7B",      "7B",   5.0, "format discipline check"),
    ModelSpec("qwen3:4b",      "A", "4B",      "4B",   3.0, "retest, prev failed", True),
    ModelSpec("phi4-mini",     "A", "3.8B",    "3.8B", 3.0, "retest, prev failed", True),

    # Группа B — MoE, малое число активных параметров
    ModelSpec("gemma4:26b",    "B", "26B MoE", "3.8B", 6.0, "MoE 128 experts, tight fit"),
    ModelSpec("qwen3-coder:30b","B","30B MoE", "3.3B", 6.0, "coding-focused MoE"),

    # Группа C — CPU offload (64 GB RAM)
    ModelSpec("gemma3:12b",    "C", "12B",     "12B",  8.0, "prev gen, dense, offload"),
    ModelSpec("qwen3:14b",     "C", "14B",     "14B",  9.0, "dense, offload"),
    ModelSpec("gemma4:31b",    "C", "31B",     "31B", 18.0, "dense, heavy offload, quality upper bound"),

    # Группа 4090 — RTX 4090 48 GB VRAM (сервер Vast.ai, $0.656/h)
    # 48 GB позволяет 70B Q4_K_M (~40–43 GB) целиком и 32B Q8_0 (~33–35 GB).
    # Tier S: 70B dense — потолок качества для consumer GPU
    ModelSpec("qwen3:72b",                         "4090", "72B",     "72B",  43.0, "TierS: top open-source, сильный русский, ~15-25 tok/s"),
    ModelSpec("llama3.3:70b",                      "4090", "70B",     "70B",  40.0, "TierS: Meta Llama 3.3, надёжный instruction-follower"),
    ModelSpec("qwen2.5:72b",                       "4090", "72B",     "72B",  43.0, "TierS: предыдущее поколение Qwen, проверенный русский"),
    # Tier 1: 31-32B dense, целиком в GPU (Q4_K_M ~20-22 GB, Q8_0 ~33-35 GB)
    ModelSpec("gemma4:31b",                        "4090", "31B",     "31B",  20.0, "Tier1: #3 Arena AI, ~40-50 tok/s, попробовать Q8 если VRAM не занят 70B"),
    ModelSpec("huihui_ai/gemma-4-abliterated:31b", "4090", "31B",     "31B",  20.0, "Tier1: abliterated gemma4:31b, no safety refusals"),
    ModelSpec("qwen3:32b",                         "4090", "32B",     "32B",  22.0, "Tier1: сильный русский, ~22 GB Q4_K_M"),
    ModelSpec("deepseek-r1:32b",                   "4090", "32B",     "32B",  20.0, "Tier1: reasoning/CoT, тест с think=False для fair compare"),
    # Tier 2: MoE — максимальная скорость, все эксперты в VRAM
    ModelSpec("gemma4:26b",                        "4090", "26B MoE", "3.8B", 16.0, "Tier2: 71% hit rate на ноутбуке, ~80-120 tok/s на 4090"),
    ModelSpec("qwen3-coder:30b",                   "4090", "30B MoE", "3.3B", 18.0, "Tier2: coding MoE, RL-trained reasoning"),
    # Tier 3: lower bound (reference)
    ModelSpec("gemma4:e2b",                        "4090", "5B MoE",  "2.3B",  3.0, "Tier3: ноутбучный чемпион, lower bound, ~80-120+ tok/s"),
]


def get_models(groups: list[str], include_failed: bool, explicit: list[str]) -> list[ModelSpec]:
    if explicit:
        by_name = {m.name: m for m in MODELS}
        result = []
        for name in explicit:
            if name in by_name:
                result.append(by_name[name])
            else:
                result.append(ModelSpec(name, "?", "?", "?", 0, "user-specified"))
        return result

    result = []
    for m in MODELS:
        if m.group not in groups:
            continue
        if m.previously_failed and not include_failed:
            continue
        result.append(m)
    return result


# ──────────────────────────────────────────────
# Тестовые кейсы с ground truth
# ──────────────────────────────────────────────

@dataclass
class TestCase:
    id: str
    book: str
    description: str
    chunk: str
    expected_hits: list[dict]        # facts that MUST be found
    expected_absent: list[str]       # typical noise that should NOT appear
    max_acceptable_facts: int        # upper bound, above = noise
    stress_type: str = "normal"      # normal, dense_dialogue, magic_system, flashback, sparse


TEST_CASES: list[TestCase] = [
    # ── Чужак: диалог Джуффин + Макс, базовый кейс ──
    TestCase(
        id="stranger_01_juffin_intro",
        book="Чужак",
        description="Джуффин вербует Макса в Тайный Сыск. Плотный диалог с фактами о мире.",
        chunk=(
            "— Послушай, Макс, — сказал Джуффин, — у меня для тебя есть одно дельце.\n"
            "— Какое ещё дельце? — спросил я, с подозрением глядя на шефа. — Опять кого-нибудь убивать?\n"
            "— Ну что ты сразу — убивать! — обиделся Джуффин. — Просто нужно заглянуть в одно место "
            "и разобраться, что там творится.\n"
            "Я вздохнул. Когда Джуффин говорит «просто заглянуть», это обычно означает, "
            "что мне предстоит провести ночь в обществе какого-нибудь обезумевшего мага, "
            "который пытается уничтожить мир. Впрочем, камру мне всё равно допить не дали."
        ),
        expected_hits=[
            {"subject": "Джуффин", "fact_contains": "шеф"},
            {"subject": "Макс", "fact_contains": "камр"},
        ],
        expected_absent=[
            "главный герой",
            "место действия",
            "персонаж, который",
        ],
        max_acceptable_facts=8,
    ),

    # ── Чужак: описание Дома у Моста ──
    TestCase(
        id="stranger_02_house_bridge",
        book="Чужак",
        description="Описание Дома у Моста — штаба Тайного Сыска. Много place/custom фактов.",
        chunk=(
            "Дом у Моста оказался именно таким, каким я его себе представлял: мрачноватое "
            "старое здание на левом берегу Хурона. Башня с часами, узкие стрельчатые окна, "
            "тяжёлые двери. Внутри, впрочем, было уютно — камины горели, пахло камрой "
            "и чем-то вкусным с кухни.\n"
            "— Располагайся, — сказал Джуффин. — Это теперь и твой дом. "
            "Малое Тайное Сыскное Войско базируется здесь с тех пор, как Гуриг Восьмой "
            "подарил нам это здание. Раньше тут была резиденция Ордена Семилистника.\n"
            "Сэр Кофа Йох сидел за столом у окна, поглощая содержимое нескольких "
            "тарелок одновременно. Не прерывая трапезы, он приветственно помахал мне рукой."
        ),
        expected_hits=[
            {"subject": "Дом у Моста", "fact_contains": "Тайн"},
            {"subject": "Дом у Моста", "fact_contains": "Хурон"},
            {"subject": "Гуриг", "fact_contains": "подарил"},
            {"subject": "Орден Семилистника", "fact_contains": "резиденци"},
            {"subject": "Кофа Йох", "fact_contains": ""},
        ],
        expected_absent=[
            "место действия",
            "персонаж, который присутствует",
            "это место, где",
        ],
        max_acceptable_facts=12,
    ),

    # ── Мастер ветров и закатов: магическая система ──
    TestCase(
        id="master_winds_03_magic",
        book="Мастер ветров и закатов",
        description="Кодекс Хрембера, ступени Очевидной магии, Холоми. Stress: magic_system.",
        chunk=(
            "— Кодекс Хрембера, — терпеливо объяснял мне Шурф, — устанавливал "
            "ограничения на использование Очевидной магии. Всего существовало "
            "двести тридцать четыре ступени, и каждому жителю Соединённого Королевства "
            "дозволялось использовать магию только до определённой ступени.\n"
            "— А что случалось с теми, кто нарушал? — спросил я.\n"
            "— Их отправляли в Холоми, — ответил Шурф. — Тюрьма-крепость, "
            "в которой невозможно колдовать. Построена ещё во времена Эпохи Орденов. "
            "Впрочем, теперь Кодекс фактически отменён, и в Холоми сидят "
            "только по-настоящему опасные преступники.\n"
            "— Безмолвная речь, — добавил Шурф, — это одна из базовых способностей. "
            "Мысленное общение на расстоянии. Ей владеет почти каждый житель Ехо."
        ),
        expected_hits=[
            {"subject": "Кодекс Хрембера", "fact_contains": "Очевидн"},
            {"subject": "Кодекс Хрембера", "fact_contains": "ступен"},
            {"subject": "Холоми", "fact_contains": "тюрьм"},
            {"subject": "Холоми", "fact_contains": "колдовать"},
            {"subject": "Безмолвная речь", "fact_contains": "общени"},
            {"subject": "Кодекс Хрембера", "fact_contains": "отменён"},
        ],
        expected_absent=[
            "место действия",
            "персонаж, который",
            "видимо",
        ],
        max_acceptable_facts=14,
        stress_type="magic_system",
    ),

    # ── Тёмная сторона: исторический флешбек ──
    TestCase(
        id="dark_side_04_flashback",
        book="Тёмная сторона",
        description="Флешбек о Смутных Временах и войне Орденов. Stress: flashback, time_scope.",
        chunk=(
            "Сэр Манга Мелифаро — не путать с моим коллегой Мелифаро — "
            "рассказывал за ужином о Смутных Временах.\n"
            "— В те годы Ордена воевали друг с другом, — говорил он, задумчиво "
            "крутя кружку с камрой. — Орден Семилистника и Орден Потаённой Травы "
            "объединились против Ордена Водяной Вороны. Лойсо Пондохва, "
            "Великий Магистр Водяной Вороны, был, пожалуй, самым могущественным магом "
            "той эпохи. Он один мог противостоять нескольким Орденам сразу.\n"
            "— А Нуфлин? — спросил я.\n"
            "— Нуфлин Мони Мах, Великий Магистр Семилистника, был хитрее. "
            "Не сильнее, а именно хитрее. Он сумел заключить союз с Кеттарийскими Охотниками. "
            "Именно они в итоге решили исход войны."
        ),
        expected_hits=[
            {"subject": "Смутные Времена", "fact_contains": "Орден"},
            {"subject": "Лойсо Пондохва", "fact_contains": "Водяной Вороны"},
            {"subject": "Лойсо Пондохва", "fact_contains": "могущественн"},
            {"subject": "Нуфлин Мони Мах", "fact_contains": "Семилистник"},
            {"subject": "Нуфлин Мони Мах", "fact_contains": "Кеттарийски"},
            {"subject": "Манга Мелифаро", "fact_contains": ""},
        ],
        expected_absent=[
            "главный герой",
            "упоминается в контексте",
            "место действия",
        ],
        max_acceptable_facts=14,
        stress_type="flashback",
    ),

    # ── Мастер ветров: плотный диалог, мало фактов ──
    TestCase(
        id="dreams_05_dense_dialogue",
        book="Мастер ветров и закатов",
        description="Болтовня Макса и Мелифаро. Stress: dense_dialogue — мало фактов ожидаемо.",
        chunk=(
            "— Ты опять не спал? — Мелифаро уставился на меня с притворным ужасом.\n"
            "— Не-а, — я зевнул. — Зато я видел потрясающий сон.\n"
            "— Про еду?\n"
            "— А про что же ещё? Мне снилась такая пастила из Гуппарока, "
            "что я чуть язык не проглотил. Даже во сне.\n"
            "— Ты безнадёжен, — вздохнул Мелифаро. — Ладно, пойдём в «Обжору Бунбу», "
            "пока ты не начал грызть мебель.\n"
            "— Лучшее предложение за сегодня, — согласился я.\n"
            "Мы вышли из Дома у Моста и направились по улице Медных Горшков. "
            "Утро было тёплым и ленивым, как кот на подоконнике."
        ),
        expected_hits=[
            {"subject": "Обжора Бунба", "fact_contains": ""},
            {"subject": "улица Медных Горшков", "fact_contains": ""},
        ],
        expected_absent=[
            "Мелифаро спросил",
            "Макс ответил",
            "обед в",
            "утренний",
        ],
        max_acceptable_facts=6,
        stress_type="dense_dialogue",
    ),

    # ── Sparse: пейзаж / внутренний монолог, 0-2 факта ──
    TestCase(
        id="dreams_06_landscape",
        book="Мастер ветров и закатов",
        description="Закат над Ехо, внутренний монолог. Stress: sparse — 0-2 факта.",
        chunk=(
            "Я сидел в своей башне на крыше Мохнатого Дома и смотрел на закат. "
            "Небо над Ехо было невозможного цвета — что-то среднее между "
            "перезревшим абрикосом и расплавленным золотом. "
            "Такие закаты случаются только здесь, я точно знаю.\n"
            "Внизу, на улицах, зажигались первые фонари. Кто-то громко смеялся "
            "в трактире на углу. Откуда-то тянуло запахом свежей выпечки.\n"
            "Я подумал, что, пожалуй, счастлив. Не в том смысле, что у меня "
            "всё хорошо, — в моей жизни хватало проблем. А в том глубоком, "
            "неизлечимом смысле, когда ты точно знаешь, что находишься "
            "именно там, где должен быть."
        ),
        expected_hits=[
            {"subject": "Мохнатый Дом", "fact_contains": "башн"},
        ],
        expected_absent=[
            "место действия",
            "описание заката",
            "настроение",
            "Макс — персонаж",
        ],
        max_acceptable_facts=4,
        stress_type="sparse",
    ),

    # ── Чужак: характеристика Шурфа ──
    TestCase(
        id="stranger_07_shurf",
        book="Чужак",
        description="Шурф: Перчатки Смерти, Безумный Рыбник, Орден Дырявой Чаши.",
        chunk=(
            "Сэр Шурф Лонли-Локли был, пожалуй, самым необычным из моих коллег. "
            "Мастер Пресекающий Ненужные Жизни, как гласил его официальный титул, "
            "говорил подчёркнуто правильно и формально, никогда не шутил "
            "и носил Перчатки Смерти — страшное оружие, которое он сам изготовил "
            "из рук убитых им магистров Ордена Ледяной Руки.\n"
            "Когда-то Шурф был Мастером Рыбником в Ордене Дырявой Чаши. "
            "Его там называли Безумным Рыбником — он выпил воду из всех "
            "орденских аквариумов и получил силу, предназначенную для "
            "шестисот человек. Несколько лет он терроризировал Ехо, "
            "не контролируя себя. Сэр Джуффин Халли спас его, "
            "отправив в Хумгат, — после чего родился нынешний сдержанный Шурф.\n"
            "Был женат на Хельне, поэтессе."
        ),
        expected_hits=[
            {"subject": "Шурф Лонли-Локли", "fact_contains": "Пресекающий"},
            {"subject": "Шурф Лонли-Локли", "fact_contains": "Перчатки Смерти"},
            {"subject": "Шурф Лонли-Локли", "fact_contains": "Ледяной Руки"},
            {"subject": "Шурф Лонли-Локли", "fact_contains": "Рыбник"},
            {"subject": "Шурф Лонли-Локли", "fact_contains": "Дырявой Чаши"},
            {"subject": "Шурф Лонли-Локли", "fact_contains": "Хумгат"},
            {"subject": "Шурф Лонли-Локли", "fact_contains": "Хельн"},
        ],
        expected_absent=[
            "персонаж, который",
            "упоминается в контексте",
        ],
        max_acceptable_facts=12,
    ),

    # ── Мастер ветров: быт мира ──
    TestCase(
        id="master_winds_08_customs",
        book="Мастер ветров и закатов",
        description="Быт Ехо: камра, амобилеры, лоохи. custom-факты.",
        chunk=(
            "Жизнь в Ехо имеет свои маленькие прелести. Утро начинается с камры — "
            "горького горячего напитка, который здесь пьют все и помногу. "
            "Камра бодрит лучше любого кофе, которого в этом мире, кстати, нет. "
            "Как нет электричества, телефонов и сигарет.\n"
            "По улицам носятся амобилеры — экипажи на магических кристаллах. "
            "Ни бензина, ни мотора — чистая магия. Я, кстати, вожу свой амобилер "
            "как бешеный, пугая половину города.\n"
            "Одежда здесь называется лоохи — что-то вроде длинной мантии. "
            "Мелифаро меняет их по несколько штук в день, каждый раз "
            "ярче предыдущей. Я же ношу что попало."
        ),
        expected_hits=[
            {"subject": "камра", "fact_contains": "горяч"},
            {"subject": "камра", "fact_contains": ""},
            {"subject": "амобилер", "fact_contains": "кристалл"},
            {"subject": "лоохи", "fact_contains": "одежд"},
        ],
        expected_absent=[
            "место действия",
            "это напиток, который",
            "Макс — персонаж",
        ],
        max_acceptable_facts=10,
    ),
]


# ──────────────────────────────────────────────
# Line-protocol prompt (из extract_dialogues.py)
# ──────────────────────────────────────────────

KNOWLEDGE_SYSTEM = """Ты — помощник для извлечения фактов из книг Макса Фрая.
Извлекаешь структурированные знания о мире Ехо, персонажах и событиях.
Отвечай СТРОГО в line-protocol формате. Никакого JSON, markdown или пояснений."""

KNOWLEDGE_LINE_PROMPT = """Из PRIMARY CHUNK извлеки автономные факты для базы знаний мира Ехо.

РЕЖИМ: WORLD_FACTS
Ищи только именованные сущности и устойчивые знания: кто кто, что где, как устроено, какие есть роли,
отношения, свойства мест, правила мира, магия, устойчивые предметы и институты.
PRIORITY: precision first. Лучше вернуть меньше фактов, чем добавить шум.

ФОРМАТ ОТВЕТА:
Верни только строки line-протокола, ОДИН ФАКТ = ОДНА СТРОКА.
Строгий формат каждой строки:
category=... | subject=... | fact=... | time_scope=...

Пример хороших строк:
category=character | subject=Кимпа | fact=Кимпа был гонщиком, прежде чем стать дворецким Джуффина. | time_scope=past
category=place | subject=Дом у Моста | fact=Дом у Моста служит штабом Тайного Сыска. | time_scope=timeless
category=place | subject=Мохнатый Дом | fact=Мохнатый Дом — дом Макса с башней на крыше. | time_scope=timeless
category=place | subject=Обжора Бунба | fact=Обжора Бунба — известная забегаловка Ехо. | time_scope=timeless
category=magic | subject=Безмолвная речь | fact=Безмолвная речь является обычным способом общения на расстоянии. | time_scope=timeless
category=history | subject=Кодекс Хрембера | fact=Кодекс Хрембера больше не действует. | time_scope=ended
category=place | subject=Холоми | fact=В Холоми невозможно колдовать. | time_scope=timeless
category=event | subject=Макс | fact=Макс впервые прибывает в Ехо. | time_scope=past
category=custom | subject=камра | fact=Камра — горький горячий напиток, популярный в Ехо. | time_scope=timeless

Категории:
- character
- place
- magic
- history
- event
- creature
- custom

Time scope:
- timeless
- past
- current
- change
- ended
- unclear

Главные правила:
- PRECISION FIRST: лучше вернуть меньше строк, чем мусор.
- Используй только факты, явно подтверждённые PRIMARY CHUNK.
- Subject должен быть полным и естественным. Не обрезай названия.
- Перед завершением ответа перепроверь короткие named facts, которые модели часто теряют:
  дома и трактиры, ордена и институты, магические термины, напитки, одежду, транспорт,
  а также relation/status facts вроде `X женат на Y`, `X был ...`, `X больше не действует`, `в X невозможно ...`.
- Не пропускай короткие, но полезные факты только потому, что они выражены в 1-2 именованных словах:
  `Мохнатый Дом`, `Обжора Бунба`, `Холоми`, `Кодекс Хрембера`, `камра`, `лоохи`, `амобилер`.
- Не пиши summary-абстракции вроде «курс адаптации», «место действия».
- Не пиши сценические пересказы и реплики-пересказы.
- Не используй символ | внутри значений полей.
- За один проход верни не более 12 новых строк.

Нельзя:
- никакого JSON, TOML, YAML, markdown, нумерации и пояснений;
- никаких placeholder-subject вроде «место», «персонаж», «улица»;
- никаких псевдоопределений вроде «Это место, где...», «Это персонаж, который...»;
- никаких фраз с неопределённостью: «видимо», «возможно», «упоминается в контексте».

Если подходящих фактов нет, верни пустой ответ без пояснений.

Материал:
---
[PRIMARY CHUNK]
{chunk}
[/PRIMARY CHUNK]
---
"""


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────

def parse_line_protocol(response: str) -> list[dict]:
    """Парсит line-protocol ответ модели в список фактов."""
    items = []
    seen = set()

    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*•]\s*", "", line)
        line = re.sub(r"^\s*\d+[.)]\s*", "", line)
        if not line:
            continue
        if line.startswith("#") or (line.startswith("[") and line.endswith("]")):
            continue
        if line.startswith("{") or line.startswith("["):
            continue

        fields = {}
        for part in line.split("|"):
            part = part.strip()
            if "=" in part:
                key, _, val = part.partition("=")
                fields[key.strip().lower()] = val.strip()

        if not fields.get("category") or not fields.get("subject") or not fields.get("fact"):
            continue

        item = {
            "category": fields["category"].strip(),
            "subject": fields["subject"].strip(),
            "fact": fields["fact"].strip(),
            "time_scope": fields.get("time_scope", "unclear").strip(),
        }

        key = f"{item['category']}|{item['subject']}|{item['fact']}".lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)

    return items


def count_parse_errors(response: str) -> int:
    """Считает строки, которые похожи на line-protocol, но не распарсились."""
    errors = 0
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*•]\s*", "", line)
        line = re.sub(r"^\s*\d+[.)]\s*", "", line)
        if not line:
            continue
        if line.startswith("#") or (line.startswith("[") and line.endswith("]")):
            continue
        if "|" in line:
            fields = {}
            for part in line.split("|"):
                part = part.strip()
                if "=" in part:
                    key, _, val = part.partition("=")
                    fields[key.strip().lower()] = val.strip()
            if not fields.get("category") or not fields.get("fact"):
                errors += 1
        elif line and not line.startswith("{"):
            errors += 1
    return errors


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_hits(facts: list[dict], expected: list[dict]) -> tuple[list[bool], list[str]]:
    hits = []
    descriptions = []
    for exp in expected:
        subj = exp.get("subject", "").lower()
        contains = exp.get("fact_contains", "").lower()
        found = False
        for f in facts:
            f_subj = f.get("subject", "").lower()
            f_fact = f.get("fact", "").lower()
            if subj and subj not in f_subj:
                continue
            if contains and contains not in f_fact:
                continue
            found = True
            break
        hits.append(found)
        desc = f"subject~'{exp.get('subject', '')}'"
        if contains:
            desc += f" fact~'{exp.get('fact_contains', '')}'"
        descriptions.append(desc)
    return hits, descriptions


def evaluate_noise(facts: list[dict], expected_absent: list[str]) -> list[str]:
    noise_found = []
    for marker in expected_absent:
        marker_low = marker.lower()
        for f in facts:
            full = f"{f.get('subject', '')} {f.get('fact', '')}".lower()
            if marker_low in full:
                noise_found.append(marker)
                break
    return noise_found


GENERIC_NOISE_PATTERNS = [
    r"персонаж,?\s*котор",
    r"место действия",
    r"это место,?\s*где",
    r"это персонаж",
    r"упоминается в контексте",
    r"видимо",
    r"возможно",
    r"вероятно",
    r"главный герой",
]


def evaluate_precision(facts: list[dict], expected_absent: list[str]) -> float:
    if not facts:
        return 1.0
    noise_count = len(evaluate_noise(facts, expected_absent))
    for f in facts:
        full = f"{f.get('subject', '')} {f.get('fact', '')}".lower()
        for pat in GENERIC_NOISE_PATTERNS:
            if re.search(pat, full):
                noise_count += 1
                break
    return max(0.0, 1.0 - noise_count / len(facts))


# ──────────────────────────────────────────────
# Ollama Native API
# ──────────────────────────────────────────────

def ollama_api(endpoint: str, payload: dict = None, method: str = "GET",
               timeout: int = 300) -> dict:
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def ollama_chat(model: str, system: str, user: str,
                timeout: int = 300, extra_options: dict = None) -> dict:
    """Native /api/chat — возвращает полный response с метриками."""
    options = {"temperature": 0.1, "num_predict": 1800}
    if extra_options:
        options.update(extra_options)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": options,
    }
    return ollama_api("/api/chat", payload, "POST", timeout)


def ollama_list_models() -> list[str]:
    try:
        data = ollama_api("/api/tags")
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ollama_pull(model: str) -> bool:
    # Timeout: 70B models ~43GB can take 30-60 min on slow connections
    timeout = 3600
    try:
        subprocess.run(
            ["ollama", "pull", model],
            check=True,
            timeout=timeout,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] Failed to pull {model}: exit status {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Pull timed out after {timeout}s for {model}")
        return False
    except Exception as e:
        print(f"  [WARN] Failed to pull {model}: {e}")
        return False


def ollama_stop(model: str):
    """Выгрузить модель (keep_alive=0)."""
    try:
        payload = {"model": model, "messages": [], "stream": False, "keep_alive": 0}
        ollama_api("/api/chat", payload, "POST", timeout=30)
    except Exception:
        pass


def ollama_ps() -> dict:
    """Читает /api/ps — info о загруженных моделях (VRAM/CPU split)."""
    try:
        data = ollama_api("/api/ps")
        models = data.get("models", [])
        if not models:
            return {}
        m = models[0]
        size_vram = m.get("size_vram", 0)
        size_total = m.get("size", 0)
        return {
            "model_name": m.get("name", ""),
            "size_vram": size_vram,
            "size_total": size_total,
            "offload_ratio": (size_vram / size_total) if size_total > 0 else 0.0,
            "size_vram_mb": size_vram / (1024 * 1024),
            "size_total_mb": size_total / (1024 * 1024),
            "quantization": m.get("details", {}).get("quantization_level", "unknown"),
            "parameter_size": m.get("details", {}).get("parameter_size", "unknown"),
            "family": m.get("details", {}).get("family", "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}


def extract_llm_metrics(resp: dict) -> dict:
    """Извлекает внутренние метрики из native ollama API response."""
    total_ns = resp.get("total_duration", 0)
    load_ns = resp.get("load_duration", 0)
    prompt_eval_count = resp.get("prompt_eval_count", 0)
    prompt_eval_ns = resp.get("prompt_eval_duration", 0)
    eval_count = resp.get("eval_count", 0)
    eval_ns = resp.get("eval_duration", 0)

    prompt_eval_rate = (
        (prompt_eval_count / (prompt_eval_ns / 1e9)) if prompt_eval_ns > 0 else 0.0
    )
    eval_rate = (
        (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0
    )
    overhead_ns = total_ns - prompt_eval_ns - eval_ns - load_ns
    overhead_ms = max(0.0, overhead_ns / 1e6)
    effective_time_sec = total_ns / 1e9 if total_ns > 0 else 0.0
    decode_efficiency = eval_rate / THEORETICAL_DECODE_CEILING_TOKS if eval_rate > 0 else 0.0

    return {
        "total_duration_ns": total_ns,
        "load_duration_ns": load_ns,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_ns": prompt_eval_ns,
        "prompt_eval_rate": round(prompt_eval_rate, 2),
        "eval_count": eval_count,
        "eval_duration_ns": eval_ns,
        "eval_rate": round(eval_rate, 2),
        "overhead_ms": round(overhead_ms, 2),
        "total_tokens": prompt_eval_count + eval_count,
        "effective_time_sec": round(effective_time_sec, 3),
        "decode_efficiency": round(decode_efficiency, 3),
    }


def ensure_ollama_running():
    """Проверяет что ollama запущен. Если нет — запускает с нужными env vars и ждёт готовности."""
    # Check if already running
    try:
        ollama_api("/api/tags", timeout=5)
        print("ollama: already running")
        return
    except Exception:
        pass

    print("ollama: not running, starting with FLASH_ATTENTION=1 KV_CACHE=q8_0 ...")
    env = os.environ.copy()
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
    env["OLLAMA_NUM_PARALLEL"] = "1"
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"

    log_path = Path("/tmp/ollama_serve.log")
    log_fh = open(log_path, "w")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    except FileNotFoundError:
        print("ERROR: 'ollama' not found in PATH. Install: curl -fsSL https://ollama.com/install.sh | sh")
        sys.exit(1)

    # Wait up to 30s for ollama to accept connections
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(1)
        try:
            ollama_api("/api/tags", timeout=3)
            print(f"ollama: ready (log: {log_path})")
            return
        except Exception:
            pass

    print(f"ERROR: ollama did not start within 30s. Check {log_path}")
    sys.exit(1)


def warmup_model(model: str):
    try:
        ollama_chat(model, "Ответь одним словом.", "Привет.", timeout=120)
    except Exception as e:
        print(f"  [WARN] Warmup failed for {model}: {e}")


# ──────────────────────────────────────────────
# GPU metrics collector (background nvidia-smi)
# ──────────────────────────────────────────────

class GpuMetricsCollector:
    """Фоновый сбор GPU-метрик через nvidia-smi каждые 2 секунды."""

    QUERY = (
        "timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,"
        "memory.total,temperature.gpu,power.draw,power.limit,"
        "clocks.current.graphics,clocks.current.memory"
    )

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._output_path: Optional[Path] = None
        self._samples: list[dict] = []

    def start(self, label: str):
        RESULTS_DIR.mkdir(exist_ok=True)
        self._output_path = RESULTS_DIR / f"gpu_metrics_{label}.csv"
        self._samples = []
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu={self.QUERY}",
                    "--format=csv,nounits",
                    "-l", "2",
                ],
                stdout=open(self._output_path, "w"),
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self._process = None

    def stop(self) -> dict:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        return self._parse_csv()

    def _parse_csv(self) -> dict:
        if not self._output_path or not self._output_path.exists():
            return {}
        try:
            with open(self._output_path, "r") as f:
                lines = f.readlines()
            if len(lines) < 2:
                return {}

            # Parse header
            header = [h.strip().lower().replace(" ", "_").replace(".", "_") for h in lines[0].split(",")]
            rows = []
            for line in lines[1:]:
                vals = line.strip().split(",")
                if len(vals) != len(header):
                    continue
                row = {}
                for h, v in zip(header, vals):
                    v = v.strip()
                    try:
                        row[h] = float(v)
                    except ValueError:
                        row[h] = v
                rows.append(row)

            if not rows:
                return {}

            def safe_stat(key, func):
                vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
                return func(vals) if vals else None

            vram_used_vals = [r.get("memory_used_[mib]", r.get("memory_used", 0)) for r in rows
                             if isinstance(r.get("memory_used_[mib]", r.get("memory_used")), (int, float))]
            vram_free_vals = [r.get("memory_free_[mib]", r.get("memory_free", 0)) for r in rows
                             if isinstance(r.get("memory_free_[mib]", r.get("memory_free")), (int, float))]
            gpu_util_vals = [r.get("utilization_gpu_[%]", r.get("utilization_gpu", 0)) for r in rows
                           if isinstance(r.get("utilization_gpu_[%]", r.get("utilization_gpu")), (int, float))]
            mem_util_vals = [r.get("utilization_memory_[%]", r.get("utilization_memory", 0)) for r in rows
                           if isinstance(r.get("utilization_memory_[%]", r.get("utilization_memory")), (int, float))]
            temp_vals = [r.get("temperature_gpu", 0) for r in rows
                        if isinstance(r.get("temperature_gpu"), (int, float))]
            power_draw_vals = [r.get("power_draw_[w]", r.get("power_draw", 0)) for r in rows
                              if isinstance(r.get("power_draw_[w]", r.get("power_draw")), (int, float))]
            power_limit_vals = [r.get("power_limit_[w]", r.get("power_limit", 0)) for r in rows
                               if isinstance(r.get("power_limit_[w]", r.get("power_limit")), (int, float))]
            clock_gpu_vals = [r.get("clocks_current_graphics_[mhz]", r.get("clocks_current_graphics", 0)) for r in rows
                             if isinstance(r.get("clocks_current_graphics_[mhz]", r.get("clocks_current_graphics")), (int, float))]
            clock_mem_vals = [r.get("clocks_current_memory_[mhz]", r.get("clocks_current_memory", 0)) for r in rows
                             if isinstance(r.get("clocks_current_memory_[mhz]", r.get("clocks_current_memory")), (int, float))]

            temp_max = max(temp_vals) if temp_vals else 0
            power_max = max(power_draw_vals) if power_draw_vals else 0
            power_limit = max(power_limit_vals) if power_limit_vals else 999

            is_throttling = (temp_max > 87) or (power_max >= power_limit * 0.95)

            return {
                "vram_used_peak_mb": max(vram_used_vals) if vram_used_vals else None,
                "vram_used_avg_mb": round(statistics.mean(vram_used_vals), 1) if vram_used_vals else None,
                "vram_free_min_mb": min(vram_free_vals) if vram_free_vals else None,
                "gpu_util_avg": round(statistics.mean(gpu_util_vals), 1) if gpu_util_vals else None,
                "gpu_util_max": max(gpu_util_vals) if gpu_util_vals else None,
                "mem_util_avg": round(statistics.mean(mem_util_vals), 1) if mem_util_vals else None,
                "temp_avg_c": round(statistics.mean(temp_vals), 1) if temp_vals else None,
                "temp_max_c": temp_max if temp_vals else None,
                "power_avg_w": round(statistics.mean(power_draw_vals), 1) if power_draw_vals else None,
                "power_max_w": power_max if power_draw_vals else None,
                "power_limit_w": power_limit if power_limit_vals else None,
                "gpu_clock_avg_mhz": round(statistics.mean(clock_gpu_vals), 0) if clock_gpu_vals else None,
                "vram_clock_avg_mhz": round(statistics.mean(clock_mem_vals), 0) if clock_mem_vals else None,
                "is_throttling": is_throttling,
                "samples": len(rows),
            }
        except Exception as e:
            return {"parse_error": str(e)}


def get_vram_baseline() -> Optional[float]:
    """Snapshot VRAM до загрузки модели (сколько занято системой/браузером)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def get_vram_snapshot() -> Optional[float]:
    """Текущее потребление VRAM в MB."""
    return get_vram_baseline()


# ──────────────────────────────────────────────
# CPU/RAM metrics (for Group C offload models)
# ──────────────────────────────────────────────

class CpuRamCollector:
    """Фоновый сбор CPU/RAM через psutil (для моделей с оффлоадом)."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._samples: list[dict] = []

    def start(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            return
        self._running = True
        self._samples = []
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def _collect(self):
        import psutil
        while self._running:
            mem = psutil.virtual_memory()
            self._samples.append({
                "ts": time.time(),
                "cpu_percent": psutil.cpu_percent(),
                "ram_used_gb": round(mem.used / 1e9, 2),
                "ram_available_gb": round(mem.available / 1e9, 2),
            })
            time.sleep(2)

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if not self._samples:
            return {}
        cpu_vals = [s["cpu_percent"] for s in self._samples]
        ram_vals = [s["ram_used_gb"] for s in self._samples]
        return {
            "cpu_util_avg": round(statistics.mean(cpu_vals), 1),
            "ram_used_peak_gb": round(max(ram_vals), 2),
            "ram_used_avg_gb": round(statistics.mean(ram_vals), 2),
            "samples": len(self._samples),
        }


# ──────────────────────────────────────────────
# Bottleneck diagnosis
# ──────────────────────────────────────────────

def diagnose_bottleneck(model_info: dict, gpu_metrics: dict) -> str:
    offload_ratio = model_info.get("offload_ratio", 1.0)
    if offload_ratio < 1.0:
        return "CPU_OFFLOAD"

    gpu_util = gpu_metrics.get("gpu_util_avg", 100)
    if gpu_util is not None and gpu_util < 50:
        return "GPU_UNDERUTILIZED"

    temp_max = gpu_metrics.get("temp_max_c", 0)
    if temp_max is not None and temp_max > 87:
        return "THERMAL_THROTTLING"

    power_max = gpu_metrics.get("power_max_w", 0)
    power_limit = gpu_metrics.get("power_limit_w", 999)
    if power_max and power_limit and power_max >= power_limit * 0.95:
        return "TDP_THROTTLING"

    mem_util = gpu_metrics.get("mem_util_avg", 0)
    if mem_util is not None and mem_util > 90:
        return "MEMORY_BANDWIDTH"

    vram_free = gpu_metrics.get("vram_free_min_mb", 9999)
    if vram_free is not None and vram_free < 200:
        return "VRAM_NEAR_OOM"

    return "COMPUTE_BOUND"


# ──────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    model: str
    # Quality
    facts: list[dict]
    raw_response: str
    hit_mask: list[bool]
    hit_descriptions: list[str]
    noise_markers: list[str]
    hit_rate: float
    precision: float
    noise_rate: float
    parse_errors: int
    total_facts: int
    # LLM metrics
    llm_metrics: dict
    # Hardware snapshot
    vram_usage_mb: Optional[float]
    # Wall clock
    time_sec: float


@dataclass
class ModelResult:
    model: str
    spec: ModelSpec
    cases: list[CaseResult]
    model_info: dict = field(default_factory=dict)
    gpu_metrics: dict = field(default_factory=dict)
    cpu_metrics: dict = field(default_factory=dict)
    bottleneck: str = "UNKNOWN"
    vram_baseline_mb: float = 0.0
    # Aggregates (computed)
    avg_hit_rate: float = 0.0
    avg_precision: float = 0.0
    avg_time_sec: float = 0.0
    format_reliability: float = 0.0
    avg_eval_rate: float = 0.0
    avg_prompt_eval_rate: float = 0.0
    avg_total_tokens: float = 0.0
    estimated_pipeline_hours: float = 0.0
    estimated_pipeline_hours_ensemble: float = 0.0
    quality_speed_score: float = 0.0
    quality_score: float = 0.0
    efficiency_score: float = 0.0
    cost_per_book_usd: float = 0.0
    cost_per_corpus_usd: float = 0.0

    def compute_aggregates(self):
        if not self.cases:
            return
        self.avg_hit_rate = statistics.mean(c.hit_rate for c in self.cases)
        self.avg_precision = statistics.mean(c.precision for c in self.cases)
        self.avg_time_sec = statistics.mean(c.time_sec for c in self.cases)
        zero_errors = sum(1 for c in self.cases if c.parse_errors == 0)
        self.format_reliability = zero_errors / len(self.cases)

        eval_rates = [c.llm_metrics.get("eval_rate", 0) for c in self.cases if c.llm_metrics.get("eval_rate")]
        self.avg_eval_rate = statistics.mean(eval_rates) if eval_rates else 0.0

        prefill_rates = [c.llm_metrics.get("prompt_eval_rate", 0) for c in self.cases if c.llm_metrics.get("prompt_eval_rate")]
        self.avg_prompt_eval_rate = statistics.mean(prefill_rates) if prefill_rates else 0.0

        total_toks = [c.llm_metrics.get("total_tokens", 0) for c in self.cases]
        self.avg_total_tokens = statistics.mean(total_toks) if total_toks else 0.0

        self.estimated_pipeline_hours = self.avg_time_sec * EST_CHUNKS_PER_BOOK * 32 / 3600
        self.estimated_pipeline_hours_ensemble = self.estimated_pipeline_hours * 1.3

        if self.avg_time_sec > 0:
            self.quality_speed_score = self.avg_hit_rate * self.avg_precision / self.avg_time_sec
        self.quality_score = self.avg_hit_rate * self.avg_precision * self.format_reliability

        vram_peak = self.gpu_metrics.get("vram_used_peak_mb", 0)
        if vram_peak and vram_peak > 0 and self.quality_speed_score > 0:
            self.efficiency_score = self.quality_speed_score / vram_peak * 1000

        if COST_PER_HOUR_USD > 0:
            self.cost_per_book_usd = (self.avg_time_sec * EST_CHUNKS_PER_BOOK / 3600) * COST_PER_HOUR_USD
            self.cost_per_corpus_usd = self.cost_per_book_usd * 32


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def run_case(model: str, case: TestCase, extra_options: dict = None) -> CaseResult:
    prompt = KNOWLEDGE_LINE_PROMPT.format(chunk=case.chunk)

    vram_before = get_vram_snapshot()
    t0 = time.monotonic()

    try:
        resp_data = ollama_chat(model, KNOWLEDGE_SYSTEM, prompt,
                                timeout=300, extra_options=extra_options)
    except Exception as e:
        return CaseResult(
            case_id=case.id, model=model, facts=[], raw_response=f"ERROR: {e}",
            hit_mask=[], hit_descriptions=[], noise_markers=[],
            hit_rate=0.0, precision=0.0, noise_rate=1.0,
            parse_errors=1, total_facts=0, llm_metrics={},
            vram_usage_mb=vram_before, time_sec=time.monotonic() - t0,
        )

    elapsed = time.monotonic() - t0
    vram_after = get_vram_snapshot()
    content = resp_data.get("message", {}).get("content", "")
    llm = extract_llm_metrics(resp_data)

    facts = parse_line_protocol(content)
    p_errors = count_parse_errors(content)

    hit_mask, hit_desc = evaluate_hits(facts, case.expected_hits)
    noise_markers = evaluate_noise(facts, case.expected_absent)
    hit_rate = sum(hit_mask) / len(hit_mask) if hit_mask else 0.0
    precision = evaluate_precision(facts, case.expected_absent)
    noise_rate = len(noise_markers) / len(case.expected_absent) if case.expected_absent else 0.0

    return CaseResult(
        case_id=case.id, model=model, facts=facts, raw_response=content,
        hit_mask=hit_mask, hit_descriptions=hit_desc, noise_markers=noise_markers,
        hit_rate=hit_rate, precision=precision, noise_rate=noise_rate,
        parse_errors=p_errors, total_facts=len(facts), llm_metrics=llm,
        vram_usage_mb=vram_after, time_sec=elapsed,
    )


def run_model(spec: ModelSpec, cases: list[TestCase],
              vram_baseline: float = 0.0) -> ModelResult:
    model = spec.name
    print(f"\n{'='*60}")
    print(f"MODEL: {model} ({spec.params_total}, active: {spec.params_active})")
    print(f"{'='*60}")

    # Check availability
    available = ollama_list_models()
    if not any(model in m for m in available):
        print(f"  Pulling {model}...")
        if not ollama_pull(model):
            print(f"  SKIP: model not available")
            return ModelResult(model=model, spec=spec, cases=[])

    # Start GPU metrics collection
    gpu_collector = GpuMetricsCollector()
    gpu_collector.start(model.replace(":", "_").replace("/", "_"))

    # Start CPU/RAM collection for Group C
    cpu_collector = CpuRamCollector()
    if spec.group == "C":
        cpu_collector.start()

    # Warmup (loads model into memory)
    print(f"  Warming up...", flush=True)
    warmup_model(model)

    # Get model info from ollama ps
    model_info = ollama_ps()
    if model_info:
        offload = model_info.get("offload_ratio", 0)
        vram_mb = model_info.get("size_vram_mb", 0)
        total_mb = model_info.get("size_total_mb", 0)
        quant = model_info.get("quantization", "?")
        print(f"  Model info: {total_mb:.0f}MB total, {vram_mb:.0f}MB VRAM, "
              f"offload={offload:.2f}, quant={quant}")

    # Run cases
    results = []
    for i, case in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {case.id} ({case.stress_type})...", end=" ", flush=True)
        cr = run_case(model, case)
        results.append(cr)

        hits = sum(cr.hit_mask)
        total_exp = len(cr.hit_mask)
        eval_rate = cr.llm_metrics.get("eval_rate", 0)
        prefill_rate = cr.llm_metrics.get("prompt_eval_rate", 0)
        print(
            f"hits={hits}/{total_exp} "
            f"facts={cr.total_facts} "
            f"prec={cr.precision:.0%} "
            f"err={cr.parse_errors} "
            f"time={cr.time_sec:.1f}s "
            f"decode={eval_rate:.1f}tok/s "
            f"prefill={prefill_rate:.0f}tok/s",
            flush=True,
        )

    # Stop collectors
    gpu_metrics = gpu_collector.stop()
    cpu_metrics = cpu_collector.stop()

    # Build result
    mr = ModelResult(
        model=model, spec=spec, cases=results,
        model_info=model_info, gpu_metrics=gpu_metrics,
        cpu_metrics=cpu_metrics, vram_baseline_mb=vram_baseline,
    )
    mr.compute_aggregates()
    mr.bottleneck = diagnose_bottleneck(model_info, gpu_metrics)

    print(f"\n  SUMMARY: hit={mr.avg_hit_rate:.0%} prec={mr.avg_precision:.0%} "
          f"fmt={mr.format_reliability:.0%} "
          f"decode={mr.avg_eval_rate:.1f}tok/s "
          f"time={mr.avg_time_sec:.1f}s "
          f"ETA={mr.estimated_pipeline_hours:.0f}h "
          f"QxS={mr.quality_speed_score:.4f} "
          f"bottleneck={mr.bottleneck}")

    return mr


# ──────────────────────────────────────────────
# Ensemble testing
# ──────────────────────────────────────────────

def run_ensemble_test(top_models: list[ModelResult], cases: list[TestCase]):
    hard_cases = []
    for case in cases:
        best_hr = 0.0
        for mr in top_models:
            for cr in mr.cases:
                if cr.case_id == case.id:
                    best_hr = max(best_hr, cr.hit_rate)
        if best_hr < 0.8:
            hard_cases.append(case)

    if not hard_cases:
        hard_cases = [c for c in cases if c.stress_type != "sparse"][:3]

    if len(top_models) < 2:
        print("\n[ENSEMBLE] Need at least 2 models, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"ENSEMBLE TEST on {len(hard_cases)} hard cases")
    print(f"{'='*60}")

    for i in range(min(len(top_models), 3)):
        for j in range(min(len(top_models), 3)):
            if i == j:
                continue
            primary = top_models[i]
            secondary = top_models[j]
            print(f"\n  Combo: {primary.model} (primary) + {secondary.model} (secondary)")

            for case in hard_cases:
                cr1 = run_case(primary.model, case)
                cr2 = run_case(secondary.model, case)

                merged = cr1.facts[:]
                seen = {f"{f['subject']}|{f['fact']}".lower() for f in merged}
                for f in cr2.facts:
                    key = f"{f['subject']}|{f['fact']}".lower()
                    if key not in seen:
                        merged.append(f)
                        seen.add(key)

                hit_mask, _ = evaluate_hits(merged, case.expected_hits)
                combo_hr = sum(hit_mask) / len(hit_mask) if hit_mask else 0.0
                combo_time = cr1.time_sec + cr2.time_sec

                print(f"    {case.id}: primary={cr1.hit_rate:.0%} secondary={cr2.hit_rate:.0%} "
                      f"merged={combo_hr:.0%} time={combo_time:.1f}s")


# ──────────────────────────────────────────────
# KV cache / Flash Attention test
# ──────────────────────────────────────────────

def run_kv_cache_test(top_models: list[ModelResult], cases: list[TestCase]):
    print(f"\n{'='*60}")
    print("KV CACHE / FLASH ATTENTION TEST")
    print(f"{'='*60}")
    print("""
This test compares ollama serve configurations.
You need to restart ollama with different env vars between runs.
The script will run the same cases and compare.

Configurations to test:
  A: ollama serve                                          (default)
  B: OLLAMA_FLASH_ATTENTION=1 ollama serve                 (flash attn, KV f16)
  C: OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0   (flash + KV q8)
  D: OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q4_0   (flash + KV q4, aggressive)

Current run measures whichever config ollama is currently running with.
""")

    subset = cases[:4]
    for mr in top_models[:2]:
        model = mr.model
        print(f"\n  Model: {model}")
        warmup_model(model)
        model_info = ollama_ps()
        offload = model_info.get("offload_ratio", 0)
        print(f"  offload_ratio={offload:.2f}")

        for case in subset:
            cr = run_case(model, case)
            hits = sum(cr.hit_mask)
            eval_rate = cr.llm_metrics.get("eval_rate", 0)
            vram = cr.vram_usage_mb
            print(f"    {case.id}: hits={hits}/{len(cr.hit_mask)} "
                  f"facts={cr.total_facts} prec={cr.precision:.0%} "
                  f"time={cr.time_sec:.1f}s decode={eval_rate:.1f}tok/s "
                  f"vram={vram}MB")


# ──────────────────────────────────────────────
# Context window test
# ──────────────────────────────────────────────

def run_context_test(top_models: list[ModelResult], cases: list[TestCase]):
    print(f"\n{'='*60}")
    print("CONTEXT WINDOW TEST (num_ctx)")
    print(f"{'='*60}")

    ctx_sizes = [2048, 4096, 8192, 16384]
    subset = cases[:3]

    for mr in top_models[:2]:
        model = mr.model
        print(f"\n  Model: {model}")

        for ctx in ctx_sizes:
            # Unload and reload with new num_ctx to force KV cache reallocation
            ollama_stop(model)
            time.sleep(3)
            warmup_model(model)  # reload

            model_info = ollama_ps()
            offload = model_info.get("offload_ratio", 0)
            print(f"\n    num_ctx={ctx} (offload_ratio={offload:.2f}):")

            for case in subset:
                cr = run_case(model, case, extra_options={"num_ctx": ctx})
                hits = sum(cr.hit_mask)
                eval_rate = cr.llm_metrics.get("eval_rate", 0)
                vram = cr.vram_usage_mb
                print(f"      {case.id}: hits={hits}/{len(cr.hit_mask)} "
                      f"facts={cr.total_facts} time={cr.time_sec:.1f}s "
                      f"decode={eval_rate:.1f}tok/s vram={vram}MB")


# ──────────────────────────────────────────────
# LOQ mode test (Balance vs Performance)
# ──────────────────────────────────────────────

def run_loq_test(top_models: list[ModelResult], cases: list[TestCase]):
    print(f"\n{'='*60}")
    print("LENOVO LOQ MODE TEST")
    print(f"{'='*60}")
    print("""
This test compares Lenovo LOQ thermal profiles:
  1. Баланс (Balance) — current default
  2. Производительность + Разгон ГП (Performance + GPU OC)

Switch mode in Lenovo Vantage, then re-run this test.
Current run measures whichever mode is currently active.

Comparing: eval_rate, gpu_clock, temp_max, power_avg, throttling.
""")

    subset = cases[:4]
    for mr in top_models[:2]:
        model = mr.model
        print(f"\n  Model: {model}")

        gpu_col = GpuMetricsCollector()
        gpu_col.start(f"loq_{model.replace(':', '_')}")

        warmup_model(model)
        for case in subset:
            cr = run_case(model, case)
            hits = sum(cr.hit_mask)
            eval_rate = cr.llm_metrics.get("eval_rate", 0)
            print(f"    {case.id}: hits={hits}/{len(cr.hit_mask)} "
                  f"decode={eval_rate:.1f}tok/s time={cr.time_sec:.1f}s")

        gpu = gpu_col.stop()
        print(f"\n  GPU summary: util_avg={gpu.get('gpu_util_avg')}% "
              f"temp_max={gpu.get('temp_max_c')}°C "
              f"power_avg={gpu.get('power_avg_w')}W "
              f"clock_avg={gpu.get('gpu_clock_avg_mhz')}MHz "
              f"throttling={gpu.get('is_throttling')}")


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────

def _fmt(val, fmt=".1f", fallback="n/a"):
    if val is None:
        return fallback
    return f"{val:{fmt}}"


def print_table1_summary(results: list[ModelResult]):
    """Таблица 1: Сводка по моделям."""
    show_cost = COST_PER_HOUR_USD > 0
    width = 160 if show_cost else 140
    print(f"\n{'='*width}")
    print("TABLE 1: MODEL SUMMARY (sorted by quality_speed_score)")
    print(f"{'='*width}")

    sorted_r = sorted(results, key=lambda r: r.quality_speed_score, reverse=True)

    header = (
        f"{'Model':<32} {'Params':<10} {'Offload':>8} {'Bottleneck':<18} "
        f"{'Hit%':>5} {'Prec%':>6} {'Fmt%':>5} "
        f"{'eval t/s':>9} {'pfill t/s':>10} "
        f"{'Sec/case':>9} {'VRAM peak':>10} "
        f"{'ETA(h)':>7} {'QS score':>9}"
    )
    if show_cost:
        header += f" {'$/book':>7} {'$/corpus':>9}"
    print(header)
    print("-" * width)

    for r in sorted_r:
        if not r.cases:
            print(f"{r.model:<32} SKIPPED")
            continue

        offload = r.model_info.get("offload_ratio")
        offload_str = f"{offload:.2f}" if offload is not None else "?"
        vram_peak = r.gpu_metrics.get("vram_used_peak_mb")
        vram_str = f"{vram_peak:.0f}MB" if vram_peak else "?"

        row = (
            f"{r.model:<32} {r.spec.params_total:<10} {offload_str:>8} {r.bottleneck:<18} "
            f"{r.avg_hit_rate:>5.0%} {r.avg_precision:>6.0%} {r.format_reliability:>5.0%} "
            f"{r.avg_eval_rate:>9.1f} {r.avg_prompt_eval_rate:>10.0f} "
            f"{r.avg_time_sec:>9.1f} {vram_str:>10} "
            f"{r.estimated_pipeline_hours:>7.0f} {r.quality_speed_score:>9.4f}"
        )
        if show_cost:
            row += f" ${r.cost_per_book_usd:>6.3f} ${r.cost_per_corpus_usd:>8.2f}"
        print(row)


def print_table2_hardware(results: list[ModelResult]):
    """Таблица 2: Аппаратная диагностика."""
    print(f"\n{'='*150}")
    print("TABLE 2: HARDWARE DIAGNOSTICS")
    print(f"{'='*150}")

    header = (
        f"{'Model':<22} {'VRAM used/free MB':>18} {'GPU%':>6} {'Mem%':>6} "
        f"{'Temp avg/max':>13} {'Power avg/max':>14} {'Throttle?':>10} "
        f"{'Offload':>8} {'Quant':<10} {'Bottleneck':<18}"
    )
    print(header)
    print("-" * 150)

    for r in sorted(results, key=lambda r: r.quality_speed_score, reverse=True):
        if not r.cases:
            continue
        g = r.gpu_metrics
        mi = r.model_info

        vram_used = g.get("vram_used_peak_mb")
        vram_free = g.get("vram_free_min_mb")
        vram_str = f"{_fmt(vram_used, '.0f')}/{_fmt(vram_free, '.0f')}"
        gpu_util = _fmt(g.get("gpu_util_avg"), ".0f")
        mem_util = _fmt(g.get("mem_util_avg"), ".0f")
        temp_avg = _fmt(g.get("temp_avg_c"), ".0f")
        temp_max = _fmt(g.get("temp_max_c"), ".0f")
        pwr_avg = _fmt(g.get("power_avg_w"), ".0f")
        pwr_max = _fmt(g.get("power_max_w"), ".0f")
        throttle = "YES" if g.get("is_throttling") else "no"
        offload = mi.get("offload_ratio")
        offload_s = f"{offload:.2f}" if offload is not None else "?"
        quant = mi.get("quantization", "?")

        print(
            f"{r.model:<22} {vram_str:>18} {gpu_util:>6} {mem_util:>6} "
            f"{temp_avg:>6}/{temp_max:<6} {pwr_avg:>6}/{pwr_max:<7} {throttle:>10} "
            f"{offload_s:>8} {quant:<10} {r.bottleneck:<18}"
        )

    # CPU metrics for Group C
    offload_models = [r for r in results if r.cpu_metrics]
    if offload_models:
        print(f"\n  CPU/RAM metrics (Group C offload models):")
        for r in offload_models:
            c = r.cpu_metrics
            print(f"    {r.model}: cpu_avg={c.get('cpu_util_avg')}% "
                  f"ram_peak={c.get('ram_used_peak_gb')}GB "
                  f"ram_avg={c.get('ram_used_avg_gb')}GB")


def print_table3_details(results: list[ModelResult]):
    """Таблица 3: Детализация по кейсам."""
    print(f"\n{'='*150}")
    print("TABLE 3: CASE DETAILS")
    print(f"{'='*150}")

    for r in sorted(results, key=lambda r: r.quality_speed_score, reverse=True):
        if not r.cases:
            continue
        print(f"\n  --- {r.model} (QxS={r.quality_speed_score:.4f}, bottleneck={r.bottleneck}) ---")

        header = (
            f"    {'Case':<35} {'Hits':>6} {'Facts':>6} {'Noise':>6} "
            f"{'Prec':>6} {'Err':>4} {'Time':>7} "
            f"{'eval t/s':>9} {'pfill t/s':>10} "
            f"{'In tok':>7} {'Out tok':>8}"
        )
        print(header)

        for cr in r.cases:
            hits = sum(cr.hit_mask)
            total = len(cr.hit_mask)
            llm = cr.llm_metrics
            print(
                f"    {cr.case_id:<35} {hits}/{total:>3} {cr.total_facts:>6} "
                f"{len(cr.noise_markers):>6} "
                f"{cr.precision:>6.0%} {cr.parse_errors:>4} {cr.time_sec:>7.1f} "
                f"{llm.get('eval_rate', 0):>9.1f} {llm.get('prompt_eval_rate', 0):>10.0f} "
                f"{llm.get('prompt_eval_count', 0):>7} {llm.get('eval_count', 0):>8}"
            )
            missed = [d for d, h in zip(cr.hit_descriptions, cr.hit_mask) if not h]
            if missed:
                print(f"      MISSED: {', '.join(missed)}")
            if cr.noise_markers:
                print(f"      NOISE:  {', '.join(cr.noise_markers)}")


def print_table4_recommendations(results: list[ModelResult]):
    """Таблица 4: Рекомендации."""
    valid = [r for r in results if r.cases and r.avg_hit_rate > 0]
    if not valid:
        print("\nNo valid results for recommendations.")
        return

    print(f"\n{'='*140}")
    print("TABLE 4: RECOMMENDATIONS")
    print(f"{'='*140}")

    by_qxs = sorted(valid, key=lambda r: r.quality_speed_score, reverse=True)
    by_qual = sorted(valid, key=lambda r: r.quality_score, reverse=True)

    # GPU-only models
    gpu_only = [r for r in valid if r.model_info.get("offload_ratio", 1.0) >= 0.99]
    gpu_by_qxs = sorted(gpu_only, key=lambda r: r.quality_speed_score, reverse=True) if gpu_only else by_qxs

    def _cost_str(r: ModelResult) -> str:
        if COST_PER_HOUR_USD > 0:
            return f", Cost: ${r.cost_per_book_usd:.3f}/book, ${r.cost_per_corpus_usd:.2f}/corpus (32 books)"
        return ""

    print("\n  BEST SINGLE MODEL FOR SPEED (highest Quality×Speed, GPU-only):")
    if gpu_by_qxs:
        r = gpu_by_qxs[0]
        print(f"    PRIMARY_MODEL={r.model}")
        print(f"    ESTIMATED_HOURS={r.estimated_pipeline_hours:.1f}")
        print(f"    Hit rate: {r.avg_hit_rate:.0%}, Precision: {r.avg_precision:.0%}")
        print(f"    Decode: {r.avg_eval_rate:.1f} tok/s, Bottleneck: {r.bottleneck}{_cost_str(r)}")

    print("\n  BEST SINGLE MODEL FOR QUALITY (highest quality_score):")
    if by_qual:
        r = by_qual[0]
        print(f"    PRIMARY_MODEL={r.model}")
        print(f"    ESTIMATED_HOURS={r.estimated_pipeline_hours:.1f}")
        print(f"    Hit rate: {r.avg_hit_rate:.0%}, Precision: {r.avg_precision:.0%}")
        print(f"    Decode: {r.avg_eval_rate:.1f} tok/s, Bottleneck: {r.bottleneck}{_cost_str(r)}")

    print("\n  RECOMMENDED MIXED CONFIG:")
    if len(by_qxs) >= 2:
        fast = gpu_by_qxs[0] if gpu_by_qxs else by_qxs[0]
        qual = by_qual[0]
        print(f"    PRIMARY_MODEL={fast.model}")
        if fast.model != qual.model:
            print(f"    SECONDARY_MODEL={qual.model}")
            print(f"    ARBITER_MODEL={qual.model}")
        else:
            second = by_qual[1] if len(by_qual) > 1 else by_qxs[1]
            print(f"    SECONDARY_MODEL={second.model}")
            print(f"    ARBITER_MODEL={fast.model}")
        eta = fast.estimated_pipeline_hours_ensemble
        print(f"    ESTIMATED_HOURS={eta:.1f}")
        print(f"    ESTIMATED_QUALITY_VS_BASELINE=better (dual extraction + arbiter)")

    # API recommendation
    offload_models = [r for r in valid if r.bottleneck == "CPU_OFFLOAD" and r.quality_score > 0.3]
    if offload_models:
        best_offload = sorted(offload_models, key=lambda r: r.quality_score, reverse=True)[0]
        print(f"\n  API ALTERNATIVE (for models limited by CPU_OFFLOAD):")
        print(f"    Model with best quality but CPU-limited: {best_offload.model}")
        print(f"    Consider: gemma-3-27b-it via Google AI Studio or DeepInfra")
        print(f"    API_ESTIMATED_COST=~$5-15 for full corpus (depends on provider)")


def print_bottleneck_analysis(results: list[ModelResult]):
    """Секция 5: Bottleneck-анализ."""
    valid = [r for r in results if r.cases]
    if not valid:
        return

    print(f"\n{'='*140}")
    print("SECTION 5: BOTTLENECK ANALYSIS")
    print(f"{'='*140}")

    groups = {}
    for r in valid:
        groups.setdefault(r.bottleneck, []).append(r)

    for bn, models in sorted(groups.items()):
        names = ", ".join(r.model for r in models)
        print(f"\n  {bn}: {names}")

        if bn == "CPU_OFFLOAD":
            print("    → Models partially offloaded to CPU RAM. Decode speed is limited by")
            print("      PCIe bandwidth, not GPU compute. For these models, consider:")
            print("      - Using via API (Google AI Studio, DeepInfra) instead of local")
            print("      - KV cache quantization to fit more layers in VRAM")
            print("      - Reducing num_ctx to free VRAM for model weights")
        elif bn == "THERMAL_THROTTLING":
            print("    → GPU hitting thermal limits (>87°C). Consider:")
            print("      - Switch to 'Производительность' mode in Lenovo Vantage")
            print("      - Use laptop on a cooling pad")
            print("      - Lower ambient temperature")
        elif bn == "TDP_THROTTLING":
            print("    → GPU hitting power limit. Consider:")
            print("      - Enable 'Разгон ГП' in Lenovo Vantage")
            print("      - This may not help much if thermal limit is also close")
        elif bn == "MEMORY_BANDWIDTH":
            print("    → VRAM bandwidth is the bottleneck. This is expected for")
            print("      memory-bound decode workloads. Flash attention may help.")
        elif bn == "VRAM_NEAR_OOM":
            print("    → Very little free VRAM. KV cache is under pressure.")
            print("      Try: KV cache q8_0/q4_0, reduce num_ctx, or use smaller model.")
        elif bn == "COMPUTE_BOUND":
            print("    → Healthy scenario — GPU compute is the bottleneck.")
            print("      Performance mode / GPU OC may give marginal improvements.")
        elif bn == "GPU_UNDERUTILIZED":
            print("    → GPU is not fully utilized. Possible causes:")
            print("      - Small batch size / short sequences")
            print("      - Scheduling overhead in ollama")
            print("      - Investigate with nvidia-smi dmon")


# ──────────────────────────────────────────────
# JSON output
# ──────────────────────────────────────────────

def build_json_report(results: list[ModelResult], vram_baseline: float, tag: str) -> dict:
    report = {
        "benchmark_meta": {
            "timestamp": datetime.now().isoformat(),
            "vram_baseline_used_mb": vram_baseline,
            "tag": tag,
            "est_chunks_per_book": EST_CHUNKS_PER_BOOK,
            "theoretical_decode_ceiling_toks": THEORETICAL_DECODE_CEILING_TOKS,
            "cost_per_hour_usd": COST_PER_HOUR_USD,
        },
        "models": {},
    }

    for mr in results:
        model_data = {
            "spec": {
                "group": mr.spec.group,
                "params_total": mr.spec.params_total,
                "params_active": mr.spec.params_active,
                "vram_est_gb": mr.spec.vram_est_gb,
                "notes": mr.spec.notes,
            },
            "model_info": mr.model_info,
            "gpu_metrics": mr.gpu_metrics,
            "cpu_metrics": mr.cpu_metrics,
            "bottleneck": mr.bottleneck,
            "aggregate": {
                "avg_hit_rate": round(mr.avg_hit_rate, 4),
                "avg_precision": round(mr.avg_precision, 4),
                "avg_time_sec": round(mr.avg_time_sec, 3),
                "format_reliability": round(mr.format_reliability, 4),
                "avg_eval_rate": round(mr.avg_eval_rate, 2),
                "avg_prompt_eval_rate": round(mr.avg_prompt_eval_rate, 2),
                "avg_total_tokens": round(mr.avg_total_tokens, 0),
                "estimated_pipeline_hours": round(mr.estimated_pipeline_hours, 1),
                "estimated_pipeline_hours_ensemble": round(mr.estimated_pipeline_hours_ensemble, 1),
                "quality_speed_score": round(mr.quality_speed_score, 6),
                "quality_score": round(mr.quality_score, 6),
                "efficiency_score": round(mr.efficiency_score, 6),
                "cost_per_book_usd": round(mr.cost_per_book_usd, 4),
                "cost_per_corpus_usd": round(mr.cost_per_corpus_usd, 4),
            },
            "cases": [],
        }

        for cr in mr.cases:
            case_data = {
                "case_id": cr.case_id,
                "quality": {
                    "hit_rate": round(cr.hit_rate, 4),
                    "precision": round(cr.precision, 4),
                    "noise_rate": round(cr.noise_rate, 4),
                    "parse_errors": cr.parse_errors,
                    "total_facts": cr.total_facts,
                    "hits": cr.hit_mask,
                    "misses": [d for d, h in zip(cr.hit_descriptions, cr.hit_mask) if not h],
                    "noise_markers": cr.noise_markers,
                    "facts": cr.facts,
                },
                "llm_metrics": cr.llm_metrics,
                "gpu_snapshot": {
                    "vram_usage_mb": cr.vram_usage_mb,
                },
                "wall_clock_sec": round(cr.time_sec, 3),
                "raw_response": cr.raw_response[:3000],
            }
            model_data["cases"].append(case_data)

        report["models"][mr.model] = model_data

    return report


def save_results(results: list[ModelResult], vram_baseline: float, tag: str = "") -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{tag}_{ts}.json" if tag else f"benchmark_{ts}.json"
    path = RESULTS_DIR / filename

    report = build_json_report(results, vram_baseline, tag)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {path}")
    return path


def apply_all_flag_defaults(args) -> None:
    """Разворачивает --all, не затирая явно запрошенную группу моделей."""
    if not getattr(args, "all", False):
        return

    if getattr(args, "group", None) == ["A", "B"]:
        args.group = ["A", "B", "C"]
    args.include_failed = True
    args.ensemble = True
    args.kv_cache_test = True
    args.context_test = True
    args.loq_test = True


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM models for Eho extraction pipeline",
    )
    parser.add_argument("--models", nargs="+", default=[],
                        help="Explicit model names to test")
    parser.add_argument("--group", nargs="+", default=["A", "B"],
                        help="Model groups (A, B, C, 4090). Use 4090 for RTX 4090 server suite.")
    parser.add_argument("--cost-per-hour", type=float, default=0.0,
                        help="Server rental cost in USD/hour (e.g. 0.23 for Vast.ai RTX 4090). "
                             "Enables cost_per_book and cost_per_corpus columns.")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include previously failed models (phi4-mini, qwen3:4b)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Run ensemble test on top-3")
    parser.add_argument("--kv-cache-test", action="store_true",
                        help="Test flash attention + KV cache quantization")
    parser.add_argument("--context-test", action="store_true",
                        help="Test num_ctx (2048/4096/8192/16384)")
    parser.add_argument("--loq-test", action="store_true",
                        help="Test LOQ thermal profile (Balance vs Performance)")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    parser.add_argument("--cases", nargs="+", default=[],
                        help="Run only specific case IDs")
    parser.add_argument("--tag", default="",
                        help="Tag for results file")

    args = parser.parse_args()

    global COST_PER_HOUR_USD
    COST_PER_HOUR_USD = args.cost_per_hour

    apply_all_flag_defaults(args)

    models = get_models(args.group, args.include_failed, args.models)
    if not models:
        print("No models selected. Use --models or --group.")
        sys.exit(1)

    cases = TEST_CASES
    if args.cases:
        cases = [c for c in TEST_CASES if c.id in args.cases]
        if not cases:
            print(f"No matching cases. Available: {[c.id for c in TEST_CASES]}")
            sys.exit(1)

    print(f"Eho Dataset Pipeline — LLM Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Models: {len(models)} ({', '.join(m.name for m in models)})")
    print(f"Cases: {len(cases)} ({', '.join(c.id for c in cases)})")
    print(f"Groups: {', '.join(sorted(set(m.group for m in models)))}")

    # Ensure ollama is running (start it if needed)
    ensure_ollama_running()

    # Baseline VRAM snapshot
    vram_baseline = get_vram_baseline() or 0
    print(f"VRAM baseline (system/browser/compositor): {vram_baseline:.0f} MB")

    # ── Single-model benchmarks ──
    all_results: list[ModelResult] = []
    for i, spec in enumerate(models):
        mr = run_model(spec, cases, vram_baseline=vram_baseline)
        all_results.append(mr)

        # Unload between models
        ollama_stop(spec.name)
        if i < len(models) - 1:
            print(f"\n  Unloading {spec.name}, waiting 10s for VRAM to settle...")
            time.sleep(10)
            # Verify VRAM returned to baseline
            vram_now = get_vram_snapshot()
            if vram_now:
                print(f"  VRAM after unload: {vram_now:.0f} MB (baseline: {vram_baseline:.0f} MB)")

    # ── Print all tables ──
    print_table1_summary(all_results)
    print_table2_hardware(all_results)
    print_table3_details(all_results)
    print_table4_recommendations(all_results)
    print_bottleneck_analysis(all_results)

    # ── Save JSON ──
    tag = args.tag or "single"
    save_results(all_results, vram_baseline, tag)

    # ── Valid results for further tests ──
    valid = sorted(
        [r for r in all_results if r.cases and r.avg_hit_rate > 0],
        key=lambda r: r.quality_speed_score,
        reverse=True,
    )

    # ── Additional tests ──
    if args.ensemble and len(valid) >= 2:
        run_ensemble_test(valid[:3], cases)

    if args.kv_cache_test and valid:
        run_kv_cache_test(valid, cases)

    if args.context_test and valid:
        run_context_test(valid, cases)

    if args.loq_test and valid:
        run_loq_test(valid, cases)

    print(f"\nDone. Models tested: {len(all_results)}, "
          f"valid: {len(valid)}, "
          f"timestamp: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
