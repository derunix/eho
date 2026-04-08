import argparse
import json
import sys
import time
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import openai  # noqa: F401
except Exception:
    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        pass

    fake_openai.OpenAI = FakeOpenAI
    sys.modules["openai"] = fake_openai

import extract_dialogues as ed
BOOK_PATH = ROOT / "books" / "1. Лабиринты Ехо" / "1. Чужак.fb2"


@dataclass(frozen=True)
class ExpectedSignal:
    subject_tokens: tuple[str, ...]
    fact_tokens: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    needle: str
    expected: tuple[ExpectedSignal, ...]


CASES = (
    BenchmarkCase(
        case_id="intro_insomnia",
        description="биографический факт о Максе",
        needle="С младенческих лет я не мог спать по ночам",
        expected=(
            ExpectedSignal(
                subject_tokens=("макс",),
                fact_tokens=("не мог спать", "ноч"),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="kimpa_butler",
        description="именованный персонаж и его роль",
        needle="Кимпа",
        expected=(
            ExpectedSignal(
                subject_tokens=("кимпа",),
                fact_tokens=("дворецк",),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="obzhora_bunba",
        description="именованное место из мира Ехо",
        needle="Сегодня мы ужинаем в «Обжоре»",
        expected=(
            ExpectedSignal(
                subject_tokens=("обжор", "бунб"),
                fact_tokens=("забегалов",),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="shurf_gloves_and_social_rule",
        description="персонажная особенность и бытовое правило мира",
        needle="толстые защитные перчатки",
        expected=(
            ExpectedSignal(
                subject_tokens=("шурф",),
                fact_tokens=("перчат",),
            ),
            ExpectedSignal(
                subject_tokens=("соединен", "королевств"),
                fact_tokens=("между ближайшими друзьями",),
            ),
        ),
    ),
)


NEGATIVE_SUBJECTS = {
    "европа",
    "два друга",
    "два персонажа",
    "действие",
    "действия",
    "character",
    "entity",
    "генерал",
    "телефон",
    "стол",
    "странный",
    "энциклопедия",
}


NEGATIVE_FACT_PATTERNS = (
    "персонаж, который",
    "является персонажем",
    "место действия",
    "упоминается в контексте",
    "не раскрывается",
    "в тексте не упоминается",
    "в данной сцене",
    "видимо",
    "возможно",
    "предметом обсуждения",
    "в состоянии, когда",
    "источник шума или внимания",
)


def norm(text: str) -> str:
    return ed.normalize_dedup_text(text)


def find_case_chunk(chunks: list[str], needle: str) -> int:
    lowered = needle.lower()
    for idx, chunk in enumerate(chunks):
        if lowered in chunk.lower():
            return idx
    raise RuntimeError(f"needle not found in chunks: {needle}")


def fact_matches_signal(item: dict, signal: ExpectedSignal) -> bool:
    subject = norm(item.get("subject", ""))
    fact = norm(item.get("fact", ""))
    if not subject or not fact:
        return False
    if not all(token in subject for token in signal.subject_tokens):
        return False
    if not all(token in fact for token in signal.fact_tokens):
        return False
    return True


def score_case(items: list[dict], case: BenchmarkCase) -> dict:
    signal_hits = 0
    matched_signals: list[dict] = []
    for signal in case.expected:
        match = next((item for item in items if fact_matches_signal(item, signal)), None)
        if match is None:
            continue
        signal_hits += 1
        matched_signals.append({
            "subject": match.get("subject", ""),
            "fact": match.get("fact", ""),
        })

    garbage_items = []
    for item in items:
        subject = norm(item.get("subject", ""))
        fact = norm(item.get("fact", ""))
        if subject in NEGATIVE_SUBJECTS or any(pattern in fact for pattern in NEGATIVE_FACT_PATTERNS):
            garbage_items.append({
                "subject": item.get("subject", ""),
                "fact": item.get("fact", ""),
            })

    return {
        "signal_hits": signal_hits,
        "expected_signals": len(case.expected),
        "garbage_items": garbage_items,
    }


def make_config(model: str, output_dir: Path, *, validator_model: str = "") -> ed.Config:
    config = ed.Config(
        output_dir=str(output_dir),
        model=model,
    )
    config.knowledge_extract_model = model
    config.knowledge_validate_model = validator_model
    config.knowledge_extraction_protocol = "lines"
    config.extraction_passes = 1
    config.knowledge_extraction_tracks = ("world", "scene")
    config.knowledge_page_max_items = 10
    config.max_tokens_knowledge = 1200
    config.knowledge_llm_validation_enabled = bool(validator_model)
    config.knowledge_linking_enabled = False
    config.request_timeout = 180
    return config


def list_local_models() -> list[str]:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as resp:
        data = json.load(resp)
    return [
        item["name"]
        for item in data.get("models", [])
        if isinstance(item, dict) and item.get("name")
    ]


def resolve_model_name(model: str, local_models: list[str]) -> str:
    if model in local_models:
        return model
    if ":" not in model:
        latest_name = f"{model}:latest"
        if latest_name in local_models:
            return latest_name
    return model


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "+" not in spec:
        return spec, ""
    extract_model, validator_model = spec.split("+", 1)
    return extract_model.strip(), validator_model.strip()


def benchmark_model(model_spec: str, chunks: list[str], cases: list[tuple[BenchmarkCase, int]], output_dir: Path) -> dict:
    model, validator_model = parse_model_spec(model_spec)
    config = make_config(model, output_dir, validator_model=validator_model)
    ed._use_ollama_native = True

    collected_logs: list[str] = []
    original_log_event = ed.log_event

    def capture_log(message: str):
        collected_logs.append(message)
        original_log_event(message)

    ed.log_event = capture_log
    t0 = time.time()
    case_results = []

    try:
        for case, chunk_idx in cases:
            case_t0 = time.time()
            chunk_payload, _ = ed.build_extraction_chunk_payload(chunks, chunk_idx, config)
            items = ed.extract_knowledge(
                client=None,
                config=config,
                chunk=chunks[chunk_idx],
                log_prefix=f"[bench {model} {case.case_id}]",
                chunk_payload=chunk_payload,
            )
            elapsed = time.time() - case_t0
            scoring = score_case(items, case)
            case_results.append({
                "case_id": case.case_id,
                "description": case.description,
                "chunk_idx": chunk_idx + 1,
                "seconds": round(elapsed, 2),
                "facts": items,
                **scoring,
            })
    finally:
        ed.log_event = original_log_event

    total_seconds = time.time() - t0
    parse_error_count = sum(
        1
        for line in collected_logs
        if "JSON parse error" in line or "ответ не удалось распарсить" in line
    )
    total_facts = sum(len(case["facts"]) for case in case_results)
    total_hits = sum(case["signal_hits"] for case in case_results)
    expected_hits = sum(case["expected_signals"] for case in case_results)
    garbage_count = sum(len(case["garbage_items"]) for case in case_results)
    nonempty_cases = sum(1 for case in case_results if case["facts"])

    return {
        "model_spec": model_spec,
        "model": model,
        "validator_model": validator_model,
        "total_seconds": round(total_seconds, 2),
        "avg_seconds_per_case": round(total_seconds / max(len(case_results), 1), 2),
        "total_facts": total_facts,
        "total_hits": total_hits,
        "expected_hits": expected_hits,
        "hit_rate": round(total_hits / max(expected_hits, 1), 3),
        "garbage_count": garbage_count,
        "parse_error_count": parse_error_count,
        "nonempty_cases": nonempty_cases,
        "case_results": case_results,
    }


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# Model benchmark",
        "",
        "| Spec | Hit rate | Hits | Facts | Garbage | Parse errors | Avg sec/case |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in results:
        lines.append(
            f"| {item['model_spec']} | {item['hit_rate']:.3f} | "
            f"{item['total_hits']}/{item['expected_hits']} | {item['total_facts']} | "
            f"{item['garbage_count']} | {item['parse_error_count']} | {item['avg_seconds_per_case']:.2f} |"
        )

    lines.append("")
    for item in results:
        lines.append(f"## {item['model_spec']}")
        for case in item["case_results"]:
            lines.append(
                f"- `{case['case_id']}` chunk {case['chunk_idx']}: "
                f"{case['signal_hits']}/{case['expected_signals']} signals, "
                f"{len(case['facts'])} facts, {case['seconds']:.2f}s"
            )
            if case["garbage_items"]:
                for garbage in case["garbage_items"][:5]:
                    lines.append(
                        f"  garbage: {garbage['subject']} -> {garbage['fact']}"
                    )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local Ollama models on real knowledge extraction cases")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma4", "qwen3:8b", "qwen3:4b", "qwen2.5:7b", "phi4-mini"],
        help="Ollama extract-model names or specs like extract+validator",
    )
    args = parser.parse_args()

    if not BOOK_PATH.exists():
        raise SystemExit(f"book not found: {BOOK_PATH}")

    text = ed.load_fb2_file(BOOK_PATH)
    chunks = ed.split_into_chunks(text, ed.Config.chunk_size, ed.Config.chunk_overlap)
    cases = [(case, find_case_chunk(chunks, case.needle)) for case in CASES]

    run_dir = ROOT / "output" / "model_benchmarks" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    local_models = list_local_models()
    resolved_specs = []
    for spec in args.models:
        extract_model, validator_model = parse_model_spec(spec)
        resolved_extract = resolve_model_name(extract_model, local_models)
        resolved_validator = resolve_model_name(validator_model, local_models) if validator_model else ""
        resolved_specs.append(
            resolved_extract if not resolved_validator else f"{resolved_extract}+{resolved_validator}"
        )

    results = []
    for requested_spec, resolved_spec in zip(args.models, resolved_specs):
        title = resolved_spec if resolved_spec == requested_spec else f"{requested_spec} -> {resolved_spec}"
        print(f"\n=== {title} ===")
        result = benchmark_model(resolved_spec, chunks, cases, run_dir)
        results.append(result)
        print(
            f"hit_rate={result['hit_rate']:.3f} "
            f"hits={result['total_hits']}/{result['expected_hits']} "
            f"facts={result['total_facts']} "
            f"garbage={result['garbage_count']} "
            f"parse_errors={result['parse_error_count']} "
            f"avg_sec={result['avg_seconds_per_case']:.2f}"
        )

    results.sort(
        key=lambda item: (
            -item["hit_rate"],
            item["garbage_count"],
            item["parse_error_count"],
            item["avg_seconds_per_case"],
        )
    )

    json_path = run_dir / "results.json"
    md_path = run_dir / "results.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
