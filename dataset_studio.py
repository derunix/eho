#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import extract_dialogues as ed


ASSETS_DIR = Path(__file__).with_name("dataset_studio_assets")
SUPPORTED_BOOK_SUFFIXES = (".fb2.zip", ".fb2", ".txt", ".zip")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def write_json_atomic(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_jsonl(path: Path, item: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def shorten(text: str, limit: int = 180) -> str:
    text = " ".join(safe_text(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def stable_hash(*parts: Any) -> str:
    payload = "\x1f".join(safe_text(part) for part in parts)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def sort_key_ci(value: str) -> str:
    return ed.normalize_dedup_text(value)


def normalized_message_key(text: str) -> str:
    return ed.normalize_dedup_text(text)


def extract_messages_fields(messages: list[dict]) -> tuple[str, str, str]:
    system_text = ""
    user_text = ""
    assistant_text = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = safe_text(message.get("role", "")).lower()
        content = safe_text(message.get("content", ""))
        if role == "system":
            system_text = content
        elif role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content
    return system_text, user_text, assistant_text


def build_messages(system_text: str, user_text: str, assistant_text: str) -> list[dict]:
    messages = []
    if safe_text(system_text):
        messages.append({"role": "system", "content": safe_text(system_text)})
    messages.append({"role": "user", "content": safe_text(user_text)})
    messages.append({"role": "assistant", "content": safe_text(assistant_text)})
    return messages


def strip_editor_fields(item: dict) -> dict:
    keep = {
        "category",
        "subject",
        "fact",
        "time_scope",
        "source_book",
        "chapter",
        "chunk_idx",
    }
    return {key: item.get(key) for key in keep if key in item and item.get(key) not in (None, "")}


def book_stem_from_output_name(filename: str, prefix: str) -> str:
    stem = Path(filename).stem
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def book_name_from_stem(stem: str) -> str:
    return stem.replace("--", "/")


def split_supported_book_suffix(name: str) -> tuple[str, str]:
    text = safe_text(name)
    lowered = text.lower()
    for suffix in SUPPORTED_BOOK_SUFFIXES:
        if lowered.endswith(suffix):
            return text[: -len(suffix)], text[-len(suffix) :]
    return text, ""


def book_name_aliases(name: str) -> list[str]:
    base = safe_text(name)
    if not base:
        return []
    root, suffix = split_supported_book_suffix(base)
    aliases = {root}
    if suffix:
        aliases.add(base)
    else:
        aliases.update(f"{root}{item}" for item in SUPPORTED_BOOK_SUFFIXES)
    return sorted(aliases)


def canonical_book_name(name: str) -> str:
    root, _ = split_supported_book_suffix(name)
    return safe_text(root)


def book_names_match(left: str, right: str) -> bool:
    left_name = canonical_book_name(left)
    right_name = canonical_book_name(right)
    return bool(left_name and right_name and left_name == right_name)


def op_sort_key(op: dict) -> tuple[str, str]:
    return safe_text(op.get("ts", "")), safe_text(op.get("op_id", ""))


@dataclass
class StudioPaths:
    output_dir: Path
    workspace_dir: Path
    facts_ops: Path
    samples_ops: Path
    themes_ops: Path
    relations_ops: Path
    exports_dir: Path
    llm_jobs_dir: Path
    llm_traces_dir: Path


def make_studio_paths(output_dir: Path, workspace_dir: Optional[Path] = None) -> StudioPaths:
    workspace = workspace_dir or (output_dir / "editor_workspace")
    return StudioPaths(
        output_dir=output_dir,
        workspace_dir=workspace,
        facts_ops=workspace / "facts_ops.jsonl",
        samples_ops=workspace / "samples_ops.jsonl",
        themes_ops=workspace / "themes_ops.jsonl",
        relations_ops=workspace / "relations_ops.jsonl",
        exports_dir=workspace / "exports",
        llm_jobs_dir=workspace / "llm_jobs",
        llm_traces_dir=output_dir / "llm_traces",
    )


def _file_fingerprint(path: Path) -> tuple[float, int]:
    try:
        st = path.stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return (0.0, 0)


def _dir_fingerprint(directory: Path, pattern: str) -> list[tuple[str, float, int]]:
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.glob(pattern)):
        mtime, size = _file_fingerprint(path)
        items.append((str(path), mtime, size))
    return items


class DatasetStudioStore:
    def __init__(self, output_dir: Path, workspace_dir: Optional[Path] = None):
        self.paths = make_studio_paths(output_dir, workspace_dir)
        self._lock = threading.RLock()
        self._config_cache: Optional[ed.Config] = None
        self._llm_trace_stats_cache: Optional[dict[str, Any]] = None
        self._state: dict[str, Any] = {}
        self._state_fingerprint: Optional[list] = None
        self._chunk_cache: Optional[list[dict]] = None
        self._chunk_cache_fingerprint: Optional[list] = None

    def _compute_state_fingerprint(self) -> list:
        return [
            _dir_fingerprint(self.paths.output_dir, "chunks_*.jsonl"),
            _dir_fingerprint(self.paths.output_dir, "knowledge_*.jsonl"),
            _dir_fingerprint(self.paths.output_dir, "voice_*.jsonl"),
            _dir_fingerprint(self.paths.output_dir, "synth_*.jsonl"),
            _dir_fingerprint(self.paths.output_dir, "synth_progress_*.jsonl"),
            _file_fingerprint(self.paths.facts_ops),
            _file_fingerprint(self.paths.samples_ops),
            _file_fingerprint(self.paths.themes_ops),
            _file_fingerprint(self.paths.relations_ops),
        ]

    def _compute_chunk_fingerprint(self) -> list:
        return _dir_fingerprint(self.paths.output_dir, "chunks_*.jsonl")

    def refresh(self):
        with self._lock:
            new_fp = self._compute_state_fingerprint()
            if self._state and self._state_fingerprint == new_fp:
                return
            self._state = self._load_state()
            self._state_fingerprint = new_fp
            self._config_cache = None

    def _load_state(self) -> dict[str, Any]:
        chunk_index = self._load_chunk_index()
        base_facts = self._load_base_facts(chunk_index)
        base_samples = self._load_base_samples(chunk_index, base_facts)
        base_themes: dict[str, dict] = {}
        base_relations: dict[str, dict] = {}

        facts, fact_ops = self._apply_ops(base_facts, self.paths.facts_ops)
        samples, sample_ops = self._apply_ops(base_samples, self.paths.samples_ops)
        themes, theme_ops = self._apply_ops(base_themes, self.paths.themes_ops)
        relations, relation_ops = self._apply_ops(base_relations, self.paths.relations_ops)

        return {
            "base_facts": base_facts,
            "base_samples": base_samples,
            "facts": facts,
            "samples": samples,
            "themes": themes,
            "relations": relations,
            "chunk_index": chunk_index,
            "ops": {
                "facts": fact_ops,
                "samples": sample_ops,
                "themes": theme_ops,
                "relations": relation_ops,
            },
        }

    def _read_ops(self, path: Path) -> list[dict]:
        ops = []
        for op in ed.read_jsonl(path):
            if isinstance(op, dict) and safe_text(op.get("op_id", "")):
                ops.append(op)
        ops.sort(key=op_sort_key)
        return ops

    def _apply_ops(self, base: dict[str, dict], path: Path) -> tuple[dict[str, dict], list[dict]]:
        current = {key: dict(value) for key, value in base.items()}
        ops = self._read_ops(path)
        for op in ops:
            entity_id = safe_text(op.get("entity_id", ""))
            after = op.get("after")
            if not entity_id:
                continue
            if after is None:
                current.pop(entity_id, None)
            elif isinstance(after, dict):
                current[entity_id] = dict(after)
        return current, ops

    def _load_chunk_index(self) -> dict[str, Any]:
        chunk_by_book: dict[str, dict[int, dict]] = {}
        voice_pair_index: dict[tuple[str, str, str], dict] = {}
        voice_assistant_index: dict[tuple[str, str], dict] = {}
        config = ed.Config()

        for path in sorted(self.paths.output_dir.glob("chunks_*.jsonl")):
            book_stem = book_stem_from_output_name(path.name, "chunks_")
            book_name = book_name_from_stem(book_stem)
            alias_names = book_name_aliases(book_name)
            per_book = chunk_by_book.setdefault(alias_names[0], {})
            for alias in alias_names[1:]:
                chunk_by_book[alias] = per_book

            for record in ed.read_jsonl(path):
                idx = record.get("idx")
                if not isinstance(idx, int):
                    continue

                source = {
                    "source_book": book_name,
                    "chapter": safe_text(record.get("chapter", "")),
                    "chunk_idx": idx,
                    "source_excerpt": safe_text(record.get("chunk_text", "")),
                }
                per_book[idx] = source

                dialogues = [item for item in record.get("dialogues", []) if isinstance(item, dict)]
                if not dialogues:
                    continue

                try:
                    pairs = ed.make_training_pairs(dialogues, config)
                except Exception:
                    pairs = []

                for pair in pairs:
                    messages = pair.get("messages", [])
                    if not isinstance(messages, list):
                        continue
                    _, user_text, assistant_text = extract_messages_fields(messages)
                    if not assistant_text:
                        continue
                    exact_key = (
                        book_stem,
                        normalized_message_key(user_text),
                        normalized_message_key(assistant_text),
                    )
                    voice_pair_index.setdefault(exact_key, source)
                    voice_assistant_index.setdefault(
                        (book_stem, normalized_message_key(assistant_text)),
                        source,
                    )

        synth_link_index = self._load_synth_links()
        return {
            "chunks": chunk_by_book,
            "voice_pairs": voice_pair_index,
            "voice_assistant": voice_assistant_index,
            "synth_links": synth_link_index,
        }

    def _load_base_facts(self, chunk_index: dict[str, Any]) -> dict[str, dict]:
        facts: dict[str, dict] = {}
        occurrence_counter: dict[str, int] = {}

        for path in sorted(self.paths.output_dir.glob("knowledge_*.jsonl")):
            if path.name.startswith("knowledge_base"):
                continue

            book_stem = book_stem_from_output_name(path.name, "knowledge_")
            book_name = book_name_from_stem(book_stem)
            for item in ed.read_jsonl(path):
                if not isinstance(item, dict):
                    continue

                category = safe_text(item.get("category", ""))
                subject = safe_text(item.get("subject", ""))
                fact_text = safe_text(item.get("fact", ""))
                time_scope = safe_text(item.get("time_scope", "")) or "unclear"
                if not category or not subject or not fact_text:
                    continue

                source_book = safe_text(item.get("source_book", "")) or book_name
                chapter = safe_text(item.get("chapter", ""))
                chunk_idx = item.get("chunk_idx")
                if not isinstance(chunk_idx, int):
                    chunk_idx = None

                signature = stable_hash(
                    source_book,
                    chunk_idx if chunk_idx is not None else "",
                    category,
                    subject,
                    fact_text,
                    time_scope,
                )
                occurrence = occurrence_counter.get(signature, 0) + 1
                occurrence_counter[signature] = occurrence
                fact_id = f"fact:{signature}:{occurrence}"

                source_info = {}
                if chunk_idx is not None:
                    for alias in book_name_aliases(source_book):
                        source_info = chunk_index["chunks"].get(alias, {}).get(chunk_idx, {})
                        if source_info:
                            break

                facts[fact_id] = {
                    "id": fact_id,
                    "kind": "fact",
                    "category": category,
                    "subject": subject,
                    "fact": fact_text,
                    "time_scope": time_scope,
                    "source_book": source_book,
                    "chapter": chapter or safe_text(source_info.get("chapter", "")),
                    "chunk_idx": chunk_idx,
                    "source_excerpt": safe_text(source_info.get("source_excerpt", "")),
                    "source_file": str(path),
                    "theme_ids": [],
                    "review_status": "pending",
                    "review_score": None,
                    "review_note": "",
                    "updated_at": "",
                    "origin": "base",
                }

        return facts

    def _load_base_samples(self, chunk_index: dict[str, Any], facts: dict[str, dict]) -> dict[str, dict]:
        samples: dict[str, dict] = {}
        occurrence_counter: dict[str, int] = {}
        fact_hash_map: dict[tuple[str, str], list[str]] = {}

        for fact_id, fact in facts.items():
            source_book = safe_text(fact.get("source_book", ""))
            fact_hash = ed.stable_fact_hash(fact)
            for alias in book_name_aliases(source_book):
                fact_hash_map.setdefault((alias, fact_hash), []).append(fact_id)

        file_specs = [
            ("voice", "voice_*.jsonl"),
            ("synth", "synth_*.jsonl"),
        ]

        for kind, pattern in file_specs:
            for path in sorted(self.paths.output_dir.glob(pattern)):
                book_stem = book_stem_from_output_name(path.name, f"{kind}_")
                book_name = book_name_from_stem(book_stem)
                for raw in ed.read_jsonl(path):
                    if not isinstance(raw, dict):
                        continue
                    messages = raw.get("messages", [])
                    if not isinstance(messages, list):
                        continue

                    system_text, user_text, assistant_text = extract_messages_fields(messages)
                    if not assistant_text:
                        continue

                    signature = stable_hash(kind, book_stem, user_text, assistant_text)
                    occurrence = occurrence_counter.get(signature, 0) + 1
                    occurrence_counter[signature] = occurrence
                    sample_id = f"sample:{kind}:{signature}:{occurrence}"

                    source_info = {}
                    linked_fact_ids: list[str] = []
                    if kind == "voice":
                        exact_key = (
                            book_stem,
                            normalized_message_key(user_text),
                            normalized_message_key(assistant_text),
                        )
                        source_info = (
                            chunk_index["voice_pairs"].get(exact_key)
                            or chunk_index["voice_assistant"].get((book_stem, normalized_message_key(assistant_text)))
                            or {}
                        )
                    else:
                        synth_key = (
                            book_stem,
                            normalized_message_key(user_text),
                            normalized_message_key(assistant_text),
                        )
                        source_info = chunk_index["synth_links"].get(synth_key, {})
                        source_book = safe_text(source_info.get("source_book", "")) or book_name
                        fact_hash = safe_text(source_info.get("fact_hash", ""))
                        if fact_hash:
                            linked_fact_ids = list(fact_hash_map.get((source_book, fact_hash), []))

                    source_book_value = safe_text(source_info.get("source_book", "")) or book_name
                    chapter_value = safe_text(source_info.get("chapter", ""))
                    chunk_idx_value = source_info.get("chunk_idx")
                    excerpt_value = safe_text(source_info.get("source_excerpt", ""))
                    if kind == "synth" and linked_fact_ids:
                        first_linked = facts.get(linked_fact_ids[0], {})
                        source_book_value = safe_text(first_linked.get("source_book", "")) or source_book_value
                        chapter_value = safe_text(first_linked.get("chapter", "")) or chapter_value
                        if chunk_idx_value is None and isinstance(first_linked.get("chunk_idx"), int):
                            chunk_idx_value = first_linked.get("chunk_idx")
                        excerpt_value = safe_text(first_linked.get("source_excerpt", "")) or excerpt_value

                    samples[sample_id] = {
                        "id": sample_id,
                        "kind": kind,
                        "messages": messages,
                        "system": system_text,
                        "user": user_text,
                        "assistant": assistant_text,
                        "source_book": source_book_value,
                        "chapter": chapter_value,
                        "chunk_idx": chunk_idx_value,
                        "source_excerpt": excerpt_value,
                        "source_file": str(path),
                        "linked_fact_ids": linked_fact_ids,
                        "theme_ids": [],
                        "review_status": "pending",
                        "review_score": None,
                        "review_note": "",
                        "updated_at": "",
                        "origin": "base",
                    }

        return samples

    def _load_synth_links(self) -> dict[tuple[str, str, str], dict]:
        links: dict[tuple[str, str, str], dict] = {}
        for path in sorted(self.paths.output_dir.glob("synth_progress_*.jsonl")):
            book_stem = book_stem_from_output_name(path.name, "synth_progress_")
            book_name = book_name_from_stem(book_stem)
            for record in ed.read_jsonl(path):
                pair = record.get("pair")
                if not isinstance(pair, dict):
                    continue
                messages = pair.get("messages", [])
                if not isinstance(messages, list):
                    continue
                _, user_text, assistant_text = extract_messages_fields(messages)
                if not user_text or not assistant_text:
                    continue
                links[(
                    book_stem,
                    normalized_message_key(user_text),
                    normalized_message_key(assistant_text),
                )] = {
                    "source_book": book_name_aliases(book_name)[0],
                    "fact_hash": safe_text(record.get("fact_hash", "")),
                }
        return links

    def _require_state(self):
        if not self._state:
            self.refresh()

    def _record_operation(
        self,
        domain: str,
        entity_id: str,
        before: Optional[dict],
        after: Optional[dict],
        note: str = "",
    ) -> dict:
        op = {
            "op_id": f"op:{uuid.uuid4().hex}",
            "ts": now_iso(),
            "domain": domain,
            "entity_id": entity_id,
            "before": before,
            "after": after,
            "note": safe_text(note),
        }
        path = getattr(self.paths, f"{domain}_ops")
        append_jsonl(path, op)
        return op

    def _last_operation(self, domain: Optional[str] = None) -> Optional[dict]:
        self._require_state()
        ops = []
        domains = [domain] if domain else ["facts", "samples", "themes", "relations"]
        for name in domains:
            ops.extend(self._state["ops"].get(name, []))
        if not ops:
            return None
        ops.sort(key=op_sort_key)
        return ops[-1]

    def undo_last(self, domain: Optional[str] = None) -> dict:
        with self._lock:
            self.refresh()
            op = self._last_operation(domain=domain)
            if op is None:
                raise ValueError("No operations to undo")
            inverse = self._record_operation(
                op["domain"],
                safe_text(op.get("entity_id", "")),
                before=op.get("after"),
                after=op.get("before"),
                note=f"undo:{safe_text(op.get('op_id', ''))}",
            )
            self.refresh()
            return inverse

    def summary(self) -> dict:
        with self._lock:
            self.refresh()
            facts = list(self._state["facts"].values())
            samples = list(self._state["samples"].values())
            themes = list(self._state["themes"].values())
            relations = list(self._state["relations"].values())
            chunk_total = sum(1 for _ in self._iter_chunk_records())
            metadata_events_total = sum(1 for _ in self._iter_metadata_events())
            llm_jobs_total = sum(1 for _ in self._iter_llm_jobs())
            timeline_overview = self.get_timeline_overview()
            llm_stats = self._llm_trace_stats_cache or {"run_count": 0, "total_traces": 0}
            books = sorted(
                {
                    safe_text(item.get("source_book", ""))
                    for item in facts + samples
                    if safe_text(item.get("source_book", ""))
                },
                key=sort_key_ci,
            )
            return {
                "counts": {
                    "facts": len(facts),
                    "voice": len([item for item in samples if item.get("kind") == "voice"]),
                    "synth": len([item for item in samples if item.get("kind") == "synth"]),
                    "themes": len(themes),
                    "relations": len(relations),
                    "chunks": chunk_total,
                    "metadata_events": metadata_events_total,
                    "llm_jobs": llm_jobs_total,
                    "llm_runs": int(llm_stats["run_count"]),
                    "llm_traces": int(llm_stats["total_traces"]),
                    "timeline_nodes": int(timeline_overview["counts"].get("nodes", 0)),
                    "timeline_edges": int(timeline_overview["counts"].get("edges", 0)),
                    "timeline_groups": int(timeline_overview["counts"].get("groups", 0)),
                },
                "books": books,
                "categories": sorted(
                    {safe_text(item.get("category", "")) for item in facts if safe_text(item.get("category", ""))}
                ),
                "review_statuses": ["pending", "approved", "needs_work", "rejected"],
                "workspace_dir": str(self.paths.workspace_dir),
            }

    def _iter_chunk_records(self) -> list[dict]:
        chunk_fp = self._compute_chunk_fingerprint()
        if self._chunk_cache is not None and self._chunk_cache_fingerprint == chunk_fp:
            return self._chunk_cache
        items = self._load_chunk_records()
        self._chunk_cache = items
        self._chunk_cache_fingerprint = chunk_fp
        return items

    def _load_chunk_records(self) -> list[dict]:
        items: list[dict] = []
        for path in sorted(self.paths.output_dir.glob("chunks_*.jsonl")):
            book_stem = book_stem_from_output_name(path.name, "chunks_")
            book_name = book_name_from_stem(book_stem)
            for record in ed.read_jsonl(path):
                idx = record.get("idx")
                if not isinstance(idx, int):
                    continue
                chapter = safe_text(record.get("chapter", ""))
                chunk_text = safe_text(record.get("chunk_text", ""))
                dialogues = [item for item in record.get("dialogues", []) if isinstance(item, dict)]
                knowledge = [item for item in record.get("knowledge", []) if isinstance(item, dict)]
                chunk_id = f"chunk:{stable_hash(book_name, idx)}"
                items.append(
                    {
                        "id": chunk_id,
                        "book_stem": book_stem,
                        "source_book": book_name,
                        "chapter": chapter,
                        "chunk_idx": idx,
                        "source_excerpt": chunk_text,
                        "preview": shorten(chunk_text, 180),
                        "dialogues": dialogues,
                        "knowledge": knowledge,
                        "dialogue_count": len(dialogues),
                        "knowledge_count": len(knowledge),
                        "path": str(path),
                    }
                )
        return items

    def list_chunks(
        self,
        *,
        search: str = "",
        book: str = "",
        chapter: str = "",
        has_dialogues: str = "",
        has_knowledge: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        items = []
        for item in self._iter_chunk_records():
            if book and not book_names_match(item.get("source_book", ""), book):
                continue
            if chapter and safe_text(item.get("chapter", "")) != safe_text(chapter):
                continue
            if has_dialogues == "yes" and int(item.get("dialogue_count", 0)) <= 0:
                continue
            if has_dialogues == "no" and int(item.get("dialogue_count", 0)) > 0:
                continue
            if has_knowledge == "yes" and int(item.get("knowledge_count", 0)) <= 0:
                continue
            if has_knowledge == "no" and int(item.get("knowledge_count", 0)) > 0:
                continue
            if not self._filter_search(
                [
                    item.get("source_book", ""),
                    item.get("chapter", ""),
                    item.get("source_excerpt", ""),
                ],
                search,
            ):
                continue
            items.append(item)
        items.sort(
            key=lambda item: (
                sort_key_ci(safe_text(item.get("source_book", ""))),
                sort_key_ci(safe_text(item.get("chapter", ""))),
                item.get("chunk_idx", 10**9),
            )
        )
        total = len(items)
        page = items[offset : offset + limit]
        return {
            "total": total,
            "items": [
                {
                    "id": item["id"],
                    "source_book": item.get("source_book", ""),
                    "chapter": item.get("chapter", ""),
                    "chunk_idx": item.get("chunk_idx"),
                    "preview": item.get("preview", ""),
                    "dialogue_count": item.get("dialogue_count", 0),
                    "knowledge_count": item.get("knowledge_count", 0),
                }
                for item in page
            ],
        }

    def get_chunk(self, chunk_id: str) -> dict:
        with self._lock:
            self.refresh()
            found = next((item for item in self._iter_chunk_records() if item.get("id") == chunk_id), None)
            if found is None:
                raise KeyError(chunk_id)
            linked_facts = [
                self._fact_summary(item)
                for item in self._state["facts"].values()
                if book_names_match(item.get("source_book", ""), found.get("source_book", ""))
                and item.get("chunk_idx") == found.get("chunk_idx")
            ]
            linked_samples = [
                self._sample_summary(item)
                for item in self._state["samples"].values()
                if book_names_match(item.get("source_book", ""), found.get("source_book", ""))
                and item.get("chunk_idx") == found.get("chunk_idx")
            ]
            return {
                "item": found,
                "linked_facts": linked_facts,
                "linked_samples": linked_samples,
            }

    def _metadata_path(self) -> Path:
        return self.paths.output_dir / "metadata.json"

    def _metadata_history_path(self) -> Path:
        return self.paths.output_dir / "metadata_history.jsonl"

    def _timeline_raw_path(self) -> Path:
        return self.paths.output_dir / "timeline_resolution_raw.json"

    def _timeline_graph_path(self) -> Path:
        return self.paths.output_dir / "timeline_graph.json"

    def _iter_metadata_events(self) -> list[dict]:
        return [item for item in ed.read_jsonl(self._metadata_history_path()) if isinstance(item, dict)]

    def get_pipeline_snapshot(self) -> dict:
        metadata = read_json(self._metadata_path(), {})
        recent_events = metadata.get("recent_events", []) if isinstance(metadata, dict) else []
        if not isinstance(recent_events, list):
            recent_events = []
        llm_stats = self._llm_trace_stats_cache or {"run_count": 0, "total_traces": 0}
        llm_jobs_total = sum(1 for _ in self._iter_llm_jobs())
        return {
            "metadata": metadata if isinstance(metadata, dict) else {},
            "recent_events": [item for item in recent_events if isinstance(item, dict)],
            "history_count": sum(1 for _ in self._iter_metadata_events()),
            "llm_jobs_count": llm_jobs_total,
            "llm_runs_count": int(llm_stats["run_count"]),
            "llm_traces_count": int(llm_stats["total_traces"]),
            "paths": {
                "metadata": str(self._metadata_path()),
                "metadata_history": str(self._metadata_history_path()),
                "llm_jobs": str(self.paths.llm_jobs_dir),
            },
        }

    def list_metadata_events(
        self,
        *,
        search: str = "",
        event_type: str = "",
        status: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        items = []
        for item in self._iter_metadata_events():
            if event_type and safe_text(item.get("type", "")) != safe_text(event_type):
                continue
            if status and safe_text(item.get("status", "")) != safe_text(status):
                continue
            if not self._filter_search(
                [
                    item.get("type", ""),
                    item.get("status", ""),
                    item.get("current_stage", ""),
                    item.get("current_book", ""),
                    item.get("message", ""),
                    json.dumps(item.get("details", {}), ensure_ascii=False),
                ],
                search,
            ):
                continue
            items.append(item)
        items.sort(key=lambda item: int(item.get("idx", 0)), reverse=True)
        total = len(items)
        page = items[offset : offset + limit]
        return {"total": total, "items": page}

    def _iter_llm_jobs(self) -> list[dict]:
        items = []
        for path in sorted(self.paths.llm_jobs_dir.glob("*.json")):
            item = read_json(path, {})
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item["_path"] = str(path)
            items.append(item)
        return items

    def list_llm_jobs(
        self,
        *,
        search: str = "",
        job_type: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        items = []
        for item in self._iter_llm_jobs():
            if job_type and safe_text(item.get("job_type", "")) != safe_text(job_type):
                continue
            request_payload = item.get("request", {}) if isinstance(item.get("request"), dict) else {}
            response_payload = item.get("response", {}) if isinstance(item.get("response"), dict) else {}
            summary = {
                "id": safe_text(item.get("job_id", "")) or Path(item.get("_path", "")).stem,
                "ts": safe_text(item.get("ts", "")),
                "job_type": safe_text(item.get("job_type", "")),
                "request_preview": shorten(
                    safe_text(request_payload.get("user", ""))
                    or safe_text(request_payload.get("prompt", "")),
                    180,
                ),
                "response_preview": shorten(
                    safe_text(response_payload.get("content", ""))
                    or safe_text(response_payload.get("raw_response", "")),
                    180,
                ),
                "path": item.get("_path", ""),
            }
            if not self._filter_search(
                [
                    summary.get("id", ""),
                    summary.get("job_type", ""),
                    summary.get("request_preview", ""),
                    summary.get("response_preview", ""),
                ],
                search,
            ):
                continue
            items.append(summary)
        items.sort(key=lambda item: safe_text(item.get("ts", "")), reverse=True)
        total = len(items)
        page = items[offset : offset + limit]
        return {"total": total, "items": page}

    def get_llm_job(self, job_id: str) -> dict:
        target = safe_text(job_id)
        for item in self._iter_llm_jobs():
            current_id = safe_text(item.get("job_id", "")) or Path(item.get("_path", "")).stem
            if current_id == target or Path(item.get("_path", "")).name == target:
                return item
        raise KeyError(job_id)

    def get_timeline_overview(self) -> dict:
        graph = read_json(self._timeline_graph_path(), {})
        raw_groups = read_json(self._timeline_raw_path(), [])
        if not isinstance(graph, dict):
            graph = {}
        if not isinstance(raw_groups, list):
            raw_groups = []
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        groups = raw_groups or graph.get("groups", [])
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []
        if not isinstance(groups, list):
            groups = []
        node_types: dict[str, int] = {}
        edge_types: dict[str, int] = {}
        books = set()
        for item in nodes:
            if not isinstance(item, dict):
                continue
            node_type = safe_text(item.get("type", "")) or "unknown"
            node_types[node_type] = node_types.get(node_type, 0) + 1
            book_name = safe_text(item.get("book", ""))
            if book_name:
                books.add(book_name)
        for item in groups:
            if isinstance(item, dict):
                book_name = safe_text(item.get("book_name", ""))
                if book_name:
                    books.add(book_name)
        for item in edges:
            if not isinstance(item, dict):
                continue
            edge_type = safe_text(item.get("type", "")) or "related_to"
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        return {
            "available": self._timeline_graph_path().exists() or self._timeline_raw_path().exists(),
            "counts": {
                "nodes": len([item for item in nodes if isinstance(item, dict)]),
                "edges": len([item for item in edges if isinstance(item, dict)]),
                "groups": len([item for item in groups if isinstance(item, dict)]),
            },
            "node_types": node_types,
            "edge_types": edge_types,
            "books": sorted(books, key=sort_key_ci),
            "paths": {
                "timeline_graph": str(self._timeline_graph_path()),
                "timeline_raw": str(self._timeline_raw_path()),
            },
        }

    def list_timeline_nodes(
        self,
        *,
        search: str = "",
        node_type: str = "",
        limit: int = 300,
        offset: int = 0,
    ) -> dict:
        graph = read_json(self._timeline_graph_path(), {})
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        if not isinstance(nodes, list):
            nodes = []
        items = []
        for item in nodes:
            if not isinstance(item, dict):
                continue
            if node_type and safe_text(item.get("type", "")) != safe_text(node_type):
                continue
            if not self._filter_search(
                [
                    item.get("id", ""),
                    item.get("label", ""),
                    item.get("type", ""),
                    item.get("summary", ""),
                    item.get("chapter", ""),
                    item.get("book", ""),
                ],
                search,
            ):
                continue
            items.append(item)
        items.sort(key=lambda item: (sort_key_ci(safe_text(item.get("type", ""))), sort_key_ci(safe_text(item.get("label", "")))))
        total = len(items)
        page = items[offset : offset + limit]
        return {"total": total, "items": page}

    def list_timeline_edges(
        self,
        *,
        search: str = "",
        relation_type: str = "",
        limit: int = 300,
        offset: int = 0,
    ) -> dict:
        graph = read_json(self._timeline_graph_path(), {})
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        edges = graph.get("edges", []) if isinstance(graph, dict) else []
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []
        label_by_id = {
            safe_text(item.get("id", "")): safe_text(item.get("label", ""))
            for item in nodes
            if isinstance(item, dict)
        }
        items = []
        for item in edges:
            if not isinstance(item, dict):
                continue
            if relation_type and safe_text(item.get("type", "")) != safe_text(relation_type):
                continue
            enriched = dict(item)
            enriched["source_label"] = label_by_id.get(safe_text(item.get("source", "")), safe_text(item.get("source", "")))
            enriched["target_label"] = label_by_id.get(safe_text(item.get("target", "")), safe_text(item.get("target", "")))
            if not self._filter_search(
                [
                    enriched.get("type", ""),
                    enriched.get("source", ""),
                    enriched.get("target", ""),
                    enriched.get("source_label", ""),
                    enriched.get("target_label", ""),
                    enriched.get("book", ""),
                    enriched.get("chapter", ""),
                    enriched.get("evidence", ""),
                ],
                search,
            ):
                continue
            items.append(enriched)
        items.sort(key=lambda item: (sort_key_ci(safe_text(item.get("type", ""))), sort_key_ci(safe_text(item.get("source_label", ""))), sort_key_ci(safe_text(item.get("target_label", "")))))
        total = len(items)
        page = items[offset : offset + limit]
        return {"total": total, "items": page}

    def list_timeline_groups(
        self,
        *,
        search: str = "",
        book: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        groups = read_json(self._timeline_raw_path(), [])
        if not isinstance(groups, list):
            groups = []
        items = []
        for index, item in enumerate(groups, 1):
            if not isinstance(item, dict):
                continue
            if book and not book_names_match(item.get("book_name", ""), book):
                continue
            summary = {
                "id": f"group:{stable_hash(item.get('book_name', ''), item.get('chapter', ''), index)}",
                "book_name": safe_text(item.get("book_name", "")),
                "chapter": safe_text(item.get("chapter", "")),
                "chunk_indices": item.get("chunk_indices", []),
                "fact_count": len([fact for fact in item.get("facts", []) if isinstance(fact, dict)]),
                "event_count": len([event for event in item.get("events", []) if isinstance(event, dict)]),
                "relation_count": len([relation for relation in item.get("relations", []) if isinstance(relation, dict)]),
                "entity_count": len([entity for entity in item.get("entities", []) if isinstance(entity, dict)]),
                "preview": "",
                "raw": item,
            }
            facts = [fact for fact in item.get("facts", []) if isinstance(fact, dict)]
            if facts:
                summary["preview"] = shorten(" | ".join(safe_text(fact.get("fact", "")) for fact in facts[:3]), 180)
            elif item.get("events"):
                summary["preview"] = shorten(" | ".join(safe_text(event.get("summary", "")) for event in item.get("events", [])[:3]), 180)
            if not self._filter_search(
                [
                    summary.get("book_name", ""),
                    summary.get("chapter", ""),
                    summary.get("preview", ""),
                ],
                search,
            ):
                continue
            items.append(summary)
        items.sort(key=lambda item: (sort_key_ci(safe_text(item.get("book_name", ""))), sort_key_ci(safe_text(item.get("chapter", "")))))
        total = len(items)
        page = items[offset : offset + limit]
        return {"total": total, "items": page}

    def _filter_search(self, haystack: list[str], query: str) -> bool:
        if not query:
            return True
        target = ed.normalize_dedup_text(" ".join(haystack))
        terms = [term for term in ed.normalize_dedup_text(query).split() if term]
        return all(term in target for term in terms)

    def _fact_summary(self, item: dict) -> dict:
        return {
            "id": item["id"],
            "category": item.get("category", ""),
            "subject": item.get("subject", ""),
            "fact": item.get("fact", ""),
            "fact_preview": shorten(item.get("fact", ""), 160),
            "time_scope": item.get("time_scope", ""),
            "source_book": item.get("source_book", ""),
            "chapter": item.get("chapter", ""),
            "chunk_idx": item.get("chunk_idx"),
            "review_status": item.get("review_status", "pending"),
            "review_score": item.get("review_score"),
            "theme_ids": item.get("theme_ids", []),
            "origin": item.get("origin", "base"),
        }

    def list_facts(
        self,
        *,
        search: str = "",
        category: str = "",
        book: str = "",
        review_status: str = "",
        theme_id: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        with self._lock:
            self.refresh()
            items = []
            for item in self._state["facts"].values():
                if category and safe_text(item.get("category", "")) != category:
                    continue
                if book and safe_text(item.get("source_book", "")) != book:
                    continue
                if review_status and safe_text(item.get("review_status", "")) != review_status:
                    continue
                if theme_id and theme_id not in item.get("theme_ids", []):
                    continue
                if not self._filter_search(
                    [
                        item.get("subject", ""),
                        item.get("fact", ""),
                        item.get("source_book", ""),
                        item.get("chapter", ""),
                    ],
                    search,
                ):
                    continue
                items.append(item)
            items.sort(
                key=lambda item: (
                    sort_key_ci(safe_text(item.get("source_book", ""))),
                    item.get("chunk_idx") if isinstance(item.get("chunk_idx"), int) else 10**9,
                    sort_key_ci(safe_text(item.get("subject", ""))),
                    sort_key_ci(safe_text(item.get("fact", ""))),
                )
            )
            total = len(items)
            page = items[offset : offset + limit]
            return {"total": total, "items": [self._fact_summary(item) for item in page]}

    def get_fact(self, fact_id: str) -> dict:
        with self._lock:
            self.refresh()
            fact = self._state["facts"].get(fact_id)
            if fact is None:
                raise KeyError(fact_id)
            relations = [
                relation
                for relation in self._state["relations"].values()
                if relation.get("source_fact_id") == fact_id or relation.get("target_fact_id") == fact_id
            ]
            themes = [self._state["themes"][theme_id] for theme_id in fact.get("theme_ids", []) if theme_id in self._state["themes"]]
            linked_samples = [
                self._sample_summary(sample)
                for sample in self._state["samples"].values()
                if fact_id in sample.get("linked_fact_ids", [])
            ]
            return {
                "item": fact,
                "original": self._state["base_facts"].get(fact_id),
                "relations": relations,
                "themes": themes,
                "linked_samples": linked_samples,
            }

    def create_fact(self, payload: dict) -> dict:
        with self._lock:
            self.refresh()
            fact_id = f"fact:manual:{uuid.uuid4().hex}"
            record = {
                "id": fact_id,
                "kind": "fact",
                "category": safe_text(payload.get("category", "")) or "character",
                "subject": safe_text(payload.get("subject", "")),
                "fact": safe_text(payload.get("fact", "")),
                "time_scope": safe_text(payload.get("time_scope", "")) or "unclear",
                "source_book": safe_text(payload.get("source_book", "")),
                "chapter": safe_text(payload.get("chapter", "")),
                "chunk_idx": payload.get("chunk_idx") if isinstance(payload.get("chunk_idx"), int) else None,
                "source_excerpt": safe_text(payload.get("source_excerpt", "")),
                "source_file": "",
                "theme_ids": list(dict.fromkeys(payload.get("theme_ids", []) or [])),
                "review_status": safe_text(payload.get("review_status", "")) or "pending",
                "review_score": payload.get("review_score"),
                "review_note": safe_text(payload.get("review_note", "")),
                "updated_at": now_iso(),
                "origin": "editor",
            }
            self._record_operation("facts", fact_id, before=None, after=record, note="create")
            self.refresh()
            return self._state["facts"][fact_id]

    def update_fact(self, fact_id: str, patch: dict) -> dict:
        with self._lock:
            self.refresh()
            before = self._state["facts"].get(fact_id)
            if before is None:
                raise KeyError(fact_id)
            after = dict(before)
            for key in (
                "category",
                "subject",
                "fact",
                "time_scope",
                "source_book",
                "chapter",
                "source_excerpt",
                "review_status",
                "review_note",
            ):
                if key in patch:
                    after[key] = safe_text(patch.get(key, ""))
            if "chunk_idx" in patch:
                after["chunk_idx"] = patch["chunk_idx"] if isinstance(patch["chunk_idx"], int) else None
            if "review_score" in patch:
                score = patch["review_score"]
                after["review_score"] = score if isinstance(score, (int, float)) else None
            if "theme_ids" in patch:
                after["theme_ids"] = list(dict.fromkeys(patch.get("theme_ids", []) or []))
            after["updated_at"] = now_iso()
            self._record_operation("facts", fact_id, before=before, after=after, note="update")
            self.refresh()
            return self._state["facts"][fact_id]

    def delete_fact(self, fact_id: str) -> dict:
        with self._lock:
            self.refresh()
            before = self._state["facts"].get(fact_id)
            if before is None:
                raise KeyError(fact_id)
            self._record_operation("facts", fact_id, before=before, after=None, note="delete")
            self.refresh()
            return {"deleted": fact_id}

    def _sample_summary(self, item: dict) -> dict:
        return {
            "id": item["id"],
            "kind": item.get("kind", ""),
            "user_preview": shorten(item.get("user", ""), 120),
            "assistant_preview": shorten(item.get("assistant", ""), 160),
            "source_book": item.get("source_book", ""),
            "chapter": item.get("chapter", ""),
            "chunk_idx": item.get("chunk_idx"),
            "review_status": item.get("review_status", "pending"),
            "linked_fact_ids": item.get("linked_fact_ids", []),
            "theme_ids": item.get("theme_ids", []),
            "origin": item.get("origin", "base"),
        }

    def list_samples(
        self,
        *,
        kind: str,
        search: str = "",
        review_status: str = "",
        linked_fact_id: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        with self._lock:
            self.refresh()
            items = []
            for item in self._state["samples"].values():
                if safe_text(item.get("kind", "")) != kind:
                    continue
                if review_status and safe_text(item.get("review_status", "")) != review_status:
                    continue
                if linked_fact_id and linked_fact_id not in item.get("linked_fact_ids", []):
                    continue
                if not self._filter_search(
                    [
                        item.get("user", ""),
                        item.get("assistant", ""),
                        item.get("source_book", ""),
                        item.get("chapter", ""),
                    ],
                    search,
                ):
                    continue
                items.append(item)
            items.sort(
                key=lambda item: (
                    sort_key_ci(safe_text(item.get("source_book", ""))),
                    sort_key_ci(safe_text(item.get("assistant", ""))),
                )
            )
            total = len(items)
            page = items[offset : offset + limit]
            return {"total": total, "items": [self._sample_summary(item) for item in page]}

    def get_sample(self, sample_id: str) -> dict:
        with self._lock:
            self.refresh()
            sample = self._state["samples"].get(sample_id)
            if sample is None:
                raise KeyError(sample_id)
            themes = [self._state["themes"][theme_id] for theme_id in sample.get("theme_ids", []) if theme_id in self._state["themes"]]
            linked_facts = [self._fact_summary(self._state["facts"][fact_id]) for fact_id in sample.get("linked_fact_ids", []) if fact_id in self._state["facts"]]
            return {
                "item": sample,
                "original": self._state["base_samples"].get(sample_id),
                "themes": themes,
                "linked_facts": linked_facts,
            }

    def create_sample(self, payload: dict) -> dict:
        with self._lock:
            self.refresh()
            kind = safe_text(payload.get("kind", "")) or "voice"
            if kind not in {"voice", "synth"}:
                raise ValueError("Invalid sample kind")
            sample_id = f"sample:{kind}:manual:{uuid.uuid4().hex}"
            system_text = safe_text(payload.get("system", "")) or ed.Config().character_system_prompt
            user_text = safe_text(payload.get("user", ""))
            assistant_text = safe_text(payload.get("assistant", ""))
            record = {
                "id": sample_id,
                "kind": kind,
                "messages": build_messages(system_text, user_text, assistant_text),
                "system": system_text,
                "user": user_text,
                "assistant": assistant_text,
                "source_book": safe_text(payload.get("source_book", "")),
                "chapter": safe_text(payload.get("chapter", "")),
                "chunk_idx": payload.get("chunk_idx") if isinstance(payload.get("chunk_idx"), int) else None,
                "source_excerpt": safe_text(payload.get("source_excerpt", "")),
                "source_file": "",
                "linked_fact_ids": list(dict.fromkeys(payload.get("linked_fact_ids", []) or [])),
                "theme_ids": list(dict.fromkeys(payload.get("theme_ids", []) or [])),
                "review_status": safe_text(payload.get("review_status", "")) or "pending",
                "review_score": payload.get("review_score"),
                "review_note": safe_text(payload.get("review_note", "")),
                "updated_at": now_iso(),
                "origin": "editor",
            }
            self._record_operation("samples", sample_id, before=None, after=record, note="create")
            self.refresh()
            return self._state["samples"][sample_id]

    def update_sample(self, sample_id: str, patch: dict) -> dict:
        with self._lock:
            self.refresh()
            before = self._state["samples"].get(sample_id)
            if before is None:
                raise KeyError(sample_id)
            after = dict(before)
            system_text = safe_text(patch.get("system", before.get("system", ""))) if "system" in patch else before.get("system", "")
            user_text = safe_text(patch.get("user", before.get("user", ""))) if "user" in patch else before.get("user", "")
            assistant_text = safe_text(patch.get("assistant", before.get("assistant", ""))) if "assistant" in patch else before.get("assistant", "")
            after["system"] = safe_text(system_text)
            after["user"] = safe_text(user_text)
            after["assistant"] = safe_text(assistant_text)
            after["messages"] = build_messages(after["system"], after["user"], after["assistant"])
            for key in ("source_book", "chapter", "source_excerpt", "review_status", "review_note"):
                if key in patch:
                    after[key] = safe_text(patch.get(key, ""))
            if "chunk_idx" in patch:
                after["chunk_idx"] = patch["chunk_idx"] if isinstance(patch["chunk_idx"], int) else None
            if "linked_fact_ids" in patch:
                after["linked_fact_ids"] = list(dict.fromkeys(patch.get("linked_fact_ids", []) or []))
            if "theme_ids" in patch:
                after["theme_ids"] = list(dict.fromkeys(patch.get("theme_ids", []) or []))
            if "review_score" in patch:
                score = patch["review_score"]
                after["review_score"] = score if isinstance(score, (int, float)) else None
            after["updated_at"] = now_iso()
            self._record_operation("samples", sample_id, before=before, after=after, note="update")
            self.refresh()
            return self._state["samples"][sample_id]

    def delete_sample(self, sample_id: str) -> dict:
        with self._lock:
            self.refresh()
            before = self._state["samples"].get(sample_id)
            if before is None:
                raise KeyError(sample_id)
            self._record_operation("samples", sample_id, before=before, after=None, note="delete")
            self.refresh()
            return {"deleted": sample_id}

    def list_themes(self) -> list[dict]:
        with self._lock:
            self.refresh()
            return sorted(self._state["themes"].values(), key=lambda item: sort_key_ci(safe_text(item.get("name", ""))))

    def create_theme(self, payload: dict) -> dict:
        with self._lock:
            self.refresh()
            theme_id = f"theme:{uuid.uuid4().hex}"
            theme = {
                "id": theme_id,
                "name": safe_text(payload.get("name", "")) or "New theme",
                "description": safe_text(payload.get("description", "")),
                "color": safe_text(payload.get("color", "")) or "#ffb347",
                "updated_at": now_iso(),
                "origin": "editor",
            }
            self._record_operation("themes", theme_id, before=None, after=theme, note="create")
            self.refresh()
            return self._state["themes"][theme_id]

    def update_theme(self, theme_id: str, patch: dict) -> dict:
        with self._lock:
            self.refresh()
            before = self._state["themes"].get(theme_id)
            if before is None:
                raise KeyError(theme_id)
            after = dict(before)
            for key in ("name", "description", "color"):
                if key in patch:
                    after[key] = safe_text(patch.get(key, ""))
            after["updated_at"] = now_iso()
            self._record_operation("themes", theme_id, before=before, after=after, note="update")
            self.refresh()
            return self._state["themes"][theme_id]

    def delete_theme(self, theme_id: str) -> dict:
        with self._lock:
            self.refresh()
            before = self._state["themes"].get(theme_id)
            if before is None:
                raise KeyError(theme_id)
            self._record_operation("themes", theme_id, before=before, after=None, note="delete")
            self.refresh()
            return {"deleted": theme_id}

    def merge_themes(self, source_theme_id: str, target_theme_id: str) -> dict:
        with self._lock:
            self.refresh()
            if source_theme_id == target_theme_id:
                raise ValueError("Theme ids must be different")
            if source_theme_id not in self._state["themes"] or target_theme_id not in self._state["themes"]:
                raise KeyError("Theme not found")
            for fact in list(self._state["facts"].values()):
                theme_ids = list(fact.get("theme_ids", []))
                if source_theme_id in theme_ids:
                    updated = [target_theme_id if item == source_theme_id else item for item in theme_ids]
                    self.update_fact(fact["id"], {"theme_ids": list(dict.fromkeys(updated))})
            for sample in list(self._state["samples"].values()):
                theme_ids = list(sample.get("theme_ids", []))
                if source_theme_id in theme_ids:
                    updated = [target_theme_id if item == source_theme_id else item for item in theme_ids]
                    self.update_sample(sample["id"], {"theme_ids": list(dict.fromkeys(updated))})
            deleted = self.delete_theme(source_theme_id)
            self.refresh()
            return {"merged": source_theme_id, "into": target_theme_id, "deleted": deleted}

    def list_relations(self, fact_id: str = "") -> list[dict]:
        with self._lock:
            self.refresh()
            items = list(self._state["relations"].values())
            if fact_id:
                items = [item for item in items if item.get("source_fact_id") == fact_id or item.get("target_fact_id") == fact_id]
            items.sort(
                key=lambda item: (
                    sort_key_ci(safe_text(item.get("source_fact_id", ""))),
                    sort_key_ci(safe_text(item.get("relation_type", ""))),
                    sort_key_ci(safe_text(item.get("target_fact_id", ""))),
                )
            )
            return items

    def create_relation(self, payload: dict) -> dict:
        with self._lock:
            self.refresh()
            relation_id = f"relation:{uuid.uuid4().hex}"
            source_fact_id = safe_text(payload.get("source_fact_id", ""))
            target_fact_id = safe_text(payload.get("target_fact_id", ""))
            if source_fact_id not in self._state["facts"] or target_fact_id not in self._state["facts"]:
                raise KeyError("Relation endpoints must point to existing facts")
            relation = {
                "id": relation_id,
                "source_fact_id": source_fact_id,
                "target_fact_id": target_fact_id,
                "relation_type": safe_text(payload.get("relation_type", "")) or "related_to",
                "note": safe_text(payload.get("note", "")),
                "updated_at": now_iso(),
                "origin": "editor",
            }
            self._record_operation("relations", relation_id, before=None, after=relation, note="create")
            self.refresh()
            return self._state["relations"][relation_id]

    def update_relation(self, relation_id: str, patch: dict) -> dict:
        with self._lock:
            self.refresh()
            before = self._state["relations"].get(relation_id)
            if before is None:
                raise KeyError(relation_id)
            after = dict(before)
            for key in ("relation_type", "note"):
                if key in patch:
                    after[key] = safe_text(patch.get(key, ""))
            for key in ("source_fact_id", "target_fact_id"):
                if key in patch:
                    candidate = safe_text(patch.get(key, ""))
                    if candidate and candidate not in self._state["facts"]:
                        raise KeyError(f"{key} not found")
                    after[key] = candidate
            after["updated_at"] = now_iso()
            self._record_operation("relations", relation_id, before=before, after=after, note="update")
            self.refresh()
            return self._state["relations"][relation_id]

    def delete_relation(self, relation_id: str) -> dict:
        with self._lock:
            self.refresh()
            before = self._state["relations"].get(relation_id)
            if before is None:
                raise KeyError(relation_id)
            self._record_operation("relations", relation_id, before=before, after=None, note="delete")
            self.refresh()
            return {"deleted": relation_id}

    def _iter_llm_trace_records(self) -> list[dict]:
        root = self.paths.llm_traces_dir
        if not root.exists():
            return []
        try:
            result = subprocess.run(
                ["find", str(root), "-type", "f", "-name", "*.json", "-printf", "%T@|%p\n"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                records = []
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or "|" not in line:
                        continue
                    raw_mtime, raw_path = line.split("|", 1)
                    try:
                        mtime = float(raw_mtime)
                    except Exception:
                        continue
                    records.append({"path": Path(raw_path), "mtime": mtime})
                if records:
                    return sorted(records, key=lambda item: str(item["path"]))
        except Exception:
            pass
        records = []
        for path in sorted(root.rglob("*.json")):
            try:
                records.append({"path": path, "mtime": path.stat().st_mtime})
            except OSError:
                continue
        return records

    def _iter_llm_trace_files(self) -> list[Path]:
        return [item["path"] for item in self._iter_llm_trace_records()]

    def _fast_llm_trace_stats(self) -> dict[str, Any]:
        runs: dict[str, dict[str, Any]] = {}
        total = 0
        for record in self._iter_llm_trace_records():
            path = record["path"]
            total += 1
            run_id = self._llm_run_id_for_trace(path)
            latest_at = datetime.fromtimestamp(record["mtime"], tz=timezone.utc).astimezone().isoformat(timespec="seconds")
            item = runs.setdefault(
                run_id,
                {
                    "id": run_id,
                    "trace_count": 0,
                    "latest_at": "",
                    "providers": [],
                    "models": [],
                },
            )
            item["trace_count"] += 1
            if latest_at > safe_text(item.get("latest_at", "")):
                item["latest_at"] = latest_at
        run_items = sorted(runs.values(), key=lambda item: safe_text(item.get("latest_at", "")), reverse=True)
        return {
            "total_traces": total,
            "runs": run_items,
            "run_count": len(run_items),
        }

    def _llm_run_id_for_trace(self, path: Path) -> str:
        root = self.paths.llm_traces_dir
        try:
            relative = path.relative_to(root)
        except ValueError:
            return "legacy_flat"
        parts = relative.parts
        if len(parts) > 1:
            return parts[0]
        return "legacy_flat"

    def _llm_trace_summary(self, trace: dict, path: Path) -> dict:
        trace_id = safe_text(trace.get("trace_id", "")) or path.stem
        attempts = trace.get("attempts", [])
        last_attempt = attempts[-1] if isinstance(attempts, list) and attempts else {}
        request_payload = trace.get("request_payload", {})
        messages = request_payload.get("messages", []) if isinstance(request_payload, dict) else []
        system_text, user_text, _ = extract_messages_fields(messages if isinstance(messages, list) else [])
        return {
            "id": f"{self._llm_run_id_for_trace(path)}/{trace_id}",
            "trace_id": trace_id,
            "run_id": self._llm_run_id_for_trace(path),
            "path": str(path),
            "created_at": safe_text(trace.get("created_at", "")),
            "updated_at": safe_text(trace.get("updated_at", "")),
            "provider": safe_text(trace.get("provider", "")),
            "model": safe_text(trace.get("model", "")),
            "log_prefix": safe_text(trace.get("log_prefix", "")),
            "max_tokens": trace.get("max_tokens"),
            "temperature": trace.get("temperature"),
            "response_format": trace.get("response_format"),
            "system_preview": shorten(system_text, 120),
            "user_preview": shorten(user_text, 180),
            "attempt_count": len(attempts) if isinstance(attempts, list) else 0,
            "last_status": safe_text(last_attempt.get("status", "")),
            "last_content_preview": shorten(safe_text(last_attempt.get("content", "")), 180),
        }

    def _resolve_trace_path(self, trace_ref: str) -> Path:
        trace_ref = safe_text(trace_ref)
        if not trace_ref:
            raise KeyError("Empty trace id")
        direct = self.paths.llm_traces_dir / trace_ref
        if direct.exists():
            return direct
        if not direct.suffix:
            direct_json = direct.with_suffix(".json")
            if direct_json.exists():
                return direct_json
        for path in self._iter_llm_trace_files():
            trace = read_json(path, {})
            trace_id = safe_text(trace.get("trace_id", "")) or path.stem
            full_id = f"{self._llm_run_id_for_trace(path)}/{trace_id}"
            if trace_ref in {trace_id, full_id, path.name, str(path.relative_to(self.paths.llm_traces_dir))}:
                return path
        raise KeyError(trace_ref)

    def list_llm_runs(self) -> list[dict]:
        stats = self._fast_llm_trace_stats()
        self._llm_trace_stats_cache = stats
        return stats["runs"]

    def list_llm_traces(self, *, run_id: str = "", search: str = "", limit: int = 200, offset: int = 0) -> dict:
        items = []
        run_counts: dict[str, dict[str, Any]] = {}
        for record in self._iter_llm_trace_records():
            path = record["path"]
            current_run_id = self._llm_run_id_for_trace(path)
            if run_id and current_run_id != run_id:
                continue
            trace = read_json(path, {})
            if not isinstance(trace, dict):
                trace = {}
            summary = self._llm_trace_summary(trace, path)
            if search and not self._filter_search(
                [
                    summary.get("trace_id", ""),
                    summary.get("run_id", ""),
                    summary.get("model", ""),
                    summary.get("provider", ""),
                    summary.get("log_prefix", ""),
                    summary.get("user_preview", ""),
                    summary.get("last_content_preview", ""),
                ],
                search,
            ):
                continue
            items.append(summary)
            bucket = run_counts.setdefault(
                summary["run_id"],
                {"id": summary["run_id"], "trace_count": 0, "latest_at": "", "providers": [], "models": []},
            )
            bucket["trace_count"] += 1
            if safe_text(summary.get("updated_at", "")) > safe_text(bucket.get("latest_at", "")):
                bucket["latest_at"] = safe_text(summary.get("updated_at", ""))
        self._llm_trace_stats_cache = {
            "total_traces": len(items),
            "runs": sorted(run_counts.values(), key=lambda item: safe_text(item.get("latest_at", "")), reverse=True),
            "run_count": len(run_counts),
        }
        items.sort(key=lambda item: safe_text(item.get("created_at", "")), reverse=True)
        total = len(items)
        page = items[offset : offset + limit]
        return {"total": total, "items": page}

    def get_llm_trace(self, trace_ref: str) -> dict:
        path = self._resolve_trace_path(trace_ref)
        trace = read_json(path, {})
        if not isinstance(trace, dict):
            raise KeyError(trace_ref)
        summary = self._llm_trace_summary(trace, path)
        request_payload = trace.get("request_payload", {})
        messages = request_payload.get("messages", []) if isinstance(request_payload, dict) else []
        system_text, user_text, assistant_text = extract_messages_fields(messages if isinstance(messages, list) else [])
        return {
            "summary": summary,
            "trace": trace,
            "editable_request": {
                "system": system_text,
                "user": user_text,
                "assistant": assistant_text,
                "model_override": safe_text(trace.get("model", "")),
                "max_tokens": trace.get("max_tokens"),
                "temperature": trace.get("temperature"),
                "response_format": trace.get("response_format"),
                "log_prefix": safe_text(trace.get("log_prefix", "")),
            },
        }

    def _parse_response_format(self, value: Any) -> Any:
        if value in (None, "", {}):
            return None
        if isinstance(value, (dict, list)):
            return value
        text = safe_text(value)
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return text

    def run_llm_prompt(
        self,
        payload: dict,
        *,
        rerun_from_trace: str = "",
    ) -> dict:
        config = self._editor_config()
        config.llm_trace_enabled = True
        config.llm_trace_run_id = f"studio_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        system = safe_text(payload.get("system", ""))
        user = safe_text(payload.get("user", ""))
        if not user:
            raise ValueError("User prompt is empty")
        max_tokens = payload.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            max_tokens = 800
        temperature = payload.get("temperature")
        if not isinstance(temperature, (int, float)):
            temperature = 0.2
        model_override = safe_text(payload.get("model_override", "")) or config.model
        log_prefix = safe_text(payload.get("log_prefix", "")) or "[studio][llm]"
        response_format = self._parse_response_format(payload.get("response_format"))
        trace_id = ed.next_llm_trace_id(log_prefix)

        if ":11434" in config.api_base:
            result = ed.call_llm_ollama_native(
                config,
                system,
                user,
                max_tokens=max_tokens,
                response_format=response_format,
                log_prefix=log_prefix,
                temperature=float(temperature),
                trace_id=trace_id,
                model_override=model_override,
            )
        else:
            client = self._llm_client()
            result = ed.call_llm_openai(
                client,
                config,
                system,
                user,
                max_tokens=max_tokens,
                response_format=response_format,
                log_prefix=log_prefix,
                temperature=float(temperature),
                trace_id=trace_id,
                model_override=model_override,
            )

        trace_path = ed.get_llm_trace_dir(config) / f"{trace_id}.json"
        trace = read_json(trace_path, {})
        run_record = {
            "job_id": f"llm_run:{uuid.uuid4().hex}",
            "ts": now_iso(),
            "job_type": "manual_llm_run" if not rerun_from_trace else "rerun_llm_trace",
            "source_trace": rerun_from_trace or None,
            "request": {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model_override": model_override,
                "response_format": response_format,
                "log_prefix": log_prefix,
                "run_id": config.llm_trace_run_id,
                "trace_id": trace_id,
            },
            "response": {
                "content": result,
                "trace_path": str(trace_path),
            },
        }
        self._record_llm_job(run_record["job_type"], run_record["request"], run_record["response"])
        return {
            "run_id": config.llm_trace_run_id,
            "trace_id": trace_id,
            "trace_ref": f"{config.llm_trace_run_id}/{trace_id}",
            "trace_path": str(trace_path),
            "content": result,
            "trace": trace,
        }

    def rerun_llm_trace(self, trace_ref: str, patch: Optional[dict] = None) -> dict:
        detail = self.get_llm_trace(trace_ref)
        editable = dict(detail["editable_request"])
        if patch:
            editable.update(patch)
        return self.run_llm_prompt(editable, rerun_from_trace=trace_ref)

    def _editor_config(self) -> ed.Config:
        if self._config_cache is not None:
            return self._config_cache
        meta = read_json(self.paths.output_dir / "metadata.json", {})
        config = ed.Config(
            output_dir=str(self.paths.output_dir),
            api_base=safe_text(meta.get("api_base", "")) or ed.Config.api_base,
            model=safe_text(meta.get("model", "")) or ed.Config.model,
        )
        config.knowledge_extract_model = safe_text(meta.get("knowledge_extract_model", "")) or config.model
        config.knowledge_validate_model = safe_text(meta.get("knowledge_validate_model", "")) or config.model
        config.knowledge_link_model = safe_text(meta.get("knowledge_link_model", "")) or config.model
        config.knowledge_extract_model_secondary = safe_text(meta.get("knowledge_extract_model_secondary", "")) or ed.Config.knowledge_extract_model_secondary
        config.knowledge_arbiter_model = safe_text(meta.get("knowledge_arbiter_model", "")) or config.model
        self._config_cache = config
        return config

    def _llm_client(self) -> Any:
        config = self._editor_config()
        if getattr(ed, "_openai_import_error", None) is not None:
            raise RuntimeError("openai package is required for LLM actions. Install with: pip install openai")
        return ed.OpenAI(base_url=config.api_base, api_key=config.api_key)

    def _call_llm_auto(
        self,
        config: ed.Config,
        system: str,
        user: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
        log_prefix: str = "",
        model_override: Optional[str] = None,
        response_format: Optional[Any] = None,
    ) -> Optional[str]:
        trace_id = ed.next_llm_trace_id(log_prefix)
        if ":11434" in config.api_base:
            return ed.call_llm_ollama_native(
                config, system, user,
                max_tokens=max_tokens,
                response_format=response_format,
                log_prefix=log_prefix,
                temperature=temperature,
                trace_id=trace_id,
                model_override=model_override,
            )
        client = self._llm_client()
        return ed.call_llm_openai(
            client, config, system, user,
            max_tokens=max_tokens,
            response_format=response_format,
            log_prefix=log_prefix,
            temperature=temperature,
            trace_id=trace_id,
            model_override=model_override,
        )

    def _record_llm_job(self, job_type: str, request_payload: dict, response_payload: dict):
        job = {
            "job_id": f"{job_type}:{uuid.uuid4().hex}",
            "ts": now_iso(),
            "job_type": job_type,
            "request": request_payload,
            "response": response_payload,
        }
        path = self.paths.llm_jobs_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{job_type}_{uuid.uuid4().hex[:8]}.json"
        write_json_atomic(path, job)

    def _parse_reanalysis_lines(self, text: str, facts: list[dict]) -> list[dict]:
        fact_by_id = {fact["id"]: fact for fact in facts}
        suggestions = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or "|" not in line:
                continue
            parts = [part.strip() for part in line.split("|")]
            verdict = parts[0].lower()
            if verdict == "drop" and len(parts) >= 3:
                suggestions.append(
                    {
                        "action": "drop",
                        "fact_id": parts[1],
                        "reason": "|".join(parts[2:]).strip(),
                        "original": fact_by_id.get(parts[1]),
                    }
                )
            elif verdict == "keep" and len(parts) >= 7:
                suggestions.append(
                    {
                        "action": "keep",
                        "fact_id": parts[1],
                        "fact": {
                            "category": parts[2],
                            "subject": parts[3],
                            "fact": parts[4],
                            "time_scope": parts[5],
                        },
                        "reason": "|".join(parts[6:]).strip(),
                        "original": fact_by_id.get(parts[1]),
                    }
                )
        return suggestions

    def reanalyze_facts(self, fact_ids: list[str], *, bundle: bool = False, model_override: str = "") -> dict:
        with self._lock:
            self.refresh()
            facts = [self._state["facts"][fact_id] for fact_id in fact_ids if fact_id in self._state["facts"]]
        if not facts:
            raise ValueError("No facts selected")

        excerpts = []
        for fact in facts:
            excerpts.append(
                {
                    "fact_id": fact["id"],
                    "source_book": fact.get("source_book", ""),
                    "chapter": fact.get("chapter", ""),
                    "chunk_idx": fact.get("chunk_idx"),
                    "source_excerpt": fact.get("source_excerpt", ""),
                    "fact": strip_editor_fields(fact),
                }
            )

        system = (
            "You review extracted facts for a structured knowledge base about the world of Echo. "
            "Do not invent details. Output plain lines only."
        )
        user = "\n".join(
            [
                "Review extracted knowledge facts against source excerpts.",
                "Only keep facts that are grounded in the excerpt and are autonomous for a world knowledge base.",
                "If a fact is salvageable, rewrite it into a precise autonomous fact.",
                "Return one line per input fact.",
                "Format:",
                "keep|FACT_ID|category|subject|fact|time_scope|reason",
                "drop|FACT_ID|reason",
                "",
                f"BUNDLE_MODE={'yes' if bundle else 'no'}",
                "",
                "INPUT:",
                json.dumps(excerpts, ensure_ascii=False, indent=2),
            ]
        )

        config = self._editor_config()
        response = self._call_llm_auto(
            config, system, user,
            max_tokens=1200,
            temperature=0.0,
            log_prefix="[studio][reanalyze]",
            model_override=model_override or config.knowledge_extract_model or config.model,
        )
        parsed = self._parse_reanalysis_lines(response or "", facts)
        self._record_llm_job(
            "reanalyze_facts",
            {
                "fact_ids": fact_ids,
                "bundle": bundle,
                "model_override": model_override or config.knowledge_extract_model or config.model,
                "system": system,
                "prompt": user,
            },
            {"raw_response": response, "parsed": parsed},
        )
        return {"suggestions": parsed, "raw_response": response or ""}

    def generate_sample_from_facts(
        self,
        fact_ids: list[str],
        *,
        kind: str = "synth",
        instruction: str = "",
        model_override: str = "",
    ) -> dict:
        with self._lock:
            self.refresh()
            facts = [self._state["facts"][fact_id] for fact_id in fact_ids if fact_id in self._state["facts"]]
            voice_pairs = [
                {"messages": item.get("messages", [])}
                for item in self._state["samples"].values()
                if item.get("kind") == "voice" and isinstance(item.get("messages"), list)
            ]
        if not facts:
            raise ValueError("No facts selected")

        fact_lines = [json.dumps(strip_editor_fields(fact), ensure_ascii=False) for fact in facts]
        style_examples = ed.pick_style_examples(voice_pairs, n=3) if voice_pairs else "(нет примеров)"
        system = "You generate a single training pair for a fine-tune dataset. Return strict JSON only."
        user = (
            f"KIND={kind}\n"
            "Use the selected facts as grounding. Keep the answer in the voice of Sir Max.\n"
            "Return JSON: {\"user\": \"...\", \"assistant\": \"...\"}\n\n"
            f"Instruction:\n{safe_text(instruction) or 'Create a natural question and answer grounded in the facts.'}\n\n"
            "Facts:\n"
            + "\n".join(fact_lines)
            + "\n\nVoice examples:\n"
            + style_examples
        )

        config = self._editor_config()
        response = self._call_llm_auto(
            config, system, user,
            max_tokens=900,
            temperature=0.7,
            log_prefix="[studio][generate]",
            model_override=model_override or config.model,
        )
        parsed, _ = ed.parse_json_response(response or "", expect="object", log_prefix="[studio][generate]")
        if not isinstance(parsed, dict):
            raise ValueError("LLM did not return valid JSON")
        user_text = safe_text(parsed.get("user", ""))
        assistant_text = safe_text(parsed.get("assistant", ""))
        if not user_text or not assistant_text:
            raise ValueError("Generated sample is empty")

        created = self.create_sample(
            {
                "kind": kind,
                "system": ed.Config().character_system_prompt,
                "user": user_text,
                "assistant": assistant_text,
                "linked_fact_ids": fact_ids,
                "review_status": "pending",
                "review_note": "generated_from_facts",
            }
        )
        self._record_llm_job(
            "generate_sample",
            {
                "fact_ids": fact_ids,
                "kind": kind,
                "instruction": instruction,
                "model_override": model_override or config.model,
                "system": system,
                "prompt": user,
            },
            {"raw_response": response, "created_sample_id": created["id"]},
        )
        return created

    def export_final(self) -> dict:
        with self._lock:
            self.refresh()
            facts = [strip_editor_fields(item) for item in self._state["facts"].values()]
            facts = [item for item in facts if safe_text(item.get("subject", "")) and safe_text(item.get("fact", ""))]
            canonical = ed.canonicalize_global_knowledge(facts, narrator="Макс")
            unique_facts = ed.deduplicate_knowledge(canonical)

            samples = list(self._state["samples"].values())
            voice_export = [{"messages": item.get("messages", [])} for item in samples if item.get("kind") == "voice" and isinstance(item.get("messages"), list)]
            synth_export = [{"messages": item.get("messages", [])} for item in samples if item.get("kind") == "synth" and isinstance(item.get("messages"), list)]
            dataset_export = ed.deduplicate(voice_export + synth_export)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = self.paths.exports_dir / ts
            export_dir.mkdir(parents=True, exist_ok=True)

            knowledge_path = export_dir / "knowledge_base_final.json"
            knowledge_txt_path = export_dir / "knowledge_base_final.txt"
            dataset_path = export_dir / "dataset_final.jsonl"
            dataset_txt_path = export_dir / "dataset_final.txt"
            voice_path = export_dir / "voice_final.jsonl"
            synth_path = export_dir / "synth_final.jsonl"
            themes_path = export_dir / "themes_final.json"
            relations_path = export_dir / "relations_final.json"

            with open(knowledge_path, "w", encoding="utf-8") as f:
                json.dump(unique_facts, f, ensure_ascii=False, indent=2)
            ed.save_readable_knowledge(unique_facts, str(knowledge_txt_path))

            with open(dataset_path, "w", encoding="utf-8") as f:
                for pair in dataset_export:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            ed.save_readable_voice(dataset_export, str(dataset_txt_path))

            with open(voice_path, "w", encoding="utf-8") as f:
                for pair in voice_export:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            with open(synth_path, "w", encoding="utf-8") as f:
                for pair in synth_export:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            write_json_atomic(themes_path, list(self._state["themes"].values()))
            write_json_atomic(relations_path, list(self._state["relations"].values()))

            manifest = {
                "exported_at": now_iso(),
                "facts": len(unique_facts),
                "dataset_pairs": len(dataset_export),
                "voice_pairs": len(voice_export),
                "synth_pairs": len(synth_export),
                "export_dir": str(export_dir),
            }
            write_json_atomic(export_dir / "manifest.json", manifest)
            return manifest


class DatasetStudioHandler(BaseHTTPRequestHandler):
    server_version = "DatasetStudio/1.0"

    @property
    def store(self) -> DatasetStudioStore:
        return self.server.store  # type: ignore[attr-defined]

    def _send_json(self, payload: Any, status: int = 200):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_text(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = 200):
        encoded = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _serve_asset(self, name: str):
        path = ASSETS_DIR / name
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "text/plain; charset=utf-8"
        if name.endswith(".html"):
            content_type = "text/html; charset=utf-8"
        elif name.endswith(".js"):
            content_type = "application/javascript; charset=utf-8"
        elif name.endswith(".css"):
            content_type = "text/css; charset=utf-8"
        self._send_text(path.read_text(encoding="utf-8"), content_type=content_type)

    def _route_error(self, exc: Exception, status: int = 400):
        self._send_json({"error": str(exc)}, status=status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                self._serve_asset("index.html")
                return
            if path.startswith("/assets/"):
                self._serve_asset(path.split("/assets/", 1)[1])
                return
            if path == "/api/summary":
                self._send_json(self.store.summary())
                return
            if path == "/api/chunks":
                limit = int(query.get("limit", ["200"])[0] or "200")
                offset = int(query.get("offset", ["0"])[0] or "0")
                self._send_json(
                    self.store.list_chunks(
                        search=safe_text(query.get("search", [""])[0]),
                        book=safe_text(query.get("book", [""])[0]),
                        chapter=safe_text(query.get("chapter", [""])[0]),
                        has_dialogues=safe_text(query.get("has_dialogues", [""])[0]),
                        has_knowledge=safe_text(query.get("has_knowledge", [""])[0]),
                        limit=max(1, min(limit, 2000)),
                        offset=max(0, offset),
                    )
                )
                return
            if path.startswith("/api/chunks/"):
                chunk_id = unquote(path.split("/api/chunks/", 1)[1])
                self._send_json(self.store.get_chunk(chunk_id))
                return
            if path == "/api/facts":
                limit = int(query.get("limit", ["200"])[0] or "200")
                offset = int(query.get("offset", ["0"])[0] or "0")
                self._send_json(
                    self.store.list_facts(
                        search=safe_text(query.get("search", [""])[0]),
                        category=safe_text(query.get("category", [""])[0]),
                        book=safe_text(query.get("book", [""])[0]),
                        review_status=safe_text(query.get("review_status", [""])[0]),
                        theme_id=safe_text(query.get("theme_id", [""])[0]),
                        limit=max(1, min(limit, 2000)),
                        offset=max(0, offset),
                    )
                )
                return
            if path.startswith("/api/facts/"):
                fact_id = unquote(path.split("/api/facts/", 1)[1])
                self._send_json(self.store.get_fact(fact_id))
                return
            if path == "/api/samples":
                limit = int(query.get("limit", ["200"])[0] or "200")
                offset = int(query.get("offset", ["0"])[0] or "0")
                kind = safe_text(query.get("kind", ["voice"])[0]) or "voice"
                self._send_json(
                    self.store.list_samples(
                        kind=kind,
                        search=safe_text(query.get("search", [""])[0]),
                        review_status=safe_text(query.get("review_status", [""])[0]),
                        linked_fact_id=safe_text(query.get("linked_fact_id", [""])[0]),
                        limit=max(1, min(limit, 2000)),
                        offset=max(0, offset),
                    )
                )
                return
            if path.startswith("/api/samples/"):
                sample_id = unquote(path.split("/api/samples/", 1)[1])
                self._send_json(self.store.get_sample(sample_id))
                return
            if path == "/api/themes":
                self._send_json({"items": self.store.list_themes()})
                return
            if path == "/api/relations":
                self._send_json({"items": self.store.list_relations(fact_id=safe_text(query.get("fact_id", [""])[0]))})
                return
            if path == "/api/llm/runs":
                self._send_json({"items": self.store.list_llm_runs()})
                return
            if path == "/api/llm/jobs":
                limit = int(query.get("limit", ["200"])[0] or "200")
                offset = int(query.get("offset", ["0"])[0] or "0")
                self._send_json(
                    self.store.list_llm_jobs(
                        search=safe_text(query.get("search", [""])[0]),
                        job_type=safe_text(query.get("job_type", [""])[0]),
                        limit=max(1, min(limit, 2000)),
                        offset=max(0, offset),
                    )
                )
                return
            if path.startswith("/api/llm/jobs/"):
                job_id = unquote(path.split("/api/llm/jobs/", 1)[1])
                self._send_json(self.store.get_llm_job(job_id))
                return
            if path == "/api/llm/traces":
                limit = int(query.get("limit", ["200"])[0] or "200")
                offset = int(query.get("offset", ["0"])[0] or "0")
                self._send_json(
                    self.store.list_llm_traces(
                        run_id=safe_text(query.get("run_id", [""])[0]),
                        search=safe_text(query.get("search", [""])[0]),
                        limit=max(1, min(limit, 2000)),
                        offset=max(0, offset),
                    )
                )
                return
            if path.startswith("/api/llm/traces/"):
                trace_ref = unquote(path.split("/api/llm/traces/", 1)[1])
                self._send_json(self.store.get_llm_trace(trace_ref))
                return
            if path == "/api/pipeline/summary":
                self._send_json(self.store.get_pipeline_snapshot())
                return
            if path == "/api/pipeline/events":
                limit = int(query.get("limit", ["200"])[0] or "200")
                offset = int(query.get("offset", ["0"])[0] or "0")
                self._send_json(
                    self.store.list_metadata_events(
                        search=safe_text(query.get("search", [""])[0]),
                        event_type=safe_text(query.get("event_type", [""])[0]),
                        status=safe_text(query.get("status", [""])[0]),
                        limit=max(1, min(limit, 2000)),
                        offset=max(0, offset),
                    )
                )
                return
            if path == "/api/timeline/overview":
                self._send_json(self.store.get_timeline_overview())
                return
            if path == "/api/timeline/nodes":
                limit = int(query.get("limit", ["300"])[0] or "300")
                offset = int(query.get("offset", ["0"])[0] or "0")
                self._send_json(
                    self.store.list_timeline_nodes(
                        search=safe_text(query.get("search", [""])[0]),
                        node_type=safe_text(query.get("node_type", [""])[0]),
                        limit=max(1, min(limit, 3000)),
                        offset=max(0, offset),
                    )
                )
                return
            if path == "/api/timeline/edges":
                limit = int(query.get("limit", ["300"])[0] or "300")
                offset = int(query.get("offset", ["0"])[0] or "0")
                self._send_json(
                    self.store.list_timeline_edges(
                        search=safe_text(query.get("search", [""])[0]),
                        relation_type=safe_text(query.get("relation_type", [""])[0]),
                        limit=max(1, min(limit, 3000)),
                        offset=max(0, offset),
                    )
                )
                return
            if path == "/api/timeline/groups":
                limit = int(query.get("limit", ["200"])[0] or "200")
                offset = int(query.get("offset", ["0"])[0] or "0")
                self._send_json(
                    self.store.list_timeline_groups(
                        search=safe_text(query.get("search", [""])[0]),
                        book=safe_text(query.get("book", [""])[0]),
                        limit=max(1, min(limit, 2000)),
                        offset=max(0, offset),
                    )
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._route_error(exc, status=404)
        except Exception as exc:
            self._route_error(exc, status=400)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_json_body()
        try:
            if path == "/api/facts":
                self._send_json(self.store.create_fact(body), status=201)
                return
            if path == "/api/facts/batch":
                fact_ids = [safe_text(item) for item in body.get("fact_ids", []) if safe_text(item)]
                patch = body.get("patch", {})
                if not isinstance(patch, dict) or not fact_ids:
                    raise ValueError("fact_ids and patch are required")
                results = []
                for fid in fact_ids:
                    try:
                        results.append(self.store.update_fact(fid, patch))
                    except KeyError:
                        pass
                self._send_json({"updated": len(results)})
                return
            if path == "/api/facts/reanalyze":
                fact_ids = [safe_text(item) for item in body.get("fact_ids", []) if safe_text(item)]
                self._send_json(
                    self.store.reanalyze_facts(
                        fact_ids,
                        bundle=bool(body.get("bundle")),
                        model_override=safe_text(body.get("model_override", "")),
                    )
                )
                return
            if path == "/api/samples":
                self._send_json(self.store.create_sample(body), status=201)
                return
            if path == "/api/samples/generate":
                fact_ids = [safe_text(item) for item in body.get("fact_ids", []) if safe_text(item)]
                self._send_json(
                    self.store.generate_sample_from_facts(
                        fact_ids,
                        kind=safe_text(body.get("kind", "")) or "synth",
                        instruction=safe_text(body.get("instruction", "")),
                        model_override=safe_text(body.get("model_override", "")),
                    ),
                    status=201,
                )
                return
            if path == "/api/themes":
                self._send_json(self.store.create_theme(body), status=201)
                return
            if path == "/api/themes/merge":
                self._send_json(
                    self.store.merge_themes(
                        safe_text(body.get("source_theme_id", "")),
                        safe_text(body.get("target_theme_id", "")),
                    )
                )
                return
            if path == "/api/relations":
                self._send_json(self.store.create_relation(body), status=201)
                return
            if path == "/api/undo":
                self._send_json(self.store.undo_last(domain=safe_text(body.get("domain", "")) or None))
                return
            if path == "/api/export/final":
                self._send_json(self.store.export_final())
                return
            if path == "/api/llm/run":
                self._send_json(self.store.run_llm_prompt(body), status=201)
                return
            if path.startswith("/api/llm/traces/") and path.endswith("/rerun"):
                trace_ref = unquote(path[len("/api/llm/traces/") : -len("/rerun")])
                self._send_json(self.store.rerun_llm_trace(trace_ref, body), status=201)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._route_error(exc, status=404)
        except Exception as exc:
            self._route_error(exc, status=400)

    def do_PATCH(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_json_body()
        try:
            if path.startswith("/api/facts/"):
                fact_id = unquote(path.split("/api/facts/", 1)[1])
                self._send_json(self.store.update_fact(fact_id, body))
                return
            if path.startswith("/api/samples/"):
                sample_id = unquote(path.split("/api/samples/", 1)[1])
                self._send_json(self.store.update_sample(sample_id, body))
                return
            if path.startswith("/api/themes/"):
                theme_id = unquote(path.split("/api/themes/", 1)[1])
                self._send_json(self.store.update_theme(theme_id, body))
                return
            if path.startswith("/api/relations/"):
                relation_id = unquote(path.split("/api/relations/", 1)[1])
                self._send_json(self.store.update_relation(relation_id, body))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._route_error(exc, status=404)
        except Exception as exc:
            self._route_error(exc, status=400)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/facts/"):
                fact_id = unquote(path.split("/api/facts/", 1)[1])
                self._send_json(self.store.delete_fact(fact_id))
                return
            if path.startswith("/api/samples/"):
                sample_id = unquote(path.split("/api/samples/", 1)[1])
                self._send_json(self.store.delete_sample(sample_id))
                return
            if path.startswith("/api/themes/"):
                theme_id = unquote(path.split("/api/themes/", 1)[1])
                self._send_json(self.store.delete_theme(theme_id))
                return
            if path.startswith("/api/relations/"):
                relation_id = unquote(path.split("/api/relations/", 1)[1])
                self._send_json(self.store.delete_relation(relation_id))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._route_error(exc, status=404)
        except Exception as exc:
            self._route_error(exc, status=400)

    def log_message(self, format: str, *args):
        return


def serve(output_dir: Path, host: str = "127.0.0.1", port: int = 8766, workspace_dir: Optional[Path] = None):
    store = DatasetStudioStore(output_dir=output_dir, workspace_dir=workspace_dir)
    store.refresh()
    server = ThreadingHTTPServer((host, port), DatasetStudioHandler)
    server.daemon_threads = True
    server.store = store  # type: ignore[attr-defined]
    print(f"Dataset Studio: http://{host}:{port}")
    print(f"Output dir: {output_dir}")
    print(f"Workspace dir: {store.paths.workspace_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Dataset Studio...")
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset Studio for Eho fine-tuning datasets")
    parser.add_argument("--output-dir", default="./output", help="Directory with pipeline output artifacts")
    parser.add_argument("--workspace-dir", default="", help="Separate editor workspace directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8766, help="Bind port")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    workspace_dir = Path(args.workspace_dir).resolve() if safe_text(args.workspace_dir) else None
    serve(output_dir=output_dir, host=args.host, port=args.port, workspace_dir=workspace_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
