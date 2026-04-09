import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from unittest import mock

import dataset_studio as ds
import extract_dialogues as ed


def write_jsonl(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class DatasetStudioFixture:
    def __init__(self, root: Path):
        self.root = root
        self.output = root / "output"
        self.output.mkdir(parents=True, exist_ok=True)
        self.book_name = "1. Лабиринты Ехо/1. Чужак.fb2"
        self.book_stem = ed.get_book_stem(self.book_name)
        self.config = ed.Config()

    def build(self):
        dialogue = {
            "type": "dialogue",
            "context": "Макс и Джуффин сидят в Доме у Моста.",
            "interlocutor": "Джуффин",
            "interlocutor_says": "Макс, выпей камры.",
            "max_says": "Я бы и без приглашения выпил камры, если уж на то пошло.",
        }
        chunk_text = (
            "Дом у Моста служит штабом Тайного Сыска. "
            "Кимпа — старый дворецкий Джуффина. "
            "Макс и Джуффин обсуждают камру."
        )
        fact = {
            "category": "place",
            "subject": "Дом у Моста",
            "fact": "Дом у Моста служит штабом Тайного Сыска.",
            "time_scope": "timeless",
            "source_book": self.book_name,
            "chapter": "Глава 1",
            "chunk_idx": 0,
        }
        voice_pair = ed.make_training_pairs([dialogue], self.config)[0]
        synth_pair = {
            "messages": [
                {"role": "system", "content": self.config.character_system_prompt},
                {"role": "user", "content": "Что такое Дом у Моста?"},
                {"role": "assistant", "content": "Дом у Моста — это место, где мы работаем и пьём камру."},
            ]
        }
        synth_progress = {
            "fact_hash": ed.stable_fact_hash(fact),
            "pair": synth_pair,
        }

        write_jsonl(
            self.output / f"chunks_{self.book_stem}.jsonl",
            [
                {
                    "idx": 0,
                    "chapter": "Глава 1",
                    "chunk_text": chunk_text,
                    "dialogues": [dialogue],
                    "knowledge": [fact],
                }
            ],
        )
        write_jsonl(self.output / f"knowledge_{self.book_stem}.jsonl", [fact])
        write_jsonl(self.output / f"voice_{self.book_stem}.jsonl", [voice_pair])
        write_jsonl(self.output / f"synth_{self.book_stem}.jsonl", [synth_pair])
        write_jsonl(self.output / f"synth_progress_{self.book_stem}.jsonl", [synth_progress])
        with open(self.output / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "gemma4:latest",
                    "api_base": "http://localhost:11434/v1",
                    "knowledge_extract_model": "gemma4:latest",
                    "knowledge_validate_model": "gemma4:latest",
                    "knowledge_link_model": "gemma4:latest",
                    "knowledge_extract_model_secondary": "qwen3:8b",
                    "knowledge_arbiter_model": "gemma4:latest",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        write_jsonl(
            self.output / "metadata_history.jsonl",
            [
                {
                    "idx": 1,
                    "ts": "2026-04-08T10:00:00+04:00",
                    "type": "run_started",
                    "status": "running",
                    "current_stage": "startup",
                    "current_book": "",
                    "message": "pipeline started",
                    "details": {"current_stage": "startup"},
                },
                {
                    "idx": 2,
                    "ts": "2026-04-08T10:01:00+04:00",
                    "type": "chunk_complete",
                    "status": "running",
                    "current_stage": "book_extraction",
                    "current_book": self.book_name,
                    "message": "chunk complete",
                    "details": {"current_book_progress": {"chunk_idx": 0, "completed_chunks": 1}},
                },
            ],
        )

        llm_root = self.output / "llm_traces"
        llm_root.mkdir(parents=True, exist_ok=True)
        ds.write_json_atomic(
            llm_root / "20260407T182406_000001_smoke.json",
            {
                "trace_id": "20260407T182406_000001_smoke",
                "created_at": "2026-04-07T18:24:06+04:00",
                "updated_at": "2026-04-07T18:24:19+04:00",
                "provider": "ollama_native",
                "api_base": "http://localhost:11434/v1",
                "model": "gemma4:latest",
                "log_prefix": "[smoke]",
                "max_tokens": 10,
                "temperature": 0.1,
                "response_format": None,
                "request_payload": {
                    "model": "gemma4:latest",
                    "messages": [
                        {"role": "system", "content": "Ответь одним словом."},
                        {"role": "user", "content": "Скажи: работает"},
                    ],
                },
                "attempts": [
                    {
                        "status": "ok",
                        "attempt": 1,
                        "ts": "2026-04-07T18:24:19+04:00",
                        "content": "Работает",
                        "elapsed_seconds": 12.6,
                    }
                ],
            },
        )
        run_dir = llm_root / "run_20260408_101010"
        run_dir.mkdir(parents=True, exist_ok=True)
        ds.write_json_atomic(
            run_dir / "20260408T101011_000001_manual.json",
            {
                "trace_id": "20260408T101011_000001_manual",
                "created_at": "2026-04-08T10:10:11+04:00",
                "updated_at": "2026-04-08T10:10:20+04:00",
                "provider": "ollama_native",
                "api_base": "http://localhost:11434/v1",
                "model": "gemma4:latest",
                "log_prefix": "[manual]",
                "max_tokens": 120,
                "temperature": 0.2,
                "response_format": None,
                "request_payload": {
                    "model": "gemma4:latest",
                    "messages": [
                        {"role": "system", "content": "Ты полезный помощник."},
                        {"role": "user", "content": "Коротко опиши Дом у Моста."},
                    ],
                },
                "attempts": [
                    {
                        "status": "ok",
                        "attempt": 1,
                        "ts": "2026-04-08T10:10:20+04:00",
                        "content": "Дом у Моста — штаб Тайного Сыска.",
                        "elapsed_seconds": 9.1,
                    }
                ],
            },
        )
        workspace = self.output / "editor_workspace"
        llm_jobs = workspace / "llm_jobs"
        llm_jobs.mkdir(parents=True, exist_ok=True)
        ds.write_json_atomic(
            llm_jobs / "20260408T101100_generate_sample_deadbeef.json",
            {
                "job_id": "generate_sample:deadbeef",
                "ts": "2026-04-08T10:11:00+04:00",
                "job_type": "generate_sample",
                "request": {
                    "fact_ids": ["fact:test"],
                    "kind": "synth",
                    "instruction": "Make a short answer",
                    "prompt": "Facts: Дом у Моста ...",
                },
                "response": {
                    "raw_response": "{\"user\":\"?\",\"assistant\":\"!\"}",
                    "created_sample_id": "sample:synth:test",
                },
            },
        )
        ds.write_json_atomic(
            self.output / "timeline_resolution_raw.json",
            [
                {
                    "book_name": self.book_name,
                    "chapter": "Глава 1",
                    "chunk_indices": [0],
                    "facts": [fact],
                    "entities": [{"label": "Дом у Моста", "type": "place"}],
                    "events": [
                        {
                            "local_id": "E1",
                            "label": "Прибытие Макса",
                            "summary": "Макс прибыл в Дом у Моста.",
                            "time_scope": "past",
                            "chunk_indices": [0],
                            "participants": ["Макс"],
                            "places": ["Дом у Моста"],
                            "objects": [],
                        }
                    ],
                    "relations": [
                        {
                            "source": "E1",
                            "target": "Дом у Моста",
                            "type": "occurs_in",
                            "evidence": "Дом у Моста служит штабом Тайного Сыска.",
                            "confidence": "explicit",
                        }
                    ],
                }
            ],
        )
        ds.write_json_atomic(
            self.output / "timeline_graph.json",
            {
                "nodes": [
                    {
                        "id": "place:house",
                        "label": "Дом у Моста",
                        "type": "place",
                        "source_books": [self.book_name],
                    },
                    {
                        "id": "event:arrival",
                        "label": "Прибытие Макса",
                        "type": "event",
                        "book": self.book_name,
                        "chapter": "Глава 1",
                        "summary": "Макс прибыл в Дом у Моста.",
                        "time_scope": "past",
                        "chunk_indices": [0],
                    },
                ],
                "edges": [
                    {
                        "source": "event:arrival",
                        "target": "place:house",
                        "type": "occurs_in",
                        "book": self.book_name,
                        "chapter": "Глава 1",
                        "chunk_indices": [0],
                        "evidence": "Дом у Моста служит штабом Тайного Сыска.",
                        "confidence": "explicit",
                    }
                ],
                "groups": [
                    {
                        "book_name": self.book_name,
                        "chapter": "Глава 1",
                        "facts": 1,
                        "events": 1,
                        "relations": 1,
                    }
                ],
            },
        )


class DatasetStudioStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.fixture = DatasetStudioFixture(self.root)
        self.fixture.build()
        self.store = ds.DatasetStudioStore(self.fixture.output)
        self.store.refresh()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_loads_base_data_with_source_excerpt_and_links(self):
        summary = self.store.summary()
        self.assertEqual(summary["counts"]["facts"], 1)
        self.assertEqual(summary["counts"]["voice"], 1)
        self.assertEqual(summary["counts"]["synth"], 1)

        fact_id = next(iter(self.store._state["facts"]))  # noqa: SLF001
        fact_payload = self.store.get_fact(fact_id)
        self.assertIn("Дом у Моста", fact_payload["item"]["source_excerpt"])
        self.assertEqual(fact_payload["item"]["chapter"], "Глава 1")

        synth_id = next(sample_id for sample_id, sample in self.store._state["samples"].items() if sample["kind"] == "synth")  # noqa: SLF001
        synth_payload = self.store.get_sample(synth_id)
        self.assertEqual(len(synth_payload["linked_facts"]), 1)
        self.assertEqual(synth_payload["linked_facts"][0]["subject"], "Дом у Моста")

    def test_fact_operations_are_reversible(self):
        created = self.store.create_fact(
            {
                "category": "character",
                "subject": "Кимпа",
                "fact": "Кимпа — старый дворецкий Джуффина.",
                "time_scope": "timeless",
                "source_book": self.fixture.book_name,
            }
        )
        self.assertEqual(created["subject"], "Кимпа")

        updated = self.store.update_fact(created["id"], {"review_status": "approved", "review_score": 4})
        self.assertEqual(updated["review_status"], "approved")
        self.assertEqual(updated["review_score"], 4)

        inverse = self.store.undo_last(domain="facts")
        self.assertEqual(inverse["domain"], "facts")
        reverted = self.store.get_fact(created["id"])["item"]
        self.assertEqual(reverted["review_status"], "pending")

        deleted = self.store.delete_fact(created["id"])
        self.assertEqual(deleted["deleted"], created["id"])
        with self.assertRaises(KeyError):
            self.store.get_fact(created["id"])

    def test_theme_merge_relation_and_export(self):
        fact_ids = list(self.store._state["facts"])  # noqa: SLF001
        extra_fact = self.store.create_fact(
            {
                "category": "custom",
                "subject": "камра",
                "fact": "Камра — горький горячий напиток Ехо.",
                "time_scope": "timeless",
                "source_book": self.fixture.book_name,
            }
        )
        theme_a = self.store.create_theme({"name": "Быт", "description": "Повседневность"})
        theme_b = self.store.create_theme({"name": "Напитки", "description": "Еда и напитки"})
        self.store.update_fact(fact_ids[0], {"theme_ids": [theme_a["id"]]})
        self.store.update_fact(extra_fact["id"], {"theme_ids": [theme_a["id"]]})
        relation = self.store.create_relation(
            {
                "source_fact_id": fact_ids[0],
                "target_fact_id": extra_fact["id"],
                "relation_type": "related_to",
                "note": "Место и напиток сцены",
            }
        )
        self.assertEqual(relation["relation_type"], "related_to")

        merge_result = self.store.merge_themes(theme_a["id"], theme_b["id"])
        self.assertEqual(merge_result["into"], theme_b["id"])
        extra_after = self.store.get_fact(extra_fact["id"])["item"]
        self.assertEqual(extra_after["theme_ids"], [theme_b["id"]])

        manifest = self.store.export_final()
        export_dir = Path(manifest["export_dir"])
        self.assertTrue((export_dir / "knowledge_base_final.json").exists())
        self.assertTrue((export_dir / "dataset_final.jsonl").exists())
        self.assertTrue((export_dir / "themes_final.json").exists())
        self.assertTrue((export_dir / "relations_final.json").exists())

    def test_llm_trace_listing_populates_model_and_preview(self):
        traces = self.store.list_llm_traces(run_id="run_20260408_101010")
        self.assertEqual(traces["total"], 1)
        item = traces["items"][0]
        self.assertEqual(item["model"], "gemma4:latest")
        self.assertEqual(item["provider"], "ollama_native")
        self.assertIn("Дом у Моста", item["user_preview"])
        self.assertEqual(item["last_status"], "ok")

    def test_llm_trace_listing_and_detail(self):
        runs = self.store.list_llm_runs()
        self.assertGreaterEqual(len(runs), 2)
        self.assertTrue(any(item["id"] == "legacy_flat" for item in runs))
        self.assertTrue(any(item["id"] == "run_20260408_101010" for item in runs))

        traces = self.store.list_llm_traces(run_id="run_20260408_101010")
        self.assertEqual(traces["total"], 1)
        detail = self.store.get_llm_trace(traces["items"][0]["id"])
        self.assertEqual(detail["summary"]["model"], "gemma4:latest")
        self.assertIn("Дом у Моста", detail["editable_request"]["user"])

    def test_llm_manual_run_and_rerun_create_new_trace(self):
        def fake_call(config, system, user, max_tokens=0, response_format=None, log_prefix="", temperature=None, trace_id="", model_override=None):
            trace_dir = ed.get_llm_trace_dir(config)
            trace_dir.mkdir(parents=True, exist_ok=True)
            ds.write_json_atomic(
                trace_dir / f"{trace_id}.json",
                {
                    "trace_id": trace_id,
                    "created_at": ds.now_iso(),
                    "updated_at": ds.now_iso(),
                    "provider": "ollama_native",
                    "api_base": config.api_base,
                    "model": model_override or config.model,
                    "log_prefix": log_prefix,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "response_format": response_format,
                    "request_payload": {
                        "model": model_override or config.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                    "attempts": [
                        {
                            "status": "ok",
                            "attempt": 1,
                            "ts": ds.now_iso(),
                            "content": "mocked response",
                            "elapsed_seconds": 0.01,
                        }
                    ],
                },
            )
            return "mocked response"

        with mock.patch.object(ed, "call_llm_ollama_native", side_effect=fake_call):
            created = self.store.run_llm_prompt(
                {
                    "system": "Ты редактор.",
                    "user": "Проверь этот факт.",
                    "model_override": "gemma4:latest",
                    "max_tokens": 150,
                    "temperature": 0.0,
                    "log_prefix": "[studio][manual]",
                }
            )
            self.assertTrue(created["trace_ref"].startswith("studio_"))
            self.assertEqual(created["content"], "mocked response")
            self.assertTrue(Path(created["trace_path"]).exists())

            rerun = self.store.rerun_llm_trace(
                "run_20260408_101010/20260408T101011_000001_manual",
                {"user": "Сделай новую версию ответа."},
            )
            self.assertTrue(Path(rerun["trace_path"]).exists())
            detail = self.store.get_llm_trace(rerun["trace_ref"])
            self.assertIn("Сделай новую версию ответа.", detail["editable_request"]["user"])


    def test_chunks_pipeline_and_timeline_are_exposed(self):
        summary = self.store.summary()
        self.assertEqual(summary["counts"]["chunks"], 1)
        self.assertEqual(summary["counts"]["metadata_events"], 2)
        self.assertEqual(summary["counts"]["llm_jobs"], 1)
        self.assertEqual(summary["counts"]["timeline_nodes"], 2)
        self.assertEqual(summary["counts"]["timeline_edges"], 1)

        chunks = self.store.list_chunks()
        self.assertEqual(chunks["total"], 1)
        chunk = self.store.get_chunk(chunks["items"][0]["id"])
        self.assertEqual(len(chunk["linked_facts"]), 1)
        self.assertGreaterEqual(len(chunk["linked_samples"]), 1)

        pipeline = self.store.get_pipeline_snapshot()
        self.assertEqual(pipeline["history_count"], 2)
        events = self.store.list_metadata_events(search="chunk")
        self.assertEqual(events["total"], 1)

        jobs = self.store.list_llm_jobs()
        self.assertEqual(jobs["total"], 1)
        job = self.store.get_llm_job(jobs["items"][0]["id"])
        self.assertEqual(job["job_type"], "generate_sample")

        overview = self.store.get_timeline_overview()
        self.assertTrue(overview["available"])
        self.assertEqual(overview["counts"]["groups"], 1)
        nodes = self.store.list_timeline_nodes(search="Дом", node_type="place")
        self.assertEqual(nodes["total"], 1)
        edges = self.store.list_timeline_edges(relation_type="occurs_in")
        self.assertEqual(edges["total"], 1)
        groups = self.store.list_timeline_groups(search="Глава 1")
        self.assertEqual(groups["total"], 1)


    def test_smart_refresh_skips_reload_when_files_unchanged(self):
        self.store.refresh()
        state_before = self.store._state  # noqa: SLF001
        self.store.refresh()
        state_after = self.store._state  # noqa: SLF001
        self.assertIs(state_before, state_after)

    def test_batch_update_changes_review_status_for_multiple_facts(self):
        extra = self.store.create_fact(
            {
                "category": "character",
                "subject": "Макс",
                "fact": "Макс пьёт камру.",
                "time_scope": "timeless",
            }
        )
        fact_ids = list(self.store._state["facts"])  # noqa: SLF001
        self.assertGreaterEqual(len(fact_ids), 2)
        for fid in fact_ids:
            self.store.update_fact(fid, {"review_status": "approved"})
        for fid in fact_ids:
            self.assertEqual(self.store.get_fact(fid)["item"]["review_status"], "approved")


class DatasetStudioApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.fixture = DatasetStudioFixture(self.root)
        self.fixture.build()
        self.store = ds.DatasetStudioStore(self.fixture.output)
        self.store.refresh()
        self.server = ds.ThreadingHTTPServer(("127.0.0.1", 0), ds.DatasetStudioHandler)
        self.server.store = self.store  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.tempdir.cleanup()

    def request(self, method: str, path: str, payload: dict | None = None) -> dict | str:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(f"{self.base}{path}", data=data, headers=headers, method=method)
        with urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            if resp.headers.get_content_type() == "application/json":
                return json.loads(body)
            return body

    def test_http_summary_and_fact_update(self):
        summary = self.request("GET", "/api/summary")
        self.assertEqual(summary["counts"]["facts"], 1)
        self.assertEqual(summary["counts"]["chunks"], 1)

        facts = self.request("GET", "/api/facts?limit=10")
        fact_id = facts["items"][0]["id"]
        updated = self.request("PATCH", f"/api/facts/{fact_id}", {"review_status": "approved"})
        self.assertEqual(updated["review_status"], "approved")

        html = self.request("GET", "/")
        self.assertIn("Dataset Studio", html)
        self.assertIn('data-tab="chunks"', html)
        self.assertIn('data-tab="pipeline"', html)
        self.assertIn('data-tab="timeline"', html)

    def test_http_reanalyze_route_can_be_mocked(self):
        with mock.patch.object(self.store, "reanalyze_facts", return_value={"suggestions": [{"action": "keep"}]}):
            payload = self.request("POST", "/api/facts/reanalyze", {"fact_ids": ["fact:1"], "bundle": True})
        self.assertEqual(payload["suggestions"][0]["action"], "keep")

    def test_http_llm_trace_routes_can_be_mocked(self):
        with mock.patch.object(self.store, "list_llm_runs", return_value=[{"id": "legacy_flat", "trace_count": 1, "latest_at": "", "providers": [], "models": []}]):
            runs = self.request("GET", "/api/llm/runs")
        self.assertEqual(runs["items"][0]["id"], "legacy_flat")

        with mock.patch.object(self.store, "run_llm_prompt", return_value={"trace_ref": "studio_x/test", "content": "ok", "trace_path": "/tmp/x.json", "trace": {}}):
            result = self.request("POST", "/api/llm/run", {"system": "s", "user": "u"})
        self.assertEqual(result["trace_ref"], "studio_x/test")

    def test_http_batch_update_facts(self):
        facts = self.request("GET", "/api/facts?limit=10")
        fact_id = facts["items"][0]["id"]
        result = self.request("POST", "/api/facts/batch", {
            "fact_ids": [fact_id],
            "patch": {"review_status": "rejected"},
        })
        self.assertEqual(result["updated"], 1)
        updated = self.request("GET", f"/api/facts/{fact_id}")
        self.assertEqual(updated["item"]["review_status"], "rejected")

    def test_http_chunk_pipeline_and_timeline_routes(self):
        chunks = self.request("GET", "/api/chunks?limit=10")
        self.assertEqual(chunks["total"], 1)
        chunk_id = chunks["items"][0]["id"]
        chunk = self.request("GET", f"/api/chunks/{chunk_id}")
        self.assertEqual(chunk["item"]["chunk_idx"], 0)

        pipeline = self.request("GET", "/api/pipeline/summary")
        self.assertEqual(pipeline["history_count"], 2)
        events = self.request("GET", "/api/pipeline/events?limit=10")
        self.assertEqual(events["total"], 2)

        jobs = self.request("GET", "/api/llm/jobs?limit=10")
        self.assertEqual(jobs["total"], 1)
        job = self.request("GET", f"/api/llm/jobs/{jobs['items'][0]['id']}")
        self.assertEqual(job["job_type"], "generate_sample")

        overview = self.request("GET", "/api/timeline/overview")
        self.assertEqual(overview["counts"]["nodes"], 2)
        nodes = self.request("GET", "/api/timeline/nodes?limit=10")
        self.assertEqual(nodes["total"], 2)
        edges = self.request("GET", "/api/timeline/edges?limit=10")
        self.assertEqual(edges["total"], 1)
        groups = self.request("GET", "/api/timeline/groups?limit=10")
        self.assertEqual(groups["total"], 1)


if __name__ == "__main__":
    unittest.main()
