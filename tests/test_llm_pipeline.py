import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock


try:
    import openai  # noqa: F401
except Exception:
    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        pass

    fake_openai.OpenAI = FakeOpenAI
    sys.modules["openai"] = fake_openai


import extract_dialogues as ed


class LLMExtractionPipelineSmokeTest(unittest.TestCase):
    def setUp(self):
        ed._STOP_REQUESTED.clear()

    def test_update_metadata_snapshot_writes_incrementally(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"
            metadata_history_path = Path(tmpdir) / "metadata_history.jsonl"
            state = {
                "status": "starting",
                "current_stage": "startup",
                "recent_events": [],
                "event_count": 0,
            }

            ed.update_metadata_snapshot(
                state,
                metadata_path,
                history_path=metadata_history_path,
                event_type="run_started",
                current_stage="startup",
                books_total=33,
            )
            ed.update_metadata_snapshot(
                state,
                metadata_path,
                history_path=metadata_history_path,
                event_type="chunk_complete",
                current_stage="book_extraction",
                current_book="1. Чужак.fb2",
                current_book_progress={
                    "completed_chunks": 3,
                    "total_chunks": 15,
                    "book_voice_pairs": 120,
                },
            )

            written = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(written["books_total"], 33)
            self.assertEqual(written["current_stage"], "book_extraction")
            self.assertEqual(written["current_book"], "1. Чужак.fb2")
            self.assertEqual(written["event_count"], 2)
            self.assertEqual(len(written["recent_events"]), 2)
            self.assertEqual(written["recent_events"][-1]["type"], "chunk_complete")
            self.assertEqual(
                written["recent_events"][-1]["details"]["current_book_progress"]["completed_chunks"],
                3,
            )

            history_lines = [
                json.loads(line)
                for line in metadata_history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(history_lines), 2)
            self.assertEqual(history_lines[0]["type"], "run_started")
            self.assertEqual(history_lines[1]["type"], "chunk_complete")
            self.assertEqual(history_lines[1]["current_book"], "1. Чужак.fb2")

    def test_call_llm_openai_saves_request_and_response_in_one_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ed.Config(output_dir=tmpdir)
            config.llm_trace_run_id = "test_run"

            response = types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="РџСЂРѕРІРµСЂРѕС‡РЅС‹Р№ РѕС‚РІРµС‚"),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=12, completion_tokens=7),
            )
            client = types.SimpleNamespace(
                chat=types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=mock.Mock(return_value=response)
                    )
                )
            )

            content = ed.call_llm_openai(
                client=client,
                config=config,
                system="system prompt",
                user="user prompt",
                max_tokens=123,
                response_format={"type": "json_object"},
                log_prefix="[trace-test]",
                temperature=0.55,
                trace_id="trace_test",
            )

            self.assertEqual(content, "РџСЂРѕРІРµСЂРѕС‡РЅС‹Р№ РѕС‚РІРµС‚")

            trace_path = Path(tmpdir) / "llm_traces" / "test_run" / "trace_test.json"

            self.assertTrue(trace_path.exists())

            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
            response_payload = trace_payload["attempts"][0]

            self.assertEqual(trace_payload["provider"], "openai_compatible")
            self.assertEqual(trace_payload["request_payload"]["messages"][0]["content"], "system prompt")
            self.assertEqual(trace_payload["request_payload"]["messages"][1]["content"], "user prompt")
            self.assertEqual(trace_payload["max_tokens"], 123)
            self.assertEqual(trace_payload["temperature"], 0.55)
            self.assertEqual(len(trace_payload["attempts"]), 1)

            self.assertEqual(response_payload["provider"], "openai_compatible")
            self.assertEqual(response_payload["status"], "ok")
            self.assertEqual(response_payload["attempt"], 1)
            self.assertEqual(response_payload["content"], "РџСЂРѕРІРµСЂРѕС‡РЅС‹Р№ РѕС‚РІРµС‚")
            self.assertEqual(response_payload["finish_reason"], "stop")
            self.assertEqual(response_payload["prompt_tokens"], 12)
            self.assertEqual(response_payload["completion_tokens"], 7)

    def test_write_global_knowledge_snapshot_updates_readable_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            facts = [
                {
                    "category": "character",
                    "subject": "Джуффин Халли",
                    "fact": "Джуффин носит белое лоохи.",
                    "time_scope": "current",
                    "source_book": "book1.fb2",
                    "chapter": "Глава 1",
                    "chunk_idx": 0,
                }
            ]
            ed.write_global_knowledge_snapshot(tmpdir, facts)

            global_txt_path = Path(tmpdir) / "knowledge_base.txt"
            self.assertTrue(global_txt_path.exists())
            first_snapshot = global_txt_path.read_text(encoding="utf-8")
            self.assertIn("Джуффин носит белое лоохи.", first_snapshot)

            facts.append(
                {
                    "category": "custom",
                    "subject": "камра",
                    "fact": "Камру пьют горячей.",
                    "time_scope": "timeless",
                    "source_book": "book2.fb2",
                    "chapter": "Глава 2",
                    "chunk_idx": 4,
                }
            )
            ed.write_global_knowledge_snapshot(tmpdir, facts)

            second_snapshot = global_txt_path.read_text(encoding="utf-8")
            self.assertIn("Джуффин носит белое лоохи.", second_snapshot)
            self.assertIn("Камру пьют горячей.", second_snapshot)
            self.assertNotEqual(first_snapshot, second_snapshot)

    def test_validate_dialogues_handles_null_llm_fields(self):
        items = [
            {
                "type": "dialogue",
                "context": None,
                "interlocutor": "Джуффин",
                "interlocutor_says": None,
                "max_says": "Если ты сейчас скажешь, что это хорошая идея, я начну нервно смеяться",
            }
        ]
        source_chunk = (
            "— Ты готов? — спросил Джуффин.\n\n"
            "— Если ты сейчас скажешь, что это хорошая идея, я начну нервно смеяться, — сказал я."
        )

        validated = ed.validate_dialogues(items, source_chunk=source_chunk)
        pairs = ed.make_training_pairs(validated, ed.Config())

        self.assertEqual(len(validated), 1)
        self.assertEqual(len(pairs), 1)
        self.assertIn("Если ты сейчас скажешь, что это хорошая идея", pairs[0]["messages"][-1]["content"])

    def test_process_book_llm_voice_and_knowledge_are_saved(self):
        book_name = "test_max_book.fb2"
        text = (
            "Предисловие\n\n"
            "— Ты готов? — спросил Джуффин.\n\n"
            "— Ладно, поехали, — сказал я.\n\n"
            "Потом мы поднялись на крышу Дома у Моста.\n\n"
            "В Ехо снова начали колдовать повара."
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ed.Config(
                output_dir=tmpdir,
                chunk_size=5000,
            )
            config.extraction_passes = 2
            config.extraction_neighbor_chunks = 1
            config.extraction_context_budget = 1200

            prompt_counters = {
                "dialogues": 0,
                "knowledge": 0,
            }

            def fake_call_llm(client, config, system, user, **kwargs):
                if "извлеки материал для обучения нейросети говорить как Сэр Макс" in user:
                    prompt_counters["dialogues"] += 1
                    if prompt_counters["dialogues"] == 1:
                        return json.dumps([
                            {
                                "type": "dialogue",
                                "context": "Разговор перед поездкой.",
                                "interlocutor": "Джуффин",
                                "interlocutor_says": "Ты готов?",
                                "max_says": "Ладно, поехали,",
                            }
                        ], ensure_ascii=False)
                    return "[]"

                if "извлеки ФАКТЫ о мире Ехо, персонажах и событиях" in user:
                    prompt_counters["knowledge"] += 1
                    if prompt_counters["knowledge"] == 1:
                        return json.dumps([
                            {
                                "category": "event",
                                "subject": "Макс",
                                "fact": "Макс соглашается ехать дальше вместе с Джуффином.",
                            },
                            {
                                "category": "custom",
                                "subject": "Ехо",
                                "fact": "В Ехо снова начали колдовать повара.",
                            },
                        ], ensure_ascii=False)
                    return "[]"

                self.fail(f"Unexpected LLM prompt: {user[:200]}")

            with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
                result = ed.process_book(
                    client=object(),
                    config=config,
                    book_name=book_name,
                    text=text,
                    skip_synth=True,
                    workers=1,
                    voice_extractor="llm",
                    books_total=1,
                    books_completed_before=0,
                    pipeline_t0=time.time(),
                )

            self.assertEqual(prompt_counters["dialogues"], 2)
            self.assertEqual(prompt_counters["knowledge"], 2)
            self.assertEqual(len(result["voice_pairs"]), 1)
            self.assertEqual(len(result["knowledge"]), 2)
            self.assertEqual(result["synth_pairs"], [])

            output_paths = ed.get_book_output_paths(tmpdir, book_name)

            self.assertTrue(output_paths["voice"].exists())
            self.assertTrue(output_paths["knowledge_stream"].exists())
            self.assertTrue(output_paths["knowledge"].exists())
            self.assertTrue(output_paths["chunks"].exists())
            self.assertTrue(output_paths["voice_txt"].exists())
            self.assertTrue(output_paths["knowledge_txt"].exists())
            self.assertTrue(output_paths["done"].exists())

            voice_lines = [json.loads(line) for line in output_paths["voice"].read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(voice_lines), 1)
            self.assertEqual(
                voice_lines[0]["messages"][-1]["content"],
                "Ладно, поехали,",
            )

            knowledge_stream_lines = [
                json.loads(line)
                for line in output_paths["knowledge_stream"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(knowledge_stream_lines), 2)

            knowledge_items = json.loads(output_paths["knowledge"].read_text(encoding="utf-8"))
            self.assertEqual(len(knowledge_items), 2)
            self.assertEqual(
                {item["category"] for item in knowledge_items},
                {"event", "custom"},
            )
            self.assertEqual({item["source_book"] for item in knowledge_items}, {book_name})
            self.assertEqual({item["chapter"] for item in knowledge_items}, {"Предисловие"})
            self.assertEqual({item["chunk_idx"] for item in knowledge_items}, {0})

            chunk_records = [
                json.loads(line)
                for line in output_paths["chunks"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(chunk_records), 1)
            self.assertEqual(chunk_records[0]["chapter"], "Предисловие")
            self.assertIn("— Ты готов?", chunk_records[0]["chunk_text"])
            self.assertEqual(len(chunk_records[0]["dialogues"]), 1)
            self.assertEqual(len(chunk_records[0]["knowledge"]), 2)

            voice_txt = output_paths["voice_txt"].read_text(encoding="utf-8")
            knowledge_txt = output_paths["knowledge_txt"].read_text(encoding="utf-8")
            self.assertIn("Ладно, поехали,", voice_txt)
            self.assertIn("Джуффин: Ты готов?", voice_txt)
            self.assertIn("Макс соглашается ехать дальше", knowledge_txt)
            self.assertIn("В Ехо снова начали колдовать повара.", knowledge_txt)


    def test_retrieval_linking_reuses_existing_subject(self):
        config = ed.Config()
        config.knowledge_link_top_k = 4
        knowledge_base = [
            {
                "category": "character",
                "subject": "Маба Калох",
                "fact": "Маба Калох — старый знакомый Джуффина.",
            }
        ]
        new_items = [
            {
                "category": "character",
                "subject": "сэр Маба",
                "fact": "Он пришел к Максу в гости.",
            }
        ]

        def fake_call_llm(client, config, system, user, **kwargs):
            self.assertIn("НОВЫЙ ФАКТ", user)
            self.assertIn("subject=Маба Калох", user)
            return json.dumps(
                {
                    "decision": "reuse_subject",
                    "subject": "Маба Калох",
                    "candidate_id": 1,
                },
                ensure_ascii=False,
            )

        with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
            linked = ed.link_knowledge_items_with_retrieval(
                client=object(),
                config=config,
                items=new_items,
                knowledge_base=knowledge_base,
                log_prefix="[test-link]",
            )

        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["subject"], "Маба Калох")

    def test_timeline_resolution_stage_builds_graph_from_facts_and_chunks(self):
        config = ed.Config(output_dir="")
        book_name = "1. Лабиринты Ехо/1. Чужак.fb2"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = ed.get_book_output_paths(tmpdir, book_name)
            ed.append_jsonl(output_paths["chunks"], [
                {
                    "idx": 0,
                    "chapter": "Предисловие",
                    "chunk_text": "Макс приезжает в Ехо. Джуффин встречает его в Доме у Моста.",
                    "dialogues": [],
                    "knowledge": [],
                },
                {
                    "idx": 1,
                    "chapter": "Предисловие",
                    "chunk_text": "Потом Макс поднимается на крышу Дома у Моста и пьет камру.",
                    "dialogues": [],
                    "knowledge": [],
                },
            ])

            raw_knowledge = [
                {
                    "category": "event",
                    "subject": "Макс",
                    "fact": "Макс приезжает в Ехо.",
                    "time_scope": "change",
                    "source_book": book_name,
                    "chapter": "Предисловие",
                    "chunk_idx": 0,
                },
                {
                    "category": "character",
                    "subject": "Джуффин Халли",
                    "fact": "Джуффин встречает Макса у Дома у Моста.",
                    "time_scope": "current",
                    "source_book": book_name,
                    "chapter": "Предисловие",
                    "chunk_idx": 0,
                },
                {
                    "category": "place",
                    "subject": "Дом у Моста",
                    "fact": "Дом у Моста служит важным местом встреч.",
                    "time_scope": "timeless",
                    "source_book": book_name,
                    "chapter": "Предисловие",
                    "chunk_idx": 1,
                },
                {
                    "category": "custom",
                    "subject": "камра",
                    "fact": "Макс пьет камру на крыше.",
                    "time_scope": "current",
                    "source_book": book_name,
                    "chapter": "Предисловие",
                    "chunk_idx": 1,
                },
            ]
            unique_knowledge = ed.deduplicate_knowledge(raw_knowledge)

            def fake_call_llm(client, config, system, user, **kwargs):
                self.assertIn("ТЕКСТОВЫЕ ФРАГМЕНТЫ", user)
                self.assertIn("Макс приезжает в Ехо", user)
                return json.dumps(
                    {
                        "entities": [
                            {"label": "Макс", "type": "character"},
                            {"label": "Джуффин Халли", "type": "character"},
                            {"label": "Дом у Моста", "type": "place"},
                            {"label": "камра", "type": "item"},
                        ],
                        "events": [
                            {
                                "local_id": "E1",
                                "label": "Макс приезжает в Ехо",
                                "summary": "Макс оказывается в Ехо и встречает Джуффина.",
                                "time_scope": "change",
                                "chunk_indices": [0],
                                "participants": ["Макс", "Джуффин Халли"],
                                "places": ["Дом у Моста"],
                                "objects": [],
                            },
                            {
                                "local_id": "E2",
                                "label": "Макс поднимается на крышу и пьет камру",
                                "summary": "Позже Макс оказывается на крыше Дома у Моста и пьет камру.",
                                "time_scope": "current",
                                "chunk_indices": [1],
                                "participants": ["Макс"],
                                "places": ["Дом у Моста"],
                                "objects": ["камра"],
                            },
                        ],
                        "relations": [
                            {
                                "source": "E1",
                                "target": "E2",
                                "type": "before",
                                "evidence": "Сначала приезд, потом сцена на крыше.",
                                "confidence": "explicit",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )

            with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
                raw_groups, graph = ed.build_timeline_resolution_artifacts(
                    client=object(),
                    config=config,
                    output_dir=tmpdir,
                    book_names=[book_name],
                    raw_knowledge=raw_knowledge,
                    unique_knowledge=unique_knowledge,
                    log_prefix="[test-timeline]",
                )

            global_paths = ed.get_global_output_paths(tmpdir)
            self.assertTrue(global_paths["timeline_raw"].exists())
            self.assertTrue(global_paths["timeline_graph"].exists())
            self.assertTrue(global_paths["timeline_graph_txt"].exists())

            self.assertEqual(len(raw_groups), 1)
            self.assertGreaterEqual(len(graph["nodes"]), 4)
            self.assertTrue(any(node["type"] == "event" for node in graph["nodes"]))
            self.assertTrue(any(edge["type"] == "before" for edge in graph["edges"]))
            self.assertTrue(any(edge["type"] == "occurs_in" for edge in graph["edges"]))


if __name__ == "__main__":
    unittest.main()
