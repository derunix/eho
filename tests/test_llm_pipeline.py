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

    def test_parse_knowledge_line_protocol_accepts_category_as_field_name(self):
        response = (
            "character=Кимпа | fact=Кимпа служит дворецким дома Джуффина. | time_scope=timeless\n"
            "place: Дом у Моста | fact: Дом у Моста служит штабом Тайного Сыска. | time_scope: timeless\n"
        )

        items, strategy = ed.parse_knowledge_line_protocol(
            response,
            log_prefix="[test-line-protocol]",
        )

        self.assertEqual(strategy, "line_protocol")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["category"], "character")
        self.assertEqual(items[0]["subject"], "Кимпа")
        self.assertEqual(items[1]["category"], "place")
        self.assertEqual(items[1]["subject"], "Дом у Моста")

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

    def test_validate_knowledge_skips_non_dict_items(self):
        items = [
            "мусор",
            123,
            {
                "category": "event",
                "subject": "Макс",
                "fact": "Макс вошел в Дом у Моста.",
                "time_scope": "current",
            },
        ]

        validated = ed.validate_knowledge(items)

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["subject"], "Макс")
        self.assertEqual(validated[0]["category"], "event")

    def test_extract_knowledge_does_not_crash_on_non_object_json_items(self):
        config = ed.Config()
        config.knowledge_llm_validation_enabled = False

        with mock.patch.object(ed, "call_llm", return_value='["не факт"]'):
            result = ed.extract_knowledge(
                client=object(),
                config=config,
                chunk="Тестовый фрагмент",
                log_prefix="[test-knowledge]",
            )

        self.assertEqual(result, [])

    def test_extract_knowledge_aggregates_world_and_scene_tracks(self):
        config = ed.Config(
            extraction_passes=1,
            knowledge_extraction_tracks=("world", "scene"),
        )
        config.knowledge_llm_validation_enabled = False

        def fake_call_llm(client, config, system, user, **kwargs):
            if "WORLD_FACTS" in user:
                return json.dumps([
                    {
                        "category": "place",
                        "subject": "Дом у Моста",
                        "fact": "Дом у Моста служит штабом Тайного Сыска.",
                        "time_scope": "timeless",
                    }
                ], ensure_ascii=False)
            if "SCENE_FACTS" in user:
                return json.dumps([
                    {
                        "category": "event",
                        "subject": "Макс",
                        "fact": "Макс поднимается на крышу Дома у Моста.",
                        "time_scope": "current",
                    }
                ], ensure_ascii=False)
            self.fail(f"Unexpected knowledge prompt: {user[:200]}")

        with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
            result = ed.extract_knowledge(
                client=object(),
                config=config,
                chunk="Макс поднимается на крышу Дома у Моста.",
                log_prefix="[test-tracks]",
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category"], "event")
        self.assertEqual(result[0]["subject"], "Макс")

    def test_extract_knowledge_scene_pagination_is_independent_from_world_track(self):
        config = ed.Config(
            extraction_passes=2,
            knowledge_extraction_tracks=("world", "scene"),
        )
        config.knowledge_llm_validation_enabled = False
        config.knowledge_dual_extraction_enabled = False
        track_page_counters = {
            "world": 0,
            "scene": 0,
        }

        def fake_call_llm(client, config, system, user, **kwargs):
            if "MODE: WORLD_FACTS" in user:
                track_page_counters["world"] += 1
                is_second_page = track_page_counters["world"] > 1
                if not is_second_page:
                    return (
                        "category=place | subject=Дом у Моста | "
                        "fact=Дом у Моста служит штабом Тайного Сыска. | time_scope=timeless"
                    )
                return ""
            if "MODE: SCENE_FACTS" in user:
                track_page_counters["scene"] += 1
                is_second_page = track_page_counters["scene"] > 1
                if not is_second_page:
                    return (
                        "category=event | subject=Макс | "
                        "fact=Макс входит в Дом у Моста и замечает странную тишину. | "
                        "time_scope=current"
                    )
                return (
                    "category=event | subject=Джуффин Халли | "
                    "fact=Джуффин Халли приказывает Максу немедленно подняться наверх. | "
                    "time_scope=current"
                )
            self.fail(f"Unexpected knowledge prompt: {user[:200]}")

        with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
            result = ed.extract_knowledge(
                client=object(),
                config=config,
                chunk="Макс входит в Дом у Моста. Джуффин Халли велит ему подняться наверх.",
                log_prefix="[test-track-pagination]",
            )

        self.assertEqual(len(result), 2)
        self.assertEqual({item["subject"] for item in result}, {"Макс", "Джуффин Халли"})

    def test_extract_knowledge_runs_secondary_and_arbiter_on_suspicious_chunk(self):
        config = ed.Config(
            extraction_passes=1,
            knowledge_extraction_tracks=("world",),
        )
        config.knowledge_llm_validation_enabled = False
        config.knowledge_extract_model = "primary-model"
        config.knowledge_dual_extraction_enabled = True
        config.knowledge_extract_model_secondary = "secondary-model"
        config.knowledge_arbiter_model = "arbiter-model"

        calls = {
            "primary": 0,
            "secondary": 0,
            "arbiter": 0,
        }

        chunk = (
            "Дом у Моста служит штабом Тайного Сыска. "
            "Кимпа был старым дворецким дома Джуффина."
        )

        def fake_call_llm(client, config, system, user, **kwargs):
            model_override = kwargs.get("model_override")
            if system == ed.KNOWLEDGE_ARBITER_SYSTEM:
                calls["arbiter"] += 1
                self.assertIn("Кимпа", user)
                self.assertIn("Дом у Моста", user)
                return "1 keep\n2 keep"
            if model_override == "primary-model":
                calls["primary"] += 1
                return (
                    "category=place | subject=Дом у Моста | "
                    "fact=Дом у Моста служит штабом Тайного Сыска. | time_scope=timeless"
                )
            if model_override == "secondary-model":
                calls["secondary"] += 1
                return (
                    "category=character | subject=Кимпа | "
                    "fact=Кимпа был старым дворецким дома Джуффина. | time_scope=timeless"
                )
            self.fail(f"Unexpected LLM call: {system[:60]} / {model_override}")

        with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
            result = ed.extract_knowledge(
                client=object(),
                config=config,
                chunk=chunk,
                log_prefix="[test-ensemble]",
            )

        self.assertEqual(calls["primary"], 1)
        self.assertEqual(calls["secondary"], 1)
        self.assertEqual(calls["arbiter"], 1)
        self.assertEqual(len(result), 2)
        self.assertEqual({item["subject"] for item in result}, {"Дом у Моста", "Кимпа"})

    def test_extract_knowledge_skips_secondary_when_primary_is_healthy(self):
        config = ed.Config(
            extraction_passes=1,
            knowledge_extraction_tracks=("world",),
        )
        config.knowledge_llm_validation_enabled = False
        config.knowledge_extract_model = "primary-model"
        config.knowledge_dual_extraction_enabled = True
        config.knowledge_extract_model_secondary = "secondary-model"
        config.knowledge_arbiter_model = "arbiter-model"

        calls = {
            "primary": 0,
            "secondary": 0,
            "arbiter": 0,
        }

        chunk = (
            "Дом у Моста служит штабом Тайного Сыска. "
            "Кимпа был старым дворецким дома Джуффина. "
            "Макс не мог спать по ночам с детства."
        )

        def fake_call_llm(client, config, system, user, **kwargs):
            model_override = kwargs.get("model_override")
            if system == ed.KNOWLEDGE_ARBITER_SYSTEM:
                calls["arbiter"] += 1
                return "1 keep"
            if model_override == "primary-model":
                calls["primary"] += 1
                return "\n".join([
                    "category=place | subject=Дом у Моста | fact=Дом у Моста служит штабом Тайного Сыска. | time_scope=timeless",
                    "category=character | subject=Кимпа | fact=Кимпа был старым дворецким дома Джуффина. | time_scope=timeless",
                    "category=character | subject=Макс | fact=Макс не мог спать по ночам с детства. | time_scope=past",
                ])
            if model_override == "secondary-model":
                calls["secondary"] += 1
                return ""
            self.fail(f"Unexpected LLM call: {system[:60]} / {model_override}")

        with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
            result = ed.extract_knowledge(
                client=object(),
                config=config,
                chunk=chunk,
                log_prefix="[test-no-secondary]",
            )

        self.assertEqual(calls["primary"], 1)
        self.assertEqual(calls["secondary"], 0)
        self.assertEqual(calls["arbiter"], 0)
        self.assertEqual(len(result), 3)

    def test_extract_knowledge_runs_secondary_when_glossary_terms_are_uncovered(self):
        config = ed.Config(
            extraction_passes=1,
            knowledge_extraction_tracks=("world",),
        )
        config.knowledge_llm_validation_enabled = False
        config.knowledge_extract_model = "primary-model"
        config.knowledge_dual_extraction_enabled = True
        config.knowledge_extract_model_secondary = "secondary-model"
        config.knowledge_arbiter_model = "arbiter-model"
        config.knowledge_ensemble_low_fact_threshold = 0
        config.knowledge_ensemble_drop_ratio_threshold = 1.0
        config.knowledge_ensemble_uncovered_glossary_threshold = 2

        calls = {
            "primary": 0,
            "secondary": 0,
            "arbiter": 0,
        }

        chunk = (
            "Дом у Моста служит штабом Тайного Сыска. "
            "Макс не мог спать по ночам с детства. "
            "Кодекс Хрембера больше не действует. "
            "Амобилеры ездят на магических кристаллах."
        )
        chunk_payload = (
            "[PRIMARY CHUNK]\n"
            f"{chunk}\n\n"
            "[SCENE GLOSSARY]\n"
            "- place: Дом у Моста\n"
            "- history: Кодекс Хрембера\n"
            "- custom: амобилер\n"
        )

        def fake_call_llm(client, config, system, user, **kwargs):
            model_override = kwargs.get("model_override")
            if system == ed.KNOWLEDGE_ARBITER_SYSTEM:
                calls["arbiter"] += 1
                self.assertIn("Кодекс Хрембера", user)
                self.assertIn("амобилер", user)
                return "1 keep\n2 keep"
            if model_override == "primary-model":
                calls["primary"] += 1
                return "\n".join([
                    "category=place | subject=Дом у Моста | fact=Дом у Моста служит штабом Тайного Сыска. | time_scope=timeless",
                    "category=character | subject=Макс | fact=Макс не мог спать по ночам с детства. | time_scope=past",
                    "category=custom | subject=Тайный Сыск | fact=Тайный Сыск расследует магические преступления. | time_scope=timeless",
                ])
            if model_override == "secondary-model":
                calls["secondary"] += 1
                return "\n".join([
                    "category=history | subject=Кодекс Хрембера | fact=Кодекс Хрембера больше не действует. | time_scope=ended",
                    "category=custom | subject=амобилер | fact=Амобилеры ездят на магических кристаллах. | time_scope=timeless",
                ])
            self.fail(f"Unexpected LLM call: {system[:60]} / {model_override}")

        with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
            result = ed.extract_knowledge(
                client=object(),
                config=config,
                chunk=chunk,
                chunk_payload=chunk_payload,
                log_prefix="[test-glossary-secondary]",
            )

        self.assertEqual(calls["primary"], 1)
        self.assertEqual(calls["secondary"], 1)
        self.assertEqual(calls["arbiter"], 1)
        subjects = {item["subject"] for item in result}
        self.assertIn("Кодекс Хрембера", subjects)
        self.assertIn("амобилер", subjects)

    def test_arbiter_can_rewrite_candidate_into_autonomous_fact(self):
        config = ed.Config()
        config.knowledge_arbiter_model = "arbiter-model"

        candidates = [
            {
                "category": "character",
                "subject": "Кимпа",
                "fact": "Кимпа был строго предупрежден хозяином, что должен встретить Макса по первому разряду.",
                "time_scope": "past",
                "_ensemble_source": "secondary",
            }
        ]
        chunk = "Кимпа был старым дворецким дома Джуффина."

        def fake_call_llm(client, config, system, user, **kwargs):
            self.assertEqual(system, ed.KNOWLEDGE_ARBITER_SYSTEM)
            self.assertIn("Кимпа", user)
            return (
                "1 rewrite | category=character | subject=Кимпа | "
                "fact=Кимпа был старым дворецким дома Джуффина. | time_scope=timeless"
            )

        with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
            resolved = ed.arbiter_resolve_knowledge_candidates_with_llm(
                client=object(),
                config=config,
                candidates=candidates,
                chunk=chunk,
                log_prefix="[test-arbiter-rewrite]",
            )

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["subject"], "Кимпа")
        self.assertEqual(resolved[0]["time_scope"], "timeless")
        self.assertIn("дворецким", resolved[0]["fact"])

    def test_extract_knowledge_recovers_wrapper_object_schema(self):
        config = ed.Config(
            extraction_passes=1,
            knowledge_extraction_tracks=("world",),
        )
        config.knowledge_llm_validation_enabled = False

        def fake_call_llm(client, config, system, user, **kwargs):
            return json.dumps(
                {
                    "characters": [
                        {
                            "name": "Макс",
                            "description": "Макс впервые оказывается в Ехо.",
                        },
                        {
                            "name": "Сэр Джуффин",
                            "description": "Сэр Джуффин помогает Максу советами.",
                        },
                    ],
                    "key_events": [
                        {
                            "event": "Прибытие Макса в Ехо",
                            "details": "Макс впервые оказывается в Ехо.",
                        }
                    ],
                    "summary": "Не должно использоваться напрямую как факт.",
                },
                ensure_ascii=False,
            )

        with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
            result = ed.extract_knowledge(
                client=object(),
                config=config,
                chunk="Макс впервые оказывается в Ехо.",
                log_prefix="[test-wrapper-schema]",
            )

        self.assertEqual(len(result), 2)
        event_subjects = {
            item["subject"]
            for item in result
            if item["category"] == "event"
        }
        self.assertIn("Прибытие Макса в Ехо", event_subjects)

    def test_coerce_knowledge_payload_recovers_action_details_schema(self):
        payload = [
            {
                "character": "Макс",
                "action": "пытается понять происходящее.",
                "details": "Он чувствует себя потерянным и постепенно адаптируется.",
            }
        ]

        result = ed.coerce_knowledge_payload_to_items(payload)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category"], "character")
        self.assertEqual(result[0]["subject"], "Макс")
        self.assertIn("адаптируется", result[0]["fact"])

    def test_coerce_knowledge_payload_ignores_weak_root_entity_catalog_schema(self):
        payload = [
            {
                "entity": "Европа",
                "type": "место",
                "description": "Место действия, где Макс Фрай оказался.",
            }
        ]

        result = ed.coerce_knowledge_payload_to_items(payload)

        self.assertEqual(result, [])

    def test_validate_knowledge_drops_entity_catalog_noise_and_keeps_grounded_fact(self):
        payload = [
            {
                "entity": "Макс Фрай",
                "type": "персонаж",
                "description": "Главный герой, который попадает в новый мир.",
            },
            {
                "entity": "Европа",
                "type": "место",
                "description": "Место действия, где Макс Фрай оказался.",
            },
            {
                "entity": "Кимпа",
                "type": "персонаж",
                "description": "Старый дворецкий дома Джуффина Халли.",
            },
        ]
        source_text = (
            "[PRIMARY CHUNK]\n"
            "Когда я впервые очутился в доме сэра Джуффина Халли, его самого не оказалось на месте. "
            "Старый дворецкий Кимпа был немало озадачен. "
            "[SCENE GLOSSARY]\n"
            "- character: Джуффин Халли\n"
            "- character: Макс\n"
            "- place: Ехо\n"
        )

        candidates = ed.coerce_knowledge_payload_to_items(payload)
        validated = ed.validate_knowledge(candidates, source_text=source_text)

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["subject"], "Кимпа")
        self.assertIn("дворецкий", validated[0]["fact"])

    def test_validate_knowledge_rejects_ungrounded_subject(self):
        items = [
            {
                "category": "place",
                "subject": "Европа",
                "fact": "Европа расположена далеко от Ехо.",
            }
        ]

        validated = ed.validate_knowledge(
            items,
            source_text="[PRIMARY CHUNK]\nМакс прибыл в Ехо и встретил Джуффина.\n[SCENE GLOSSARY]\n- place: Ехо\n",
        )

        self.assertEqual(validated, [])

    def test_validate_knowledge_rejects_fact_grounded_only_in_supporting_context(self):
        items = [
            {
                "category": "character",
                "subject": "Макс",
                "fact": "Макс не мог спать по ночам с младенчества.",
            }
        ]

        source_text = (
            "[PRIMARY CHUNK #2]\n"
            "Кимпа встретил меня в холле и молча поклонился.\n\n"
            "[SUPPORTING CONTEXT]\n"
            "[SUPPORTING PREV CHUNK #1 | excerpt]\n"
            "С младенческих лет я не мог спать по ночам.\n"
        )

        validated = ed.validate_knowledge(items, source_text=source_text)

        self.assertEqual(validated, [])

    def test_fact_tokens_grounded_in_primary_rejects_unrelated_summary_fact(self):
        primary = (
            "Сэр Джуффин Халли хлопнул Макса между лопаток. "
            "В Соединенном Королевстве это допустимо только между ближайшими друзьями."
        )

        self.assertFalse(
            ed.fact_tokens_grounded_in_primary(
                "В Ехо архитектура приемлет исключительно приземистые и просторные здания.",
                primary,
                category="place",
                subject="Ехо",
            )
        )

    def test_validate_knowledge_rejects_subject_fact_focus_mismatch(self):
        items = [
            {
                "category": "character",
                "subject": "Макс",
                "fact": "Сэр Джуффин Халли предложил Максу помощь в трудоустройстве.",
            }
        ]

        source_text = (
            "[PRIMARY CHUNK]\n"
            "Сэр Джуффин Халли предложил Максу помощь в трудоустройстве.\n"
        )

        validated = ed.validate_knowledge(items, source_text=source_text)

        self.assertEqual(validated, [])

    def test_validate_knowledge_items_with_llm_drops_non_autonomous_facts(self):
        config = ed.Config()
        items = [
            {
                "category": "place",
                "subject": "Дом у Моста",
                "fact": "Дом у Моста служит штабом Тайного Сыска.",
                "time_scope": "timeless",
            },
            {
                "category": "character",
                "subject": "Джуффин Халли",
                "fact": "Джуффин Халли удивился зову, спросив: «Кому это приспичило?»",
                "time_scope": "current",
            },
        ]

        def fake_call_llm(client, config, system, user, **kwargs):
            self.assertIn("PRIMARY CHUNK", user)
            self.assertIn("Дом у Моста служит штабом Тайного Сыска.", user)
            return json.dumps(
                [
                    {"idx": 1, "decision": "keep", "reason": "standalone"},
                    {"idx": 2, "decision": "drop", "reason": "too local"},
                ],
                ensure_ascii=False,
            )

        with mock.patch.object(ed, "call_llm", side_effect=fake_call_llm):
            validated = ed.validate_knowledge_items_with_llm(
                client=object(),
                config=config,
                items=items,
                chunk="Дом у Моста был штабом, а Джуффин удивился зову.",
                log_prefix="[test-llm-validate]",
            )

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["subject"], "Дом у Моста")

    def test_merge_knowledge_items_keeps_distinct_facts_during_extraction(self):
        existing = [
            {
                "category": "character",
                "subject": "Шурф Лонли-Локли",
                "fact": "Шурф носит Перчатки Смерти.",
                "time_scope": "timeless",
            }
        ]
        new_items = [
            {
                "category": "character",
                "subject": "Шурф Лонли-Локли",
                "fact": "Шурф носит тюрбан.",
                "time_scope": "current",
            }
        ]

        merged, added = ed.merge_knowledge_items(existing, new_items)

        self.assertEqual(added, 1)
        self.assertEqual(len(merged), 2)

    def test_parse_json_response_repairs_unescaped_quotes_in_strings(self):
        response = (
            '[{"category": "place", "subject": "Дом", '
            '"fact": "Место, где происходят события, возможно, в контексте "Дома у Моста"."}]'
        )

        data, strategy = ed.parse_json_response(response, expect="array")

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["subject"], "Дом")
        self.assertIn("Дома у Моста", data[0]["fact"])
        self.assertIn("repair_quotes", strategy)

    def test_parse_json_response_recovers_partial_truncated_array(self):
        response = (
            '[{"category":"character","subject":"Мелифаро","fact":"Мелифаро любит яркую одежду."},'
            '{"category":"place","subject":"Дом у Моста","fact":"Дом у Моста является важным местом."},'
            '{"category":"event","subject":"Макс","fact":"Макс'
        )

        data, strategy = ed.parse_json_response(response, expect="array")

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["subject"], "Мелифаро")
        self.assertEqual(data[1]["subject"], "Дом у Моста")
        self.assertIn("partial_array", strategy)

    def test_validate_knowledge_normalizes_russian_schema_keys(self):
        items = [
            {
                "категория": "персонаж",
                "имя": "Мелифаро",
                "описание": "Мелифаро любит яркую одежду.",
                "время": "timeless",
            }
        ]

        validated = ed.validate_knowledge(items)

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["category"], "character")
        self.assertEqual(validated[0]["subject"], "Мелифаро")
        self.assertEqual(validated[0]["fact"], "Мелифаро любит яркую одежду.")
        self.assertEqual(validated[0]["time_scope"], "timeless")

    def test_validate_knowledge_normalizes_extended_category_aliases(self):
        items = [
            {
                "type": "historical_period",
                "entity": "Эпоха Кодекса",
                "description": "Эпоха Кодекса началась после принятия Кодекса Хрембера.",
            }
        ]

        validated = ed.validate_knowledge(items)

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["category"], "history")
        self.assertEqual(validated[0]["subject"], "Эпоха Кодекса")

    def test_validate_knowledge_drops_generic_entity_descriptions(self):
        items = [
            {
                "category": "character",
                "subject": "Мелифаро",
                "fact": "Персонаж, который присутствует в сцене.",
            },
            {
                "category": "place",
                "subject": "Дом у Моста",
                "fact": "Место, где происходят события.",
            },
            {
                "category": "custom",
                "subject": "Обжор",
                "fact": "Это место, где персонажи заказывают еду и напитки.",
            },
            {
                "category": "magic",
                "subject": "Дитя Багровой Жемчужины Гурига VII",
                "fact": "Это миф, то, чего нет, и его обнаружил мудрый старый Магистр.",
            },
        ]

        validated = ed.validate_knowledge(items)

        self.assertEqual(validated, [])

    def test_validate_knowledge_drops_generic_or_ephemeral_character_facts(self):
        items = [
            {
                "category": "character",
                "subject": "старушка",
                "fact": "старушка наблюдательна и умеет замечать, что говор человека не является столичным.",
            },
            {
                "category": "character",
                "subject": "Джуффин Халли",
                "fact": "Джуффин Халли удивился зову, спросив: «Кому это приспичило?»",
            },
            {
                "category": "character",
                "subject": "Джуффин Халли",
                "fact": "Джуффин Халли посоветовал сэру Максу ограничить зону своей разрушительной деятельности этим кабинетом.",
            },
            {
                "category": "character",
                "subject": "Кимпа",
                "fact": "Кимпа был строго предупрежден хозяином, что должен встретить Макса по первому разряду.",
            },
            {
                "category": "character",
                "subject": "Кимпа",
                "fact": "Кимпа помог Максу одеться, чтобы тот выглядел пристойно для коренного жителя Ехо.",
            },
            {
                "category": "character",
                "subject": "Сэр Шурф Лонли-Локли",
                "fact": "Фамилия Сэр Шурф Лонли-Локли состоит из десятка букв.",
            },
            {
                "category": "character",
                "subject": "Сэр Шурф Лонли-Локли",
                "fact": "Сэр Шурф Лонли-Локли потребовал от Мелифаро запомнить его фамилию.",
            },
        ]

        validated = ed.validate_knowledge(items)

        self.assertEqual(validated, [])

    def test_validate_knowledge_drops_scene_summary_noise(self):
        items = [
            {
                "category": "character",
                "subject": "Руди",
                "fact": "Руди — это персонаж, который, по-видимому, является источником шума или внимания в данной сцене.",
            },
            {
                "category": "place",
                "subject": "гостиная",
                "fact": "В центре гостиной стоял огромный прозрачный сосуд, в котором произрастал гигантский светящийся гриб.",
                "time_scope": "current",
            },
            {
                "category": "place",
                "subject": "след",
                "fact": "Место, где Макс обнаружил следы, было местом, где он смог определить направление.",
            },
            {
                "category": "event",
                "subject": "обед в «Обжоре Бунбу»",
                "fact": "Лонли-Локли и Макс отправились обедать в «Обжору Бунбу», где их ждал сэр Джуффин.",
                "time_scope": "current",
            },
            {
                "category": "event",
                "subject": "утренний «подвиг» Макса",
                "fact": "Макс рассказал Шурфу о своем утреннем «подвиге», который вызвал озабоченность Лонли-Локли.",
                "time_scope": "current",
            },
            {
                "category": "event",
                "subject": "Сэр Шурф",
                "fact": "Сэр Шурф унес Мелифаро под мышкой, как свернутый в рулон ковер.",
                "time_scope": "past",
            },
            {
                "category": "event",
                "subject": "Джуффин Халли",
                "fact": "Джуффин Халли назвал Макса лихим ветром.",
                "time_scope": "past",
            },
            {
                "category": "custom",
                "subject": "Сэр Джуффин Халли",
                "fact": "Сэр Джуффин Халли хлопнул Макса между лопаток, что в Соединённом Королевстве допустимо только между ближайшими друзьями.",
                "time_scope": "past",
            },
            {
                "category": "custom",
                "subject": "Сон",
                "fact": "Сон унесет незаконнорожденные гримасы иного мира, и все забудется.",
                "time_scope": "past",
            },
            {
                "category": "event",
                "subject": "Джуффин Халли",
                "fact": "Джуффин Халли приказывает Максу и Шурфу отправиться в «Обжору» на праздник воскрешения Мелифаро.",
                "time_scope": "past",
            },
            {
                "category": "character",
                "subject": "Сэр Макс",
                "fact": "Сэр Макс заявляет, что те, кто могли бы рассказать о зеркальцах, умолкли навеки.",
                "time_scope": "past",
            },
        ]

        validated = ed.validate_knowledge(items)

        self.assertEqual(validated, [])

    def test_process_chunks_parallel_survives_chunk_exception(self):
        config = ed.Config()

        with mock.patch.object(ed, "extract_knowledge", side_effect=RuntimeError("boom")):
            dialogues, knowledge = ed.process_chunks_parallel(
                client=object(),
                config=config,
                chunks=["Первый чанк"],
                do_voice=False,
                do_knowledge=True,
                workers=1,
                return_results=True,
            )

        self.assertEqual(dialogues, [])
        self.assertEqual(knowledge, [])

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
                knowledge_extraction_tracks=("world",),
            )
            config.knowledge_llm_validation_enabled = False
            config.knowledge_dual_extraction_enabled = False
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

                if "автономные факты для базы знаний мира Ехо" in user:
                    prompt_counters["knowledge"] += 1
                    if prompt_counters["knowledge"] == 1:
                        return (
                            "category=event | subject=Макс | "
                            "fact=Макс соглашается ехать дальше вместе с Джуффином. | "
                            "time_scope=current\n"
                            "category=custom | subject=Ехо | "
                            "fact=В Ехо снова начали колдовать повара. | "
                            "time_scope=current"
                        )
                    return ""

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
            self.assertEqual(len(result["knowledge"]), 1)
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
            self.assertEqual(len(knowledge_stream_lines), 1)

            knowledge_items = json.loads(output_paths["knowledge"].read_text(encoding="utf-8"))
            self.assertEqual(len(knowledge_items), 1)
            self.assertEqual({item["category"] for item in knowledge_items}, {"custom"})
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
            self.assertEqual(len(chunk_records[0]["knowledge"]), 1)

            voice_txt = output_paths["voice_txt"].read_text(encoding="utf-8")
            knowledge_txt = output_paths["knowledge_txt"].read_text(encoding="utf-8")
            self.assertIn("Ладно, поехали,", voice_txt)
            self.assertIn("Джуффин: Ты готов?", voice_txt)
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

    def test_retrieval_linking_does_not_drop_non_duplicate_fact(self):
        config = ed.Config()
        config.knowledge_link_top_k = 4
        knowledge_base = [
            {
                "category": "character",
                "subject": "Маба Калох",
                "fact": "Маба Калох — старый знакомый Джуффина.",
                "time_scope": "timeless",
            }
        ]
        new_items = [
            {
                "category": "character",
                "subject": "сэр Маба",
                "fact": "Маба Калох пришел к Максу в гости.",
                "time_scope": "current",
            }
        ]

        def fake_call_llm(client, config, system, user, **kwargs):
            return json.dumps(
                {
                    "decision": "drop_duplicate",
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
                log_prefix="[test-link-drop]",
            )

        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["fact"], "Маба Калох пришел к Максу в гости.")

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
