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

            chunk_records = [
                json.loads(line)
                for line in output_paths["chunks"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(chunk_records), 1)
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


if __name__ == "__main__":
    unittest.main()
