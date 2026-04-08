import sys
import types
import unittest
from pathlib import Path


try:
    import openai  # noqa: F401
except Exception:
    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        pass

    fake_openai.OpenAI = FakeOpenAI
    sys.modules["openai"] = fake_openai


import extract_dialogues as ed


REAL_BOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "books"
    / "1. Лабиринты Ехо"
    / "1. Чужак.fb2"
)


@unittest.skipUnless(REAL_BOOK_PATH.exists(), "real book fixture is unavailable")
class RealBookRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ed.Config()
        cls.text = ed.load_fb2_file(REAL_BOOK_PATH)
        cls.chunks = ed.split_into_chunks(
            cls.text,
            cls.config.chunk_size,
            cls.config.chunk_overlap,
        )

    def test_real_book_chunking_keeps_expected_granularity(self):
        self.assertGreater(len(self.text), 1_000_000)
        self.assertGreater(len(self.chunks), 120)
        self.assertLess(len(self.chunks), 250)
        self.assertIn("Никогда не знаешь, где тебе повезет.", self.chunks[1])
        self.assertIn("Наконец меня оставили в покое", self.chunks[2])

    def test_real_book_scene_glossary_finds_real_entities(self):
        glossary = ed.build_scene_glossary(self.chunks[1], limit=16)

        self.assertIn("character: Джуффин Халли", glossary)
        self.assertIn("place: Ехо", glossary)
        self.assertNotIn("Европа", glossary)
        self.assertNotIn("Два персонажа", glossary)

    def test_real_book_validate_knowledge_keeps_grounded_fact_and_drops_noise(self):
        payload, _ = ed.build_extraction_chunk_payload(self.chunks, 2, self.config)
        items = [
            {
                "category": "character",
                "subject": "Кимпа",
                "fact": "Кимпа — старый дворецкий.",
            },
            {
                "category": "place",
                "subject": "сарайчик",
                "fact": "В конце сада находится небольшой нарядный сарайчик.",
            },
            {
                "category": "place",
                "subject": "Европа",
                "fact": "Место действия, где Макс оказался.",
            },
        ]

        validated = ed.validate_knowledge(items, source_text=payload)

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["subject"], "Кимпа")
        self.assertEqual(validated[0]["fact"], "Кимпа — старый дворецкий.")

    def test_real_book_validate_knowledge_keeps_stable_rule_and_drops_meta_group(self):
        payload, _ = ed.build_extraction_chunk_payload(self.chunks, 20, self.config)
        items = [
            {
                "category": "character",
                "subject": "Шурф Лонли-Локли",
                "fact": "Шурф Лонли-Локли носит толстые защитные перчатки.",
            },
            {
                "category": "custom",
                "subject": "Соединенное Королевство",
                "fact": "В Соединенном Королевстве хлопать между лопаток допустимо только между ближайшими друзьями.",
            },
            {
                "category": "character",
                "subject": "Два друга",
                "fact": "Два друга, видимо, были в состоянии, когда им было необходимо, чтобы их оставили в покое.",
            },
        ]

        validated = ed.validate_knowledge(items, source_text=payload)
        by_subject = {item["subject"]: item for item in validated}

        self.assertEqual(set(by_subject), {"Шурф Лонли-Локли", "Соединенное Королевство"})
        self.assertEqual(by_subject["Соединенное Королевство"]["time_scope"], "timeless")

    def test_real_book_regex_voice_survives_validation(self):
        dialogues, stats = ed.extract_voice_with_regex(self.chunks[20])
        validated = ed.validate_dialogues(dialogues, source_chunk=self.chunks[20])

        self.assertGreaterEqual(stats["monologue"], 1)
        self.assertGreaterEqual(len(dialogues), 1)
        self.assertGreaterEqual(len(validated), 1)
        self.assertTrue(any(item["type"] == "monologue" for item in validated))


if __name__ == "__main__":
    unittest.main()
