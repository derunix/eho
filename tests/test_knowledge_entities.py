import json
import sys
import tempfile
import types
import unittest


try:
    import openai  # noqa: F401
except Exception:
    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        pass

    fake_openai.OpenAI = FakeOpenAI
    sys.modules["openai"] = fake_openai


import extract_dialogues as ed


class KnowledgeEntityResolutionTest(unittest.TestCase):
    def test_character_aliases_and_placeholders_are_cleaned(self):
        items = [
            {
                "category": "character",
                "subject": "Андэ",
                "fact": "Он упоминает, что ему скучно.",
            },
            {
                "category": "character",
                "subject": "Андэ Пу",
                "fact": "Он работает репортером для «Королевского голоса».",
            },
            {
                "category": "character",
                "subject": "Леди Меламори Блимм",
                "fact": "Она улыбается.",
            },
            {
                "category": "character",
                "subject": "Меламори",
                "fact": "Меламори пришла раньше.",
            },
            {
                "category": "place",
                "subject": "Место",
                "fact": "В доме с Максом и его компанией было много мест, где происходили события.",
            },
            {
                "category": "place",
                "subject": "Место действия",
                "fact": "В конце дня группа направляется в сторону, где их ждет отдых.",
            },
        ]

        cleaned = ed.canonicalize_book_knowledge(items, narrator="Макс")

        subjects = [item["subject"] for item in cleaned]
        self.assertEqual(subjects.count("Андэ Пу"), 2)
        self.assertEqual(subjects.count("Меламори Блимм"), 2)
        self.assertNotIn("Место", subjects)
        self.assertNotIn("Место действия", subjects)

        ande_facts = [item["fact"] for item in cleaned if item["subject"] == "Андэ Пу"]
        self.assertTrue(any(fact.startswith("Андэ Пу") for fact in ande_facts))

        melamori_facts = [item["fact"] for item in cleaned if item["subject"] == "Меламори Блимм"]
        self.assertTrue(any(fact.startswith("Меламори Блимм") for fact in melamori_facts))

    def test_short_name_is_not_merged_when_full_candidates_are_ambiguous(self):
        items = [
            {
                "category": "character",
                "subject": "Андэ",
                "fact": "Он пришел первым.",
            },
            {
                "category": "character",
                "subject": "Андэ Пу",
                "fact": "Андэ Пу работает репортером.",
            },
            {
                "category": "character",
                "subject": "Андэ Бим",
                "fact": "Андэ Бим служит при дворе.",
            },
        ]

        cleaned = ed.canonicalize_book_knowledge(items, narrator="Макс")
        subjects = [item["subject"] for item in cleaned]

        self.assertIn("Андэ", subjects)
        self.assertIn("Андэ Пу", subjects)
        self.assertIn("Андэ Бим", subjects)

    def test_honorifics_are_always_collapsed_for_character_subjects(self):
        items = [
            {
                "category": "character",
                "subject": "Сэр Джуффин Халли",
                "fact": "Он задал вопрос.",
            },
            {
                "category": "character",
                "subject": "Джуффин",
                "fact": "Джуффин смотрел на Макса с насмешкой.",
            },
        ]

        cleaned = ed.canonicalize_book_knowledge(items, narrator="Макс")
        subjects = [item["subject"] for item in cleaned]

        self.assertEqual(subjects.count("Джуффин Халли"), 2)
        self.assertNotIn("Сэр Джуффин Халли", subjects)

    def test_place_subjects_are_canonicalized_and_generic_places_are_dropped(self):
        items = [
            {
                "category": "place",
                "subject": "К новая квартира на улице Желтых Камней",
                "fact": "Квартира имеет шесть огромных комнат.",
            },
            {
                "category": "place",
                "subject": "на улице Желтых Камней",
                "fact": "Улица была все еще темной.",
            },
            {
                "category": "place",
                "subject": "улице Желтых Камней",
                "fact": "Там было темно.",
            },
            {
                "category": "place",
                "subject": "Кабинет",
                "fact": "Это место, где Король указал гостям садиться и где было подано угощение.",
            },
            {
                "category": "place",
                "subject": "улица",
                "fact": "Улица была все еще темной.",
            },
            {
                "category": "place",
                "subject": "в кабинете Короля",
                "fact": "Это место, где Король указал гостям садиться и где было подано угощение.",
            },
        ]

        cleaned = ed.canonicalize_book_knowledge(items, narrator="Макс")
        subjects = [item["subject"] for item in cleaned]

        self.assertIn("новая квартира на улице Желтых Камней", subjects)
        self.assertEqual(subjects.count("улица Желтых Камней"), 2)
        self.assertIn("кабинет Короля", subjects)
        self.assertNotIn("Кабинет", subjects)
        self.assertNotIn("улица", subjects)

        cabinet_facts = [item["fact"] for item in cleaned if item["subject"] == "кабинет Короля"]
        self.assertTrue(any(fact.startswith("кабинет Короля") for fact in cabinet_facts))

    def test_descriptive_scene_fragments_are_not_kept_as_places(self):
        items = [
            {
                "category": "place",
                "subject": "дом и сад",
                "fact": "Рассказчик осмотрел дом и сад, где смог расслабиться на траве.",
            },
            {
                "category": "place",
                "subject": "парадная дверь с надписью «Здесь живет сэр Маклук»",
                "fact": "Эта дверь ведет в место, где происходит встреча, и к ней подходят четверо слуг в одинаковых серых одеждах.",
            },
            {
                "category": "place",
                "subject": "Дом у Моста",
                "fact": "Дом у Моста служит штабом Тайного Сыска.",
            },
        ]

        cleaned = ed.canonicalize_book_knowledge(items, narrator="Макс")
        subjects = [item["subject"] for item in cleaned]

        self.assertNotIn("дом и сад", subjects)
        self.assertNotIn("парадная дверь с надписью «Здесь живет сэр Маклук»", subjects)
        self.assertIn("Дом у Моста", subjects)

    def test_canonical_character_dictionary_is_applied(self):
        items = [
            {
                "category": "character",
                "subject": "сэр Маба",
                "fact": "Он внимательно посмотрел на Макса.",
            },
            {
                "category": "character",
                "subject": "Кофа",
                "fact": "Он отправился пить камру.",
            },
        ]

        cleaned = ed.canonicalize_book_knowledge(items, narrator="Макс")
        subjects = [item["subject"] for item in cleaned]

        self.assertIn("Маба Калох", subjects)
        self.assertIn("Кофа Йох", subjects)
        self.assertNotIn("Безумный Рыбник", subjects)

    def test_global_canonicalization_merges_subjects_between_books(self):
        items = [
            {
                "category": "character",
                "subject": "Маба Калох",
                "fact": "Маба Калох дружит с Джуффином.",
            },
            {
                "category": "character",
                "subject": "сэр Маба",
                "fact": "Он пришел к Максу в гости.",
            },
            {
                "category": "place",
                "subject": "на улице Желтых Камней",
                "fact": "Там часто бывало темно по утрам.",
            },
            {
                "category": "place",
                "subject": "улица Желтых Камней",
                "fact": "Улица тянулась вдоль домов Старого Города.",
            },
        ]

        cleaned = ed.canonicalize_global_knowledge(items)
        subjects = [item["subject"] for item in cleaned]

        self.assertEqual(subjects.count("Маба Калох"), 2)
        self.assertEqual(subjects.count("улица Желтых Камней"), 2)

    def test_global_knowledge_stage_collects_per_book_files_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            book_one = "1. Лабиринты Ехо/1. Чужак.fb2"
            book_two = "2. Волонтеры Вечности/1. История.fb2"

            book_one_paths = ed.get_book_output_paths(tmpdir, book_one)
            book_two_paths = ed.get_book_output_paths(tmpdir, book_two)

            first_book_items = [
                {
                    "category": "character",
                    "subject": "сэр Маба",
                    "fact": "Он пришел к Максу в гости.",
                },
                {
                    "category": "place",
                    "subject": "на улице Желтых Камней",
                    "fact": "Там было темно.",
                },
            ]
            second_book_items = [
                {
                    "category": "character",
                    "subject": "Маба Калох",
                    "fact": "Маба Калох пришел к Максу в гости.",
                },
                {
                    "category": "place",
                    "subject": "улица Желтых Камней",
                    "fact": "улица Желтых Камней была темной.",
                },
            ]

            with open(book_one_paths["knowledge"], "w", encoding="utf-8") as f:
                json.dump(first_book_items, f, ensure_ascii=False, indent=2)
            with open(book_two_paths["knowledge"], "w", encoding="utf-8") as f:
                json.dump(second_book_items, f, ensure_ascii=False, indent=2)

            raw_global, unique_global = ed.build_global_knowledge_base(
                tmpdir,
                [book_one, book_two],
                log_prefix="[test-global-stage]",
            )
            global_paths = ed.get_global_output_paths(tmpdir)

            self.assertTrue(global_paths["knowledge_raw"].exists())
            self.assertTrue(global_paths["knowledge_raw_txt"].exists())
            self.assertTrue(global_paths["knowledge"].exists())
            self.assertTrue(global_paths["knowledge_txt"].exists())

            self.assertEqual(len(raw_global), 4)
            self.assertLess(len(unique_global), len(raw_global))

            subjects = [item["subject"] for item in unique_global]
            self.assertIn("Маба Калох", subjects)
            self.assertIn("улица Желтых Камней", subjects)
            self.assertNotIn("сэр Маба", subjects)
            self.assertNotIn("на улице Желтых Камней", subjects)

            with open(global_paths["knowledge_raw"], encoding="utf-8") as f:
                persisted_raw = json.load(f)
            with open(global_paths["knowledge"], encoding="utf-8") as f:
                persisted_unique = json.load(f)

            self.assertEqual(len(persisted_raw), 4)
            self.assertEqual(len(persisted_unique), len(unique_global))

    def test_placeholder_subject_patterns_are_dropped(self):
        items = [
            {
                "category": "character",
                "subject": "этот человек",
                "fact": "Он молча стоял у двери.",
            },
            {
                "category": "character",
                "subject": "неизвестный маг",
                "fact": "Он попытался заговорить первым.",
            },
            {
                "category": "character",
                "subject": "его друг",
                "fact": "Он пришел вместе с ним.",
            },
            {
                "category": "place",
                "subject": "где-то в Ехо",
                "fact": "Там было тихо и сыро.",
            },
            {
                "category": "place",
                "subject": "в каком-то месте",
                "fact": "Там можно было укрыться от дождя.",
            },
        ]

        cleaned = ed.canonicalize_book_knowledge(items, narrator="Макс")
        self.assertEqual(cleaned, [])

    def test_subjects_with_different_canonical_names_are_not_duplicates(self):
        self.assertFalse(ed.subjects_look_duplicate("Кофа", "Шурф"))
        self.assertTrue(ed.subjects_look_duplicate("Кофа", "Кофа Йох"))
        self.assertFalse(ed.subjects_look_duplicate("Безумный Рыбник", "Шурф Лонли-Локли"))
        self.assertFalse(ed.subjects_look_duplicate("Мелифаро", "Анчифа Мелифаро"))

    def test_special_cases_for_subpersonality_and_surname_only_character(self):
        items = [
            {
                "category": "character",
                "subject": "Безумный Рыбник",
                "fact": "Он проявляется как отдельная безумная часть сознания Шурфа.",
            },
            {
                "category": "character",
                "subject": "Шурф",
                "fact": "Он тщательно контролирует себя и говорит подчеркнуто правильно.",
            },
            {
                "category": "character",
                "subject": "Мелифаро",
                "fact": "Он работает в Тайном Сыске и постоянно шутит.",
            },
            {
                "category": "character",
                "subject": "Анчифа Мелифаро",
                "fact": "Он приходится Мелифаро родственником.",
            },
        ]

        cleaned = ed.canonicalize_book_knowledge(items, narrator="Макс")
        subjects = [item["subject"] for item in cleaned]

        self.assertIn("Безумный Рыбник", subjects)
        self.assertIn("Шурф Лонли-Локли", subjects)
        self.assertIn("Мелифаро", subjects)
        self.assertIn("Анчифа Мелифаро", subjects)
        self.assertEqual(subjects.count("Безумный Рыбник"), 1)
        self.assertEqual(subjects.count("Шурф Лонли-Локли"), 1)

    def test_fact_duplicate_detection_is_less_aggressive(self):
        self.assertFalse(
            ed.facts_look_duplicate(
                "Шурф носит Перчатки Смерти и никогда не снимает их на людях.",
                "Шурф носит тюрбан и говорит подчеркнуто вежливо.",
            )
        )
        self.assertTrue(
            ed.facts_look_duplicate(
                "Шурф почти постоянно носит Перчатки Смерти.",
                "Шурф носит Перчатки Смерти почти постоянно.",
            )
        )


    def test_prompts_constrain_custom_category_to_stable_world_concepts(self):
        self.assertIn('Для category="custom" извлекай только УСТОЙЧИВЫЕ элементы мира и быта', ed.KNOWLEDGE_PROMPT)
        self.assertIn('Не создавай category="custom" для разовых сценических деталей', ed.KNOWLEDGE_PROMPT)
        self.assertIn('полумесяц из плотной ткани с карманами', ed.KNOWLEDGE_PROMPT)
        self.assertIn('Для `custom` извлекай только устойчивые элементы мира и быта', ed.COMBINED_PROMPT)
        self.assertIn('Не используй `custom` для разовых сценических деталей', ed.COMBINED_PROMPT)
        self.assertIn('ведьмочки', ed.COMBINED_PROMPT)


if __name__ == "__main__":
    unittest.main()
