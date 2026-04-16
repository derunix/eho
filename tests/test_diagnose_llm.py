import types
import unittest


import diagnose_llm as dl


class DiagnoseCliDefaultsTest(unittest.TestCase):
    def test_all_keeps_explicit_group_selection(self):
        args = types.SimpleNamespace(
            all=True,
            group=["4090"],
            include_failed=False,
            ensemble=False,
            kv_cache_test=False,
            context_test=False,
            loq_test=False,
        )

        dl.apply_all_flag_defaults(args)

        self.assertEqual(args.group, ["4090"])
        self.assertTrue(args.include_failed)
        self.assertTrue(args.ensemble)
        self.assertTrue(args.kv_cache_test)
        self.assertTrue(args.context_test)
        self.assertTrue(args.loq_test)

    def test_all_expands_default_group_to_abc(self):
        args = types.SimpleNamespace(
            all=True,
            group=["A", "B"],
            include_failed=False,
            ensemble=False,
            kv_cache_test=False,
            context_test=False,
            loq_test=False,
        )

        dl.apply_all_flag_defaults(args)

        self.assertEqual(args.group, ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
