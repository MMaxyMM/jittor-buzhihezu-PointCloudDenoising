import unittest

from src.config_override import apply_overrides


class ConfigOverrideTest(unittest.TestCase):
    def test_typed_and_nested_values(self):
        original = {"steps": 4, "condition": True, "nested": {"value": 1}}
        updated = apply_overrides(
            original,
            [
                "steps=8",
                "condition=false",
                "nested.value=0.1",
                "prediction_type=epsilon",
            ],
        )
        self.assertEqual(updated["steps"], 8)
        self.assertFalse(updated["condition"])
        self.assertAlmostEqual(updated["nested"]["value"], 0.1)
        self.assertEqual(updated["prediction_type"], "epsilon")
        self.assertEqual(original["steps"], 4)


if __name__ == "__main__":
    unittest.main()
