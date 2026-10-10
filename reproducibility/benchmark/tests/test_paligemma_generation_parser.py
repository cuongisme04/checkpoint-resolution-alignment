import unittest

from scripts.bench_hf_paligemma_classification import (
    format_generation_class_list,
    generation_class_indices,
    generation_prompt,
    parse_generated_label,
)


CLASS_IDS = [
    "Abyssinian",
    "Siamese",
    "Persian",
    "german_shorthaired",
]
CLASS_NAMES = [
    "abyssinian",
    "siamese",
    "persian",
    "german shorthaired",
]


class PaliGemmaGenerationParserTest(unittest.TestCase):
    def test_accepts_exact_label_formatting(self):
        index, status = parse_generated_label("german_shorthaired", CLASS_IDS, CLASS_NAMES)
        self.assertEqual(index, 3)
        self.assertEqual(status, "exact")

    def test_accepts_full_sentence_with_alias(self):
        index, status = parse_generated_label(
            "The breed is a German Shorthaired Pointer.", CLASS_IDS, CLASS_NAMES
        )
        self.assertEqual(index, 3)
        self.assertEqual(status, "substring")

    def test_accepts_numbered_option(self):
        index, status = parse_generated_label(
            "option 2",
            CLASS_IDS,
            CLASS_NAMES,
            option_class_indices=[2, 0, 3, 1],
        )
        self.assertEqual(index, 0)
        self.assertEqual(status, "option_number")

    def test_rejects_out_of_range_numbered_option(self):
        index, status = parse_generated_label(
            "option 99",
            CLASS_IDS,
            CLASS_NAMES,
            option_class_indices=[2, 0, 3, 1],
        )
        self.assertIsNone(index)
        self.assertEqual(status, "unknown_option_number")

    def test_rejects_empty_generation(self):
        index, status = parse_generated_label("   ", CLASS_IDS, CLASS_NAMES)
        self.assertIsNone(index)
        self.assertEqual(status, "empty")

    def test_rejects_unknown_generation(self):
        index, status = parse_generated_label("tabby cat", CLASS_IDS, CLASS_NAMES)
        self.assertIsNone(index)
        self.assertEqual(status, "unknown")

    def test_rejects_ambiguous_generation(self):
        index, status = parse_generated_label("siamese or persian", CLASS_IDS, CLASS_NAMES)
        self.assertIsNone(index)
        self.assertEqual(status, "ambiguous")

    def test_numbered_prompt_uses_requested_class_order(self):
        class_indices = [2, 0, 3, 1]
        class_list = format_generation_class_list(
            CLASS_NAMES, class_indices, list_format="numbered"
        )
        self.assertEqual(
            class_list,
            "1. persian\n2. abyssinian\n3. german shorthaired\n4. siamese",
        )
        prompt = generation_prompt(
            "Choose one:\n{class_list}", CLASS_NAMES, class_indices, "numbered"
        )
        self.assertIn("2. abyssinian", prompt)

    def test_shuffle_order_is_deterministic(self):
        first = generation_class_indices(len(CLASS_NAMES), "shuffle", seed=7)
        second = generation_class_indices(len(CLASS_NAMES), "shuffle", seed=7)
        self.assertEqual(first, second)
        self.assertCountEqual(first, range(len(CLASS_NAMES)))


if __name__ == "__main__":
    unittest.main()
