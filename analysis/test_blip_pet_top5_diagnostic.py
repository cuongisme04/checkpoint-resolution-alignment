import unittest

from blip_pet_top5_diagnostic import summarize_pair


def row(sample, top1, top5):
    return {"sample_id": sample, "correct_top1": str(top1), "correct_top5": str(top5)}


class TopFiveDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.a = [row("gain", 1, 1), row("loss", 0, 1), row("neither", 0, 0)]
        self.b = [row("gain", 0, 1), row("loss", 1, 1), row("neither", 0, 1)]

    def test_hand_counted_changes_and_row_order(self):
        result = summarize_pair(self.a, list(reversed(self.b)))
        self.assertEqual(result["paired_images"], 3)
        self.assertEqual(result["top1_changes_total"], 2)
        self.assertEqual(result["top1_changes_true_label_in_top5_at_both"], 2)
        self.assertEqual(result["metrics"]["top1"]["gains_at_336"], 1)
        self.assertEqual(result["metrics"]["top1"]["losses_at_336"], 1)
        self.assertAlmostEqual(result["metrics"]["top5"]["difference_percentage_points"], -100 / 3)
        self.assertEqual(result, summarize_pair(list(reversed(self.a)), self.b))

    def test_direction_and_non_top_five_changes(self):
        result = summarize_pair([row("x", 1, 1)], [row("x", 0, 0)])
        self.assertEqual(result["top1_changes_total"], 1)
        self.assertEqual(result["top1_changes_true_label_in_top5_at_both"], 0)
        self.assertEqual(result["metrics"]["top1"]["difference_percentage_points"], 100)
        swapped = summarize_pair(self.b, self.a)
        self.assertAlmostEqual(swapped["metrics"]["top5"]["difference_percentage_points"], 100 / 3)

    def test_rejects_lost_pairs_duplicates_and_invalid_outcomes(self):
        bad_inputs = [self.b[:-1], self.b + self.b[:1],
                      [row("gain", 1, 0), *self.b[1:]],
                      [row("gain", 2, 1), *self.b[1:]],
                      [row("other", 0, 1), *self.b[1:]]]
        for rows in bad_inputs:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_pair(self.a, rows)
        with self.assertRaises(ValueError):
            summarize_pair([], [])


if __name__ == "__main__":
    unittest.main()
