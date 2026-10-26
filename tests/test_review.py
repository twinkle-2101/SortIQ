import unittest

from src.review import category_for_file, get_pending_corrections


class ReviewTests(unittest.TestCase):
    def test_confident_file_uses_prediction_and_needs_no_category_override(self):
        category = category_for_file(
            "report.pdf",
            "academic",
            confidence=0.91,
            confidence_threshold=0.65,
            user_overrides={"report.pdf": "receipt"},
        )

        self.assertEqual(category, "academic")

    def test_pending_corrections_include_only_ambiguous_changed_labels(self):
        file_names = ["uncertain.pdf", "confident.pdf", "unchanged.pdf"]
        predictions = {
            "uncertain.pdf": {
                "predicted_category": "academic",
                "confidence": 0.52,
            },
            "confident.pdf": {
                "predicted_category": "receipt",
                "confidence": 0.9,
            },
            "unchanged.pdf": {
                "predicted_category": "image",
                "confidence": 0.4,
            },
        }

        pending = get_pending_corrections(
            file_names,
            predictions,
            user_overrides={
                "uncertain.pdf": "receipt",
                "confident.pdf": "image",
                "unchanged.pdf": "image",
            },
            submitted_feedback={},
            confidence_threshold=0.65,
        )

        self.assertEqual(pending, {"uncertain.pdf": "receipt"})

    def test_previously_submitted_same_correction_is_not_pending(self):
        pending = get_pending_corrections(
            ["uncertain.pdf"],
            {
                "uncertain.pdf": {
                    "predicted_category": "academic",
                    "confidence": 0.5,
                }
            },
            user_overrides={"uncertain.pdf": "receipt"},
            submitted_feedback={"uncertain.pdf": "receipt"},
            confidence_threshold=0.65,
        )

        self.assertEqual(pending, {})


if __name__ == "__main__":
    unittest.main()
