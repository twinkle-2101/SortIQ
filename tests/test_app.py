"""Unit tests for Streamlit app helpers and fallback prediction logic."""

import tempfile
import unittest
from pathlib import Path

from app import create_demo_folder, load_model_metrics, rule_based_fallback_predict
from src.file_scanner import scan_directory


class AppHelperTests(unittest.TestCase):
    def test_rule_based_fallback_predict(self):
        names = [
            "machine_learning_lecture_notes.pdf",
            "jane_doe_resume_2025.pdf",
            "starbucks_coffee_receipt.pdf",
            "quarterly_board_meeting_deck.pptx",
            "family_vacation_photo.jpg",
            "random_setup_file.pkg",
        ]
        results = rule_based_fallback_predict(names)
        self.assertEqual(len(results), len(names))

        cat_map = {r["file_name"]: r["predicted_category"] for r in results}
        self.assertEqual(cat_map["machine_learning_lecture_notes.pdf"], "academic")
        self.assertEqual(cat_map["jane_doe_resume_2025.pdf"], "resume")
        self.assertEqual(cat_map["starbucks_coffee_receipt.pdf"], "receipt")
        self.assertEqual(cat_map["quarterly_board_meeting_deck.pptx"], "presentation")
        self.assertEqual(cat_map["family_vacation_photo.jpg"], "image")
        self.assertEqual(cat_map["random_setup_file.pkg"], "other")

    def test_create_demo_folder(self):
        demo_path = create_demo_folder()
        self.assertTrue(demo_path.is_dir())
        scanned = scan_directory(demo_path)
        self.assertGreaterEqual(len(scanned), 10)

    def test_load_model_metrics(self):
        metrics = load_model_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn("models", metrics)
        self.assertIn("selected_model", metrics)


    def test_ensure_sandbox_files(self):
        from app import ensure_sandbox_files, get_sandbox_path
        sandbox_path = get_sandbox_path()
        self.assertTrue(sandbox_path.is_dir())
        count = ensure_sandbox_files()
        scanned = scan_directory(sandbox_path)
        self.assertGreaterEqual(len(scanned), 20)


if __name__ == "__main__":
    unittest.main()
