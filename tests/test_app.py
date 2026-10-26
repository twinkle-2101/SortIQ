"""Unit tests for Streamlit app helpers and fallback prediction logic."""

import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import create_demo_folder, load_model_metrics, rule_based_fallback_predict
from src.file_scanner import scan_directory


class AppHelperTests(unittest.TestCase):
    def test_category_review_controls_are_only_shown_for_ambiguous_files(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertFalse(app.exception)

        predictions = app.session_state["predictions"]
        threshold = next(
            slider.value
            for slider in app.slider
            if slider.label == "Ambiguity Review Threshold"
        )
        ambiguous_files = {
            file_name
            for file_name, prediction in predictions.items()
            if prediction.get("confidence", 0.5) < threshold
        }
        review_controls = {
            selectbox.label.removeprefix("Review category for ")
            for selectbox in app.selectbox
            if selectbox.label.startswith("Review category for ")
        }

        self.assertEqual(review_controls, ambiguous_files)

    def test_confirmation_is_reset_when_review_changes_the_move_plan(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        confirmation = next(
            checkbox
            for checkbox in app.checkbox
            if checkbox.label.startswith("I have verified the dry-run ledger")
        )
        confirmation.check().run()
        execute_button = next(
            button
            for button in app.button
            if button.label == "Execute File Organization"
        )
        self.assertFalse(execute_button.disabled)

        category_control = next(
            selectbox
            for selectbox in app.selectbox
            if selectbox.label.startswith("Review category for ")
        )
        alternate_category = next(
            option for option in category_control.options if option != category_control.value
        )
        category_control.select(alternate_category).run()

        confirmation = next(
            checkbox
            for checkbox in app.checkbox
            if checkbox.label.startswith("I have verified the dry-run ledger")
        )
        execute_button = next(
            button
            for button in app.button
            if button.label == "Execute File Organization"
        )
        self.assertFalse(confirmation.value)
        self.assertTrue(execute_button.disabled)

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
