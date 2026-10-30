"""Unit tests for Streamlit app helpers and fallback prediction logic."""

import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import (
    create_demo_folder,
    create_fresh_demo_folder,
    load_model,
    load_model_metrics,
    rule_based_fallback_predict,
)
from src.file_scanner import scan_directory
from src.ml_pipeline import predict_file_names
from src.organizer import COLLISION_SKIP, plan_organization


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
        self.assertEqual(threshold, 0.75)
        ambiguous_files = {
            file_name
            for file_name, prediction in predictions.items()
            if prediction.get("confidence", 0.5) < threshold
        }
        self.assertEqual(len(predictions), 100)
        self.assertGreaterEqual(len(ambiguous_files), 10)
        self.assertLessEqual(len(ambiguous_files), 18)
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

    def test_demo_folder_button_loads_and_reports_loaded(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            Path(temporary_directory, "meeting_agenda.docx").write_text(
                "Project meeting agenda", encoding="utf-8"
            )
            app = AppTest.from_file(str(app_path), default_timeout=30).run()
            folder_input = next(
                item for item in app.text_input if item.label == "Folder path"
            )
            folder_input.set_value(temporary_directory)
            next(button for button in app.button if button.label == "Scan folder").click().run()

            self.assertEqual(
                app.session_state["current_scan_path"],
                str(Path(temporary_directory).resolve()),
            )
            next(button for button in app.button if button.label == "Use demo folder").click().run()

            self.assertEqual(
                app.session_state["current_scan_path"],
                app.session_state["demo_folder_path"],
            )
            self.assertGreaterEqual(len(app.session_state["scanned_files"]), 100)
            self.assertTrue(
                any(button.label == "Demo folder loaded ✓" for button in app.button)
            )

    def test_name_conflict_filter_shows_only_colliding_demo_files(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        conflict_filter = next(
            option
            for option in next(
                item for item in app.selectbox if item.label == "Show"
            ).options
            if option.startswith("Name conflicts")
        )
        self.assertEqual(conflict_filter, "Name conflicts (6)")

        next(item for item in app.selectbox if item.label == "Show").select(
            conflict_filter
        ).run()

        visible_name_inputs = {
            item.key.removeprefix("destination_name_")
            for item in app.text_input
            if item.key.startswith("destination_name_")
        }
        self.assertEqual(
            visible_name_inputs,
            {
                "computer_networks_unit4_lecture_notes_dbms_unit_3_lecture_notes.pdf",
                "alex_morgan_software_engineer_resume_software_engineer_resume.pdf",
                "amazon_order_tax_invoice_10928_grocery_market_receipt.pdf",
                "quarterly_sales_enablement_training_deck_database_normalization_slides.pptx",
                "family_hiking_trail_photo_2026_mountain_sunrise_landscape_photo.jpg",
                "birthday_party_guest_checklist_final.txt",
            },
        )

    def test_each_fresh_demo_run_has_a_clean_and_consistent_preview(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            demo_parent = Path(temporary_directory)
            first_run = create_fresh_demo_folder(demo_parent)
            second_run = create_fresh_demo_folder(demo_parent)

            first_files = scan_directory(first_run)
            second_files = scan_directory(second_run)
            self.assertEqual(
                {file_info.file_name for file_info in first_files},
                {file_info.file_name for file_info in second_files},
            )
            self.assertEqual(len(first_files), 100)
            model = load_model()
            predictions = predict_file_names(
                model,
                [file_info.file_name for file_info in first_files],
            )
            category_markers = {
                "_dbms_unit_3_lecture_notes.": "academic",
                "_software_engineer_resume.": "resume",
                "_grocery_market_receipt.": "receipt",
                "_database_normalization_slides.": "presentation",
                "_mountain_sunrise_landscape_photo.": "image",
            }
            for prediction in predictions:
                expected_category = next(
                    (
                        category
                        for marker, category in category_markers.items()
                        if marker in prediction["file_name"]
                    ),
                    "other",
                )
                self.assertEqual(
                    prediction["predicted_category"],
                    expected_category,
                    prediction["file_name"],
                )
            predicted_categories = {
                prediction["file_name"]: prediction["predicted_category"]
                for prediction in predictions
            }
            for run_dir, files in (
                (first_run, first_files),
                (second_run, second_files),
            ):
                plan = plan_organization(
                    [
                        {
                            "file_name": file_info.file_name,
                            "source_path": file_info.full_path,
                            "category": predicted_categories[file_info.file_name],
                        }
                        for file_info in files
                    ],
                    destination_root=run_dir,
                )
                self.assertEqual(len(plan), 100)
                collision_items = [
                    item for item in plan if item.status == "rename"
                ]
                self.assertEqual(
                    {item.file_name for item in collision_items},
                    {
                        "computer_networks_unit4_lecture_notes_dbms_unit_3_lecture_notes.pdf",
                        "alex_morgan_software_engineer_resume_software_engineer_resume.pdf",
                        "amazon_order_tax_invoice_10928_grocery_market_receipt.pdf",
                        "quarterly_sales_enablement_training_deck_database_normalization_slides.pptx",
                        "family_hiking_trail_photo_2026_mountain_sunrise_landscape_photo.jpg",
                        "birthday_party_guest_checklist_final.txt",
                    },
                )
                self.assertEqual(
                    {item.destination_dir.name for item in collision_items},
                    {
                        "Academic",
                        "Resumes",
                        "Receipts",
                        "Presentations",
                        "Images",
                        "Other",
                    },
                )
                self.assertTrue(
                    all(item.status == "ready" for item in plan if item not in collision_items)
                )
                skip_plan = plan_organization(
                    [
                        {
                            "file_name": file_info.file_name,
                            "source_path": file_info.full_path,
                            "category": predicted_categories[file_info.file_name],
                        }
                        for file_info in files
                    ],
                    destination_root=run_dir,
                    collision_policy=COLLISION_SKIP,
                )
                skipped_items = [item for item in skip_plan if item.status == "skip"]
                self.assertEqual(
                    {item.file_name for item in skipped_items},
                    {item.file_name for item in collision_items},
                )

    def test_review_only_allows_renaming_when_destination_collides(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        filename = "team_meeting_agenda.docx"
        with tempfile.TemporaryDirectory() as temporary_directory:
            Path(temporary_directory, filename).write_text(
                "Agenda for the project team", encoding="utf-8"
            )
            app = AppTest.from_file(str(app_path), default_timeout=30).run()
            next(
                item for item in app.text_input if item.label == "Folder path"
            ).set_value(temporary_directory)
            next(button for button in app.button if button.label == "Scan folder").click().run()
            name_input_key = f"destination_name_{filename}"
            self.assertFalse(any(item.key == name_input_key for item in app.text_input))

            prediction = app.session_state["predictions"][filename]
            destination_folder = {
                "academic": "Academic",
                "resume": "Resumes",
                "receipt": "Receipts",
                "presentation": "Presentations",
                "image": "Images",
                "other": "Other",
            }[prediction["predicted_category"]]
            destination_directory = Path(temporary_directory, destination_folder)
            destination_directory.mkdir()
            (destination_directory / filename).write_text(
                "Existing destination file", encoding="utf-8"
            )
            next(button for button in app.button if button.label == "Scan folder").click().run()

            name_input = next(
                item
                for item in app.text_input
                if item.key == name_input_key
            )
            name_input.set_value("project_team_agenda.docx").run()

            ledger = next(
                table.value
                for table in app.dataframe
                if "Destination Path" in table.value.columns
            )
            self.assertIn(
                str(Path("Other") / "project_team_agenda.docx"),
                ledger["Destination Path"].tolist(),
            )
            self.assertTrue(Path(temporary_directory, filename).is_file())

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
        self.assertGreaterEqual(len(scanned), 100)


if __name__ == "__main__":
    unittest.main()
