"""Integration tests for end-to-end SortIQ workflow (Tests I1, I2)."""

import tempfile
import unittest
from pathlib import Path

from src.category_mapping import get_destination_folder
from src.file_scanner import scan_directory
from src.ml_pipeline import load_model, predict_file_names
from src.organizer import execute_move, plan_organization


class IntegrationWorkflowTests(unittest.TestCase):
    def test_i1_missing_model_file_handling(self):
        """Test I1: Missing model file raises clean FileNotFoundError with actionable instructions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "does_not_exist.joblib"
            with self.assertRaises(FileNotFoundError) as ctx:
                load_model(missing_path)
            self.assertIn("Run python -m src.run_experiments", str(ctx.exception))

    def test_i2_full_workflow_scan_predict_review_confirm_move(self):
        """Test I2: Full workflow: Scan -> Predict -> Review & Override -> Dry-run -> Confirm Move -> Verify."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloads_dir = Path(temp_dir).resolve() / "Downloads"
            downloads_dir.mkdir()

            # Create realistic test files
            test_files = {
                "cs101_algorithms_lecture_notes.pdf": "academic content",
                "alex_morgan_software_engineer_resume.pdf": "resume content",
                "amazon_order_invoice_october.pdf": "receipt content",
                "quarterly_financial_review_slides.pptx": "presentation content",
                "vacation_mountain_sunset.jpg": "image bytes",
                "setup_installer_v2.exe": "installer binary",
            }

            for name, content in test_files.items():
                (downloads_dir / name).write_text(content, encoding="utf-8")

            # 1. SCAN
            scanned = scan_directory(downloads_dir)
            self.assertEqual(len(scanned), len(test_files))
            file_names = [f.file_name for f in scanned]

            # 2. PREDICT
            model = load_model()  # loads models/best_model.joblib
            predictions = predict_file_names(model, file_names)
            self.assertEqual(len(predictions), len(test_files))

            pred_map = {p["file_name"]: p["predicted_category"] for p in predictions}
            self.assertEqual(pred_map["cs101_algorithms_lecture_notes.pdf"], "academic")
            self.assertEqual(pred_map["alex_morgan_software_engineer_resume.pdf"], "resume")
            self.assertEqual(pred_map["amazon_order_invoice_october.pdf"], "receipt")
            self.assertEqual(pred_map["quarterly_financial_review_slides.pptx"], "presentation")
            self.assertEqual(pred_map["vacation_mountain_sunset.jpg"], "image")

            # 3. REVIEW & OVERRIDE
            # Suppose user overrides 'setup_installer_v2.exe' to 'other' explicitly
            reviewed_items = []
            for file_info in scanned:
                predicted_cat = pred_map[file_info.file_name]
                final_cat = "other" if file_info.file_name == "setup_installer_v2.exe" else predicted_cat
                reviewed_items.append(
                    {
                        "file_name": file_info.file_name,
                        "source_path": file_info.full_path,
                        "category": final_cat,
                    }
                )

            # 4. DRY-RUN PREVIEW (Confirmed=False)
            plan = plan_organization(reviewed_items, destination_root=downloads_dir)
            self.assertEqual(len(plan), len(test_files))
            self.assertTrue(all(item.status == "ready" for item in plan))

            # Verify files are still in original downloads directory
            for name in test_files:
                self.assertTrue((downloads_dir / name).is_file())

            # 5. CONFIRM MOVE
            results = execute_move(plan, confirmed=True)
            self.assertEqual(results["moved_count"], len(test_files))
            self.assertEqual(results["failed_count"], 0)
            self.assertEqual(results["skipped_count"], 0)

            # 6. VERIFY FINAL DIRECTORY STRUCTURE
            for item in reviewed_items:
                name = item["file_name"]
                cat = item["category"]
                expected_folder = get_destination_folder(cat)
                dest_file = downloads_dir / expected_folder / name
                self.assertTrue(dest_file.is_file(), f"Expected {dest_file} to exist")
                self.assertEqual(dest_file.read_text(encoding="utf-8"), test_files[name])
                # Source file no longer in top-level directory
                self.assertFalse((downloads_dir / name).exists())


if __name__ == "__main__":
    unittest.main()
