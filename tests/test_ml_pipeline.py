import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ml_pipeline import (
    _group_key,
    grouped_train_test_split,
    load_model,
    predict_file_names,
    train_and_evaluate,
)
from src.feedback import load_feedback, save_correction


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "file_dataset.csv"


class MLPipelineTests(unittest.TestCase):
    def test_grouped_split_keeps_filename_variants_together_and_balanced(self):
        data = pd.read_csv(DATASET_PATH)
        train_indices, test_indices = grouped_train_test_split(data)
        train = data.iloc[train_indices]
        test = data.iloc[test_indices]
        train_families = {_group_key(name) for name in train["file_name"]}
        test_families = {_group_key(name) for name in test["file_name"]}
        self.assertFalse(train_families & test_families)
        self.assertEqual(len(train_indices), 960)
        self.assertEqual(len(test_indices), 240)
        self.assertEqual(
            test["category"].value_counts().to_dict(),
            {category: 40 for category in data["category"].unique()},
        )

    def test_training_saves_metrics_model_and_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = train_and_evaluate(
                model_path=output / "best_model.joblib",
                metrics_path=output / "metrics.json",
                confusion_matrix_path=output / "confusion_matrix.png",
            )

            self.assertEqual(
                set(metrics["models"]),
                {
                    "tfidf_multinomial_nb",
                    "tfidf_logistic_regression",
                    "tfidf_svd_logistic_regression",
                },
            )
            self.assertIn(metrics["selected_model"], metrics["models"])
            self.assertTrue((output / "best_model.joblib").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "confusion_matrix.png").is_file())

            model = load_model(output / "best_model.joblib")
            predictions = predict_file_names(
                model, ["machine_learning_course_notes.pdf", "orange_cat_photo.jpg"]
            )
            self.assertEqual(len(predictions), 2)
            self.assertTrue(
                all("confidence" in prediction for prediction in predictions)
            )
            self.assertEqual(
                [prediction["predicted_category"] for prediction in predictions],
                ["academic", "image"],
            )

    def test_prediction_rejects_blank_names(self):
        with self.assertRaises(ValueError):
            predict_file_names(object(), [""])

    def test_load_model_reports_missing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_model = Path(directory) / "missing_model.joblib"
            with self.assertRaisesRegex(FileNotFoundError, "Run python -m"):
                load_model(missing_model)

    def test_prediction_returns_empty_list_for_empty_input(self):
        self.assertEqual(predict_file_names(object(), []), [])

    def test_feedback_saves_and_replaces_a_correction_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback_path = Path(directory) / "user_feedback.csv"
            self.assertEqual(
                save_correction("Review_Me.PDF", "academic", feedback_path), 1
            )
            self.assertEqual(
                save_correction("review_me.pdf", "receipt", feedback_path), 1
            )

            feedback = load_feedback(feedback_path)
            self.assertEqual(len(feedback), 1)
            self.assertEqual(feedback.iloc[0]["file_name"], "review_me.pdf")
            self.assertEqual(feedback.iloc[0]["extension"], ".pdf")
            self.assertEqual(feedback.iloc[0]["category"], "receipt")

    def test_feedback_rejects_paths_and_unapproved_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback_path = Path(directory) / "user_feedback.csv"
            with self.assertRaisesRegex(ValueError, "filename, not a path"):
                save_correction("C:\\private\\receipt.pdf", "receipt", feedback_path)
            with self.assertRaisesRegex(ValueError, "one of"):
                save_correction("receipt.pdf", "personal", feedback_path)
            self.assertFalse(feedback_path.exists())

    def test_training_explicitly_applies_local_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            feedback_path = output / "user_feedback.csv"
            save_correction(
                "feedbackonlykeyword_grocery_receipt.pdf",
                "receipt",
                feedback_path,
            )
            metrics = train_and_evaluate(
                model_path=output / "best_model.joblib",
                metrics_path=output / "metrics.json",
                confusion_matrix_path=output / "confusion_matrix.png",
                feedback_path=feedback_path,
            )

            self.assertEqual(metrics["feedback"]["correction_count"], 1)
            self.assertEqual(metrics["feedback"]["final_training_rows"], 1201)
            model = load_model(output / "best_model.joblib")
            self.assertIn("feedbackonlykeyword", model.named_steps["tfidf"].vocabulary_)


if __name__ == "__main__":
    unittest.main()
