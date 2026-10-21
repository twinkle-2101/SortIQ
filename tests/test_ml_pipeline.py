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


if __name__ == "__main__":
    unittest.main()
