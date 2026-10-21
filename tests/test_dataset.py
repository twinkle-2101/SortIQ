import csv
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "file_dataset.csv"
EXPECTED_CATEGORIES = {
    "academic",
    "resume",
    "receipt",
    "presentation",
    "image",
    "other",
}


class DatasetTests(unittest.TestCase):
    def test_dataset_has_contract_columns_and_balanced_categories(self):
        with DATASET_PATH.open(newline="", encoding="utf-8") as dataset_file:
            reader = csv.DictReader(dataset_file)
            self.assertEqual(
                reader.fieldnames, ["file_name", "extension", "category"]
            )
            rows = list(reader)

        self.assertEqual(len(rows), 1200)
        self.assertEqual(
            Counter(row["category"] for row in rows),
            {category: 200 for category in EXPECTED_CATEGORIES},
        )

    def test_dataset_has_valid_unique_filenames_and_extensions(self):
        with DATASET_PATH.open(newline="", encoding="utf-8") as dataset_file:
            rows = list(csv.DictReader(dataset_file))

        filenames = [row["file_name"] for row in rows]
        self.assertEqual(len(filenames), len(set(filenames)))
        for row in rows:
            self.assertTrue(row["file_name"])
            self.assertIn(row["category"], EXPECTED_CATEGORIES)
            self.assertEqual(row["extension"], Path(row["file_name"]).suffix)
            self.assertEqual(row["extension"], row["extension"].lower())


if __name__ == "__main__":
    unittest.main()
