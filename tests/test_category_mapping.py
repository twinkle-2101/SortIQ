"""Unit tests for category-to-folder mapping and destination resolution."""

import unittest
from pathlib import Path

from src.category_mapping import (
    DEFAULT_CATEGORY_MAPPING,
    get_destination_folder,
    get_destination_path,
)


class CategoryMappingTests(unittest.TestCase):
    def test_default_categories_map_to_standard_folders(self):
        expected = {
            "academic": "Academic",
            "resume": "Resumes",
            "receipt": "Receipts",
            "presentation": "Presentations",
            "image": "Images",
            "other": "Other",
        }
        for cat, folder in expected.items():
            self.assertEqual(get_destination_folder(cat), folder)

    def test_case_insensitivity_and_unknown_category_fallback(self):
        self.assertEqual(get_destination_folder("ACADEMIC"), "Academic")
        self.assertEqual(get_destination_folder("Resume"), "Resumes")
        self.assertEqual(get_destination_folder("unknown_label"), "Other")

    def test_custom_mapping_override(self):
        custom = {"academic": "University", "other": "Misc"}
        self.assertEqual(get_destination_folder("academic", custom), "University")
        self.assertEqual(get_destination_folder("unknown", custom), "Misc")

    def test_get_destination_path_resolution(self):
        base = Path("/downloads")
        path = get_destination_path(base, "resume", "my_cv.pdf")
        self.assertEqual(path, base / "Resumes" / "my_cv.pdf")


if __name__ == "__main__":
    unittest.main()
