"""Unit tests for file scanner and path validation (Tests F1, F2)."""

import tempfile
import unittest
from pathlib import Path

from src.file_scanner import (
    format_file_size,
    scan_directory,
    validate_directory_path,
)


class FileScannerTests(unittest.TestCase):
    def test_format_file_size(self):
        self.assertEqual(format_file_size(0), "0 B")
        self.assertEqual(format_file_size(500), "500 B")
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1048576), "1.0 MB")
        self.assertEqual(format_file_size(1073741824), "1.0 GB")

    def test_f1_scan_valid_folder_read_only(self):
        """Test F1: Scan valid folder: files listed; no changes made."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            file1 = dir_path / "lecture_notes.pdf"
            file2 = dir_path / "cat_photo.jpg"
            file1.write_text("dummy academic notes", encoding="utf-8")
            file2.write_bytes(b"\xff\xd8\xff\xe0" + b"dummyimage" * 50)

            # Record initial state
            mtime1 = file1.stat().st_mtime
            mtime2 = file2.stat().st_mtime
            initial_contents = set(dir_path.iterdir())

            # Perform scan
            scanned = scan_directory(dir_path)

            self.assertEqual(len(scanned), 2)
            names = [f.file_name for f in scanned]
            self.assertEqual(names, ["cat_photo.jpg", "lecture_notes.pdf"])

            pdf_item = next(f for f in scanned if f.file_name == "lecture_notes.pdf")
            self.assertEqual(pdf_item.extension, ".pdf")
            self.assertTrue(pdf_item.size_bytes > 0)
            self.assertTrue(pdf_item.size_formatted.endswith("B"))

            # Integrity check: files must remain completely unchanged
            self.assertEqual(set(dir_path.iterdir()), initial_contents)
            self.assertEqual(file1.stat().st_mtime, mtime1)
            self.assertEqual(file2.stat().st_mtime, mtime2)

    def test_f2_invalid_folder_clear_error(self):
        """Test F2: Invalid folder provides clear error message."""
        # Non-existent path
        non_existent = Path("/path/that/definitely/does/not/exist_12345")
        is_valid, msg = validate_directory_path(non_existent)
        self.assertFalse(is_valid)
        self.assertIn("does not exist", msg)
        with self.assertRaises(FileNotFoundError):
            scan_directory(non_existent)

        # Empty string
        is_valid, msg = validate_directory_path("")
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", msg)

        # File path instead of directory
        with tempfile.NamedTemporaryFile() as temp_file:
            is_valid, msg = validate_directory_path(temp_file.name)
            self.assertFalse(is_valid)
            self.assertIn("Path is a file, not a directory", msg)
            with self.assertRaises(NotADirectoryError):
                scan_directory(temp_file.name)

    def test_non_recursive_by_default(self):
        """Ensure subdirectories are not scanned by default (safety requirement)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            (dir_path / "top_level.txt").write_text("hello", encoding="utf-8")
            sub_dir = dir_path / "subfolder"
            sub_dir.mkdir()
            (sub_dir / "nested_file.txt").write_text("world", encoding="utf-8")

            # Non-recursive scan
            scanned = scan_directory(dir_path, recursive=False)
            self.assertEqual(len(scanned), 1)
            self.assertEqual(scanned[0].file_name, "top_level.txt")

            # Recursive scan
            scanned_rec = scan_directory(dir_path, recursive=True)
            self.assertEqual(len(scanned_rec), 2)
            rec_names = {f.file_name for f in scanned_rec}
            self.assertEqual(rec_names, {"top_level.txt", "nested_file.txt"})

    def test_hidden_files_handling(self):
        """Verify hidden files (starting with dot) are excluded by default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            (dir_path / "visible.pdf").write_text("visible", encoding="utf-8")
            (dir_path / ".DS_Store").write_bytes(b"hidden")

            scanned = scan_directory(dir_path, include_hidden=False)
            self.assertEqual(len(scanned), 1)
            self.assertEqual(scanned[0].file_name, "visible.pdf")

            scanned_with_hidden = scan_directory(dir_path, include_hidden=True)
            self.assertEqual(len(scanned_with_hidden), 2)


if __name__ == "__main__":
    unittest.main()
