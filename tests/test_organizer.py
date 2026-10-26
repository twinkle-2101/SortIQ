"""Unit tests for organizer: dry-run planning, collision handling, and safe moves (Tests F3-F8)."""

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.organizer import (
    COLLISION_RENAME,
    COLLISION_SKIP,
    execute_move,
    organization_plan_signature,
    plan_organization,
    resolve_collision_path,
)


class OrganizerTests(unittest.TestCase):
    def test_plan_rejects_path_traversal_filenames_and_mappings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source_file = root / "notes.pdf"
            source_file.write_text("notes", encoding="utf-8")
            traversal_filename = plan_organization(
                [
                    {
                        "file_name": "..\\outside.pdf",
                        "source_path": source_file,
                        "category": "academic",
                    }
                ],
                destination_root=root,
            )
            traversal_mapping = plan_organization(
                [
                    {
                        "file_name": source_file.name,
                        "source_path": source_file,
                        "category": "academic",
                    }
                ],
                destination_root=root,
                custom_mapping={"academic": "..\\outside"},
            )

            self.assertEqual(traversal_filename[0].status, "error")
            self.assertIn("Invalid filename", traversal_filename[0].error_message)
            self.assertEqual(traversal_mapping[0].status, "error")
            self.assertIn("inside the selected root", traversal_mapping[0].error_message)
            self.assertTrue(source_file.is_file())

    def test_plan_signature_changes_when_category_or_destination_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source_file = root / "notes.pdf"
            source_file.write_text("notes", encoding="utf-8")
            academic_plan = plan_organization(
                [
                    {
                        "file_name": source_file.name,
                        "source_path": source_file,
                        "category": "academic",
                    }
                ],
                destination_root=root,
            )
            receipt_plan = plan_organization(
                [
                    {
                        "file_name": source_file.name,
                        "source_path": source_file,
                        "category": "receipt",
                    }
                ],
                destination_root=root,
            )

            academic_signature = organization_plan_signature(
                academic_plan, root, COLLISION_RENAME
            )
            receipt_signature = organization_plan_signature(
                receipt_plan, root, COLLISION_RENAME
            )
            other_policy_signature = organization_plan_signature(
                academic_plan, root, COLLISION_SKIP
            )
            original_signature = organization_plan_signature(
                academic_plan, root, COLLISION_RENAME
            )
            source_stat = source_file.stat()
            os.utime(
                source_file,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
            )
            changed_file_signature = organization_plan_signature(
                academic_plan, root, COLLISION_RENAME
            )

            self.assertNotEqual(academic_signature, receipt_signature)
            self.assertNotEqual(academic_signature, other_policy_signature)
            self.assertNotEqual(original_signature, changed_file_signature)

    def test_f3_preview_recommendations_changes_nothing(self):
        """Test F3: Preview recommendations: files remain completely unchanged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir).resolve()
            file1 = dir_path / "resume_john_doe.pdf"
            file1.write_text("resume content", encoding="utf-8")
            mtime = file1.stat().st_mtime

            items = [
                {
                    "file_name": "resume_john_doe.pdf",
                    "source_path": file1,
                    "category": "resume",
                }
            ]

            plan = plan_organization(items, destination_root=dir_path)

            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0].status, "ready")
            self.assertEqual(plan[0].destination_dir, dir_path / "Resumes")
            self.assertEqual(
                plan[0].resolved_destination_path,
                dir_path / "Resumes" / "resume_john_doe.pdf",
            )

            # Dry-run invariant: original file is still in place, destination not created yet
            self.assertTrue(file1.is_file())
            self.assertEqual(file1.stat().st_mtime, mtime)
            self.assertFalse((dir_path / "Resumes").exists())

    def test_f4_manual_category_override(self):
        """Test F4: Manual category override updates destination recommendation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir).resolve()
            file1 = dir_path / "doc.pdf"
            file1.write_text("content", encoding="utf-8")

            # Originally predicted as receipt, user overrides to academic
            items = [
                {
                    "file_name": "doc.pdf",
                    "source_path": file1,
                    "category": "academic",  # overridden
                }
            ]

            plan = plan_organization(items, destination_root=dir_path)
            self.assertEqual(plan[0].destination_dir, dir_path / "Academic")
            self.assertEqual(
                plan[0].resolved_destination_path,
                dir_path / "Academic" / "doc.pdf",
            )

    def test_f5_cancel_or_unconfirmed_move_moves_nothing(self):
        """Test F5: Cancel/unconfirmed: execute_move raises ValueError if confirmed is False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir).resolve()
            file1 = dir_path / "invoice.pdf"
            file1.write_text("invoice", encoding="utf-8")

            items = [
                {
                    "file_name": "invoice.pdf",
                    "source_path": file1,
                    "category": "receipt",
                }
            ]
            plan = plan_organization(items, destination_root=dir_path)

            # Attempt execute_move without confirmed=True
            with self.assertRaises(ValueError):
                execute_move(plan, confirmed=False)

            # Verify file did not move
            self.assertTrue(file1.is_file())
            self.assertFalse((dir_path / "Receipts" / "invoice.pdf").exists())

    def test_f6_confirm_test_move_moves_only_selected_files(self):
        """Test F6: Confirm test move: only selected files move."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir).resolve()
            file1 = dir_path / "move_me.pdf"
            file2 = dir_path / "keep_me.pdf"
            file1.write_text("move this", encoding="utf-8")
            file2.write_text("keep this", encoding="utf-8")

            # Only file1 is selected for moving
            items = [
                {
                    "file_name": "move_me.pdf",
                    "source_path": file1,
                    "category": "academic",
                }
            ]
            plan = plan_organization(items, destination_root=dir_path)
            results = execute_move(plan, confirmed=True)

            self.assertEqual(results["moved_count"], 1)
            self.assertEqual(results["failed_count"], 0)
            self.assertFalse(file1.exists())
            self.assertTrue((dir_path / "Academic" / "move_me.pdf").is_file())

            # File 2 remains untouched in source directory
            self.assertTrue(file2.is_file())

    def test_f7_destination_name_collision_handling(self):
        """Test F7: Destination collision: existing files are NEVER overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir).resolve()
            academic_dir = dir_path / "Academic"
            academic_dir.mkdir()

            # Pre-existing file in destination
            existing_dest_file = academic_dir / "notes.pdf"
            existing_dest_file.write_text("original destination file", encoding="utf-8")

            # Source file with identical name
            source_file = dir_path / "notes.pdf"
            source_file.write_text("new incoming file", encoding="utf-8")

            # Case A: Rename policy (default)
            plan_rename = plan_organization(
                [
                    {
                        "file_name": "notes.pdf",
                        "source_path": source_file,
                        "category": "academic",
                    }
                ],
                destination_root=dir_path,
                collision_policy=COLLISION_RENAME,
            )
            self.assertTrue(plan_rename[0].collision_detected)
            self.assertEqual(plan_rename[0].status, "rename")
            self.assertEqual(
                plan_rename[0].resolved_destination_path,
                academic_dir / "notes (1).pdf",
            )

            # Execute rename move
            results = execute_move(plan_rename, confirmed=True)
            self.assertEqual(results["moved_count"], 1)
            # Original file remains untouched!
            self.assertEqual(
                existing_dest_file.read_text(encoding="utf-8"),
                "original destination file",
            )
            # New file placed with unique name!
            self.assertTrue((academic_dir / "notes (1).pdf").is_file())
            self.assertEqual(
                (academic_dir / "notes (1).pdf").read_text(encoding="utf-8"),
                "new incoming file",
            )

            # Case B: Skip policy
            source_file2 = dir_path / "notes.pdf"
            source_file2.write_text("another file", encoding="utf-8")
            plan_skip = plan_organization(
                [
                    {
                        "file_name": "notes.pdf",
                        "source_path": source_file2,
                        "category": "academic",
                    }
                ],
                destination_root=dir_path,
                collision_policy=COLLISION_SKIP,
            )
            self.assertEqual(plan_skip[0].status, "skip")
            results_skip = execute_move(plan_skip, confirmed=True)
            self.assertEqual(results_skip["moved_count"], 0)
            self.assertEqual(results_skip["skipped_count"], 1)
            self.assertTrue(source_file2.is_file())

    def test_move_does_not_overwrite_destination_created_after_preview(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source_file = root / "notes.pdf"
            destination_file = root / "Academic" / "notes.pdf"
            source_file.write_text("source", encoding="utf-8")
            plan = plan_organization(
                [
                    {
                        "file_name": source_file.name,
                        "source_path": source_file,
                        "category": "academic",
                    }
                ],
                destination_root=root,
            )
            real_link = os.link

            def simulate_concurrent_destination(source, destination):
                Path(destination).write_text("existing", encoding="utf-8")
                return real_link(source, destination)

            with patch(
                "src.organizer.os.link",
                side_effect=simulate_concurrent_destination,
            ):
                result = execute_move(plan, confirmed=True)

            self.assertEqual(result["moved_count"], 0)
            self.assertEqual(result["failed_count"], 1)
            self.assertEqual(source_file.read_text(encoding="utf-8"), "source")
            self.assertEqual(destination_file.read_text(encoding="utf-8"), "existing")

    def test_cross_device_move_uses_exclusive_copy_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source_file = root / "notes.pdf"
            source_file.write_text("source", encoding="utf-8")
            plan = plan_organization(
                [
                    {
                        "file_name": source_file.name,
                        "source_path": source_file,
                        "category": "academic",
                    }
                ],
                destination_root=root,
            )
            with patch(
                "src.organizer.os.link",
                side_effect=OSError(errno.EXDEV, "cross-device"),
            ):
                result = execute_move(plan, confirmed=True)

            destination_file = root / "Academic" / "notes.pdf"
            self.assertEqual(result["moved_count"], 1)
            self.assertFalse(source_file.exists())
            self.assertEqual(destination_file.read_text(encoding="utf-8"), "source")

    def test_f8_permission_and_move_error_handling(self):
        """Test F8: Permission or move errors report clear per-file failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir).resolve()
            file1 = dir_path / "protected.pdf"
            file1.write_text("data", encoding="utf-8")

            items = [
                {
                    "file_name": "protected.pdf",
                    "source_path": file1,
                    "category": "academic",
                }
            ]
            plan = plan_organization(items, destination_root=dir_path)

            # Mock the no-overwrite move primitive to simulate a PermissionError
            with patch(
                "src.organizer.os.link",
                side_effect=PermissionError("Access denied"),
            ):
                results = execute_move(plan, confirmed=True)

            self.assertEqual(results["failed_count"], 1)
            self.assertEqual(results["moved_count"], 0)
            self.assertEqual(len(results["errors"]), 1)
            self.assertEqual(results["errors"][0]["file_name"], "protected.pdf")
            self.assertIn("Access denied", results["errors"][0]["error"])

    def test_missing_source_file_error(self):
        """Ensure plan handles missing source file gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir).resolve()
            non_existent = dir_path / "ghost.pdf"
            items = [
                {
                    "file_name": "ghost.pdf",
                    "source_path": non_existent,
                    "category": "other",
                }
            ]
            plan = plan_organization(items, destination_root=dir_path)
            self.assertEqual(plan[0].status, "error")
            self.assertIn("does not exist", plan[0].error_message)

            results = execute_move(plan, confirmed=True)
            self.assertEqual(results["failed_count"], 1)
            self.assertEqual(results["moved_count"], 0)

    def test_already_in_place_safety(self):
        """Ensure file already in destination folder is marked already_in_place and skipped."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir).resolve()
            images_dir = dir_path / "Images"
            images_dir.mkdir()
            img_file = images_dir / "sunset.jpg"
            img_file.write_bytes(b"image")

            items = [
                {
                    "file_name": "sunset.jpg",
                    "source_path": img_file,
                    "category": "image",
                }
            ]
            plan = plan_organization(items, destination_root=dir_path)
            self.assertEqual(plan[0].status, "already_in_place")

            results = execute_move(plan, confirmed=True)
            self.assertEqual(results["skipped_count"], 1)
            self.assertEqual(results["moved_count"], 0)


if __name__ == "__main__":
    unittest.main()
