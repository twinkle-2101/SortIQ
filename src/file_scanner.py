"""Read-only directory scanner and path validator for downloads organizer."""

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileInfo:
    """Metadata for a scanned file."""
    file_name: str
    extension: str
    size_bytes: int
    size_formatted: str
    modified_timestamp: float
    modified_str: str
    full_path: Path

    def to_dict(self) -> dict[str, Any]:
        """Convert FileInfo dataclass to dictionary."""
        data = asdict(self)
        data["full_path"] = str(self.full_path)
        return data


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string (B, KB, MB, GB)."""
    if size_bytes < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} B"
    return f"{size:.1f} {units[unit_index]}"


def validate_directory_path(path_input: str | Path) -> tuple[bool, str]:
    """Validate that the given path exists, is a directory, and is accessible.
    
    Returns (is_valid, error_message). If valid, error_message is empty.
    """
    if not path_input:
        return False, "Directory path cannot be empty."

    try:
        path = Path(path_input).expanduser().resolve()
    except Exception as exc:
        return False, f"Invalid path syntax: {exc}"

    if not path.exists():
        return False, f"Directory does not exist: {path}"

    if not path.is_dir():
        return False, f"Path is a file, not a directory: {path}"

    # Check read permissions
    if not os.access(path, os.R_OK):
        return False, f"Directory is not readable (permission denied): {path}"

    return True, ""


def scan_directory(
    directory_path: str | Path,
    recursive: bool = False,
    include_hidden: bool = False,
) -> list[FileInfo]:
    """Perform a read-only scan of files in the specified directory.
    
    Args:
        directory_path: Path to the directory to scan.
        recursive: If True, scans subfolders. Defaults to False (safety requirement).
        include_hidden: If True, includes hidden files (starting with dot). Defaults to False.
        
    Returns:
        List of FileInfo objects sorted alphabetically by filename.
        
    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
        PermissionError: If read permission is denied.
    """
    is_valid, error_message = validate_directory_path(directory_path)
    if not is_valid:
        resolved = Path(directory_path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(error_message)
        if not resolved.is_dir():
            raise NotADirectoryError(error_message)
        raise PermissionError(error_message)

    path = Path(directory_path).expanduser().resolve()
    scanned_files: list[FileInfo] = []

    try:
        if recursive:
            entries = path.rglob("*")
        else:
            entries = path.iterdir()

        for entry in entries:
            try:
                # Symlinks are excluded so a scan cannot target files outside the chosen folder.
                if entry.is_symlink() or not entry.is_file():
                    continue

                # Hidden file filter
                if not include_hidden and entry.name.startswith("."):
                    continue

                stat = entry.stat()
                mod_time = stat.st_mtime
                mod_dt = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")

                file_info = FileInfo(
                    file_name=entry.name,
                    extension=entry.suffix.lower(),
                    size_bytes=stat.st_size,
                    size_formatted=format_file_size(stat.st_size),
                    modified_timestamp=mod_time,
                    modified_str=mod_dt,
                    full_path=entry.resolve(),
                )
                scanned_files.append(file_info)
            except (PermissionError, OSError):
                # Skip individual files that cannot be read due to file-level locks or permissions
                continue

    except PermissionError as exc:
        raise PermissionError(f"Permission denied while reading directory: {path}") from exc

    return sorted(scanned_files, key=lambda f: f.file_name.casefold())
