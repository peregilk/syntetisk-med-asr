#!/usr/bin/env python3
"""Utilities for JSONL files and partitioned JSONL directories."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable, Iterator


MAX_PART_SIZE_BYTES = 60 * 1024 * 1024
_PART_PATTERN = re.compile(r"^(?P<prefix>.*)part_(?P<index>\d+)\.jsonl$")


def is_jsonl_file_path(path: Path) -> bool:
    """Return True when path should be treated as a single JSONL file."""
    return path.suffix.lower() == ".jsonl"


def is_partitioned_path(path: Path) -> bool:
    """Return True when path should be treated as a partition directory."""
    return not is_jsonl_file_path(path)


def _list_partitions(path: Path) -> list[tuple[int, Path]]:
    if not path.exists():
        return []
    if not path.is_dir():
        raise ValueError(f"Expected directory for partitioned JSONL: {path}")

    indexed: list[tuple[int, Path]] = []
    prefixes: set[str] = set()
    jsonl_file_count = 0
    for child in path.iterdir():
        if not child.is_file():
            continue
        if child.suffix.lower() == ".jsonl":
            jsonl_file_count += 1
        matched = _PART_PATTERN.match(child.name)
        if matched is None:
            continue
        prefixes.add(matched.group("prefix"))
        indexed.append((int(matched.group("index")), child))

    if len(prefixes) > 1:
        raise ValueError(
            f"Partition directory has mixed prefixes; expected one prefix: {path}"
        )

    if jsonl_file_count > 0 and not indexed:
        raise ValueError(
            "Partition directory contains JSONL files but none match '*part_xx.jsonl' "
            f"suffix pattern: {path}"
        )

    indexed.sort(key=lambda item: item[0])
    for expected_index, (actual_index, _) in enumerate(indexed):
        if expected_index != actual_index:
            raise ValueError(
                f"Partition sequence must be contiguous from *_part_00.jsonl: {path}"
            )
    return indexed


def iter_jsonl_lines(path: Path) -> Iterator[tuple[Path, int, str]]:
    """Yield tuples of (source_path, line_number, raw_line) for file or partitions."""
    if is_jsonl_file_path(path):
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                yield path, line_number, raw_line
        return

    for _, part_path in _list_partitions(path):
        with part_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                yield part_path, line_number, raw_line


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL records from a single file or a partition directory."""
    items: list[dict] = []
    for _source, _line_number, raw_line in iter_jsonl_lines(path):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items


def _part_path(directory: Path, index: int, prefix: str | None = None) -> Path:
    if prefix is None:
        prefix = _partition_prefix(directory)
    return directory / f"{prefix}part_{index:02d}.jsonl"


def _partition_prefix(directory: Path) -> str:
    partitions = _list_partitions(directory)
    if partitions:
        first_name = partitions[0][1].name
        matched = _PART_PATTERN.match(first_name)
        if matched is not None:
            return matched.group("prefix")
    return f"{directory.name}_"


def _normalize_partition_prefix(directory: Path) -> None:
    """Rename partition files in-place so they use the directory-derived prefix."""
    partitions = _list_partitions(directory)
    if not partitions:
        return

    target_prefix = f"{directory.name}_"
    current_name = partitions[0][1].name
    matched = _PART_PATTERN.match(current_name)
    if matched is None:
        return
    current_prefix = matched.group("prefix")
    if current_prefix == target_prefix:
        return

    staged_names: list[tuple[Path, Path]] = []
    for index, source_path in partitions:
        staged_path = directory / f".__rename_stage_{index:02d}.tmp"
        source_path.rename(staged_path)
        target_path = directory / f"{target_prefix}part_{index:02d}.jsonl"
        staged_names.append((staged_path, target_path))

    for staged_path, target_path in staged_names:
        staged_path.rename(target_path)


def _record_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"


def _record_size_bytes(record: dict) -> int:
    return len(_record_line(record).encode("utf-8"))


def _remove_partition_files(directory: Path) -> None:
    if not directory.exists():
        return
    if not directory.is_dir():
        raise ValueError(f"Expected partition directory path: {directory}")
    for _, part_path in _list_partitions(directory):
        part_path.unlink()


def clear_jsonl_target(path: Path) -> None:
    """Clear existing records from a target path without deleting the directory itself."""
    if is_jsonl_file_path(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return

    path.mkdir(parents=True, exist_ok=True)
    _remove_partition_files(path)


def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    """Overwrite JSONL data to file path or partitioned directory path."""
    if is_jsonl_file_path(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(_record_line(item))
        return

    path.mkdir(parents=True, exist_ok=True)
    prefix = _partition_prefix(path)
    _remove_partition_files(path)

    part_index = 0
    current_size = 0
    handle = None
    try:
        for item in items:
            line = _record_line(item)
            line_size = len(line.encode("utf-8"))
            if handle is None:
                handle = _part_path(path, part_index, prefix=prefix).open("w", encoding="utf-8")
                current_size = 0
            elif current_size + line_size > MAX_PART_SIZE_BYTES and current_size > 0:
                handle.close()
                part_index += 1
                handle = _part_path(path, part_index, prefix=prefix).open("w", encoding="utf-8")
                current_size = 0

            handle.write(line)
            current_size += line_size
    finally:
        if handle is not None:
            handle.close()


def append_jsonl_record(path: Path, record: dict) -> None:
    """Append a single JSONL record to file path or partitioned directory path."""
    if is_jsonl_file_path(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_record_line(record))
        return

    path.mkdir(parents=True, exist_ok=True)
    partitions = _list_partitions(path)
    line = _record_line(record)
    line_size = len(line.encode("utf-8"))

    if not partitions:
        target_path = _part_path(path, 0)
    else:
        last_index, last_path = partitions[-1]
        last_size = last_path.stat().st_size
        if last_size + line_size > MAX_PART_SIZE_BYTES and last_size > 0:
            target_path = _part_path(path, last_index + 1)
        else:
            target_path = last_path

    with target_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def make_temp_target(path: Path, suffix: str = ".tmp") -> Path:
    """Build a temporary target path preserving file-vs-directory mode."""
    if is_jsonl_file_path(path):
        return path.with_suffix(path.suffix + suffix)
    return path.parent / f"{path.name}{suffix}"


def remove_jsonl_target(path: Path) -> None:
    """Remove file target or full partition directory target if present."""
    if not path.exists():
        return
    if is_jsonl_file_path(path):
        if path.is_file():
            path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def replace_jsonl_target(source: Path, destination: Path) -> None:
    """Replace destination with source for either file or partitioned directory mode."""
    source_is_file = is_jsonl_file_path(source)
    destination_is_file = is_jsonl_file_path(destination)
    if source_is_file != destination_is_file:
        raise ValueError("Cannot replace JSONL targets across file/directory modes")

    if not source.exists():
        clear_jsonl_target(destination)
        return

    if destination_is_file:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    remove_jsonl_target(destination)
    source.rename(destination)
    _normalize_partition_prefix(destination)
