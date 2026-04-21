#!/usr/bin/env python3
"""Extract only id/result fields from JSONL output files.

Behavior:
- If --output-path ends with .jsonl, all input files are merged into one output file.
- If --output-path is a directory:
  - single input .jsonl file -> writes one output file inside that directory
  - directory input (multiple .jsonl files) -> writes one output file per input file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_jsonl_file(path: Path) -> bool:
    """Return True when path should be treated as a single JSONL file path."""
    return path.suffix.lower() == ".jsonl"


def list_input_files(input_path: Path) -> list[Path]:
    """Resolve input path into one or more JSONL files."""
    if is_jsonl_file(input_path):
        if not input_path.exists() or not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return [input_path]

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    files = sorted(
        child for child in input_path.iterdir() if child.is_file() and child.suffix.lower() == ".jsonl"
    )
    if not files:
        raise ValueError(f"No .jsonl files found in input directory: {input_path}")
    return files


def extract_record_fields(record: dict) -> dict | None:
    """Keep only id and result fields; return None when either field is missing."""
    record_id = record.get("id")
    result = record.get("result")
    if record_id is None or result is None:
        return None
    return {"id": record_id, "result": result}


def process_file(input_file: Path, output_file: Path) -> tuple[int, int]:
    """Extract id/result records from one input file into one output file.

    Returns (written_count, skipped_count).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0
    skipped_count = 0

    with input_file.open("r", encoding="utf-8") as source, output_file.open(
        "w", encoding="utf-8"
    ) as target:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                skipped_count += 1
                continue

            if not isinstance(parsed, dict):
                skipped_count += 1
                continue

            extracted = extract_record_fields(parsed)
            if extracted is None:
                skipped_count += 1
                continue

            target.write(json.dumps(extracted, ensure_ascii=False) + "\n")
            written_count += 1

    return written_count, skipped_count


def merge_files(input_files: list[Path], output_file: Path) -> tuple[int, int]:
    """Merge extracted id/result records from many files into one output JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0
    skipped_count = 0

    with output_file.open("w", encoding="utf-8") as target:
        for input_file in input_files:
            with input_file.open("r", encoding="utf-8") as source:
                for raw_line in source:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_count += 1
                        continue

                    if not isinstance(parsed, dict):
                        skipped_count += 1
                        continue

                    extracted = extract_record_fields(parsed)
                    if extracted is None:
                        skipped_count += 1
                        continue

                    target.write(json.dumps(extracted, ensure_ascii=False) + "\n")
                    written_count += 1

    return written_count, skipped_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract id/result from JSONL files with flexible file/directory I/O",
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Input .jsonl file or directory")
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output .jsonl file or output directory",
    )
    args = parser.parse_args()

    input_files = list_input_files(args.input_path)

    total_written = 0
    total_skipped = 0
    written_outputs: list[Path] = []

    if is_jsonl_file(args.output_path):
        total_written, total_skipped = merge_files(input_files, args.output_path)
        written_outputs.append(args.output_path)
    else:
        args.output_path.mkdir(parents=True, exist_ok=True)

        if len(input_files) == 1 and is_jsonl_file(args.input_path):
            output_file = args.output_path / input_files[0].name
            written, skipped = process_file(input_files[0], output_file)
            total_written += written
            total_skipped += skipped
            written_outputs.append(output_file)
        else:
            for input_file in input_files:
                output_file = args.output_path / input_file.name
                written, skipped = process_file(input_file, output_file)
                total_written += written
                total_skipped += skipped
                written_outputs.append(output_file)

    print("Extraction summary:")
    print(f"  input_files={len(input_files)}")
    print(f"  output_files={len(written_outputs)}")
    print(f"  written_records={total_written}")
    print(f"  skipped_records={total_skipped}")


if __name__ == "__main__":
    main()
