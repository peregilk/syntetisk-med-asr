#!/usr/bin/env python3
"""Shared output filtering helpers for CLI and generation pipeline."""

from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class FilterConfig:
    """Filter thresholds for generated output records."""

    min_used_terms: int = 4
    min_chars: int = 400
    max_chars: int = 600


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating one output record."""

    accepted: bool
    reason: str | None
    text: str | None


def _parse_result_container(result_value: object) -> dict | None:
    """Parse the result payload as a literal dict when possible."""
    if isinstance(result_value, dict):
        return result_value
    if not isinstance(result_value, str):
        return None
    try:
        parsed = ast.literal_eval(result_value)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def validate_output_record(record: dict, config: FilterConfig) -> ValidationResult:
    """Validate one output record against formatting, term count, and length rules."""
    used_terms = record.get("used_terms")
    used_term_count = len(used_terms) if isinstance(used_terms, list) else 0
    if used_term_count < config.min_used_terms:
        return ValidationResult(False, "too_few_used_terms", None)

    result_dict = _parse_result_container(record.get("result"))
    if result_dict is None:
        return ValidationResult(False, "invalid_result_literal", None)

    text = result_dict.get("text") if isinstance(result_dict, dict) else None
    if not isinstance(text, str) or not text.strip():
        return ValidationResult(False, "missing_text", None)

    text_length = len(text)
    if text_length < config.min_chars:
        return ValidationResult(False, "too_short", text)
    if text_length > config.max_chars:
        return ValidationResult(False, "too_long", text)

    return ValidationResult(True, None, text)


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file while skipping invalid rows."""
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def write_jsonl_line(path: Path, record: dict) -> None:
    """Append one JSON object to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def filter_jsonl_file(
    input_file: Path,
    output_file: Path,
    config: FilterConfig,
    rejected_file: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Filter records from input_file and write accepted/rejected outputs."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if overwrite:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("", encoding="utf-8")
        if rejected_file is not None:
            rejected_file.parent.mkdir(parents=True, exist_ok=True)
            rejected_file.write_text("", encoding="utf-8")

    processed = 0
    accepted = 0
    rejected = 0
    reasons: Counter[str] = Counter()

    for record in iter_jsonl(input_file):
        processed += 1
        validation = validate_output_record(record, config)
        if validation.accepted:
            accepted += 1
            write_jsonl_line(output_file, record)
            continue

        rejected += 1
        reason = validation.reason or "unknown"
        reasons[reason] += 1
        if rejected_file is not None:
            audited = dict(record)
            audited["filter_reason"] = reason
            write_jsonl_line(rejected_file, audited)

    return {
        "processed": processed,
        "accepted": accepted,
        "rejected": rejected,
        "reasons": dict(reasons),
    }
