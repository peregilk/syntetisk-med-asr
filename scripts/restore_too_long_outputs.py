#!/usr/bin/env python3
"""Restore previously rejected filtered outputs with relaxed thresholds.

This helper is intended for ad-hoc use. It restores records from rejected -> output
only when all conditions are met:
- original filter reason is one of the configured restore reasons
- record is not marked obsolete
- record ID is not already present in output
- record passes all current filtering criteria using a relaxed max length (default 650)

Optional: with --plan-file, restore can require that each restored record contributes
at least one currently under-covered term according to the plan.

It supports both single JSONL files and partitioned directories.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt_creation.chunked_jsonl import append_jsonl_record
from prompt_creation.chunked_jsonl import read_jsonl
from prompt_creation.chunked_jsonl import write_jsonl
from prompt_creation.output_filter import FilterConfig
from prompt_creation.output_filter import validate_output_record


def _is_obsolete_reason(reason: object) -> bool:
    if not isinstance(reason, str):
        return False
    return reason.startswith("obsolete")


def _parse_restore_reasons(value: str) -> set[str]:
    reasons = {token.strip() for token in value.split(",") if token.strip()}
    if not reasons:
        raise ValueError("--restore-reasons must contain at least one reason")
    return reasons


def _required_output_counts(plan_file: Path) -> dict[str, int]:
    required: dict[str, int] = {}
    for entry in read_jsonl(plan_file):
        if not isinstance(entry, dict):
            continue

        term = entry.get("term")
        if not isinstance(term, str) or not term.strip():
            continue

        target_count = entry.get("target_count", 0)
        corpus_count = entry.get("corpus_count", 0)
        if not isinstance(target_count, int):
            target_count = 0
        if not isinstance(corpus_count, int):
            corpus_count = 0

        required[term] = max(0, target_count - corpus_count)

    return required


def _normalize_used_terms(record: dict) -> list[str]:
    used_terms = record.get("used_terms")
    if not isinstance(used_terms, list):
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for term in used_terms:
        if not isinstance(term, str) or not term.strip() or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _current_term_counts(records: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        counts.update(_normalize_used_terms(record))
    return counts


def restore_filtered_outputs(
    output_file: Path,
    rejected_file: Path,
    min_used_terms: int,
    min_chars: int,
    max_chars: int,
    restore_reasons: set[str],
    plan_file: Path | None,
    dry_run: bool,
) -> dict[str, int]:
    """Restore valid filtered records from rejected storage back to output."""
    output_records = read_jsonl(output_file)
    rejected_records = read_jsonl(rejected_file)

    output_ids = {
        record.get("id")
        for record in output_records
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }

    obsolete_ids = {
        record.get("id")
        for record in rejected_records
        if isinstance(record, dict)
        and _is_obsolete_reason(record.get("filter_reason"))
        and isinstance(record.get("id"), str)
    }

    config = FilterConfig(
        min_used_terms=min_used_terms,
        min_chars=min_chars,
        max_chars=max_chars,
    )

    required_counts: dict[str, int] | None = None
    current_counts: Counter[str] | None = None
    if plan_file is not None:
        if not plan_file.exists():
            raise FileNotFoundError(f"Plan file/path not found: {plan_file}")
        required_counts = _required_output_counts(plan_file)
        current_counts = _current_term_counts(output_records)

    restored_ids: set[str] = set()
    kept_rejected: list[dict] = []

    stats = {
        "rejected_total": len(rejected_records),
        "restore_reason_candidates": 0,
        "restored": 0,
        "skipped_obsolete": 0,
        "skipped_existing_output": 0,
        "skipped_invalid": 0,
        "skipped_redundant_terms": 0,
    }

    for record in rejected_records:
        if not isinstance(record, dict):
            continue

        record_id = record.get("id")
        reason = record.get("filter_reason")

        if reason not in restore_reasons:
            kept_rejected.append(record)
            continue

        stats["restore_reason_candidates"] += 1

        if not isinstance(record_id, str) or not record_id:
            stats["skipped_invalid"] += 1
            kept_rejected.append(record)
            continue

        if record_id in obsolete_ids:
            stats["skipped_obsolete"] += 1
            kept_rejected.append(record)
            continue

        if record_id in output_ids or record_id in restored_ids:
            stats["skipped_existing_output"] += 1
            kept_rejected.append(record)
            continue

        validation = validate_output_record(record, config)
        if not validation.accepted:
            stats["skipped_invalid"] += 1
            kept_rejected.append(record)
            continue

        if required_counts is not None and current_counts is not None:
            used_terms = _normalize_used_terms(record)
            contributes_needed_term = any(
                current_counts.get(term, 0) < required_counts.get(term, 0)
                for term in used_terms
            )
            if not contributes_needed_term:
                stats["skipped_redundant_terms"] += 1
                kept_rejected.append(record)
                continue

        restored = dict(record)
        restored.pop("filter_reason", None)

        if not dry_run:
            append_jsonl_record(output_file, restored)
        output_ids.add(record_id)
        restored_ids.add(record_id)
        if current_counts is not None:
            current_counts.update(_normalize_used_terms(restored))

    stats["restored"] = len(restored_ids)

    if not dry_run:
        write_jsonl(rejected_file, kept_rejected)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore rejected filtered outputs using current filter thresholds",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSONL file or partition directory",
    )
    parser.add_argument(
        "--rejected-file",
        type=Path,
        required=True,
        help="Rejected JSONL file or partition directory",
    )
    parser.add_argument(
        "--min-used-terms",
        type=int,
        default=3,
        help="Minimum used terms required",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=400,
        help="Minimum text length",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=650,
        help="Maximum text length for restoration checks",
    )
    parser.add_argument(
        "--restore-reasons",
        type=str,
        default="too_long,too_short,too_few_used_terms",
        help="Comma-separated rejection reasons eligible for restoration",
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=None,
        help=(
            "Optional plan JSONL. If set, only restore records that still contribute "
            "to under-covered terms (prevents restoring redundant records)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report only; do not modify files",
    )

    args = parser.parse_args()

    if not args.rejected_file.exists():
        raise FileNotFoundError(f"Rejected file/path not found: {args.rejected_file}")

    summary = restore_filtered_outputs(
        output_file=args.output_file,
        rejected_file=args.rejected_file,
        min_used_terms=args.min_used_terms,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        restore_reasons=_parse_restore_reasons(args.restore_reasons),
        plan_file=args.plan_file,
        dry_run=args.dry_run,
    )

    print("Restore summary:")
    for key in (
        "rejected_total",
        "restore_reason_candidates",
        "restored",
        "skipped_obsolete",
        "skipped_existing_output",
        "skipped_invalid",
        "skipped_redundant_terms",
    ):
        print(f"  {key}={summary[key]}")

    if args.dry_run:
        print("Dry-run mode: no files were modified")


if __name__ == "__main__":
    main()
