#!/usr/bin/env python3
"""Identify and move obsolete output records while preserving term coverage."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from prompt_creation.output_filter import iter_jsonl


@dataclass(frozen=True)
class PruneSummary:
    """Summary metrics for obsolete-pruning pass."""

    processed: int
    kept: int
    moved: int
    moved_missing_used_terms: int
    moved_saturated_terms: int


def _required_output_counts(plan_entries: list[dict]) -> dict[str, int]:
    """Return required output counts per term based on target and corpus counts."""
    required: dict[str, int] = {}
    for entry in plan_entries:
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
    """Return unique used terms in order, accepting only non-empty strings."""
    used_terms = record.get("used_terms")
    if not isinstance(used_terms, list):
        return []

    unique_terms: list[str] = []
    seen: set[str] = set()
    for term in used_terms:
        if not isinstance(term, str) or not term.strip():
            continue
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
    return unique_terms


def _record_generation_key(record: dict, index: int) -> tuple[int, str, int]:
    """Return ordering key so all *_001 records come before *_002, and so on."""
    record_id = record.get("id")
    if isinstance(record_id, str) and "_" in record_id:
        prefix, suffix = record_id.rsplit("_", 1)
        if suffix.isdigit():
            return int(suffix), prefix, index
    # Unknown formats are placed last in ascending order.
    return 10**9, "", index


def prune_obsolete_jsonl(
    plan_file: Path,
    input_file: Path,
    output_file: Path,
    rejected_file: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Prune obsolete records from input_file into output_file and rejected_file."""
    if not plan_file.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    plan_entries = list(iter_jsonl(plan_file))
    required_counts = _required_output_counts(plan_entries)

    # First pass: build per-record normalized used terms and term totals.
    terms_by_index: list[list[str]] = []
    generation_order_keys: list[tuple[int, str, int]] = []
    current_counts: Counter[str] = Counter()
    for index, record in enumerate(iter_jsonl(input_file)):
        terms = _normalize_used_terms(record)
        terms_by_index.append(terms)
        generation_order_keys.append(_record_generation_key(record, index))
        current_counts.update(terms)
    baseline_counts = Counter(current_counts)

    obsolete_indexes: set[int] = set()
    moved_missing_used_terms = 0
    moved_saturated_terms = 0

    # Prune from latest generation-step first: ..._003 before ..._002 before ..._001.
    ordered_indexes = [
        idx for _, _, idx in sorted(generation_order_keys, key=lambda item: (item[0], item[1], item[2]))
    ]
    for index in reversed(ordered_indexes):
        terms = terms_by_index[index]
        if not terms:
            obsolete_indexes.add(index)
            moved_missing_used_terms += 1
            continue

        removable = True
        for term in terms:
            required = required_counts.get(term, 0)
            if current_counts.get(term, 0) - 1 < required:
                removable = False
                break

        if not removable:
            continue

        obsolete_indexes.add(index)
        moved_saturated_terms += 1
        for term in terms:
            current_counts[term] -= 1

    coverage_regressions: list[str] = []
    for term, required in required_counts.items():
        baseline = baseline_counts.get(term, 0)
        floor = min(baseline, required)
        if current_counts.get(term, 0) < floor:
            coverage_regressions.append(term)
    if coverage_regressions:
        preview = ", ".join(coverage_regressions[:20])
        raise ValueError(
            "Coverage validation failed after pruning. "
            f"Coverage regressions ({len(coverage_regressions)}): {preview}"
        )

    kept_count = 0
    moved_count = 0
    processed = len(terms_by_index)

    if overwrite:
        temp_output = output_file.with_suffix(output_file.suffix + ".tmp")
        temp_rejected = rejected_file.with_suffix(rejected_file.suffix + ".tmp")
        temp_output.parent.mkdir(parents=True, exist_ok=True)
        temp_rejected.parent.mkdir(parents=True, exist_ok=True)

        with temp_output.open("w", encoding="utf-8") as kept_handle, temp_rejected.open(
            "w", encoding="utf-8"
        ) as rejected_handle:
            for index, record in enumerate(iter_jsonl(input_file)):
                if index in obsolete_indexes:
                    obsolete = dict(record)
                    obsolete["filter_reason"] = "obsolete_terms_saturated"
                    rejected_handle.write(json.dumps(obsolete, ensure_ascii=False) + "\n")
                    moved_count += 1
                else:
                    kept_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept_count += 1

        temp_output.replace(output_file)
        temp_rejected.replace(rejected_file)
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        rejected_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("a", encoding="utf-8") as kept_handle, rejected_file.open(
            "a", encoding="utf-8"
        ) as rejected_handle:
            for index, record in enumerate(iter_jsonl(input_file)):
                if index in obsolete_indexes:
                    obsolete = dict(record)
                    obsolete["filter_reason"] = "obsolete_terms_saturated"
                    rejected_handle.write(json.dumps(obsolete, ensure_ascii=False) + "\n")
                    moved_count += 1
                else:
                    kept_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept_count += 1

    summary = PruneSummary(
        processed=processed,
        kept=kept_count,
        moved=moved_count,
        moved_missing_used_terms=moved_missing_used_terms,
        moved_saturated_terms=moved_saturated_terms,
    )

    return {
        "processed": summary.processed,
        "kept": summary.kept,
        "moved": summary.moved,
        "moved_missing_used_terms": summary.moved_missing_used_terms,
        "moved_saturated_terms": summary.moved_saturated_terms,
        "coverage_regressions": len(coverage_regressions),
    }
