#!/usr/bin/env python3
"""
Generate prompts and maintain a term usage plan.

Workflow:
1) init-plan: Build data/terms_to_use.jsonl from data/preprocessed/snomed.jsonl.
2) update-plan: Count term usage in data/output.jsonl and update remaining counts.
3) generate: Produce prompts.jsonl from the plan and template.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from prompt_creation.prompts import _generate_prompts
from prompt_creation.prompts import _init_plan
from prompt_creation.prompts import _read_jsonl
from prompt_creation.prompts import _update_plan
from prompt_creation.prompts import _write_jsonl


def _build_parser() -> argparse.ArgumentParser:
	"""Create CLI parser for plan and prompt operations."""
	script_dir = Path(__file__).parent
	project_root = script_dir
	parser = argparse.ArgumentParser(
		description="Generate prompts and maintain term usage plan.",
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	init_parser = subparsers.add_parser("init-plan", help="Create terms_to_use.jsonl")
	init_parser.add_argument(
		"--snomed-file",
		type=Path,
		default=project_root / "data" / "preprocessed" / "snomed.jsonl",
		help="Path to preprocessed SNOMED terms",
	)
	init_parser.add_argument(
		"--plan-file",
		type=Path,
		default=project_root / "data" / "terms_to_use.jsonl",
		help="Output plan file",
	)
	init_parser.add_argument(
		"--target-count",
		type=int,
		default=100,
		help="Target total usage per term",
	)

	update_parser = subparsers.add_parser("update-plan", help="Update plan from output.jsonl")
	update_parser.add_argument(
		"--plan-file",
		type=Path,
		default=project_root / "data" / "terms_to_use.jsonl",
		help="Plan file to update",
	)
	update_parser.add_argument(
		"--output-file",
		type=Path,
		default=project_root / "data" / "outputs" / "output.jsonl",
		help="JSONL with model responses",
	)
	update_parser.add_argument(
		"--accumulate",
		action="store_true",
		help="Add new counts to existing used_in_output",
	)

	generate_parser = subparsers.add_parser("generate", help="Generate prompts from plan")
	generate_parser.add_argument(
		"--plan-file",
		type=Path,
		default=project_root / "data" / "terms_to_use.jsonl",
		help="Plan file to use",
	)
	generate_parser.add_argument(
		"--snomed-file",
		type=Path,
		default=project_root / "data" / "preprocessed" / "snomed.jsonl",
		help="Path to preprocessed SNOMED terms",
	)
	generate_parser.add_argument(
		"--template",
		type=Path,
		default=project_root / "templates" / "a.txt",
		help="Prompt template",
	)
	generate_parser.add_argument(
		"--output-file",
		type=Path,
		default=project_root / "prompts" / "generated_prompts.jsonl",
		help="Output prompts JSONL",
	)
	generate_parser.add_argument(
		"--optional-count",
		type=int,
		default=10,
		help="Number of optional terms per prompt",
	)
	generate_parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed for reproducibility",
	)
	generate_parser.add_argument(
		"--generate-all",
		action="store_true",
		help="Generate all prompts at once using estimated target_remaining",
	)
	generate_parser.add_argument(
		"--reasoning-effort",
		type=str,
		default="none",
		choices=["low", "medium", "high", "none"],
		help="Reasoning effort used when estimating prompt coverage",
	)

	return parser


def _estimate_prompt_targets(
	plan_file: Path,
	optional_count: int,
	reasoning_effort: str,
) -> None:
	"""Estimate prompt counts per term and write them to target_remaining."""
	plan_entries = _read_jsonl(plan_file)
	if not plan_entries:
		raise FileNotFoundError(f"No plan entries found in {plan_file}")

	base_avg_total = 4.5 if reasoning_effort == "none" else 6.1
	optional_scale = max(0, optional_count) / 10
	expected_total = 1 + (base_avg_total - 1) * optional_scale
	expected_total = max(1.0, expected_total)

	for entry in plan_entries:
		current_remaining = entry.get("target_remaining")
		if not isinstance(current_remaining, int) or current_remaining < 0:
			target_count = entry.get("target_count", 0)
			corpus_count = entry.get("corpus_count", 0)
			used_in_output = entry.get("used_in_output", 0)
			
			current_remaining = max(0, target_count - corpus_count - used_in_output)

		estimated_prompts = (
			math.ceil(current_remaining / expected_total)
			if current_remaining > 0
			else 0
		)
		entry["target_remaining"] = estimated_prompts

	_write_jsonl(plan_file, plan_entries)


def main() -> None:
	"""Entrypoint for CLI commands."""
	parser = _build_parser()
	args = parser.parse_args()

	if args.command == "init-plan":
		_init_plan(args.snomed_file, args.plan_file, args.target_count)
		return
	if args.command == "update-plan":
		_update_plan(args.plan_file, args.output_file, args.accumulate)
		return
	if args.command == "generate":
		if args.generate_all:
			_estimate_prompt_targets(
				args.plan_file,
				args.optional_count,
				args.reasoning_effort,
			)
		_generate_prompts(
			args.plan_file,
			args.snomed_file,
			args.template,
			args.output_file,
			args.optional_count,
			args.seed,
			generate_all=args.generate_all,
		)
		return

	raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
	main()