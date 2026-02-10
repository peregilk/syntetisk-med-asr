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
from pathlib import Path

from prompt_creation.prompts import _generate_prompts
from prompt_creation.prompts import _init_plan
from prompt_creation.prompts import _update_plan


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

	return parser


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
		_generate_prompts(
			args.plan_file,
			args.snomed_file,
			args.template,
			args.output_file,
			args.optional_count,
			args.seed,
		)
		return

	raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
	main()