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
import json
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from processing import _normalize_term
from processing import build_variant_maps
from processing import tokenize_terms
from tools import get_random_scenario


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")  # Approximate sentence boundary.


def _read_jsonl(path: Path) -> list[dict]:
	"""Load JSONL into memory, skipping invalid lines."""
	items: list[dict] = []
	if not path.exists():
		return items
	for raw in path.read_text(encoding="utf-8").splitlines():
		line = raw.strip()
		if not line:
			continue
		try:
			items.append(json.loads(line))
		except json.JSONDecodeError:
			continue
	return items


def _write_jsonl(path: Path, items: Iterable[dict]) -> None:
	"""Write JSONL to disk with UTF-8 encoding."""
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for item in items:
			handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _extract_text_from_output(record: dict) -> str | None:
	"""Extract response text from multiple known output.jsonl formats."""
	if "text" in record and isinstance(record["text"], str):
		return record["text"]
	response = record.get("response")
	if isinstance(response, dict):
		text = response.get("text")
		if isinstance(text, str):
			return text
	if isinstance(response, str):
		try:
			parsed = json.loads(response)
		except json.JSONDecodeError:
			return None
		if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
			return parsed["text"]
	return None


def _iter_sentence_tokens(text: str) -> Iterable[tuple[str, bool]]:
	"""Yield tokens with a flag for sentence start for capitalization rules."""
	for sentence in _SENTENCE_SPLIT_PATTERN.split(text):
		tokens = tokenize_terms(sentence)
		for index, token in enumerate(tokens):
			yield token, index == 0


def _is_titlecase_word(token: str) -> bool:
	"""Return True for tokens like 'Tumor' (only first letter uppercase)."""
	return token[:1].isupper() and token[1:].islower()


def _match_term_token(
	token: str,
	sentence_start: bool,
	term_set: set[str],
	normalized_to_term: dict[str, str],
	stemmer,
) -> str | None:
	"""Match a token to a plan term with strict casing + sentence-start exception."""
	if token in term_set:
		return token

	allow_casefold = False
	if sentence_start and _is_titlecase_word(token):
		lowered = token.lower()
		allow_casefold = True
		if lowered in term_set:
			return lowered
	if token.islower() or allow_casefold:
		normalized = _normalize_term(token.lower() if allow_casefold else token, stemmer)
		return normalized_to_term.get(normalized)
	return None


def _count_used_terms(texts: Iterable[str], plan_entries: list[dict]) -> dict[str, int]:
	"""Count usage of plan terms in the generated outputs."""
	term_set = {entry["term"] for entry in plan_entries}
	corpus_counts = {entry["term"]: entry.get("corpus_count", 0) for entry in plan_entries}
	# Reuse processing.py variant normalization to match inflected forms.
	stemmer, _variant_map, _normalized_term_set, _merged_counts = build_variant_maps(
		term_set,
		corpus_counts,
		merge_enabled=True,
	)
	normalized_to_term: dict[str, str] = {}
	for term in term_set:
		normalized = _normalize_term(term, stemmer)
		if normalized in normalized_to_term and normalized_to_term[normalized] != term:
			existing = normalized_to_term[normalized]
			print(
				"Warning: multiple terms normalize to the same key:"
				f" {existing} / {term}",
			)
			continue
		normalized_to_term[normalized] = term

	counts: dict[str, int] = defaultdict(int)
	for text in texts:
		for token, sentence_start in _iter_sentence_tokens(text):
			matched = _match_term_token(
				token,
				sentence_start,
				term_set,
				normalized_to_term,
				stemmer,
			)
			if matched is not None:
				counts[matched] += 1
	return dict(counts)


def _init_plan(
	snomed_file: Path,
	plan_file: Path,
	target_count: int,
) -> None:
	"""Initialize terms_to_use.jsonl from preprocessed SNOMED output."""
	snomed_entries = _read_jsonl(snomed_file)
	plan_entries: list[dict] = []
	for entry in snomed_entries:
		term = entry.get("term")
		if not isinstance(term, str) or not term.strip():
			continue
		# Keep usage list for 50/50 selection later.
		usage = entry.get("usage") if isinstance(entry.get("usage"), list) else []
		corpus_count = entry.get("corpus_count", 0)
		if not isinstance(corpus_count, int):
			corpus_count = 0
		# Remaining count is based on static corpus baseline.
		target_remaining = max(0, target_count - corpus_count)
		plan_entries.append(
			{
				"term": term,
				"corpus_count": corpus_count,
				"target_count": target_count,
				"used_in_output": 0,
				"target_remaining": target_remaining,
				"usage": usage,
			}
		)

	# Sort by remaining counts to prioritize under-covered terms.
	plan_entries.sort(key=lambda item: (item["target_remaining"], item["term"]))
	_write_jsonl(plan_file, plan_entries)


def _update_plan(
	plan_file: Path,
	output_file: Path,
	accumulate: bool,
) -> None:
	"""Update used_in_output and target_remaining based on outputs."""
	plan_entries = _read_jsonl(plan_file)
	if not plan_entries:
		raise FileNotFoundError(f"No plan entries found in {plan_file}")

	output_records = _read_jsonl(output_file)
	# Extract text from output records using known response formats.
	texts = [text for record in output_records if (text := _extract_text_from_output(record))]
	used_counts = _count_used_terms(texts, plan_entries)

	for entry in plan_entries:
		term = entry.get("term")
		if not isinstance(term, str):
			continue
		current_used = entry.get("used_in_output", 0)
		if not isinstance(current_used, int):
			current_used = 0
		new_used = used_counts.get(term, 0)
		# When accumulate is True we add counts to the existing plan.
		entry["used_in_output"] = current_used + new_used if accumulate else new_used
		corpus_count = entry.get("corpus_count", 0)
		target_count = entry.get("target_count", 0)
		if not isinstance(corpus_count, int):
			corpus_count = 0
		if not isinstance(target_count, int):
			target_count = 0
		# Remaining budget excludes both corpus and output usage.
		entry["target_remaining"] = max(
			0,
			target_count - corpus_count - entry["used_in_output"],
		)

	plan_entries.sort(key=lambda item: (item["target_remaining"], item["term"]))
	_write_jsonl(plan_file, plan_entries)


def _choose_expression(term: str, usage: list[str], rng: random.Random) -> str:
	"""Pick term or a usage phrase with 50/50 probability."""
	if usage and rng.random() < 0.5:
		return rng.choice(usage)
	return term


def _generate_prompts(
	plan_file: Path,
	template_file: Path,
	output_file: Path,
	optional_count: int,
	seed: int | None,
) -> None:
	"""Generate prompts using required + optional term expressions."""
	plan_entries = _read_jsonl(plan_file)
	if not plan_entries:
		raise FileNotFoundError(f"No plan entries found in {plan_file}")

	template_content = template_file.read_text(encoding="utf-8")
	template = string.Template(template_content)
	rng = random.Random(seed)

	prompts: list[dict] = []
	for index, entry in enumerate(plan_entries, start=1):
		term = entry.get("term")
		if not isinstance(term, str) or not term.strip():
			continue
		usage = entry.get("usage") if isinstance(entry.get("usage"), list) else []
		# Required expression uses the same 50/50 rule as optional expressions.
		required_expr = _choose_expression(term, usage, rng)

		optional_terms: list[str] = []
		for _ in range(optional_count):
			# Optional expressions are drawn with replacement over all terms.
			optional_entry = rng.choice(plan_entries)
			opt_term = optional_entry.get("term")
			if not isinstance(opt_term, str) or not opt_term.strip():
				continue
			opt_usage = (
				optional_entry.get("usage")
				if isinstance(optional_entry.get("usage"), list)
				else []
			)
			optional_terms.append(_choose_expression(opt_term, opt_usage, rng))

		prompt_text = template.substitute(
			scenario=get_random_scenario(),
			snomed_required=required_expr,
			snomed_optional=", ".join(optional_terms),
		)
		prompts.append(
			{
				"id": f"term_{index}",
				"template": template_file.name,
				"prompt": prompt_text,
			}
		)

	_write_jsonl(output_file, prompts)


def _build_parser() -> argparse.ArgumentParser:
	"""Create CLI parser for plan and prompt operations."""
	script_dir = Path(__file__).parent
	project_root = script_dir.parent
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
		default=project_root / "data" / "output.jsonl",
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
			args.template,
			args.output_file,
			args.optional_count,
			args.seed,
		)
		return

	raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
	main()
