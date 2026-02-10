#!/usr/bin/env python3
"""
Prompt generation utilities for SNOMED-based prompt planning.

This module provides:
- plan initialization and updates
- term usage counting
- prompt generation using templates
"""

from __future__ import annotations

import json
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from prompt_creation.processing import _normalize_term
from prompt_creation.processing import build_variant_maps
from prompt_creation.processing import tokenize_terms
from prompt_creation.tools import get_random_scenario


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
		term_id = entry.get("term_id")
		if not isinstance(term_id, str) or not term_id.strip():
			term_id = f"term_{len(plan_entries) + 1:04d}"
		# Keep usage list for 50/50 selection later.
		usage = entry.get("usage") if isinstance(entry.get("usage"), list) else []
		corpus_count = entry.get("corpus_count", 0)
		if not isinstance(corpus_count, int):
			corpus_count = 0
		# Remaining count is based on static corpus baseline.
		target_remaining = max(0, target_count - corpus_count)
		plan_entries.append(
			{
				"term_id": term_id,
				"term": term,
				"corpus_count": corpus_count,
				"target_count": target_count,
				"used_in_output": 0,
				"prompt_counter": 0,
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

	# Build counters from existing output to avoid reusing IDs.
	existing_prompts = _read_jsonl(output_file)
	max_counters: dict[str, int] = {}
	for item in existing_prompts:
		prompt_id = item.get("id")
		if not isinstance(prompt_id, str) or "_" not in prompt_id:
			continue
		term_id, suffix = prompt_id.rsplit("_", 1)
		if not suffix.isdigit():
			continue
		count = int(suffix)
		current_max = max_counters.get(term_id, 0)
		if count > current_max:
			max_counters[term_id] = count

	template_content = template_file.read_text(encoding="utf-8")
	template = string.Template(template_content)
	rng = random.Random(seed)

	prompts: list[dict] = []
	for index, entry in enumerate(plan_entries, start=1):
		term = entry.get("term")
		if not isinstance(term, str) or not term.strip():
			continue
		term_id = entry.get("term_id")
		if not isinstance(term_id, str) or not term_id.strip():
			term_id = f"term_{index:04d}"
		target_remaining = entry.get("target_remaining", 0)
		if not isinstance(target_remaining, int) or target_remaining <= 0:
			continue
		prompt_counter = max_counters.get(term_id, entry.get("prompt_counter", 0))
		if not isinstance(prompt_counter, int) or prompt_counter < 0:
			prompt_counter = 0
		prompt_counter += 1
		entry["prompt_counter"] = prompt_counter
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
				"id": f"{term_id}_{prompt_counter:03d}",
				"template": template_file.name,
				"prompt": prompt_text,
			}
		)

	if prompts:
		with output_file.open("a", encoding="utf-8") as handle:
			for item in prompts:
				handle.write(json.dumps(item, ensure_ascii=False) + "\n")
	_write_jsonl(plan_file, plan_entries)