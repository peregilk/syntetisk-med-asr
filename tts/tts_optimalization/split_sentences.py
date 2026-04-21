#!/usr/bin/env python3
"""Split long JSONL result texts into sentence-based chunks for TTS.

Expected record format:
	{"id": "...", "result": "{\"text\": \"...\"}"}

The script also handles `result` as a dict with a `text` field.
It is deterministic and only splits on sentence typography (.?!).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm


MAX_CHARS = 300
TARGET_MIN_CHARS = 150
TARGET_MAX_CHARS = 300
TARGET_MID_CHARS = 225
DEFAULT_INPUT_PATH = Path("data/outputs/output.jsonl")

# Split on whitespace after sentence punctuation, optionally including closing quote marks.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?:(?<=[.!?])|(?<=[.!?]['»’]))\s+")
# Last-resort fallbacks for overlong fragments.
_COMMA_FALLBACK_SPLIT_PATTERN = re.compile(r"(?<=,)\s*")
_CAPITAL_FALLBACK_SPLIT_PATTERN = re.compile(r"\s+(?=[A-ZÆØÅ])")
_SPACE_FALLBACK_SPLIT_PATTERN = re.compile(r"\s+")
_JSON_DECODER = json.JSONDecoder()
_INVALID_ESCAPE_PATTERN = re.compile(r"\\([^\"\\/bfnrtu])")


@dataclass(frozen=True)
class _PartitionResult:
	cost: int
	chunks: list[str]


def _read_text_from_result(result_value: object) -> tuple[str | None, str]:
	"""Extract text from result payload.

	Returns (text, mode) where mode is "string" or "dict" for writeback strategy.
	"""
	if isinstance(result_value, str):
		try:
			parsed = json.loads(result_value)
		except json.JSONDecodeError:
			sanitized_result_value = _sanitize_invalid_json_escapes(result_value)
			if sanitized_result_value != result_value:
				try:
					parsed = json.loads(sanitized_result_value)
				except json.JSONDecodeError:
					parsed = None
				if isinstance(parsed, dict):
					text = parsed.get("text")
					if isinstance(text, str):
						return text, "string"

			recovered_text = _read_text_from_malformed_result_string(result_value)
			if recovered_text is not None:
				return recovered_text, "string"

			recovered_text = _read_text_from_malformed_result_string(sanitized_result_value)
			if recovered_text is not None:
				return recovered_text, "string"
			return None, "string"
		if not isinstance(parsed, dict):
			return None, "string"
		text = parsed.get("text")
		if not isinstance(text, str):
			return None, "string"
		return text, "string"

	if isinstance(result_value, dict):
		text = result_value.get("text")
		if isinstance(text, str):
			return text, "dict"

	return None, "string"


def _join_text_fragments(fragments: list[str]) -> str:
	if not fragments:
		return ""

	combined = fragments[0]
	for fragment in fragments[1:]:
		# Adjacent JSON strings imply missing escaping in the producer.
		if combined and fragment and not combined[-1].isspace() and not fragment[0].isspace():
			combined += " "
		combined += fragment
	return combined


def _sanitize_invalid_json_escapes(text: str) -> str:
	"""Drop backslashes before non-JSON escape characters.

	Example: \\' -> ' while preserving valid escapes like \\n, \\" and \\u00f8.
	"""
	return _INVALID_ESCAPE_PATTERN.sub(r"\1", text)


def _read_text_from_malformed_result_string(result_text: str) -> str | None:
	"""Recover `text` from malformed result JSON when possible.

	Handles common producer errors where `text` becomes multiple adjacent
	JSON string literals, for example: {"text": "A" "B"}.
	"""
	key_index = result_text.find('"text"')
	if key_index < 0:
		return None

	colon_index = result_text.find(":", key_index)
	if colon_index < 0:
		return None

	cursor = colon_index + 1
	text_length = len(result_text)
	while cursor < text_length and result_text[cursor].isspace():
		cursor += 1

	fragments: list[str] = []
	while cursor < text_length:
		if result_text[cursor] != '"':
			break
		try:
			fragment, next_cursor = _JSON_DECODER.raw_decode(result_text, cursor)
		except json.JSONDecodeError:
			return None

		if not isinstance(fragment, str):
			return None
		fragments.append(fragment)
		cursor = next_cursor

		while cursor < text_length and result_text[cursor].isspace():
			cursor += 1

		if cursor >= text_length:
			break
		if result_text[cursor] == '"':
			# Continue parsing adjacent string literals.
			continue
		if result_text[cursor] in {"}", ","}:
			break
		return None

	if not fragments:
		return None
	return _join_text_fragments(fragments)


def _write_text_back(result_value: object, mode: str, new_text: str) -> object:
	if mode == "dict" and isinstance(result_value, dict):
		updated = dict(result_value)
		updated["text"] = new_text
		return updated

	payload = {"text": new_text}
	return json.dumps(payload, ensure_ascii=False)


def _jsonl_files_from_input(input_path: Path) -> list[Path]:
	if input_path.suffix.lower() == ".jsonl":
		if not input_path.exists() or not input_path.is_file():
			raise FileNotFoundError(f"Input file not found: {input_path}")
		return [input_path]

	if not input_path.exists() or not input_path.is_dir():
		raise FileNotFoundError(f"Input directory not found: {input_path}")

	files = sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".jsonl")
	if not files:
		raise ValueError(f"No .jsonl files found in input directory: {input_path}")
	return files


def _split_output_name(input_file: Path) -> str:
	return f"{input_file.stem}_split.jsonl"


def _default_output_path(input_path: Path) -> Path:
	if input_path.suffix.lower() == ".jsonl":
		return input_path.with_name(_split_output_name(input_path))
	return input_path.parent / f"{input_path.name}_split"


def _count_non_empty_lines(path: Path) -> int:
	count = 0
	with path.open("r", encoding="utf-8") as handle:
		for raw_line in handle:
			if raw_line.strip():
				count += 1
	return count


def _record_id(record: dict, input_file: Path, line_number: int) -> str:
	raw_id = record.get("id")
	if raw_id is None:
		return f"{input_file.name}:{line_number}"
	return str(raw_id)


def _split_into_sentences(text: str) -> list[str]:
	sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(text.strip()) if sentence.strip()]
	if not sentences:
		return [text.strip()]
	return sentences


def _split_long_sentences_with_fallbacks(sentences: list[str]) -> list[str]:
	"""Split overlong fragments with ordered last-resort boundaries.

	Order is strict and deterministic:
	1) normal comma
	2) before capital letter
	3) on space
	"""
	patterns = (
		_COMMA_FALLBACK_SPLIT_PATTERN,
		_CAPITAL_FALLBACK_SPLIT_PATTERN,
		_SPACE_FALLBACK_SPLIT_PATTERN,
	)

	refined_sentences: list[str] = []
	for sentence in sentences:
		fragments = [sentence]
		for pattern in patterns:
			next_fragments: list[str] = []
			for fragment in fragments:
				if len(fragment) <= MAX_CHARS:
					next_fragments.append(fragment)
					continue

				parts = [part.strip() for part in pattern.split(fragment.strip()) if part.strip()]
				if len(parts) <= 1:
					next_fragments.append(fragment)
					continue
				next_fragments.extend(parts)
			fragments = next_fragments
		refined_sentences.extend(fragments)

	return refined_sentences


def _build_sentence_lengths(sentences: list[str]) -> list[int]:
	lengths = [0]
	running = 0
	for index, sentence in enumerate(sentences):
		running += len(sentence)
		if index > 0:
			running += 1
		lengths.append(running)
	return lengths


def _slice_length(prefix_lengths: list[int], start: int, end: int) -> int:
	if start == end:
		return 0
	raw = prefix_lengths[end] - prefix_lengths[start]
	# Remove spaces that were counted before `start`.
	return raw - start


def _join_sentences(sentences: list[str], start: int, end: int) -> str:
	return " ".join(sentences[start:end])


def _chunk_penalty(length: int) -> int:
	# Keep chunks close to 200-300 and centered around 250 characters.
	penalty = (length - TARGET_MID_CHARS) ** 2
	if length < TARGET_MIN_CHARS:
		penalty += (TARGET_MIN_CHARS - length) ** 2 * 3
	if length > TARGET_MAX_CHARS:
		penalty += (length - TARGET_MAX_CHARS) ** 2 * 2
	return penalty


def _partition_for_k(sentences: list[str], k: int) -> Optional[_PartitionResult]:
	n = len(sentences)
	if k <= 0 or k > n:
		return None

	prefix_lengths = _build_sentence_lengths(sentences)
	inf = 10**18
	dp = [[inf] * (n + 1) for _ in range(k + 1)]
	cut = [[-1] * (n + 1) for _ in range(k + 1)]
	dp[0][0] = 0

	for parts in range(1, k + 1):
		for end in range(parts, n + 1):
			start_min = parts - 1
			start_max = end - 1
			best_cost = inf
			best_start = -1

			for start in range(start_min, start_max + 1):
				previous = dp[parts - 1][start]
				if previous == inf:
					continue
				length = _slice_length(prefix_lengths, start, end)
				if length > MAX_CHARS:
					continue
				current_cost = previous + _chunk_penalty(length)
				if current_cost < best_cost:
					best_cost = current_cost
					best_start = start

			dp[parts][end] = best_cost
			cut[parts][end] = best_start

	if dp[k][n] == inf:
		return None

	chunks: list[str] = []
	end = n
	parts = k
	while parts > 0:
		start = cut[parts][end]
		if start < 0:
			return None
		chunks.append(_join_sentences(sentences, start, end))
		end = start
		parts -= 1
	chunks.reverse()
	return _PartitionResult(cost=int(dp[k][n]), chunks=chunks)


def _split_text_by_sentences(text: str) -> list[str]:
	clean_text = text.strip()
	if len(clean_text) <= MAX_CHARS:
		return [clean_text]

	sentences = _split_into_sentences(clean_text)
	if any(len(sentence) > MAX_CHARS for sentence in sentences):
		sentences = _split_long_sentences_with_fallbacks(sentences)

	if any(len(sentence) > MAX_CHARS for sentence in sentences):
		longest = max(len(sentence) for sentence in sentences)
		raise ValueError(
			"Found a single sentence longer than max chars "
			f"({longest}>{MAX_CHARS}); cannot split without breaking sentence boundaries or fallback boundaries. Sentence: '{clean_text}'"
		)

	total_len = len(" ".join(sentences))
	min_chunks = max(1, math.ceil(total_len / MAX_CHARS))
	max_chunks = max(min_chunks, math.ceil(total_len / TARGET_MIN_CHARS))

	best: Optional[_PartitionResult] = None
	for k in range(min_chunks, max_chunks + 1):
		candidate = _partition_for_k(sentences, k)
		if candidate is None:
			continue
		if best is None or candidate.cost < best.cost:
			best = candidate

	if best is None:
		raise ValueError("Unable to build sentence chunks within max length")

	return best.chunks


def _split_record(record: dict, input_file: Path, line_number: int) -> list[dict]:
	record_id = _record_id(record, input_file, line_number)
	text, mode = _read_text_from_result(record.get("result"))
	if text is None:
		raise ValueError(f"Missing or invalid result.text payload in record with id '{record_id}' at {input_file}:{line_number}")

	chunks = _split_text_by_sentences(text)
	output_records: list[dict] = []
	for index, chunk in enumerate(chunks, start=1):
		updated_record = dict(record)
		updated_record["id"] = f"{record_id}_{index}"
		updated_record["result"] = _write_text_back(record.get("result"), mode, chunk)
		output_records.append(updated_record)
	return output_records


def _is_unsplittable_error(exc: ValueError) -> bool:
	message = str(exc)
	return (
		"single sentence longer than max chars" in message
		or "Unable to build sentence chunks within max length" in message
	)


def _transform_file(input_file: Path, output_file: Path) -> tuple[int, int, int]:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	temp_output_file = output_file.with_name(f"{output_file.stem}.split.tmp.jsonl")

	processed_records = 0
	emitted_records = 0
	skipped_unsplittable_records = 0
	total_non_empty_lines = _count_non_empty_lines(input_file)

	try:
		with input_file.open("r", encoding="utf-8") as source, temp_output_file.open(
			"w", encoding="utf-8"
		) as target, tqdm(
			total=total_non_empty_lines,
			desc=f"Splitting {input_file.name}",
			unit="record",
		) as progress:
			for line_number, raw_line in enumerate(source, start=1):
				line = raw_line.strip()
				if not line:
					continue

				try:
					record = json.loads(line)
				except json.JSONDecodeError as exc:
					raise ValueError(
						f"Invalid JSON at {input_file.name}:{line_number}: {exc}"
					) from exc

				if not isinstance(record, dict):
					raise ValueError(
						f"Expected JSON object at {input_file.name}:{line_number}, got {type(record).__name__}"
					)

				processed_records += 1
				try:
					split_records = _split_record(record, input_file, line_number)
				except ValueError as exc:
					if _is_unsplittable_error(exc):
						skipped_unsplittable_records += 1
						progress.update(1)
						continue
					raise
				emitted_records += len(split_records)
				for split_record in split_records:
					target.write(json.dumps(split_record, ensure_ascii=False) + "\n")

				progress.update(1)

		temp_output_file.replace(output_file)
	finally:
		if temp_output_file.exists():
			temp_output_file.unlink()

	return processed_records, emitted_records, skipped_unsplittable_records


def main() -> None:
	parser = argparse.ArgumentParser(description="Split long JSONL result.text into sentence-based chunks")
	parser.add_argument(
		"--input-path",
		type=Path,
		default=DEFAULT_INPUT_PATH,
		help="Input .jsonl file or directory",
	)
	parser.add_argument(
		"--output-path",
		type=Path,
		default=None,
		help="Optional output .jsonl file or directory. Defaults to *_split path.",
	)
	args = parser.parse_args()

	input_files = _jsonl_files_from_input(args.input_path)
	output_path = args.output_path if args.output_path is not None else _default_output_path(args.input_path)

	total_processed = 0
	total_emitted = 0
	total_skipped_unsplittable = 0
	output_files: list[Path] = []

	if output_path.suffix.lower() == ".jsonl":
		if len(input_files) != 1:
			raise ValueError("When output-path is a .jsonl file, input must be a single .jsonl file")
		processed, emitted, skipped_unsplittable = _transform_file(input_files[0], output_path)
		total_processed += processed
		total_emitted += emitted
		total_skipped_unsplittable += skipped_unsplittable
		output_files.append(output_path)
	else:
		output_path.mkdir(parents=True, exist_ok=True)
		for input_file in input_files:
			output_file = output_path / _split_output_name(input_file)
			processed, emitted, skipped_unsplittable = _transform_file(input_file, output_file)
			total_processed += processed
			total_emitted += emitted
			total_skipped_unsplittable += skipped_unsplittable
			output_files.append(output_file)

	print("Sentence split summary:")
	print(f"  max_chars={MAX_CHARS}")
	print(f"  target_range={TARGET_MIN_CHARS}-{TARGET_MAX_CHARS}")
	print(f"  input_files={len(input_files)}")
	print(f"  output_files={len(output_files)}")
	print(f"  processed_records={total_processed}")
	print(f"  emitted_records={total_emitted}")
	print(f"  skipped_unsplittable_records={total_skipped_unsplittable}")


if __name__ == "__main__":
	main()
