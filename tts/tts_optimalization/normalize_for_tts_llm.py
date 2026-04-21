#!/usr/bin/env python3
"""Normalize medical JSONL text for Norwegian TTS using a DeepInfra-hosted LLM.

Expected record format:
    {"id": "...", "result": "{\"text\": \"...\"}"}

The script also handles result as a dict with a text field.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt_creation.chunked_jsonl import append_jsonl_record
from prompt_creation.generate_outputs import _resolve_api_key, generate_outputs


MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
# MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# MODEL_NAME = "microsoft/phi-4"
# MODEL_NAME = "google/gemma-4-26B-A4B-it"
# MODEL_NAME = "google/gemma-4-31B-it"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_INPUT_PATH = Path("data/outputs/tts/id_result_parts")
GENERATION_CONCURRENCY = 200
GENERATION_MAX_RETRIES = 0
GENERATION_RETRY_BACKOFF_S = 0.5
GENERATION_BATCH_SIZE = 200
GENERATION_RESPONSE_FORMAT = {"type": "json_object"}
_JSON_DECODER = json.JSONDecoder()
_INVALID_ESCAPE_PATTERN = re.compile(r"\\([^\"\\/bfnrtu])")
_FORBIDDEN_OUTPUT_PATTERN = re.compile(r"[0-9-]")


@dataclass
class _PendingNormalization:
    line_number: int
    record_id: str
    original_text: str
    result_value: object
    mode: str
    updated_record: dict
    prompt: str


@dataclass
class _LazyStatusLog:
    path: Path
    handle: Optional[object] = None
    created: bool = False

    @property
    def was_created(self) -> bool:
        return self.created

    def write_record(self, payload: dict[str, str | int]) -> None:
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", encoding="utf-8")
            self.created = True
        self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


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


def _tts_output_name(input_file: Path) -> str:
    return f"{input_file.stem}.jsonl"


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".jsonl":
        return input_path.with_name(_tts_output_name(input_path))
    return input_path.parent / f"{input_path.name}_tts_llm"


def _default_status_log_path(output_path: Path) -> Path:
    if output_path.suffix.lower() == ".jsonl":
        return output_path.with_name(f"{output_path.stem}_status_log.jsonl")
    return output_path.parent / f"{output_path.name}_status_log.jsonl"


def _record_id_for_log(record: dict, input_file: Path, line_number: int) -> str:
    raw_id = record.get("id")
    if raw_id is None:
        return f"{input_file.name}:{line_number}"
    return str(raw_id)


def _write_status_log(
    status_log_handle: Optional[_LazyStatusLog],
    *,
    input_file: Path,
    line_number: int,
    record_id: str,
    status: str,
    details: Optional[str] = None,
) -> None:
    if status_log_handle is None:
        return

    payload: dict[str, str | int] = {
        "input_file": input_file.name,
        "line_number": line_number,
        "id": record_id,
        "status": status,
    }
    if details:
        payload["details"] = details

    status_log_handle.write_record(payload)


def _count_non_empty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                count += 1
    return count


def _load_completed_output_ids(output_file: Path) -> set[str]:
    """Load IDs already present in output so interrupted runs can resume.

    Records without a string/int id are ignored for resume tracking.
    """
    if not output_file.exists() or not output_file.is_file():
        return set()

    completed_ids: set[str] = set()
    with output_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            output_id = payload.get("id")
            if isinstance(output_id, (str, int)):
                completed_ids.add(str(output_id))
    return completed_ids


def _load_system_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists() or not prompt_path.is_file():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"System prompt file is empty: {prompt_path}")
    return prompt


def _extract_output_text(response_content: str) -> str:
    """Extract normalized sentence from model JSON output.

    Expected format:
    {"output":"..."}
    """
    if not response_content.strip():
        raise ValueError("Model response was empty")

    try:
        payload = json.loads(response_content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_content, flags=re.DOTALL)
        if not match:
            raise ValueError("Model response did not contain valid JSON") from None
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("Model response JSON is not an object")

    output_text = payload.get("output")
    if not isinstance(output_text, str):
        raise ValueError("Model response JSON missing string 'output' field")

    return output_text


def _build_generation_prompt(text: str) -> str:
    return text


def _contains_forbidden_output_chars(text: str) -> bool:
    return bool(_FORBIDDEN_OUTPUT_PATTERN.search(text))


def _apply_batch_results(
    pending_batch: list[_PendingNormalization],
    generation_results: list,
    input_file: Path,
    status_log_handle: Optional[_LazyStatusLog],
    verbose: bool,
) -> tuple[int, int, set[str]]:
    """Apply model outputs to pending records.

    Returns (changed_records, failed_records, successful_ids).
    """
    changed_records = 0
    failed_records = 0
    successful_ids: set[str] = set()

    if len(generation_results) != len(pending_batch):
        missing = len(pending_batch) - len(generation_results)
        if missing > 0:
            for item in pending_batch[-missing:]:
                failed_records += 1
                _write_status_log(
                    status_log_handle,
                    input_file=input_file,
                    line_number=item.line_number,
                    record_id=item.record_id,
                    status="failed_llm_normalization",
                    details="ProviderError: Missing result for input prompt",
                )
                if verbose:
                    print(
                        f"[fail] id={item.record_id} status=failed_llm_normalization "
                        "error=ProviderError: Missing result for input prompt"
                    )

    for item, result in zip(pending_batch, generation_results):
        if result.error:
            failed_records += 1
            _write_status_log(
                status_log_handle,
                input_file=input_file,
                line_number=item.line_number,
                record_id=item.record_id,
                status="failed_llm_normalization",
                details=f"ProviderError: {result.error}",
            )
            if verbose:
                print(
                    f"[fail] id={item.record_id} status=failed_llm_normalization "
                    f"error=ProviderError: {result.error}"
                )
            continue

        try:
            normalized_text = _extract_output_text(result.content)
        except Exception as exc:
            failed_records += 1
            _write_status_log(
                status_log_handle,
                input_file=input_file,
                line_number=item.line_number,
                record_id=item.record_id,
                status="failed_llm_normalization",
                details=f"ParseError: {type(exc).__name__}: {exc}",
            )
            if verbose:
                print(
                    f"[fail] id={item.record_id} status=failed_llm_normalization "
                    f"error=ParseError: {type(exc).__name__}: {exc}"
                )
            continue

        if normalized_text != item.original_text:
            changed_records += 1
        item.updated_record["result"] = _write_text_back(item.result_value, item.mode, normalized_text)
        successful_ids.add(item.record_id)

    return changed_records, failed_records, successful_ids


def _flush_pending_batch(
    *,
    pending_normalizations: list[_PendingNormalization],
    input_file: Path,
    api_key: str | None,
    concurrency: int,
    batch_size: int,
    system_prompt: str,
    reasoning_effort: str,
    status_log_handle: Optional[_LazyStatusLog],
    verbose: bool,
) -> tuple[int, int, set[str]]:
    """Normalize pending records in explicit chunk loops.

    Returns (changed_records, failed_records, successful_ids).
    """
    if not pending_normalizations:
        return 0, 0, set()

    changed_records = 0
    failed_records = 0
    successful_ids: set[str] = set()

    for start_index in range(0, len(pending_normalizations), batch_size):
        batch = pending_normalizations[start_index : start_index + batch_size]
        prompts = [item.prompt for item in batch]

        resolved_key = _resolve_api_key(api_key)
        try:
            generation_results = generate_outputs(
                prompts=prompts,
                system_prompt=system_prompt,
                api_key=resolved_key,
                concurrency=concurrency,
                max_retries=GENERATION_MAX_RETRIES,
                retry_backoff_s=GENERATION_RETRY_BACKOFF_S,
                batch_size=len(prompts),
                reasoning_effort=reasoning_effort,
                model_name=MODEL_NAME,
                base_url=DEEPINFRA_BASE_URL,
                response_format=GENERATION_RESPONSE_FORMAT,
            )
        except Exception as exc:
            for item in batch:
                failed_records += 1
                _write_status_log(
                    status_log_handle,
                    input_file=input_file,
                    line_number=item.line_number,
                    record_id=item.record_id,
                    status="failed_llm_normalization",
                    details=f"BatchGenerationError: {type(exc).__name__}: {exc}",
                )
                if verbose:
                    print(
                        f"[fail] id={item.record_id} status=failed_llm_normalization "
                        f"error=BatchGenerationError: {type(exc).__name__}: {exc}"
                    )
            continue

        batch_changed, batch_failed, batch_successful_ids = _apply_batch_results(
            pending_batch=batch,
            generation_results=generation_results,
            input_file=input_file,
            status_log_handle=status_log_handle,
            verbose=verbose,
        )
        changed_records += batch_changed
        failed_records += batch_failed
        successful_ids.update(batch_successful_ids)

    return changed_records, failed_records, successful_ids


def _transform_file(
    input_file: Path,
    output_file: Path,
    api_key: str | None,
    concurrency: int,
    batch_size: int,
    system_prompt: str,
    reasoning_effort: str,
    status_log_handle: Optional[_LazyStatusLog],
    verbose: bool,
    regenerate_failed: bool,
) -> tuple[int, int, int, int, int]:
    """Transform one JSONL file.

    Returns (processed_records, changed_records, skipped_records, failed_records, resumed_records).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = _load_completed_output_ids(output_file)

    processed_records = 0
    changed_records = 0
    skipped_records = 0
    failed_records = 0
    resumed_records = 0
    pending_normalizations: list[_PendingNormalization] = []
    total_non_empty_lines = _count_non_empty_lines(input_file)

    if regenerate_failed:
        in_place_overwrite = input_file.resolve() == output_file.resolve()
        temp_output_file: Path | None = None
        if in_place_overwrite:
            temp_output_file = output_file.with_name(f"{output_file.stem}.regenerate.tmp.jsonl")
            if temp_output_file.exists():
                temp_output_file.unlink()
            target_path = temp_output_file
        else:
            target_path = output_file

        try:
            with input_file.open("r", encoding="utf-8") as source, target_path.open(
                "w", encoding="utf-8"
            ) as target, tqdm(
                total=total_non_empty_lines,
                desc=f"Regenerating {input_file.name}",
                unit="record",
            ) as progress:
                for line_number, raw_line in enumerate(source, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_records += 1
                        if verbose:
                            print(f"[skip] {input_file.name}:{line_number} status=skipped_invalid_json")
                        target.write(raw_line)
                        progress.update(1)
                        continue

                    if not isinstance(record, dict):
                        skipped_records += 1
                        if verbose:
                            print(
                                f"[skip] {input_file.name}:{line_number} status=skipped_non_object_record"
                            )
                        target.write(json.dumps(record, ensure_ascii=False) + "\n")
                        progress.update(1)
                        continue

                    processed_records += 1
                    result_value = record.get("result")
                    text, mode = _read_text_from_result(result_value)
                    if text is None:
                        skipped_records += 1
                        if verbose:
                            record_id = _record_id_for_log(record, input_file, line_number)
                            print(f"[skip] id={record_id} status=skipped_unreadable_result")
                        target.write(json.dumps(record, ensure_ascii=False) + "\n")
                        progress.update(1)
                        continue

                    if not _contains_forbidden_output_chars(text):
                        target.write(json.dumps(record, ensure_ascii=False) + "\n")
                        progress.update(1)
                        continue

                    record_id = _record_id_for_log(record, input_file, line_number)
                    pending_normalizations.append(
                        _PendingNormalization(
                            line_number=line_number,
                            record_id=record_id,
                            original_text=text,
                            result_value=result_value,
                            mode=mode,
                            updated_record=dict(record),
                            prompt=_build_generation_prompt(text),
                        )
                    )
                    progress.update(1)

                    if len(pending_normalizations) >= batch_size:
                        batch_changed, batch_failed, _batch_successful_ids = _flush_pending_batch(
                            pending_normalizations=pending_normalizations,
                            input_file=input_file,
                            api_key=api_key,
                            concurrency=concurrency,
                            batch_size=batch_size,
                            system_prompt=system_prompt,
                            reasoning_effort=reasoning_effort,
                            status_log_handle=status_log_handle,
                            verbose=verbose,
                        )
                        changed_records += batch_changed
                        failed_records += batch_failed
                        for item in pending_normalizations:
                            target.write(json.dumps(item.updated_record, ensure_ascii=False) + "\n")
                        pending_normalizations = []

                if pending_normalizations:
                    batch_changed, batch_failed, _batch_successful_ids = _flush_pending_batch(
                        pending_normalizations=pending_normalizations,
                        input_file=input_file,
                        api_key=api_key,
                        concurrency=concurrency,
                        batch_size=batch_size,
                        system_prompt=system_prompt,
                        reasoning_effort=reasoning_effort,
                        status_log_handle=status_log_handle,
                        verbose=verbose,
                    )
                    changed_records += batch_changed
                    failed_records += batch_failed
                    for item in pending_normalizations:
                        target.write(json.dumps(item.updated_record, ensure_ascii=False) + "\n")

            if in_place_overwrite and temp_output_file is not None:
                temp_output_file.replace(output_file)
        finally:
            if temp_output_file is not None and temp_output_file.exists():
                temp_output_file.unlink()

        return processed_records, changed_records, skipped_records, failed_records, resumed_records

    with input_file.open("r", encoding="utf-8") as source, tqdm(
        total=total_non_empty_lines,
        desc=f"Normalizing {input_file.name}",
        unit="record",
    ) as progress:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_records += 1
                if verbose:
                    print(f"[skip] {input_file.name}:{line_number} status=skipped_invalid_json")
                progress.update(1)
                continue

            if not isinstance(record, dict):
                skipped_records += 1
                if verbose:
                    print(f"[skip] {input_file.name}:{line_number} status=skipped_non_object_record")
                progress.update(1)
                continue

            processed_records += 1
            record_id = _record_id_for_log(record, input_file, line_number)
            if record_id in completed_ids:
                resumed_records += 1
                progress.update(1)
                continue

            result_value = record.get("result")
            text, mode = _read_text_from_result(result_value)

            if text is None:
                skipped_records += 1
                if verbose:
                    print(f"[skip] id={record_id} status=skipped_unreadable_result")
                progress.update(1)
                continue

            updated_record = dict(record)
            pending_normalizations.append(
                _PendingNormalization(
                    line_number=line_number,
                    record_id=record_id,
                    original_text=text,
                    result_value=result_value,
                    mode=mode,
                    updated_record=updated_record,
                    prompt=_build_generation_prompt(text),
                )
            )
            progress.update(1)

            if len(pending_normalizations) >= batch_size:
                batch_changed, batch_failed, batch_successful_ids = _flush_pending_batch(
                    pending_normalizations=pending_normalizations,
                    input_file=input_file,
                    api_key=api_key,
                    concurrency=concurrency,
                    batch_size=batch_size,
                    system_prompt=system_prompt,
                    reasoning_effort=reasoning_effort,
                    status_log_handle=status_log_handle,
                    verbose=verbose,
                )
                changed_records += batch_changed
                failed_records += batch_failed
                for item in pending_normalizations:
                    if item.record_id in batch_successful_ids:
                        append_jsonl_record(output_file, item.updated_record)
                        completed_ids.add(item.record_id)
                pending_normalizations = []

        if pending_normalizations:
            batch_changed, batch_failed, batch_successful_ids = _flush_pending_batch(
                pending_normalizations=pending_normalizations,
                input_file=input_file,
                api_key=api_key,
                concurrency=concurrency,
                batch_size=batch_size,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
                status_log_handle=status_log_handle,
                verbose=verbose,
            )
            changed_records += batch_changed
            failed_records += batch_failed
            for item in pending_normalizations:
                if item.record_id in batch_successful_ids:
                    append_jsonl_record(output_file, item.updated_record)
                    completed_ids.add(item.record_id)

    return processed_records, changed_records, skipped_records, failed_records, resumed_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize JSONL text for Norwegian TTS using an LLM")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        required=True,
        help="Input .jsonl file or directory (default: data/outputs/tts/id_result_parts)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        required=True,
        help="Output file/directory.",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=Path,
        default=Path("tts_optimization/tts_system_prompt.txt"),
        help="Path to system prompt instructions used for normalization.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=GENERATION_CONCURRENCY,
        help="Maximum number of parallel model requests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=GENERATION_BATCH_SIZE,
        help="Number of prompts per batch for async generation.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="none",
        help="Reasoning effort.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional DeepInfra API key. If omitted, DEEPINFRA_API_KEY is used.",
    )
    parser.add_argument(
        "--status-log-path",
        type=Path,
        default=None,
        help=(
            "Path to JSONL status log for failed records. "
            "Defaults to sibling *_status_log.jsonl next to output path."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-record skip/failure details while processing.",
    )
    parser.add_argument(
        "--regenerate-failed",
        action="store_true",
        help=(
            "Regenerate only records whose current result.text contains numbers or hyphens. "
            "When input and output paths are the same, files are safely overwritten in place."
        ),
    )
    args = parser.parse_args()
    system_prompt = _load_system_prompt(args.system_prompt_path)

    input_files = _jsonl_files_from_input(args.input_path)
    output_path = args.output_path if args.output_path is not None else _default_output_path(args.input_path)
    status_log_path = args.status_log_path if args.status_log_path is not None else _default_status_log_path(output_path)
    status_log_handle = _LazyStatusLog(status_log_path)

    total_processed = 0
    total_changed = 0
    total_skipped = 0
    total_failed = 0
    total_resumed = 0
    output_files: list[Path] = []

    try:
        if output_path.suffix.lower() == ".jsonl":
            if len(input_files) != 1:
                raise ValueError("When output-path is a .jsonl file, input must be a single .jsonl file")
            processed, changed, skipped, failed, resumed = _transform_file(
                input_file=input_files[0],
                output_file=output_path,
                api_key=args.api_key,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                system_prompt=system_prompt,
                reasoning_effort=args.reasoning_effort,
                status_log_handle=status_log_handle,
                verbose=args.verbose,
                regenerate_failed=args.regenerate_failed,
            )
            total_processed += processed
            total_changed += changed
            total_skipped += skipped
            total_failed += failed
            total_resumed += resumed
            output_files.append(output_path)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            for input_file in input_files:
                output_file = output_path / _tts_output_name(input_file)
                processed, changed, skipped, failed, resumed = _transform_file(
                    input_file=input_file,
                    output_file=output_file,
                    api_key=args.api_key,
                    concurrency=args.concurrency,
                    batch_size=args.batch_size,
                    system_prompt=system_prompt,
                    reasoning_effort=args.reasoning_effort,
                    status_log_handle=status_log_handle,
                    verbose=args.verbose,
                    regenerate_failed=args.regenerate_failed,
                )
                total_processed += processed
                total_changed += changed
                total_skipped += skipped
                total_failed += failed
                total_resumed += resumed
                output_files.append(output_file)
    finally:
        status_log_handle.close()

    status_log_summary = status_log_path if status_log_handle.was_created else "not_written"

    print("TTS LLM normalization summary:")
    print(f"  model={MODEL_NAME}")
    print(f"  input_files={len(input_files)}")
    print(f"  output_files={len(output_files)}")
    print(f"  status_log={status_log_summary}")
    print(f"  processed_records={total_processed}")
    print(f"  changed_records={total_changed}")
    print(f"  skipped_records={total_skipped}")
    print(f"  failed_records={total_failed}")
    print(f"  resumed_records={total_resumed}")
    if args.regenerate_failed:
        print("Regenerate failed summary:")
        print(f"  outputs_changed={total_changed}")


if __name__ == "__main__":
    main()
