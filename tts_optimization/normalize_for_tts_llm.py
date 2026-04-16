#!/usr/bin/env python3
"""Normalize medical JSONL text for Norwegian TTS using a DeepInfra-hosted LLM.

Expected record format:
    {"id": "...", "result": "{\"text\": \"...\"}"}

The script also handles result as a dict with a text field.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, TextIO

from openai import OpenAI


# MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_INPUT_PATH = Path("data/outputs/tts/id_result_parts")


def _resolve_api_key(api_key: Optional[str]) -> str:
    resolved_key = api_key or os.environ.get("DEEPINFRA_API_KEY")
    if not resolved_key:
        raise EnvironmentError("Missing DEEPINFRA_API_KEY environment variable")
    return resolved_key


def _read_text_from_result(result_value: object) -> tuple[str | None, str]:
    """Extract text from result payload.

    Returns (text, mode) where mode is "string" or "dict" for writeback strategy.
    """
    if isinstance(result_value, str):
        try:
            parsed = json.loads(result_value)
        except json.JSONDecodeError:
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
    return f"{input_file.stem}_tts_llm.jsonl"


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
    status_log_handle: Optional[TextIO],
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

    status_log_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
    {"stil":"tts_optimalisert","input":"...","output":"..."}
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


def _normalize_text_with_llm(client: OpenAI, system_prompt: str, text: str, reasoning_effort: str) -> str:
    user_prompt = f"{system_prompt}\n{text}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        reasoning_effort=reasoning_effort,
        messages=[{"role": "user", "content": user_prompt}],
    )
    content = response.choices[0].message.content or ""
    return _extract_output_text(content)


def _transform_file(
    input_file: Path,
    output_file: Path,
    client: OpenAI,
    system_prompt: str,
    reasoning_effort: str,
    status_log_handle: Optional[TextIO],
    verbose: bool,
) -> tuple[int, int, int, int]:
    """Transform one JSONL file.

    Returns (processed_records, changed_records, skipped_records, failed_records).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processed_records = 0
    changed_records = 0
    skipped_records = 0
    failed_records = 0

    with input_file.open("r", encoding="utf-8") as source, output_file.open("w", encoding="utf-8") as target:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_records += 1
                _write_status_log(
                    status_log_handle,
                    input_file=input_file,
                    line_number=line_number,
                    record_id=f"{input_file.name}:{line_number}",
                    status="skipped_invalid_json",
                    details="Line is not valid JSON",
                )
                if verbose:
                    print(f"[skip] {input_file.name}:{line_number} status=skipped_invalid_json")
                continue

            if not isinstance(record, dict):
                skipped_records += 1
                _write_status_log(
                    status_log_handle,
                    input_file=input_file,
                    line_number=line_number,
                    record_id=f"{input_file.name}:{line_number}",
                    status="skipped_non_object_record",
                    details="Top-level JSON value is not an object",
                )
                if verbose:
                    print(f"[skip] {input_file.name}:{line_number} status=skipped_non_object_record")
                continue

            processed_records += 1
            record_id = _record_id_for_log(record, input_file, line_number)
            result_value = record.get("result")
            text, mode = _read_text_from_result(result_value)

            if text is None:
                skipped_records += 1
                _write_status_log(
                    status_log_handle,
                    input_file=input_file,
                    line_number=line_number,
                    record_id=record_id,
                    status="skipped_unreadable_result",
                    details="Missing or invalid result.text payload",
                )
                if verbose:
                    print(f"[skip] id={record_id} status=skipped_unreadable_result")
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            updated_record = dict(record)
            try:
                normalized_text = _normalize_text_with_llm(
                    client=client,
                    system_prompt=system_prompt,
                    text=text,
                    reasoning_effort=reasoning_effort,
                )
                if normalized_text != text:
                    changed_records += 1
                updated_record["result"] = _write_text_back(result_value, mode, normalized_text)
            except Exception as exc:
                # Preserve the original record so downstream scripts can still consume the file.
                failed_records += 1
                _write_status_log(
                    status_log_handle,
                    input_file=input_file,
                    line_number=line_number,
                    record_id=record_id,
                    status="failed_llm_normalization",
                    details=f"{type(exc).__name__}: {exc}",
                )
                if verbose:
                    print(
                        f"[fail] id={record_id} status=failed_llm_normalization "
                        f"error={type(exc).__name__}: {exc}"
                    )

            target.write(json.dumps(updated_record, ensure_ascii=False) + "\n")

    return processed_records, changed_records, skipped_records, failed_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize JSONL text for Norwegian TTS using an LLM")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input .jsonl file or directory (default: data/outputs/tts/id_result_parts)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output file/directory. Defaults to matching _tts_llm suffix path.",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=Path,
        default=Path("tts_optimization/tts_system_prompt.txt"),
        help="Path to system prompt instructions used for normalization.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="none",
        help="DeepSeek reasoning effort.",
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
            "Path to JSONL status log for skipped/failed records. "
            "Defaults to sibling *_status_log.jsonl next to output path."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-record skip/failure details while processing.",
    )
    args = parser.parse_args()

    resolved_key = _resolve_api_key(args.api_key)
    client = OpenAI(api_key=resolved_key, base_url=DEEPINFRA_BASE_URL)
    system_prompt = _load_system_prompt(args.system_prompt_path)

    input_files = _jsonl_files_from_input(args.input_path)
    output_path = args.output_path if args.output_path is not None else _default_output_path(args.input_path)
    status_log_path = args.status_log_path if args.status_log_path is not None else _default_status_log_path(output_path)

    total_processed = 0
    total_changed = 0
    total_skipped = 0
    total_failed = 0
    output_files: list[Path] = []

    status_log_path.parent.mkdir(parents=True, exist_ok=True)
    with status_log_path.open("w", encoding="utf-8") as status_log_handle:
        if output_path.suffix.lower() == ".jsonl":
            if len(input_files) != 1:
                raise ValueError("When output-path is a .jsonl file, input must be a single .jsonl file")
            processed, changed, skipped, failed = _transform_file(
                input_file=input_files[0],
                output_file=output_path,
                client=client,
                system_prompt=system_prompt,
                reasoning_effort=args.reasoning_effort,
                status_log_handle=status_log_handle,
                verbose=args.verbose,
            )
            total_processed += processed
            total_changed += changed
            total_skipped += skipped
            total_failed += failed
            output_files.append(output_path)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            for input_file in input_files:
                output_file = output_path / _tts_output_name(input_file)
                processed, changed, skipped, failed = _transform_file(
                    input_file=input_file,
                    output_file=output_file,
                    client=client,
                    system_prompt=system_prompt,
                    reasoning_effort=args.reasoning_effort,
                    status_log_handle=status_log_handle,
                    verbose=args.verbose,
                )
                total_processed += processed
                total_changed += changed
                total_skipped += skipped
                total_failed += failed
                output_files.append(output_file)

    print("TTS LLM normalization summary:")
    print(f"  model={MODEL_NAME}")
    print(f"  input_files={len(input_files)}")
    print(f"  output_files={len(output_files)}")
    print(f"  status_log={status_log_path}")
    print(f"  processed_records={total_processed}")
    print(f"  changed_records={total_changed}")
    print(f"  skipped_records={total_skipped}")
    print(f"  failed_records={total_failed}")


if __name__ == "__main__":
    main()
