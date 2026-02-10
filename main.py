"""Orchestrate prompt generation and LLM outputs for synthetic data."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from prompt_creation.generate_outputs import generate_outputs
from prompt_creation.prompts import _generate_prompts
from prompt_creation.prompts import _init_plan
from prompt_creation.prompts import _read_jsonl
from prompt_creation.prompts import _update_plan


def load_prompts(prompt_file: Path) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    with prompt_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {prompt_file}"
                ) from exc
            if "id" not in payload or "prompt" not in payload:
                raise ValueError(
                    f"Missing 'id' or 'prompt' on line {line_number} in {prompt_file}"
                )
            prompts.append({"id": payload["id"], "prompt": payload["prompt"]})
    return prompts


def load_existing_output(output_file: Path) -> List[Dict[str, str]]:
    if not output_file.exists():
        return []
    existing: List[Dict[str, str]] = []
    with output_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {output_file}"
                ) from exc
            if "id" not in payload:
                raise ValueError(
                    f"Missing 'id' on line {line_number} in {output_file}"
                )
            existing.append(payload)
    return existing


def write_output(output_file: Path, entries: List[Dict[str, str]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_output(output_file: Path, entry: Dict[str, str]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _plan_complete(plan_file: Path) -> bool:
    plan_entries = _read_jsonl(plan_file)
    if not plan_entries:
        return False
    for entry in plan_entries:
        remaining = entry.get("target_remaining", 0)
        if isinstance(remaining, int) and remaining > 0:
            return False
    return True


def _generate_outputs(
    prompt_file: Path,
    output_file: Path,
    overwrite: bool,
    concurrency: int,
    batch_size: int,
    max_retries: int,
    retry_backoff_s: float,
) -> None:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    prompts = load_prompts(prompt_file)
    existing = load_existing_output(output_file)
    existing_index = {entry.get("id"): idx for idx, entry in enumerate(existing)}

    if not overwrite:
        prompts = [prompt for prompt in prompts if prompt["id"] not in existing_index]

    prompt_texts = [prompt["prompt"] for prompt in prompts]
    prompt_ids = [prompt["id"] for prompt in prompts]

    results = generate_outputs(
        prompt_texts,
        concurrency=concurrency,
        batch_size=batch_size,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )

    for prompt_id, prompt_text, result in tqdm(
        zip(prompt_ids, prompt_texts, results),
        desc="Writing",
        unit="prompt",
        total=len(results),
    ):
        existing_position = existing_index.get(prompt_id)
        entry = {
            "id": prompt_id,
            "prompt": prompt_text,
            "response": result.content,
        }
        if result.error:
            entry["error"] = result.error

        if existing_position is None:
            existing_index[prompt_id] = len(existing)
            existing.append(entry)
        else:
            existing[existing_position] = entry

        if overwrite:
            write_output(output_file, existing)
        else:
            if existing_position is None:
                append_output(output_file, entry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prompts and LLM outputs until target coverage is met."
    )
    parser.add_argument(
        "--snomed-file",
        type=Path,
        default=Path("data/preprocessed/snomed.jsonl"),
        help="Path to preprocessed SNOMED terms.",
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=Path("data/terms_to_use.jsonl"),
        help="Path to the term usage plan JSONL.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
        help="Target total usage per term.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("templates/a.txt"),
        help="Prompt template file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("prompts/generated_prompts.jsonl"),
        help="Prompt JSONL output file.",
    )
    parser.add_argument(
        "--optional-count",
        type=int,
        default=10,
        help="Number of optional terms per prompt.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/outputs/output.jsonl"),
        help="LLM output JSONL file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing entries with the same id.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Maximum number of parallel model requests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of prompts per batch for async generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of retries per prompt on transient errors.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=0.5,
        help="Base backoff in seconds for retries.",
    )
    parser.add_argument(
        "--init-plan",
        action="store_true",
        help="Rebuild the plan from the SNOMED file before looping.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Safety limit for loop iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for prompt generation.",
    )
    args = parser.parse_args()

    if args.init_plan or not args.plan_file.exists():
        _init_plan(args.snomed_file, args.plan_file, args.target_count)

    if not args.overwrite and args.output_file.exists():
        _update_plan(args.plan_file, args.output_file, accumulate=False)

    for iteration in range(1, args.max_iterations + 1):
        if _plan_complete(args.plan_file):
            break
        _generate_prompts(
            args.plan_file,
            args.snomed_file,
            args.template,
            args.prompt_file,
            args.optional_count,
            args.seed,
        )
        _generate_outputs(
            args.prompt_file,
            args.output_file,
            args.overwrite,
            args.concurrency,
            args.batch_size,
            args.max_retries,
            args.retry_backoff_s,
        )
        _update_plan(args.plan_file, args.output_file, accumulate=False)


if __name__ == "__main__":
    main()
