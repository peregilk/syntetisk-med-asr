"""Generate synthetic responses from prompts using DeepSeek via DeepInfra."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm

MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic responses from prompt JSONL files."
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Path to JSONL prompt file (id + prompt).",
    )
    parser.add_argument(
        "--output-file",
        default="data/outputs/output.jsonl",
        help="Path to JSONL output file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing entries with the same id.",
    )
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file)
    output_file = Path(args.output_file)

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing DEEPINFRA_API_KEY environment variable")

    client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)

    prompts = load_prompts(prompt_file)
    existing = load_existing_output(output_file)
    existing_index = {entry.get("id"): idx for idx, entry in enumerate(existing)}

    for prompt in tqdm(prompts, desc="Generating", unit="prompt"):
        existing_position = existing_index.get(prompt["id"])
        if existing_position is not None and not args.overwrite:
            continue
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt["prompt"]}],
        )
        content = response.choices[0].message.content or ""
        entry = {
            "id": prompt["id"],
            "prompt": prompt["prompt"],
            "response": content,
        }

        if existing_position is None:
            existing_index[prompt["id"]] = len(existing)
            existing.append(entry)
        else:
            existing[existing_position] = entry

        if args.overwrite:
            write_output(output_file, existing)
        else:
            if existing_position is None:
                append_output(output_file, entry)


if __name__ == "__main__":
    main()
