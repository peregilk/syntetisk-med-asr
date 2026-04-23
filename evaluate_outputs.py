"""Evaluate LLM outputs with DeepSeek via DeepInfra."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from prompt_creation.generate_outputs import DEEPINFRA_BASE_URL
from prompt_creation.generate_outputs import MODEL_NAME
from prompt_creation.generate_outputs import _resolve_api_key


@dataclass
class EvaluationResult:
    """Result for a single evaluation request."""

    score: Optional[int]
    error: Optional[str] = None


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_text_from_response(response: str) -> str:
    """Extract the text content to evaluate from a response payload.

    The response may already be plain text, or JSON like {"text": "..."}.
    """
    response = response.strip()
    if not response:
        return ""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return response
    if isinstance(payload, dict) and "text" in payload:
        text_value = payload.get("text")
        return text_value if isinstance(text_value, str) else response
    return response


def _render_user_prompt(template: str, text: str) -> str:
    if "${text}" in template:
        return template.replace("${text}", text)
    if "{text}" in template:
        try:
            return template.format(text=text)
        except Exception:
            return template + "\n\n" + text
    return template + "\n\n" + text


def _parse_score(raw_output: str) -> int:
    """Parse a score integer from the model output."""
    raw_output = raw_output.strip()
    try:
        payload = json.loads(raw_output)
        if isinstance(payload, dict) and "score" in payload:
            value = payload["score"]
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                return int(value.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\b(10|[1-9])\b", raw_output)
    if not match:
        raise ValueError(f"Could not parse score from: {raw_output}")
    return int(match.group(1))


def evaluate_text(
    client: OpenAI, system_prompt: str, user_template: str, text: str
) -> int:
    user_prompt = _render_user_prompt(user_template, text)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return _parse_score(content)


async def _evaluate_one(
    client: AsyncOpenAI,
    system_prompt: str,
    user_template: str,
    text: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    retry_backoff_s: float,
    reasoning_effort: str,
) -> EvaluationResult:
    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                user_prompt = _render_user_prompt(user_template, text)
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    reasoning_effort=reasoning_effort,  # low, medium, high, or none
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content or ""
                score = _parse_score(content)
                return EvaluationResult(score=score)
            except Exception as error:  # noqa: BLE001 - capture provider errors
                if attempt >= max_retries:
                    return EvaluationResult(score=None, error=str(error))
                await asyncio.sleep(retry_backoff_s * (2**attempt))
        return EvaluationResult(score=None, error="Unknown error")


async def evaluate_texts_async(
    texts: Iterable[str],
    system_prompt: str,
    user_template: str,
    api_key: Optional[str] = None,
    concurrency: int = 50,
    max_retries: int = 2,
    retry_backoff_s: float = 0.5,
    batch_size: int = 100,
    reasoning_effort: str = "low",
    on_batch_complete: Optional[
        Callable[[list[EvaluationResult]], Awaitable[None]]
    ] = None,
) -> list[EvaluationResult]:
    resolved_key = _resolve_api_key(api_key)
    semaphore = asyncio.Semaphore(concurrency)
    results: list[EvaluationResult] = []
    text_list = list(texts)

    async with AsyncOpenAI(api_key=resolved_key, base_url=DEEPINFRA_BASE_URL) as client:
        for start_index in range(0, len(text_list), batch_size):
            batch_texts = text_list[start_index : start_index + batch_size]
            tasks = [
                _evaluate_one(
                    client=client,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    text=text,
                    semaphore=semaphore,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                    reasoning_effort=reasoning_effort,
                )
                for text in batch_texts
            ]
            batch_results = list(await asyncio.gather(*tasks))
            results.extend(batch_results)
            if on_batch_complete is not None:
                await on_batch_complete(batch_results)

    return results


def evaluate_texts(
    texts: Iterable[str],
    system_prompt: str,
    user_template: str,
    api_key: Optional[str] = None,
    concurrency: int = 50,
    max_retries: int = 2,
    retry_backoff_s: float = 0.5,
    batch_size: int = 100,
    reasoning_effort: str = "low",
    on_batch_complete: Optional[
        Callable[[list[EvaluationResult]], Awaitable[None]]
    ] = None,
) -> list[EvaluationResult]:
    return asyncio.run(
        evaluate_texts_async(
            texts=texts,
            system_prompt=system_prompt,
            user_template=user_template,
            api_key=api_key,
            concurrency=concurrency,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            batch_size=batch_size,
            reasoning_effort=reasoning_effort,
            on_batch_complete=on_batch_complete,
        )
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSON object on line {line_number} in {path}")
            entries.append(payload)
    return entries


def _write_jsonl(path: Path, entries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate JSONL outputs using DeepSeek via DeepInfra."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        required=True,
        help="Path to JSONL outputs file to score (updated in place).",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=Path("templates/evaluation_system_prompt.txt"),
        help="Path to evaluation system prompt.",
    )
    parser.add_argument(
        "--user-prompt",
        type=Path,
        default=Path("templates/evaluation_user_prompt.txt"),
        help="Path to evaluation user prompt template.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional DeepInfra API key (otherwise uses DEEPINFRA_API_KEY).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Maximum number of parallel evaluation requests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of evaluations per batch.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of retries per evaluation on transient errors.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=0.5,
        help="Base backoff in seconds for retries.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high", "none"],
        help="Reasoning effort for evaluation (low, medium, high, or none).",
    )

    args = parser.parse_args()

    api_key = _resolve_api_key(args.api_key)
    system_prompt = _read_text_file(args.system_prompt)
    user_template = _read_text_file(args.user_prompt)

    entries = _read_jsonl(args.outputs)
    work_items: list[tuple[int, str]] = []
    for index, entry in enumerate(entries):
        response_text = entry.get("response", "")
        if not isinstance(response_text, str) or not response_text.strip():
            entry["score_error"] = "Missing or empty response"
            continue
        text_to_evaluate = _extract_text_from_response(response_text)
        work_items.append((index, text_to_evaluate))

    progress = tqdm(total=len(work_items), desc="Scoring", unit="item")
    try:
        for start_index in range(0, len(work_items), args.batch_size):
            batch = work_items[start_index : start_index + args.batch_size]
            batch_indices = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]

            batch_results = evaluate_texts(
                texts=batch_texts,
                system_prompt=system_prompt,
                user_template=user_template,
                api_key=api_key,
                concurrency=args.concurrency,
                batch_size=len(batch_texts),
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff_s,
                reasoning_effort=args.reasoning_effort,
            )

            for entry_index, result in zip(batch_indices, batch_results):
                entry = entries[entry_index]
                if result.error:
                    entry["score_error"] = result.error
                    entry.pop("score", None)
                else:
                    entry["score"] = result.score
                    entry.pop("score_error", None)

            _write_jsonl(args.outputs, entries)
            progress.update(len(batch_results))
    finally:
        progress.close()


if __name__ == "__main__":
    main()
