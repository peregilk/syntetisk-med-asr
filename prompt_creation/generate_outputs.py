"""LLM output generation using DeepInfra (DeepSeek)."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional

from openai import AsyncOpenAI, OpenAI

MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


@dataclass
class GenerationResult:
    """Result for a single prompt generation."""

    prompt: str
    content: str
    error: Optional[str] = None


def _resolve_api_key(api_key: Optional[str]) -> str:
    resolved_key = api_key or os.environ.get("DEEPINFRA_API_KEY")
    if not resolved_key:
        raise EnvironmentError("Missing DEEPINFRA_API_KEY environment variable")
    return resolved_key


def generate_output(prompt: str, api_key: Optional[str] = None) -> str:
    """Generate a response for a single prompt.

    Args:
        prompt: The prompt text to send to the model.
        api_key: Optional DeepInfra API key. If not provided, uses
            DEEPINFRA_API_KEY environment variable.

    Returns:
        The model response content as a string (may be empty).
    """
    resolved_key = _resolve_api_key(api_key)
    client = OpenAI(api_key=resolved_key, base_url=DEEPINFRA_BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        reasoning_effort="none",  # low, medium, high, or none
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


async def _generate_one(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    retry_backoff_s: float,
) -> GenerationResult:
    """Generate a response for a single prompt with retries."""
    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    reasoning_effort="none",  # low, medium, high, or none
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content or ""
                return GenerationResult(prompt=prompt, content=content)
            except Exception as error:  # noqa: BLE001 - capture provider errors
                if attempt >= max_retries:
                    return GenerationResult(prompt=prompt, content="", error=str(error))
                await asyncio.sleep(retry_backoff_s * (2**attempt))
        return GenerationResult(prompt=prompt, content="", error="Unknown error")


async def generate_outputs_async(
    prompts: Iterable[str],
    api_key: Optional[str] = None,
    concurrency: int = 50,
    max_retries: int = 2,
    retry_backoff_s: float = 0.5,
    batch_size: int = 100,
    on_batch_complete: Optional[Callable[[list[GenerationResult]], Awaitable[None]]] = None,
) -> list[GenerationResult]:
    """Generate outputs for many prompts concurrently.

    The prompts are processed in batches so you can "touch base" between
    batches via the optional `on_batch_complete` callback.

    Args:
        prompts: Iterable of prompt strings.
        api_key: Optional DeepInfra API key. If not provided, uses
            DEEPINFRA_API_KEY environment variable.
        concurrency: Maximum number of in-flight requests.
        max_retries: Number of retry attempts per prompt.
        retry_backoff_s: Base backoff (seconds) for retries.
        batch_size: Number of prompts per batch before touching base.
        on_batch_complete: Async callback called after each batch completes.

    Returns:
        List of GenerationResult in input order.
    """
    resolved_key = _resolve_api_key(api_key)
    semaphore = asyncio.Semaphore(concurrency)
    results: list[GenerationResult] = []
    prompt_list = list(prompts)

    async with AsyncOpenAI(api_key=resolved_key, base_url=DEEPINFRA_BASE_URL) as client:
        for start_index in range(0, len(prompt_list), batch_size):
            batch_prompts = prompt_list[start_index : start_index + batch_size]
            tasks = [
                _generate_one(
                    client=client,
                    prompt=prompt,
                    semaphore=semaphore,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                )
                for prompt in batch_prompts
            ]
            batch_results = list(await asyncio.gather(*tasks))
            results.extend(batch_results)
            if on_batch_complete is not None:
                await on_batch_complete(batch_results)

    return results


def generate_outputs(
    prompts: Iterable[str],
    api_key: Optional[str] = None,
    concurrency: int = 50,
    max_retries: int = 2,
    retry_backoff_s: float = 0.5,
    batch_size: int = 100,
    on_batch_complete: Optional[Callable[[list[GenerationResult]], Awaitable[None]]] = None,
) -> list[GenerationResult]:
    """Synchronous wrapper for generate_outputs_async()."""
    return asyncio.run(
        generate_outputs_async(
            prompts=prompts,
            api_key=api_key,
            concurrency=concurrency,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            batch_size=batch_size,
            on_batch_complete=on_batch_complete,
        )
    )
