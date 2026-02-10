"""LLM output generation using DeepInfra (DeepSeek)."""

import os
from typing import Optional

from openai import OpenAI

MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


def generate_output(prompt: str, api_key: Optional[str] = None) -> str:
    """Generate a response for a single prompt.

    Args:
        prompt: The prompt text to send to the model.
        api_key: Optional DeepInfra API key. If not provided, uses
            DEEPINFRA_API_KEY environment variable.

    Returns:
        The model response content as a string (may be empty).
    """
    resolved_key = api_key or os.environ.get("DEEPINFRA_API_KEY")
    if not resolved_key:
        raise EnvironmentError("Missing DEEPINFRA_API_KEY environment variable")

    client = OpenAI(api_key=resolved_key, base_url=DEEPINFRA_BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        reasoning_effort="none",  # low, medium, high, or none
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""
