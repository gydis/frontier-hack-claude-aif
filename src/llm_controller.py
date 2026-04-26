"""
LLM Controller — uses Anthropic Claude to decide difficulty adjustments.

Takes discretized labels + current difficulty + history and returns
a parsed response with next difficulty and reasoning text.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TARGET = "keep player performance in the 'normal' band"

_PROMPT_PATH = Path(__file__).parent / "prompts" / "difficulty_controller.txt"


def _load_prompt(target: str) -> str:
    """Load and format the system prompt template."""
    template = _PROMPT_PATH.read_text()
    return template.replace("{target}", target)


def _parse_response(text: str) -> dict | None:
    """
    Extract JSON from LLM response.
    Handles raw JSON or markdown code blocks.
    Returns None on parse failure.
    """
    text = text.strip()

    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        data = json.loads(text)
        if "difficulty" in data and "reasoning" in data:
            difficulty = int(data["difficulty"])
            if 1 <= difficulty <= 5:
                return {"difficulty": difficulty, "reasoning": str(data["reasoning"])}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return None


def _build_user_message(labels: dict, current_difficulty: int, history: list[dict] | None) -> str:
    """Build the user message with current state."""
    payload = {
        "labels": labels,
        "current_difficulty": current_difficulty,
        "history": history[-5:] if history else [],
    }
    return json.dumps(payload)


def get_difficulty_decision(
    labels: dict,
    current_difficulty: int,
    history: list[dict] | None = None,
    target: str = DEFAULT_TARGET,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Get difficulty decision from LLM.

    Args:
        labels: Discretized performance labels from discretize_stats()
        current_difficulty: Current difficulty level (1-5)
        history: Recent decisions [{"difficulty": int, "reasoning": str}, ...]
        target: Goal statement for the controller
        model: Anthropic model to use

    Returns:
        {"difficulty": int, "reasoning": str}
        On failure: {"difficulty": current_difficulty, "reasoning": "..."}
    """
    client = anthropic.Anthropic()
    system_prompt = _load_prompt(target)
    user_message = _build_user_message(labels, current_difficulty, history)

    fallback = {
        "difficulty": current_difficulty,
        "reasoning": "Parse error, maintaining current difficulty",
    }

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            text = response.content[0].text
            result = _parse_response(text)

            if result is not None:
                return result

        except anthropic.APIError as e:
            fallback["reasoning"] = f"API error: {e}"
            break

    return fallback
