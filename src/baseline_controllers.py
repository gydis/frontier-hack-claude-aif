"""
Baseline Controllers — fixed and rule-based difficulty controllers for comparison.

Same interface as LLM controller for easy swapping via config.
"""

from __future__ import annotations


class FixedController:
    """Always returns the same difficulty level."""

    def __init__(self, difficulty: int = 3):
        if difficulty not in range(1, 6):
            raise ValueError(f"Difficulty must be 1-5, got {difficulty}")
        self.difficulty = difficulty

    def get_difficulty_decision(
        self,
        labels: dict,
        current_difficulty: int,
        history: list[dict] | None = None,
        target: str | None = None,
    ) -> dict:
        return {
            "difficulty": self.difficulty,
            "reasoning": f"Fixed difficulty mode (level {self.difficulty})",
        }


class RuleBasedController:
    """
    Simple threshold-based controller.

    Logic:
    - Majority "poor" → decrease difficulty
    - Majority "high" → increase difficulty
    - Otherwise → maintain current
    """

    def get_difficulty_decision(
        self,
        labels: dict,
        current_difficulty: int,
        history: list[dict] | None = None,
        target: str | None = None,
    ) -> dict:
        counts = {"poor": 0, "normal": 0, "high": 0}
        for label in labels.values():
            if label in counts:
                counts[label] += 1

        total = sum(counts.values())
        majority_threshold = total / 2

        if counts["poor"] > majority_threshold:
            new_difficulty = max(1, current_difficulty - 1)
            reasoning = f"Majority poor metrics ({counts['poor']}/{total}). Lowering difficulty."
        elif counts["high"] > majority_threshold:
            new_difficulty = min(5, current_difficulty + 1)
            reasoning = f"Majority high metrics ({counts['high']}/{total}). Raising difficulty."
        else:
            new_difficulty = current_difficulty
            reasoning = f"Mixed metrics (poor={counts['poor']}, normal={counts['normal']}, high={counts['high']}). Maintaining difficulty."

        return {"difficulty": new_difficulty, "reasoning": reasoning}


class LLMControllerWrapper:
    """Wrapper to give LLM controller the same class interface."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    def get_difficulty_decision(
        self,
        labels: dict,
        current_difficulty: int,
        history: list[dict] | None = None,
        target: str | None = None,
    ) -> dict:
        from src.llm_controller import get_difficulty_decision

        return get_difficulty_decision(
            labels=labels,
            current_difficulty=current_difficulty,
            history=history,
            target=target or "keep player performance in the 'normal' band",
            model=self.model,
        )


def create_controller(controller_type: str, **kwargs):
    """
    Factory function to create controllers.

    Args:
        controller_type: "llm" | "fixed" | "rule_based"
        **kwargs: Controller-specific arguments
            - fixed: difficulty (int)
            - llm: model (str)

    Returns:
        Controller with get_difficulty_decision() method
    """
    if controller_type == "fixed":
        return FixedController(difficulty=kwargs.get("difficulty", 3))
    elif controller_type == "rule_based":
        return RuleBasedController()
    elif controller_type == "llm":
        return LLMControllerWrapper(model=kwargs.get("model", "claude-sonnet-4-20250514"))
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
