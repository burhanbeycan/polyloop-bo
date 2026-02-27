from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class PromptConfig:
    system_role: str = "You are an expert materials chemist and optimisation scientist."
    max_history: int = 8


def build_llm_prompt(
    history: pd.DataFrame,
    candidate: Dict[str, float],
    objective_description: str,
    cfg: PromptConfig = PromptConfig(),
) -> str:
    """Build a prompt for an LLM to provide human-readable guidance or constraints.

    This module is intentionally API-agnostic: you can plug in OpenAI, local LLMs, etc.
    It is designed to support 'human-in-the-loop' workflows where the model proposes:
    - plausibility checks (solubility, viscosity, safety)
    - failure mode warnings (beads, clogging)
    - recommendations for ranges based on literature

    Parameters
    ----------
    history:
        Previous experiments with outcomes.
    candidate:
        Proposed next experiment parameters.
    objective_description:
        What we are trying to optimise (multi-objective description).

    Returns
    -------
    prompt: str
        A formatted prompt string.
    """
    recent = history.tail(cfg.max_history).copy()

    lines = []
    lines.append(f"SYSTEM: {cfg.system_role}")
    lines.append("")
    lines.append("We are running a closed-loop electrospinning optimisation.")
    lines.append(f"Objective: {objective_description}")
    lines.append("")
    lines.append("Recent experiments (most recent last):")
    lines.append(recent.to_csv(index=False))
    lines.append("")
    lines.append("Proposed next candidate:")
    for k, v in candidate.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("TASK:")
    lines.append("1) Check feasibility and typicality vs electrospinning practice.")
    lines.append("2) Suggest any parameter adjustments to reduce failure risk (beads, clogging).")
    lines.append("3) Provide 2-3 short, actionable recommendations for the lab operator.")
    return "\n".join(lines)
