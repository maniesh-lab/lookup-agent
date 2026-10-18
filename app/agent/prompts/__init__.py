"""System prompts, one text file per version (v1.txt, v2.txt, ...).

Keeping each version as its own file means old versions stay runnable,
so evals can compare them, and each trace records which one it used.
"""

from datetime import date
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def available_versions() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("v*.txt"))


def load_prompt(version: str, today: date | None = None) -> str:
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        raise ValueError(f"Unknown prompt version {version!r}. Available: {available_versions()}")
    return path.read_text(encoding="utf-8").format(today=(today or date.today()).isoformat())
