"""Prompt and option string conventions.

Two conventions matter for scoring and are easy to get wrong:

* The option is scored as a *continuation* of the prompt. The prompt ends with a cue
  (``"Answer:"``) and every option gets a leading space, so the first option token
  is the natural next token.
* Every option ends with a terminator (default ``"\\n"``) that is scored too. Without
  it, an option whose tokens are a prefix of another option's tokens (``"New York"``
  vs ``"New York City"``) can never score lower.

``terminator=""`` reproduces the lm-evaluation-harness convention (no terminator),
used only for the sanity-check tasks.

All tasks list the option set in the prompt (the model sees the schema before it
scores), which is what a decision API implies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionFormat:
    terminator: str = "\n"

    def render(self, option: str) -> str:
        """The scored string: a leading space (unless the option brings its own), the option, the terminator."""
        text = option if option.startswith(" ") else " " + option
        return text + self.terminator


DEFAULT_FORMAT = OptionFormat()
LM_EVAL_FORMAT = OptionFormat(terminator="")


def classify_prompt(text: str, labels: list[str], instruction: str) -> str:
    """Label a piece of text with one of the listed labels."""
    return f"{instruction.strip()}\nLabels: {', '.join(labels)}\nText: {text.strip()}\nLabel:"


def mcq_prompt(question: str, options: list[str], context: str | None = None) -> str:
    """Answer a question with one of the listed options (options may be long, one per line)."""
    listed = "\n".join(f"- {o}" for o in options)
    head = f"{context.strip()}\n" if context else ""
    return f"{head}Question: {question.strip()}\nOptions:\n{listed}\nAnswer:"


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def symbol_prompt(prompt: str, options: list[str]) -> tuple[str, list[str]]:
    """The first-token 'symbol' baseline: options become lettered lines and the model
    is scored on the letter. Returns ``(prompt, letters)``. Only for up to 26 options."""
    if len(options) > len(LETTERS):
        raise ValueError("symbol scoring supports at most 26 options")
    body = prompt.rsplit("\n", 1)[0]  # drop the "Answer:" / "Label:" cue
    listed = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{body}\nChoices:\n{listed}\nAnswer:", list(LETTERS[: len(options)])
