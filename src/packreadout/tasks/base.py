"""Task registry. A task turns a dataset split into ``(prompt, options, gold)`` examples.

Roles: ``train`` tasks form the fine-tuning mixture, ``heldout`` tasks are never
trained on and test generalization to unseen option sets, ``sanity`` tasks use the
lm-evaluation-harness prompt format for checking the scorer against the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from packreadout.formatting import DEFAULT_FORMAT, OptionFormat

Loader = Callable[[str, int | None], list["Example"]]  # (split, limit) -> examples


@dataclass
class Example:
    prompt: str
    options: list[str]
    gold: int


@dataclass
class Task:
    name: str
    loader: Loader
    role: str  # train | heldout | sanity
    fmt: OptionFormat = DEFAULT_FORMAT

    def load(self, split: str = "test", limit: int | None = None) -> list[Example]:
        return self.loader(split, limit)


TASKS: dict[str, Task] = {}


def register(name: str, role: str, fmt: OptionFormat = DEFAULT_FORMAT):
    def deco(loader: Loader):
        TASKS[name] = Task(name=name, loader=loader, role=role, fmt=fmt)
        return loader

    return deco


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]


def tasks_with_role(role: str) -> list[Task]:
    return [t for t in TASKS.values() if t.role == role]


def take(ds, limit: int | None, first: bool = False):
    """A seeded shuffle, cut to ``limit`` rows (or the first ``limit`` rows unshuffled, as lm-eval does).
    Shuffling even without a limit keeps dataset order (often sorted by label) out of the folds."""
    if not first:
        ds = ds.shuffle(seed=0)
    if limit is None or limit >= len(ds):
        return ds
    return ds.select(range(limit))
