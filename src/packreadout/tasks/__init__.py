from packreadout.tasks import classify, mcq  # noqa: F401  (registers tasks)
from packreadout.tasks.base import TASKS, Example, Task, get_task, tasks_with_role

__all__ = ["TASKS", "Example", "Task", "get_task", "tasks_with_role"]
