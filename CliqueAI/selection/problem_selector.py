import random

from CliqueAI.selection.miner_selector import MinerSelector
from pydantic import BaseModel, Field

max_int = 1e9
DEFAULT_TIME_LIMITS = [6, 7.5, 10, 15, 30]


class Range(BaseModel):
    min: int
    max: int


class Problem(BaseModel):
    label: str
    vertex_range: Range
    edge_range: Range
    difficulty: float
    time_limits: list[float] = Field(default_factory=lambda: DEFAULT_TIME_LIMITS.copy())


PROBLEMS = [
    Problem(
        label="general",
        vertex_range=Range(min=290, max=300),
        edge_range=Range(min=100, max=max_int),
        difficulty=0.7,
    ),
    Problem(
        label="general",
        vertex_range=Range(min=490, max=500),
        edge_range=Range(min=0, max=max_int),
        difficulty=0.8,
    ),
    Problem(
        label="general",
        vertex_range=Range(min=690, max=700),
        edge_range=Range(min=0, max=max_int),
        difficulty=0.9,
    ),
    Problem(
        label="general",
        vertex_range=Range(min=890, max=900),
        edge_range=Range(min=0, max=max_int),
        difficulty=1.0,
        time_limits=[7.5, 10, 15, 30],
    ),
]


class ProblemSelector:
    def __init__(self, miner_selector: MinerSelector):
        self.miner_selector = miner_selector

    def select_problem(self):
        selected_problem = random.choice(PROBLEMS)
        return selected_problem

    def select_time_limit(self, problem: Problem | None = None):
        time_limits = problem.time_limits if problem is not None else DEFAULT_TIME_LIMITS
        selected_time_limit = random.choice(time_limits)
        return selected_time_limit
