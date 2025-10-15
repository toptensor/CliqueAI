import random

from CliqueAI.selection.miner_selector import MinerSelector
from pydantic import BaseModel

max_int = 1e9


class Range(BaseModel):
    min: int
    max: int


class Problem(BaseModel):
    label: str
    vertex_range: Range
    edge_range: Range
    difficulty: float


PROBLEMS = [
    Problem(
        label="general",
        vertex_range=Range(min=90, max=100),
        edge_range=Range(min=0, max=max_int),
        difficulty=0.1,
    ),
    Problem(
        label="general",
        vertex_range=Range(min=290, max=300),
        edge_range=Range(min=100, max=max_int),
        difficulty=0.2,
    ),
    Problem(
        label="general",
        vertex_range=Range(min=490, max=500),
        edge_range=Range(min=0, max=max_int),
        difficulty=0.3,
    ),
    Problem(
        label="general",
        vertex_range=Range(min=690, max=700),
        edge_range=Range(min=0, max=max_int),
        difficulty=0.4,
    )
]


class ProblemSelector:
    def __init__(self, miner_selector: MinerSelector):
        self.miner_selector = miner_selector

    def select_problem(self):
        selected_problem = random.choice(PROBLEMS)
        return selected_problem
