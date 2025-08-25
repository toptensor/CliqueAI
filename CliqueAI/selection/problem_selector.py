import random
import sys

from bittensor import Metagraph
from pydantic import BaseModel

from .miner_selector import MinerSelector

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
        difficulty=0.4,
    ),
]


class ProblemSelector:
    def __init__(self, metagraph: Metagraph, miner_selector: MinerSelector):
        self.metagraph = metagraph
        self.miner_selector = miner_selector

    def select_problem(self):
        weights = [
            1 / sum(w)
            for w in [self.miner_selector.miner_weights(p.difficulty) for p in PROBLEMS]
        ]
        selected_problem = random.choices(PROBLEMS, weights=weights, k=1)[0]
        return selected_problem
