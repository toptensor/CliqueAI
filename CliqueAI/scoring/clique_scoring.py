from collections import Counter
from typing import List

import numpy as np
from CliqueAI.graph.model import LambdaGraph


class CliqueScoreCalculator:
    def __init__(
        self, graph: LambdaGraph, difficulty: float, responses: List[List[int]]
    ):
        """
        Initializes the scoring calculator.

        Args:
        - graph (LambdaGraph): The graph to validate against.
        - difficulty (float): The difficulty level for scoring.
        - responses (List[List[int]]): List of node sets returned by miners.
        """
        self.graph = graph
        self.difficulty = difficulty
        self.responses = responses

    def is_valid_maximum_clique(self, nodes: List[int]) -> bool:
        """
        Returns True if the given nodes form a clique in the graph.
        """
        node_set = set(nodes)
        # 0. Check if the node set is empty
        if len(node_set) == 0:
            return False

        # 1. Check for duplicates or out-of-range nodes
        if len(node_set) != len(nodes):
            return False
        if not node_set.issubset(range(self.graph.number_of_nodes)):
            return False

        # 2. Check if all pairs of nodes are connected (i.e., form a clique)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if nodes[j] not in self.graph.adjacency_list[nodes[i]]:
                    return False

        # 3. Check if any other node can be added to form a larger clique
        all_nodes = set(range(self.graph.number_of_nodes))
        remaining_nodes = all_nodes - node_set
        for candidate in remaining_nodes:
            # Candidate must be connected to all nodes in the current clique
            if node_set.issubset(self.graph.adjacency_list[candidate]):
                return False  # Clique can be extended, so it's not maximum

        return True

    def optimality(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the optimality scores for each response.
        """
        val = np.array(
            [
                1 if self.is_valid_maximum_clique(response) else 0
                for response in self.responses
            ]
        )
        size = np.array([len(response) for response in self.responses]) * val
        zeros = np.zeros(len(self.responses))
        if len(size) == 0:
            return zeros, zeros, zeros, zeros

        max_size = np.max(size)
        if max_size <= 0:
            return zeros, zeros, zeros, zeros

        rel = size / max_size
        pr = np.array([np.sum(size > size[i]) / len(size) for i in range(len(size))])

        omega = np.zeros(len(self.responses))
        for i, valid in enumerate(val):
            if valid:
                omega[i] = np.exp(-pr[i] / rel[i])

        max_omega = np.max(omega)
        if max_omega == 0:
            return rel, pr, omega, omega
        omega_normalized = omega / max_omega
        return rel, pr, omega, omega_normalized

    def diversity(self) -> np.ndarray:
        """
        Calculate the diversity scores for each response.
        """
        val = np.array(
            [
                1 if self.is_valid_maximum_clique(response) else 0
                for response in self.responses
            ]
        )

        canonical_responses = [tuple(sorted(r)) for r in self.responses]
        counts = Counter(canonical_responses)
        unq = np.array([1 / counts[sol] for sol in canonical_responses])

        delta = val * unq
        if len(delta) == 0:
            return np.array([])

        max_delta = np.max(delta)
        if max_delta == 0:
            return delta
        delta_normalized = delta / max_delta
        return delta_normalized

    def get_scores(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute normalized scores.
        """
        rel, pr, omega, optimality = self.optimality()
        diversity = self.diversity()

        rewards = optimality * (1 + self.difficulty) + diversity
        return rel, pr, omega, optimality, diversity, rewards
