import numpy as np
from CliqueAI.chain.snapshot import Snapshot


class MinerSelector:
    def __init__(
        self,
        current_block: int,
        snapshot: Snapshot,
    ):
        self.current_block = current_block
        self.snapshot = snapshot
        self.miner_uids = self._filter_validators()
        self.reference_r = 1.5

    def _filter_validators(self) -> list[int]:
        """
        Filter out uids that are validators in the metagraph.
        """
        uids = []
        for uid in range(self.snapshot.metagraph.n):
            if self.snapshot.metagraph.validator_trust[uid] > 0:
                continue

            if (
                self.current_block - self.snapshot.metagraph.last_update[uid]
                <= self.snapshot.epoch_length
            ):
                continue

            uids.append(uid)
        return uids

    def miner_selection_probabilities(self, difficulty: float) -> np.ndarray:
        """
        Calculate equal sampling probabilities for all miners.

        Args:
            difficulty (float): The difficulty threshold for sampling miners.

        Returns:
            np.ndarray: An array of selection probabilities for each miner.
        """
        if not self.miner_uids:
            return np.array([])

        x_m = np.sqrt(1 + self.reference_r)
        delta = x_m - difficulty - 0.5
        P = 1 - np.exp(-np.maximum(0, delta))
        return np.full(len(self.miner_uids), P)

    def sample_miner_uids(self, difficulty: float) -> list[int]:
        """
        Sample miners using equal selection probabilities.

        Args:
            difficulty (float): The difficulty threshold for sampling miners.

        Returns:
            list[int]: A list of selected miner UIDs.
        """
        P = self.miner_selection_probabilities(difficulty)

        random_vals = np.random.rand(len(P))
        selected_mask = random_vals < P
        selected_uids = np.array(self.miner_uids)[selected_mask]
        return selected_uids.tolist()
