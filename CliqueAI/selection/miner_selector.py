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
        self.miner_weights_cache = {}  # difficulty -> weights mapping
        self.miner_uids = self._filter_validators()

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

    def miner_weights(self, difficulty: float) -> np.ndarray:
        """
        Calculate the weights of miners based on their stake and the difficulty.

        Args:
            difficulty (float): The difficulty threshold for sampling miners.

        Returns:
            np.ndarray: An array of weights for each miner.
        """
        if difficulty in self.miner_weights_cache:
            return self.miner_weights_cache[difficulty]

        s_m = [
            self.snapshot.alpha_stakes[uid]
            + self.snapshot.stakes_on_owner_validator[uid]
            for uid in self.miner_uids
        ]
        S = sum(s_m) / len(self.miner_uids)
        if S == 0:
            x_m = np.array([1] * len(self.miner_uids))
        else:
            x_m = np.sqrt(1 + s_m / S)

        delta = x_m - difficulty - 0.5
        P = 1 - np.exp(-np.maximum(0, delta))
        self.miner_weights_cache[difficulty] = P
        return P

    def sample_miner_uids(self, difficulty: float) -> list[int]:
        """
        Sample miners based on their stake weights and a given difficulty.

        Args:
            difficulty (float): The difficulty threshold for sampling miners.

        Returns:
            list[int]: A list of selected miner UIDs.
        """
        P = self.miner_weights(difficulty)

        random_vals = np.random.rand(len(P))
        selected_mask = random_vals < P
        selected_uids = np.array(self.miner_uids)[selected_mask]
        return selected_uids.tolist()
