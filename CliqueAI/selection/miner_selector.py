from collections import defaultdict

import numpy as np
from bittensor import Metagraph, Subtensor


class MinerSelector:
    def __init__(
        self,
        subtensor: Subtensor,
        metagraph: Metagraph,
        epoch_length: int,
        owner_hotkey: str,
        netuid: int,
    ):
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.epoch_length = epoch_length
        self.owner_hotkey = owner_hotkey
        self.netuid = netuid
        self.miner_uids = self.filter_validators()
        self.hotkeys = None
        self.coldkeys = None
        self.miner_weights_cache = {}  # difficulty -> weights mapping

    def get_miner_hotkeys(self) -> list[str]:
        """
        Get the hotkeys of the miners.
        """
        if self.hotkeys is None:
            self.hotkeys = [self.metagraph.hotkeys[uid] for uid in self.miner_uids]
        return self.hotkeys

    def get_miner_coldkeys(self) -> list[str]:
        """
        Get the coldkeys of the miners.
        """
        if self.coldkeys is None:
            self.coldkeys = [
                self.subtensor.get_hotkey_owner(hotkey)
                for hotkey in self.get_miner_hotkeys()
            ]
        return self.coldkeys

    def get_miner_keys_by_uid(self, uid: int) -> tuple[str, str]:
        """
        Get the hotkey and coldkey of a miner by UID.
        """
        if uid not in self.miner_uids:
            raise ValueError(f"UID {uid} is not a valid miner UID.")
        index = self.miner_uids.index(uid)
        return self.hotkeys[index], self.coldkeys[index]

    def filter_validators(self) -> list[int]:
        """
        Filter out uids that are active in the metagraph.
        """
        current_block = self.subtensor.get_current_block()
        return [
            uid
            for uid in range(self.metagraph.n)
            if (current_block - self.metagraph.last_update[uid]) > self.epoch_length
        ]

    def stake_weight(self, coldkeys: list[str]) -> np.ndarray:
        """
        Calculate the stake weight for each hotkey based on the stakes on the owner and miner.

        Args:
            coldkeys (list[str]): List of coldkeys to calculate stake weights for.

        Returns:
            np.ndarray: An array of stake weights for each hotkey.
        """
        coldkey_to_hotkey_count = defaultdict(int)
        for coldkey in coldkeys:
            coldkey_to_hotkey_count[coldkey] += 1

        coldkey_to_owner_stake = {
            ck: self.subtensor.get_stake(ck, self.owner_hotkey, netuid=self.netuid).rao
            for ck in set(coldkeys)
        }
        stake_on_miner = np.array(
            self.metagraph.alpha_stake[self.miner_uids], dtype=float
        )

        weights = np.array(
            [
                stake_on_miner[i]
                + coldkey_to_owner_stake[ck] / coldkey_to_hotkey_count[ck]
                for i, ck in enumerate(coldkeys)
            ],
            dtype=float,
        )
        return weights

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

        hotkeys = self.get_miner_hotkeys()
        coldkeys = self.get_miner_coldkeys()

        s_m = self.stake_weight(coldkeys)
        S = sum(s_m) / len(hotkeys)
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
