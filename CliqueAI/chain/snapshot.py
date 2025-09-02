from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from bittensor import Metagraph


@dataclass(frozen=True)
class Snapshot:
    netuid: int
    epoch_length: int
    block: int
    owner_hotkey: str
    metagraph: Metagraph
    hotkeys: list[str]
    coldkeys: list[str]
    alpha_stakes: npt.NDArray[np.int64]
    stakes_on_owner_validator: npt.NDArray[np.int64]
