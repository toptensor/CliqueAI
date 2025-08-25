from pydantic import BaseModel

from .base import SignedRequest


class WandbRunInitPayload(BaseModel):
    timestamp: float
    hotkey: str
    netuid: int
    version: str


class WandbRunLogData(BaseModel):
    timestamp: float
    uuid: str
    type: str
    label: str
    difficulty: float
    number_of_nodes: int
    adjacency_list: list
    miner_uids: list
    miner_hotkeys: list
    miner_coldkeys: list
    miner_ans: list
    miner_rel: list
    miner_pr: list
    miner_omega: list
    miner_optimality: list
    miner_diversity: list
    miner_rewards: list


class WandbRunLogPayload(BaseModel):
    timestamp: float
    hotkey: str
    netuid: int
    version: str
    run_id: str
    data: WandbRunLogData


WandbRunInitRequest = SignedRequest[WandbRunInitPayload]
WandbRunLogRequest = SignedRequest[WandbRunLogPayload]


class WandbRunInitResponse(BaseModel):
    run_id: str
    message: str


class WandbRunLogResponse(BaseModel):
    message: str
