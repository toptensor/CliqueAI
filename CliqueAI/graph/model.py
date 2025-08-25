from common.base.wandb_logging.base import SignedRequest
from pydantic import BaseModel


class GraphPayload(BaseModel):
    timestamp: float
    hotkey: str
    uuid: str
    netuid: int
    label: str | None = None
    number_of_nodes_min: int | None = None
    number_of_nodes_max: int | None = None
    number_of_edges_min: int | None = None
    number_of_edges_max: int | None = None


ValidatorGraphRequest = SignedRequest[GraphPayload]


class LambdaGraph(BaseModel):
    uuid: str
    label: str
    number_of_nodes: int
    adjacency_list: list[list[int]]
