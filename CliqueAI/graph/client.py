import time
import uuid

import bittensor as bt
import httpx
from CliqueAI.graph.model import GraphPayload, LambdaGraph, ValidatorGraphRequest


async def get_graph(
    wallet: bt.Wallet,
    netuid: int,
    label: str | None = None,
    number_of_nodes_min: int | None = None,
    number_of_nodes_max: int | None = None,
    number_of_edges_min: int | None = None,
    number_of_edges_max: int | None = None,
) -> LambdaGraph:
    """
    This function is called by the validator to get a graph.

    Args:
        wallet (bt.Wallet): Validator wallet.
    """
    url = "http://lambda.toptensor.ai/graph/lambda"
    payload = GraphPayload(
        timestamp=time.time(),
        hotkey=wallet.hotkey.ss58_address,
        uuid=str(uuid.uuid4()),
        netuid=netuid,
        label=label,
        number_of_nodes_min=number_of_nodes_min,
        number_of_nodes_max=number_of_nodes_max,
        number_of_edges_min=number_of_edges_min,
        number_of_edges_max=number_of_edges_max,
    )
    signature = wallet.hotkey.sign(payload.model_dump_json()).hex()
    body = ValidatorGraphRequest(
        payload=payload,
        signature=signature,
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=body.model_dump())
        response.raise_for_status()
        return LambdaGraph.model_validate(response.json())
