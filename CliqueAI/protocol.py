import typing

import bittensor as bt
from pydantic import Field


class MaximumCliqueOfLambdaGraph(bt.Synapse):
    # The input to the miner.
    uuid: str = Field(
        title="uuid",
        description="The UUID of the synapse.",
        default="",
    )
    label: str = Field(
        title="label",
        description="The label of the synapse.",
        default="",
    )
    number_of_nodes: int = Field(
        title="number_of_nodes",
        description="The number of nodes in the graph.",
        default=0,
    )
    adjacency_list: typing.List[typing.List[int]] = Field(
        title="adjacency_list",
        description="The adjacency list of the graph.",
        default=[],
    )

    # The output of the miner.
    maximum_clique: typing.List[int] = Field(
        title="maximum_clique",
        description="The maximum clique in the graph.",
        default=[],
    )

    def deserialize(self) -> typing.List[int]:
        return self.maximum_clique
