import typing

import bittensor as bt
from pydantic import Field

MAX_SEALED_CHARS = 62_019


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
    encoded_matrix: str = Field(
        title="encoded_matrix",
        description="The base92 encoded adjacency matrix of the graph.",
        default="",
    )

    # The output of the miner.
    maximum_clique: typing.List[int] = Field(
        title="maximum_clique",
        description="The maximum clique in the graph.",
        default=[],
    )

    def deserialize(self) -> typing.List[int]:
        return self.maximum_clique


class MaximumCliqueSealedGraph(bt.Synapse):
    """Commit-reveal challenge sent as a sealed graph followed by its key."""

    required_hash_fields: typing.ClassVar[tuple[str, ...]] = (
        "uuid",
        "sealed_matrix",
        "digest",
        "key",
        "time_limit",
    )

    # The input to the miner.
    uuid: str = Field(
        title="uuid",
        description="The UUID of the synapse, matching a commit to its reveal.",
        default="",
        max_length=64,
    )
    sealed_matrix: str = Field(
        title="sealed_matrix",
        description="Commit: the sealed base92 encoded adjacency matrix of the graph.",
        default="",
        max_length=MAX_SEALED_CHARS,
    )
    digest: str = Field(
        title="digest",
        description="Commit: commitment to the graph, checked after unsealing.",
        default="",
        max_length=64,
    )
    time_limit: float = Field(
        title="time_limit",
        description="Reveal: the solve budget in seconds.",
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    key: str = Field(
        title="key",
        description="Reveal: the hex key that unseals the committed payload.",
        default="",
        max_length=64,
    )

    # The output of the miner.
    maximum_clique: typing.List[int] = Field(
        title="maximum_clique",
        description="Reveal only: the maximum clique in the graph.",
        default=[],
    )

    def deserialize(self) -> typing.List[int]:
        return self.maximum_clique
