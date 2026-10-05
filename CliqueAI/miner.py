import asyncio
import time
import typing

import bittensor as bt
from CliqueAI.clique_algorithms import networkx_algorithm
from CliqueAI.commit_reveal import CommitRevealResponder
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph, MaximumCliqueSealedGraph
from common.base.miner import BaseMinerNeuron


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom

    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self.codec = GraphCodec()
        self.commit_reveal = CommitRevealResponder(solve=self.solve_graph)
        self.axon.attach(
            forward_fn=self.forward_sealed_graph,
            blacklist_fn=self.blacklist_sealed_graph,
            priority_fn=self.priority_sealed_graph,
            verify_fn=self.verify_sealed_graph,
        )
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.blacklist_graph,
            priority_fn=self.priority_graph,
            verify_fn=self.verify_graph,
        )

    def solve_graph(
        self,
        number_of_nodes: int,
        adjacency_list: list[list[int]],
        time_limit: float = 0.0,
    ) -> list[int]:
        if time_limit <= 0:
            return []
        maximum_clique: list[int] = networkx_algorithm(number_of_nodes, adjacency_list)
        # or use GNN models
        # from CliqueAI.clique_algorithms import scattering_clique_algorithm
        # maximum_clique = scattering_clique_algorithm(number_of_nodes, adjacency_list)
        bt.logging.info(
            f"Maximum clique found: {maximum_clique} with size {len(maximum_clique)}"
        )
        return maximum_clique

    async def forward_sealed_graph(
        self, synapse: MaximumCliqueSealedGraph
    ) -> MaximumCliqueSealedGraph:
        return await self.commit_reveal.handle(synapse)

    async def blacklist_sealed_graph(
        self, synapse: MaximumCliqueSealedGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_sealed_graph(self, synapse: MaximumCliqueSealedGraph) -> float:
        return await self.priority(synapse)

    async def verify_sealed_graph(self, synapse: MaximumCliqueSealedGraph) -> None:
        await self.verify(synapse)

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        adjacency_list = self.codec.matrix_to_list(
            self.codec.decode_matrix(synapse.encoded_matrix)
        )
        synapse.maximum_clique = await asyncio.to_thread(
            self.solve_graph,
            synapse.number_of_nodes,
            adjacency_list,
            float(synapse.timeout or 0.0),
        )
        return synapse

    async def blacklist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)

    async def verify_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> None:
        await self.verify(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Miner has started running.")
        while True:
            if miner.should_exit:
                bt.logging.info("Miner is exiting.")
                break
            time.sleep(1)
