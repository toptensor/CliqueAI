import asyncio
import copy
import random
import time

import aiohttp
import bittensor as bt
from CliqueAI import validator_version, version_int_version
from CliqueAI.graph.client import get_graph
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator
from CliqueAI.selection.miner_selector import MinerSelector
from CliqueAI.selection.problem_selector import ProblemSelector
from CliqueAI.transport.axon_requester import AxonRequester
from common.base.validator import BaseValidatorNeuron
from common.base.wandb_logging.model import WandbRunLogData


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    validator_version = validator_version

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        if not self.config.wandb.off:
            self.wandb_client.init(version=self.validator_version)

        self.forward_interval = self.config.forward.interval
        self.spec_version = version_int_version
        self.owner_hotkey = None

    def get_owner_hotkey(self):
        if self.owner_hotkey is None:
            self.owner_hotkey = self.subtensor.get_subnet_owner_hotkey(
                netuid=self.config.netuid
            )
        bt.logging.info(f"Owner hotkey: {self.owner_hotkey}")
        return self.owner_hotkey

    async def forward(self):
        start_time = time.time()
        bt.logging.info(f"Validator forward started at {start_time}")

        miner_selector = MinerSelector(
            subtensor=self.subtensor,
            metagraph=self.metagraph,
            epoch_length=self.config.neuron.epoch_length,
            owner_hotkey=self.get_owner_hotkey(),
            netuid=self.config.netuid,
        )
        problem_selector = ProblemSelector(
            metagraph=self.metagraph,
            miner_selector=miner_selector,
        )

        # Problem Selection
        problem = problem_selector.select_problem()
        try:
            graph = await get_graph(
                wallet=self.wallet,
                netuid=self.config.netuid,
                label=problem.label,
                number_of_nodes_min=problem.vertex_range.min,
                number_of_nodes_max=problem.vertex_range.max,
                number_of_edges_min=problem.edge_range.min,
                number_of_edges_max=problem.edge_range.max,
            )
        except Exception as e:
            bt.logging.error(f"Error fetching graph: {e}")
            await asyncio.sleep(30)
            return
        bt.logging.info(f"Selected problem: {problem}")

        # Shuffle the graph
        old_adjacency_list = graph.adjacency_list
        old_vertices = list(range(graph.number_of_nodes))
        new_vertices = copy.deepcopy(old_vertices)
        random.shuffle(new_vertices)
        vertex_map = dict(zip(old_vertices, new_vertices))
        bt.logging.debug(f"Vertex mapping (old -> new): {vertex_map}")

        new_adjacency_list = [[] for _ in range(graph.number_of_nodes)]
        for u in range(graph.number_of_nodes):
            new_u = vertex_map[u]
            mapped_neighbors = [vertex_map[v] for v in old_adjacency_list[u]]
            new_adjacency_list[new_u] = sorted(mapped_neighbors)
        graph.adjacency_list = new_adjacency_list

        # Synapse
        synapse = MaximumCliqueOfLambdaGraph(
            uuid=graph.uuid,
            label=graph.label,
            number_of_nodes=graph.number_of_nodes,
            adjacency_list=graph.adjacency_list,
            timeout=self.config.forward.timeout,
        )
        bt.logging.info(f"Synapse UUID: {synapse.uuid}")

        # Miner Selection
        selector = MinerSelector(
            subtensor=self.subtensor,
            metagraph=self.metagraph,
            epoch_length=self.config.neuron.epoch_length,
            owner_hotkey=self.get_owner_hotkey(),
            netuid=self.config.netuid,
        )
        selected_uids = selector.sample_miner_uids(difficulty=problem.difficulty)
        bt.logging.info(f"Selected UIDs: {selected_uids}")
        axons = [self.metagraph.axons[uid] for uid in selected_uids]

        # querying.
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=256)
        ) as session:
            axon_requester = AxonRequester(session=session, dendrite=self.dendrite)
            responses = await axon_requester.forward(
                synapse=synapse,
                axons=axons,
            )
        responses = [response.deserialize() for response in responses]

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {responses}")

        # Score the responses.
        clique_score_calculator = CliqueScoreCalculator(
            graph=graph,
            difficulty=problem.difficulty,
            responses=responses,
        )
        (
            rel,
            pr,
            omega,
            optimality,
            diversity,
            rewards,
        ) = clique_score_calculator.get_scores()
        bt.logging.debug(f"Rel: {rel}")
        bt.logging.debug(f"PR: {pr}")
        bt.logging.debug(f"Omega: {omega}")
        bt.logging.info(f"Optimality: {optimality}")
        bt.logging.info(f"Diversity: {diversity}")
        bt.logging.info(f"Rewards: {rewards}")

        # Update the scores.
        self.update_scores(rewards, selected_uids)

        if not self.config.wandb.off and len(selected_uids) > 0:
            miner_hotheys, miner_coldkeys = map(
                list,
                zip(*[selector.get_miner_keys_by_uid(uid) for uid in selected_uids]),
            )

            self.wandb_client.log(
                WandbRunLogData(
                    timestamp=time.time(),
                    uuid=synapse.uuid,
                    type=synapse.__class__.__name__,
                    label=synapse.label,
                    difficulty=problem.difficulty,
                    number_of_nodes=synapse.number_of_nodes,
                    adjacency_list=synapse.adjacency_list,
                    miner_uids=selected_uids,
                    miner_hotkeys=miner_hotheys,
                    miner_coldkeys=miner_coldkeys,
                    miner_ans=responses,
                    miner_rel=rel,
                    miner_pr=pr,
                    miner_omega=omega,
                    miner_optimality=optimality,
                    miner_diversity=diversity,
                    miner_rewards=rewards,
                )
            )

        end_time = time.time()
        bt.logging.info(
            f"Validator forward completed in {end_time - start_time} seconds"
        )
        await asyncio.sleep(max(0, self.forward_interval - (end_time - start_time)))


if __name__ == "__main__":
    with Validator() as validator:
        bt.logging.info("Validator has started running.")
        while True:
            if validator.should_exit:
                bt.logging.info("Validator is exiting.")
                break
            time.sleep(1)
