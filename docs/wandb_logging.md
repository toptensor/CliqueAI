# W&B Logging

W&B Dashboard: https://wandb.ai/toptensor-ai/CliqueAI/table

This document describes the data logged by the Maximum Clique Validator to Weights & Biases (W&B) for monitoring, analysis, and debugging purposes.

## W&B Data Model
The structured logging format is defined using a Pydantic model in [WandbRunLogData](https://github.com/toptensor/CliqueAI/blob/main/common/base/wandb_logging/model.py#L13-L30).

## Field Descriptions
### Global Information
| Field        | Type    | Description                                              |
| ------------ | ------- | -------------------------------------------------------- |
| **timestamp**| float   | Unix timestamp when the validator uploaded the results.  |
| **uuid**     | str     | Unique identifier for this run (matches synapse UUID).   |
| **type**     | str     | Task type, e.g., MaximumCliqueOfLambdaGraph.             |

### Synapse Information
| Field               | Type              | Description                                                     |
| ------------------- | ---------------- | ---------------------------------------------------------------- |
| **label**           | str               | Problem category, e.g., general.                                |
| **difficulty**      | float             | Problem difficulty in [0,1], derived from problem attributes.   |
| **number_of_nodes** | int               | The number of vertices in the graph.                            |
| **adjacency_list**  | list[list[int]]   | Adjacency list representation of the graph **after shuffling**. |

### Miner Information
Detailed scoring logic is described in the Mechanism documentation.

> All list fields are index-aligned, meaning miner_uids[i] corresponds to miner_hotkeys[i], miner_ans[i], etc.


| Field                 | Type              | Description                                               |
| --------------------- | ----------------  | --------------------------------------------------------- |
| **miner_uids**        | list[int]         | UID of each selected miner.                               |
| **miner_hotkeys**     | list[str]         | Hotkeys of each miner.                                    |
| **miner_coldkeys**    | list[str]         | Coldkeys associated with each miner.                      |
| **miner_ans**         | list[list[int]]   | Miner solutions: list of vertex IDs forming a clique.     |
| **miner_rel**         | list[float]       | Rel score of each miner.                                  |
| **miner_pr**          | list[float]       | PR score of each miner.                                   |
| **miner_optimality**  | list[float]       | Optimality score of each miner.                           |
| **miner_omega**       | list[float]       | Normalized optimality score of each miner.                |
| **miner_diversity**   | list[float]       | Normalized diversity score measuring solution uniqueness. |
| **miner_rewards**     | list[float]       | Final aggregated reward for each miner.                   |
