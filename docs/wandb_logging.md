# W&B Logging

W&B Dashboard: https://wandb.ai/toptensor-ai/CliqueAI/table

This document describes the data logged by the Maximum Clique Validator to Weights & Biases (W&B) for monitoring, analysis, and debugging purposes.

## W&B Data Model
The structured logging format is defined using a Pydantic model in [WandbRunLogData](https://github.com/toptensor/CliqueAI/blob/main/common/base/wandb_logging/model.py#L13-L35).

## Field Descriptions
### Global Information
| Field        | Type    | Description                                              |
| ------------ | ------- | -------------------------------------------------------- |
| **timestamp**| float   | Unix timestamp when the validator uploaded the results.  |
| **uuid**     | str     | Unique identifier for this run (matches synapse UUID).   |
| **type**     | str     | Task type, e.g., MaximumCliqueSealedGraph.               |

### Synapse Information
| Field               | Type              | Description                                                     |
| ------------------- | ---------------- | ---------------------------------------------------------------- |
| **label**           | str               | Problem category, e.g., general.                                |
| **difficulty**      | float             | Difficulty assigned to the selected problem type.               |
| **time_limit**      | float             | Time limit sent to miners, in seconds.                          |
| **number_of_nodes** | int               | The number of vertices in the graph.                            |
| **adjacency_list**  | list[list[int]]   | Adjacency list representation of the graph **after shuffling**. |
| **encoded_matrix**  | str               | The base92 encoded adjacency matrix of the graph.               |
| **challenge_digest**| str               | The commitment the miners were sent for this graph.             |
| **challenge_sealed_matrix** | str       | The sealed payload the miners were sent in the commit phase.    |
| **challenge_key**   | str               | The key released in the reveal phase, which unseals it.         |

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

## Verifying a Challenge Was Distributed as Logged
Validators deliver each challenge as a commit-reveal pair. The whole exchange is recorded here, so a round can be replayed offline without a miner's cooperation:

```
unseal(challenge_sealed_matrix, challenge_key) == encoded_matrix
challenge_digest == sha256(uint32_be(len(uuid)) || uuid || encoded_matrix)
```

Both checks use only the fields above, and both fail if the validator's own pipeline was inconsistent.

Detecting a validator that *distributed* something other than what it logged needs one more party. A miner receives **challenge_digest** in the commit, covered by the validator's request signature, and only answers once the revealed key opens the sealed graph onto that digest. If the digest logged here differs from the one it was sent, the graph published for scoring is not the graph it solved, and the signed commit is the evidence.

`CliqueAI.commit_reveal` provides `commitment` and `unseal`; the construction is defined in `CliqueAI/commit_reveal/sealer.py`.
