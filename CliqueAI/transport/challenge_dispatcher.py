import math
import time
from dataclasses import dataclass, field

import bittensor as bt
from CliqueAI.commit_reveal import SealedChallenge
from CliqueAI.protocol import MaximumCliqueSealedGraph
from CliqueAI.transport.axon_requester import AxonRequester

DEFAULT_COMMIT_TIMEOUT = 30.0
MAX_COMMIT_TIMEOUT = 60.0


def validate_commit_timeout(commit_timeout: float) -> None:
    """Raise ValueError unless commit_timeout lies in (0, MAX_COMMIT_TIMEOUT]."""
    if (
        not math.isfinite(commit_timeout)
        or not 0 < commit_timeout <= MAX_COMMIT_TIMEOUT
    ):
        raise ValueError(f"commit_timeout must be between 0 and {MAX_COMMIT_TIMEOUT}")


@dataclass
class MinerDispatch:
    hotkey: str
    clique: list[int] = field(default_factory=list)
    commit_seconds: float | None = None
    reveal_seconds: float | None = None


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}s"


async def dispatch_challenge(
    *,
    axon_requester: AxonRequester,
    sealed: SealedChallenge,
    uuid: str,
    axons: list[bt.AxonInfo],
    time_limit: float,
    commit_timeout: float = DEFAULT_COMMIT_TIMEOUT,
) -> list[MinerDispatch]:
    """Send a sealed graph, reveal its key, and collect miner answers."""
    if not axons:
        return []
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    validate_commit_timeout(commit_timeout)

    # Commit
    commit_started = time.perf_counter()
    commit_responses = await axon_requester.forward(
        synapse=MaximumCliqueSealedGraph(
            uuid=uuid,
            sealed_matrix=sealed.payload,
            digest=sealed.digest,
            timeout=commit_timeout,
        ),
        axons=axons,
    )
    commit_seconds = time.perf_counter() - commit_started

    # Reveal
    reveal_started = time.perf_counter()
    reveal_responses = await axon_requester.forward(
        synapse=MaximumCliqueSealedGraph(
            uuid=uuid,
            key=sealed.key,
            time_limit=time_limit,
            timeout=time_limit,
        ),
        axons=axons,
    )
    reveal_seconds = time.perf_counter() - reveal_started

    miners = [
        MinerDispatch(
            hotkey=axon.hotkey,
            clique=reveal.deserialize(),
            commit_seconds=commit.dendrite.process_time,
            reveal_seconds=reveal.dendrite.process_time,
        )
        for axon, commit, reveal in zip(axons, commit_responses, reveal_responses)
    ]

    for miner in miners:
        bt.logging.debug(
            f"{miner.hotkey}: commit {_seconds(miner.commit_seconds)}, "
            f"reveal {_seconds(miner.reveal_seconds)}, "
            f"clique size {len(miner.clique)}"
        )
    bt.logging.info(
        f"Answered by {sum(1 for miner in miners if miner.clique)}/{len(axons)} "
        f"miners (commit phase {commit_seconds:.3f}s, "
        f"reveal phase {reveal_seconds:.3f}s)"
    )
    return miners
