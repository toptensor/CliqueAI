import asyncio
import inspect
import math
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import bittensor as bt
from CliqueAI.commit_reveal.sealer import SealError, unseal
from CliqueAI.graph.codec import DEFAULT_BASE92_ALPHABET, GraphCodec
from CliqueAI.protocol import (
    MAX_SEALED_CHARS,
    MaximumCliqueSealedGraph,
)

DEFAULT_MAX_PENDING = 64
DEFAULT_MAX_PENDING_PER_HOTKEY = 10
DEFAULT_PENDING_TTL = 120.0


@dataclass
class PendingChallenge:
    sealed_matrix: str
    digest: str
    created_at: float


class CommitRevealResponder:
    """Store sealed challenges per validator hotkey and solve them after reveal.

    Pending commits are keyed by (validator hotkey, uuid), so two validators
    holding the same uuid cannot collide, and a reveal only opens a commit made
    by the same, signature-verified hotkey. A per-hotkey quota keeps one hotkey
    from filling the table and starving the other validators: a hotkey that
    exceeds it displaces its own oldest pending commit, never another's.
    """

    def __init__(
        self,
        solve: Callable[..., Any],
        max_sealed_chars: int = MAX_SEALED_CHARS,
        max_pending: int = DEFAULT_MAX_PENDING,
        max_pending_per_hotkey: int = DEFAULT_MAX_PENDING_PER_HOTKEY,
        pending_ttl: float = DEFAULT_PENDING_TTL,
    ):
        if not 0 < max_sealed_chars <= MAX_SEALED_CHARS:
            raise ValueError(
                f"max_sealed_chars must be between 1 and {MAX_SEALED_CHARS}"
            )
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if max_pending_per_hotkey <= 0:
            raise ValueError("max_pending_per_hotkey must be positive")
        if not math.isfinite(pending_ttl) or pending_ttl <= 0:
            raise ValueError("pending_ttl must be positive and finite")

        self.solve = solve
        self.codec = GraphCodec()
        self.max_sealed_chars = max_sealed_chars
        self.max_pending = max_pending
        self.max_pending_per_hotkey = max_pending_per_hotkey
        self.pending_ttl = pending_ttl
        self._pending: OrderedDict[tuple[str, str], PendingChallenge] = OrderedDict()
        self._pending_per_hotkey: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._pending)

    async def handle(
        self, synapse: MaximumCliqueSealedGraph
    ) -> MaximumCliqueSealedGraph:
        synapse.maximum_clique = []
        self._prune()

        if synapse.sealed_matrix and synapse.key:
            synapse.sealed_matrix = ""
            synapse.digest = ""
            synapse.key = ""
            bt.logging.warning(f"Challenge {synapse.uuid}: invalid commit-reveal phase")
            return synapse
        if synapse.sealed_matrix:
            return self._commit(synapse)
        if synapse.key:
            return await self._reveal(synapse)

        bt.logging.warning(f"Challenge {synapse.uuid}: missing commit and reveal data")
        return synapse

    def _commit(self, synapse: MaximumCliqueSealedGraph) -> MaximumCliqueSealedGraph:
        payload, synapse.sealed_matrix = synapse.sealed_matrix, ""
        digest, synapse.digest = synapse.digest, ""
        synapse.key = ""

        hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not hotkey:
            bt.logging.warning(
                f"Rejecting commit {synapse.uuid}: missing dendrite hotkey"
            )
            return synapse
        if not synapse.uuid or len(synapse.uuid) > 64:
            bt.logging.warning("Rejecting commit: invalid uuid")
            return synapse
        if len(payload) > self.max_sealed_chars:
            bt.logging.warning(
                f"Rejecting commit {synapse.uuid}: sealed_matrix too large, "
                f"{len(payload)} chars > {self.max_sealed_chars}"
            )
            return synapse
        if any(char not in DEFAULT_BASE92_ALPHABET for char in payload):
            bt.logging.warning(
                f"Rejecting commit {synapse.uuid}: invalid sealed_matrix"
            )
            return synapse
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            bt.logging.warning(f"Rejecting commit {synapse.uuid}: invalid digest")
            return synapse

        pending_key = (hotkey, synapse.uuid)
        if pending_key in self._pending:
            bt.logging.warning(f"Rejecting commit {synapse.uuid}: already committed")
            return synapse
        if self._pending_per_hotkey[hotkey] >= self.max_pending_per_hotkey:
            self._evict_oldest(hotkey)
        if len(self._pending) >= self.max_pending:
            bt.logging.warning(
                f"Rejecting commit {synapse.uuid}: pending limit reached"
            )
            return synapse

        self._pending[pending_key] = PendingChallenge(
            sealed_matrix=payload,
            digest=digest,
            created_at=time.monotonic(),
        )
        self._pending_per_hotkey[hotkey] += 1

        bt.logging.info(
            f"Committed challenge {synapse.uuid} from {hotkey} ({len(payload)} chars)"
        )
        return synapse

    async def _reveal(
        self, synapse: MaximumCliqueSealedGraph
    ) -> MaximumCliqueSealedGraph:
        revealed_at = time.perf_counter()
        key, synapse.key = synapse.key, ""
        synapse.sealed_matrix = ""
        synapse.digest = ""

        hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not hotkey:
            bt.logging.warning(
                f"Rejecting reveal {synapse.uuid}: missing dendrite hotkey"
            )
            return synapse
        if not synapse.uuid or len(synapse.uuid) > 64:
            bt.logging.warning("Rejecting reveal: invalid uuid")
            return synapse
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            bt.logging.warning(f"Reveal {synapse.uuid}: invalid key")
            return synapse
        if not math.isfinite(synapse.time_limit) or synapse.time_limit <= 0:
            bt.logging.warning(f"Reveal {synapse.uuid}: invalid time limit")
            return synapse

        pending_key = (hotkey, synapse.uuid)
        challenge = self._pending.get(pending_key)
        if challenge is None:
            bt.logging.warning(
                f"Reveal {synapse.uuid}: no committed challenge from {hotkey}"
            )
            return synapse

        try:
            encoded_matrix = unseal(
                uuid=synapse.uuid,
                payload=challenge.sealed_matrix,
                key=key,
                digest=challenge.digest,
            )
            adjacency_list = self.codec.matrix_to_list(
                self.codec.decode_matrix(encoded_matrix)
            )
        except (SealError, ValueError) as error:
            bt.logging.error(f"Reveal {synapse.uuid}: unseal failed: {error}")
            return synapse

        self._pop(pending_key)
        remaining = max(0.0, synapse.time_limit - (time.perf_counter() - revealed_at))
        bt.logging.info(
            f"Unsealed challenge {synapse.uuid} with {len(adjacency_list)} nodes, "
            f"{remaining:.3f}s left to solve"
        )
        if remaining <= 0:
            return synapse

        try:
            clique = await asyncio.wait_for(
                self._solve(
                    number_of_nodes=len(adjacency_list),
                    adjacency_list=adjacency_list,
                    time_limit=remaining,
                ),
                timeout=remaining,
            )
        except TimeoutError:
            bt.logging.warning(f"Reveal {synapse.uuid}: solve timed out")
            return synapse
        except Exception as error:
            bt.logging.error(f"Reveal {synapse.uuid}: solve failed: {error}")
            return synapse

        answer = list(clique or [])
        synapse.maximum_clique = answer
        bt.logging.info(
            f"Answered challenge {synapse.uuid} with clique size "
            f"{len(synapse.maximum_clique)} in {time.perf_counter() - revealed_at:.3f}s"
        )
        return synapse

    async def _solve(self, **kwargs) -> Any:
        if inspect.iscoroutinefunction(self.solve):
            return await self.solve(**kwargs)
        result = await asyncio.to_thread(self.solve, **kwargs)
        return await result if inspect.isawaitable(result) else result

    def _pop(self, pending_key: tuple[str, str]) -> None:
        if self._pending.pop(pending_key, None) is not None:
            self._release(pending_key[0])

    def _evict_oldest(self, hotkey: str) -> None:
        for pending_key in self._pending:
            if pending_key[0] == hotkey:
                del self._pending[pending_key]
                self._release(hotkey)
                bt.logging.warning(
                    f"Evicting pending commit {pending_key[1]} from {hotkey}: "
                    "per-hotkey pending limit reached"
                )
                return

    def _release(self, hotkey: str) -> None:
        self._pending_per_hotkey[hotkey] -= 1
        if self._pending_per_hotkey[hotkey] <= 0:
            del self._pending_per_hotkey[hotkey]

    def _prune(self) -> None:
        expires_before = time.monotonic() - self.pending_ttl
        while self._pending:
            pending_key, challenge = next(iter(self._pending.items()))
            if challenge.created_at > expires_before:
                break
            del self._pending[pending_key]
            self._release(pending_key[0])
