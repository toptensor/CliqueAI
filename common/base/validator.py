import argparse
import asyncio
import copy
import threading
import time
from traceback import print_exception
from typing import List, Union

import bittensor as bt
import numpy as np
from common.base import validator_int_version
from common.base.middleware.bypass_axon_middleware import replace_axon_middleware
from common.base.neuron import BaseNeuron
from common.base.utils.signature import verify_signature
from common.base.utils.state_storage import (
    get_all_validator_state,
    load_latest_validator_state,
    save_validator_state,
)
from common.base.wandb_logging.client import WandbClient
from common.utils.autoupdate import update_repo_if_needed
from common.utils.config import add_validator_args
from fastapi import HTTPException


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"
    POWER_TARGET_TOP_HALF_SHARE: float = 0.80
    POWER_MAX_GAMMA: float = 32.0

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        self.dendrite = bt.Dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.ema_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)  # Debiased EMA
        self.ema_step_count = np.zeros(self.metagraph.n, dtype=np.int32)

        bt.logging.info("load_state()")
        self.load_state()

        self.axon = bt.Axon(
            wallet=self.wallet,
            config=self.config,
        )

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()
        self.wandb_client = WandbClient(
            wallet=self.wallet,
            base_version=self.base_version,
            netuid=self.config.netuid,
        )

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.Axon(wallet=self.wallet, config=self.config)

            try:
                bt.logging.info(
                    f"Serving validator axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )

                # Add the bypass axon middleware to the axon app.
                replace_axon_middleware(
                    axon=self.axon,
                    exclude_paths=[
                        "/validator/state",  # Exclude the validator state endpoint from the middleware.
                    ],
                )

                # Add the validator state endpoint to the axon router.
                self.axon.router.add_api_route(
                    path="/validator/state",
                    endpoint=self.get_validator_state,
                    methods=["POST"],
                )
                self.axon.app.include_router(self.axon.router)

                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.network} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()
        self.resync_metagraph()  # update snapshot

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
            self.axon.start()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                if not self.should_exit:
                    self.loop.run_until_complete(self.concurrent_forward())

                # Check autoupdate status.
                if self.config.neuron.autoupdate:
                    bt.logging.info("Checking for updates...")
                    if update_repo_if_needed():
                        raise KeyboardInterrupt()

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            self.should_exit = True

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))
            self.should_exit = True

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

        if self.wandb_client.run_id:
            bt.logging.info("Stopping WandB client.")
            self.wandb_client.finish()
            bt.logging.info("WandB client stopped.")

    def _sigmoid_weight(self, normalized_incentives: np.ndarray, midpoint: float, steepness: float) -> np.ndarray:
        """
        Apply sigmoid scaling to normalized incentives.
        """
        x = (normalized_incentives - midpoint) / steepness
        return 1 / (1 + np.exp(-x))

    def _top_half_share(self, weights: np.ndarray, rank_scores: np.ndarray) -> float:
        total = np.sum(weights)
        if total <= 0:
            return 0.0
        order = np.argsort(-rank_scores)
        top = order[: len(order) // 2]
        return float(np.sum(weights[top]) / total)

    def _power_weight(
        self,
        sigmoid_incentives: np.ndarray,
        rank_scores: np.ndarray,
        target_top_half_share: float,
    ) -> tuple[np.ndarray, float]:
        active = sigmoid_incentives > 0
        if not np.any(active):
            return np.zeros_like(sigmoid_incentives, dtype=np.float64), 0.0

        def transform(gamma: float) -> np.ndarray:
            weights = np.zeros_like(sigmoid_incentives, dtype=np.float64)
            weights[active] = np.power(sigmoid_incentives[active], gamma)
            return weights

        lo = 0.0
        max_gamma = self.POWER_MAX_GAMMA
        hi = min(1.0, max_gamma)
        while (
            self._top_half_share(transform(hi), rank_scores) < target_top_half_share
            and hi < max_gamma
        ):
            hi = min(hi * 2.0, max_gamma)

        for _ in range(80):
            mid = (lo + hi) / 2.0
            if self._top_half_share(transform(mid), rank_scores) < target_top_half_share:
                lo = mid
            else:
                hi = mid

        return transform(hi), hi

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        LAMBDA_WEIGHTS = 1.0

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )
            # Replace NaN values with 0 to avoid issues in weight calculations.
            self.scores = np.nan_to_num(self.scores, nan=0.0)

        # Normalize the scores to ensure they are in the range [0, 1].
        min_val = np.min(self.scores)
        max_val = np.max(self.scores)
        range_val = max_val - min_val

        if range_val == 0:
            normalized = np.zeros_like(self.scores)
        else:
            normalized = (self.scores - min_val) / range_val

        # Apply sigmoid scaling with power scaling.
        zero_mask = (normalized == 0)
        nonzero_normalized = normalized[~zero_mask]
        if nonzero_normalized.size > 0:
            midpoint = np.median(nonzero_normalized)
            steepness = max(np.percentile(nonzero_normalized, 75) - np.percentile(nonzero_normalized, 25), 0.1)  # IQR
            sigmoid_scores = self._sigmoid_weight(normalized, midpoint=midpoint, steepness=steepness)
            sigmoid_scores[zero_mask] = 0.0
            sigmoid_weights, power_gamma = self._power_weight(
                sigmoid_scores,
                normalized,
                self.POWER_TARGET_TOP_HALF_SHARE,
            )
            bt.logging.debug(
                f"Power weight gamma={power_gamma:.9f}, target_top_half_share={self.POWER_TARGET_TOP_HALF_SHARE:.3f}"
            )
        else:
            sigmoid_weights = normalized

        # Combine the sigmoid weights with a lambda to determine the final weights.
        weights = LAMBDA_WEIGHTS * sigmoid_weights
        weights[0] = (1.0 - LAMBDA_WEIGHTS) * sum(sigmoid_weights)
        uids = np.asarray(self.metagraph.uids, dtype=np.int64)
        bt.logging.debug("weights", weights)
        bt.logging.debug("uids", uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uids,
            weights=weights,
            wait_for_finalization=False,
            wait_for_inclusion=True,
            version_key=validator_int_version,
        )
        if result is True:
            self.last_set_weight = self.block
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                # hotkey has been replaced
                self.ema_scores[uid] = 0
                self.scores[uid] = 0
                self.ema_step_count[uid] = 0

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the ema scores.
            new_ema_scores = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.ema_scores))
            new_ema_scores[:min_len] = self.ema_scores[:min_len]
            self.ema_scores = new_ema_scores

            # Update the size of the scores.
            new_scores = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_scores[:min_len] = self.scores[:min_len]
            self.scores = new_scores

            # Update the size of the ema_step_count.
            new_ema_step_count = np.zeros((self.metagraph.n), dtype=np.int32)
            min_len = min(len(self.hotkeys), len(self.ema_step_count))
            new_ema_step_count[:min_len] = self.ema_step_count[:min_len]
            self.ema_step_count = new_ema_step_count

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Update scores with rewards produced by this step.
        alpha: float = self.config.neuron.ema_alpha
        self.ema_step_count[uids_array] += 1
        bt.logging.debug(f"EMA step count: {self.ema_step_count}")
        self.ema_scores[uids_array] = (
            alpha * rewards + (1 - alpha) * self.ema_scores[uids_array]
        )
        correction = 1 - np.power(1 - alpha, self.ema_step_count)
        self.scores = np.divide(
            self.ema_scores,
            correction,
            out=np.zeros_like(self.ema_scores),
            where=correction != 0,
        )
        bt.logging.debug(f"Updated EMA scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator."""
        bt.logging.info("Saving validator state.")

        save_validator_state(
            path=self.config.neuron.full_path,
            step=self.step,
            ema_scores=self.ema_scores,
            hotkeys=self.hotkeys,
            ema_step_count=self.ema_step_count,
        )
        bt.logging.info(f"Validator state saved. step: {self.step}.")

    def load_state(self):
        """Loads the state of the validator."""
        bt.logging.info("Loading validator state.")

        try:
            step, ema_scores, hotkeys, ema_step_count = load_latest_validator_state(
                path=self.config.neuron.full_path
            )
            self.step = self.init_step = step
            self.ema_scores = ema_scores
            self.hotkeys = hotkeys
            self.ema_step_count = ema_step_count

            bt.logging.info(f"Loaded validator state. step: {self.step}.")
        except FileNotFoundError:
            bt.logging.warning(
                "No previous validator state found. Starting from scratch."
            )

    def _get_owner_hotkey(self) -> str:
        if hasattr(self.subtensor, "get_subnet_owner_hotkey"):
            try:
                owner_hotkey = self.subtensor.get_subnet_owner_hotkey(
                    netuid=self.config.netuid,
                    block=self.block,
                )
                if owner_hotkey:
                    return owner_hotkey
            except TypeError:
                try:
                    owner_hotkey = self.subtensor.get_subnet_owner_hotkey(
                        netuid=self.config.netuid
                    )
                    if owner_hotkey:
                        return owner_hotkey
                except Exception:
                    pass
            except Exception:
                pass

        owner_hotkey = getattr(self.metagraph, "owner_hotkey", None)
        if owner_hotkey:
            return owner_hotkey

        hotkeys = getattr(self.metagraph, "hotkeys", [])
        if hotkeys:
            return hotkeys[0]

        raise ValueError("Unable to determine subnet owner hotkey.")

    async def get_validator_state(
        self,
        timestamp: float,
        owner_signature: str,
        num: int | None = None,
    ) -> list[tuple[int, list[float], list[str], list[int]]]:
        bt.logging.info("Received request for validator state.")
        now = time.time()
        if abs(now - timestamp) > 60:
            bt.logging.warning("Request expired.")
            raise HTTPException(status_code=400, detail="Request expired.")
        if num is not None and num <= 0:
            raise HTTPException(status_code=400, detail="num must be greater than 0.")

        try:
            owner_hotkey = self._get_owner_hotkey()
        except Exception as exc:
            detail = (
                "Failed to determine subnet owner hotkey: "
                f"{type(exc).__name__}: {exc}"
            )
            bt.logging.error(detail)
            raise HTTPException(status_code=500, detail=detail) from exc

        if not verify_signature(owner_signature, timestamp, owner_hotkey):
            bt.logging.warning("Invalid owner signature.")
            raise HTTPException(status_code=403, detail="Invalid owner signature.")

        try:
            return get_all_validator_state(
                path=self.config.neuron.full_path,
                num=num,
            )
        except Exception as exc:
            detail = (
                "Failed to load validator state: "
                f"{type(exc).__name__}: {exc}"
            )
            bt.logging.error(detail)
            raise HTTPException(status_code=500, detail=detail) from exc
