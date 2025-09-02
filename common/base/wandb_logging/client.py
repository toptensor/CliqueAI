import time

import httpx
from bittensor import Wallet
from common.base.consts import LAMBDA_URL
from common.base.wandb_logging.model import (
    WandbRunInitPayload,
    WandbRunInitRequest,
    WandbRunInitResponse,
    WandbRunLogData,
    WandbRunLogPayload,
    WandbRunLogRequest,
    WandbRunLogResponse,
)


class WandbClient:
    def __init__(self, wallet: Wallet, base_version: str, netuid: int):
        self.wallet = wallet
        self.netuid = netuid
        self.hotkey = wallet.hotkey.ss58_address
        self.base_version = str(base_version)
        self.client = httpx.Client(base_url=LAMBDA_URL)
        self.key = wallet.hotkey
        self.run_id = None
        self.validator_version = None

    def init(self, version: str) -> WandbRunInitResponse:
        self.validator_version = str(version)
        payload = WandbRunInitPayload(
            timestamp=time.time(),
            hotkey=self.hotkey,
            netuid=self.netuid,
            version=self.validator_version,
        )
        signature = self.key.sign(payload.model_dump_json()).hex()
        body = WandbRunInitRequest(
            payload=payload, hotkey=self.hotkey, signature=signature
        )
        response = self.client.post(
            "/wandb/init",
            json=body.model_dump(),
        )
        response.raise_for_status()
        response = WandbRunInitResponse.model_validate(response.json())
        self.run_id = response.run_id
        return response

    def log(self, data: WandbRunLogData) -> WandbRunLogResponse:
        if not self.run_id:
            raise ValueError("WandB run is not started.")
        payload = WandbRunLogPayload(
            timestamp=time.time(),
            hotkey=self.hotkey,
            netuid=self.netuid,
            version=self.validator_version,
            run_id=self.run_id,
            data=data,
        )
        signature = self.key.sign(payload.model_dump_json()).hex()
        body = WandbRunLogRequest(
            payload=payload, hotkey=self.hotkey, signature=signature
        )
        response = self.client.post(
            "/wandb/log",
            json=body.model_dump(),
        )
        response.raise_for_status()
        return WandbRunLogResponse.model_validate(response.json())

    def finish(self) -> WandbRunInitResponse:
        if not self.run_id:
            raise ValueError("WandB run is not started.")
        payload = WandbRunInitPayload(
            timestamp=time.time(),
            hotkey=self.hotkey,
            netuid=self.netuid,
            version=self.validator_version,
        )
        signature = self.key.sign(payload.model_dump_json()).hex()
        body = WandbRunInitRequest(
            payload=payload, hotkey=self.hotkey, signature=signature
        )
        response = self.client.post(
            "/wandb/finish",
            json=body.model_dump(),
        )
        response.raise_for_status()
        self.run_id = None
        return WandbRunInitResponse.model_validate(response.json())
