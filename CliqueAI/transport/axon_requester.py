import asyncio
import json
import time

import aiohttp
import bittensor as bt

MAX_CONTENT_LENGTH = 100 * 1024  # 100 KB


class AxonRequester:
    """Sends requests to multiple axons and collects their responses."""

    def __init__(self, session: aiohttp.ClientSession, dendrite: bt.Dendrite):
        self.session = session
        self.dendrite = dendrite

    def _prepare_requests(
        self, synapse: bt.Synapse, axons: list[bt.AxonInfo]
    ) -> tuple[list[bt.Synapse], list[dict]]:
        request_name = synapse.__class__.__name__
        miner_synapses = []
        requests_data = []
        for axon in axons:
            url = self.dendrite._get_endpoint_url(axon, request_name=request_name)
            miner_synapse = self.dendrite.preprocess_synapse_for_request(
                axon, synapse.model_copy(), synapse.timeout
            )
            miner_synapses.append(miner_synapse)
            requests_data.append(
                {
                    "url": url,
                    "headers": miner_synapse.to_headers(),
                    "body": miner_synapse.model_dump(),
                    "timeout": synapse.timeout,
                }
            )
        return miner_synapses, requests_data

    async def _send_request(
        self,
        miner_synapse: bt.Synapse,
        url: str,
        headers: dict,
        body: dict,
        timeout: float,
    ) -> bt.Synapse:
        started = time.perf_counter()
        try:
            async with self.session.post(
                url, headers=headers, json=body, timeout=timeout
            ) as response:
                miner_synapse = await self._process_response(miner_synapse, response)
        except Exception as error:
            miner_synapse = self.dendrite.process_error_message(
                synapse=miner_synapse,
                request_name=error.__class__.__name__,
                exception=error,
            )
        miner_synapse.dendrite.process_time = time.perf_counter() - started
        return miner_synapse

    async def _process_response(
        self, miner_synapse: bt.Synapse, response: aiohttp.ClientResponse
    ) -> bt.Synapse:
        try:
            json_response = await self._read_json(response)
            self.dendrite.process_server_response(
                response, json_response, miner_synapse
            )
        except Exception as error:
            miner_synapse = self.dendrite.process_error_message(
                synapse=miner_synapse,
                request_name=response.__class__.__name__,
                exception=error,
            )
        return miner_synapse

    async def _read_json(self, response: aiohttp.ClientResponse) -> dict:
        if (
            response.content_length is not None
            and response.content_length > MAX_CONTENT_LENGTH
        ):
            raise ValueError(f"Response too large: {response.content_length} bytes")

        content = bytearray()
        async for chunk in response.content.iter_chunked(16 * 1024):
            if len(content) + len(chunk) > MAX_CONTENT_LENGTH:
                response.close()
                raise ValueError("Response exceeds 100 KB")
            content.extend(chunk)
        return json.loads(content)

    async def forward(
        self, synapse: bt.Synapse, axons: list[bt.AxonInfo]
    ) -> list[bt.Synapse]:
        miner_synapses, requests_data = self._prepare_requests(synapse, axons)
        if not requests_data:
            return []
        return await asyncio.gather(
            *[
                self._send_request(miner_synapse, **request)
                for miner_synapse, request in zip(miner_synapses, requests_data)
            ]
        )
