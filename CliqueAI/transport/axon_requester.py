import asyncio

import aiohttp
import bittensor as bt

MAX_CONTENT_LENGTH = 100 * 1024  # 100 KB


class AxonRequester:
    """
    Sends requests to multiple axons asynchronously and collects responses.
    Prevent miner timeouts in dendrite forward caused by slow response verification.
    """

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
        self, url: str, headers: dict, body: dict, timeout: float
    ) -> aiohttp.ClientResponse:
        response = await self.session.post(
            url, headers=headers, json=body, timeout=timeout
        )
        return response

    async def _process_response(
        self, miner_synapse: bt.Synapse, response: aiohttp.ClientResponse
    ) -> bt.Synapse:
        if isinstance(response, Exception):
            miner_synapse = self.dendrite.process_error_message(
                synapse=miner_synapse,
                request_name=response.__class__.__name__,
                exception=response,
            )
            return miner_synapse

        # Check if the response is too large
        if int(response.headers.get("Content-Length", 0)) > MAX_CONTENT_LENGTH:
            miner_synapse = self.dendrite.process_error_message(
                synapse=miner_synapse,
                request_name=response.__class__.__name__,
                exception=ValueError(
                    f"Response too large: {response.headers.get('Content-Length')} bytes"
                ),
            )
            return miner_synapse

        try:
            json_response = await response.json()
            self.dendrite.process_server_response(
                response, json_response, miner_synapse
            )
        except Exception as e:
            miner_synapse = self.dendrite.process_error_message(
                synapse=miner_synapse,
                request_name=response.__class__.__name__,
                exception=e,
            )
        return miner_synapse

    async def forward(
        self, synapse: bt.Synapse, axons: list[bt.AxonInfo]
    ) -> list[bt.Synapse]:
        miner_synapses, requests_data = self._prepare_requests(synapse, axons)
        if not requests_data:
            return []
        responses = await asyncio.gather(
            *[self._send_request(**req) for req in requests_data],
            return_exceptions=True,
        )
        miner_synapses = await asyncio.gather(
            *[
                self._process_response(miner_synapse, response)
                for miner_synapse, response in zip(miner_synapses, responses)
            ]
        )
        return miner_synapses
