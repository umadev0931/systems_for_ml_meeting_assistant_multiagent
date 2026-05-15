"""Extractor worker agent."""
from __future__ import annotations

import httpx

from agents import config
from agents.base import Agent, CallResult
from agents.worker_util import compose_worker_input


class Extractor(Agent):
    def __init__(self, client: httpx.AsyncClient, **kw):
        super().__init__(config.AGENT_SPECS["extractor"], client, **kw)

    async def run(self, brief: str, transcript: str,
                  critique: str | None = None, revision: int = 0) -> CallResult:
        body = compose_worker_input(brief, transcript, critique)
        return await self.call(body, revision=revision)
