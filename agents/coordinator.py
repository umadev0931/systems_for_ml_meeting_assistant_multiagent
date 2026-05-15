"""Coordinator agent: triage + task framing.

Ingests the raw transcript and emits a compact "brief" (context boundaries,
participant list, decision cues) that is prepended to each worker's input so
the workers operate on a shared, normalised framing.
"""
from __future__ import annotations

import httpx

from agents import config
from agents.base import Agent, CallResult


class Coordinator(Agent):
    def __init__(self, client: httpx.AsyncClient, **kw):
        super().__init__(config.AGENT_SPECS["coordinator"], client, **kw)

    async def run(self, transcript: str) -> CallResult:
        user = (
            "Here is a raw meeting transcript. Produce a short coordination brief "
            "that other agents will use as shared context.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )
        return await self.call(user)
