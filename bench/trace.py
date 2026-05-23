"""Lightweight tracing: JSON-line spans collected per run.

A `Tracer` accumulates spans (one per agent call plus pipeline-level spans) and
serialises them to a single JSON object per benchmark run. Keeping traces as
plain dicts keeps downstream pandas analysis trivial.
"""
from __future__ import annotations

import time
import contextlib
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Span:
    name: str
    kind: str                # "agent" | "pipeline" | "phase"
    start_ts: float
    end_ts: float
    duration_s: float
    attrs: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects spans for one pipeline execution."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._t0 = time.perf_counter()

    @contextlib.contextmanager
    def span(self, name: str, kind: str = "phase", **attrs: Any):
        start = time.perf_counter()
        start_wall = time.time()
        rec: dict[str, Any] = {}
        try:
            yield rec  # caller may stuff extra attrs into rec
        finally:
            end = time.perf_counter()
            merged = {**attrs, **rec}
            self.spans.append(
                Span(
                    name=name,
                    kind=kind,
                    start_ts=start_wall,
                    end_ts=start_wall + (end - start),
                    duration_s=end - start,
                    attrs=merged,
                )
            )

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def to_dict(self) -> dict[str, Any]:
        return {"spans": [asdict(s) for s in self.spans]}
