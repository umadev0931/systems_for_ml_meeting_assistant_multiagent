"""Scrape vLLM's Prometheus /metrics endpoint.

vLLM exposes KV-cache, queue, throughput, and prefix-cache metrics in Prometheus
text format. We snapshot the endpoint before/after a run (for counters: take the
delta) and sample gauges during the run. Metric *names* have shifted across vLLM
versions, so we match a set of known aliases and degrade gracefully when a
metric is absent (e.g. on a mock server or older build).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

# name -> list of acceptable Prometheus metric names (newest first)
_GAUGES = {
    "gpu_cache_usage_perc": ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"],
    "num_requests_running": ["vllm:num_requests_running"],
    "num_requests_waiting": ["vllm:num_requests_waiting"],
}
_COUNTERS = {
    "prompt_tokens_total":      ["vllm:prompt_tokens_total"],
    "generation_tokens_total":  ["vllm:generation_tokens_total"],
    "prefix_cache_queries_total": [
        "vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"],
    "prefix_cache_hits_total": [
        "vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
    "num_requests_total": ["vllm:request_success_total", "vllm:num_requests_total"],
}

_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE+\-.]+)\s*$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum all samples for each metric name (ignoring labels)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, _labels, val = m.group(1), m.group(2), m.group(3)
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            continue
    return out


def _pick(raw: dict[str, float], aliases: list[str]) -> Optional[float]:
    for a in aliases:
        if a in raw:
            return raw[a]
    return None


@dataclass
class MetricSnapshot:
    available: bool
    gauges: dict[str, Optional[float]] = field(default_factory=dict)
    counters: dict[str, Optional[float]] = field(default_factory=dict)


class VllmMetricsScraper:
    def __init__(self, metrics_url: str, client: httpx.Client | None = None):
        self.url = metrics_url
        self._client = client or httpx.Client(timeout=5.0)

    def snapshot(self) -> MetricSnapshot:
        try:
            resp = self._client.get(self.url)
            resp.raise_for_status()
            raw = parse_prometheus(resp.text)
        except (httpx.HTTPError, Exception):
            return MetricSnapshot(available=False)
        gauges = {k: _pick(raw, al) for k, al in _GAUGES.items()}
        counters = {k: _pick(raw, al) for k, al in _COUNTERS.items()}
        return MetricSnapshot(available=True, gauges=gauges, counters=counters)

    @staticmethod
    def counter_delta(before: MetricSnapshot, after: MetricSnapshot) -> dict[str, Optional[float]]:
        out: dict[str, Optional[float]] = {}
        for k in _COUNTERS:
            b, a = before.counters.get(k), after.counters.get(k)
            out[k] = (a - b) if (b is not None and a is not None) else None
        # derived server-side prefix-cache hit rate over the run window
        q = out.get("prefix_cache_queries_total")
        h = out.get("prefix_cache_hits_total")
        out["server_prefix_cache_hit_rate"] = (h / q) if (q and q > 0 and h is not None) else None
        return out

    def close(self) -> None:
        self._client.close()
