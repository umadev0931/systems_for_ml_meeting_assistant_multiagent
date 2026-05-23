"""Benchmark runner.

Sweeps transcript_size x pipeline_mode x repeats against the single 8B vLLM
server, capturing token, latency, and system metrics for each run. Writes one
JSON file per run into results/runs/.

Usage:
    python -m bench.runner --sizes small medium large \
        --modes sequential parallel langgraph --repeats 3

Transcripts are read from data/normalized/<size>.txt (run
scripts/generate_transcripts.py first, or pass --transcript-dir).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import httpx

from agents import config
from bench.metrics import build_run_record
from bench.system_sampler import SystemSampler
from bench.trace import Tracer
from bench.vllm_metrics import VllmMetricsScraper
from pipeline import PIPELINES


def _read_transcript(transcript_dir: str, size: str) -> str:
    path = os.path.join(transcript_dir, f"{size}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Transcript not found: {path}. Run scripts/generate_transcripts.py first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


async def run_one(mode: str, size: str, transcript: str, repeat: int,
                  client: httpx.AsyncClient, scraper: VllmMetricsScraper | None,
                  stream: bool) -> dict:
    tracer = Tracer()
    before = scraper.snapshot() if scraper else None

    t0 = time.perf_counter()
    with SystemSampler(scraper=scraper) as sampler:
        pipeline_fn = PIPELINES[mode]
        result = await pipeline_fn(transcript, size, client, tracer, stream)
    e2e_wall_s = time.perf_counter() - t0

    after = scraper.snapshot() if scraper else None
    if scraper and before and after:
        counter_delta = VllmMetricsScraper.counter_delta(before, after)
        server_available = before.available and after.available
    else:
        counter_delta, server_available = {}, False

    record = build_run_record(
        result, e2e_wall_s=e2e_wall_s, samples=sampler.samples,
        counter_delta=counter_delta, server_available=server_available, repeat=repeat)
    record["spans"] = tracer.to_dict()["spans"]
    return record


async def main_async(args) -> None:
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    scraper = None if args.no_server_metrics else VllmMetricsScraper(config.VLLM_METRICS_URL)

    transcripts = {s: _read_transcript(args.transcript_dir, s) for s in args.sizes}

    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    async with httpx.AsyncClient(limits=limits) as client:
        for size in args.sizes:
            for mode in args.modes:
                for r in range(args.repeats):
                    label = f"{size}__{mode}__rep{r}"
                    print(f"[run] {label} ...", flush=True)
                    try:
                        record = await run_one(mode, size, transcripts[size], r,
                                               client, scraper, not args.no_stream)
                    except Exception as exc:  # keep the sweep alive
                        print(f"[run] {label} FAILED: {type(exc).__name__}: {exc}")
                        record = {"mode": mode, "transcript_size": size, "repeat": r,
                                  "errored": True, "error": f"{type(exc).__name__}: {exc}"}
                    out_path = os.path.join(config.RUNS_DIR, f"{label}.json")
                    with open(out_path, "w", encoding="utf-8") as fh:
                        json.dump(record, fh, indent=2, default=str)
                    et = record.get("e2e_wall_s")
                    print(f"[run] {label} done"
                          + (f" ({et:.2f}s, {record.get('total_tokens')} tok, "
                             f"prefix-hit={record.get('prefix_cache_hit_ratio'):.2%})"
                             if et else ""), flush=True)

    if scraper:
        scraper.close()
    print(f"\nWrote run records to {config.RUNS_DIR}")
    print("Aggregate + plot with: python -m bench.plot")


def parse_args():
    p = argparse.ArgumentParser(description="Meeting-assistant benchmark runner")
    p.add_argument("--sizes", nargs="+", default=list(config.TRANSCRIPT_SIZES))
    p.add_argument("--modes", nargs="+", default=list(config.PIPELINE_MODES),
                   choices=list(config.PIPELINE_MODES))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--transcript-dir", default=config.NORMALIZED_DIR)
    p.add_argument("--no-server-metrics", action="store_true",
                   help="Skip scraping vLLM /metrics (host metrics still sampled).")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable streaming (TTFT will be unavailable).")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
