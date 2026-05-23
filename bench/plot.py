"""Generate all figures for the report from results/runs/.

Produces PNGs in results/figures/ plus a tidy CSV summary. Each figure groups
by transcript size and pipeline mode. Run after bench.runner.

    python -m bench.plot
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agents import config
from bench.metrics import aggregate, summarize

MODE_ORDER = list(config.PIPELINE_MODES)

def _make_abbrev(sizes: list[str]) -> dict[str, str]:
    """Build short abbreviations from size names, e.g. large_finance -> LF."""
    abbrevs = {}
    for s in sizes:
        parts = s.split("_")
        abbrev = "".join(p[0].upper() for p in parts if p)
        # ensure uniqueness by appending extra chars if needed
        base, n = abbrev, 2
        while abbrev in abbrevs.values():
            abbrev = base + str(n)
            n += 1
        abbrevs[s] = abbrev
    return abbrevs


def _bar_by_size_mode(summary: pd.DataFrame, column: str, ylabel: str,
                      title: str, out: str, scale: float = 1.0) -> None:
    if column not in summary.columns or summary[column].dropna().empty:
        print(f"[plot] skip {column} (no data)")
        return
    sizes = sorted(summary["transcript_size"].unique())
    modes = [m for m in MODE_ORDER if m in summary["mode"].unique()]
    abbrevs = _make_abbrev(sizes)
    x = range(len(sizes))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, mode in enumerate(modes):
        vals = []
        for s in sizes:
            row = summary[(summary["transcript_size"] == s) & (summary["mode"] == mode)]
            vals.append(row[column].mean() * scale if not row.empty else 0)
        ax.bar([xi + i * width for xi in x], vals, width=width, label=mode)
    ax.set_xticks([xi + width * (len(modes) - 1) / 2 for xi in x])
    ax.set_xticklabels([abbrevs[s] for s in sizes])
    ax.set_xlabel("Transcript")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Pipeline")
    ax.grid(axis="y", alpha=0.3)
    key = "\n".join(f"{v} = {k}" for k, v in abbrevs.items())
    ax.text(1.01, 0.99, key, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")


FIGURES = [
    ("e2e_wall_s", "End-to-end latency (s)", "End-to-end latency by size & pipeline", "latency_e2e.png"),
    ("ttft_mean_s", "Mean TTFT (s)", "Mean time-to-first-token (prefill proxy)", "ttft.png"),
    ("throughput_tok_per_s", "Tokens / s", "Aggregate throughput", "throughput.png"),
    ("delta_server_prefix_cache_hit_rate", "KV prefix-cache hit rate", "KV prefix-cache reuse by size & pipeline", "prefix_cache_hit.png"),
    ("total_tokens", "Total tokens", "Token usage (prompt + completion)", "token_usage.png"),
    ("vllm_gpu_cache_usage_perc_max", "Max KV-cache usage (%)", "Peak vLLM KV-cache occupancy", "kv_cache_usage.png", 100),
    ("vllm_num_requests_running_max", "Max concurrent running", "Server concurrency (serialisation check)", "concurrency.png"),
    ("orchestrator_rss_mb_max", "Peak RSS (MB)", "Orchestrator memory footprint", "memory_rss.png"),
    ("host_cpu_percent_mean", "Mean CPU %", "Host CPU utilisation", "cpu.png"),
]


def main() -> None:
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    df = aggregate(config.RUNS_DIR)
    if df.empty:
        print("No runs found in", config.RUNS_DIR)
        return

    summary = summarize(df)
    csv_path = os.path.join(config.FIGURES_DIR, "summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"[plot] wrote {csv_path}")

    for entry in FIGURES:
        col, ylabel, title, fname = entry[:4]
        scale = entry[4] if len(entry) > 4 else 1.0
        _bar_by_size_mode(summary, col, ylabel, title,
                          os.path.join(config.FIGURES_DIR, fname), scale=scale)

    print("\nSummary (mean over repeats):")
    cols = [c for c in ("transcript_size", "mode", "e2e_wall_s", "total_tokens",
                        "delta_server_prefix_cache_hit_rate", "throughput_tok_per_s")
            if c in summary.columns]
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
