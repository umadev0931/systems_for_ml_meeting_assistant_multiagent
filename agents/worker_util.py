"""Shared helpers for worker agents."""
from __future__ import annotations


def compose_worker_input(brief: str, transcript: str, critique: str | None) -> str:
    """Build the worker prompt body.

    The (brief + transcript) prefix is identical across Summarizer, Extractor,
    and Drafter so that vLLM prefix KV-cache reuse is observable when the three
    run concurrently.
    """
    parts = [f"COORDINATION BRIEF:\n{brief}", f"TRANSCRIPT:\n{transcript}"]
    if critique:
        parts.append(
            "A reviewer flagged the following issues with your previous output. "
            f"Revise to address them:\n{critique}"
        )
    return "\n\n".join(parts)
