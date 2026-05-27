"""Generate synthetic meeting transcripts at controlled sizes.

GROUND TRUTH HAS BEEN REMOVED: this script emits ONLY transcript text (no
summaries, action-item labels, or any reference JSON). Output:
    data/synthetic/<size>.txt    raw generated transcript
    data/normalized/<size>.txt   whitespace-normalised transcript (used by bench)

Two generation paths:
  - default (model): asks the 8B vLLM server to write a transcript using
    prompts/transcript_generator.md, sized to each bucket's target tokens.
  - --offline: procedurally assembles a transcript locally (no server needed),
    useful for smoke tests / CI / no-GPU environments.

Usage:
    python scripts/generate_transcripts.py --sizes small medium large
    python scripts/generate_transcripts.py --offline
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys

# allow running as a script: add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import config  # noqa: E402

SPEAKERS = ["Alice", "Bob", "Carol", "Dan", "Erin", "Frank", "Grace", "Heidi"]
TOPICS = ["the Q3 roadmap", "the migration to the new data warehouse",
          "the customer churn spike", "the hiring plan", "the incident postmortem",
          "the pricing experiment", "the mobile redesign", "the vendor contract"]
UTTERANCES = [
    "I think we should prioritise this before the end of the sprint.",
    "Can someone own that and report back next week?",
    "Let's hold off until we have the numbers from analytics.",
    "I'll take the first pass and share a doc by Friday.",
    "We agreed last time, so let's just move forward with option B.",
    "There's a dependency on the platform team we need to flag.",
    "Honestly I'm not sure the deadline is realistic given the scope.",
    "Okay, decision made: we ship the limited beta first.",
    "Quick tangent, did anyone see the latency dashboard this morning?",
    "Let's circle back to that, it's out of scope for today.",
    "I'll sync with Dan offline and we'll bring a proposal next meeting.",
    "Action item for me: update the runbook and notify support.",
]


def _approx_tokens(text: str) -> int:
    # rough heuristic ~0.75 words/token; good enough for sizing buckets
    return int(len(text.split()) / 0.75)


def offline_transcript(spec: config.TranscriptSize, seed: int = 0) -> str:
    rng = random.Random(seed + hash(spec.label) % 10_000)
    speakers = SPEAKERS[: spec.n_speakers]
    topics = rng.sample(TOPICS, k=min(spec.n_topics, len(TOPICS)))
    lines: list[str] = []
    lines.append(f"{speakers[0]}: Thanks everyone for joining. Today we cover "
                 + ", ".join(topics) + ".")
    while _approx_tokens("\n".join(lines)) < spec.target_tokens:
        topic = rng.choice(topics)
        lines.append(f"{speakers[0]}: Let's talk about {topic}.")
        for _ in range(rng.randint(3, 7)):
            spk = rng.choice(speakers)
            lines.append(f"{spk}: {rng.choice(UTTERANCES)}")
    lines.append(f"{speakers[0]}: Great, let's wrap up. Thanks all.")
    return "\n".join(lines)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def model_transcript(spec: config.TranscriptSize) -> str:
    """Generate via the vLLM server (synchronous, one shot)."""
    import httpx
    with open(os.path.join(config.PROMPTS_DIR, "transcript_generator.md"),
              encoding="utf-8") as fh:
        system = fh.read()
    user = (f"Generate a meeting transcript with {spec.n_speakers} speakers, "
            f"{spec.n_topics} distinct topics, and approximately "
            f"{spec.target_tokens} tokens of dialogue.")
    payload = {
        "model": config.MODEL_NAME,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.8,
        "max_tokens": min(spec.target_tokens + 1024, 16384),
        "stream": False,
    }
    url = f"{config.VLLM_BASE_URL}/v1/chat/completions"
    resp = httpx.post(url, json=payload, timeout=config.REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", default=list(config.TRANSCRIPT_SIZES),
                   choices=list(config.TRANSCRIPT_SIZES))
    p.add_argument("--offline", action="store_true",
                   help="Generate locally without contacting the model server.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(config.SYNTHETIC_DIR, exist_ok=True)
    os.makedirs(config.NORMALIZED_DIR, exist_ok=True)

    for size in args.sizes:
        spec = config.TRANSCRIPT_SIZES[size]
        if args.offline:
            raw = offline_transcript(spec, seed=args.seed)
        else:
            try:
                raw = model_transcript(spec)
            except Exception as exc:
                print(f"[gen] model generation failed ({exc}); falling back to offline.")
                raw = offline_transcript(spec, seed=args.seed)
        norm = normalize(raw)
        with open(os.path.join(config.SYNTHETIC_DIR, f"{size}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(raw)
        with open(os.path.join(config.NORMALIZED_DIR, f"{size}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(norm)
        print(f"[gen] {size}: ~{_approx_tokens(norm)} tokens "
              f"-> data/normalized/{size}.txt")


if __name__ == "__main__":
    main()
