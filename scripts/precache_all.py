"""
Pre-cache all 33 sample transcripts for all 3 LLMs across 4 cache types:
  1. "workflow"      — V3 (17-node) pipeline result
  2. "workflow_v1"   — V1 (5-node)  pipeline result
  3. "v1_comparison" — V1 baseline run saved alongside V3 (powers Gains tab)
  4. benchmark       — per-model summarization + QA benchmark entry

Run with real LLMs:
    MOCK_LLM=false venv/bin/python scripts/precache_all.py

Run with mocks (zero API cost, populates cache structure only):
    MOCK_LLM=true  venv/bin/python scripts/precache_all.py

Already-cached entries are skipped automatically — safe to re-run.
"""

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("precache")

from agents.intake_agent import IntakeAgent
from agents.summarization_agent import SummarizationAgent
from agents.quality_score_agent import QualityScoreAgent
from workflow.langgraph_flow import create_workflow, run_workflow
from workflow.langgraph_flow_v1 import create_workflow_v1, run_workflow_v1
from utils.cache import get_cached, save_cache
from utils.validation import sanitize_transcript

SAMPLE_DIR = ROOT / "data" / "sample_transcripts"
MODELS = ["claude", "gpt4", "gemini"]
SAMPLE_FILES = sorted(SAMPLE_DIR.glob("*.json"))


def load_sample(path: Path) -> tuple[str, str]:
    """Return (call_id, transcript_text)."""
    with open(path) as f:
        data = json.load(f)
    return data.get("call_id", path.stem.upper()), data.get("transcript", "")


# ── Cache helpers ──────────────────────────────────────────────────────────────

def precache_v3(transcript: str, call_id: str, model: str) -> bool:
    """Run V3 workflow and cache under 'workflow'. Returns True if newly cached."""
    if get_cached(transcript, model, "workflow"):
        return False
    intake = IntakeAgent()
    call_input = intake.process(transcript_text=transcript, metadata={"call_id": call_id})
    wf = create_workflow(llm_name=model)
    result = run_workflow(wf, call_input, llm_name=model)
    data = result.model_dump()
    data["_cache_type"] = "workflow"
    data["_llm_name"] = model
    save_cache(transcript, model, "workflow", data)
    return True


def precache_v1(transcript: str, call_id: str, model: str) -> bool:
    """Run V1 workflow and cache under 'workflow_v1'. Returns True if newly cached."""
    if get_cached(transcript, model, "workflow_v1"):
        return False
    intake = IntakeAgent()
    call_input = intake.process(transcript_text=transcript, metadata={"call_id": call_id})
    wf = create_workflow_v1(llm_name=model)
    result = run_workflow_v1(wf, call_input, llm_name=model)
    data = result.model_dump()
    data["_cache_type"] = "workflow_v1"
    data["_llm_name"] = model
    save_cache(transcript, model, "workflow_v1", data)
    return True


def precache_v1_comparison(transcript: str, call_id: str, model: str) -> bool:
    """Run V1 and cache under 'v1_comparison' (used by V1→V3 Gains tab). Returns True if newly cached."""
    if get_cached(transcript, model, "v1_comparison"):
        return False
    # Reuse workflow_v1 result if already cached to avoid a second run
    v1_cached = get_cached(transcript, model, "workflow_v1")
    if v1_cached:
        data = dict(v1_cached)
        data["_cache_type"] = "v1_comparison"
        data["_llm_name"] = model
        save_cache(transcript, model, "v1_comparison", data)
        return True
    # Otherwise run V1 fresh
    intake = IntakeAgent()
    call_input = intake.process(transcript_text=transcript, metadata={"call_id": call_id})
    wf = create_workflow_v1(llm_name=model)
    result = run_workflow_v1(wf, call_input, llm_name=model)
    data = result.model_dump()
    data["_cache_type"] = "v1_comparison"
    data["_llm_name"] = model
    save_cache(transcript, model, "v1_comparison", data)
    return True


def precache_benchmark(transcript: str, call_id: str, model: str) -> bool:
    """Run summarization + QA benchmark entry. Returns True if newly cached."""
    cache_key = f"benchmark_full_benchmark_{model}"
    if get_cached(transcript, model, cache_key):
        return False
    t0 = time.time()
    summarizer = SummarizationAgent(llm_name=model)
    scorer = QualityScoreAgent(llm_name=model)
    summary = summarizer.process(call_id=call_id, transcript=transcript)
    qa = scorer.process(call_id=call_id, transcript=transcript)
    save_cache(transcript, model, cache_key, {
        "summary": summary.model_dump(),
        "qa": qa.model_dump(),
        "timing": round(time.time() - t0, 2),
        "token_counts": {},
        "call_id": call_id,
        "_cache_type": cache_key,
        "_llm_name": model,
    })
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Found {len(SAMPLE_FILES)} sample files · {len(MODELS)} models → "
                f"up to {len(SAMPLE_FILES) * len(MODELS) * 4} cache entries")

    counters = {"workflow": 0, "workflow_v1": 0, "v1_comparison": 0, "benchmark": 0}
    errors = []

    for i, path in enumerate(SAMPLE_FILES, 1):
        call_id, transcript = load_sample(path)
        if not transcript.strip():
            logger.warning(f"[{i}/{len(SAMPLE_FILES)}] Skipping {path.name} — empty")
            continue

        transcript = sanitize_transcript(transcript)

        for model in MODELS:
            label = f"[{i}/{len(SAMPLE_FILES)}] {path.name} / {model}"
            try:
                results = []

                # Order matters: cache v1 first so v1_comparison can reuse it
                if precache_v1(transcript, call_id, model):
                    counters["workflow_v1"] += 1
                    results.append("V1 cached")

                if precache_v1_comparison(transcript, call_id, model):
                    counters["v1_comparison"] += 1
                    results.append("V1-comparison cached")

                if precache_v3(transcript, call_id, model):
                    counters["workflow"] += 1
                    results.append("V3 cached")

                if precache_benchmark(transcript, call_id, model):
                    counters["benchmark"] += 1
                    results.append("benchmark cached")

                status = ", ".join(results) if results else "all already cached"
                logger.info(f"  ✓ {label}: {status}")

            except Exception as e:
                logger.error(f"  ✗ {label}: {e}")
                errors.append(f"{label}: {e}")

    logger.info(
        f"\n{'='*60}\n"
        f"Done.\n"
        f"  V3 workflow     : {counters['workflow']} newly cached\n"
        f"  V1 workflow     : {counters['workflow_v1']} newly cached\n"
        f"  V1 comparison   : {counters['v1_comparison']} newly cached\n"
        f"  Benchmark       : {counters['benchmark']} newly cached\n"
        f"  Errors          : {len(errors)}"
    )
    if errors:
        logger.warning("Errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
