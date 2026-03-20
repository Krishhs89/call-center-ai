"""
Pre-cache all sample transcripts for all 3 models.
Runs workflow + full benchmark for each sample × model and saves to disk.
No API calls are made when MOCK_LLM=true in .env.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precache")

from agents.intake_agent import IntakeAgent
from agents.summarization_agent import SummarizationAgent
from agents.quality_score_agent import QualityScoreAgent
from workflow.langgraph_flow import create_workflow, run_workflow
from utils.cache import get_cached, save_cache
from utils.validation import sanitize_transcript

SAMPLE_DIR = ROOT / "data" / "sample_transcripts"
MODELS = ["claude", "gpt4", "gemini"]

SAMPLE_FILES = sorted(SAMPLE_DIR.glob("*.json"))


def load_transcript(path: Path) -> tuple[str, str]:
    """Return (call_id, transcript_text)."""
    with open(path) as f:
        data = json.load(f)
    return data.get("call_id", path.stem.upper()), data.get("transcript", "")


def precache_workflow(transcript: str, call_id: str, model: str) -> bool:
    """Run workflow and cache result. Returns True if newly cached."""
    transcript = sanitize_transcript(transcript)
    if get_cached(transcript, model, "workflow"):
        return False  # already cached

    intake = IntakeAgent()
    call_input = intake.process(
        transcript_text=transcript,
        metadata={"call_id": call_id},
    )
    workflow = create_workflow(llm_name=model)
    result = run_workflow(workflow, call_input, llm_name=model)

    data = result.model_dump()
    data["_cache_type"] = "workflow"
    data["_llm_name"] = model
    save_cache(transcript, model, "workflow", data)
    return True


def precache_benchmark(transcript: str, call_id: str, model: str) -> bool:
    """Run full benchmark for one model and cache. Returns True if newly cached."""
    cache_key = f"benchmark_full_benchmark_{model}"
    if get_cached(transcript, model, cache_key):
        return False  # already cached

    start = time.time()
    summarizer = SummarizationAgent(llm_name=model)
    scorer = QualityScoreAgent(llm_name=model)

    summary = summarizer.process(call_id=call_id, transcript=transcript)
    qa = scorer.process(call_id=call_id, transcript=transcript)
    elapsed = round(time.time() - start, 2)

    save_cache(transcript, model, cache_key, {
        "summary": summary.model_dump(),
        "qa": qa.model_dump(),
        "timing": elapsed,
        "token_counts": {},
        "call_id": call_id,
        "_cache_type": cache_key,
        "_llm_name": model,
    })
    return True


def main():
    logger.info(f"Found {len(SAMPLE_FILES)} sample files")
    total_workflow = 0
    total_benchmark = 0
    errors = []

    for sample_path in SAMPLE_FILES:
        call_id, transcript = load_transcript(sample_path)
        if not transcript.strip():
            logger.warning(f"Skipping {sample_path.name} — empty transcript")
            continue

        for model in MODELS:
            label = f"{sample_path.name} / {model}"
            try:
                wf_new = precache_workflow(transcript, call_id, model)
                bm_new = precache_benchmark(transcript, call_id, model)
                status = []
                if wf_new:
                    status.append("workflow cached")
                    total_workflow += 1
                if bm_new:
                    status.append("benchmark cached")
                    total_benchmark += 1
                if not status:
                    status = ["already cached"]
                logger.info(f"  ✓ {label}: {', '.join(status)}")
            except Exception as e:
                logger.error(f"  ✗ {label}: {e}")
                errors.append(f"{label}: {e}")

    logger.info(
        f"\nDone. Newly cached: {total_workflow} workflow + {total_benchmark} benchmark entries."
    )
    if errors:
        logger.warning(f"{len(errors)} errors:\n" + "\n".join(errors))
    else:
        logger.info("No errors.")


if __name__ == "__main__":
    main()
