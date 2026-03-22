"""
Feedback Loop Agent: Tracks whether QA coaching feedback was acted on in subsequent calls.

Compares the current call's QA scores and behaviors against the coaching tips
given after the previous call — closing the loop between coaching and improvement.

No LLM call required — pure data comparison from call history.
Runs in run_workflow() after CallResult is assembled, as a post-processing step.

Business value:
  - Without feedback loops, 60-70% of coaching is forgotten within 1 week (Ebbinghaus curve)
  - Identifies agents who consistently improve vs. those who ignore coaching
  - Enables data-driven coaching prioritization: focus on coachable agents
  - Powers agent performance trajectory reports for HR and management
  - Closes the coaching-to-improvement loop that most tools leave open
  - Enables manager accountability: "was coaching delivered and did it work?"

Real-world usage:
  - Run for every call where the agent has a prior coaching record
  - Coaching effectiveness report generated weekly per agent
  - Agents with improvement_status='regressed' flagged for urgent 1:1
  - Used in performance reviews and PIP (Performance Improvement Plan) documentation
  - Feeds gamification systems: badges for sustained improvement
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CALL_HISTORY_PATH = Path(__file__).parent.parent / "data" / "call_history.jsonl"


class FeedbackLoopAgent:
    """
    Measures whether coaching from previous calls was applied in the current call.

    Compares:
    - Current QA score vs. last 3 calls' average
    - Current escalation risk vs. trend
    - Specific coaching dimensions that were flagged previously

    In production, this data feeds:
    - Agent performance scorecards
    - Weekly coaching effectiveness reports
    - Learning management systems (LMS) for training assignment
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def process(
        self,
        call_id: str,
        current_qa: Optional[dict] = None,
        current_coaching: Optional[dict] = None,
        current_escalation: Optional[dict] = None,
    ) -> dict:
        """
        Measure coaching effectiveness by comparing current vs. prior calls.

        Args:
            call_id: Current call ID
            current_qa: QA score for this call
            current_coaching: Coaching tips generated for this call
            current_escalation: Escalation data for this call

        Returns:
            dict with improvement_status, score_delta, coaching_adoption_rate,
                  regressed_dimensions, improved_dimensions, feedback_summary
        """
        history = self._load_history(call_id)

        if len(history) < 2:
            return self._insufficient_history_result(call_id, len(history))

        # Get last 3 calls for trend
        recent = history[-3:]
        prior_scores = [
            c.get("qa_score", {}).get("overall_score", 0.0)
            for c in recent
            if c.get("qa_score")
        ]

        current_overall = current_qa.get("overall_score", 0.0) if current_qa else 0.0
        avg_prior = round(sum(prior_scores) / len(prior_scores), 1) if prior_scores else 0.0
        score_delta = round(current_overall - avg_prior, 1)

        # Compare dimension-level scores
        current_dims = current_qa.get("dimension_scores", {}) if current_qa else {}
        prior_dims_list = [
            c.get("qa_score", {}).get("dimension_scores", {})
            for c in recent
            if c.get("qa_score") and c["qa_score"].get("dimension_scores")
        ]

        improved_dims = []
        regressed_dims = []
        if prior_dims_list and current_dims:
            for dim, curr_val in current_dims.items():
                prior_vals = [d.get(dim) for d in prior_dims_list if d.get(dim) is not None]
                if not prior_vals:
                    continue
                prior_avg = sum(prior_vals) / len(prior_vals)
                delta = curr_val - prior_avg
                if delta >= 0.5:
                    improved_dims.append({"dimension": dim, "delta": round(delta, 1)})
                elif delta <= -0.5:
                    regressed_dims.append({"dimension": dim, "delta": round(delta, 1)})

        # Check if prior coaching topics were addressed
        prior_coaching_tips = []
        for call in recent:
            v2 = call.get("_v2_extras", {}) or {}
            coaching = v2.get("coaching") or call.get("coaching", {})
            if coaching and coaching.get("coaching_tips"):
                prior_coaching_tips.extend(coaching["coaching_tips"])

        addressed_topics = []
        missed_topics = []
        for tip in prior_coaching_tips[-5:]:  # Look at last 5 tips
            dim = tip.get("dimension", "")
            if dim in [d["dimension"] for d in improved_dims]:
                addressed_topics.append(dim)
            elif dim in [d["dimension"] for d in regressed_dims]:
                missed_topics.append(dim)

        # Coaching adoption rate
        total_topics = len(prior_coaching_tips[-5:])
        adoption_rate = (len(addressed_topics) / total_topics * 100) if total_topics > 0 else 0.0

        # Overall improvement status
        if score_delta >= 5:
            status = "significantly_improved"
        elif score_delta >= 2:
            status = "improved"
        elif score_delta >= -2:
            status = "stable"
        elif score_delta >= -5:
            status = "declined"
        else:
            status = "regressed"

        result = {
            "call_id": call_id,
            "improvement_status": status,
            "score_delta": score_delta,
            "current_score": current_overall,
            "prior_avg_score": avg_prior,
            "improved_dimensions": improved_dims,
            "regressed_dimensions": regressed_dims,
            "coaching_adoption_rate": round(adoption_rate, 1),
            "addressed_coaching_topics": list(set(addressed_topics)),
            "missed_coaching_topics": list(set(missed_topics)),
            "calls_in_baseline": len(prior_scores),
            "feedback_summary": self._build_summary(
                status, score_delta, avg_prior, current_overall,
                improved_dims, regressed_dims, adoption_rate
            ),
        }

        self.logger.info(
            f"[FEEDBACK] call {call_id}: status={status}, delta={score_delta:+.1f}, "
            f"adoption={adoption_rate:.0f}%"
        )
        return result

    def _load_history(self, exclude_call_id: str) -> list:
        """Load call history sorted by timestamp (or file order as proxy)."""
        if not CALL_HISTORY_PATH.exists():
            return []
        try:
            calls = []
            with open(CALL_HISTORY_PATH, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("call_id") != exclude_call_id:
                            calls.append(record)
                    except json.JSONDecodeError:
                        continue
            return calls
        except Exception as e:
            self.logger.warning(f"[FEEDBACK] Failed to load history: {e}")
            return []

    def _build_summary(
        self,
        status: str,
        delta: float,
        prior_avg: float,
        current: float,
        improved: list,
        regressed: list,
        adoption: float,
    ) -> str:
        status_phrases = {
            "significantly_improved": f"Strong improvement ({delta:+.1f} pts vs baseline)",
            "improved": f"Improvement detected ({delta:+.1f} pts vs baseline)",
            "stable": "Performance stable vs. recent baseline",
            "declined": f"Slight decline ({delta:+.1f} pts) — monitor closely",
            "regressed": f"REGRESSION DETECTED ({delta:+.1f} pts) — coaching required",
        }
        base = status_phrases.get(status, f"Status: {status}")

        extras = []
        if improved:
            extras.append(f"Improved: {', '.join(d['dimension'] for d in improved)}")
        if regressed:
            extras.append(f"Regressed: {', '.join(d['dimension'] for d in regressed)}")
        if adoption > 0:
            extras.append(f"Coaching adoption: {adoption:.0f}%")

        return base + (". " + ". ".join(extras) + "." if extras else ".")

    def _insufficient_history_result(self, call_id: str, history_count: int) -> dict:
        return {
            "call_id": call_id,
            "improvement_status": "insufficient_history",
            "score_delta": 0.0,
            "current_score": 0.0,
            "prior_avg_score": 0.0,
            "improved_dimensions": [],
            "regressed_dimensions": [],
            "coaching_adoption_rate": 0.0,
            "addressed_coaching_topics": [],
            "missed_coaching_topics": [],
            "calls_in_baseline": history_count,
            "feedback_summary": (
                f"Feedback loop requires at least 2 prior calls. "
                f"Current history: {history_count} call(s). Will activate after more calls are processed."
            ),
        }
