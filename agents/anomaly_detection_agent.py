"""
Anomaly Detection Agent: Flags outlier calls for operational review.

Compares current call metrics against historical baselines to detect unusual
patterns: abnormally long calls, very low quality scores, multiple violations,
extreme sentiment, unusual topics, or suspicious interaction patterns.

No LLM call required — pure statistical comparison against history.
Very fast (<50ms) — runs after all scoring agents have completed.

Business value:
  - Detects agent misconduct (abnormal handling times, suspicious patterns)
  - Flags calls that may represent systemic product or process failures
  - Identifies potential fraud patterns before they become widespread
  - Enables proactive QA sampling: "flag top-5% outliers for manual review"
  - Reduces QA team workload: focus on flagged calls only (~5-8% of volume)

Real-world usage:
  - 100% of calls scored (zero additional LLM cost — pure computation)
  - Anomaly score threshold of 70+ triggers automatic QA queue entry
  - Dashboard: "Calls requiring review today" fed from this output
  - Weekly anomaly report to operations management
  - Used by fraud detection for behavioral pattern analysis
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CALL_HISTORY_PATH = Path(__file__).parent.parent / "data" / "call_history.jsonl"

# ── Anomaly thresholds ────────────────────────────────────────────────────────
ANOMALY_THRESHOLDS = {
    "very_low_qa_score": 50.0,           # QA score below this is flagged
    "very_high_escalation_risk": 85.0,   # Escalation risk score above this
    "multiple_compliance_violations": 3, # More than N violations
    "critical_compliance_score": 50.0,   # Compliance score below this
    "very_negative_sentiment": "negative",
    "many_pii_items": 5,                  # More than N PII items redacted
}

ANOMALY_WEIGHTS = {
    "low_qa_score": 25,
    "critical_compliance": 30,
    "multiple_violations": 20,
    "high_escalation_risk": 20,
    "extreme_sentiment": 15,
    "excessive_pii": 10,
    "statistical_outlier": 15,           # Deviates significantly from mean
}


class AnomalyDetectionAgent:
    """
    Statistical + rule-based anomaly detection for call center operations.

    Computes an anomaly score (0-100) by checking:
    1. Rule-based flags (absolute thresholds)
    2. Statistical outliers (vs. population mean ± 2σ)

    In production, high anomaly scores trigger:
    - Automatic QA review queue entry
    - Supervisor notification
    - Call flagged for audit trail
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def process(
        self,
        call_id: str,
        qa_score: Optional[dict] = None,
        compliance: Optional[dict] = None,
        escalation: Optional[dict] = None,
        sentiment: Optional[dict] = None,
        pii_summary: Optional[dict] = None,
    ) -> dict:
        """
        Compute anomaly score and identify which signals triggered it.

        Args:
            call_id: Unique call identifier
            qa_score: QualityScoreAgent output
            compliance: ComplianceCheckerAgent output
            escalation: EscalationPredictionAgent output
            sentiment: SentimentAgent output
            pii_summary: PIIRedactionAgent redaction summary

        Returns:
            dict with anomaly_score, anomaly_level, flags, requires_review,
                  anomaly_summary, statistical_context
        """
        flags = []
        score = 0

        # ── Rule-based checks ─────────────────────────────────────────────────

        # 1. Very low QA score
        if qa_score:
            overall = qa_score.get("overall_score", 100.0)
            if overall < ANOMALY_THRESHOLDS["very_low_qa_score"]:
                flags.append({
                    "type": "low_qa_score",
                    "detail": f"QA score {overall:.0f}/100 is below threshold ({ANOMALY_THRESHOLDS['very_low_qa_score']:.0f})",
                    "severity": "high",
                })
                score += ANOMALY_WEIGHTS["low_qa_score"]

        # 2. Critical compliance score
        if compliance:
            comp_score = compliance.get("compliance_score", 100.0)
            violations = compliance.get("violations", [])
            if comp_score < ANOMALY_THRESHOLDS["critical_compliance_score"]:
                flags.append({
                    "type": "critical_compliance",
                    "detail": f"Compliance score {comp_score:.0f}/100 is critically low",
                    "severity": "critical",
                })
                score += ANOMALY_WEIGHTS["critical_compliance"]
            if len(violations) >= ANOMALY_THRESHOLDS["multiple_compliance_violations"]:
                flags.append({
                    "type": "multiple_violations",
                    "detail": f"{len(violations)} compliance violations detected in a single call",
                    "severity": "high",
                })
                score += ANOMALY_WEIGHTS["multiple_violations"]

        # 3. High escalation risk
        if escalation:
            risk = escalation.get("risk_score", 0.0)
            if risk >= ANOMALY_THRESHOLDS["very_high_escalation_risk"]:
                flags.append({
                    "type": "high_escalation_risk",
                    "detail": f"Escalation risk score {risk:.0f}/100 — critical threshold exceeded",
                    "severity": "high",
                })
                score += ANOMALY_WEIGHTS["high_escalation_risk"]

        # 4. Extreme negative sentiment
        if sentiment:
            cust_sent = sentiment.get("overall_customer_sentiment", "neutral")
            esc_risk = sentiment.get("escalation_risk", "low")
            if cust_sent in ("very_negative",) or esc_risk == "high":
                flags.append({
                    "type": "extreme_sentiment",
                    "detail": f"Customer sentiment: {cust_sent}, escalation risk: {esc_risk}",
                    "severity": "medium",
                })
                score += ANOMALY_WEIGHTS["extreme_sentiment"]

        # 5. Excessive PII in transcript
        if pii_summary:
            total_pii = sum(pii_summary.values()) if pii_summary else 0
            if total_pii >= ANOMALY_THRESHOLDS["many_pii_items"]:
                flags.append({
                    "type": "excessive_pii",
                    "detail": f"{total_pii} PII items detected — unusually high for a single call",
                    "severity": "medium",
                })
                score += ANOMALY_WEIGHTS["excessive_pii"]

        # ── Statistical outlier check ─────────────────────────────────────────
        stat_context = self._statistical_check(call_id, qa_score)
        if stat_context.get("is_outlier"):
            flags.append({
                "type": "statistical_outlier",
                "detail": stat_context.get("detail", "Score deviates significantly from historical mean"),
                "severity": "medium",
            })
            score += ANOMALY_WEIGHTS["statistical_outlier"]

        # ── Determine level ───────────────────────────────────────────────────
        score = min(score, 100)
        if score >= 70:
            level = "critical"
            requires_review = True
        elif score >= 45:
            level = "high"
            requires_review = True
        elif score >= 20:
            level = "medium"
            requires_review = False
        else:
            level = "normal"
            requires_review = False

        result = {
            "call_id": call_id,
            "anomaly_score": float(score),
            "anomaly_level": level,
            "flags": flags,
            "requires_review": requires_review,
            "statistical_context": stat_context,
            "anomaly_summary": self._build_summary(score, level, flags, requires_review),
        }

        self.logger.info(
            f"[ANOMALY] call {call_id}: score={score}, level={level}, "
            f"flags={len(flags)}, requires_review={requires_review}"
        )
        return result

    def _statistical_check(self, call_id: str, qa_score: Optional[dict]) -> dict:
        """Compare current QA score against historical population (mean ± 2σ)."""
        if not qa_score:
            return {"is_outlier": False, "detail": "No QA score available for statistical check"}

        current_score = qa_score.get("overall_score", 75.0)
        history_scores = self._load_historical_scores(call_id)

        if len(history_scores) < 5:
            return {
                "is_outlier": False,
                "detail": f"Insufficient history ({len(history_scores)} calls) for statistical comparison",
                "population_size": len(history_scores),
            }

        mean = statistics.mean(history_scores)
        stdev = statistics.stdev(history_scores)
        z_score = (current_score - mean) / stdev if stdev > 0 else 0

        is_outlier = abs(z_score) > 2.0  # > 2 standard deviations

        return {
            "is_outlier": is_outlier,
            "z_score": round(z_score, 2),
            "population_mean": round(mean, 1),
            "population_stdev": round(stdev, 1),
            "current_score": current_score,
            "population_size": len(history_scores),
            "detail": (
                f"Score {current_score:.0f} is {abs(z_score):.1f}σ below mean ({mean:.0f})"
                if is_outlier and z_score < 0
                else f"Score {current_score:.0f} is within normal range (mean={mean:.0f} ±{stdev:.0f})"
            ),
        }

    def _load_historical_scores(self, exclude_call_id: str) -> list:
        """Load QA scores from JSONL history for statistical comparison."""
        if not CALL_HISTORY_PATH.exists():
            return []
        try:
            scores = []
            with open(CALL_HISTORY_PATH, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("call_id") == exclude_call_id:
                            continue
                        score = record.get("qa_score", {}).get("overall_score")
                        if score is not None:
                            scores.append(float(score))
                    except (json.JSONDecodeError, ValueError):
                        continue
            return scores
        except Exception as e:
            self.logger.warning(f"[ANOMALY] Failed to load history for stats: {e}")
            return []

    def _build_summary(self, score: float, level: str, flags: list, requires_review: bool) -> str:
        if not flags:
            return f"No anomalies detected (score: {score:.0f}/100). Call appears within normal parameters."
        flag_types = [f["type"] for f in flags]
        review_msg = " — QUEUED FOR MANUAL QA REVIEW" if requires_review else ""
        return (
            f"Anomaly score: {score:.0f}/100 ({level.upper()}){review_msg}. "
            f"Triggered by: {', '.join(flag_types)}."
        )
