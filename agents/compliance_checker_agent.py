"""
Compliance Checker Agent: Scans call transcripts for regulatory violations.

Checks against: HIPAA, GDPR, PCI-DSS, TCPA, and general financial regulations.
Runs on the PII-redacted transcript — after PIIRedactionAgent, before SummarizationAgent.

Business value:
  - GDPR fines: up to €20M or 4% of annual revenue
  - HIPAA fines: $100–$50,000 per violation
  - PCI-DSS: $5,000–$100,000/month + card scheme penalties
  - Replaces manual compliance auditing — flags 100% of calls, not 2% sample

Real-world usage:
  - Banking, healthcare, insurance: every call
  - Retail, telecom: spot-check mode (~20%)
  - Output fed to compliance dashboard, supervisor alerts, agent coaching
"""

import logging
import time
from typing import Literal

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Compliance categories and their rule sets ──────────────────────────────────

COMPLIANCE_RULES = {
    "HIPAA": [
        "Agent disclosed patient health information to unverified third party",
        "PHI (diagnosis, medication, treatment) discussed without patient consent verification",
        "Medical record details shared without confirming caller identity",
    ],
    "GDPR": [
        "Personal data processed or disclosed without explicit consent confirmation",
        "Agent did not inform customer of data recording/retention policy when asked",
        "Customer's right to erasure or data portability request was ignored or deflected",
        "Data shared with third party without proper disclosure",
    ],
    "PCI-DSS": [
        "Agent requested CVV/security code verbally over the phone",
        "Full card number spoken or repeated aloud during call",
        "Card expiry date combined with card number disclosed verbally",
    ],
    "TCPA": [
        "Call recording started without customer consent notification",
        "Customer not informed this call may be recorded for quality purposes",
        "Marketing pitch made to customer on Do-Not-Call list",
    ],
    "Financial": [
        "Agent made unsuitable product recommendation without assessing customer needs",
        "Misleading or unsubstantiated claim about product returns or guarantees",
        "Agent confirmed transaction without proper identity verification",
        "Agent disclosed account balance or transaction details without ID verification",
        "Agent promised a specific outcome or compensation not within authority to grant",
    ],
    "General": [
        "Agent became abusive, dismissive, or unprofessional toward customer",
        "Agent placed customer on hold for more than 5 minutes without check-in",
        "Agent failed to offer escalation when customer explicitly requested supervisor",
    ],
}


class ComplianceViolation(BaseModel):
    category: str = Field(..., description="Regulation category: HIPAA/GDPR/PCI-DSS/TCPA/Financial/General")
    severity: str = Field(..., description="critical / high / medium / low")
    description: str = Field(..., description="What specific violation occurred")
    transcript_evidence: str = Field(..., description="Short quote or paraphrase from transcript that triggered this flag")
    remediation: str = Field(..., description="Recommended corrective action")


class ComplianceSchema(BaseModel):
    overall_compliance_status: str = Field(..., description="compliant / minor_issues / major_violations / critical")
    violations: list[ComplianceViolation] = Field(default_factory=list, description="List of detected violations")
    compliance_score: float = Field(..., ge=0, le=100, description="Compliance score 0-100 (100=fully compliant)")
    requires_immediate_review: bool = Field(..., description="True if a supervisor must review this call urgently")
    summary: str = Field(..., description="One sentence compliance summary")


class ComplianceCheckerAgent:
    """
    Scans call transcripts for regulatory compliance violations.

    Produces a structured violation report with severity levels and
    remediation recommendations for each detected issue.
    """

    def __init__(self, llm_name: Literal["claude", "gpt4", "gemini"] = "claude"):
        self.llm_name = llm_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.llm = self._initialize_llm(llm_name)

    def _initialize_llm(self, llm_name: str) -> BaseLanguageModel:
        try:
            if llm_name == "claude":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=settings.CLAUDE_MODEL,
                    api_key=settings.ANTHROPIC_API_KEY,
                    temperature=0.1,  # Very low — compliance needs consistency
                )
            elif llm_name == "gpt4":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=settings.GPT4_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.1,
                )
            elif llm_name == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    api_key=settings.GOOGLE_API_KEY,
                    temperature=0.1,
                )
            else:
                raise ValueError(f"Unsupported LLM: {llm_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for compliance: {e}")
            raise

    def process(self, call_id: str, transcript: str) -> dict:
        """
        Scan the transcript for compliance violations.

        Args:
            call_id: Unique call identifier
            transcript: PII-redacted transcript text

        Returns:
            dict with overall_compliance_status, violations, compliance_score,
                  requires_immediate_review, summary
        """
        if not transcript or not transcript.strip():
            return self._empty_result(call_id)

        if settings.MOCK_LLM:
            self.logger.info(f"[MOCK] Returning mock compliance check for {call_id}")
            return self._mock_compliance(call_id, transcript)

        self.logger.info(f"[COMPLIANCE] Checking call {call_id} with {self.llm_name}")
        start = time.time()

        try:
            method = "json_mode" if self.llm_name == "gemini" else None
            structured_llm = (
                self.llm.with_structured_output(ComplianceSchema, method=method)
                if method
                else self.llm.with_structured_output(ComplianceSchema)
            )
            result: ComplianceSchema = structured_llm.invoke(self._build_prompt(transcript))

            elapsed = time.time() - start
            self.logger.info(
                f"[COMPLIANCE] Completed in {elapsed:.2f}s — "
                f"status: {result.overall_compliance_status}, "
                f"violations: {len(result.violations)}, "
                f"score: {result.compliance_score:.1f}/100"
            )

            return {
                "call_id": call_id,
                "overall_compliance_status": result.overall_compliance_status,
                "violations": [v.model_dump() for v in result.violations],
                "compliance_score": result.compliance_score,
                "requires_immediate_review": result.requires_immediate_review,
                "summary": result.summary,
            }

        except Exception as e:
            self.logger.error(f"[COMPLIANCE] Failed for {call_id}: {e}")
            return self._empty_result(call_id)

    def _build_prompt(self, transcript: str) -> str:
        rules_text = "\n".join(
            f"\n{cat}:\n" + "\n".join(f"  - {r}" for r in rules)
            for cat, rules in COMPLIANCE_RULES.items()
        )

        return f"""You are a compliance auditor reviewing a call center transcript for regulatory violations.

COMPLIANCE RULES TO CHECK:
{rules_text}

TRANSCRIPT:
{transcript}

Carefully review the transcript against each rule above.

For each violation found, provide:
- category: which regulation was violated (HIPAA/GDPR/PCI-DSS/TCPA/Financial/General)
- severity: critical (immediate supervisor action needed) / high (compliance team review) / medium (coaching required) / low (minor deviation)
- description: what specifically went wrong
- transcript_evidence: the exact quote or paraphrase that triggered this flag
- remediation: specific corrective action the agent or business should take

compliance_score: 100 if no violations, reduce by 5 for each low, 15 for medium, 25 for high, 40 for critical.
requires_immediate_review: true only if any critical or multiple high violations.
overall_compliance_status: compliant (score>=95) / minor_issues (80-94) / major_violations (50-79) / critical (<50).

If no violations are found, return an empty violations list with score=100 and status=compliant."""

    def _mock_compliance(self, call_id: str, transcript: str) -> dict:
        """Pattern-based mock compliance check for testing without API calls."""
        t = transcript.lower()
        violations = []

        # PCI-DSS: CVV requested
        if any(w in t for w in ["cvv", "security code", "3 digits", "card number"]):
            violations.append({
                "category": "PCI-DSS",
                "severity": "critical",
                "description": "Agent may have requested or repeated card security details verbally",
                "transcript_evidence": "Detected payment card security data discussion",
                "remediation": "Redirect customer to secure IVR or web portal for card input. Never repeat card numbers verbally.",
            })

        # GDPR: no consent mentioned but data processed
        if any(w in t for w in ["record", "recording"]) and "consent" not in t and "permission" not in t:
            violations.append({
                "category": "TCPA",
                "severity": "medium",
                "description": "Call may have been recorded without explicit customer consent notification",
                "transcript_evidence": "Recording mentioned but no consent confirmation detected",
                "remediation": "Always inform customer at call start: 'This call may be recorded for quality and training purposes.'",
            })

        # Financial: balance/account disclosed
        if any(w in t for w in ["balance", "account number", "transaction"]):
            if not any(w in t for w in ["verify", "verification", "date of birth", "password", "pin", "security question"]):
                violations.append({
                    "category": "Financial",
                    "severity": "high",
                    "description": "Agent may have disclosed account information without completing identity verification",
                    "transcript_evidence": "Account/balance discussion detected without prior identity verification steps",
                    "remediation": "Always complete full ID verification (name, DOB, account pin) before disclosing any account details.",
                })

        # General: escalation refused
        if any(w in t for w in ["speak to manager", "want a supervisor", "escalate"]):
            if not any(w in t for w in ["transfer", "connect you", "supervisor will", "arrange"]):
                violations.append({
                    "category": "General",
                    "severity": "high",
                    "description": "Customer requested supervisor/escalation but agent may not have facilitated it",
                    "transcript_evidence": "Escalation request detected without confirmed transfer action",
                    "remediation": "When a customer requests a supervisor, always action the transfer or schedule a callback within the same interaction.",
                })

        # Calculate score
        severity_deductions = {"critical": 40, "high": 25, "medium": 15, "low": 5}
        score = max(0.0, 100.0 - sum(severity_deductions.get(v["severity"], 0) for v in violations))

        if not violations:
            status = "compliant"
        elif score >= 80:
            status = "minor_issues"
        elif score >= 50:
            status = "major_violations"
        else:
            status = "critical"

        requires_review = any(v["severity"] in ("critical", "high") for v in violations)

        summary = (
            f"[MOCK — {self.llm_name}] "
            + (f"{len(violations)} violation(s) detected. Score: {score:.0f}/100."
               if violations else "No compliance violations detected. Score: 100/100.")
        )

        return {
            "call_id": call_id,
            "overall_compliance_status": status,
            "violations": violations,
            "compliance_score": score,
            "requires_immediate_review": requires_review,
            "summary": summary,
        }

    def _empty_result(self, call_id: str) -> dict:
        return {
            "call_id": call_id,
            "overall_compliance_status": "unknown",
            "violations": [],
            "compliance_score": 100.0,
            "requires_immediate_review": False,
            "summary": "Compliance check skipped — no transcript available",
        }
