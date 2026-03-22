"""
Auto Tagging Agent: Multi-label classification for intelligent call routing and analytics.

Assigns structured tags to every call across 5 taxonomies:
  primary_category, sub_category, intent_tags, product_tags, routing_tags

Business value:
  - Replaces manual call coding (analysts tagging 2% of calls) with 100% automated coverage
  - Feeds omni-channel routing: tagged calls go to the right queue/team automatically
  - Powers call analytics dashboards (Tableau, PowerBI) for trend detection
  - Enables automated SLA tracking per category
  - Replicates Verint, NICE, and AWS Contact Center Intelligence capabilities

Real-world usage:
  - 100% of calls tagged in real-time
  - Primary category drives queue routing (billing → billing team, tech → tier-1 tech)
  - Tags feed CRM opportunity detection (e.g. "upgrade" tag → sales team notified)
  - Analytics: track top-N issues week-over-week, seasonal trends, product feedback
  - Used by workforce management for staffing forecasting by category volume
"""

import logging
import time
from typing import Literal, Optional

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Tag taxonomy ──────────────────────────────────────────────────────────────

PRIMARY_CATEGORIES = [
    "billing", "technical_support", "account_management",
    "product_inquiry", "complaint", "cancellation", "refund",
    "shipping_delivery", "general_inquiry", "fraud_security",
]

INTENT_TAGS = [
    "wants_refund", "wants_escalation", "wants_information",
    "wants_cancellation", "wants_upgrade", "reporting_issue",
    "making_complaint", "requesting_callback", "comparing_options",
    "fraudulent_activity_suspected",
]

SENTIMENT_TAGS = ["urgent", "frustrated", "satisfied", "confused", "threatening_legal_action"]

ROUTING_TAGS = [
    "route_to_billing", "route_to_tech_tier1", "route_to_tech_tier2",
    "route_to_retention", "route_to_fraud", "route_to_supervisor",
    "no_routing_needed",
]


class AutoTaggingSchema(BaseModel):
    primary_category: str = Field(..., description=f"One of: {', '.join(PRIMARY_CATEGORIES)}")
    sub_category: str = Field(..., description="More specific sub-topic within primary_category")
    intent_tags: list[str] = Field(default_factory=list, description="Customer's intent(s) for this call")
    sentiment_tags: list[str] = Field(default_factory=list, description="Emotional/urgency tags")
    routing_tags: list[str] = Field(default_factory=list, description="Where this call should be routed")
    product_tags: list[str] = Field(default_factory=list, description="Products/services mentioned (e.g. 'mobile_plan', 'savings_account')")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence in primary_category (0-1)")
    tagging_rationale: str = Field(..., description="One sentence explaining why these tags were assigned")


class AutoTaggingAgent:
    """
    Multi-label call classification for routing, analytics, and CRM enrichment.

    Tags are assigned post-transcription and post-PII-redaction.
    In production, tags are written back to the CRM record and used to:
    - Route the call to the correct queue (real-time)
    - Tag the CRM record for reporting
    - Trigger automated workflows (e.g., sales follow-up on 'wants_upgrade')
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
                    temperature=0.1,
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
            self.logger.error(f"Failed to initialize LLM for auto-tagging: {e}")
            raise

    def process(
        self,
        call_id: str,
        transcript: str,
        summary: Optional[dict] = None,
    ) -> dict:
        """
        Assign multi-label tags to a call.

        Args:
            call_id: Unique call identifier
            transcript: PII-redacted transcript
            summary: Output from SummarizationAgent (enriches tagging accuracy)

        Returns:
            dict with primary_category, sub_category, intent_tags,
                  sentiment_tags, routing_tags, product_tags, confidence_score
        """
        if not transcript or not transcript.strip():
            return self._empty_result(call_id)

        if settings.MOCK_LLM:
            self.logger.info(f"[MOCK] Returning mock auto-tags for {call_id}")
            return self._mock_tagging(call_id, transcript, summary)

        self.logger.info(f"[TAGGING] Tagging call {call_id} with {self.llm_name}")
        start = time.time()

        try:
            method = "json_mode" if self.llm_name == "gemini" else None
            structured_llm = (
                self.llm.with_structured_output(AutoTaggingSchema, method=method)
                if method
                else self.llm.with_structured_output(AutoTaggingSchema)
            )
            result: AutoTaggingSchema = structured_llm.invoke(
                self._build_prompt(transcript, summary)
            )

            self.logger.info(
                f"[TAGGING] Completed in {time.time() - start:.2f}s — "
                f"category={result.primary_category} (conf={result.confidence_score:.2f})"
            )
            return {
                "call_id": call_id,
                "primary_category": result.primary_category,
                "sub_category": result.sub_category,
                "intent_tags": result.intent_tags,
                "sentiment_tags": result.sentiment_tags,
                "routing_tags": result.routing_tags,
                "product_tags": result.product_tags,
                "confidence_score": result.confidence_score,
                "tagging_rationale": result.tagging_rationale,
            }

        except Exception as e:
            self.logger.error(f"[TAGGING] Failed for {call_id}: {e}")
            return self._empty_result(call_id)

    def _build_prompt(self, transcript: str, summary: Optional[dict]) -> str:
        summary_block = ""
        if summary:
            summary_block = (
                f"\nCALL SUMMARY:\n"
                f"  Issue: {summary.get('main_issue', 'N/A')}\n"
                f"  Resolution: {summary.get('resolution_status', 'N/A')}\n"
                f"  Action items: {', '.join(summary.get('action_items', []))}"
            )

        categories_str = ", ".join(PRIMARY_CATEGORIES)
        intents_str = ", ".join(INTENT_TAGS)
        routing_str = ", ".join(ROUTING_TAGS)

        return f"""You are a call classification system assigning structured tags to a call center interaction.
{summary_block}

TRANSCRIPT:
{transcript}

Assign tags from these taxonomies:

primary_category (choose ONE): {categories_str}

sub_category: more specific topic within primary_category (e.g., "credit_card_charge" within "billing")

intent_tags (choose ALL that apply): {intents_str}

sentiment_tags (choose ALL that apply): urgent, frustrated, satisfied, confused, threatening_legal_action

routing_tags (choose ALL that apply): {routing_str}

product_tags: list all products/services mentioned (use snake_case, e.g. "mobile_plan", "savings_account")

confidence_score: 0.0-1.0 how confident you are in primary_category

tagging_rationale: one sentence explaining the most important tagging decision"""

    def _mock_tagging(self, call_id: str, transcript: str, summary: Optional[dict]) -> dict:
        t = transcript.lower()

        # Determine primary category
        if any(w in t for w in ["bill", "charge", "invoice", "payment", "overcharged", "fee"]):
            primary = "billing"
            sub = "incorrect_charge"
            routing = ["route_to_billing"]
        elif any(w in t for w in ["refund", "money back", "reimburse"]):
            primary = "refund"
            sub = "refund_request"
            routing = ["route_to_billing"]
        elif any(w in t for w in ["not working", "broken", "error", "bug", "crash", "technical", "outage"]):
            primary = "technical_support"
            sub = "product_not_working"
            routing = ["route_to_tech_tier1"]
        elif any(w in t for w in ["cancel", "cancellation", "close account", "terminate"]):
            primary = "cancellation"
            sub = "account_cancellation"
            routing = ["route_to_retention"]
        elif any(w in t for w in ["fraud", "unauthorized", "suspicious", "stolen"]):
            primary = "fraud_security"
            sub = "unauthorized_transaction"
            routing = ["route_to_fraud"]
        elif any(w in t for w in ["ship", "delivery", "package", "order", "tracking"]):
            primary = "shipping_delivery"
            sub = "delivery_delay"
            routing = ["route_to_billing"]
        else:
            primary = "general_inquiry"
            sub = "general_question"
            routing = ["no_routing_needed"]

        # Intent tags
        intents = []
        if any(w in t for w in ["refund", "money back"]):
            intents.append("wants_refund")
        if any(w in t for w in ["supervisor", "manager", "escalate"]):
            intents.append("wants_escalation")
            routing.append("route_to_supervisor")
        if any(w in t for w in ["cancel"]):
            intents.append("wants_cancellation")
        if any(w in t for w in ["how", "what", "when", "why", "where"]):
            intents.append("wants_information")
        if any(w in t for w in ["not working", "broken", "issue", "problem"]):
            intents.append("reporting_issue")
        if not intents:
            intents = ["wants_information"]

        # Sentiment tags
        sentiment_tags = []
        if any(w in t for w in ["urgent", "asap", "immediately", "emergency"]):
            sentiment_tags.append("urgent")
        if any(w in t for w in ["frustrated", "angry", "furious", "unacceptable", "terrible"]):
            sentiment_tags.append("frustrated")
        if any(w in t for w in ["lawsuit", "legal", "sue", "lawyer"]):
            sentiment_tags.append("threatening_legal_action")
        if any(w in t for w in ["thank", "great", "perfect", "appreciate", "happy"]):
            sentiment_tags.append("satisfied")

        # Product tags
        product_tags = []
        if any(w in t for w in ["account", "savings", "current"]):
            product_tags.append("bank_account")
        if any(w in t for w in ["card", "credit", "debit"]):
            product_tags.append("payment_card")
        if any(w in t for w in ["app", "mobile", "website", "portal"]):
            product_tags.append("digital_channel")
        if any(w in t for w in ["plan", "subscription", "service"]):
            product_tags.append("subscription_service")

        confidence = 0.85 if primary != "general_inquiry" else 0.65

        return {
            "call_id": call_id,
            "primary_category": primary,
            "sub_category": sub,
            "intent_tags": intents,
            "sentiment_tags": sentiment_tags,
            "routing_tags": list(set(routing)),
            "product_tags": product_tags if product_tags else ["unspecified"],
            "confidence_score": confidence,
            "tagging_rationale": f"[MOCK — {self.llm_name}] Primary category '{primary}' assigned based on dominant keyword signals in transcript.",
        }

    def _empty_result(self, call_id: str) -> dict:
        return {
            "call_id": call_id,
            "primary_category": "general_inquiry",
            "sub_category": "unknown",
            "intent_tags": [],
            "sentiment_tags": [],
            "routing_tags": ["no_routing_needed"],
            "product_tags": [],
            "confidence_score": 0.0,
            "tagging_rationale": "Auto-tagging skipped — no transcript available",
        }
