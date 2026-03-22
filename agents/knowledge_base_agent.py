"""
Knowledge Base Agent: RAG retrieval against internal SOPs, product FAQs, and policy documents.

Distinct from the call-history RAG (rag_retrieval_agent.py) which searches past calls.
This agent searches a SEPARATE ChromaDB collection: structured knowledge documents
(SOPs, product sheets, FAQ pages, compliance guides).

Business value:
  - Agents spend 25-40% of handle time searching for answers — this eliminates that
  - Reduces incorrect information given to customers (costly errors, NPS impact)
  - New agent ramp time reduced by ~40% when real-time KB is available
  - Enables "Agent Assist" use case: surfaced during live call, not just post-call
  - Replicates Salesforce Knowledge, Zendesk Guide, ServiceNow KB capabilities

Real-world usage:
  - Triggered for 100% of calls that have a clear category/topic
  - Results surfaced in real-time agent assist overlay during live call
  - Post-call: verifies whether agent followed the correct SOP
  - Used by QA teams to flag "agent gave wrong answer not in KB"
"""

import logging
import time
from typing import Literal, Optional

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Default knowledge base articles (used when no ChromaDB KB is seeded) ──────

DEFAULT_KB_ARTICLES = [
    {
        "id": "kb_001",
        "title": "Refund Policy — Standard Purchases",
        "content": "Customers may request a full refund within 30 days of purchase for unused products. After 30 days, store credit only. Digital products are non-refundable unless faulty. Agent must verify purchase date before issuing refund.",
        "category": "refund",
        "tags": ["refund", "return", "money back", "purchase"],
    },
    {
        "id": "kb_002",
        "title": "Account Verification Procedure",
        "content": "Always verify identity before disclosing account details: (1) Full name, (2) Date of birth, (3) Last 4 digits of account number OR security question. Do NOT proceed without completing all 3 steps. Document verification in the CRM.",
        "category": "security",
        "tags": ["verification", "identity", "security", "account"],
    },
    {
        "id": "kb_003",
        "title": "Escalation to Supervisor — Process",
        "content": "When customer requests a supervisor: (1) Acknowledge request immediately, (2) Attempt one resolution offer first, (3) If customer still requests, initiate transfer within 2 minutes. Never refuse escalation. Log reason for escalation in CRM.",
        "category": "escalation",
        "tags": ["supervisor", "manager", "escalate", "transfer"],
    },
    {
        "id": "kb_004",
        "title": "Billing Dispute Resolution",
        "content": "For billing disputes: (1) Pull up account and transaction history, (2) Verify identity per KB-002, (3) If charge is < $50 and clearly erroneous, credit immediately, (4) If > $50 or unclear, raise dispute ticket (SLA: 5 business days), (5) Inform customer of SLA and provide ticket number.",
        "category": "billing",
        "tags": ["billing", "charge", "dispute", "overcharge", "credit"],
    },
    {
        "id": "kb_005",
        "title": "Technical Support — First Response Protocol",
        "content": "For technical issues: (1) Confirm device and software version, (2) Ask customer to restart device, (3) Check service status page for known outages, (4) If issue persists, collect error message and device logs, (5) Escalate to Tier 2 if unresolved after 10 minutes.",
        "category": "technical",
        "tags": ["technical", "not working", "error", "bug", "broken", "outage"],
    },
    {
        "id": "kb_006",
        "title": "TCPA Compliance — Recording Consent",
        "content": "Mandatory script at call start: 'This call may be recorded or monitored for quality and training purposes. By continuing, you consent to this recording.' This must be stated BEFORE discussing any account information. Failure is a TCPA violation.",
        "category": "compliance",
        "tags": ["recording", "consent", "TCPA", "compliance", "legal"],
    },
    {
        "id": "kb_007",
        "title": "Shipping Delays — Customer Communication",
        "content": "If order is delayed: (1) Apologise proactively, (2) Provide updated estimated delivery date, (3) If delay > 7 days beyond original date, offer 10% discount on next order as goodwill gesture, (4) Offer expedited shipping upgrade if available at no extra cost.",
        "category": "shipping",
        "tags": ["shipping", "delivery", "delay", "order", "package"],
    },
]


class KBArticle(BaseModel):
    article_id: str = Field(..., description="KB article identifier")
    title: str = Field(..., description="Article title")
    relevance_score: float = Field(..., description="Relevance to this call (0-1)")
    key_points: list[str] = Field(default_factory=list, description="3-5 bullet points from the article most relevant to this call")
    was_agent_compliant: str = Field(..., description="yes / no / partial / not_applicable — did agent follow this SOP?")
    agent_deviation: str = Field(default="", description="If not compliant, what did the agent do wrong?")


class KnowledgeBaseSchema(BaseModel):
    relevant_articles: list[KBArticle] = Field(default_factory=list, description="Most relevant KB articles for this call")
    sop_compliance_score: float = Field(..., ge=0, le=100, description="% of applicable SOPs the agent followed correctly")
    missed_knowledge_opportunities: list[str] = Field(default_factory=list, description="Moments where agent could have used KB but didn't")
    recommended_training_articles: list[str] = Field(default_factory=list, description="Article IDs agent should study based on deviations")
    kb_summary: str = Field(..., description="One sentence KB lookup summary")


class KnowledgeBaseAgent:
    """
    Retrieves relevant SOPs and product knowledge, then verifies agent compliance.

    Two modes:
    1. Retrieval mode (real-time): surfaces relevant KB articles during/before LLM
    2. Compliance mode (post-call): checks whether agent followed retrieved SOPs

    In production, integrates with Confluence, Zendesk Guide, SharePoint,
    or any REST-accessible knowledge management system.
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
            self.logger.error(f"Failed to initialize LLM for knowledge base: {e}")
            raise

    def retrieve_context(self, transcript: str, call_id: Optional[str] = None) -> str:
        """
        Lightweight KB lookup — returns a formatted context string for injection
        into downstream LLM prompts (used before summarization/quality_score).

        Uses keyword matching against DEFAULT_KB_ARTICLES in mock/offline mode.
        In production, would query ChromaDB KB collection or external KB API.
        """
        matched = self._keyword_match(transcript)
        if not matched:
            return ""

        lines = ["RELEVANT KNOWLEDGE BASE ARTICLES:"]
        for article in matched[:3]:
            lines.append(f"\n[{article['id']}] {article['title']}")
            lines.append(article["content"])

        context = "\n".join(lines)
        self.logger.info(f"[KB] Retrieved {len(matched)} article(s) for call {call_id or 'unknown'}")
        return context

    def process(
        self,
        call_id: str,
        transcript: str,
        category: Optional[str] = None,
    ) -> dict:
        """
        Full KB analysis: retrieval + SOP compliance check.

        Args:
            call_id: Unique call identifier
            transcript: PII-redacted transcript
            category: Call category from AutoTaggingAgent or metadata (optional)

        Returns:
            dict with relevant_articles, sop_compliance_score,
                  missed_knowledge_opportunities, recommended_training_articles
        """
        if not transcript or not transcript.strip():
            return self._empty_result(call_id)

        matched_articles = self._keyword_match(transcript, category)

        if settings.MOCK_LLM:
            self.logger.info(f"[MOCK] Returning mock KB analysis for {call_id}")
            return self._mock_kb(call_id, transcript, matched_articles)

        self.logger.info(f"[KB] Analysing call {call_id} with {self.llm_name}")
        start = time.time()

        try:
            method = "json_mode" if self.llm_name == "gemini" else None
            structured_llm = (
                self.llm.with_structured_output(KnowledgeBaseSchema, method=method)
                if method
                else self.llm.with_structured_output(KnowledgeBaseSchema)
            )
            result: KnowledgeBaseSchema = structured_llm.invoke(
                self._build_prompt(transcript, matched_articles)
            )

            self.logger.info(
                f"[KB] Completed in {time.time() - start:.2f}s — "
                f"SOP compliance: {result.sop_compliance_score:.0f}%"
            )
            return {
                "call_id": call_id,
                "relevant_articles": [a.model_dump() for a in result.relevant_articles],
                "sop_compliance_score": result.sop_compliance_score,
                "missed_knowledge_opportunities": result.missed_knowledge_opportunities,
                "recommended_training_articles": result.recommended_training_articles,
                "kb_summary": result.kb_summary,
            }

        except Exception as e:
            self.logger.error(f"[KB] Failed for {call_id}: {e}")
            return self._empty_result(call_id)

    def _keyword_match(self, transcript: str, category: Optional[str] = None) -> list:
        """Fast keyword matching against DEFAULT_KB_ARTICLES."""
        t = transcript.lower()
        matched = []
        for article in DEFAULT_KB_ARTICLES:
            score = sum(1 for tag in article["tags"] if tag in t)
            if category and article["category"] == category.lower():
                score += 3
            if score > 0:
                matched.append({**article, "_score": score})
        matched.sort(key=lambda x: x["_score"], reverse=True)
        return matched[:3]

    def _build_prompt(self, transcript: str, articles: list) -> str:
        articles_text = "\n\n".join(
            f"[{a['id']}] {a['title']}\n{a['content']}"
            for a in articles
        ) if articles else "No specific KB articles matched — use general call center best practices."

        return f"""You are a knowledge management auditor reviewing a call center interaction.

RELEVANT KB ARTICLES:
{articles_text}

TRANSCRIPT:
{transcript}

For each relevant KB article, determine:
1. Was the agent compliant with the SOP? (yes / no / partial / not_applicable)
2. If not compliant, what specific deviation occurred?
3. What were the 3-5 most relevant points from the article for this call?

Also identify:
- Moments where the agent needed KB information but seemed unsure or gave incorrect info
- Which articles the agent should review before their next shift

sop_compliance_score: % of applicable SOPs followed (100 = perfect adherence)
kb_summary: one sentence about KB usage on this call"""

    def _mock_kb(self, call_id: str, transcript: str, matched_articles: list) -> dict:
        t = transcript.lower()

        relevant_articles = []
        missed_opportunities = []
        training_articles = []
        total_score = 100.0

        for article in matched_articles:
            # Check compliance heuristically
            compliant = "yes"
            deviation = ""

            if article["id"] == "kb_002":  # Verification
                has_verify = any(w in t for w in ["verify", "date of birth", "last 4", "security question", "password"])
                has_account_info = any(w in t for w in ["balance", "account", "transaction"])
                if has_account_info and not has_verify:
                    compliant = "no"
                    deviation = "Agent disclosed account information without completing identity verification (KB-002 requires name, DOB, and last 4 digits before any account disclosure)."
                    total_score -= 20
                    training_articles.append(article["id"])

            elif article["id"] == "kb_006":  # Recording consent
                has_consent = any(w in t for w in ["recorded", "recording", "monitored", "consent"])
                if not has_consent:
                    compliant = "partial"
                    deviation = "No recording consent statement detected at call start (TCPA requires this before account discussion)."
                    total_score -= 10
                    missed_opportunities.append("Mandatory recording consent statement was not given at call start.")
                    training_articles.append(article["id"])

            elif article["id"] == "kb_003":  # Escalation
                wants_supervisor = any(w in t for w in ["supervisor", "manager", "want to speak"])
                did_transfer = any(w in t for w in ["transfer", "connect you", "put you through"])
                if wants_supervisor and not did_transfer:
                    compliant = "no"
                    deviation = "Customer requested supervisor but transfer was not initiated within the interaction."
                    total_score -= 15
                    training_articles.append(article["id"])

            relevant_articles.append({
                "article_id": article["id"],
                "title": article["title"],
                "relevance_score": min(1.0, article["_score"] / 5),
                "key_points": [article["content"][:100] + "..."],
                "was_agent_compliant": compliant,
                "agent_deviation": deviation,
            })

        if not missed_opportunities:
            missed_opportunities = ["No major missed KB opportunities detected"]

        sop_score = max(0.0, total_score)
        return {
            "call_id": call_id,
            "relevant_articles": relevant_articles,
            "sop_compliance_score": round(sop_score, 1),
            "missed_knowledge_opportunities": missed_opportunities,
            "recommended_training_articles": list(set(training_articles)),
            "kb_summary": (
                f"[MOCK — {self.llm_name}] {len(relevant_articles)} KB article(s) relevant. "
                f"SOP compliance: {sop_score:.0f}%. "
                + (f"Training recommended: {', '.join(set(training_articles))}." if training_articles else "No training gaps identified.")
            ),
        }

    def _empty_result(self, call_id: str) -> dict:
        return {
            "call_id": call_id,
            "relevant_articles": [],
            "sop_compliance_score": 100.0,
            "missed_knowledge_opportunities": [],
            "recommended_training_articles": [],
            "kb_summary": "KB check skipped — no transcript available",
        }
