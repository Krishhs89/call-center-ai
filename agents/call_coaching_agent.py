"""
Call Coaching Agent: Generates personalised coaching feedback for call center agents.

Analyses QA score dimensions (empathy, resolution, communication, compliance)
and the transcript to produce targeted, actionable coaching tips with specific
examples from the actual call.

Business value:
  - Replaces manual supervisor coaching (one-to-many, generic) with per-call,
    data-driven feedback tied to specific transcript moments
  - 15-20% improvement in QA scores within 90 days (industry benchmark)
  - Reduces new-agent ramp time by ~30% when coaching is tied to real calls
  - Replicates Calabrio, NICE CXone, Qualtrics coaching capabilities

Real-world usage:
  - Generated for 100% of scored calls
  - Delivered to agent via supervisor dashboard or direct email/Slack message
  - Used by QA teams for monthly 1:1 coaching sessions
  - Feeds into agent performance scorecards and career development plans
"""

import logging
import time
from typing import Literal, Optional

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class CoachingTip(BaseModel):
    dimension: str = Field(..., description="QA dimension: empathy / resolution / communication / compliance / product_knowledge / process_adherence")
    current_score: float = Field(..., description="Agent's score on this dimension (0-10)")
    priority: str = Field(..., description="immediate / high / medium / low — based on score gap")
    what_happened: str = Field(..., description="What the agent did or failed to do, with a quote from the transcript")
    what_to_do_instead: str = Field(..., description="Specific alternative action or phrase the agent should use next time")
    example_script: str = Field(..., description="Word-for-word example of what the agent should say")


class CallCoachingSchema(BaseModel):
    agent_strengths: list[str] = Field(default_factory=list, description="2-3 things the agent did well — acknowledge before coaching")
    coaching_tips: list[CoachingTip] = Field(default_factory=list, description="Prioritised coaching tips, worst dimension first")
    overall_coaching_priority: str = Field(..., description="immediate / high / medium / low — overall urgency")
    coaching_summary: str = Field(..., description="One paragraph coaching narrative for the supervisor to read to the agent")
    next_call_focus: str = Field(..., description="The single most important thing the agent should focus on in their next call")
    estimated_improvement: str = Field(..., description="Expected QA score improvement if agent applies this coaching (e.g. '+8-12 points in 2 weeks')")


class CallCoachingAgent:
    """
    Generates personalised, transcript-grounded coaching feedback.

    Input: QA score dimensions + transcript + optional sentiment signals.
    Output: Prioritised coaching tips with exact script examples.

    In production, coaching output is delivered via:
    - Supervisor dashboard (Calabrio, NICE CXone)
    - Agent self-coaching portal
    - Automated Slack/Teams message to agent after call
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
                    temperature=0.4,  # Slightly higher — coaching benefits from varied phrasing
                )
            elif llm_name == "gpt4":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=settings.GPT4_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.4,
                )
            elif llm_name == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    api_key=settings.GOOGLE_API_KEY,
                    temperature=0.4,
                )
            else:
                raise ValueError(f"Unsupported LLM: {llm_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for coaching: {e}")
            raise

    def process(
        self,
        call_id: str,
        transcript: str,
        qa_score: Optional[dict] = None,
        sentiment: Optional[dict] = None,
    ) -> dict:
        """
        Generate coaching feedback for the agent on this call.

        Args:
            call_id: Unique call identifier
            transcript: PII-redacted transcript
            qa_score: Output from QualityScoreAgent (scores per dimension)
            sentiment: Output from SentimentAgent (agent tone signals)

        Returns:
            dict with agent_strengths, coaching_tips, coaching_summary,
                  next_call_focus, estimated_improvement
        """
        if not transcript or not transcript.strip():
            return self._empty_result(call_id)

        if settings.MOCK_LLM:
            self.logger.info(f"[MOCK] Returning mock coaching for {call_id}")
            return self._mock_coaching(call_id, transcript, qa_score, sentiment)

        self.logger.info(f"[COACHING] Generating coaching for call {call_id} with {self.llm_name}")
        start = time.time()

        try:
            method = "json_mode" if self.llm_name == "gemini" else None
            structured_llm = (
                self.llm.with_structured_output(CallCoachingSchema, method=method)
                if method
                else self.llm.with_structured_output(CallCoachingSchema)
            )
            result: CallCoachingSchema = structured_llm.invoke(
                self._build_prompt(transcript, qa_score, sentiment)
            )

            self.logger.info(
                f"[COACHING] Completed in {time.time() - start:.2f}s — "
                f"{len(result.coaching_tips)} tip(s), priority={result.overall_coaching_priority}"
            )
            return {
                "call_id": call_id,
                "agent_strengths": result.agent_strengths,
                "coaching_tips": [t.model_dump() for t in result.coaching_tips],
                "overall_coaching_priority": result.overall_coaching_priority,
                "coaching_summary": result.coaching_summary,
                "next_call_focus": result.next_call_focus,
                "estimated_improvement": result.estimated_improvement,
            }

        except Exception as e:
            self.logger.error(f"[COACHING] Failed for {call_id}: {e}")
            return self._empty_result(call_id)

    def _build_prompt(self, transcript: str, qa_score: Optional[dict], sentiment: Optional[dict]) -> str:
        score_block = ""
        if qa_score:
            dims = qa_score.get("dimension_scores", {})
            score_block = (
                f"\nQA SCORES (0-10 per dimension):\n"
                + "\n".join(f"  {k}: {v}" for k, v in dims.items())
                + f"\n  Overall: {qa_score.get('overall_score', 'N/A')}/100"
            )

        sentiment_block = ""
        if sentiment:
            sentiment_block = (
                f"\nSENTIMENT SIGNALS:\n"
                f"  Agent tone: {sentiment.get('agent_tone')}\n"
                f"  Customer sentiment: {sentiment.get('overall_customer_sentiment')}\n"
                f"  Escalation risk: {sentiment.get('escalation_risk')}"
            )

        return f"""You are an expert call center quality coach providing personalised feedback to an agent.
{score_block}{sentiment_block}

TRANSCRIPT:
{transcript}

Generate specific, actionable coaching grounded in the actual transcript.

Rules:
1. Acknowledge 2-3 strengths FIRST — agents disengage if feedback is only negative
2. Focus coaching tips on the LOWEST scoring dimensions — that's where the biggest gain is
3. Each tip must quote or paraphrase a SPECIFIC moment from the transcript
4. Provide a word-for-word example_script the agent can practise and reuse
5. overall_coaching_priority:
   - immediate: any dimension score < 4 or overall score < 60
   - high: overall score 60-74
   - medium: overall score 75-84
   - low: overall score 85+
6. estimated_improvement: be realistic, e.g. "+5-8 points within 1 week with daily practice"
7. next_call_focus: pick ONE thing — the highest-leverage change"""

    def _mock_coaching(self, call_id: str, transcript: str, qa_score: Optional[dict], sentiment: Optional[dict]) -> dict:
        t = transcript.lower()
        dims = qa_score.get("dimension_scores", {}) if qa_score else {}
        overall = qa_score.get("overall_score", 75.0) if qa_score else 75.0

        # Find weakest dimension
        weakest_dim = "empathy"
        weakest_score = 10.0
        for dim, score in dims.items():
            if isinstance(score, (int, float)) and score < weakest_score:
                weakest_score = score
                weakest_dim = dim

        # Agent tone from sentiment
        agent_tone = sentiment.get("agent_tone", "neutral") if sentiment else "neutral"

        coaching_tips = []

        # Tip 1: weakest dimension
        if weakest_score < 7:
            coaching_tips.append({
                "dimension": weakest_dim,
                "current_score": float(weakest_score),
                "priority": "immediate" if weakest_score < 4 else "high",
                "what_happened": f"Agent scored {weakest_score}/10 on {weakest_dim}. Specific moments in the transcript showed room for improvement in this area.",
                "what_to_do_instead": f"Actively improve {weakest_dim} by following the recommended script and checklist for this dimension.",
                "example_script": f"'I completely understand how frustrating this must be for you. Let me make sure we resolve this fully today.'",
            })

        # Tip 2: empathy if poor sentiment
        if "frustrat" in t or "angry" in t or "upset" in t:
            if weakest_dim != "empathy":
                coaching_tips.append({
                    "dimension": "empathy",
                    "current_score": float(dims.get("empathy", 6.0)),
                    "priority": "high",
                    "what_happened": "Customer showed signs of frustration but the agent did not explicitly acknowledge the emotional state before moving to resolution.",
                    "what_to_do_instead": "Use an empathy statement BEFORE moving to the solution. Customers need to feel heard first.",
                    "example_script": "'I can hear how frustrated you are, and that's completely understandable. Let me sort this out for you right now.'",
                })

        # Tip 3: closing / resolution
        if "resolution" in dims and dims.get("resolution", 10) < 8:
            coaching_tips.append({
                "dimension": "resolution",
                "current_score": float(dims.get("resolution", 6.5)),
                "priority": "medium",
                "what_happened": "The resolution was not confirmed clearly at the end of the call.",
                "what_to_do_instead": "Always close the loop: summarise what was resolved, confirm next steps, and check if there is anything else.",
                "example_script": "'So to confirm: I've processed your refund of $X and you'll see it in 3-5 business days. Is there anything else I can help you with today?'",
            })

        if not coaching_tips:
            coaching_tips.append({
                "dimension": "communication",
                "current_score": float(dims.get("communication", 7.5)),
                "priority": "low",
                "what_happened": "Agent communicated clearly but could improve pace and signposting.",
                "what_to_do_instead": "Use transition phrases to signal when you are moving between topics.",
                "example_script": "'Great, I've got that sorted. Now let me also check your account to make sure everything else looks good.'",
            })

        strengths = [
            "Maintained a professional and polite tone throughout the interaction",
            "Listened to the customer without interrupting",
        ]
        if "thank" in t:
            strengths.append("Used positive closing language and expressed appreciation for the customer's patience")

        if overall < 60:
            priority, improvement = "immediate", "+12-18 points with focused coaching over 2 weeks"
        elif overall < 75:
            priority, improvement = "high", "+8-12 points within 1 week"
        elif overall < 85:
            priority, improvement = "medium", "+5-8 points with daily practice"
        else:
            priority, improvement = "low", "+2-4 points by refining closing technique"

        return {
            "call_id": call_id,
            "agent_strengths": strengths,
            "coaching_tips": coaching_tips,
            "overall_coaching_priority": priority,
            "coaching_summary": (
                f"[MOCK — {self.llm_name}] The agent demonstrated professionalism but has a clear opportunity to improve "
                f"in {weakest_dim}. With targeted practice on the provided example scripts, a score improvement of "
                f"{improvement} is achievable."
            ),
            "next_call_focus": f"Apply the empathy statement before moving to resolution — this single change typically adds 5-8 points to customer satisfaction scores.",
            "estimated_improvement": improvement,
        }

    def _empty_result(self, call_id: str) -> dict:
        return {
            "call_id": call_id,
            "agent_strengths": [],
            "coaching_tips": [],
            "overall_coaching_priority": "unknown",
            "coaching_summary": "Coaching skipped — no transcript or QA score available",
            "next_call_focus": "N/A",
            "estimated_improvement": "N/A",
        }
