"""
Sentiment Analysis Agent: Analyzes per-turn sentiment and customer emotional arc.
Provides escalation risk score and agent tone across the full call.
A single LLM call covers the entire transcript — efficient and consistent.
"""

import logging
import time
from typing import Literal

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class TurnSentiment(BaseModel):
    speaker: str = Field(..., description="Agent or Customer")
    text_snippet: str = Field(..., description="Short quote from this turn (max 120 chars)")
    sentiment: str = Field(..., description="positive / neutral / negative")
    score: float = Field(..., ge=-1.0, le=1.0, description="-1=very negative, 1=very positive")


class SentimentSchema(BaseModel):
    overall_customer_sentiment: str = Field(..., description="positive / neutral / negative")
    customer_sentiment_trend: str = Field(..., description="improving / stable / degrading")
    agent_tone: str = Field(..., description="Brief agent tone description")
    escalation_risk: str = Field(..., description="low / medium / high")
    escalation_risk_reason: str = Field(..., description="One sentence reason for this risk level")
    turns: list[TurnSentiment] = Field(..., description="Per-turn sentiment, up to 10 representative turns")


class SentimentAgent:
    """
    Analyzes turn-by-turn sentiment and customer emotional trajectory.
    Useful for real-time QA monitoring and escalation prediction.
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
                    temperature=0.3,
                )
            elif llm_name == "gpt4":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=settings.GPT4_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.3,
                )
            elif llm_name == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    api_key=settings.GOOGLE_API_KEY,
                    temperature=0.3,
                )
            else:
                raise ValueError(f"Unsupported LLM: {llm_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for sentiment: {e}")
            raise

    def process(self, call_id: str, transcript: str) -> dict:
        """
        Analyze sentiment across the full call transcript.

        Returns:
            dict with overall_customer_sentiment, customer_sentiment_trend,
            agent_tone, escalation_risk, escalation_risk_reason, turns
        """
        if not transcript or not transcript.strip():
            return self._empty_result(call_id)

        if settings.MOCK_LLM:
            self.logger.info(f"[MOCK] Returning mock sentiment for {call_id}")
            return self._mock_sentiment(call_id, transcript)

        self.logger.info(f"[SENTIMENT] Analyzing call {call_id} with {self.llm_name}")
        start = time.time()

        try:
            method = "json_mode" if self.llm_name == "gemini" else None
            structured_llm = (
                self.llm.with_structured_output(SentimentSchema, method=method)
                if method
                else self.llm.with_structured_output(SentimentSchema)
            )
            result: SentimentSchema = structured_llm.invoke(self._build_prompt(transcript))

            self.logger.info(
                f"[SENTIMENT] Completed in {time.time() - start:.2f}s "
                f"— sentiment: {result.overall_customer_sentiment}, risk: {result.escalation_risk}"
            )
            return {
                "call_id": call_id,
                "overall_customer_sentiment": result.overall_customer_sentiment,
                "customer_sentiment_trend": result.customer_sentiment_trend,
                "agent_tone": result.agent_tone,
                "escalation_risk": result.escalation_risk,
                "escalation_risk_reason": result.escalation_risk_reason,
                "turns": [t.model_dump() for t in result.turns],
            }
        except Exception as e:
            self.logger.error(f"[SENTIMENT] Failed for {call_id}: {e}")
            return self._empty_result(call_id)

    def _build_prompt(self, transcript: str) -> str:
        return f"""Analyze the sentiment in the following call transcript.

TRANSCRIPT:
{transcript}

For each conversation turn (Agent/Customer lines), extract:
- speaker (Agent or Customer)
- text_snippet (short quote, max 120 chars)
- sentiment (positive / neutral / negative)
- score (float from -1.0 very negative to 1.0 very positive)

Include up to 10 representative turns covering the full arc of the conversation.

Also determine:
- overall_customer_sentiment: positive / neutral / negative (dominant customer mood)
- customer_sentiment_trend: improving / stable / degrading (did mood change across the call?)
- agent_tone: brief description (e.g. "empathetic and calm", "rushed but professional")
- escalation_risk: low / medium / high (based on frustration signals and unresolved tension)
- escalation_risk_reason: one sentence explaining the risk level"""

    def _mock_sentiment(self, call_id: str, transcript: str) -> dict:
        t = transcript.lower()
        is_angry = any(w in t for w in ["furious", "unacceptable", "terrible", "outraged", "lawsuit", "horrible"])
        is_happy = any(w in t for w in ["thank you so much", "amazing", "wonderful", "great job", "appreciate"])
        is_escalation = any(w in t for w in ["supervisor", "manager", "escalat"])

        lines = [ln for ln in transcript.split('\n') if ln.strip() and ':' in ln]
        turns = []
        for i, line in enumerate(lines[:10]):
            speaker = "Agent" if line.strip().startswith("Agent") else "Customer"
            snippet = line.strip()[:110]
            if speaker == "Customer" and is_angry:
                sentiment, score = "negative", round(-0.6 - (i * 0.03), 2)
            elif is_happy:
                sentiment, score = "positive", 0.75
            else:
                sentiment, score = "neutral", 0.05
            turns.append({"speaker": speaker, "text_snippet": snippet,
                           "sentiment": sentiment, "score": max(-1.0, min(1.0, score))})

        if is_angry or is_escalation:
            overall, trend, risk = "negative", "degrading", "high"
            reason = "Customer expressed repeated frustration without satisfactory resolution"
            tone = f"[MOCK — {self.llm_name}] Calm under pressure but resolution was delayed"
        elif is_happy:
            overall, trend, risk = "positive", "improving", "low"
            reason = "Customer expressed clear satisfaction throughout the interaction"
            tone = f"[MOCK — {self.llm_name}] Warm, engaged, and customer-centric"
        else:
            overall, trend, risk = "neutral", "stable", "low"
            reason = "No strong frustration signals detected in the transcript"
            tone = f"[MOCK — {self.llm_name}] Professional and solution-focused"

        return {
            "call_id": call_id,
            "overall_customer_sentiment": overall,
            "customer_sentiment_trend": trend,
            "agent_tone": tone,
            "escalation_risk": risk,
            "escalation_risk_reason": reason,
            "turns": turns,
        }

    def _empty_result(self, call_id: str) -> dict:
        return {
            "call_id": call_id,
            "overall_customer_sentiment": "unknown",
            "customer_sentiment_trend": "unknown",
            "agent_tone": "Unable to assess",
            "escalation_risk": "unknown",
            "escalation_risk_reason": "",
            "turns": [],
        }
