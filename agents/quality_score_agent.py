"""
Quality Score Agent: Evaluates agent performance on multiple dimensions.
Scores empathy, professionalism, resolution, and compliance.
Uses structured output for reliable scoring.
"""

import logging
from typing import Literal
import time

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field

from utils.schemas import QAScore
from config.settings import settings

logger = logging.getLogger(__name__)


class QAScoreSchema(BaseModel):
    """Schema for structured QA scoring output."""
    empathy_score: float = Field(..., ge=0, le=25, description="Empathy score (0-25)")
    professionalism_score: float = Field(..., ge=0, le=25, description="Professionalism score (0-25)")
    resolution_score: float = Field(..., ge=0, le=25, description="Resolution effectiveness (0-25)")
    compliance_score: float = Field(..., ge=0, le=25, description="Compliance score (0-25)")
    tone: str = Field(..., description="Overall tone assessment")
    strengths: list[str] = Field(..., description="List of 2-3 identified strengths")
    improvements: list[str] = Field(..., description="List of 2-3 improvement suggestions")


class QualityScoreAgent:
    """
    Evaluates call quality on multiple dimensions.
    Provides scores and actionable feedback.
    """

    def __init__(self, llm_name: Literal["claude", "gpt4", "gemini"] = "claude"):
        """
        Initialize QA scoring agent with specified LLM.

        Args:
            llm_name: LLM to use (claude, gpt4, or gemini)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.llm_name = llm_name
        self.llm = self._initialize_llm(llm_name)

    def _initialize_llm(self, llm_name: Literal["claude", "gpt4", "gemini"]) -> BaseLanguageModel:
        """
        Initialize the specified LLM.

        Args:
            llm_name: LLM to initialize

        Returns:
            BaseLanguageModel: Initialized language model

        Raises:
            ValueError: If LLM not available or API key missing
        """
        try:
            if llm_name == "claude":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=settings.CLAUDE_MODEL,
                    api_key=settings.ANTHROPIC_API_KEY,
                    temperature=0.7,
                )
            elif llm_name == "gpt4":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=settings.GPT4_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.7,
                )
            elif llm_name == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    api_key=settings.GOOGLE_API_KEY,
                    temperature=0.7,
                )
            else:
                raise ValueError(f"Unsupported LLM: {llm_name}")
        except ImportError as e:
            self.logger.error(f"Failed to import LLM: {e}")
            raise ValueError(f"LLM {llm_name} not available. Install required dependencies.")

    def process(self, call_id: str, transcript: str, rag_context: str = "") -> QAScore:
        """
        Score a call transcript across multiple dimensions.

        Args:
            call_id: Call identifier
            transcript: Call transcript text
            rag_context: Formatted context from similar past calls (optional)

        Returns:
            QAScore: Multi-dimensional quality assessment

        Raises:
            ValueError: If transcript is empty or scoring fails
        """
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")

        # Mock mode: return context-aware response without hitting any API
        if settings.MOCK_LLM:
            self.logger.info(f"[MOCK] Returning mock QA score for {call_id}")
            return self._mock_qa_score(call_id, transcript)

        self.logger.info(f"Scoring call {call_id} with {self.llm_name}" +
                         (" [+RAG]" if rag_context else ""))
        start_time = time.time()

        prompt = self._build_prompt(transcript, rag_context)

        try:
            # Gemini function-calling rejects list[str] fields without explicit items;
            # use json_mode instead which works across all Gemini versions.
            method = "json_mode" if self.llm_name == "gemini" else None
            structured_llm = (
                self.llm.with_structured_output(QAScoreSchema, method=method)
                if method
                else self.llm.with_structured_output(QAScoreSchema)
            )
            result = structured_llm.invoke(prompt)

            elapsed_time = time.time() - start_time

            # Calculate overall score (sum of all dimension scores)
            overall_score = (
                result.empathy_score +
                result.professionalism_score +
                result.resolution_score +
                result.compliance_score
            )

            output = QAScore(
                call_id=call_id,
                overall_score=overall_score,
                empathy_score=result.empathy_score,
                professionalism_score=result.professionalism_score,
                resolution_score=result.resolution_score,
                compliance_score=result.compliance_score,
                tone=result.tone,
                strengths=result.strengths,
                improvements=result.improvements,
            )

            self.logger.info(f"QA scoring completed for {call_id} in {elapsed_time:.2f}s. Score: {overall_score:.1f}/100")
            return output

        except Exception as e:
            self.logger.error(f"QA scoring failed for {call_id}: {e}")
            # Return a minimal valid response
            return QAScore(
                call_id=call_id,
                overall_score=0.0,
                empathy_score=0,
                professionalism_score=0,
                resolution_score=0,
                compliance_score=0,
                tone="Unable to assess",
                strengths=[],
                improvements=[],
            )

    def _mock_qa_score(self, call_id: str, transcript: str) -> QAScore:
        """Generate a context-aware mock QA score by parsing the transcript."""
        t = transcript.lower()

        is_compliment = any(w in t for w in ["thank you so much", "amazing", "wonderful", "great job", "you've been fantastic", "happy", "pleased", "excellent service", "appreciate", "compliment"])
        is_angry = any(w in t for w in ["unacceptable", "furious", "terrible", "horrible", "outraged", "ridiculous", "lawsuit"])
        is_escalation = any(w in t for w in ["escalat", "supervisor", "manager", "transfer to"])
        is_technical = any(w in t for w in ["internet", "wifi", "wi-fi", "router", "connection", "error", "crash", "reboot", "reset", "device", "login", "password", "software", "network"])
        is_healthcare = any(w in t for w in ["prescription", "medication", "doctor", "physician", "appointment", "patient", "diagnosis", "treatment"])

        # Tune scores based on detected context
        if is_compliment:
            empathy, professionalism, resolution, compliance = 24.0, 24.0, 23.0, 23.0
            tone = f"[MOCK — {self.llm_name}] Warm, positive, and customer-centric"
            strengths = [
                "Excellent rapport built with the customer throughout the call",
                "Agent acknowledged and expressed genuine appreciation for the feedback",
                "Professional tone maintained from start to finish",
            ]
            improvements = [
                "Could proactively invite the customer to share feedback online",
                "Consider offering loyalty program information during positive interactions",
            ]
        elif is_angry or is_escalation:
            empathy, professionalism, resolution, compliance = 18.0, 20.0, 16.0, 20.0
            tone = f"[MOCK — {self.llm_name}] Calm under pressure but resolution incomplete"
            strengths = [
                "Agent remained calm and professional despite customer frustration",
                "Correctly followed escalation protocol when required",
                "Accurately documented the issue for the next tier",
            ]
            improvements = [
                "De-escalation techniques could be applied earlier in the call",
                "Offer more empathy statements to acknowledge customer frustration",
                "Provide a clearer estimated timeline for resolution",
            ]
        elif is_healthcare:
            empathy, professionalism, resolution, compliance = 23.0, 22.0, 21.0, 22.0
            tone = f"[MOCK — {self.llm_name}] Empathetic and medically informed"
            strengths = [
                "Agent showed genuine concern for patient wellbeing",
                "Provided clear and actionable next steps for the patient",
                "Correctly followed healthcare compliance protocols",
            ]
            improvements = [
                "Could confirm patient understanding by repeating key instructions",
                "Proactively mention alternative care options earlier in the call",
            ]
        elif is_technical:
            empathy, professionalism, resolution, compliance = 20.0, 22.0, 21.0, 21.0
            tone = f"[MOCK — {self.llm_name}] Patient and methodical"
            strengths = [
                "Agent guided customer through troubleshooting steps clearly",
                "Demonstrated strong technical knowledge",
                "Maintained patience throughout the diagnostic process",
            ]
            improvements = [
                "Summarize the troubleshooting steps taken before ending the call",
                "Offer a follow-up contact in case the issue recurs",
            ]
        else:
            empathy, professionalism, resolution, compliance = 21.0, 21.0, 20.0, 21.0
            tone = f"[MOCK — {self.llm_name}] Professional and solution-focused"
            strengths = [
                "Agent communicated clearly and maintained a professional tone",
                "Issue was addressed efficiently within a reasonable timeframe",
                "Customer was kept informed throughout the interaction",
            ]
            improvements = [
                "Could offer proactive solutions before the customer asks",
                "Verify understanding by summarizing the issue back to the customer",
            ]

        overall = empathy + professionalism + resolution + compliance
        return QAScore(
            call_id=call_id,
            overall_score=overall,
            empathy_score=empathy,
            professionalism_score=professionalism,
            resolution_score=resolution,
            compliance_score=compliance,
            tone=tone,
            strengths=strengths,
            improvements=improvements,
        )

    def _build_prompt(self, transcript: str, rag_context: str = "") -> str:
        """
        Build the prompt for QA scoring, optionally including RAG context.

        Args:
            transcript: Call transcript
            rag_context: Formatted similar-call context from vector store

        Returns:
            str: Formatted prompt
        """
        context_block = ""
        if rag_context:
            context_block = (
                f"\n\n{rag_context}\n\n"
                "Consider the similar past calls above when calibrating your scores "
                "(e.g. consistent scoring patterns for similar issue types).\n"
            )

        return f"""Evaluate the following call transcript and provide a quality assessment of the agent's performance.{context_block}
TRANSCRIPT:
{transcript}

Score the agent on the following dimensions (0-25 scale each):
1. EMPATHY (0-25): Did the agent understand and acknowledge customer emotions? Show genuine concern?
2. PROFESSIONALISM (0-25): Was the communication clear, courteous, and professional throughout?
3. RESOLUTION (0-25): Did the agent effectively resolve or address the customer's issue?
4. COMPLIANCE (0-25): Were company policies and procedures followed correctly?

Also provide:
- Overall tone assessment (e.g., "professional and empathetic", "neutral and efficient", etc.)
- 2-3 identified strengths
- 2-3 actionable improvement suggestions

Ensure all scores are between 0-25 and reflect the agent's actual performance based on the transcript."""

    def validate_scores(self, score: QAScore) -> bool:
        """
        Validate that all scores are in proper ranges.

        Args:
            score: QAScore object to validate

        Returns:
            bool: True if all scores are valid
        """
        dimension_scores = [
            score.empathy_score,
            score.professionalism_score,
            score.resolution_score,
            score.compliance_score,
        ]

        # Check individual dimensions are 0-25
        for s in dimension_scores:
            if not (0 <= s <= 25):
                self.logger.warning(f"Score {s} out of range [0, 25]")
                return False

        # Check overall is 0-100
        if not (0 <= score.overall_score <= 100):
            self.logger.warning(f"Overall score {score.overall_score} out of range [0, 100]")
            return False

        return True
