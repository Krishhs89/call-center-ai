"""
Summarization Agent: Extracts key information from call transcripts.
Uses LangChain with configurable LLM (Claude, GPT-4, or Gemini).
Structured output with function calling for reliable JSON responses.
"""

import json
import logging
from typing import Optional, Literal
import time

from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field

from utils.schemas import SummaryOutput, ResolutionStatus
from config.settings import settings

logger = logging.getLogger(__name__)


class SummarizationSchema(BaseModel):
    """Schema for structured summarization output."""
    summary: str = Field(..., description="Concise summary of the call (2-3 sentences)")
    key_points: list[str] = Field(..., description="List of 3-5 key discussion points")
    action_items: list[str] = Field(..., description="List of 2-5 action items resulting from the call")
    customer_issue: str = Field(..., description="The primary issue or concern the customer raised")
    resolution_status: str = Field(
        ...,
        description="Resolution status: 'resolved', 'unresolved', or 'escalated'"
    )


class SummarizationAgent:
    """
    Summarizes call transcripts using configurable LLM.
    Extracts summary, key points, action items, and resolution status.
    """

    def __init__(self, llm_name: Literal["claude", "gpt4", "gemini"] = "claude"):
        """
        Initialize summarization agent with specified LLM.

        Args:
            llm_name: LLM to use (claude, gpt4, or gemini)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.llm_name = llm_name
        self.llm = self._initialize_llm(llm_name)
        self.token_count = 0

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

    def process(self, call_id: str, transcript: str) -> SummaryOutput:
        """
        Summarize a call transcript.

        Args:
            call_id: Call identifier
            transcript: Call transcript text

        Returns:
            SummaryOutput: Structured summary with key information

        Raises:
            ValueError: If transcript is empty or summarization fails
        """
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")

        # Mock mode: return context-aware response without hitting any API
        if settings.MOCK_LLM:
            self.logger.info(f"[MOCK] Returning mock summary for {call_id}")
            return self._mock_summary(call_id, transcript)

        self.logger.info(f"Summarizing call {call_id} with {self.llm_name}")
        start_time = time.time()

        prompt = self._build_prompt(transcript)

        try:
            # Gemini function-calling rejects list[str] fields without explicit items;
            # use json_mode instead which works across all Gemini versions.
            method = "json_mode" if self.llm_name == "gemini" else None
            structured_llm = (
                self.llm.with_structured_output(SummarizationSchema, method=method)
                if method
                else self.llm.with_structured_output(SummarizationSchema)
            )
            result = structured_llm.invoke(prompt)

            elapsed_time = time.time() - start_time

            # Validate resolution status
            resolution_status = result.resolution_status.lower()
            if resolution_status not in ["resolved", "unresolved", "escalated"]:
                resolution_status = "unresolved"

            output = SummaryOutput(
                call_id=call_id,
                summary=result.summary,
                key_points=result.key_points,
                action_items=result.action_items,
                customer_issue=result.customer_issue,
                resolution_status=ResolutionStatus(resolution_status),
            )

            self.logger.info(f"Summarization completed for {call_id} in {elapsed_time:.2f}s")
            return output

        except Exception as e:
            self.logger.error(f"Summarization failed for {call_id}: {e}")
            # Return a minimal valid response
            return SummaryOutput(
                call_id=call_id,
                summary="Unable to summarize transcript",
                key_points=[],
                action_items=[],
                customer_issue="Unable to determine",
                resolution_status=ResolutionStatus.UNRESOLVED,
            )

    def _mock_summary(self, call_id: str, transcript: str) -> SummaryOutput:
        """Generate a context-aware mock summary by parsing the transcript."""
        t = transcript.lower()

        # Detect sentiment/call type
        is_compliment = any(w in t for w in ["thank you so much", "amazing", "wonderful", "great job", "you've been fantastic", "happy", "pleased", "excellent service", "appreciate", "compliment"])
        is_angry = any(w in t for w in ["unacceptable", "furious", "terrible", "horrible", "outraged", "ridiculous", "this is insane", "lawsuit", "furious"])
        is_escalation = any(w in t for w in ["escalat", "supervisor", "manager", "transfer to", "speak to someone"])
        is_technical = any(w in t for w in ["internet", "wifi", "wi-fi", "router", "connection", "outage", "error", "crash", "reboot", "reset", "device", "login", "password", "software", "update", "install", "network", "server"])
        is_billing = any(w in t for w in ["bill", "billing", "charge", "invoice", "payment", "refund", "credit", "overcharg", "fee"])
        is_order = any(w in t for w in ["order", "shipment", "delivery", "tracking", "package", "item", "product", "return", "exchange"])
        is_healthcare = any(w in t for w in ["prescription", "medication", "doctor", "physician", "appointment", "referral", "insurance", "clinic", "hospital", "patient", "diagnosis", "treatment"])
        is_travel = any(w in t for w in ["flight", "booking", "reservation", "hotel", "ticket", "airline", "cancel", "seat", "baggage", "itinerary"])

        # Detect resolution from transcript end
        resolved_signals = ["resolved", "fixed", "taken care", "all set", "sorted out", "confirmed", "confirmed satisfaction", "thank you for calling", "have a great day", "is there anything else"]
        unresolved_signals = ["still not working", "not resolved", "follow up", "callback", "will check back", "no solution"]
        escalated_signals = ["escalating", "transfer", "supervisor", "manager will", "escalated"]

        if is_escalation or any(w in t for w in escalated_signals):
            resolution = ResolutionStatus.ESCALATED
        elif any(w in t for w in unresolved_signals):
            resolution = ResolutionStatus.UNRESOLVED
        else:
            resolution = ResolutionStatus.RESOLVED

        # Extract first customer name if present (look for "I'm <Name>" or "This is <Name>")
        import re
        name_match = re.search(r"(?:i'?m|this is|my name is)\s+([A-Z][a-z]+)", transcript)
        customer_name = name_match.group(1) if name_match else "the customer"

        # Build context-specific content
        if is_compliment:
            issue = "Customer called to express satisfaction and compliment the service"
            summary = (
                f"[MOCK — {self.llm_name}] {customer_name.capitalize()} contacted support to share positive feedback about a recent interaction. "
                "The agent thanked the customer and noted the compliment for team recognition. "
                "The call concluded on a positive note with no outstanding issues."
            )
            key_points = [
                "Customer expressed satisfaction with recent service experience",
                "Agent acknowledged the feedback and thanked the customer",
                "Compliment logged for team recognition",
                "No issues or action items raised during the call",
            ]
            action_items = [
                "Log compliment in CRM for team recognition",
                "Share feedback with the relevant team member",
            ]
        elif is_healthcare:
            issue = "Patient seeking prescription refill or medical service assistance"
            summary = (
                f"[MOCK — {self.llm_name}] {customer_name.capitalize()} contacted patient services regarding a medical or prescription-related concern. "
                "The agent reviewed the account and coordinated with the appropriate medical team. "
                "A resolution path was provided to the patient."
            )
            key_points = [
                "Patient raised a healthcare or prescription-related concern",
                "Agent verified patient identity and account details",
                "Coordination with medical team or pharmacy was initiated",
                "Patient was informed of next steps and timeline",
            ]
            action_items = [
                "Send urgent request to prescribing physician",
                "Confirm patient pre-registration or appointment",
                "Follow up if no response within 24 hours",
            ]
        elif is_technical:
            issue = "Technical issue with service or device requiring troubleshooting"
            summary = (
                f"[MOCK — {self.llm_name}] {customer_name.capitalize()} contacted support with a technical issue. "
                "The agent walked through troubleshooting steps to diagnose and address the problem. "
                "{'The issue was resolved during the call.' if resolution == ResolutionStatus.RESOLVED else 'The issue required further investigation and a follow-up was scheduled.'}"
            )
            key_points = [
                "Customer reported a technical problem affecting service",
                "Agent guided customer through diagnostic and troubleshooting steps",
                "Root cause was identified during the interaction",
                "Resolution steps were communicated clearly",
            ]
            action_items = [
                "Send troubleshooting steps via email for reference",
                "Escalate to tier-2 support if issue recurs",
                "Schedule follow-up check within 48 hours",
            ]
        elif is_order:
            issue = "Order, delivery, or return inquiry"
            summary = (
                f"[MOCK — {self.llm_name}] {customer_name.capitalize()} contacted support regarding an order or delivery concern. "
                "The agent located the order details and provided a status update or initiated the necessary action. "
                "The customer was informed of the expected resolution timeline."
            )
            key_points = [
                "Customer inquired about an order or delivery status",
                "Agent located the order and reviewed current status",
                "Customer was provided with tracking or resolution details",
                "Any necessary return or exchange process was initiated",
            ]
            action_items = [
                "Send updated order tracking information to customer",
                "Initiate return or exchange process if requested",
                "Confirm delivery resolution with logistics team",
            ]
        elif is_billing:
            issue = "Billing inquiry or payment discrepancy"
            summary = (
                f"[MOCK — {self.llm_name}] {customer_name.capitalize()} contacted support regarding a billing or payment concern. "
                "The agent reviewed the account and addressed the discrepancy. "
                "{'The issue was resolved and any applicable credits were applied.' if resolution == ResolutionStatus.RESOLVED else 'The billing issue requires further review by the finance team.'}"
            )
            key_points = [
                "Customer raised a concern about a charge or billing statement",
                "Agent reviewed account transaction history",
                "Discrepancy was identified and addressed",
                "Customer was informed of any credits or adjustments applied",
            ]
            action_items = [
                "Send billing adjustment confirmation to customer",
                "Flag account for billing team review",
                "Update case notes with resolution details",
            ]
        elif is_travel:
            issue = "Travel booking, cancellation, or reservation inquiry"
            summary = (
                f"[MOCK — {self.llm_name}] {customer_name.capitalize()} contacted support regarding a travel booking or reservation. "
                "The agent reviewed the booking details and assisted with the requested changes. "
                "The customer confirmed the updated arrangements."
            )
            key_points = [
                "Customer inquired about a travel reservation or booking",
                "Agent reviewed the booking and available options",
                "Changes or cancellations were processed as requested",
                "Customer received confirmation of updated itinerary",
            ]
            action_items = [
                "Send updated booking confirmation via email",
                "Process any applicable refunds or credits",
                "Confirm changes with partner airline or hotel",
            ]
        else:
            issue = "General customer service inquiry"
            summary = (
                f"[MOCK — {self.llm_name}] {customer_name.capitalize()} contacted support with a general inquiry. "
                "The agent assisted the customer and addressed their concerns effectively. "
                "The call was concluded with a clear resolution or next steps."
            )
            key_points = [
                "Customer raised a service or account-related inquiry",
                "Agent gathered relevant details and reviewed the account",
                "Appropriate action was taken to address the concern",
                "Customer was provided with a clear resolution or follow-up plan",
            ]
            action_items = [
                "Update CRM with call outcome and notes",
                "Follow up with customer if issue is not fully resolved",
            ]

        return SummaryOutput(
            call_id=call_id,
            summary=summary,
            key_points=key_points,
            action_items=action_items,
            customer_issue=issue,
            resolution_status=resolution,
        )

    def _build_prompt(self, transcript: str) -> str:
        """
        Build the prompt for summarization.

        Args:
            transcript: Call transcript

        Returns:
            str: Formatted prompt
        """
        return f"""Analyze the following call transcript and provide a structured summary.

TRANSCRIPT:
{transcript}

Please extract:
1. A concise 2-3 sentence summary of the call
2. 3-5 key discussion points
3. 2-5 action items that should be taken
4. The primary customer issue
5. The resolution status (resolved, unresolved, or escalated)

Provide your response in the specified format."""

    def get_token_count(self) -> int:
        """
        Get approximate token count for the last operation.

        Returns:
            int: Token count
        """
        return self.token_count
