"""
Pydantic models and schemas for the Call Center AI Assistant.
Defines the data structures for all agents and workflow states.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ResolutionStatus(str, Enum):
    """Enumeration for call resolution status."""
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ESCALATED = "escalated"


class CallInput(BaseModel):
    """
    Input data structure for a call.
    Supports both audio files and transcript text.
    """
    call_id: str = Field(..., description="Unique identifier for the call")
    audio_path: Optional[str] = Field(None, description="Path to audio file (optional)")
    transcript_text: Optional[str] = Field(None, description="Raw transcript text (optional)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "call_id": "CALL_001",
                "transcript_text": "Agent: Hello, how can I help? Customer: I have a billing issue.",
                "metadata": {"category": "billing", "duration_seconds": 300}
            }
        }


class TranscriptOutput(BaseModel):
    """
    Output from transcription agent.
    Contains normalized transcript with speaker labels and metadata.
    """
    call_id: str = Field(..., description="Unique call identifier")
    transcript: str = Field(..., description="Normalized transcript text")
    speakers: List[str] = Field(default_factory=list, description="List of speakers (e.g., ['Agent', 'Customer'])")
    duration_seconds: Optional[int] = Field(None, description="Call duration in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "call_id": "CALL_001",
                "transcript": "Agent: Hello. Customer: Hi, I have an issue.",
                "speakers": ["Agent", "Customer"],
                "duration_seconds": 300
            }
        }


class SummaryOutput(BaseModel):
    """
    Output from summarization agent.
    Contains comprehensive call summary with action items and resolution status.
    """
    call_id: str = Field(..., description="Unique call identifier")
    summary: str = Field(..., description="Concise summary of the call")
    key_points: List[str] = Field(..., description="List of key discussion points")
    action_items: List[str] = Field(..., description="List of action items from the call")
    customer_issue: str = Field(..., description="Description of the customer's main issue")
    resolution_status: ResolutionStatus = Field(..., description="Current resolution status")

    class Config:
        json_schema_extra = {
            "example": {
                "call_id": "CALL_001",
                "summary": "Customer reported unauthorized charge of $50.",
                "key_points": ["Unauthorized charge", "Account compromised"],
                "action_items": ["Refund $50", "Reset password"],
                "customer_issue": "Fraudulent charge on account",
                "resolution_status": "resolved"
            }
        }


class QAScore(BaseModel):
    """
    Quality Assurance scoring output.
    Provides multi-dimensional assessment of agent performance.
    """
    call_id: str = Field(..., description="Unique call identifier")
    overall_score: float = Field(..., ge=0, le=100, description="Overall score (0-100)")
    empathy_score: float = Field(..., ge=0, le=25, description="Empathy score (0-25)")
    professionalism_score: float = Field(..., ge=0, le=25, description="Professionalism score (0-25)")
    resolution_score: float = Field(..., ge=0, le=25, description="Resolution effectiveness (0-25)")
    compliance_score: float = Field(..., ge=0, le=25, description="Compliance and adherence (0-25)")
    tone: str = Field(..., description="Overall tone assessment (e.g., professional, empathetic, neutral)")
    strengths: List[str] = Field(..., description="List of identified strengths")
    improvements: List[str] = Field(..., description="List of suggested improvements")

    class Config:
        json_schema_extra = {
            "example": {
                "call_id": "CALL_001",
                "overall_score": 85.5,
                "empathy_score": 24,
                "professionalism_score": 23,
                "resolution_score": 22,
                "compliance_score": 21,
                "tone": "Professional and empathetic",
                "strengths": ["Clear communication", "Problem-solving"],
                "improvements": ["Faster resolution", "More follow-up"]
            }
        }


class CallResult(BaseModel):
    """
    Complete call analysis result combining all agent outputs.
    This is the final deliverable from the workflow.
    """
    call_id: str = Field(..., description="Unique call identifier")
    input_data: CallInput = Field(..., description="Original input data")
    transcript: Optional[TranscriptOutput] = Field(None, description="Transcription output")
    summary: Optional[SummaryOutput] = Field(None, description="Summary output")
    qa_score: Optional[QAScore] = Field(None, description="Quality assessment")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    current_step: str = Field(default="initialized", description="Current workflow step")

    class Config:
        json_schema_extra = {
            "example": {
                "call_id": "CALL_001",
                "input_data": {"call_id": "CALL_001", "transcript_text": "..."},
                "transcript": {"call_id": "CALL_001", "transcript": "...", "speakers": ["Agent", "Customer"]},
                "summary": {"call_id": "CALL_001", "summary": "...", "key_points": [], "action_items": [], "customer_issue": "...", "resolution_status": "resolved"},
                "qa_score": {"call_id": "CALL_001", "overall_score": 85},
                "errors": [],
                "current_step": "completed"
            }
        }


class BenchmarkResult(BaseModel):
    """
    Result from multi-LLM benchmark comparison.
    Compares performance across Claude, GPT-4, and Gemini.
    """
    call_id: str = Field(..., description="Call identifier")
    claude_summary: Optional[SummaryOutput] = Field(None, description="Claude summarization result")
    gpt4_summary: Optional[SummaryOutput] = Field(None, description="GPT-4 summarization result")
    gemini_summary: Optional[SummaryOutput] = Field(None, description="Gemini summarization result")
    claude_qa: Optional[QAScore] = Field(None, description="Claude QA score")
    gpt4_qa: Optional[QAScore] = Field(None, description="GPT-4 QA score")
    gemini_qa: Optional[QAScore] = Field(None, description="Gemini QA score")
    timing: Dict[str, float] = Field(default_factory=dict, description="Processing time per LLM (seconds)")
    token_counts: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Token usage per LLM")
    errors: Dict[str, str] = Field(default_factory=dict, description="Any errors per LLM")

    class Config:
        json_schema_extra = {
            "example": {
                "call_id": "CALL_001",
                "timing": {"claude": 1.5, "gpt4": 2.1, "gemini": 1.8},
                "token_counts": {"claude": {"input": 100, "output": 200}},
                "errors": {}
            }
        }
