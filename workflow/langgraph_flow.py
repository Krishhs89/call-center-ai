"""
LangGraph Workflow: Orchestrates the multi-agent call processing pipeline.
Version 2 — adds PII Redaction, RAG Retrieval, and Sentiment Analysis nodes.
Implements conditional routing and error handling.
Compatible with LangSmith tracing.
"""

import logging
from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from agents.intake_agent import IntakeAgent
from agents.transcription_agent import TranscriptionAgent
from agents.pii_redaction_agent import PIIRedactionAgent
from agents.rag_retrieval_agent import RAGRetrievalAgent
from agents.sentiment_agent import SentimentAgent
from agents.compliance_checker_agent import ComplianceCheckerAgent
from agents.summarization_agent import SummarizationAgent
from agents.quality_score_agent import QualityScoreAgent
from agents.routing_agent import RoutingAgent
from utils.schemas import CallResult, CallInput, TranscriptOutput, SummaryOutput, QAScore
from utils.memory import call_memory
from utils.vector_store import store_call_embedding

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State dictionary for the LangGraph workflow (Version 2)."""
    call_id: str
    input_data: CallInput
    transcript: Optional[TranscriptOutput]
    redacted_transcript: str        # PII-masked version of the transcript
    pii_summary: dict               # {field: count} of redacted PII items
    rag_context: str                # Formatted context from similar past calls
    sentiment: Optional[dict]       # Per-turn sentiment + escalation risk
    compliance: Optional[dict]      # Compliance violations + score
    summary: Optional[SummaryOutput]
    qa_score: Optional[QAScore]
    errors: list[str]
    current_step: str


def create_workflow(
    llm_name: Literal["claude", "gpt4", "gemini"] = "claude"
) -> CompiledGraph:
    """
    Create and compile the call processing workflow.

    Pipeline (Version 2):
      intake → transcription → pii_redaction → rag_retrieval
             → sentiment → summarization → quality_score → end

    Args:
        llm_name: LLM to use for LLM-powered nodes

    Returns:
        CompiledGraph: Compiled LangGraph workflow
    """
    logger.info(f"Creating V2 workflow with LLM: {llm_name}")

    # Initialize agents
    intake_agent = IntakeAgent()
    transcription_agent = TranscriptionAgent()
    pii_agent = PIIRedactionAgent()
    rag_agent = RAGRetrievalAgent(top_k=3)
    sentiment_agent = SentimentAgent(llm_name=llm_name)
    compliance_agent = ComplianceCheckerAgent(llm_name=llm_name)
    summarization_agent = SummarizationAgent(llm_name=llm_name)
    quality_score_agent = QualityScoreAgent(llm_name=llm_name)
    routing_agent = RoutingAgent()

    workflow = StateGraph(WorkflowState)

    # ── Node definitions ──────────────────────────────────────────────────────

    def intake_node(state: WorkflowState) -> WorkflowState:
        logger.info(f"[NODE] intake: Processing call {state['call_id']}")
        state["current_step"] = "intake"
        try:
            if not state["input_data"]:
                raise ValueError("No input data provided")
            logger.info("[NODE] intake: Input validated")
        except Exception as e:
            logger.error(f"[NODE] intake: Error - {e}")
            state["errors"].append(f"Intake error: {str(e)}")
        return state

    def transcription_node(state: WorkflowState) -> WorkflowState:
        logger.info(f"[NODE] transcription: Processing call {state['call_id']}")
        state["current_step"] = "transcription"
        try:
            result = transcription_agent.process(
                call_id=state["call_id"],
                audio_path=state["input_data"].audio_path,
                transcript_text=state["input_data"].transcript_text,
            )
            state["transcript"] = result
            logger.info("[NODE] transcription: Transcript normalized")
        except Exception as e:
            logger.error(f"[NODE] transcription: Error - {e}")
            state["errors"].append(f"Transcription error: {str(e)}")
            if state["input_data"].transcript_text:
                state["transcript"] = TranscriptOutput(
                    call_id=state["call_id"],
                    transcript=state["input_data"].transcript_text,
                    speakers=["Agent", "Customer"],
                )
        return state

    def pii_redaction_node(state: WorkflowState) -> WorkflowState:
        """Mask PII before any LLM sees the transcript."""
        logger.info(f"[NODE] pii_redaction: Processing call {state['call_id']}")
        state["current_step"] = "pii_redaction"
        try:
            raw = ""
            if state["transcript"]:
                raw = state["transcript"].transcript
            elif state["input_data"].transcript_text:
                raw = state["input_data"].transcript_text

            if raw:
                result = pii_agent.redact(raw)
                state["redacted_transcript"] = result.redacted_transcript
                state["pii_summary"] = result.redaction_summary
                if result.total_redactions > 0:
                    logger.info(
                        f"[NODE] pii_redaction: {result.total_redactions} PII item(s) masked: "
                        f"{result.redaction_summary}"
                    )
            else:
                state["redacted_transcript"] = ""
                state["pii_summary"] = {}
        except Exception as e:
            logger.warning(f"[NODE] pii_redaction: {e} — using raw transcript")
            raw = (state["transcript"].transcript if state["transcript"]
                   else state["input_data"].transcript_text or "")
            state["redacted_transcript"] = raw
            state["pii_summary"] = {}
        return state

    def rag_retrieval_node(state: WorkflowState) -> WorkflowState:
        """Retrieve semantically similar past calls for RAG context."""
        logger.info(f"[NODE] rag_retrieval: Processing call {state['call_id']}")
        state["current_step"] = "rag_retrieval"
        try:
            text = state.get("redacted_transcript") or ""
            if text:
                context = rag_agent.retrieve_context(text, state["call_id"])
                state["rag_context"] = context
                logger.info(
                    f"[NODE] rag_retrieval: {'Context retrieved' if context else 'No similar calls found'}"
                )
        except Exception as e:
            logger.warning(f"[NODE] rag_retrieval: {e} — continuing without RAG")
            state["rag_context"] = ""
        return state

    def sentiment_node(state: WorkflowState) -> WorkflowState:
        """Analyze per-turn sentiment and escalation risk."""
        logger.info(f"[NODE] sentiment: Processing call {state['call_id']}")
        state["current_step"] = "sentiment"
        try:
            text = state.get("redacted_transcript") or ""
            if text:
                state["sentiment"] = sentiment_agent.process(state["call_id"], text)
                risk = state["sentiment"].get("escalation_risk", "unknown")
                logger.info(f"[NODE] sentiment: escalation_risk={risk}")
        except Exception as e:
            logger.warning(f"[NODE] sentiment: {e} — continuing without sentiment data")
            state["sentiment"] = None
        return state

    def compliance_check_node(state: WorkflowState) -> WorkflowState:
        """Check transcript for regulatory compliance violations."""
        logger.info(f"[NODE] compliance_check: Processing call {state['call_id']}")
        state["current_step"] = "compliance_check"
        try:
            text = state.get("redacted_transcript") or ""
            if text:
                result = compliance_agent.process(state["call_id"], text)
                state["compliance"] = result
                n = len(result.get("violations", []))
                score = result.get("compliance_score", 100)
                logger.info(f"[NODE] compliance_check: {n} violation(s), score={score:.0f}/100")
        except Exception as e:
            logger.warning(f"[NODE] compliance_check: {e} — continuing without compliance data")
            state["compliance"] = None
        return state

    def summarization_node(state: WorkflowState) -> WorkflowState:
        """Generate summary using PII-safe redacted transcript + RAG context."""
        logger.info(f"[NODE] summarization: Processing call {state['call_id']}")
        state["current_step"] = "summarization"
        try:
            text = state.get("redacted_transcript") or ""
            if not text:
                raise ValueError("No transcript available for summarization")
            result = summarization_agent.process(
                call_id=state["call_id"],
                transcript=text,
                rag_context=state.get("rag_context", ""),
            )
            state["summary"] = result
            logger.info(f"[NODE] summarization: {result.resolution_status.value}")
        except Exception as e:
            logger.error(f"[NODE] summarization: Error - {e}")
            state["errors"].append(f"Summarization error: {str(e)}")
        return state

    def quality_score_node(state: WorkflowState) -> WorkflowState:
        """Score call quality using PII-safe transcript + RAG context."""
        logger.info(f"[NODE] quality_score: Processing call {state['call_id']}")
        state["current_step"] = "quality_score"
        try:
            text = state.get("redacted_transcript") or ""
            if not text:
                raise ValueError("No transcript available for quality scoring")
            result = quality_score_agent.process(
                call_id=state["call_id"],
                transcript=text,
                rag_context=state.get("rag_context", ""),
            )
            state["qa_score"] = result
            logger.info(f"[NODE] quality_score: {result.overall_score:.1f}/100")
        except Exception as e:
            logger.error(f"[NODE] quality_score: Error - {e}")
            state["errors"].append(f"Quality score error: {str(e)}")
        return state

    def error_handler_node(state: WorkflowState) -> WorkflowState:
        logger.warning(f"[NODE] error_handler: Errors — {state['errors']}")
        state["current_step"] = "error_handler"
        return state

    def end_node(state: WorkflowState) -> WorkflowState:
        logger.info(f"[NODE] end: Finalizing call {state['call_id']}")
        state["current_step"] = "completed"
        return state

    # ── Build graph ───────────────────────────────────────────────────────────

    workflow.add_node("intake", intake_node)
    workflow.add_node("transcription", transcription_node)
    workflow.add_node("pii_redaction", pii_redaction_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("compliance_check", compliance_check_node)
    workflow.add_node("summarization", summarization_node)
    workflow.add_node("quality_score", quality_score_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("end", end_node)

    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "transcription")

    # Transcription: errors → error_handler, else → pii_redaction
    workflow.add_conditional_edges(
        "transcription",
        lambda s: "error_handler" if s["errors"] else "pii_redaction",
        {"error_handler": "error_handler", "pii_redaction": "pii_redaction"},
    )

    workflow.add_edge("pii_redaction", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "sentiment")
    workflow.add_edge("sentiment", "compliance_check")
    workflow.add_edge("compliance_check", "summarization")
    workflow.add_edge("summarization", "quality_score")
    workflow.add_edge("quality_score", "end")

    # Error handler recovery (skips PII/RAG/Sentiment on error path)
    workflow.add_conditional_edges(
        "error_handler",
        lambda s: "summarization" if not s["summary"] else "quality_score" if not s["qa_score"] else "end",
        {"summarization": "summarization", "quality_score": "quality_score", "end": "end"},
    )

    workflow.add_edge("end", END)

    logger.info("V2 Workflow graph compiled successfully")
    return workflow.compile()


def run_workflow(
    compiled_graph: CompiledGraph,
    call_input: CallInput,
    llm_name: Literal["claude", "gpt4", "gemini"] = "claude",
) -> CallResult:
    """
    Execute the V2 workflow on a single call.

    Returns:
        CallResult: Complete call analysis result
    """
    logger.info(f"[WORKFLOW] Starting V2 execution for call {call_input.call_id}")

    initial_state: WorkflowState = {
        "call_id": call_input.call_id,
        "input_data": call_input,
        "transcript": None,
        "redacted_transcript": "",
        "pii_summary": {},
        "rag_context": "",
        "sentiment": None,
        "compliance": None,
        "summary": None,
        "qa_score": None,
        "errors": [],
        "current_step": "initialized",
    }

    try:
        final_state = compiled_graph.invoke(initial_state)

        result = CallResult(
            call_id=final_state["call_id"],
            input_data=final_state["input_data"],
            transcript=final_state["transcript"],
            summary=final_state["summary"],
            qa_score=final_state["qa_score"],
            errors=final_state["errors"],
            current_step=final_state["current_step"],
        )

        logger.info(f"[WORKFLOW] V2 completed for call {call_input.call_id}")

        # Persist to memory layer
        try:
            call_memory.add_call(result.model_dump(), llm_name=llm_name)
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to save call history: {e}")

        # Store embedding in vector DB for future RAG retrieval
        try:
            if result.summary and result.transcript:
                store_call_embedding(
                    call_id=result.call_id,
                    transcript=final_state.get("redacted_transcript") or result.transcript.transcript,
                    summary=result.summary.summary,
                    metadata={
                        "category": result.input_data.metadata.get("category", "unknown"),
                        "resolution_status": result.summary.resolution_status.value,
                        "overall_score": result.qa_score.overall_score if result.qa_score else 0.0,
                        "llm_name": llm_name,
                    },
                )
        except Exception as e:
            logger.warning(f"[VECTORDB] Failed to store embedding: {e}")

        # Attach extra V2 data to result for UI consumption via session state
        result._v2_extras = {
            "redacted_transcript": final_state.get("redacted_transcript", ""),
            "pii_summary": final_state.get("pii_summary", {}),
            "rag_context": final_state.get("rag_context", ""),
            "sentiment": final_state.get("sentiment"),
            "compliance": final_state.get("compliance"),
        }

        return result

    except Exception as e:
        logger.error(f"[WORKFLOW] Failed for call {call_input.call_id}: {e}")
        raise ValueError(f"Workflow execution failed: {str(e)}")
