"""
LangGraph Workflow: Orchestrates the multi-agent call processing pipeline.
Version 3 — adds CallCoaching, KnowledgeBase, CustomerProfile, AutoTagging,
             AnomalyDetection, and FeedbackLoop agents (17 nodes total).
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
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.customer_profile_agent import CustomerProfileAgent
from agents.sentiment_agent import SentimentAgent
from agents.compliance_checker_agent import ComplianceCheckerAgent
from agents.escalation_prediction_agent import EscalationPredictionAgent
from agents.summarization_agent import SummarizationAgent
from agents.auto_tagging_agent import AutoTaggingAgent
from agents.quality_score_agent import QualityScoreAgent
from agents.call_coaching_agent import CallCoachingAgent
from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.feedback_loop_agent import FeedbackLoopAgent
from agents.routing_agent import RoutingAgent
from utils.schemas import CallResult, CallInput, TranscriptOutput, SummaryOutput, QAScore
from utils.memory import call_memory
from utils.vector_store import store_call_embedding

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State dictionary for the LangGraph workflow (Version 3)."""
    call_id: str
    input_data: CallInput
    transcript: Optional[TranscriptOutput]
    redacted_transcript: str        # PII-masked version of the transcript
    pii_summary: dict               # {field: count} of redacted PII items
    customer_profile: Optional[dict]  # Cross-call customer journey
    rag_context: str                # Formatted context from similar past calls
    kb_context: str                 # Formatted context from KB articles
    kb_analysis: Optional[dict]     # Full KB compliance analysis
    sentiment: Optional[dict]       # Per-turn sentiment + escalation risk
    compliance: Optional[dict]      # Compliance violations + score
    escalation: Optional[dict]      # Escalation risk score + triggers
    summary: Optional[SummaryOutput]
    tags: Optional[dict]            # Auto-assigned call tags
    qa_score: Optional[QAScore]
    coaching: Optional[dict]        # Agent coaching tips
    anomaly: Optional[dict]         # Anomaly detection result
    feedback_loop: Optional[dict]   # Coaching feedback loop result
    errors: list[str]
    current_step: str


def create_workflow(
    llm_name: Literal["claude", "gpt4", "gemini"] = "claude"
) -> CompiledGraph:
    """
    Create and compile the call processing workflow.

    Pipeline (Version 3 — 17 nodes):
      intake → customer_profile → transcription → pii_redaction
             → rag_retrieval → kb_retrieval → sentiment
             → compliance_check → escalation_prediction → summarization
             → auto_tagging → quality_score → call_coaching
             → anomaly_detection → end
      (+ error_handler branch)
      FeedbackLoopAgent runs post-workflow in run_workflow()

    Args:
        llm_name: LLM to use for LLM-powered nodes

    Returns:
        CompiledGraph: Compiled LangGraph workflow
    """
    logger.info(f"Creating V3 workflow with LLM: {llm_name}")

    # Initialize agents
    intake_agent = IntakeAgent()
    transcription_agent = TranscriptionAgent()
    pii_agent = PIIRedactionAgent()
    customer_profile_agent = CustomerProfileAgent()
    rag_agent = RAGRetrievalAgent(top_k=3)
    kb_agent = KnowledgeBaseAgent(llm_name=llm_name)
    sentiment_agent = SentimentAgent(llm_name=llm_name)
    compliance_agent = ComplianceCheckerAgent(llm_name=llm_name)
    escalation_agent = EscalationPredictionAgent(llm_name=llm_name)
    summarization_agent = SummarizationAgent(llm_name=llm_name)
    auto_tagging_agent = AutoTaggingAgent(llm_name=llm_name)
    quality_score_agent = QualityScoreAgent(llm_name=llm_name)
    coaching_agent = CallCoachingAgent(llm_name=llm_name)
    anomaly_agent = AnomalyDetectionAgent()
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

    def customer_profile_node(state: WorkflowState) -> WorkflowState:
        """Build cross-call customer journey from history."""
        logger.info(f"[NODE] customer_profile: Processing call {state['call_id']}")
        state["current_step"] = "customer_profile"
        try:
            category = state["input_data"].metadata.get("category") if state["input_data"].metadata else None
            transcript_text = state["input_data"].transcript_text or ""
            result = customer_profile_agent.process(
                call_id=state["call_id"],
                transcript=transcript_text,
                category=category,
            )
            state["customer_profile"] = result
            risk = result.get("risk_tier", "unknown")
            total = result.get("total_calls_in_history", 0)
            logger.info(f"[NODE] customer_profile: {total} historical calls, risk_tier={risk}")
        except Exception as e:
            logger.warning(f"[NODE] customer_profile: {e} — continuing without profile")
            state["customer_profile"] = None
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

    def kb_retrieval_node(state: WorkflowState) -> WorkflowState:
        """Retrieve relevant SOPs and product knowledge from the knowledge base."""
        logger.info(f"[NODE] kb_retrieval: Processing call {state['call_id']}")
        state["current_step"] = "kb_retrieval"
        try:
            text = state.get("redacted_transcript") or ""
            if text:
                category = None
                if state["input_data"].metadata:
                    category = state["input_data"].metadata.get("category")
                # Quick context for LLM injection
                kb_context = kb_agent.retrieve_context(text, state["call_id"])
                state["kb_context"] = kb_context
                # Full analysis (SOP compliance check) — runs in mock or production
                kb_analysis = kb_agent.process(state["call_id"], text, category=category)
                state["kb_analysis"] = kb_analysis
                sop_score = kb_analysis.get("sop_compliance_score", 100.0)
                logger.info(f"[NODE] kb_retrieval: SOP compliance={sop_score:.0f}%")
        except Exception as e:
            logger.warning(f"[NODE] kb_retrieval: {e} — continuing without KB")
            state["kb_context"] = ""
            state["kb_analysis"] = None
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

    def escalation_prediction_node(state: WorkflowState) -> WorkflowState:
        """Predict escalation risk using transcript + sentiment + compliance signals."""
        logger.info(f"[NODE] escalation_prediction: Processing call {state['call_id']}")
        state["current_step"] = "escalation_prediction"
        try:
            text = state.get("redacted_transcript") or ""
            if text:
                result = escalation_agent.process(
                    call_id=state["call_id"],
                    transcript=text,
                    sentiment=state.get("sentiment"),
                    compliance=state.get("compliance"),
                )
                state["escalation"] = result
                logger.info(
                    f"[NODE] escalation_prediction: risk={result.get('risk_score', 0):.0f}/100 "
                    f"({result.get('risk_level', 'unknown')})"
                )
        except Exception as e:
            logger.warning(f"[NODE] escalation_prediction: {e} — continuing")
            state["escalation"] = None
        return state

    def summarization_node(state: WorkflowState) -> WorkflowState:
        """Generate summary using PII-safe redacted transcript + RAG + KB context."""
        logger.info(f"[NODE] summarization: Processing call {state['call_id']}")
        state["current_step"] = "summarization"
        try:
            text = state.get("redacted_transcript") or ""
            if not text:
                raise ValueError("No transcript available for summarization")
            # Combine RAG context + KB context for richer summarization
            combined_context = "\n\n".join(
                filter(None, [state.get("rag_context", ""), state.get("kb_context", "")])
            )
            result = summarization_agent.process(
                call_id=state["call_id"],
                transcript=text,
                rag_context=combined_context,
            )
            state["summary"] = result
            logger.info(f"[NODE] summarization: {result.resolution_status.value}")
        except Exception as e:
            logger.error(f"[NODE] summarization: Error - {e}")
            state["errors"].append(f"Summarization error: {str(e)}")
        return state

    def auto_tagging_node(state: WorkflowState) -> WorkflowState:
        """Assign multi-label tags for routing and analytics."""
        logger.info(f"[NODE] auto_tagging: Processing call {state['call_id']}")
        state["current_step"] = "auto_tagging"
        try:
            text = state.get("redacted_transcript") or ""
            if text:
                summary_dict = state["summary"].model_dump() if state.get("summary") else None
                result = auto_tagging_agent.process(
                    call_id=state["call_id"],
                    transcript=text,
                    summary=summary_dict,
                )
                state["tags"] = result
                cat = result.get("primary_category", "unknown")
                conf = result.get("confidence_score", 0.0)
                logger.info(f"[NODE] auto_tagging: category={cat} (conf={conf:.2f})")
        except Exception as e:
            logger.warning(f"[NODE] auto_tagging: {e} — continuing without tags")
            state["tags"] = None
        return state

    def quality_score_node(state: WorkflowState) -> WorkflowState:
        """Score call quality using PII-safe transcript + RAG + KB context."""
        logger.info(f"[NODE] quality_score: Processing call {state['call_id']}")
        state["current_step"] = "quality_score"
        try:
            text = state.get("redacted_transcript") or ""
            if not text:
                raise ValueError("No transcript available for quality scoring")
            combined_context = "\n\n".join(
                filter(None, [state.get("rag_context", ""), state.get("kb_context", "")])
            )
            result = quality_score_agent.process(
                call_id=state["call_id"],
                transcript=text,
                rag_context=combined_context,
            )
            state["qa_score"] = result
            logger.info(f"[NODE] quality_score: {result.overall_score:.1f}/100")
        except Exception as e:
            logger.error(f"[NODE] quality_score: Error - {e}")
            state["errors"].append(f"Quality score error: {str(e)}")
        return state

    def call_coaching_node(state: WorkflowState) -> WorkflowState:
        """Generate personalised agent coaching based on QA scores."""
        logger.info(f"[NODE] call_coaching: Processing call {state['call_id']}")
        state["current_step"] = "call_coaching"
        try:
            text = state.get("redacted_transcript") or ""
            if text:
                qa_dict = state["qa_score"].model_dump() if state.get("qa_score") else None
                result = coaching_agent.process(
                    call_id=state["call_id"],
                    transcript=text,
                    qa_score=qa_dict,
                    sentiment=state.get("sentiment"),
                )
                state["coaching"] = result
                priority = result.get("overall_coaching_priority", "unknown")
                n_tips = len(result.get("coaching_tips", []))
                logger.info(f"[NODE] call_coaching: {n_tips} tip(s), priority={priority}")
        except Exception as e:
            logger.warning(f"[NODE] call_coaching: {e} — continuing without coaching")
            state["coaching"] = None
        return state

    def anomaly_detection_node(state: WorkflowState) -> WorkflowState:
        """Flag outlier calls for QA review."""
        logger.info(f"[NODE] anomaly_detection: Processing call {state['call_id']}")
        state["current_step"] = "anomaly_detection"
        try:
            qa_dict = state["qa_score"].model_dump() if state.get("qa_score") else None
            result = anomaly_agent.process(
                call_id=state["call_id"],
                qa_score=qa_dict,
                compliance=state.get("compliance"),
                escalation=state.get("escalation"),
                sentiment=state.get("sentiment"),
                pii_summary=state.get("pii_summary"),
            )
            state["anomaly"] = result
            level = result.get("anomaly_level", "normal")
            score = result.get("anomaly_score", 0.0)
            logger.info(f"[NODE] anomaly_detection: level={level}, score={score:.0f}")
        except Exception as e:
            logger.warning(f"[NODE] anomaly_detection: {e} — continuing")
            state["anomaly"] = None
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
    workflow.add_node("customer_profile", customer_profile_node)
    workflow.add_node("transcription", transcription_node)
    workflow.add_node("pii_redaction", pii_redaction_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("kb_retrieval", kb_retrieval_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("compliance_check", compliance_check_node)
    workflow.add_node("escalation_prediction", escalation_prediction_node)
    workflow.add_node("summarization", summarization_node)
    workflow.add_node("auto_tagging", auto_tagging_node)
    workflow.add_node("quality_score", quality_score_node)
    workflow.add_node("call_coaching", call_coaching_node)
    workflow.add_node("anomaly_detection", anomaly_detection_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("end", end_node)

    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "customer_profile")
    workflow.add_edge("customer_profile", "transcription")

    # Transcription: errors → error_handler, else → pii_redaction
    workflow.add_conditional_edges(
        "transcription",
        lambda s: "error_handler" if s["errors"] else "pii_redaction",
        {"error_handler": "error_handler", "pii_redaction": "pii_redaction"},
    )

    workflow.add_edge("pii_redaction", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "kb_retrieval")
    workflow.add_edge("kb_retrieval", "sentiment")
    workflow.add_edge("sentiment", "compliance_check")
    workflow.add_edge("compliance_check", "escalation_prediction")
    workflow.add_edge("escalation_prediction", "summarization")
    workflow.add_edge("summarization", "auto_tagging")
    workflow.add_edge("auto_tagging", "quality_score")
    workflow.add_edge("quality_score", "call_coaching")
    workflow.add_edge("call_coaching", "anomaly_detection")
    workflow.add_edge("anomaly_detection", "end")

    # Error handler recovery (skips enrichment agents on error path)
    workflow.add_conditional_edges(
        "error_handler",
        lambda s: "summarization" if not s["summary"] else "quality_score" if not s["qa_score"] else "end",
        {"summarization": "summarization", "quality_score": "quality_score", "end": "end"},
    )

    workflow.add_edge("end", END)

    logger.info("V3 Workflow graph compiled successfully")
    return workflow.compile()


def run_workflow(
    compiled_graph: CompiledGraph,
    call_input: CallInput,
    llm_name: Literal["claude", "gpt4", "gemini"] = "claude",
) -> CallResult:
    """
    Execute the V3 workflow on a single call.

    Returns:
        CallResult: Complete call analysis result
    """
    logger.info(f"[WORKFLOW] Starting V3 execution for call {call_input.call_id}")

    initial_state: WorkflowState = {
        "call_id": call_input.call_id,
        "input_data": call_input,
        "transcript": None,
        "redacted_transcript": "",
        "pii_summary": {},
        "customer_profile": None,
        "rag_context": "",
        "kb_context": "",
        "kb_analysis": None,
        "sentiment": None,
        "compliance": None,
        "escalation": None,
        "summary": None,
        "tags": None,
        "qa_score": None,
        "coaching": None,
        "anomaly": None,
        "feedback_loop": None,
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

        logger.info(f"[WORKFLOW] V3 completed for call {call_input.call_id}")

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

        # Run FeedbackLoopAgent post-workflow (requires assembled result)
        feedback_loop_result = None
        try:
            from agents.feedback_loop_agent import FeedbackLoopAgent
            fl_agent = FeedbackLoopAgent()
            qa_dict = result.qa_score.model_dump() if result.qa_score else None
            feedback_loop_result = fl_agent.process(
                call_id=result.call_id,
                current_qa=qa_dict,
                current_coaching=final_state.get("coaching"),
                current_escalation=final_state.get("escalation"),
            )
            logger.info(
                f"[FEEDBACK] status={feedback_loop_result.get('improvement_status')}, "
                f"delta={feedback_loop_result.get('score_delta', 0):+.1f}"
            )
        except Exception as e:
            logger.warning(f"[FEEDBACK] FeedbackLoopAgent failed: {e}")

        # Attach all V3 extras to result for UI consumption via session state
        result._v2_extras = {
            "redacted_transcript": final_state.get("redacted_transcript", ""),
            "pii_summary": final_state.get("pii_summary", {}),
            "customer_profile": final_state.get("customer_profile"),
            "rag_context": final_state.get("rag_context", ""),
            "kb_context": final_state.get("kb_context", ""),
            "kb_analysis": final_state.get("kb_analysis"),
            "sentiment": final_state.get("sentiment"),
            "compliance": final_state.get("compliance"),
            "escalation": final_state.get("escalation"),
            "tags": final_state.get("tags"),
            "coaching": final_state.get("coaching"),
            "anomaly": final_state.get("anomaly"),
            "feedback_loop": feedback_loop_result,
        }

        return result

    except Exception as e:
        logger.error(f"[WORKFLOW] Failed for call {call_input.call_id}: {e}")
        raise ValueError(f"Workflow execution failed: {str(e)}")
