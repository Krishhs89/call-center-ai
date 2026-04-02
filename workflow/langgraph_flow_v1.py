"""
LangGraph Workflow: V1 Baseline Pipeline (5 nodes).
intake → transcription → summarization → quality_score → end
(+ error_handler branch on transcription failure)

No PII redaction, RAG, sentiment, compliance, or coaching.
Used for side-by-side comparison with V3 in the UI.
"""

import logging
from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from agents.intake_agent import IntakeAgent
from agents.transcription_agent import TranscriptionAgent
from agents.summarization_agent import SummarizationAgent
from agents.quality_score_agent import QualityScoreAgent
from agents.routing_agent import RoutingAgent
from utils.schemas import CallResult, CallInput, TranscriptOutput, SummaryOutput, QAScore
from utils.memory import call_memory

logger = logging.getLogger(__name__)


class WorkflowStateV1(TypedDict):
    """State dictionary for the V1 baseline workflow."""
    call_id: str
    input_data: CallInput
    transcript: Optional[TranscriptOutput]
    summary: Optional[SummaryOutput]
    qa_score: Optional[QAScore]
    errors: list
    current_step: str


def create_workflow_v1(
    llm_name: Literal["claude", "gpt4", "gemini"] = "claude"
) -> CompiledGraph:
    """
    Create and compile the V1 baseline workflow.

    Pipeline (5 nodes):
      intake → transcription → summarization → quality_score → end
      (+ error_handler on transcription failure)

    Args:
        llm_name: LLM to use for summarization and QA scoring

    Returns:
        CompiledGraph: Compiled LangGraph workflow
    """
    logger.info(f"Creating V1 workflow with LLM: {llm_name}")

    intake_agent = IntakeAgent()
    transcription_agent = TranscriptionAgent()
    summarization_agent = SummarizationAgent(llm_name=llm_name)
    quality_score_agent = QualityScoreAgent(llm_name=llm_name)
    routing_agent = RoutingAgent()

    workflow = StateGraph(WorkflowStateV1)

    # ── Node definitions ──────────────────────────────────────────────────────

    def intake_node(state: WorkflowStateV1) -> WorkflowStateV1:
        logger.info(f"[V1][NODE] intake: Processing call {state['call_id']}")
        state["current_step"] = "intake"
        try:
            if not state["input_data"]:
                raise ValueError("No input data provided")
        except Exception as e:
            state["errors"].append(f"Intake error: {str(e)}")
        return state

    def transcription_node(state: WorkflowStateV1) -> WorkflowStateV1:
        logger.info(f"[V1][NODE] transcription: Processing call {state['call_id']}")
        state["current_step"] = "transcription"
        try:
            result = transcription_agent.process(
                call_id=state["call_id"],
                audio_path=state["input_data"].audio_path,
                transcript_text=state["input_data"].transcript_text,
            )
            state["transcript"] = result
        except Exception as e:
            logger.error(f"[V1][NODE] transcription: Error - {e}")
            state["errors"].append(f"Transcription error: {str(e)}")
            if state["input_data"].transcript_text:
                state["transcript"] = TranscriptOutput(
                    call_id=state["call_id"],
                    transcript=state["input_data"].transcript_text,
                    speakers=["Agent", "Customer"],
                )
        return state

    def summarization_node(state: WorkflowStateV1) -> WorkflowStateV1:
        logger.info(f"[V1][NODE] summarization: Processing call {state['call_id']}")
        state["current_step"] = "summarization"
        try:
            text = state["transcript"].transcript if state["transcript"] else ""
            if not text:
                raise ValueError("No transcript available for summarization")
            result = summarization_agent.process(
                call_id=state["call_id"],
                transcript=text,
                rag_context="",
            )
            state["summary"] = result
            logger.info(f"[V1][NODE] summarization: {result.resolution_status.value}")
        except Exception as e:
            logger.error(f"[V1][NODE] summarization: Error - {e}")
            state["errors"].append(f"Summarization error: {str(e)}")
        return state

    def quality_score_node(state: WorkflowStateV1) -> WorkflowStateV1:
        logger.info(f"[V1][NODE] quality_score: Processing call {state['call_id']}")
        state["current_step"] = "quality_score"
        try:
            text = state["transcript"].transcript if state["transcript"] else ""
            if not text:
                raise ValueError("No transcript available for quality scoring")
            result = quality_score_agent.process(
                call_id=state["call_id"],
                transcript=text,
                rag_context="",
            )
            state["qa_score"] = result
            logger.info(f"[V1][NODE] quality_score: {result.overall_score:.1f}/100")
        except Exception as e:
            logger.error(f"[V1][NODE] quality_score: Error - {e}")
            state["errors"].append(f"Quality score error: {str(e)}")
        return state

    def error_handler_node(state: WorkflowStateV1) -> WorkflowStateV1:
        logger.warning(f"[V1][NODE] error_handler: Errors — {state['errors']}")
        state["current_step"] = "error_handler"
        return state

    def end_node(state: WorkflowStateV1) -> WorkflowStateV1:
        logger.info(f"[V1][NODE] end: Finalizing call {state['call_id']}")
        state["current_step"] = "completed"
        return state

    # ── Build graph ───────────────────────────────────────────────────────────

    workflow.add_node("intake", intake_node)
    workflow.add_node("transcription", transcription_node)
    workflow.add_node("summarization", summarization_node)
    workflow.add_node("quality_score", quality_score_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("end", end_node)

    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "transcription")

    workflow.add_conditional_edges(
        "transcription",
        lambda s: "error_handler" if s["errors"] else "summarization",
        {"error_handler": "error_handler", "summarization": "summarization"},
    )

    workflow.add_edge("summarization", "quality_score")
    workflow.add_edge("quality_score", "end")

    workflow.add_conditional_edges(
        "error_handler",
        lambda s: "summarization" if not s["summary"] else "quality_score" if not s["qa_score"] else "end",
        {"summarization": "summarization", "quality_score": "quality_score", "end": "end"},
    )

    workflow.add_edge("end", END)

    logger.info("V1 Workflow graph compiled successfully")
    return workflow.compile()


def run_workflow_v1(
    compiled_graph: CompiledGraph,
    call_input: CallInput,
    llm_name: Literal["claude", "gpt4", "gemini"] = "claude",
) -> CallResult:
    """
    Execute the V1 workflow on a single call.

    Returns:
        CallResult: Complete call analysis result
    """
    logger.info(f"[V1 WORKFLOW] Starting execution for call {call_input.call_id}")

    initial_state: WorkflowStateV1 = {
        "call_id": call_input.call_id,
        "input_data": call_input,
        "transcript": None,
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

        logger.info(f"[V1 WORKFLOW] Completed for call {call_input.call_id}")

        try:
            call_memory.add_call(result.model_dump(), llm_name=llm_name)
        except Exception as e:
            logger.warning(f"[V1 MEMORY] Failed to save call history: {e}")

        # V1 has no extras — empty dict keeps UI compatible
        result._v2_extras = {}

        return result

    except Exception as e:
        logger.error(f"[V1 WORKFLOW] Failed for call {call_input.call_id}: {e}")
        raise ValueError(f"V1 Workflow execution failed: {str(e)}")
