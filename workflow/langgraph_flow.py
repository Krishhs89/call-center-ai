"""
LangGraph Workflow: Orchestrates the multi-agent call processing pipeline.
Implements conditional routing and error handling.
Compatible with LangSmith tracing.
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


class WorkflowState(TypedDict):
    """State dictionary for the LangGraph workflow."""
    call_id: str
    input_data: CallInput
    transcript: Optional[TranscriptOutput]
    summary: Optional[SummaryOutput]
    qa_score: Optional[QAScore]
    errors: list[str]
    current_step: str


def create_workflow(
    llm_name: Literal["claude", "gpt4", "gemini"] = "claude"
) -> CompiledGraph:
    """
    Create and compile the call processing workflow.

    Args:
        llm_name: LLM to use for summarization and QA scoring

    Returns:
        CompiledGraph: Compiled LangGraph workflow

    Raises:
        ValueError: If LLM is not available
    """
    logger.info(f"Creating workflow with LLM: {llm_name}")

    # Initialize agents
    intake_agent = IntakeAgent()
    transcription_agent = TranscriptionAgent()
    summarization_agent = SummarizationAgent(llm_name=llm_name)
    quality_score_agent = QualityScoreAgent(llm_name=llm_name)
    routing_agent = RoutingAgent()

    # Create state graph
    workflow = StateGraph(WorkflowState)

    # Define nodes
    def intake_node(state: WorkflowState) -> WorkflowState:
        """Process intake and validate input."""
        logger.info(f"[NODE] intake: Processing call {state['call_id']}")
        state["current_step"] = "intake"

        try:
            # Input data should already be validated, just confirm it
            if not state["input_data"]:
                raise ValueError("No input data provided")
            logger.info(f"[NODE] intake: Input validated")
        except Exception as e:
            logger.error(f"[NODE] intake: Error - {e}")
            state["errors"].append(f"Intake error: {str(e)}")

        return state

    def transcription_node(state: WorkflowState) -> WorkflowState:
        """Normalize and process transcript."""
        logger.info(f"[NODE] transcription: Processing call {state['call_id']}")
        state["current_step"] = "transcription"

        try:
            result = transcription_agent.process(
                call_id=state["call_id"],
                audio_path=state["input_data"].audio_path,
                transcript_text=state["input_data"].transcript_text,
            )
            state["transcript"] = result
            logger.info(f"[NODE] transcription: Transcript normalized")
        except Exception as e:
            logger.error(f"[NODE] transcription: Error - {e}")
            state["errors"].append(f"Transcription error: {str(e)}")
            # Create fallback transcript from input text
            if state["input_data"].transcript_text:
                state["transcript"] = TranscriptOutput(
                    call_id=state["call_id"],
                    transcript=state["input_data"].transcript_text,
                    speakers=["Agent", "Customer"],
                )

        return state

    def summarization_node(state: WorkflowState) -> WorkflowState:
        """Generate summary of the call."""
        logger.info(f"[NODE] summarization: Processing call {state['call_id']}")
        state["current_step"] = "summarization"

        try:
            # Use transcript if available, otherwise use input text
            transcript_text = ""
            if state["transcript"]:
                transcript_text = state["transcript"].transcript
            elif state["input_data"].transcript_text:
                transcript_text = state["input_data"].transcript_text

            if not transcript_text:
                raise ValueError("No transcript available for summarization")

            result = summarization_agent.process(
                call_id=state["call_id"],
                transcript=transcript_text,
            )
            state["summary"] = result
            logger.info(f"[NODE] summarization: Summary generated - {result.resolution_status.value}")
        except Exception as e:
            logger.error(f"[NODE] summarization: Error - {e}")
            state["errors"].append(f"Summarization error: {str(e)}")

        return state

    def quality_score_node(state: WorkflowState) -> WorkflowState:
        """Score the call quality."""
        logger.info(f"[NODE] quality_score: Processing call {state['call_id']}")
        state["current_step"] = "quality_score"

        try:
            # Use transcript if available, otherwise use input text
            transcript_text = ""
            if state["transcript"]:
                transcript_text = state["transcript"].transcript
            elif state["input_data"].transcript_text:
                transcript_text = state["input_data"].transcript_text

            if not transcript_text:
                raise ValueError("No transcript available for quality scoring")

            result = quality_score_agent.process(
                call_id=state["call_id"],
                transcript=transcript_text,
            )
            state["qa_score"] = result
            logger.info(f"[NODE] quality_score: QA score {result.overall_score:.1f}/100")
        except Exception as e:
            logger.error(f"[NODE] quality_score: Error - {e}")
            state["errors"].append(f"Quality score error: {str(e)}")

        return state

    def error_handler_node(state: WorkflowState) -> WorkflowState:
        """Handle errors and log them."""
        logger.warning(f"[NODE] error_handler: Processing call {state['call_id']}")
        logger.warning(f"[NODE] error_handler: Errors - {state['errors']}")
        state["current_step"] = "error_handler"
        return state

    def end_node(state: WorkflowState) -> WorkflowState:
        """Finalize the workflow."""
        logger.info(f"[NODE] end: Finalizing call {state['call_id']}")
        state["current_step"] = "completed"
        return state

    # Add nodes to workflow
    workflow.add_node("intake", intake_node)
    workflow.add_node("transcription", transcription_node)
    workflow.add_node("summarization", summarization_node)
    workflow.add_node("quality_score", quality_score_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("end", end_node)

    # Define edges
    workflow.set_entry_point("intake")

    # Intake always goes to transcription
    workflow.add_edge("intake", "transcription")

    # Transcription error handling
    workflow.add_conditional_edges(
        "transcription",
        lambda state: "error_handler" if state["errors"] else "summarization",
        {"error_handler": "error_handler", "summarization": "summarization"},
    )

    # Summarization always continues
    workflow.add_edge("summarization", "quality_score")

    # Quality score always continues
    workflow.add_edge("quality_score", "end")

    # Error handler attempts recovery
    workflow.add_conditional_edges(
        "error_handler",
        lambda state: "summarization" if not state["summary"] else "quality_score" if not state["qa_score"] else "end",
        {"summarization": "summarization", "quality_score": "quality_score", "end": "end"},
    )

    # End is final
    workflow.add_edge("end", END)

    logger.info("Workflow graph created successfully")
    return workflow.compile()


def run_workflow(
    compiled_graph: CompiledGraph,
    call_input: CallInput,
    llm_name: Literal["claude", "gpt4", "gemini"] = "claude",
) -> CallResult:
    """
    Execute the workflow on a single call.

    Args:
        compiled_graph: Compiled LangGraph workflow
        call_input: Input call data
        llm_name: LLM name for reference (not used here, passed to workflow)

    Returns:
        CallResult: Complete call analysis result

    Raises:
        ValueError: If workflow execution fails
    """
    logger.info(f"[WORKFLOW] Starting execution for call {call_input.call_id}")

    # Initialize state
    initial_state: WorkflowState = {
        "call_id": call_input.call_id,
        "input_data": call_input,
        "transcript": None,
        "summary": None,
        "qa_score": None,
        "errors": [],
        "current_step": "initialized",
    }

    try:
        # Execute workflow
        final_state = compiled_graph.invoke(initial_state)

        # Build result
        result = CallResult(
            call_id=final_state["call_id"],
            input_data=final_state["input_data"],
            transcript=final_state["transcript"],
            summary=final_state["summary"],
            qa_score=final_state["qa_score"],
            errors=final_state["errors"],
            current_step=final_state["current_step"],
        )

        logger.info(f"[WORKFLOW] Completed for call {call_input.call_id}")

        # Persist to memory layer
        try:
            call_memory.add_call(result.model_dump(), llm_name=llm_name)
        except Exception as mem_err:
            logger.warning(f"[MEMORY] Failed to save call history: {mem_err}")

        return result

    except Exception as e:
        logger.error(f"[WORKFLOW] Failed for call {call_input.call_id}: {e}")
        raise ValueError(f"Workflow execution failed: {str(e)}")
