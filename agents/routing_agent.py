"""
Routing Agent: Orchestrates the workflow and routes between agents.
Manages state transitions, error handling, and logging.
"""

import logging
from typing import Any, Dict, Optional
from utils.schemas import CallResult, CallInput

logger = logging.getLogger(__name__)


class RoutingAgent:
    """
    Routes calls through the processing workflow.
    Manages state transitions and error handling.
    """

    def __init__(self):
        """Initialize the routing agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def route_intake(self, call_result: CallResult) -> str:
        """
        Route after intake processing.

        Args:
            call_result: Current call result state

        Returns:
            str: Next node ("transcription" or "summarization")
        """
        self.logger.info(f"[ROUTE] Call {call_result.call_id}: Intake -> determining next step")

        # If audio_path exists, go to transcription
        if call_result.input_data.audio_path:
            self.logger.info(f"[ROUTE] Call {call_result.call_id}: Audio detected, routing to transcription")
            return "transcription"
        else:
            # Skip transcription if only text
            self.logger.info(f"[ROUTE] Call {call_result.call_id}: Text input, skipping transcription")
            return "transcription"  # Still route to transcription for normalization

    def route_transcription(self, call_result: CallResult) -> str:
        """
        Route after transcription processing.

        Args:
            call_result: Current call result state

        Returns:
            str: Next node ("summarization" or "error_handler")
        """
        self.logger.info(f"[ROUTE] Call {call_result.call_id}: Transcription -> determining next step")

        if call_result.errors:
            self.logger.warning(f"[ROUTE] Call {call_result.call_id}: Errors detected, routing to error handler")
            return "error_handler"

        if call_result.transcript:
            self.logger.info(f"[ROUTE] Call {call_result.call_id}: Transcript ready, routing to summarization")
            return "summarization"
        else:
            self.logger.error(f"[ROUTE] Call {call_result.call_id}: No transcript available")
            call_result.errors.append("Transcription produced no output")
            return "error_handler"

    def route_summarization(self, call_result: CallResult) -> str:
        """
        Route after summarization processing.

        Args:
            call_result: Current call result state

        Returns:
            str: Next node ("quality_score" or "error_handler")
        """
        self.logger.info(f"[ROUTE] Call {call_result.call_id}: Summarization -> determining next step")

        if call_result.errors:
            self.logger.warning(f"[ROUTE] Call {call_result.call_id}: Errors detected")

        if call_result.summary:
            self.logger.info(f"[ROUTE] Call {call_result.call_id}: Summary ready, routing to QA scoring")
            return "quality_score"
        else:
            self.logger.warning(f"[ROUTE] Call {call_result.call_id}: No summary produced, continuing to QA")
            return "quality_score"

    def route_quality_score(self, call_result: CallResult) -> str:
        """
        Route after quality scoring.

        Args:
            call_result: Current call result state

        Returns:
            str: Next node ("end")
        """
        self.logger.info(f"[ROUTE] Call {call_result.call_id}: Quality Score -> completing workflow")

        if call_result.qa_score:
            self.logger.info(f"[ROUTE] Call {call_result.call_id}: QA score {call_result.qa_score.overall_score:.1f}/100")
        else:
            self.logger.warning(f"[ROUTE] Call {call_result.call_id}: No QA score produced")

        return "end"

    def handle_error(self, call_result: CallResult) -> str:
        """
        Handle errors and attempt recovery.

        Args:
            call_result: Current call result state

        Returns:
            str: Next node
        """
        self.logger.error(f"[ROUTE] Call {call_result.call_id}: Error handler triggered")
        self.logger.error(f"[ROUTE] Errors: {call_result.errors}")

        # Log all errors
        for error in call_result.errors:
            self.logger.error(f"[ROUTE] - {error}")

        # Always continue to summarization even if transcription failed
        if not call_result.summary:
            self.logger.info(f"[ROUTE] Call {call_result.call_id}: Attempting to continue to summarization")
            return "summarization"
        elif not call_result.qa_score:
            self.logger.info(f"[ROUTE] Call {call_result.call_id}: Attempting to continue to quality scoring")
            return "quality_score"
        else:
            self.logger.info(f"[ROUTE] Call {call_result.call_id}: Routing to end")
            return "end"

    def log_state_transition(self, call_id: str, from_node: str, to_node: str, state: Dict[str, Any]) -> None:
        """
        Log a state transition for debugging and monitoring.
        Compatible with LangSmith tracing.

        Args:
            call_id: Call identifier
            from_node: Source node name
            to_node: Target node name
            state: Current state dictionary
        """
        self.logger.info(f"[STATE_TRANSITION] {call_id}: {from_node} -> {to_node}")

    def validate_state(self, call_result: CallResult) -> bool:
        """
        Validate call result state.

        Args:
            call_result: Call result to validate

        Returns:
            bool: True if state is valid
        """
        if not call_result.call_id:
            self.logger.error("Call result missing call_id")
            return False

        if not call_result.input_data:
            self.logger.error(f"Call {call_result.call_id}: Missing input_data")
            return False

        return True

    def end_workflow(self, call_result: CallResult) -> CallResult:
        """
        Finalize workflow and prepare output.

        Args:
            call_result: Final call result

        Returns:
            CallResult: Finalized call result
        """
        self.logger.info(f"[WORKFLOW_END] Call {call_result.call_id}: Processing complete")
        call_result.current_step = "completed"

        # Summary log
        summary_info = []
        if call_result.transcript:
            summary_info.append(f"Transcript: {len(call_result.transcript.transcript)} chars")
        if call_result.summary:
            summary_info.append(f"Summary: {call_result.summary.resolution_status.value}")
        if call_result.qa_score:
            summary_info.append(f"QA Score: {call_result.qa_score.overall_score:.1f}/100")

        self.logger.info(f"[WORKFLOW_END] Call {call_result.call_id}: {' | '.join(summary_info)}")

        if call_result.errors:
            self.logger.warning(f"[WORKFLOW_END] Call {call_result.call_id}: Completed with {len(call_result.errors)} errors")

        return call_result
