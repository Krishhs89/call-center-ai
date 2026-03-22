"""
RAG Retrieval Agent: Queries ChromaDB for semantically similar past calls.
Formats retrieved context for injection into summarization and QA prompts.
Runs as a LangGraph node between TranscriptionAgent and SummarizationAgent.
"""

import logging
from typing import Optional

from utils.vector_store import retrieve_similar_calls

logger = logging.getLogger(__name__)


class RAGRetrievalAgent:
    """
    Retrieves top-K similar past calls and formats them as prompt context.
    Gracefully returns an empty string if the vector store is empty or unavailable.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def retrieve_context(self, transcript: str, call_id: Optional[str] = None) -> str:
        """
        Retrieve similar past calls and format as a context block for LLM prompts.

        Args:
            transcript: Current call transcript
            call_id: Current call ID to exclude from results (avoids self-match)

        Returns:
            Formatted context string, or empty string if no similar calls found
        """
        similar = retrieve_similar_calls(
            transcript=transcript,
            top_k=self.top_k,
            exclude_call_id=call_id,
        )

        if not similar:
            return ""

        lines = ["--- SIMILAR PAST CALLS (RAG context) ---"]
        for i, call in enumerate(similar, 1):
            lines.append(
                f"[{i}] Call: {call['call_id']} | "
                f"Category: {call['category']} | "
                f"Resolution: {call['resolution_status']} | "
                f"QA Score: {call['overall_score']}/100"
            )
            lines.append(f"    Summary: {call['summary_snippet']}")
        lines.append("---")

        context = "\n".join(lines)
        self.logger.info(f"[RAG] {len(similar)} similar call(s) retrieved for context")
        return context
