"""
Memory Layer for Call Center AI Assistant.
Implements LangGraph Memory concepts using a file-based JSONL store.
Retains and recalls call context across sessions.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

HISTORY_FILE = Path(__file__).parent.parent / "data" / "call_history.jsonl"


class CallMemory:
    """
    Persistent memory store for processed calls.
    Uses JSONL (one JSON object per line) for append-efficient storage.
    Implements LangGraph Memory concepts: add, retrieve, search, summarize.
    """

    def __init__(self, history_file: Optional[Path] = None):
        self.history_file = Path(history_file) if history_file else HISTORY_FILE
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_call(self, call_result: Dict[str, Any], llm_name: str = "unknown") -> None:
        """
        Persist a processed call result to memory.

        Args:
            call_result: Serialized CallResult dict
            llm_name: Which LLM was used
        """
        entry = {
            "stored_at": datetime.utcnow().isoformat() + "Z",
            "llm_name": llm_name,
            "call_id": call_result.get("call_id", ""),
            "category": call_result.get("input_data", {}).get("metadata", {}).get("category", "unknown"),
            "resolution_status": (
                call_result.get("summary", {}).get("resolution_status", "unknown")
                if call_result.get("summary") else "unknown"
            ),
            "overall_score": (
                call_result.get("qa_score", {}).get("overall_score")
                if call_result.get("qa_score") else None
            ),
            "summary_text": (
                call_result.get("summary", {}).get("summary", "")
                if call_result.get("summary") else ""
            ),
            "customer_issue": (
                call_result.get("summary", {}).get("customer_issue", "")
                if call_result.get("summary") else ""
            ),
            "action_items": (
                call_result.get("summary", {}).get("action_items", [])
                if call_result.get("summary") else []
            ),
            "errors": call_result.get("errors", []),
            "full_result": call_result,
        }

        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self.logger.info(f"[MEMORY] Saved call {entry['call_id']} to history")

    # ── Read ──────────────────────────────────────────────────────────────────

    def _load_all(self) -> List[Dict[str, Any]]:
        """Load all entries from the history file."""
        if not self.history_file.exists():
            return []
        entries = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        self.logger.warning(f"[MEMORY] Skipping malformed line in history")
        return entries

    def get_recent_calls(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Return the n most recent calls (newest first).

        Args:
            n: Number of calls to return

        Returns:
            List of call summary dicts
        """
        entries = self._load_all()
        return list(reversed(entries))[:n]

    def get_call_by_id(self, call_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific call by call_id (most recent match).

        Args:
            call_id: Call identifier

        Returns:
            Call entry dict, or None
        """
        entries = self._load_all()
        for entry in reversed(entries):
            if entry.get("call_id") == call_id:
                return entry
        return None

    def search_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Search calls by category (case-insensitive).

        Args:
            category: Category string to search

        Returns:
            List of matching call entries (newest first)
        """
        entries = self._load_all()
        matches = [e for e in entries if category.lower() in (e.get("category") or "").lower()]
        return list(reversed(matches))

    def search_by_resolution(self, status: str) -> List[Dict[str, Any]]:
        """
        Filter calls by resolution status.

        Args:
            status: 'resolved', 'unresolved', or 'escalated'

        Returns:
            List of matching call entries
        """
        entries = self._load_all()
        matches = [e for e in entries if e.get("resolution_status") == status.lower()]
        return list(reversed(matches))

    def get_stats(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics over all stored calls.

        Returns:
            Dict with counts, average score, resolution breakdown
        """
        entries = self._load_all()
        if not entries:
            return {"total_calls": 0}

        scores = [e["overall_score"] for e in entries if e.get("overall_score") is not None]
        resolution_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}

        for e in entries:
            r = e.get("resolution_status", "unknown")
            resolution_counts[r] = resolution_counts.get(r, 0) + 1
            c = e.get("category", "unknown")
            category_counts[c] = category_counts.get(c, 0) + 1

        return {
            "total_calls": len(entries),
            "avg_qa_score": round(sum(scores) / len(scores), 1) if scores else None,
            "min_qa_score": round(min(scores), 1) if scores else None,
            "max_qa_score": round(max(scores), 1) if scores else None,
            "resolution_breakdown": resolution_counts,
            "category_breakdown": category_counts,
        }

    # ── Delete ────────────────────────────────────────────────────────────────

    def clear_history(self) -> int:
        """
        Delete all call history. Returns number of entries deleted.
        """
        entries = self._load_all()
        if self.history_file.exists():
            self.history_file.unlink()
        self.logger.info(f"[MEMORY] Cleared {len(entries)} history entries")
        return len(entries)

    def total_calls(self) -> int:
        """Return total number of stored calls."""
        return len(self._load_all())


# Singleton instance
call_memory = CallMemory()
