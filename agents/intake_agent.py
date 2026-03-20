"""
Intake Agent: Validates and processes input calls.
Handles both file paths and raw transcript text, extracting metadata.
"""

import json
import os
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import logging

from utils.schemas import CallInput

logger = logging.getLogger(__name__)


class IntakeAgent:
    """
    Validates and processes incoming call data.
    Supports JSON transcript files and plain text input.
    """

    def __init__(self):
        """Initialize the intake agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def process(
        self,
        file_path: Optional[str] = None,
        transcript_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CallInput:
        """
        Process call input from file or text.

        Args:
            file_path: Path to audio or transcript file
            transcript_text: Raw transcript text
            metadata: Additional metadata to attach to the call

        Returns:
            CallInput: Validated call input object

        Raises:
            ValueError: If neither file_path nor transcript_text provided
            FileNotFoundError: If file_path doesn't exist
            json.JSONDecodeError: If JSON file is malformed
        """
        if not file_path and not transcript_text:
            raise ValueError("Either file_path or transcript_text must be provided")

        if metadata is None:
            metadata = {}

        call_id = metadata.get("call_id") or self._generate_call_id()

        # Process file if provided
        if file_path:
            transcript_text, extracted_metadata = self._process_file(file_path)
            # Merge extracted metadata with provided metadata
            extracted_metadata.update(metadata)
            metadata = extracted_metadata

        # Validate transcript text
        if not transcript_text or not transcript_text.strip():
            raise ValueError("Transcript text cannot be empty")

        # Create and validate CallInput
        call_input = CallInput(
            call_id=call_id,
            audio_path=file_path if file_path and file_path.endswith(('.mp3', '.wav', '.m4a', '.webm')) else None,
            transcript_text=transcript_text,
            metadata=metadata,
        )

        self.logger.info(f"Processed call {call_id} with {len(transcript_text)} characters")
        return call_input

    def _process_file(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """
        Read and process a file (JSON or text).

        Args:
            file_path: Path to the file

        Returns:
            tuple: (transcript_text, extracted_metadata)

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()
        metadata: Dict[str, Any] = {}

        if file_ext == ".json":
            transcript_text, metadata = self._process_json_file(file_path)
        else:
            # Treat as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            metadata["source_file"] = file_path

        return transcript_text, metadata

    def _process_json_file(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """
        Extract transcript from JSON file.

        JSON can be in several formats:
        - {"transcript": "...", "call_id": "...", ...}
        - {"text": "...", ...}
        - Direct string value of "transcript" key

        Args:
            file_path: Path to JSON file

        Returns:
            tuple: (transcript_text, extracted_metadata)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata: Dict[str, Any] = {"source_file": file_path}

        # Extract transcript text
        transcript_text = None
        if isinstance(data, str):
            transcript_text = data
        elif isinstance(data, dict):
            # Try common keys
            for key in ["transcript", "text", "content", "call_text"]:
                if key in data:
                    transcript_text = data[key]
                    break

            # Extract metadata
            if "call_id" in data:
                metadata["call_id"] = data["call_id"]
            if "category" in data:
                metadata["category"] = data["category"]
            if "duration_seconds" in data:
                metadata["duration_seconds"] = data["duration_seconds"]
            if "timestamp" in data:
                metadata["timestamp"] = data["timestamp"]
            if "agents" in data:
                metadata["agents"] = data["agents"]
            if "metadata" in data:
                metadata.update(data["metadata"])

        if not transcript_text:
            raise ValueError(f"Could not extract transcript from {file_path}")

        return transcript_text, metadata

    def _generate_call_id(self) -> str:
        """
        Generate a unique call ID.

        Returns:
            str: Generated call ID (timestamp-based)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"CALL_{timestamp}"

    def validate_transcript_format(self, transcript_text: str) -> bool:
        """
        Validate that transcript has proper speaker labels.

        Args:
            transcript_text: Transcript to validate

        Returns:
            bool: True if transcript has Agent/Customer labels or other valid format
        """
        lines = transcript_text.strip().split('\n')
        if not lines:
            return False

        # Check for speaker labels
        has_labels = any(line.startswith(('Agent:', 'Customer:', 'Speaker:', 'Agent ', 'Customer '))
                        for line in lines if line.strip())

        return len(lines) > 0 and has_labels
