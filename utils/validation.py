"""
Validation utilities for Call Center AI Assistant.
Validates transcripts, audio files, call IDs, and structured outputs.
"""

import re
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac"}

# Max transcript length (characters)
MAX_TRANSCRIPT_LENGTH = 50_000

# Min transcript length (characters)
MIN_TRANSCRIPT_LENGTH = 10


def validate_transcript_text(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a transcript text string.

    Args:
        text: Transcript text to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if not text or not isinstance(text, str):
        return False, "Transcript must be a non-empty string"

    stripped = text.strip()

    if len(stripped) < MIN_TRANSCRIPT_LENGTH:
        return False, f"Transcript is too short (minimum {MIN_TRANSCRIPT_LENGTH} characters)"

    if len(stripped) > MAX_TRANSCRIPT_LENGTH:
        return False, f"Transcript is too long (maximum {MAX_TRANSCRIPT_LENGTH} characters)"

    # Warn but don't fail if no speaker labels
    has_speaker_labels = bool(re.search(r'^(Agent|Customer|Speaker)[\s:]', stripped, re.MULTILINE | re.IGNORECASE))
    if not has_speaker_labels:
        logger.warning("Transcript has no speaker labels (Agent:/Customer:) — results may be less accurate")

    return True, None


def validate_audio_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an audio file path.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"Audio file not found: {file_path}"

    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        return False, (
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
        )

    # Basic size check (100 MB max)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 100:
        return False, f"Audio file too large ({size_mb:.1f} MB). Maximum is 100 MB."

    return True, None


def validate_call_id(call_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a call ID string.

    Args:
        call_id: Call ID to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if not call_id or not isinstance(call_id, str):
        return False, "Call ID must be a non-empty string"

    stripped = call_id.strip()

    if len(stripped) < 3:
        return False, "Call ID must be at least 3 characters"

    if len(stripped) > 64:
        return False, "Call ID must not exceed 64 characters"

    # Allow alphanumeric, hyphens, underscores
    if not re.match(r'^[A-Za-z0-9_\-]+$', stripped):
        return False, "Call ID must contain only letters, digits, hyphens, and underscores"

    return True, None


def validate_qa_scores(scores: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate QA score dictionary values are within expected ranges.

    Args:
        scores: Dict with keys: overall_score, empathy_score,
                professionalism_score, resolution_score, compliance_score

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    dimension_keys = ["empathy_score", "professionalism_score", "resolution_score", "compliance_score"]

    for key in dimension_keys:
        if key not in scores:
            return False, f"Missing required score field: {key}"
        val = scores[key]
        if not (0 <= val <= 25):
            return False, f"{key} out of range: {val} (expected 0-25)"

    if "overall_score" in scores:
        overall = scores["overall_score"]
        if not (0 <= overall <= 100):
            return False, f"overall_score out of range: {overall} (expected 0-100)"

    return True, None


def validate_resolution_status(status: str) -> Tuple[bool, Optional[str]]:
    """
    Validate resolution status string.

    Args:
        status: Resolution status

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    valid_statuses = {"resolved", "unresolved", "escalated"}
    if status.lower() not in valid_statuses:
        return False, f"Invalid resolution status '{status}'. Must be one of: {valid_statuses}"
    return True, None


def sanitize_transcript(text: str) -> str:
    """
    Sanitize transcript text for safe processing.
    Removes null bytes and normalizes line endings.

    Args:
        text: Raw transcript text

    Returns:
        str: Sanitized transcript text
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
