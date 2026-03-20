"""
Transcription Agent: Converts audio to text and normalizes speaker labels.
Uses OpenAI Whisper for audio processing with fallback to raw text.
"""

import re
import logging
from typing import Optional
from pathlib import Path

from utils.schemas import TranscriptOutput
from config.settings import settings

logger = logging.getLogger(__name__)


class TranscriptionAgent:
    """
    Handles audio transcription and transcript normalization.
    Uses OpenAI Whisper API if audio is provided, normalizes speaker labels.
    """

    SPEAKER_PATTERNS = [
        r'^(Agent|Customer|Speaker\s*\d+|Rep|Representative|Client)[\s:]+',
        r'^(\[Agent\]|\[Customer\]|\[Speaker.*?\])[\s:]*',
    ]

    def __init__(self):
        """Initialize the transcription agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._whisper_client = None

    def _get_whisper_client(self):
        """Lazily initialize OpenAI client for Whisper."""
        if self._whisper_client is None:
            try:
                from openai import OpenAI
                self._whisper_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            except ImportError:
                self.logger.warning("OpenAI client not available")
        return self._whisper_client

    def process(self, call_id: str, audio_path: Optional[str] = None, transcript_text: Optional[str] = None) -> TranscriptOutput:
        """
        Process audio or transcript text.

        Args:
            call_id: Call identifier
            audio_path: Path to audio file (optional)
            transcript_text: Raw transcript text (optional)

        Returns:
            TranscriptOutput: Normalized transcript with speaker labels

        Raises:
            ValueError: If neither audio_path nor transcript_text provided
        """
        if not audio_path and not transcript_text:
            raise ValueError("Either audio_path or transcript_text must be provided")

        # Process audio if provided
        if audio_path:
            transcript_text = self._transcribe_audio(audio_path)
            if not transcript_text:
                self.logger.warning(f"Audio transcription failed for {call_id}, using fallback")
                transcript_text = ""

        # Normalize transcript
        normalized_text, speakers = self._normalize_transcript(transcript_text)

        output = TranscriptOutput(
            call_id=call_id,
            transcript=normalized_text,
            speakers=speakers,
        )

        self.logger.info(f"Transcription processed for {call_id}. Speakers: {speakers}")
        return output

    def _transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio using OpenAI Whisper API.

        Args:
            audio_path: Path to audio file

        Returns:
            str: Transcribed text, or None if transcription fails

        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            client = self._get_whisper_client()
            if not client:
                self.logger.warning("OpenAI client unavailable for Whisper transcription")
                return None

            self.logger.info(f"Transcribing audio: {audio_path}")

            with open(audio_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=settings.WHISPER_MODEL,
                    file=audio_file,
                )

            return transcript.text

        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            return None

    def _normalize_transcript(self, transcript_text: str) -> tuple[str, list[str]]:
        """
        Normalize transcript by standardizing speaker labels.

        Converts various speaker formats to "Agent:" and "Customer:" labels.

        Args:
            transcript_text: Raw transcript text

        Returns:
            tuple: (normalized_text, list_of_speakers)
        """
        if not transcript_text:
            return "", []

        lines = transcript_text.strip().split('\n')
        normalized_lines = []
        speakers_set = set()
        current_speaker = None

        for line in lines:
            if not line.strip():
                if normalized_lines and normalized_lines[-1]:  # Avoid consecutive empty lines
                    normalized_lines.append("")
                continue

            # Try to extract speaker label
            speaker = self._extract_speaker(line)
            if speaker:
                current_speaker = self._normalize_speaker(speaker)
                speakers_set.add(current_speaker)
                # Remove the original speaker label and add normalized one
                content = self._remove_speaker_label(line)
                if content.strip():
                    normalized_lines.append(f"{current_speaker}: {content.strip()}")
            else:
                # Continuation of previous speaker's line
                if current_speaker:
                    normalized_lines.append(f"{current_speaker}: {line.strip()}")
                else:
                    normalized_lines.append(line.strip())

        normalized_text = '\n'.join(normalized_lines)
        speakers = sorted(list(speakers_set))

        return normalized_text, speakers

    def _extract_speaker(self, line: str) -> Optional[str]:
        """
        Extract speaker label from a line.

        Args:
            line: Line to extract speaker from

        Returns:
            str: Speaker name if found, None otherwise
        """
        for pattern in self.SPEAKER_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _normalize_speaker(self, speaker: str) -> str:
        """
        Normalize speaker name to standard format.

        Maps various speaker formats to "Agent" or "Customer".

        Args:
            speaker: Speaker name to normalize

        Returns:
            str: Normalized speaker name
        """
        speaker_lower = speaker.lower().strip()

        # Map to Agent
        if any(x in speaker_lower for x in ['agent', 'rep', 'representative', 'support', 'staff']):
            return "Agent"

        # Map to Customer
        if any(x in speaker_lower for x in ['customer', 'client', 'caller', 'user']):
            return "Customer"

        # Default: keep speaker name with first letter capitalized
        return speaker.capitalize()

    def _remove_speaker_label(self, line: str) -> str:
        """
        Remove speaker label from line.

        Args:
            line: Line with speaker label

        Returns:
            str: Content without speaker label
        """
        for pattern in self.SPEAKER_PATTERNS:
            line = re.sub(pattern, '', line, count=1)
        return line.strip()
