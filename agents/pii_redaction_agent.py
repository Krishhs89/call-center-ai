"""
PII Redaction Agent: Detects and masks Personally Identifiable Information.
Uses regex patterns for fast, API-free redaction before LLM processing.
Critical for GDPR, HIPAA, and PCI-DSS compliance in production environments.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# (compiled pattern, replacement label, description key)
_RAW_PATTERNS = [
    (r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', 'phone_numbers'),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', 'email_addresses'),
    (
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?'
        r'|5[1-5][0-9]{14}'
        r'|3[47][0-9]{13}'
        r'|3(?:0[0-5]|[68][0-9])[0-9]{11})\b',
        '[CARD]', 'credit_cards',
    ),
    (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', '[SSN]', 'ssn'),
    (r'(?i)\b(?:account|acct)[\s#:]*\d{6,12}\b', '[ACCOUNT]', 'account_numbers'),
    (r'(?i)\b(?:DOB|date\s+of\s+birth|born)[:\s]+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', '[DOB]', 'dates_of_birth'),
    (r'\b\d{5}(?:[-\s]\d{4})?\b', '[ZIP]', 'zip_codes'),
]


@dataclass
class RedactionResult:
    redacted_transcript: str
    redaction_summary: dict = field(default_factory=dict)
    total_redactions: int = 0


class PIIRedactionAgent:
    """
    Masks PII from transcripts before they are sent to LLMs.
    Runs entirely with regex — zero API calls, zero latency overhead.

    Production use cases:
    - GDPR: masks EU personal data before cloud LLM processing
    - HIPAA: removes patient identifiers from healthcare call transcripts
    - PCI-DSS: prevents credit card numbers from reaching LLM APIs
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._compiled = [
            (re.compile(pat, re.IGNORECASE), label, desc)
            for pat, label, desc in _RAW_PATTERNS
        ]

    def redact(self, transcript: str) -> RedactionResult:
        """
        Scan and mask all detected PII in a transcript.

        Args:
            transcript: Raw transcript text

        Returns:
            RedactionResult with masked text and summary of what was found
        """
        redacted = transcript
        summary: dict[str, int] = {}

        for pattern, label, desc in self._compiled:
            matches = pattern.findall(redacted)
            if matches:
                summary[desc] = len(matches)
                redacted = pattern.sub(label, redacted)

        total = sum(summary.values())
        if total > 0:
            self.logger.info(f"[PII] Redacted {total} item(s): {summary}")
        else:
            self.logger.info("[PII] No PII detected in transcript")

        return RedactionResult(
            redacted_transcript=redacted,
            redaction_summary=summary,
            total_redactions=total,
        )
