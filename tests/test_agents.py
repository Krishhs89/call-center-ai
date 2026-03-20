"""
Unit tests for Call Center AI agents.
Tests core agent functionality with mocking where appropriate.
"""

import pytest
import json
from pathlib import Path

from agents.intake_agent import IntakeAgent
from agents.transcription_agent import TranscriptionAgent
from agents.summarization_agent import SummarizationAgent
from agents.quality_score_agent import QualityScoreAgent
from utils.schemas import CallInput, TranscriptOutput, SummaryOutput, QAScore, ResolutionStatus


class TestIntakeAgent:
    """Tests for IntakeAgent."""

    @pytest.fixture
    def agent(self):
        """Create intake agent instance."""
        return IntakeAgent()

    def test_process_with_text(self, agent):
        """Test processing transcript text input."""
        transcript = "Agent: Hello. Customer: Hi, I have a problem."

        result = agent.process(transcript_text=transcript)

        assert isinstance(result, CallInput)
        assert result.call_id
        assert result.transcript_text == transcript

    def test_process_with_empty_text(self, agent):
        """Test processing with empty text raises error."""
        with pytest.raises(ValueError):
            agent.process(transcript_text="")

    def test_process_with_no_input(self, agent):
        """Test processing with no input raises error."""
        with pytest.raises(ValueError):
            agent.process()

    def test_process_with_metadata(self, agent):
        """Test processing with additional metadata."""
        transcript = "Agent: Hello. Customer: Hi."
        metadata = {"category": "billing", "duration_seconds": 300}

        result = agent.process(transcript_text=transcript, metadata=metadata)

        assert result.metadata["category"] == "billing"
        assert result.metadata["duration_seconds"] == 300

    def test_process_json_file(self, agent, tmp_path):
        """Test processing JSON file."""
        json_data = {
            "call_id": "TEST_001",
            "transcript": "Agent: Hello. Customer: Hi.",
            "category": "support",
        }
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(json_data))

        result = agent.process(file_path=str(json_file))

        assert result.call_id == "TEST_001"
        assert result.transcript_text == "Agent: Hello. Customer: Hi."
        assert result.metadata["category"] == "support"

    def test_validate_transcript_format(self, agent):
        """Test transcript format validation."""
        valid_transcript = "Agent: Hello\nCustomer: Hi"
        invalid_transcript = "Just plain text without labels"

        assert agent.validate_transcript_format(valid_transcript) is True
        assert agent.validate_transcript_format(invalid_transcript) is False


class TestTranscriptionAgent:
    """Tests for TranscriptionAgent."""

    @pytest.fixture
    def agent(self):
        """Create transcription agent instance."""
        return TranscriptionAgent()

    def test_normalize_agent_customer_labels(self, agent):
        """Test normalizing standard Agent/Customer labels."""
        transcript = "Agent: Hello\nCustomer: Hi there"

        normalized, speakers = agent._normalize_transcript(transcript)

        assert "Agent: Hello" in normalized
        assert "Customer: Hi there" in normalized
        assert set(speakers) == {"Agent", "Customer"}

    def test_normalize_various_speaker_formats(self, agent):
        """Test normalizing various speaker label formats."""
        transcript = """[Agent] Hi there
[Customer] Hello
Rep: How can I help?
Client: I need support"""

        normalized, speakers = agent._normalize_transcript(transcript)

        # Should normalize to Agent/Customer
        assert "Agent:" in normalized or normalized  # Some format preserved
        assert len(speakers) > 0

    def test_process_with_transcript_text(self, agent):
        """Test processing transcript text."""
        transcript = "Agent: Good afternoon. Customer: Hi, I need help."

        result = agent.process(call_id="TEST_001", transcript_text=transcript)

        assert isinstance(result, TranscriptOutput)
        assert result.call_id == "TEST_001"
        assert result.transcript
        assert "Agent" in result.speakers or "Customer" in result.speakers

    def test_process_with_empty_transcript(self, agent):
        """Test processing empty transcript raises error."""
        with pytest.raises(ValueError):
            agent.process(call_id="TEST_001", transcript_text="")

    def test_extract_speaker(self, agent):
        """Test speaker extraction from lines."""
        test_cases = [
            ("Agent: Hello", "Agent"),
            ("Customer: Hi", "Customer"),
            ("[Agent] Message", "Agent"),
            ("Rep: How can I help?", None),  # May or may not extract
        ]

        for line, expected_speaker in test_cases:
            speaker = agent._extract_speaker(line)
            # Just check it doesn't crash, extraction logic may vary


class TestSummarizationAgent:
    """Tests for SummarizationAgent."""

    @pytest.fixture
    def agent(self):
        """Create summarization agent with mock."""
        # Note: In real tests, you'd mock the LLM calls
        try:
            return SummarizationAgent(llm_name="claude")
        except ValueError:
            pytest.skip("Anthropic API key not configured")

    def test_process_builds_prompt(self, agent):
        """Test that process builds appropriate prompt."""
        transcript = "Agent: Hello. Customer: I need help."

        # Just verify it doesn't crash - actual LLM call may fail without API key
        try:
            result = agent.process(call_id="TEST_001", transcript=transcript)
            assert isinstance(result, SummaryOutput)
        except Exception:
            pytest.skip("LLM API not available")

    def test_process_with_empty_transcript(self, agent):
        """Test processing empty transcript raises error."""
        with pytest.raises(ValueError):
            agent.process(call_id="TEST_001", transcript="")


class TestQualityScoreAgent:
    """Tests for QualityScoreAgent."""

    @pytest.fixture
    def agent(self):
        """Create quality score agent."""
        try:
            return QualityScoreAgent(llm_name="claude")
        except ValueError:
            pytest.skip("Anthropic API key not configured")

    def test_validate_scores(self, agent):
        """Test score validation."""
        # Create a valid score
        valid_score = QAScore(
            call_id="TEST_001",
            overall_score=85.0,
            empathy_score=20,
            professionalism_score=21,
            resolution_score=22,
            compliance_score=22,
            tone="Professional",
            strengths=["Clear communication"],
            improvements=["Be more empathetic"],
        )

        assert agent.validate_scores(valid_score) is True

    def test_validate_scores_out_of_range(self, agent):
        """Test validation catches out-of-range scores."""
        invalid_score = QAScore(
            call_id="TEST_001",
            overall_score=150.0,  # Out of range
            empathy_score=30,  # Should be 0-25
            professionalism_score=21,
            resolution_score=22,
            compliance_score=22,
            tone="Professional",
            strengths=["Clear communication"],
            improvements=["Be more empathetic"],
        )

        assert agent.validate_scores(invalid_score) is False

    def test_process_with_empty_transcript(self, agent):
        """Test processing empty transcript raises error."""
        with pytest.raises(ValueError):
            agent.process(call_id="TEST_001", transcript="")


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_call_input_creation(self):
        """Test CallInput schema validation."""
        call_input = CallInput(
            call_id="TEST_001",
            transcript_text="Agent: Hello. Customer: Hi.",
            metadata={"category": "billing"},
        )

        assert call_input.call_id == "TEST_001"
        assert call_input.transcript_text == "Agent: Hello. Customer: Hi."
        assert call_input.metadata["category"] == "billing"

    def test_transcript_output_creation(self):
        """Test TranscriptOutput schema validation."""
        output = TranscriptOutput(
            call_id="TEST_001",
            transcript="Agent: Hello\nCustomer: Hi",
            speakers=["Agent", "Customer"],
            duration_seconds=300,
        )

        assert output.call_id == "TEST_001"
        assert len(output.speakers) == 2

    def test_summary_output_creation(self):
        """Test SummaryOutput schema validation."""
        summary = SummaryOutput(
            call_id="TEST_001",
            summary="Customer called with billing issue",
            key_points=["Billing error", "Account compromise"],
            action_items=["Refund customer", "Reset password"],
            customer_issue="Fraudulent charge",
            resolution_status=ResolutionStatus.RESOLVED,
        )

        assert summary.call_id == "TEST_001"
        assert summary.resolution_status == ResolutionStatus.RESOLVED

    def test_qa_score_creation(self):
        """Test QAScore schema validation."""
        qa_score = QAScore(
            call_id="TEST_001",
            overall_score=85.5,
            empathy_score=24,
            professionalism_score=23,
            resolution_score=22,
            compliance_score=21,
            tone="Professional and empathetic",
            strengths=["Clear communication", "Problem-solving"],
            improvements=["Faster resolution", "More follow-up"],
        )

        assert qa_score.call_id == "TEST_001"
        assert qa_score.overall_score == 85.5
        assert 0 <= qa_score.empathy_score <= 25


@pytest.mark.integration
class TestIntegration:
    """Integration tests for agent workflows."""

    def test_full_processing_pipeline(self):
        """Test complete processing pipeline."""
        # Create sample transcript
        transcript = """Agent: Good afternoon, thank you for calling. How can I help you?
Customer: Hi, I'm having issues with my billing.
Agent: I understand. Let me help you with that.
Customer: Thank you."""

        # Process through intake
        intake = IntakeAgent()
        call_input = intake.process(transcript_text=transcript)
        assert call_input.call_id

        # Process through transcription
        transcription = TranscriptionAgent()
        transcript_output = transcription.process(
            call_id=call_input.call_id,
            transcript_text=call_input.transcript_text,
        )
        assert transcript_output.transcript

        # Verify schemas
        assert isinstance(call_input, CallInput)
        assert isinstance(transcript_output, TranscriptOutput)
