"""
Multi-LLM Benchmark: Compares performance across Claude, GPT-4, and Gemini.
The impressive feature for demonstrating multi-model evaluation capabilities.
"""

import logging
import time
from typing import Optional, Literal
import concurrent.futures

from agents.summarization_agent import SummarizationAgent
from agents.quality_score_agent import QualityScoreAgent
from utils.schemas import BenchmarkResult, SummaryOutput, QAScore

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs multi-LLM comparison benchmarks.
    Executes the same task across Claude, GPT-4, and Gemini simultaneously.
    """

    MODELS = ["claude", "gpt4", "gemini"]

    def __init__(self):
        """Initialize benchmark runner."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_summarization_benchmark(
        self,
        call_id: str,
        transcript: str,
        timeout: int = 60,
    ) -> BenchmarkResult:
        """
        Run summarization across all LLMs in parallel.

        Args:
            call_id: Call identifier
            transcript: Call transcript
            timeout: Timeout per LLM in seconds

        Returns:
            BenchmarkResult: Comparison results with timings and token counts
        """
        self.logger.info(f"Starting summarization benchmark for call {call_id}")

        result = BenchmarkResult(call_id=call_id)

        # Run summarization for each model in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._run_summarization_for_model,
                    model,
                    call_id,
                    transcript,
                ): model
                for model in self.MODELS
            }

            for future in concurrent.futures.as_completed(futures, timeout=timeout * 3):
                model = futures[future]
                try:
                    summary, elapsed_time, token_count = future.result(timeout=timeout)
                    setattr(result, f"{model}_summary", summary)
                    result.timing[model] = elapsed_time
                    result.token_counts[model] = token_count
                    self.logger.info(
                        f"Summarization complete for {model}: {elapsed_time:.2f}s"
                    )
                except Exception as e:
                    self.logger.error(f"Summarization failed for {model}: {e}")
                    result.errors[model] = str(e)

        return result

    def run_qa_benchmark(
        self,
        call_id: str,
        transcript: str,
        timeout: int = 60,
    ) -> BenchmarkResult:
        """
        Run QA scoring across all LLMs in parallel.

        Args:
            call_id: Call identifier
            transcript: Call transcript
            timeout: Timeout per LLM in seconds

        Returns:
            BenchmarkResult: Comparison results with timings and token counts
        """
        self.logger.info(f"Starting QA benchmark for call {call_id}")

        result = BenchmarkResult(call_id=call_id)

        # Run QA scoring for each model in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._run_qa_for_model,
                    model,
                    call_id,
                    transcript,
                ): model
                for model in self.MODELS
            }

            for future in concurrent.futures.as_completed(futures, timeout=timeout * 3):
                model = futures[future]
                try:
                    qa_score, elapsed_time, token_count = future.result(timeout=timeout)
                    setattr(result, f"{model}_qa", qa_score)
                    result.timing[model] = elapsed_time
                    result.token_counts[model] = token_count
                    self.logger.info(
                        f"QA scoring complete for {model}: {elapsed_time:.2f}s, Score: {qa_score.overall_score:.1f}/100"
                    )
                except Exception as e:
                    self.logger.error(f"QA scoring failed for {model}: {e}")
                    result.errors[model] = str(e)

        return result

    def run_full_benchmark(
        self,
        call_id: str,
        transcript: str,
        timeout: int = 60,
    ) -> BenchmarkResult:
        """
        Run full benchmark (summarization + QA scoring) across all LLMs.

        Args:
            call_id: Call identifier
            transcript: Call transcript
            timeout: Timeout per LLM per task in seconds

        Returns:
            BenchmarkResult: Complete comparison results
        """
        self.logger.info(f"Starting full benchmark for call {call_id}")

        result = BenchmarkResult(call_id=call_id)

        # Run all tasks for each model in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}

            # Submit all tasks
            for model in self.MODELS:
                futures[executor.submit(
                    self._run_full_for_model,
                    model,
                    call_id,
                    transcript,
                )] = model

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures, timeout=timeout * 6):
                model = futures[future]
                try:
                    summary, qa_score, timing, token_count = future.result(timeout=timeout * 2)
                    setattr(result, f"{model}_summary", summary)
                    setattr(result, f"{model}_qa", qa_score)
                    result.timing[model] = timing
                    result.token_counts[model] = token_count
                    self.logger.info(
                        f"Full benchmark complete for {model}: {timing:.2f}s"
                    )
                except Exception as e:
                    self.logger.error(f"Full benchmark failed for {model}: {e}")
                    result.errors[model] = str(e)

        self.logger.info(f"Benchmark complete for call {call_id}")
        return result

    def _run_summarization_for_model(
        self,
        model: Literal["claude", "gpt4", "gemini"],
        call_id: str,
        transcript: str,
    ) -> tuple[Optional[SummaryOutput], float, dict]:
        """
        Run summarization for a single model.

        Args:
            model: Model name
            call_id: Call identifier
            transcript: Transcript text

        Returns:
            tuple: (summary, elapsed_time, token_count)
        """
        self.logger.info(f"Starting summarization for {model}")
        start_time = time.time()

        try:
            agent = SummarizationAgent(llm_name=model)
            summary = agent.process(call_id=call_id, transcript=transcript)
            elapsed_time = time.time() - start_time

            # Estimate token count (approximate)
            token_count = {
                "input": len(transcript.split()),
                "output": len(summary.summary.split()) + sum(len(p.split()) for p in summary.key_points),
            }

            return summary, elapsed_time, token_count
        except Exception as e:
            self.logger.error(f"Summarization failed for {model}: {e}")
            raise

    def _run_qa_for_model(
        self,
        model: Literal["claude", "gpt4", "gemini"],
        call_id: str,
        transcript: str,
    ) -> tuple[Optional[QAScore], float, dict]:
        """
        Run QA scoring for a single model.

        Args:
            model: Model name
            call_id: Call identifier
            transcript: Transcript text

        Returns:
            tuple: (qa_score, elapsed_time, token_count)
        """
        self.logger.info(f"Starting QA scoring for {model}")
        start_time = time.time()

        try:
            agent = QualityScoreAgent(llm_name=model)
            qa_score = agent.process(call_id=call_id, transcript=transcript)
            elapsed_time = time.time() - start_time

            # Estimate token count
            token_count = {
                "input": len(transcript.split()),
                "output": len(qa_score.tone.split()) + sum(len(s.split()) for s in qa_score.strengths),
            }

            return qa_score, elapsed_time, token_count
        except Exception as e:
            self.logger.error(f"QA scoring failed for {model}: {e}")
            raise

    def _run_full_for_model(
        self,
        model: Literal["claude", "gpt4", "gemini"],
        call_id: str,
        transcript: str,
    ) -> tuple[Optional[SummaryOutput], Optional[QAScore], float, dict]:
        """
        Run full benchmark (summarization + QA) for a single model.

        Args:
            model: Model name
            call_id: Call identifier
            transcript: Transcript text

        Returns:
            tuple: (summary, qa_score, total_time, token_count)
        """
        start_time = time.time()

        try:
            # Run summarization
            summary_output, _, summary_tokens = self._run_summarization_for_model(model, call_id, transcript)

            # Run QA scoring
            qa_output, _, qa_tokens = self._run_qa_for_model(model, call_id, transcript)

            total_time = time.time() - start_time

            token_count = {
                "input": summary_tokens["input"] + qa_tokens["input"],
                "output": summary_tokens["output"] + qa_tokens["output"],
            }

            return summary_output, qa_output, total_time, token_count
        except Exception as e:
            self.logger.error(f"Full benchmark failed for {model}: {e}")
            raise

    def compare_results(self, result: BenchmarkResult) -> dict:
        """
        Generate a summary comparison of benchmark results.

        Args:
            result: Benchmark result

        Returns:
            dict: Comparison summary with insights
        """
        comparison = {
            "call_id": result.call_id,
            "timing_by_model": result.timing,
            "fastest_model": min(result.timing.items(), key=lambda x: x[1])[0] if result.timing else None,
            "average_time": sum(result.timing.values()) / len(result.timing) if result.timing else None,
        }

        # Compare summaries if available
        if result.claude_summary and result.gpt4_summary and result.gemini_summary:
            comparison["summaries_available"] = 3
            comparison["summary_comparison"] = {
                "claude": result.claude_summary.resolution_status.value,
                "gpt4": result.gpt4_summary.resolution_status.value,
                "gemini": result.gemini_summary.resolution_status.value,
            }

        # Compare QA scores if available
        if result.claude_qa and result.gpt4_qa and result.gemini_qa:
            comparison["qa_scores_available"] = 3
            comparison["qa_comparison"] = {
                "claude": result.claude_qa.overall_score,
                "gpt4": result.gpt4_qa.overall_score,
                "gemini": result.gemini_qa.overall_score,
            }
            comparison["highest_qa_score"] = max(comparison["qa_comparison"].items(), key=lambda x: x[1])[0]

        # Report errors
        if result.errors:
            comparison["errors"] = result.errors
            comparison["success_rate"] = (3 - len(result.errors)) / 3 * 100

        return comparison
