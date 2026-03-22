"""
Streamlit UI for Call Center AI Assistant.
Provides tabs for upload, results, QA scoring, and multi-LLM benchmarking.
"""

import sys
import streamlit as st
import json
import logging
import os
from pathlib import Path
from typing import Optional

# Ensure the repo root is on sys.path so all modules resolve correctly
# whether running locally (from repo root) or on Streamlit Cloud (from ui/)
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from agents.intake_agent import IntakeAgent
from workflow.langgraph_flow import create_workflow, run_workflow
from evaluation.benchmark import BenchmarkRunner
from utils.schemas import CallInput, CallResult, BenchmarkResult, SummaryOutput, QAScore
from utils.cache import get_cached, save_cache, list_cache_entries, delete_cache_entry, clear_cache
from utils.memory import call_memory
from utils.vector_store import get_vector_store_stats
from utils.validation import validate_transcript_text, validate_audio_file, sanitize_transcript, SUPPORTED_AUDIO_FORMATS
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Call Center AI Assistant",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("📞 Call Center AI Assistant")
st.markdown(
    "Analyze call transcripts with AI-powered summarization and quality assessment."
)

# Initialize session state
if "call_result" not in st.session_state:
    st.session_state.call_result = None
if "benchmark_result" not in st.session_state:
    st.session_state.benchmark_result = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "active_llm" not in st.session_state:
    st.session_state.active_llm = "claude"
if "active_call_id" not in st.session_state:
    st.session_state.active_call_id = None
if "audio_transcript" not in st.session_state:
    st.session_state.audio_transcript = None
if "audio_filename" not in st.session_state:
    st.session_state.audio_filename = None
if "v2_extras" not in st.session_state:
    st.session_state.v2_extras = {}  # pii_summary, sentiment, rag_context


def load_sample_transcript(sample_name: str) -> dict:
    """Load a sample transcript from the data directory."""
    sample_path = Path(__file__).parent.parent / "data" / "sample_transcripts" / f"{sample_name}.json"
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            return json.load(f)
    return None


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # LLM Selection
    llm_choice = st.radio(
        "Select LLM for Analysis:",
        options=["claude", "gpt4", "gemini"],
        help="Choose which LLM to use for summarization and QA scoring",
    )
    # Clear stale results when LLM changes
    if llm_choice != st.session_state.active_llm:
        st.session_state.call_result = None
        st.session_state.benchmark_result = None
        st.session_state.active_call_id = None
        st.session_state.active_llm = llm_choice

    # API Key Check
    st.subheader("🔑 API Keys Status")
    if settings.ANTHROPIC_API_KEY:
        st.success("✓ Anthropic API Key configured")
    else:
        st.warning("⚠ Anthropic API Key not configured")

    if settings.OPENAI_API_KEY:
        st.success("✓ OpenAI API Key configured")
    else:
        st.warning("⚠ OpenAI API Key not configured")

    if settings.GOOGLE_API_KEY:
        st.success("✓ Google API Key configured")
    else:
        st.warning("⚠ Google API Key not configured")

    # Sample Transcripts
    st.subheader("📝 Sample Transcripts")
    SAMPLE_LABELS = [
        ("sample_call_1",  "Call 1 — Banking (Fraud)"),
        ("sample_call_2",  "Call 2 — Telecom (Billing)"),
        ("sample_call_3",  "Call 3 — Healthcare (Appointment)"),
        ("sample_call_4",  "Call 4 — Retail (Return)"),
        ("sample_call_5",  "Call 5 — Insurance (Claim)"),
        ("sample_call_6",  "Call 6 — Tech Support (Internet)"),
        ("sample_call_7",  "Call 7 — Banking (Loan)"),
        ("sample_call_8",  "Call 8 — Utilities (Outage)"),
        ("sample_call_9",  "Call 9 — Hotel (Booking)"),
        ("sample_call_10", "Call 10 — Travel (Flight)"),
        ("sample_call_11", "Call 11 — Software (Billing)"),
        ("sample_call_12", "Call 12 — Automotive (Roadside)"),
        ("sample_call_13", "Call 13 — Healthcare (Prescription)"),
    ]

    st.markdown("**📁 Archive Recordings**")
    ARCHIVE_LABELS = [
        ("call_recording_01", "REC-01 — Product Inquiry (Neutral)"),
        ("call_recording_02", "REC-02 — Complaint (Angry)"),
        ("call_recording_03", "REC-03 — Technical Issue (Frustrated)"),
        ("call_recording_04", "REC-04 — Compliment (Happy)"),
        ("call_recording_05", "REC-05 — Order Placement (Neutral)"),
        ("call_recording_06", "REC-06 — Product Inquiry (Confused)"),
        ("call_recording_07", "REC-07 — Complaint (Angry)"),
        ("call_recording_08", "REC-08 — Technical Issue (Neutral)"),
        ("call_recording_09", "REC-09 — Order Placement (Happy)"),
        ("call_recording_10", "REC-10 — Compliment (Happy)"),
        ("call_recording_11", "REC-11 — Product Inquiry (Neutral)"),
        ("call_recording_12", "REC-12 — Complaint (Angry)"),
        ("call_recording_13", "REC-13 — Technical Issue (Frustrated)"),
        ("call_recording_14", "REC-14 — Order Placement (Neutral)"),
        ("call_recording_15", "REC-15 — Compliment (Happy)"),
        ("call_recording_16", "REC-16 — Product Inquiry (Neutral)"),
        ("call_recording_17", "REC-17 — Complaint (Angry)"),
        ("call_recording_18", "REC-18 — Technical Issue (Frustrated)"),
        ("call_recording_19", "REC-19 — Order Placement (Neutral)"),
        ("call_recording_20", "REC-20 — Compliment (Happy)"),
    ]

    ALL_SAMPLES = SAMPLE_LABELS + ARCHIVE_LABELS
    for sample_key, label in ALL_SAMPLES:
        if st.button(f"Load {label}", key=f"btn_{sample_key}"):
            sample = load_sample_transcript(sample_key)
            if sample:
                st.session_state.transcript_text = sample["transcript"]
                st.session_state.sample_metadata = {
                    "call_id": sample["call_id"],
                    "category": sample.get("category", ""),
                    "duration_seconds": sample.get("duration_seconds", 0),
                }
                # Clear old results so tabs don't show stale data
                st.session_state.call_result = None
                st.session_state.benchmark_result = None
                st.session_state.active_call_id = None

    # Cache Management
    st.subheader("💾 Cache")
    entries = list_cache_entries()
    if entries:
        st.caption(f"{len(entries)} cached result(s)")
        for e in entries:
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.caption(f"**{e['type']}** / {e['llm']}\n`{e['call_id']}` · {e['size_kb']} KB")
            with col_btn:
                if st.button("🗑️", key=f"del_{e['file']}", help=f"Delete {e['file']}"):
                    delete_cache_entry(e["file"])
                    st.rerun()
        st.markdown("")
        if st.button("🗑️ Clear All Cache"):
            n = clear_cache()
            st.success(f"Cleared {n} file(s)")
            st.rerun()
    else:
        st.info("No cached results yet")

    # Vector DB (RAG)
    st.subheader("🔍 Vector DB (RAG)")
    vdb = get_vector_store_stats()
    if vdb["available"]:
        st.success(f"✓ ChromaDB active — {vdb['count']} embedding(s) stored")
        if vdb["count"] > 0:
            st.caption("Similar past calls will be retrieved to enrich LLM prompts.")
        else:
            st.caption("Process a call to start building the RAG knowledge base.")
    else:
        st.warning("⚠ ChromaDB unavailable (install `chromadb`)")


def _derive_tags(summary) -> list:
    """Derive keyword tags from a SummaryOutput without extra API calls."""
    tags = []
    issue = (summary.customer_issue or "").lower()
    text = issue + " " + (summary.summary or "").lower()

    tag_map = {
        "billing": ["bill", "charge", "payment", "invoice", "fee", "overcharg", "refund"],
        "technical": ["technical", "error", "issue", "bug", "not working", "connectivity", "outage"],
        "refund": ["refund", "return", "money back", "reimburse"],
        "complaint": ["complaint", "unhappy", "dissatisfied", "frustrated", "angry"],
        "account": ["account", "login", "password", "access", "locked"],
        "shipping": ["ship", "delivery", "track", "package", "order"],
        "upgrade": ["upgrade", "plan", "tier", "premium"],
        "cancellation": ["cancel", "terminat", "close account"],
        "medical": ["medical", "health", "prescription", "medication", "doctor"],
        "fraud": ["fraud", "unauthorized", "stolen", "scam"],
        "insurance": ["claim", "insurance", "coverage", "deductible", "policy"],
        "travel": ["flight", "cancel", "delay", "booking", "reservation", "hotel"],
    }
    for tag, keywords in tag_map.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)

    # Always add resolution status as a tag
    tags.append(summary.resolution_status.value)

    return sorted(set(tags))


def _derive_highlights(summary) -> list:
    """Extract top highlights from a SummaryOutput."""
    highlights = []
    if summary.key_points:
        highlights.extend(summary.key_points[:3])
    if summary.action_items:
        highlights.append(f"Action: {summary.action_items[0]}")
    return highlights[:4]


# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📤 Upload", "📋 Results", "⭐ QA Score", "🔬 Benchmark", "🗺️ Workflow", "📜 Call History", "🏗️ Architecture"]
)

# TAB 1: Upload
with tab1:
    st.header("Upload Call Transcript")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Option 1: Paste Transcript")
        transcript_input = st.text_area(
            "Paste transcript text:",
            value=st.session_state.transcript_text,
            height=250,
            help="Paste call transcript with Agent: and Customer: labels",
        )

    with col2:
        st.subheader("Option 2: Upload File")
        audio_exts = [f[1:] for f in SUPPORTED_AUDIO_FORMATS]  # strip leading dot
        uploaded_file = st.file_uploader(
            "Upload transcript (JSON/TXT) or audio file",
            type=["json", "txt"] + audio_exts,
            help="Upload a transcript file (JSON/plain text) or audio (mp3/wav/m4a/webm) for Whisper transcription",
        )

        if uploaded_file:
            file_content = uploaded_file.read()
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix == ".json":
                try:
                    data = json.loads(file_content)
                    transcript_input = data.get("transcript", "")
                    st.success(f"✓ Loaded transcript: {data.get('call_id', uploaded_file.name)}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file")
                    transcript_input = ""
            elif suffix == ".txt":
                transcript_input = file_content.decode("utf-8")
                st.success(f"✓ Loaded text: {uploaded_file.name}")
            elif suffix in SUPPORTED_AUDIO_FORMATS:
                # Save audio to a temp file and transcribe via Whisper
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                try:
                    st.info(f"🎙️ Audio file detected ({uploaded_file.name}). Transcribing via Whisper...")
                    from agents.transcription_agent import TranscriptionAgent
                    ta = TranscriptionAgent()
                    t_out = ta.process(call_id="audio_upload", audio_path=tmp_path)
                    transcript_input = t_out.transcript
                    # Store in session state so we can show it in a side panel
                    st.session_state.audio_transcript = transcript_input
                    st.session_state.audio_filename = uploaded_file.name
                    st.success(f"✓ Transcribed {uploaded_file.name} — {len(transcript_input)} chars")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    transcript_input = ""
                    st.session_state.audio_transcript = None
                    st.session_state.audio_filename = None
                finally:
                    os.unlink(tmp_path)
            else:
                transcript_input = file_content.decode("utf-8", errors="replace")
                st.success(f"✓ Loaded: {uploaded_file.name}")

    # Audio Transcript Side Panel — shown only when an audio file was just transcribed
    if st.session_state.audio_transcript:
        st.markdown("---")
        st.subheader(f"🎙️ Whisper Transcript — `{st.session_state.audio_filename}`")
        st.caption("This is the raw text returned by OpenAI Whisper. It has been copied to the text area above and will be processed when you click **Process Call**.")
        with st.expander("📝 View full transcript", expanded=True):
            st.text_area(
                "Transcribed text:",
                value=st.session_state.audio_transcript,
                height=200,
                disabled=True,
                key="audio_transcript_display",
            )
        st.markdown("---")

    # Metadata
    st.subheader("Call Metadata (Optional)")
    col_meta1, col_meta2 = st.columns(2)

    with col_meta1:
        call_id = st.text_input(
            "Call ID:",
            value=st.session_state.get("sample_metadata", {}).get("call_id", ""),
            help="Leave blank to auto-generate",
        )
        category = st.text_input(
            "Category:",
            value=st.session_state.get("sample_metadata", {}).get("category", ""),
            help="e.g., banking, telecom, healthcare",
        )

    with col_meta2:
        duration = st.number_input(
            "Duration (seconds):",
            value=st.session_state.get("sample_metadata", {}).get("duration_seconds", 0),
            min_value=0,
        )

    # Process button
    if st.button("🚀 Process Call", type="primary", use_container_width=True):
        is_valid, val_err = validate_transcript_text(transcript_input)
        if not is_valid:
            st.error(f"Invalid input: {val_err}")
        else:
            transcript_input = sanitize_transcript(transcript_input)
            try:
                # ── Check cache first ──────────────────────────────────────────
                cached = get_cached(transcript_input, llm_choice, "workflow")
                if cached:
                    st.session_state.call_result = CallResult.model_validate(cached)
                    st.session_state.benchmark_result = None  # clear stale benchmark
                    st.session_state.active_call_id = st.session_state.call_result.call_id
                    st.session_state.active_llm = llm_choice
                    # Clear audio transcript since this came from cache
                    st.session_state.audio_transcript = None
                    st.session_state.audio_filename = None
                    st.success(
                        f"⚡ Loaded from cache — no LLM API call made! "
                        f"Call {st.session_state.call_result.call_id} · LLM: {llm_choice}"
                    )
                else:
                    with st.spinner("Processing call..."):
                        intake = IntakeAgent()
                        metadata = {}
                        if call_id:
                            metadata["call_id"] = call_id
                        if category:
                            metadata["category"] = category
                        if duration > 0:
                            metadata["duration_seconds"] = duration

                        call_input = intake.process(
                            transcript_text=transcript_input,
                            metadata=metadata,
                        )

                        workflow = create_workflow(llm_name=llm_choice)
                        call_result = run_workflow(workflow, call_input, llm_name=llm_choice)

                        # Save to cache with metadata tags
                        data = call_result.model_dump()
                        data["_cache_type"] = "workflow"
                        data["_llm_name"] = llm_choice
                        save_cache(transcript_input, llm_choice, "workflow", data)

                        st.session_state.call_result = call_result
                        st.session_state.benchmark_result = None
                        st.session_state.active_call_id = call_result.call_id
                        st.session_state.active_llm = llm_choice
                        # Capture V2 extras (PII, sentiment, RAG)
                        st.session_state.v2_extras = getattr(call_result, "_v2_extras", {})
                        st.success(f"✓ Call {call_result.call_id} processed and cached")

            except Exception as e:
                st.error(f"Error processing call: {str(e)}")
                logger.error(f"Processing error: {e}", exc_info=True)


# TAB 2: Results
with tab2:
    st.header("Call Analysis Results")

    if not st.session_state.call_result:
        st.info("👈 Upload and process a call in the Upload tab first")
    else:
        st.caption(f"🔵 Showing: **{st.session_state.active_call_id}** · LLM: **{st.session_state.active_llm}**")
        result = st.session_state.call_result

        # Call Info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Call ID", result.call_id)
        with col2:
            status = result.summary.resolution_status.value if result.summary else "N/A"
            st.metric("Resolution Status", status.upper())
        with col3:
            qa_score = f"{result.qa_score.overall_score:.1f}/100" if result.qa_score else "N/A"
            st.metric("QA Score", qa_score)
        with col4:
            step = result.current_step
            st.metric("Current Step", step)

        # Transcript
        if result.transcript:
            with st.expander("📝 Normalized Transcript", expanded=False):
                st.text(result.transcript.transcript)
                st.caption(f"Speakers: {', '.join(result.transcript.speakers)}")

        # Summary
        if result.summary:
            st.subheader("📊 Summary")
            st.write(result.summary.summary)

            col_summary1, col_summary2 = st.columns(2)

            with col_summary1:
                st.write("**Key Points:**")
                for point in result.summary.key_points:
                    st.write(f"• {point}")

            with col_summary2:
                st.write("**Action Items:**")
                for item in result.summary.action_items:
                    st.write(f"• {item}")

            st.write(f"**Customer Issue:** {result.summary.customer_issue}")

        # Tags & Highlights
        if result.summary:
            st.subheader("🏷️ Tags & Highlights")
            tag_col, highlight_col = st.columns(2)

            with tag_col:
                st.markdown("**Tags**")
                tags = _derive_tags(result.summary)
                if tags:
                    st.markdown(" ".join(f"`{t}`" for t in tags))
                else:
                    st.caption("No tags extracted")

            with highlight_col:
                st.markdown("**Highlights**")
                highlights = _derive_highlights(result.summary)
                for h in highlights:
                    st.info(f"💡 {h}")

        # ── V2: Sentiment Analysis ────────────────────────────────────────────
        sentiment = st.session_state.v2_extras.get("sentiment")
        if sentiment and sentiment.get("overall_customer_sentiment", "unknown") != "unknown":
            st.subheader("🎭 Sentiment Analysis")
            sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)
            risk = sentiment.get("escalation_risk", "unknown")
            risk_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk, "⚪")
            trend_icon = {"improving": "📈", "stable": "➡️", "degrading": "📉"}.get(
                sentiment.get("customer_sentiment_trend", ""), "")
            with sent_col1:
                st.metric("Customer Sentiment", sentiment.get("overall_customer_sentiment", "N/A").title())
            with sent_col2:
                st.metric("Sentiment Trend", f"{trend_icon} {sentiment.get('customer_sentiment_trend', 'N/A').title()}")
            with sent_col3:
                st.metric("Escalation Risk", f"{risk_color} {risk.title()}")
            with sent_col4:
                st.metric("Agent Tone", "See below")
            st.caption(f"**Agent tone:** {sentiment.get('agent_tone', 'N/A')}")
            if sentiment.get("escalation_risk_reason"):
                st.caption(f"**Risk reason:** {sentiment['escalation_risk_reason']}")

            # Turn-by-turn sentiment chart
            turns = sentiment.get("turns", [])
            if turns:
                import pandas as pd
                df_turns = pd.DataFrame(turns)
                if "score" in df_turns.columns and "speaker" in df_turns.columns:
                    df_turns["turn"] = range(1, len(df_turns) + 1)
                    with st.expander("📊 Turn-by-Turn Sentiment Chart", expanded=False):
                        st.bar_chart(df_turns.set_index("turn")[["score"]])
                        st.caption("Score: -1.0 = very negative → 1.0 = very positive")

        # ── V2: PII Redaction Summary ─────────────────────────────────────────
        pii_summary = st.session_state.v2_extras.get("pii_summary", {})
        rag_ctx = st.session_state.v2_extras.get("rag_context", "")
        if pii_summary or rag_ctx:
            with st.expander("🔒 V2 Processing Details", expanded=False):
                if pii_summary:
                    total_pii = sum(pii_summary.values())
                    st.success(f"🔒 PII Redaction: {total_pii} item(s) masked before LLM processing")
                    for field, count in pii_summary.items():
                        st.caption(f"  • {field.replace('_', ' ').title()}: {count}")
                else:
                    st.info("🔒 PII Redaction: No PII detected in transcript")
                if rag_ctx:
                    st.success(f"🔍 RAG: Similar past calls retrieved ({len(rag_ctx)} chars of context injected)")
                else:
                    st.info("🔍 RAG: No similar past calls yet (vector store building up)")

        # Errors
        if result.errors:
            with st.expander("⚠️ Errors", expanded=False):
                for error in result.errors:
                    st.error(error)


# TAB 3: QA Score
with tab3:
    st.header("Quality Assurance Score")

    if not st.session_state.call_result or not st.session_state.call_result.qa_score:
        st.info("👈 Process a call first to see QA scores")
    else:
        st.caption(f"🔵 Showing: **{st.session_state.active_call_id}** · LLM: **{st.session_state.active_llm}**")
        qa = st.session_state.call_result.qa_score

        # Overall Score
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Overall Score", f"{qa.overall_score:.1f}/100")
        with col2:
            # Progress bar representation
            percentage = qa.overall_score / 100
            color = "🟢" if percentage >= 0.8 else "🟡" if percentage >= 0.6 else "🔴"
            st.write(f"{color} {percentage*100:.0f}%")

        st.markdown("---")

        # Dimension Scores
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Empathy", f"{qa.empathy_score:.1f}/25")
            st.progress(qa.empathy_score / 25)

        with col2:
            st.metric("Professionalism", f"{qa.professionalism_score:.1f}/25")
            st.progress(qa.professionalism_score / 25)

        with col3:
            st.metric("Resolution", f"{qa.resolution_score:.1f}/25")
            st.progress(qa.resolution_score / 25)

        with col4:
            st.metric("Compliance", f"{qa.compliance_score:.1f}/25")
            st.progress(qa.compliance_score / 25)

        st.markdown("---")

        # Tone and Feedback
        col_feedback1, col_feedback2 = st.columns(2)

        with col_feedback1:
            st.write("**Tone:**")
            st.info(qa.tone)

            st.write("**Strengths:**")
            for strength in qa.strengths:
                st.success(f"✓ {strength}")

        with col_feedback2:
            st.write("**Improvements:**")
            for improvement in qa.improvements:
                st.warning(f"→ {improvement}")


# TAB 4: Benchmark
with tab4:
    st.header("Multi-LLM Benchmark")

    if not st.session_state.call_result or not st.session_state.call_result.transcript:
        st.info("👈 Process a call first to run benchmarks")
    else:
        st.caption(f"🔵 Benchmarking: **{st.session_state.active_call_id}**")
        st.write(
            "Compare Claude, GPT-4, and Gemini performance on the same transcript."
        )

        benchmark_type = st.radio(
            "Select benchmark type:",
            options=["Summarization", "QA Scoring", "Full Benchmark"],
        )

        if st.button("🔬 Run Benchmark", type="primary", use_container_width=True):
            try:
                import concurrent.futures as _cf
                transcript = st.session_state.call_result.transcript.transcript
                call_id_bm = st.session_state.call_result.call_id
                bm_prefix = f"benchmark_{benchmark_type.lower().replace(' ', '_')}"
                models = ["claude", "gpt4", "gemini"]

                # ── Helpers to detect agent fallback responses (API failure) ────
                def _is_bad_summary(s: dict) -> bool:
                    return (
                        s.get("summary") in ("Unable to summarize transcript", "", None)
                        or (not s.get("key_points") and not s.get("action_items")
                            and s.get("customer_issue") in ("Unable to determine", "", None))
                    )

                def _is_bad_qa(q: dict) -> bool:
                    return q.get("overall_score", -1) == 0.0 and not q.get("strengths") and not q.get("improvements")

                # ── Load per-model cache (skip stale/fallback entries) ───────────
                result = BenchmarkResult(call_id=call_id_bm)
                models_to_run = []
                cached_models = []

                # Also check old whole-benchmark cache as a migration fallback
                legacy_cache = get_cached(transcript, "all", bm_prefix)

                for model in models:
                    cache_key = f"{bm_prefix}_{model}"
                    cached_m = get_cached(transcript, model, cache_key)

                    # Fall back to legacy whole-cache if per-model entry is missing
                    if not cached_m and legacy_cache:
                        raw_sum = legacy_cache.get(f"{model}_summary")
                        raw_qa  = legacy_cache.get(f"{model}_qa")
                        if raw_sum or raw_qa:
                            cached_m = {
                                "summary": raw_sum,
                                "qa": raw_qa,
                                "timing": legacy_cache.get("timing", {}).get(model),
                                "token_counts": legacy_cache.get("token_counts", {}).get(model),
                            }

                    # Validate cached data is real (not an agent fallback response)
                    has_good_summary = (
                        cached_m and cached_m.get("summary")
                        and not _is_bad_summary(cached_m["summary"])
                    )
                    has_good_qa = (
                        cached_m and cached_m.get("qa")
                        and not _is_bad_qa(cached_m["qa"])
                    )

                    # Decide if this model needs a real API call
                    needs_summary = benchmark_type in ("Summarization", "Full Benchmark")
                    needs_qa      = benchmark_type in ("QA Scoring", "Full Benchmark")

                    if (needs_summary and not has_good_summary) or (needs_qa and not has_good_qa):
                        models_to_run.append(model)
                        continue

                    # Restore good data from cache
                    if has_good_summary:
                        setattr(result, f"{model}_summary",
                                SummaryOutput.model_validate(cached_m["summary"]))
                    if has_good_qa:
                        setattr(result, f"{model}_qa",
                                QAScore.model_validate(cached_m["qa"]))
                    if cached_m.get("timing") is not None:
                        result.timing[model] = cached_m["timing"]
                    if cached_m.get("token_counts"):
                        result.token_counts[model] = cached_m["token_counts"]
                    cached_models.append(model)

                if cached_models:
                    st.info(f"⚡ Loaded from cache: {', '.join(cached_models)}")

                # ── Run API only for uncached / previously-failed models ─────────
                if models_to_run:
                    with st.spinner(
                        f"Calling API for: {', '.join(models_to_run)} …"
                    ):
                        benchmark = BenchmarkRunner()

                        if benchmark_type == "Summarization":
                            _fn = benchmark._run_summarization_for_model
                        elif benchmark_type == "QA Scoring":
                            _fn = benchmark._run_qa_for_model
                        else:
                            _fn = benchmark._run_full_for_model

                        with _cf.ThreadPoolExecutor(max_workers=len(models_to_run)) as executor:
                            futures = {
                                executor.submit(_fn, m, call_id_bm, transcript): m
                                for m in models_to_run
                            }
                            for future in _cf.as_completed(futures):
                                model = futures[future]
                                cache_key = f"{bm_prefix}_{model}"
                                try:
                                    if benchmark_type == "Summarization":
                                        summary, timing, tc = future.result()
                                        if _is_bad_summary(summary.model_dump()):
                                            result.errors[model] = "API returned fallback — quota may be exhausted. Try again later."
                                        else:
                                            setattr(result, f"{model}_summary", summary)
                                            result.timing[model] = timing
                                            result.token_counts[model] = tc
                                            save_cache(transcript, model, cache_key, {
                                                "summary": summary.model_dump(),
                                                "timing": timing, "token_counts": tc,
                                                "call_id": call_id_bm,
                                                "_cache_type": cache_key, "_llm_name": model,
                                            })
                                    elif benchmark_type == "QA Scoring":
                                        qa, timing, tc = future.result()
                                        if _is_bad_qa(qa.model_dump()):
                                            result.errors[model] = "API returned fallback — quota may be exhausted. Try again later."
                                        else:
                                            setattr(result, f"{model}_qa", qa)
                                            result.timing[model] = timing
                                            result.token_counts[model] = tc
                                            save_cache(transcript, model, cache_key, {
                                                "qa": qa.model_dump(),
                                                "timing": timing, "token_counts": tc,
                                                "call_id": call_id_bm,
                                                "_cache_type": cache_key, "_llm_name": model,
                                            })
                                    else:
                                        summary, qa, timing, tc = future.result()
                                        bad_s = summary and _is_bad_summary(summary.model_dump())
                                        bad_q = qa and _is_bad_qa(qa.model_dump())
                                        if bad_s or bad_q:
                                            result.errors[model] = "API returned fallback — quota may be exhausted. Try again later."
                                        else:
                                            if summary:
                                                setattr(result, f"{model}_summary", summary)
                                            if qa:
                                                setattr(result, f"{model}_qa", qa)
                                            result.timing[model] = timing
                                            result.token_counts[model] = tc
                                            save_cache(transcript, model, cache_key, {
                                                "summary": summary.model_dump() if summary else None,
                                                "qa": qa.model_dump() if qa else None,
                                                "timing": timing, "token_counts": tc,
                                                "call_id": call_id_bm,
                                                "_cache_type": cache_key, "_llm_name": model,
                                            })
                                except Exception as e:
                                    result.errors[model] = str(e)
                                    logger.error(f"Benchmark failed for {model}: {e}")

                    newly_cached = [m for m in models_to_run if m not in result.errors]
                    if newly_cached:
                        st.success(f"✓ Completed and cached: {', '.join(newly_cached)}")
                    if result.errors:
                        st.warning(
                            f"⚠️ These models need a retry (quota or API issue): "
                            f"{', '.join(result.errors.keys())}. "
                            f"Click **Run Benchmark** again once the API is available."
                        )
                else:
                    st.success("⚡ All models loaded from cache — no API calls made")

                st.session_state.benchmark_result = result

            except Exception as e:
                st.error(f"Benchmark failed: {str(e)}")
                logger.error(f"Benchmark error: {e}", exc_info=True)

        # Display Results
        if st.session_state.benchmark_result:
            result = st.session_state.benchmark_result

            st.markdown("---")
            st.subheader("📊 Results")

            def _badge(v: str) -> str:
                return "🟢" if v == "resolved" else "🟡" if v == "escalated" else "🔴"

            def _safe(t: str) -> str:
                """Escape $ to prevent Streamlit LaTeX rendering."""
                return (t or "").replace("$", r"\$")

            # ── Timing row (always populate all 3 columns) ────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱ Claude", f"{result.timing['claude']:.2f}s" if "claude" in result.timing else "—")
            with col2:
                st.metric("⏱ GPT-4", f"{result.timing['gpt4']:.2f}s" if "gpt4" in result.timing else "—")
            with col3:
                st.metric("⏱ Gemini", f"{result.timing['gemini']:.2f}s" if "gemini" in result.timing else "—")

            st.markdown("---")

            # ── Summary Comparison ────────────────────────────────────────────
            if result.claude_summary or result.gpt4_summary or result.gemini_summary or result.errors:
                st.subheader("Summary Comparison")
                col1, col2, col3 = st.columns(3)

                with col1:
                    with st.container(border=True):
                        st.markdown("**🤖 Claude**")
                        if result.claude_summary:
                            s = result.claude_summary
                            st.markdown(_safe(s.summary))
                            st.caption(f"{_badge(s.resolution_status.value)} {s.resolution_status.value.upper()}")
                            if s.key_points:
                                st.markdown("**Key Points:**")
                                for kp in s.key_points:
                                    st.markdown(f"- {_safe(kp)}")
                        elif "claude" in result.errors:
                            st.error(result.errors["claude"][:200])
                        else:
                            st.info("Not run")

                with col2:
                    with st.container(border=True):
                        st.markdown("**🤖 GPT-4**")
                        if result.gpt4_summary:
                            s = result.gpt4_summary
                            st.markdown(_safe(s.summary))
                            st.caption(f"{_badge(s.resolution_status.value)} {s.resolution_status.value.upper()}")
                            if s.key_points:
                                st.markdown("**Key Points:**")
                                for kp in s.key_points:
                                    st.markdown(f"- {_safe(kp)}")
                        elif "gpt4" in result.errors:
                            st.error(result.errors["gpt4"][:200])
                        else:
                            st.info("Not run")

                with col3:
                    with st.container(border=True):
                        st.markdown("**🤖 Gemini**")
                        if result.gemini_summary:
                            s = result.gemini_summary
                            st.markdown(_safe(s.summary))
                            st.caption(f"{_badge(s.resolution_status.value)} {s.resolution_status.value.upper()}")
                            if s.key_points:
                                st.markdown("**Key Points:**")
                                for kp in s.key_points:
                                    st.markdown(f"- {_safe(kp)}")
                        elif "gemini" in result.errors:
                            st.error(result.errors["gemini"][:200])
                        else:
                            st.info("Not run")

            # ── QA Score Comparison ───────────────────────────────────────────
            if result.claude_qa or result.gpt4_qa or result.gemini_qa or result.errors:
                st.subheader("QA Score Comparison")
                col1, col2, col3 = st.columns(3)

                with col1:
                    with st.container(border=True):
                        st.markdown("**🤖 Claude**")
                        if result.claude_qa:
                            q = result.claude_qa
                            st.metric("Overall", f"{q.overall_score:.1f}/100")
                            st.progress(q.overall_score / 100)
                            st.caption(
                                f"Empathy {q.empathy_score:.0f} · "
                                f"Prof {q.professionalism_score:.0f} · "
                                f"Res {q.resolution_score:.0f} · "
                                f"Comp {q.compliance_score:.0f}"
                            )
                        elif "claude" in result.errors:
                            st.error(result.errors["claude"][:200])
                        else:
                            st.info("Not run")

                with col2:
                    with st.container(border=True):
                        st.markdown("**🤖 GPT-4**")
                        if result.gpt4_qa:
                            q = result.gpt4_qa
                            st.metric("Overall", f"{q.overall_score:.1f}/100")
                            st.progress(q.overall_score / 100)
                            st.caption(
                                f"Empathy {q.empathy_score:.0f} · "
                                f"Prof {q.professionalism_score:.0f} · "
                                f"Res {q.resolution_score:.0f} · "
                                f"Comp {q.compliance_score:.0f}"
                            )
                        elif "gpt4" in result.errors:
                            st.error(result.errors["gpt4"][:200])
                        else:
                            st.info("Not run")

                with col3:
                    with st.container(border=True):
                        st.markdown("**🤖 Gemini**")
                        if result.gemini_qa:
                            q = result.gemini_qa
                            st.metric("Overall", f"{q.overall_score:.1f}/100")
                            st.progress(q.overall_score / 100)
                            st.caption(
                                f"Empathy {q.empathy_score:.0f} · "
                                f"Prof {q.professionalism_score:.0f} · "
                                f"Res {q.resolution_score:.0f} · "
                                f"Comp {q.compliance_score:.0f}"
                            )
                        elif "gemini" in result.errors:
                            st.error(result.errors["gemini"][:200])
                        else:
                            st.info("Not run")


# TAB 5: Workflow Visualization
with tab5:
    st.header("🗺️ LangGraph Workflow")

    # ── SECTION 1: Graph Diagram + Side Explanation ───────────────────────────
    st.subheader("📊 Graph Diagram")
    st.caption(
        "📌 **LangGraph workflow** — matches the compiled graph in `workflow/langgraph_flow.py`. "
        "Solid arrows = fixed edges. Dashed orange arrows = conditional routing."
    )
    diag_col, exp_col = st.columns([1.2, 1], gap="large")
    with exp_col:
        st.markdown("### How to read this diagram")
        st.markdown("""
**Node colours**
| Colour | Meaning |
|--------|---------|
| 🟢 Green oval | Entry / Exit point |
| 🔵 Blue box | Processing agent (LLM or logic) |
| 🟠 Orange box | Error recovery handler |

---
**Arrow types**
- **Solid arrow** `→` — always taken, no condition
- **Dashed arrow** `-->` — conditional, depends on state

---
**Step-by-step walk-through**

1. **START** kicks off the pipeline with the user's input.
2. **intake** validates the transcript and assigns a `call_id`.
3. **transcription** normalises speaker labels to `Agent:` / `Customer:`.
4. *Conditional split:*
   - No errors → **rag_retrieval** queries ChromaDB for similar past calls
   - Errors detected → **error_handler** attempts recovery, then rejoins
5. **rag_retrieval** embeds the transcript and retrieves the top-3 semantically similar past calls from ChromaDB to form a RAG context block.
6. **summarization** calls the selected LLM with the transcript + RAG context injected into the prompt.
7. **quality_score** calls the LLM again with the same RAG context to calibrate scores against comparable past calls.
8. **end** packages everything into a `CallResult` and stores the call's embedding in ChromaDB for future RAG retrieval.
9. **END** returns control to the Streamlit UI.

---
**error_handler recovery logic**

If transcription fails the handler checks what is still missing and skips ahead to the earliest incomplete step — so no work is ever duplicated.
        """)

    with diag_col:
        # Static graphviz DOT diagram — always renders on Streamlit Cloud without extra dependencies
        _dot_source = """
digraph LangGraph {
    rankdir=TD
    bgcolor=white
    fontname=Helvetica
    node [fontname=Helvetica fontsize=11]

    __start__     [label="START" shape=oval style=filled fillcolor="#c8e6c9" color="#4caf50"]
    intake        [label="intake\n(IntakeAgent)"               shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    transcription [label="transcription\n(TranscriptionAgent)" shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    error_handler [label="error_handler\n(RoutingAgent)"       shape=box style=filled fillcolor="#ffe0b2" color="#f57c00"]
    pii_redaction [label="pii_redaction\n(PIIRedactionAgent)"  shape=box style=filled fillcolor="#fce4ec" color="#c62828"]
    rag_retrieval [label="rag_retrieval\n(RAGAgent + ChromaDB)" shape=box style=filled fillcolor="#e1bee7" color="#7b1fa2"]
    sentiment     [label="sentiment\n(SentimentAgent)"          shape=box style=filled fillcolor="#fff9c4" color="#f9a825"]
    summarization [label="summarization\n(SummarizationAgent)" shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    quality_score [label="quality_score\n(QualityScoreAgent)"  shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    end_node      [label="end\n(store ChromaDB embedding)"     shape=box style=filled fillcolor="#c8e6c9" color="#4caf50"]
    __end__       [label="END" shape=oval style=filled fillcolor="#c8e6c9" color="#4caf50"]

    __start__     -> intake          [color="#333333"]
    intake        -> transcription   [color="#333333"]
    transcription -> pii_redaction   [color="#333333" label="errors == []"]
    transcription -> error_handler   [color="#f57c00" style=dashed label="errors != []"]
    pii_redaction -> rag_retrieval   [color="#c62828" label="redacted transcript"]
    rag_retrieval -> sentiment       [color="#7b1fa2" label="+ RAG context"]
    sentiment     -> summarization   [color="#f9a825" label="+ sentiment data"]
    error_handler -> summarization   [color="#f57c00" style=dashed label="summary missing"]
    error_handler -> quality_score   [color="#f57c00" style=dashed label="qa missing"]
    error_handler -> end_node        [color="#f57c00" style=dashed label="both present"]
    summarization -> quality_score   [color="#333333"]
    quality_score -> end_node        [color="#333333"]
    end_node      -> __end__         [color="#333333"]
}
"""
        st.graphviz_chart(_dot_source)

    st.markdown("---")

    # ── SECTION 2: Workflow Architecture (static reference) ────────────────────
    st.subheader("📐 Workflow Architecture")
    st.caption("📌 **Static reference** — this table describes the fixed graph structure defined in `workflow/langgraph_flow.py`.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nodes (Agents) — Version 2**")
        st.markdown("""
| Node | Agent | Role |
|------|-------|------|
| `intake` | IntakeAgent | Validates input, generates call_id |
| `transcription` | TranscriptionAgent | Normalizes speaker labels / runs Whisper |
| `pii_redaction` | PIIRedactionAgent | Masks PII before any LLM sees the text |
| `rag_retrieval` | RAGRetrievalAgent | Queries ChromaDB for similar past calls |
| `sentiment` | SentimentAgent | Per-turn sentiment + escalation risk |
| `summarization` | SummarizationAgent | LLM → summary + key points (+ RAG) |
| `quality_score` | QualityScoreAgent | LLM → 4-dimension scores (+ RAG) |
| `error_handler` | RoutingAgent | Catches errors, decides recovery path |
| `end` | — | Packages CallResult + stores ChromaDB embedding |
        """)
    with col2:
        st.markdown("**Edges**")
        st.markdown("""
| From | To | Type | Condition |
|------|----|------|-----------|
| START | `intake` | fixed | always |
| `intake` | `transcription` | fixed | always |
| `transcription` | `pii_redaction` | **conditional** | `errors == []` |
| `transcription` | `error_handler` | **conditional** | `errors != []` |
| `pii_redaction` | `rag_retrieval` | fixed | always |
| `rag_retrieval` | `sentiment` | fixed | always |
| `sentiment` | `summarization` | fixed | always |
| `summarization` | `quality_score` | fixed | always |
| `quality_score` | `end` | fixed | always |
| `error_handler` | `summarization` | **conditional** | summary missing |
| `error_handler` | `quality_score` | **conditional** | qa_score missing |
| `error_handler` | `end` | **conditional** | both present |
| `end` | END | fixed | always |
        """)

    st.markdown("---")

    # ── SECTION 3: Shared State — schema (static) + actual values (dynamic) ────
    st.subheader("🗂️ Shared State (WorkflowState)")

    state_col1, state_col2 = st.columns(2)

    with state_col1:
        st.caption("📌 **Static** — schema defined in `workflow/langgraph_flow.py`")
        st.code("""class WorkflowState(TypedDict):
    call_id:      str             # unique call identifier
    input_data:   CallInput       # raw transcript + metadata
    transcript:   TranscriptOutput# normalized text + speakers
    rag_context:  str             # similar past calls from ChromaDB
    summary:      SummaryOutput   # summary, key_points, action_items
    qa_score:     QAScore         # empathy/prof/resolution/compliance
    errors:       list[str]       # drives conditional routing
    current_step: str             # last node that ran""", language="python")

    with state_col2:
        st.caption("✅ **Dynamic** — actual values from last processed call")
        if not st.session_state.call_result:
            st.info("Process a call in the Upload tab to see live state values here.")
        else:
            r = st.session_state.call_result
            st.markdown(f"**`call_id`** → `{r.call_id}`")
            st.markdown(f"**`current_step`** → `{r.current_step}`")
            st.markdown(f"**`errors`** → `{r.errors if r.errors else '[]  ← no errors'}`")

            st.markdown("**`input_data`** →")
            st.json({
                "call_id": r.input_data.call_id,
                "category": r.input_data.metadata.get("category", "N/A"),
                "duration_seconds": r.input_data.metadata.get("duration_seconds", "N/A"),
                "transcript_preview": (r.input_data.transcript_text or "")[:120] + "...",
            })

            if r.transcript:
                st.markdown("**`transcript`** →")
                st.json({
                    "speakers": r.transcript.speakers,
                    "transcript_preview": r.transcript.transcript[:120] + "...",
                })

            if r.summary:
                st.markdown("**`summary`** →")
                st.json({
                    "summary": r.summary.summary[:150] + "...",
                    "resolution_status": r.summary.resolution_status.value,
                    "key_points_count": len(r.summary.key_points),
                    "action_items_count": len(r.summary.action_items),
                })

            if r.qa_score:
                st.markdown("**`qa_score`** →")
                st.json({
                    "overall_score": r.qa_score.overall_score,
                    "empathy": r.qa_score.empathy_score,
                    "professionalism": r.qa_score.professionalism_score,
                    "resolution": r.qa_score.resolution_score,
                    "compliance": r.qa_score.compliance_score,
                })

    st.markdown("---")

    # ── SECTION 4: Routing decisions for last call (dynamic) ───────────────────
    st.subheader("🔀 Routing Decisions (Last Call)")
    st.caption("✅ **Dynamic** — inferred from the actual state of the last processed call.")

    if not st.session_state.call_result:
        st.info("Process a call to see which routing path was taken.")
    else:
        r = st.session_state.call_result
        had_errors = bool(r.errors)

        # Node-by-node trace
        steps = [
            ("START", "→ intake", "fixed", "green"),
            ("intake", "→ transcription", "fixed (always)", "green"),
            ("transcription",
             "→ **error_handler** ⚠️" if had_errors else "→ **summarization** ✅",
             f"conditional — `errors={'non-empty' if had_errors else '[]'}`",
             "red" if had_errors else "green"),
        ]
        if had_errors:
            steps.append(("error_handler",
                          "→ summarization (recovery)",
                          "conditional — summary was missing",
                          "orange"))
        steps += [
            ("summarization", "→ quality_score", "fixed", "green"),
            ("quality_score", "→ end", "fixed", "green"),
            ("end", "→ END", "fixed", "green"),
        ]

        for from_node, to_node, condition, color in steps:
            icon = "🔴" if color == "red" else "🟠" if color == "orange" else "🟢"
            st.markdown(f"{icon} **`{from_node}`** {to_node} &nbsp;&nbsp; _(condition: {condition})_")

        if had_errors:
            st.error(f"⚠️ Errors encountered: {r.errors}")
        else:
            st.success("✅ Happy path — no errors, no error_handler triggered.")

    st.markdown("---")

    # ── SECTION 5: Router Agent logic explained ────────────────────────────────
    st.subheader("🧠 How the Router Agent Works")
    st.caption("📌 **Static** — explains the conditional edge logic in `workflow/langgraph_flow.py` and `agents/routing_agent.py`.")

    st.markdown("""
LangGraph routing is controlled by **conditional edges** — Python lambda functions that
inspect the shared state and return which node to go to next.

There are **two conditional branch points** in this workflow:
    """)

    st.markdown("#### Branch 1 — After `transcription`")
    st.code("""# In langgraph_flow.py → create_workflow()
workflow.add_conditional_edges(
    "transcription",                                          # source node
    lambda state: "error_handler" if state["errors"]         # routing function
                  else "summarization",
    {"error_handler": "error_handler",                        # mapping: return value → node name
     "summarization": "summarization"},
)

# What triggers this:
#   state["errors"] is populated if TranscriptionAgent raises an exception.
#   If errors = []  → go to summarization  (happy path)
#   If errors != [] → go to error_handler  (recovery path)""", language="python")

    st.markdown("#### Branch 2 — After `error_handler`")
    st.code("""# In langgraph_flow.py → create_workflow()
workflow.add_conditional_edges(
    "error_handler",
    lambda state: "summarization"  if not state["summary"]   # still need summary?
             else "quality_score"  if not state["qa_score"]  # still need QA?
             else "end",                                      # both done, finish
    {"summarization": "summarization",
     "quality_score": "quality_score",
     "end": "end"},
)

# Logic: error_handler checks what's missing and skips ahead to the
# earliest incomplete step, so no work is duplicated after recovery.""", language="python")

    st.markdown("#### RoutingAgent — helper class in `agents/routing_agent.py`")
    st.markdown("""
The `RoutingAgent` class provides **helper methods** that mirror the same logic, but are used
for logging and validation — **not** for the actual LangGraph routing (those are the lambdas above).

| Method | Purpose |
|--------|---------|
| `route_intake()` | Logs the intake → transcription decision |
| `route_transcription()` | Checks errors, returns `"error_handler"` or `"summarization"` |
| `route_summarization()` | Checks summary, returns `"quality_score"` |
| `route_quality_score()` | Always returns `"end"` |
| `handle_error()` | Checks missing outputs, returns recovery target |
| `log_state_transition()` | LangSmith-compatible transition logging |
| `validate_state()` | Guards against missing `call_id` / `input_data` |

> **Key insight:** The LangGraph graph edges (the lambdas) are what actually control routing.
> The `RoutingAgent` methods are helpers for logging/testing that express the same logic in an OOP style.
    """)


# TAB 6: Call History (Memory Layer)
with tab6:
    st.header("📜 Call History")
    st.caption("Powered by the Memory Layer — persists processed calls across sessions.")

    stats = call_memory.get_stats()
    if stats["total_calls"] == 0:
        st.info("No calls in history yet. Process a call in the Upload tab to start building history.")
    else:
        # Stats row
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            st.metric("Total Calls", stats["total_calls"])
        with s_col2:
            st.metric("Avg QA Score", f"{stats['avg_qa_score']}/100" if stats.get("avg_qa_score") else "N/A")
        with s_col3:
            resolved = stats.get("resolution_breakdown", {}).get("resolved", 0)
            st.metric("Resolved", resolved)
        with s_col4:
            escalated = stats.get("resolution_breakdown", {}).get("escalated", 0)
            st.metric("Escalated", escalated)

        st.markdown("---")

        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            resolution_filter = st.selectbox(
                "Filter by Resolution:",
                ["All", "resolved", "unresolved", "escalated"],
            )
        with filter_col2:
            search_category = st.text_input("Search by Category:", placeholder="e.g. billing, healthcare")

        # Load entries
        if search_category.strip():
            entries = call_memory.search_by_category(search_category.strip())
        elif resolution_filter != "All":
            entries = call_memory.search_by_resolution(resolution_filter)
        else:
            entries = call_memory.get_recent_calls(n=50)

        st.caption(f"Showing {len(entries)} call(s)")

        for entry in entries:
            badge = "🟢" if entry.get("resolution_status") == "resolved" else \
                    "🟡" if entry.get("resolution_status") == "escalated" else "🔴"
            score_str = f"{entry['overall_score']:.1f}/100" if entry.get("overall_score") is not None else "N/A"
            with st.expander(
                f"{badge} {entry.get('call_id', 'Unknown')} — "
                f"{entry.get('category', 'unknown')} — "
                f"QA: {score_str} — "
                f"{entry.get('stored_at', '')[:10]}"
            ):
                if entry.get("summary_text"):
                    st.markdown(f"**Summary:** {entry['summary_text']}")
                if entry.get("customer_issue"):
                    st.markdown(f"**Issue:** {entry['customer_issue']}")
                if entry.get("action_items"):
                    st.markdown("**Action Items:**")
                    for item in entry["action_items"]:
                        st.markdown(f"- {item}")
                if entry.get("errors"):
                    st.warning(f"Errors: {entry['errors']}")
                st.caption(f"LLM: {entry.get('llm_name', 'unknown')} | Stored: {entry.get('stored_at', '')}")

        st.markdown("---")
        if st.button("🗑️ Clear All History", type="secondary"):
            n = call_memory.clear_history()
            st.success(f"Cleared {n} history entries")
            st.rerun()

        # Category breakdown chart
        if stats.get("category_breakdown"):
            st.subheader("📊 Calls by Category")
            cat_data = stats["category_breakdown"]
            import pandas as pd
            df = pd.DataFrame(list(cat_data.items()), columns=["Category", "Count"])
            df = df.sort_values("Count", ascending=False)
            st.bar_chart(df.set_index("Category"))


# TAB 7: Architecture
with tab7:
    st.header("🏗️ System Architecture")
    st.caption("Version 2 — Production-ready multi-agent call center AI with RAG, PII redaction, and sentiment analysis.")

    # ── Overview Diagram ───────────────────────────────────────────────────────
    st.subheader("📐 Full System Diagram")
    _arch_dot = """
digraph Architecture {
    rankdir=LR
    bgcolor=white
    fontname=Helvetica
    node [fontname=Helvetica fontsize=11 style=filled]

    subgraph cluster_ui {
        label="Streamlit UI"
        color="#1976d2"
        fontcolor="#1976d2"
        style=dashed
        ui [label="Web Browser\n(7 Tabs)" shape=box fillcolor="#e3f2fd" color="#1976d2"]
    }

    subgraph cluster_workflow {
        label="LangGraph Workflow (9 Nodes)"
        color="#4caf50"
        fontcolor="#4caf50"
        style=dashed
        intake      [label="intake"        fillcolor="#bbdefb" color="#1976d2" shape=box]
        transcr     [label="transcription" fillcolor="#bbdefb" color="#1976d2" shape=box]
        pii         [label="pii_redaction" fillcolor="#fce4ec" color="#c62828" shape=box]
        rag         [label="rag_retrieval" fillcolor="#e1bee7" color="#7b1fa2" shape=box]
        sent        [label="sentiment"     fillcolor="#fff9c4" color="#f9a825" shape=box]
        summ        [label="summarization" fillcolor="#bbdefb" color="#1976d2" shape=box]
        qa          [label="quality_score" fillcolor="#bbdefb" color="#1976d2" shape=box]
        end_n       [label="end"           fillcolor="#c8e6c9" color="#4caf50" shape=box]
        errh        [label="error_handler" fillcolor="#ffe0b2" color="#f57c00" shape=box]
    }

    subgraph cluster_llm {
        label="LLM Layer"
        color="#9c27b0"
        fontcolor="#9c27b0"
        style=dashed
        claude  [label="Claude\nSonnet 4.5"   fillcolor="#f3e5f5" color="#9c27b0" shape=box]
        gpt4    [label="GPT-4o"               fillcolor="#f3e5f5" color="#9c27b0" shape=box]
        gemini  [label="Gemini 2.5\nFlash"    fillcolor="#f3e5f5" color="#9c27b0" shape=box]
        whisper [label="Whisper\n(audio STT)" fillcolor="#f3e5f5" color="#9c27b0" shape=box]
    }

    subgraph cluster_stores {
        label="Data Stores"
        color="#ff6f00"
        fontcolor="#ff6f00"
        style=dashed
        cache   [label="SHA-256 Cache\n(file-based)" fillcolor="#fff3e0" color="#ff6f00" shape=cylinder]
        chroma  [label="ChromaDB\n(vector store)"    fillcolor="#fff3e0" color="#ff6f00" shape=cylinder]
        jsonl   [label="Call History\n(JSONL)"        fillcolor="#fff3e0" color="#ff6f00" shape=cylinder]
    }

    ui       -> cache   [label="check first" color="#ff6f00"]
    ui       -> intake  [label="cache miss" color="#4caf50"]
    intake   -> transcr
    transcr  -> pii
    pii      -> rag
    rag      -> chroma  [label="query" color="#7b1fa2" style=dashed]
    chroma   -> rag     [label="top-3 results" color="#7b1fa2" style=dashed]
    rag      -> sent
    sent     -> summ
    summ     -> claude  [color="#9c27b0" style=dashed]
    summ     -> gpt4    [color="#9c27b0" style=dashed]
    summ     -> gemini  [color="#9c27b0" style=dashed]
    qa       -> claude  [color="#9c27b0" style=dashed]
    qa       -> gpt4    [color="#9c27b0" style=dashed]
    qa       -> gemini  [color="#9c27b0" style=dashed]
    summ     -> qa
    qa       -> end_n
    end_n    -> cache   [label="save" color="#ff6f00"]
    end_n    -> chroma  [label="store\nembedding" color="#7b1fa2"]
    end_n    -> jsonl   [label="call history" color="#ff6f00"]
    end_n    -> ui      [label="result" color="#4caf50"]
    transcr  -> whisper [label="audio" color="#9c27b0" style=dashed]
    transcr  -> errh    [style=dashed color="#f57c00" label="errors"]
    errh     -> summ    [style=dashed color="#f57c00"]
}
"""
    st.graphviz_chart(_arch_dot)

    st.markdown("---")

    # ── Version History ────────────────────────────────────────────────────────
    st.subheader("🗂️ Version History")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown("**Version 1 — Baseline**")
        st.markdown("""
- 5 agents: intake, transcription, summarization, quality_score, routing
- LangGraph state machine with conditional routing
- 3 LLMs: Claude, GPT-4o, Gemini 2.5 Flash
- File-based SHA-256 cache
- JSONL call history (memory layer)
- Streamlit UI: 6 tabs
- Docker + docker-compose
- OpenAI Whisper audio transcription
        """)
    with v_col2:
        st.markdown("**Version 2 — Production-Ready (current)**")
        st.markdown("""
- ✅ **PIIRedactionAgent** — masks phone, email, SSN, card# before LLMs
- ✅ **RAGRetrievalAgent** — ChromaDB semantic search of past calls
- ✅ **SentimentAgent** — per-turn sentiment + escalation risk scoring
- ✅ 9-node LangGraph pipeline (intake → transcription → pii → rag → sentiment → summarize → qa → end)
- ✅ Vector embeddings stored after every processed call
- ✅ RAG context injected into summarization + QA prompts
- ✅ Architecture tab (this view)
- ✅ Sentiment chart + PII audit in Results tab
        """)

    st.markdown("---")

    # ── Tech Stack ─────────────────────────────────────────────────────────────
    st.subheader("🛠️ Tech Stack")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown("**AI / Orchestration**")
        st.markdown("""
| Component | Library |
|-----------|---------|
| Agent orchestration | LangGraph |
| LLM integration | LangChain |
| Structured output | Pydantic v2 |
| Claude | `langchain-anthropic` |
| GPT-4o | `langchain-openai` |
| Gemini | `langchain-google-genai` |
| Whisper STT | OpenAI API |
        """)
    with t_col2:
        st.markdown("**Data / Storage**")
        st.markdown("""
| Component | Technology |
|-----------|------------|
| Vector store | ChromaDB (local) |
| Embeddings | `text-embedding-3-small` |
| Exact cache | SHA-256 / JSON files |
| Call history | JSONL flat file |
| Config | `.env` + Pydantic Settings |
| Model control | `config/mcp.yaml` |
        """)
    with t_col3:
        st.markdown("**Infrastructure**")
        st.markdown("""
| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Containerization | Docker + docker-compose |
| Cloud UI | Streamlit Community Cloud |
| Cloud infra | AWS EC2 + ECR + SSM |
| IaC | CloudFormation |
| Observability | LangSmith tracing |
        """)

    st.markdown("---")

    # ── Agent Roadmap ──────────────────────────────────────────────────────────
    st.subheader("🚀 Agent Roadmap")
    st.caption("Ideas for Version 3+ — each adds a new capability without breaking the existing pipeline.")

    road_col1, road_col2 = st.columns(2)
    with road_col1:
        st.markdown("**Agents in progress / planned**")
        st.markdown("""
| Agent | What it does | Value |
|-------|-------------|-------|
| `ComplianceCheckerAgent` | Scans transcript for HIPAA/GDPR/PCI policy violations | Regulatory risk reduction |
| `EscalationPredictionAgent` | Predicts escalation probability mid-call | Real-time supervisor alert |
| `CallCoachingAgent` | Generates personalised coaching tips per agent | Agent performance uplift |
| `KnowledgeBaseAgent` | RAG against internal SOPs and product docs | Faster, more accurate responses |
| `CustomerProfileAgent` | Cross-call customer journey tracking | Repeat-caller context |
| `AutoTaggingAgent` | Multi-label classification (billing, fraud, etc.) | Routing + analytics |
| `AnomalyDetectionAgent` | Flags statistical outliers (very long calls, re-escalations) | Ops monitoring |
| `FeedbackLoopAgent` | Tracks if QA feedback was acted on | Closed-loop quality |
        """)
    with road_col2:
        st.markdown("**Infrastructure upgrades**")
        st.markdown("""
| Upgrade | Benefit |
|---------|---------|
| FastAPI sidecar | REST API for external integrations |
| Redis cache | Shared cache across instances (vs file-based) |
| Pinecone / Weaviate | Managed vector DB with metadata filtering |
| Async LangGraph | Parallel LLM calls (sentiment + summarization concurrently) |
| Streaming responses | Real-time token streaming to UI |
| Webhook support | Push results to CRM / Slack on completion |
| Role-based access | Multi-tenant auth (per team / per agent) |
| Batch processing | Process hundreds of calls overnight |
        """)

    st.markdown("---")

    # ── Production Checklist ───────────────────────────────────────────────────
    st.subheader("✅ Production Readiness Checklist")
    check_col1, check_col2 = st.columns(2)
    with check_col1:
        st.markdown("""
**Implemented**
- ✅ PII redaction before LLM calls (GDPR / HIPAA)
- ✅ File-based caching (zero duplicate API spend)
- ✅ ChromaDB vector store (semantic memory)
- ✅ Error recovery (conditional routing)
- ✅ LangSmith observability tracing
- ✅ Multi-LLM fallback support
- ✅ Docker containerization
- ✅ AWS EC2 deploy scripts + SSM secrets
- ✅ Sentiment + escalation risk scoring
        """)
    with check_col2:
        st.markdown("""
**Next steps for scale**
- ⬜ Redis / Memcached for distributed cache
- ⬜ PostgreSQL for persistent call storage
- ⬜ Managed vector DB (Pinecone / Weaviate)
- ⬜ Async processing (FastAPI + Celery)
- ⬜ Streaming LLM responses to UI
- ⬜ Webhook / CRM integration
- ⬜ Role-based access control
- ⬜ Load balancing (multiple EC2 / ECS)
- ⬜ Auto-scaling policies
        """)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Call Center AI Assistant • Built with LangChain, LangGraph & Streamlit</p>
        <p style='font-size: 0.8rem'>Supports Claude, GPT-4, and Google Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)
