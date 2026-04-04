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
from workflow.langgraph_flow_v1 import create_workflow_v1, run_workflow_v1
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
if "pipeline_version" not in st.session_state:
    st.session_state.pipeline_version = "V3"
if "v1_comparison_result" not in st.session_state:
    st.session_state.v1_comparison_result = None  # V1 baseline run alongside V3 for comparison tab


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

    # Pipeline Version Selection
    pipeline_version = st.radio(
        "Pipeline Version:",
        options=["V1 — Baseline (5 nodes)", "V3 — Full AI Suite (17 nodes)"],
        index=1 if st.session_state.pipeline_version == "V3" else 0,
        help="V1: intake→transcription→summarization→QA. V3: full 17-node pipeline with PII, RAG, sentiment, compliance, coaching.",
    )
    selected_version = "V3" if "V3" in pipeline_version else "V1"
    if selected_version != st.session_state.pipeline_version:
        st.session_state.pipeline_version = selected_version
        st.session_state.call_result = None
        st.session_state.benchmark_result = None
        st.session_state.active_call_id = None
        st.session_state.v2_extras = {}

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

    # Sample Audio Files
    _audio_dir = Path(__file__).parent.parent / "data" / "sample_audio"
    _audio_files = sorted(_audio_dir.glob("*.wav")) if _audio_dir.exists() else []
    if _audio_files:
        st.subheader("🎙️ Sample Audio (WAV)")
        st.caption("Click to load a pre-built audio call. Whisper will transcribe it, then the full V3 pipeline runs.")
        _AUDIO_LABELS = {
            "sample_audio_billing.wav":     "💳 Billing Dispute",
            "sample_audio_escalation.wav":  "🚨 Escalation Request",
            "sample_audio_tech_support.wav":"🔧 Tech Support",
            "sample_audio_fraud.wav":       "🔒 Fraud Report",
            "sample_audio_complaint.wav":   "😡 Complaint",
            "sample_audio_account.wav":     "👤 Account Reset",
        }
        for _wav in _audio_files:
            _label = _AUDIO_LABELS.get(_wav.name, _wav.stem.replace("_", " ").title())
            if st.button(f"Load {_label}", key=f"wav_{_wav.stem}"):
                with st.spinner(f"🎙️ Transcribing {_wav.name} via Whisper..."):
                    try:
                        from agents.transcription_agent import TranscriptionAgent
                        _ta = TranscriptionAgent()
                        _t_out = _ta.process(call_id="audio_sample", audio_path=str(_wav))
                        st.session_state.transcript_text = _t_out.transcript
                        st.session_state.audio_transcript = _t_out.transcript
                        st.session_state.audio_filename = _wav.name
                        st.session_state.sample_metadata = {
                            "call_id": f"audio_{_wav.stem}",
                            "category": _wav.stem.replace("sample_audio_", "").replace("_", " "),
                        }
                        st.session_state.call_result = None
                        st.session_state.benchmark_result = None
                        st.session_state.active_call_id = None
                        st.success(f"✓ Transcribed — {len(_t_out.transcript)} chars")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Transcription failed: {_e}")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["📤 Upload", "📋 Results", "⭐ QA Score", "🔬 Benchmark", "🗺️ Workflow", "📜 Call History", "🏗️ Architecture", "📈 V1→V3 Gains", "🎯 Pitch Deck"]
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
                # V1 and V3 use separate cache keys so they never overwrite each other
                _wf_cache_type = "workflow_v1" if st.session_state.pipeline_version == "V1" else "workflow"
                cached = get_cached(transcript_input, llm_choice, _wf_cache_type)
                if cached:
                    st.session_state.call_result = CallResult.model_validate(cached)
                    st.session_state.benchmark_result = None  # clear stale benchmark
                    st.session_state.active_call_id = st.session_state.call_result.call_id
                    st.session_state.active_llm = llm_choice
                    # Clear audio transcript since this came from cache
                    st.session_state.audio_transcript = None
                    st.session_state.audio_filename = None
                    # Restore V1 comparison from cache so Gains tab works on cache hits
                    if st.session_state.pipeline_version == "V3":
                        v1c = get_cached(transcript_input, llm_choice, "v1_comparison")
                        st.session_state.v1_comparison_result = CallResult.model_validate(v1c) if v1c else None
                    st.success(
                        f"⚡ Loaded from cache — no LLM API call made! "
                        f"Call {st.session_state.call_result.call_id} · LLM: {llm_choice} · [{st.session_state.pipeline_version}]"
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

                        if st.session_state.pipeline_version == "V1":
                            workflow = create_workflow_v1(llm_name=llm_choice)
                            call_result = run_workflow_v1(workflow, call_input, llm_name=llm_choice)
                            st.session_state.v1_comparison_result = None
                        else:
                            workflow = create_workflow(llm_name=llm_choice)
                            call_result = run_workflow(workflow, call_input, llm_name=llm_choice)

                            # Also run V1 silently for the V1→V3 comparison tab (cached)
                            v1_cached = get_cached(transcript_input, llm_choice, "v1_comparison")
                            if v1_cached:
                                st.session_state.v1_comparison_result = CallResult.model_validate(v1_cached)
                            else:
                                with st.spinner("Running V1 baseline for comparison..."):
                                    v1_wf = create_workflow_v1(llm_name=llm_choice)
                                    v1_result = run_workflow_v1(v1_wf, call_input, llm_name=llm_choice)
                                    v1_data = v1_result.model_dump()
                                    v1_data["_cache_type"] = "v1_comparison"
                                    v1_data["_llm_name"] = llm_choice
                                    save_cache(transcript_input, llm_choice, "v1_comparison", v1_data)
                                    st.session_state.v1_comparison_result = v1_result

                        # Save to cache — V1 and V3 use separate keys
                        data = call_result.model_dump()
                        data["_cache_type"] = _wf_cache_type
                        data["_llm_name"] = llm_choice
                        save_cache(transcript_input, llm_choice, _wf_cache_type, data)

                        st.session_state.call_result = call_result
                        st.session_state.benchmark_result = None
                        st.session_state.active_call_id = call_result.call_id
                        st.session_state.active_llm = llm_choice
                        # Capture V2 extras (PII, sentiment, RAG)
                        st.session_state.v2_extras = getattr(call_result, "_v2_extras", {})
                        st.success(f"✓ Call {call_result.call_id} processed [{st.session_state.pipeline_version}] and cached")

            except Exception as e:
                st.error(f"Error processing call: {str(e)}")
                logger.error(f"Processing error: {e}", exc_info=True)


# TAB 2: Results
with tab2:
    st.header("Call Analysis Results")

    if not st.session_state.call_result:
        st.info("👈 Upload and process a call in the Upload tab first")
    else:
        st.caption(f"🔵 Showing: **{st.session_state.active_call_id}** · LLM: **{st.session_state.active_llm}** · Pipeline: **{st.session_state.pipeline_version}**")
        if st.session_state.pipeline_version == "V1":
            st.info("ℹ️ **V1 Pipeline** — Core summary and QA score only. Switch to V3 in the sidebar for PII redaction, RAG, sentiment, compliance, coaching, and anomaly detection.")
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

        # ── V3: Escalation Prediction ─────────────────────────────────────────
        escalation = st.session_state.v2_extras.get("escalation")
        if escalation and escalation.get("risk_level", "unknown") != "unknown":
            risk_score = escalation.get("risk_score", 0)
            risk_level = escalation.get("risk_level", "unknown")
            risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(risk_level, "⚪")

            if risk_level in ("high", "critical"):
                st.error(
                    f"🚨 **Escalation Alert — Risk {risk_score:.0f}/100** · "
                    f"{escalation.get('recommended_intervention', '')}"
                )
            elif risk_level == "medium":
                st.warning(f"⚠️ **Escalation Risk {risk_score:.0f}/100** (Medium) — Monitor this call")

            with st.expander(f"{risk_icon} Escalation Prediction — {risk_level.upper()} ({risk_score:.0f}/100)", expanded=risk_level in ("high", "critical")):
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.metric("Risk Score", f"{risk_score:.0f}/100")
                with e2:
                    st.metric("Predicted Outcome", escalation.get("predicted_outcome", "N/A").replace("_", " ").title())
                with e3:
                    st.metric("Triggers Found", len(escalation.get("triggers", [])))

                st.markdown(f"**Recommended Action:** {escalation.get('recommended_intervention', 'N/A')}")
                st.markdown(f"**Frustration Peak:** {escalation.get('customer_frustration_peak', 'N/A')}")
                st.markdown(f"**What could have prevented this:** {escalation.get('would_have_prevented', 'N/A')}")

                triggers = escalation.get("triggers", [])
                if triggers:
                    st.markdown("**Trigger Moments:**")
                    for trig in triggers:
                        impact_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(trig.get("impact", ""), "⚪")
                        st.caption(f"{impact_icon} [{trig.get('trigger_type', '').replace('_', ' ').upper()}] {trig.get('turn_reference', '')}")

        # ── V3: Compliance Check ──────────────────────────────────────────────
        compliance = st.session_state.v2_extras.get("compliance")
        if compliance and compliance.get("overall_compliance_status", "unknown") != "unknown":
            st.subheader("⚖️ Compliance Check")
            status = compliance.get("overall_compliance_status", "unknown")
            score = compliance.get("compliance_score", 100)
            violations = compliance.get("violations", [])
            requires_review = compliance.get("requires_immediate_review", False)

            status_color = {
                "compliant": "🟢", "minor_issues": "🟡",
                "major_violations": "🔴", "critical": "🚨"
            }.get(status, "⚪")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Compliance Score", f"{score:.0f}/100")
            with c2:
                st.metric("Status", f"{status_color} {status.replace('_', ' ').title()}")
            with c3:
                st.metric("Violations Found", len(violations))
            with c4:
                st.metric("Immediate Review", "🚨 YES" if requires_review else "✅ No")

            if requires_review:
                st.error(f"🚨 **Supervisor action required** — {compliance.get('summary', '')}")
            elif violations:
                st.warning(compliance.get("summary", ""))
            else:
                st.success(compliance.get("summary", "No violations detected."))

            if violations:
                sev_icon = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}
                for v in violations:
                    icon = sev_icon.get(v.get("severity", ""), "⚪")
                    with st.expander(
                        f"{icon} [{v.get('category')}] {v.get('severity', '').upper()} — {v.get('description', '')[:80]}"
                    ):
                        st.markdown(f"**Evidence:** {v.get('transcript_evidence', 'N/A')}")
                        st.markdown(f"**Remediation:** {v.get('remediation', 'N/A')}")

        # ── V3: Customer Profile ──────────────────────────────────────────────
        customer_profile = st.session_state.v2_extras.get("customer_profile")
        if customer_profile and not customer_profile.get("is_first_call"):
            total_calls = customer_profile.get("total_calls_in_history", 0)
            risk_tier = customer_profile.get("risk_tier", "regular")
            tier_icon = {"vip": "⭐", "at_risk": "⚠️", "churning": "🚨", "regular": "👤"}.get(risk_tier, "👤")
            tier_color = {"vip": "success", "at_risk": "warning", "churning": "error", "regular": "info"}.get(risk_tier, "info")

            if risk_tier == "churning":
                st.error(f"🚨 **CHURN RISK** — {customer_profile.get('profile_summary', '')}")
            elif risk_tier == "at_risk":
                st.warning(f"⚠️ **At-Risk Customer** — {customer_profile.get('profile_summary', '')}")

            with st.expander(f"{tier_icon} Customer Profile — {risk_tier.upper().replace('_', ' ')} ({total_calls} prior call(s))", expanded=risk_tier in ("at_risk", "churning")):
                cp1, cp2, cp3, cp4 = st.columns(4)
                with cp1:
                    st.metric("Prior Calls", total_calls)
                with cp2:
                    avg_qa = customer_profile.get("avg_qa_score", 0.0)
                    st.metric("Avg QA Score", f"{avg_qa:.0f}/100" if avg_qa else "N/A")
                with cp3:
                    st.metric("Escalations", customer_profile.get("escalation_count", 0))
                with cp4:
                    trend = customer_profile.get("sentiment_trend", "unknown")
                    trend_icon = {"improving": "📈", "stable": "➡️", "deteriorating": "📉"}.get(trend, "❓")
                    st.metric("Sentiment Trend", f"{trend_icon} {trend.title()}")

                top_issues = customer_profile.get("top_issues", [])
                if top_issues:
                    st.markdown("**Top Issues:**")
                    for issue in top_issues:
                        st.caption(f"  • {issue['category']} ({issue['count']} calls)")

                st.caption(customer_profile.get("profile_summary", ""))

        elif customer_profile and customer_profile.get("is_first_call"):
            st.info("👤 **New Customer** — No prior call history found")

        # ── V3: Auto Tags ─────────────────────────────────────────────────────
        tags = st.session_state.v2_extras.get("tags")
        if tags and tags.get("primary_category"):
            with st.expander(f"🏷️ Auto Tags — {tags.get('primary_category', '').replace('_', ' ').title()}", expanded=False):
                t1, t2, t3 = st.columns(3)
                with t1:
                    st.markdown(f"**Primary Category**\n`{tags.get('primary_category', 'N/A')}`")
                    st.markdown(f"**Sub Category**\n`{tags.get('sub_category', 'N/A')}`")
                with t2:
                    intents = tags.get("intent_tags", [])
                    if intents:
                        st.markdown("**Intent Tags**")
                        st.markdown(" ".join(f"`{i}`" for i in intents))
                    routing = tags.get("routing_tags", [])
                    if routing:
                        st.markdown("**Routing**")
                        st.markdown(" ".join(f"`{r}`" for r in routing))
                with t3:
                    sent_tags = tags.get("sentiment_tags", [])
                    if sent_tags:
                        st.markdown("**Sentiment Tags**")
                        st.markdown(" ".join(f"`{s}`" for s in sent_tags))
                    products = tags.get("product_tags", [])
                    if products:
                        st.markdown("**Products**")
                        st.markdown(" ".join(f"`{p}`" for p in products))
                conf = tags.get("confidence_score", 0.0)
                st.caption(f"Confidence: {conf:.0%} — {tags.get('tagging_rationale', '')}")

        # ── V3: Knowledge Base ────────────────────────────────────────────────
        kb_analysis = st.session_state.v2_extras.get("kb_analysis")
        if kb_analysis and kb_analysis.get("relevant_articles"):
            sop_score = kb_analysis.get("sop_compliance_score", 100.0)
            sop_icon = "🟢" if sop_score >= 90 else "🟡" if sop_score >= 70 else "🔴"
            with st.expander(f"📚 Knowledge Base — SOP Compliance {sop_icon} {sop_score:.0f}%", expanded=sop_score < 80):
                st.metric("SOP Compliance Score", f"{sop_score:.0f}%")

                articles = kb_analysis.get("relevant_articles", [])
                for art in articles:
                    comp = art.get("was_agent_compliant", "yes")
                    comp_icon = {"yes": "✅", "no": "❌", "partial": "⚠️", "not_applicable": "⚫"}.get(comp, "⚫")
                    with st.expander(f"{comp_icon} [{art.get('article_id')}] {art.get('title', '')} — {comp.upper()}"):
                        if art.get("agent_deviation"):
                            st.error(f"**Deviation:** {art['agent_deviation']}")
                        for kp in art.get("key_points", []):
                            st.caption(f"• {kp}")

                missed = kb_analysis.get("missed_knowledge_opportunities", [])
                if missed and missed != ["No major missed KB opportunities detected"]:
                    st.warning("**Missed KB opportunities:**")
                    for m in missed:
                        st.caption(f"→ {m}")

                training = kb_analysis.get("recommended_training_articles", [])
                if training:
                    st.info(f"**Recommended training:** {', '.join(training)}")

                st.caption(kb_analysis.get("kb_summary", ""))

        # ── V3: Call Coaching ─────────────────────────────────────────────────
        coaching = st.session_state.v2_extras.get("coaching")
        if coaching and coaching.get("coaching_tips") is not None:
            priority = coaching.get("overall_coaching_priority", "low")
            priority_icon = {"immediate": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            with st.expander(f"🎓 Agent Coaching — {priority_icon} {priority.upper()} Priority", expanded=priority in ("immediate", "high")):
                strengths = coaching.get("agent_strengths", [])
                if strengths:
                    st.markdown("**Agent Strengths:**")
                    for s in strengths:
                        st.success(f"✓ {s}")

                tips = coaching.get("coaching_tips", [])
                if tips:
                    st.markdown("**Coaching Tips (prioritised):**")
                    for tip in tips:
                        tip_icon = {"immediate": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(tip.get("priority", ""), "⚪")
                        with st.expander(f"{tip_icon} [{tip.get('dimension', '').replace('_', ' ').title()}] Score: {tip.get('current_score', 0):.1f}/10 — {tip.get('priority', '').upper()}"):
                            st.markdown(f"**What happened:** {tip.get('what_happened', 'N/A')}")
                            st.markdown(f"**What to do instead:** {tip.get('what_to_do_instead', 'N/A')}")
                            st.info(f"**Example script:** *\"{tip.get('example_script', '')}\"*")

                st.markdown(f"**Next call focus:** {coaching.get('next_call_focus', 'N/A')}")
                st.caption(f"**Estimated improvement:** {coaching.get('estimated_improvement', 'N/A')}")
                st.caption(coaching.get("coaching_summary", ""))

        # ── V3: Anomaly Detection ─────────────────────────────────────────────
        anomaly = st.session_state.v2_extras.get("anomaly")
        if anomaly:
            anomaly_score = anomaly.get("anomaly_score", 0.0)
            anomaly_level = anomaly.get("anomaly_level", "normal")
            requires_review = anomaly.get("requires_review", False)
            level_icon = {"normal": "✅", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(anomaly_level, "⚪")

            if requires_review:
                st.error(f"🚨 **Anomaly Detected — Queued for QA Review** (score: {anomaly_score:.0f}/100)")

            with st.expander(f"{level_icon} Anomaly Detection — {anomaly_level.upper()} ({anomaly_score:.0f}/100)", expanded=requires_review):
                a1, a2 = st.columns(2)
                with a1:
                    st.metric("Anomaly Score", f"{anomaly_score:.0f}/100")
                with a2:
                    st.metric("QA Review Required", "YES" if requires_review else "No")

                flags = anomaly.get("flags", [])
                if flags:
                    st.markdown("**Anomaly Flags:**")
                    for flag in flags:
                        flag_icon = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(flag.get("severity", ""), "⚪")
                        st.caption(f"{flag_icon} [{flag.get('type', '').replace('_', ' ').upper()}] {flag.get('detail', '')}")
                else:
                    st.success("No anomalies detected — call within normal parameters")

                stat_ctx = anomaly.get("statistical_context", {})
                if stat_ctx and stat_ctx.get("population_size", 0) >= 5:
                    st.caption(
                        f"Statistical context: mean={stat_ctx.get('population_mean', 'N/A')}, "
                        f"σ={stat_ctx.get('population_stdev', 'N/A')}, "
                        f"z={stat_ctx.get('z_score', 'N/A')} "
                        f"(n={stat_ctx.get('population_size', 0)} calls)"
                    )

        # ── V3: Feedback Loop ─────────────────────────────────────────────────
        feedback_loop = st.session_state.v2_extras.get("feedback_loop")
        if feedback_loop and feedback_loop.get("improvement_status") not in (None, "insufficient_history"):
            status = feedback_loop.get("improvement_status", "stable")
            delta = feedback_loop.get("score_delta", 0.0)
            status_icon = {
                "significantly_improved": "📈", "improved": "↗️", "stable": "➡️",
                "declined": "↘️", "regressed": "📉"
            }.get(status, "➡️")

            with st.expander(f"{status_icon} Feedback Loop — {status.replace('_', ' ').title()} ({delta:+.1f} pts)", expanded=status == "regressed"):
                fl1, fl2, fl3 = st.columns(3)
                with fl1:
                    st.metric("Score Delta", f"{delta:+.1f}")
                with fl2:
                    st.metric("Prior Avg Score", f"{feedback_loop.get('prior_avg_score', 0):.0f}/100")
                with fl3:
                    st.metric("Coaching Adoption", f"{feedback_loop.get('coaching_adoption_rate', 0):.0f}%")

                improved = feedback_loop.get("improved_dimensions", [])
                regressed = feedback_loop.get("regressed_dimensions", [])
                if improved:
                    st.success(f"**Improved dimensions:** {', '.join(d['dimension'] for d in improved)}")
                if regressed:
                    st.warning(f"**Regressed dimensions:** {', '.join(d['dimension'] for d in regressed)}")

                st.caption(feedback_loop.get("feedback_summary", ""))

        elif feedback_loop and feedback_loop.get("improvement_status") == "insufficient_history":
            with st.expander("🔄 Feedback Loop — Building baseline...", expanded=False):
                st.info(feedback_loop.get("feedback_summary", "Processing more calls to build the feedback baseline."))

        # ── V2/V3: Processing Details ─────────────────────────────────────────
        pii_summary = st.session_state.v2_extras.get("pii_summary", {})
        rag_ctx = st.session_state.v2_extras.get("rag_context", "")
        kb_ctx = st.session_state.v2_extras.get("kb_context", "")
        with st.expander("🔒 Processing Details (PII / RAG / KB)", expanded=False):
            if st.session_state.pipeline_version == "V1":
                st.caption("Not available in V1 pipeline — these features require V3.")
            else:
                if pii_summary:
                    total_pii = sum(pii_summary.values())
                    st.success(f"🔒 PII Redaction: {total_pii} item(s) masked before LLM processing")
                    for field, count in pii_summary.items():
                        st.caption(f"  • {field.replace('_', ' ').title()}: {count}")
                else:
                    st.info("🔒 PII Redaction: No PII detected in transcript")
                if rag_ctx:
                    st.success(f"🔍 Call-History RAG: {len(rag_ctx)} chars of context from similar past calls")
                else:
                    st.info("🔍 Call-History RAG: No similar past calls yet (vector store building up)")
                if kb_ctx:
                    st.success(f"📚 Knowledge Base: {len(kb_ctx)} chars of KB articles injected into prompts")
                else:
                    st.info("📚 Knowledge Base: No KB articles matched for this call")

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
    _pv = st.session_state.pipeline_version  # "V1" or "V3"
    st.header(f"🗺️ LangGraph Workflow — {_pv}")

    # ── SECTION 1: Graph Diagram + Side Explanation ───────────────────────────
    st.subheader("📊 Graph Diagram")
    _flow_file = "workflow/langgraph_flow_v1.py" if _pv == "V1" else "workflow/langgraph_flow.py"
    st.caption(
        f"📌 **LangGraph workflow** — matches the compiled graph in `{_flow_file}`. "
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
""")
        if _pv == "V1":
            st.markdown("""
---
**Step-by-step walk-through (V1 — 5 nodes)**

1. **START** kicks off the pipeline with the user's input.
2. **intake** validates the transcript and assigns a `call_id`.
3. **transcription** normalises speaker labels to `Agent:` / `Customer:`.
4. *Conditional split:*
   - No errors → **summarization** calls the LLM to produce summary + key points
   - Errors → **error_handler** attempts recovery, then rejoins at summarization
5. **quality_score** calls the LLM to score the call on 4 dimensions (0–100).
6. **end** packages everything into a `CallResult` and saves to JSONL call history.
7. **END** returns control to the Streamlit UI.

---
**error_handler recovery logic**

Checks what's still missing and skips ahead to the earliest incomplete step.
""")
        else:
            st.markdown("""
---
**Step-by-step walk-through (V3 — 17 nodes)**

1. **START** kicks off the pipeline with the user's input.
2. **intake** validates the transcript and assigns a `call_id`.
3. **customer_profile** loads cross-call history and assigns a risk tier (no LLM).
4. **transcription** normalises speaker labels to `Agent:` / `Customer:`.
5. *Conditional split:*
   - No errors → **pii_redaction** masks phone/email/SSN/card before any LLM sees it
   - Errors → **error_handler** attempts recovery, then rejoins
6. **rag_retrieval** embeds the transcript and fetches top-3 similar past calls from ChromaDB.
7. **kb_retrieval** finds relevant SOPs and scores SOP compliance.
8. **sentiment** scores each turn and calculates escalation risk signal.
9. **compliance_check** scans for HIPAA/GDPR/PCI-DSS/TCPA violations.
10. **escalation_prediction** combines all signals to produce a risk score 0–100.
11. **summarization** calls the LLM with transcript + RAG + KB context injected.
12. **auto_tagging** assigns multi-label classification (category, intent, routing, product).
13. **quality_score** calls the LLM with the same context to calibrate scores.
14. **call_coaching** generates personalised coaching tips + example scripts.
15. **anomaly_detection** z-scores the call vs history (no LLM).
16. **end** stores the embedding in ChromaDB, saves to JSONL, and runs FeedbackLoopAgent.
17. **END** returns control to the Streamlit UI.

---
**error_handler recovery logic**

Checks what's missing and skips ahead to the earliest incomplete step — no work is duplicated.
""")

    with diag_col:
        if _pv == "V1":
            _dot_source = """
digraph LangGraph {
    rankdir=TD
    bgcolor=white
    fontname=Helvetica
    node [fontname=Helvetica fontsize=10]

    __start__     [label="START" shape=oval style=filled fillcolor="#c8e6c9" color="#4caf50"]
    intake        [label="intake\\n(IntakeAgent)"               shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    transcription [label="transcription\\n(TranscriptionAgent)" shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    error_handler [label="error_handler\\n(recovery)"           shape=box style=filled fillcolor="#ffe0b2" color="#f57c00"]
    summarization [label="summarization\\n(SummarizationAgent)" shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    quality_score [label="quality_score\\n(QualityScoreAgent)"  shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    end_node      [label="end"                                  shape=box style=filled fillcolor="#c8e6c9" color="#4caf50"]
    __end__       [label="END" shape=oval style=filled fillcolor="#c8e6c9" color="#4caf50"]

    __start__     -> intake        [color="#333333"]
    intake        -> transcription [color="#333333"]
    transcription -> summarization [color="#333333" label="no errors"]
    transcription -> error_handler [color="#f57c00" style=dashed label="errors"]
    summarization -> quality_score [color="#333333"]
    quality_score -> end_node      [color="#333333"]
    error_handler -> summarization [color="#f57c00" style=dashed]
    error_handler -> quality_score [color="#f57c00" style=dashed]
    error_handler -> end_node      [color="#f57c00" style=dashed]
    end_node      -> __end__       [color="#333333"]
}
"""
        else:
            _dot_source = """
digraph LangGraph {
    rankdir=TD
    bgcolor=white
    fontname=Helvetica
    node [fontname=Helvetica fontsize=10]

    __start__          [label="START" shape=oval style=filled fillcolor="#c8e6c9" color="#4caf50"]
    intake             [label="intake\\n(IntakeAgent)"               shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    customer_profile   [label="customer_profile\\n(CustomerProfileAgent)" shape=box style=filled fillcolor="#e8f5e9" color="#2e7d32"]
    transcription      [label="transcription\\n(TranscriptionAgent)" shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    error_handler      [label="error_handler\\n(recovery)"           shape=box style=filled fillcolor="#ffe0b2" color="#f57c00"]
    pii_redaction      [label="pii_redaction\\n(PIIRedactionAgent)"  shape=box style=filled fillcolor="#fce4ec" color="#c62828"]
    rag_retrieval      [label="rag_retrieval\\n(RAGAgent + ChromaDB)" shape=box style=filled fillcolor="#e1bee7" color="#7b1fa2"]
    kb_retrieval       [label="kb_retrieval\\n(KnowledgeBaseAgent)"  shape=box style=filled fillcolor="#e8eaf6" color="#3949ab"]
    sentiment          [label="sentiment\\n(SentimentAgent)"          shape=box style=filled fillcolor="#fff9c4" color="#f9a825"]
    compliance_check   [label="compliance_check\\n(ComplianceAgent)" shape=box style=filled fillcolor="#fce4ec" color="#c62828"]
    escalation_prediction [label="escalation_pred\\n(EscalationAgent)" shape=box style=filled fillcolor="#ffe0b2" color="#e65100"]
    summarization      [label="summarization\\n(SummarizationAgent)" shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    auto_tagging       [label="auto_tagging\\n(AutoTaggingAgent)"    shape=box style=filled fillcolor="#e8f5e9" color="#2e7d32"]
    quality_score      [label="quality_score\\n(QualityScoreAgent)"  shape=box style=filled fillcolor="#bbdefb" color="#1976d2"]
    call_coaching      [label="call_coaching\\n(CallCoachingAgent)"  shape=box style=filled fillcolor="#e8f5e9" color="#2e7d32"]
    anomaly_detection  [label="anomaly_detect\\n(AnomalyAgent)"      shape=box style=filled fillcolor="#fce4ec" color="#c62828"]
    end_node           [label="end\\n(+ FeedbackLoop)"               shape=box style=filled fillcolor="#c8e6c9" color="#4caf50"]
    __end__            [label="END" shape=oval style=filled fillcolor="#c8e6c9" color="#4caf50"]

    __start__           -> intake               [color="#333333"]
    intake              -> customer_profile     [color="#2e7d32"]
    customer_profile    -> transcription        [color="#333333"]
    transcription       -> pii_redaction        [color="#333333" label="no errors"]
    transcription       -> error_handler        [color="#f57c00" style=dashed label="errors"]
    pii_redaction       -> rag_retrieval        [color="#c62828"]
    rag_retrieval       -> kb_retrieval         [color="#7b1fa2"]
    kb_retrieval        -> sentiment            [color="#3949ab"]
    sentiment           -> compliance_check     [color="#f9a825"]
    compliance_check    -> escalation_prediction [color="#c62828"]
    escalation_prediction -> summarization      [color="#e65100"]
    summarization       -> auto_tagging         [color="#333333"]
    auto_tagging        -> quality_score        [color="#2e7d32"]
    quality_score       -> call_coaching        [color="#333333"]
    call_coaching       -> anomaly_detection    [color="#2e7d32"]
    anomaly_detection   -> end_node             [color="#c62828"]
    error_handler       -> summarization        [color="#f57c00" style=dashed]
    error_handler       -> quality_score        [color="#f57c00" style=dashed]
    error_handler       -> end_node             [color="#f57c00" style=dashed]
    end_node            -> __end__              [color="#333333"]
}
"""
        st.graphviz_chart(_dot_source)

    st.markdown("---")

    # ── SECTION 2: Workflow Architecture (static reference) ────────────────────
    st.subheader("📐 Workflow Architecture")
    st.caption(f"📌 **Static reference** — graph structure defined in `{_flow_file}`.")

    col1, col2 = st.columns(2)
    with col1:
        if _pv == "V1":
            st.markdown("**Nodes (Agents) — Version 1 (5 nodes)**")
            st.markdown("""
| Node | Agent | Role |
|------|-------|------|
| `intake` | IntakeAgent | Input validation, call_id |
| `transcription` | TranscriptionAgent | Audio → text, speaker normalisation |
| `summarization` | SummarizationAgent | LLM summary, key points, action items |
| `quality_score` | QualityScoreAgent | 4-dimension QA scoring 0–100 |
| `end` | — | Assemble CallResult, save history |
| `error_handler` | recovery logic | Skip-ahead on transcription failure |
            """)
        else:
            st.markdown("**Nodes (Agents) — Version 3 (17 nodes)**")
            st.markdown("""
| Node | Agent | Version |
|------|-------|---------|
| `intake` | IntakeAgent | V1 |
| `customer_profile` | CustomerProfileAgent | **V3** |
| `transcription` | TranscriptionAgent | V1 |
| `pii_redaction` | PIIRedactionAgent | V2 |
| `rag_retrieval` | RAGRetrievalAgent | V2 |
| `kb_retrieval` | KnowledgeBaseAgent | **V3** |
| `sentiment` | SentimentAgent | V2 |
| `compliance_check` | ComplianceCheckerAgent | V2 |
| `escalation_prediction` | EscalationPredictionAgent | V2 |
| `summarization` | SummarizationAgent | V1 |
| `auto_tagging` | AutoTaggingAgent | **V3** |
| `quality_score` | QualityScoreAgent | V1 |
| `call_coaching` | CallCoachingAgent | **V3** |
| `anomaly_detection` | AnomalyDetectionAgent | **V3** |
| `end` + `feedback_loop` | FeedbackLoopAgent | **V3** |
| `error_handler` | recovery logic | V1 |
            """)
    with col2:
        st.markdown("**Key Edges**")
        if _pv == "V1":
            st.markdown("""
| From | To | Condition |
|------|----|-----------|
| START | `intake` | always |
| `intake` | `transcription` | always |
| `transcription` | `summarization` | `errors == []` |
| `transcription` | `error_handler` | `errors != []` |
| `summarization` | `quality_score` | always |
| `quality_score` | `end` | always |
| `end` | END | always |
            """)
        else:
            st.markdown("""
| From | To | Condition |
|------|----|-----------|
| START | `intake` | always |
| `intake` | `customer_profile` | always |
| `customer_profile` | `transcription` | always |
| `transcription` | `pii_redaction` | `errors == []` |
| `transcription` | `error_handler` | `errors != []` |
| `pii_redaction` | `rag_retrieval` | always |
| `rag_retrieval` | `kb_retrieval` | always |
| `kb_retrieval` | `sentiment` | always |
| `sentiment` | `compliance_check` | always |
| `compliance_check` | `escalation_prediction` | always |
| `escalation_prediction` | `summarization` | always |
| `summarization` | `auto_tagging` | always |
| `auto_tagging` | `quality_score` | always |
| `quality_score` | `call_coaching` | always |
| `call_coaching` | `anomaly_detection` | always |
| `anomaly_detection` | `end` | always |
| `end` | END | always |
            """)

    st.markdown("---")

    # ── SECTION 3: Shared State — schema (static) + actual values (dynamic) ────
    st.subheader("🗂️ Shared State (WorkflowState)")

    state_col1, state_col2 = st.columns(2)

    with state_col1:
        st.caption(f"📌 **Static** — schema defined in `{_flow_file}`")
        if _pv == "V1":
            st.code("""class WorkflowStateV1(TypedDict):
    call_id:      str              # unique call identifier
    input_data:   CallInput        # raw transcript + metadata
    transcript:   TranscriptOutput # normalized text + speakers
    summary:      SummaryOutput    # summary, key_points, action_items
    qa_score:     QAScore          # empathy/prof/resolution/compliance
    errors:       list[str]        # drives conditional routing
    current_step: str              # last node that ran""", language="python")
        else:
            st.code("""class WorkflowState(TypedDict):
    call_id:          str              # unique call identifier
    input_data:       CallInput        # raw transcript + metadata
    transcript:       TranscriptOutput # normalized text + speakers
    redacted_transcript: str           # PII-masked transcript
    pii_summary:      dict             # {field: count} redacted items
    customer_profile: Optional[dict]   # cross-call customer journey
    rag_context:      str              # top-3 similar calls (ChromaDB)
    kb_context:       str              # SOP context (KnowledgeBase)
    kb_analysis:      Optional[dict]   # full SOP compliance analysis
    sentiment:        Optional[dict]   # per-turn sentiment + escalation
    compliance:       Optional[dict]   # HIPAA/GDPR violations + score
    escalation:       Optional[dict]   # risk score 0-100 + triggers
    summary:          Optional[SummaryOutput]
    tags:             Optional[dict]   # multi-label classification
    qa_score:         Optional[QAScore]
    coaching:         Optional[dict]   # personalised coaching tips
    anomaly:          Optional[dict]   # anomaly score + flags
    feedback_loop:    Optional[dict]   # coaching adoption delta
    errors:           list[str]        # drives conditional routing
    current_step:     str              # last node that ran""", language="python")

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

        if _pv == "V1":
            steps = [
                ("START", "→ intake", "fixed", "green"),
                ("intake", "→ transcription", "fixed", "green"),
                ("transcription",
                 "→ **error_handler** ⚠️" if had_errors else "→ **summarization** ✅",
                 f"conditional — `errors={'non-empty' if had_errors else '[]'}`",
                 "red" if had_errors else "green"),
            ]
            if had_errors:
                steps.append(("error_handler", "→ summarization (recovery)", "conditional — summary missing", "orange"))
            steps += [
                ("summarization", "→ quality_score", "fixed", "green"),
                ("quality_score", "→ end", "fixed", "green"),
                ("end", "→ END", "fixed", "green"),
            ]
        else:
            steps = [
                ("START", "→ intake", "fixed", "green"),
                ("intake", "→ customer_profile", "fixed", "green"),
                ("customer_profile", "→ transcription", "fixed", "green"),
                ("transcription",
                 "→ **error_handler** ⚠️" if had_errors else "→ **pii_redaction** ✅",
                 f"conditional — `errors={'non-empty' if had_errors else '[]'}`",
                 "red" if had_errors else "green"),
            ]
            if had_errors:
                steps.append(("error_handler", "→ summarization (recovery)", "conditional — summary missing", "orange"))
            else:
                steps += [
                    ("pii_redaction", "→ rag_retrieval", "fixed", "green"),
                    ("rag_retrieval", "→ kb_retrieval", "fixed", "green"),
                    ("kb_retrieval", "→ sentiment", "fixed", "green"),
                    ("sentiment", "→ compliance_check", "fixed", "green"),
                    ("compliance_check", "→ escalation_prediction", "fixed", "green"),
                    ("escalation_prediction", "→ summarization", "fixed", "green"),
                ]
            steps += [
                ("summarization", "→ auto_tagging", "fixed", "green"),
                ("auto_tagging", "→ quality_score", "fixed", "green"),
                ("quality_score", "→ call_coaching", "fixed", "green"),
                ("call_coaching", "→ anomaly_detection", "fixed", "green"),
                ("anomaly_detection", "→ end", "fixed", "green"),
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
    st.caption(f"📌 **Static** — explains the conditional edge logic in `{_flow_file}` and `agents/routing_agent.py`.")

    st.markdown(f"""
LangGraph routing is controlled by **conditional edges** — Python lambda functions that
inspect the shared state and return which node to go to next.

There are **two conditional branch points** in the {_pv} workflow:
    """)

    if _pv == "V1":
        st.markdown("#### Branch 1 — After `transcription`")
        st.code("""# In langgraph_flow_v1.py → create_workflow_v1()
workflow.add_conditional_edges(
    "transcription",
    lambda s: "error_handler" if s["errors"] else "summarization",
    {"error_handler": "error_handler", "summarization": "summarization"},
)

# If errors = []  → go to summarization  (happy path)
# If errors != [] → go to error_handler  (recovery path)""", language="python")

        st.markdown("#### Branch 2 — After `error_handler`")
        st.code("""# In langgraph_flow_v1.py → create_workflow_v1()
workflow.add_conditional_edges(
    "error_handler",
    lambda s: "summarization" if not s["summary"]
         else "quality_score" if not s["qa_score"]
         else "end",
    {"summarization": "summarization", "quality_score": "quality_score", "end": "end"},
)

# error_handler checks what's missing and skips to earliest incomplete step.""", language="python")
    else:
        st.markdown("#### Branch 1 — After `transcription`")
        st.code("""# In langgraph_flow.py → create_workflow()
workflow.add_conditional_edges(
    "transcription",
    lambda s: "error_handler" if s["errors"] else "pii_redaction",
    {"error_handler": "error_handler", "pii_redaction": "pii_redaction"},
)

# If errors = []  → go to pii_redaction  (happy path — PII masked before any LLM)
# If errors != [] → go to error_handler  (recovery path)""", language="python")

        st.markdown("#### Branch 2 — After `error_handler`")
        st.code("""# In langgraph_flow.py → create_workflow()
workflow.add_conditional_edges(
    "error_handler",
    lambda s: "summarization" if not s["summary"]
         else "quality_score" if not s["qa_score"]
         else "end",
    {"summarization": "summarization", "quality_score": "quality_score", "end": "end"},
)

# error_handler skips the enrichment nodes (PII/RAG/KB/sentiment/compliance/escalation)
# and rejoins at summarization to ensure a result is always produced.""", language="python")

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

    st.markdown("---")

    # ── SECTION 6: Evaluation & Guardrails ─────────────────────────────────────
    st.subheader("🛡️ Evaluation & Guardrails")
    st.caption("How the pipeline is made robust — input validation, output guardrails, LLM-as-judge, and observability.")

    eg_col1, eg_col2 = st.columns(2, gap="large")
    with eg_col1:
        st.markdown("#### 🔒 Guardrail Layers")
        if _pv == "V1":
            st.markdown("""
| Layer | Where | What it does |
|---|---|---|
| **Input sanitization** | `utils/validation.py` | Length limits (10–50K chars), null bytes, encoding |
| **Structured output** | `utils/schemas.py` (Pydantic) | Every agent output validated; parse failures return safe defaults |
| **Error accumulation** | `WorkflowStateV1.errors` | All errors collected in state — never silently dropped |
| **Error recovery** | `error_handler` node | Skips to earliest incomplete step — always produces a result |
| **Temperature control** | Agent constructors | Summarization `0.7`, QA scoring `0.7` — balanced responses |
| **Mock mode** | `config/settings.py` | `MOCK_LLM=true` for zero-cost deterministic testing |
| **LangSmith tracing** | All agent calls | Full token + latency trace per node in `call-center-ai` project |
""")
        else:
            st.markdown("""
| Layer | Where | What it does |
|---|---|---|
| **Input sanitization** | `utils/validation.py` | Length limits (10–50K chars), null bytes, encoding |
| **Structured output** | `utils/schemas.py` (Pydantic) | Every agent output validated; parse failures return safe defaults |
| **PII guardrail** | `pii_redaction` node | Regex masks SSN/card/phone/email/DOB **before any LLM call** |
| **Compliance scanning** | `compliance_check` node | HIPAA / GDPR / PCI-DSS / TCPA / Financial — `temperature=0.1` |
| **Anomaly detection** | `anomaly_detect` node | Rule-based flags + z-score vs historical mean ± 2σ |
| **Error accumulation** | `WorkflowState.errors` | All errors collected in state — never silently dropped |
| **Error recovery** | `error_handler` node | Skips enrichment nodes, rejoins at summarization — always produces a result |
| **Temperature control** | Agent constructors | Compliance `0.1`, Escalation `0.2`, Sentiment `0.3`, Summarization `0.7` |
| **Mock mode** | `config/settings.py` | `MOCK_LLM=true` for zero-cost deterministic testing |
| **LangSmith tracing** | All agent calls | Full token + latency trace per node in `call-center-ai` project |
""")

    with eg_col2:
        st.markdown("#### 🧑‍⚖️ LLM-as-Judge / Evaluation")
        st.markdown("""
**Multi-LLM Benchmark (cross-model judge)**

The **Benchmark tab** runs the same transcript through all three LLMs and compares:
- Summary quality and resolution status agreement
- QA score distribution (highest performer highlighted)
- Latency per model (tokens/second proxy)

This acts as a cross-model judge — if Claude and GPT-4o agree but Gemini diverges, it signals a potential output quality issue.
""")
        st.markdown("""
**Per-output validation (schema-level judge)**

Every agent output is validated against a Pydantic schema before it enters the shared state:

```python
# QAScore — enforced ranges
empathy_score:         float  # ge=0, le=25
professionalism_score: float  # ge=0, le=25
resolution_score:      float  # ge=0, le=25
compliance_score:      float  # ge=0, le=25
overall_score:         float  # ge=0, le=100
```
Out-of-range values are rejected; the agent returns a safe default rather than corrupt state.
""")
        if _pv == "V3":
            st.markdown("""
**Anomaly detection as an automated evaluator**

`AnomalyDetectionAgent` acts as a post-hoc judge on every call:
- Flags QA score < 50 as `critical`
- Flags compliance score < 50 as `high`
- Z-scores the call against historical average ± 2σ
- Sets `requires_review=True` to route outliers to human QA
""")
        st.markdown("""
**Feedback loop as a learning signal**

`FeedbackLoopAgent` (V3, `end` node) compares current call scores against the agent's last 3 calls to measure whether coaching is improving performance — a closed-loop evaluation signal built into the pipeline.
""" if _pv == "V3" else """
**Feedback loop** (V3 only — upgrade from V1)

V3 adds `FeedbackLoopAgent` that compares agent scores across calls to measure coaching effectiveness — a closed-loop evaluation signal not present in V1.
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
    _pv7 = st.session_state.pipeline_version
    _arch_captions = {
        "V1": "Version 1 — Baseline 5-node pipeline: intake, transcription, summarization, quality_score, end.",
        "V3": "Version 3 — Full AI Suite: 17-node pipeline with PII, RAG, KB, sentiment, compliance, coaching, anomaly.",
    }
    st.header("🏗️ System Architecture")
    st.caption(f"**Active pipeline: {_pv7}** — {_arch_captions[_pv7]}")

    # ── Overview Diagram ───────────────────────────────────────────────────────
    st.subheader("📐 Full System Diagram")
    if _pv7 == "V1":
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
        ui [label="Web Browser\\n(7 Tabs)" shape=box fillcolor="#e3f2fd" color="#1976d2"]
    }

    subgraph cluster_workflow {
        label="LangGraph Workflow V1 (5 Nodes)"
        color="#4caf50"
        fontcolor="#4caf50"
        style=dashed
        intake  [label="intake"        fillcolor="#bbdefb" color="#1976d2" shape=box]
        transcr [label="transcription" fillcolor="#bbdefb" color="#1976d2" shape=box]
        summ    [label="summarization" fillcolor="#bbdefb" color="#1976d2" shape=box]
        qa      [label="quality_score" fillcolor="#bbdefb" color="#1976d2" shape=box]
        end_n   [label="end"           fillcolor="#c8e6c9" color="#4caf50" shape=box]
        errh    [label="error_handler" fillcolor="#ffe0b2" color="#f57c00" shape=box]
    }

    subgraph cluster_llm {
        label="LLM Layer"
        color="#9c27b0"
        fontcolor="#9c27b0"
        style=dashed
        claude  [label="Claude\\nSonnet 4.5"   fillcolor="#f3e5f5" color="#9c27b0" shape=box]
        gpt4    [label="GPT-4o"               fillcolor="#f3e5f5" color="#9c27b0" shape=box]
        gemini  [label="Gemini 2.5\\nFlash"    fillcolor="#f3e5f5" color="#9c27b0" shape=box]
        whisper [label="Whisper\\n(audio STT)" fillcolor="#f3e5f5" color="#9c27b0" shape=box]
    }

    subgraph cluster_stores {
        label="Data Stores"
        color="#ff6f00"
        fontcolor="#ff6f00"
        style=dashed
        cache [label="SHA-256 Cache\\n(file-based)" fillcolor="#fff3e0" color="#ff6f00" shape=cylinder]
        jsonl [label="Call History\\n(JSONL)"        fillcolor="#fff3e0" color="#ff6f00" shape=cylinder]
    }

    ui      -> cache   [label="check first" color="#ff6f00"]
    ui      -> intake  [label="cache miss"  color="#4caf50"]
    intake  -> transcr
    transcr -> summ    [label="no errors"   color="#333333"]
    transcr -> errh    [style=dashed color="#f57c00" label="errors"]
    errh    -> summ    [style=dashed color="#f57c00"]
    transcr -> whisper [label="audio"       color="#9c27b0" style=dashed]
    summ    -> claude  [color="#9c27b0" style=dashed]
    summ    -> gpt4    [color="#9c27b0" style=dashed]
    summ    -> gemini  [color="#9c27b0" style=dashed]
    qa      -> claude  [color="#9c27b0" style=dashed]
    qa      -> gpt4    [color="#9c27b0" style=dashed]
    qa      -> gemini  [color="#9c27b0" style=dashed]
    summ    -> qa
    qa      -> end_n
    end_n   -> cache   [label="save"         color="#ff6f00"]
    end_n   -> jsonl   [label="call history" color="#ff6f00"]
    end_n   -> ui      [label="result"       color="#4caf50"]
}
"""
    else:
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
        ui [label="Web Browser\\n(7 Tabs)" shape=box fillcolor="#e3f2fd" color="#1976d2"]
    }

    subgraph cluster_workflow {
        label="LangGraph Workflow V3 (17 Nodes)"
        color="#4caf50"
        fontcolor="#4caf50"
        style=dashed
        intake      [label="intake"             fillcolor="#bbdefb" color="#1976d2" shape=box]
        cust_prof   [label="customer_profile"   fillcolor="#e8f5e9" color="#2e7d32" shape=box]
        transcr     [label="transcription"      fillcolor="#bbdefb" color="#1976d2" shape=box]
        pii         [label="pii_redaction"      fillcolor="#fce4ec" color="#c62828" shape=box]
        rag         [label="rag_retrieval"      fillcolor="#e1bee7" color="#7b1fa2" shape=box]
        kb          [label="kb_retrieval"       fillcolor="#e8eaf6" color="#3949ab" shape=box]
        sent        [label="sentiment"          fillcolor="#fff9c4" color="#f9a825" shape=box]
        comp        [label="compliance_check"   fillcolor="#fce4ec" color="#c62828" shape=box]
        esc         [label="escalation_pred"    fillcolor="#ffe0b2" color="#e65100" shape=box]
        summ        [label="summarization"      fillcolor="#bbdefb" color="#1976d2" shape=box]
        tags        [label="auto_tagging"       fillcolor="#e8f5e9" color="#2e7d32" shape=box]
        qa          [label="quality_score"      fillcolor="#bbdefb" color="#1976d2" shape=box]
        coach       [label="call_coaching"      fillcolor="#e8f5e9" color="#2e7d32" shape=box]
        anom        [label="anomaly_detect"     fillcolor="#fce4ec" color="#c62828" shape=box]
        end_n       [label="end+feedback"       fillcolor="#c8e6c9" color="#4caf50" shape=box]
        errh        [label="error_handler"      fillcolor="#ffe0b2" color="#f57c00" shape=box]
    }

    subgraph cluster_llm {
        label="LLM Layer"
        color="#9c27b0"
        fontcolor="#9c27b0"
        style=dashed
        claude  [label="Claude\\nSonnet 4.5"   fillcolor="#f3e5f5" color="#9c27b0" shape=box]
        gpt4    [label="GPT-4o"               fillcolor="#f3e5f5" color="#9c27b0" shape=box]
        gemini  [label="Gemini 2.5\\nFlash"    fillcolor="#f3e5f5" color="#9c27b0" shape=box]
        whisper [label="Whisper\\n(audio STT)" fillcolor="#f3e5f5" color="#9c27b0" shape=box]
    }

    subgraph cluster_stores {
        label="Data Stores"
        color="#ff6f00"
        fontcolor="#ff6f00"
        style=dashed
        cache   [label="SHA-256 Cache\\n(file-based)" fillcolor="#fff3e0" color="#ff6f00" shape=cylinder]
        chroma  [label="ChromaDB\\n(vector store)"    fillcolor="#fff3e0" color="#ff6f00" shape=cylinder]
        jsonl   [label="Call History\\n(JSONL)"        fillcolor="#fff3e0" color="#ff6f00" shape=cylinder]
    }

    ui        -> cache      [label="check first" color="#ff6f00"]
    ui        -> intake     [label="cache miss" color="#4caf50"]
    intake    -> cust_prof
    cust_prof -> transcr
    transcr   -> pii
    pii       -> rag
    rag       -> chroma     [label="query" color="#7b1fa2" style=dashed]
    chroma    -> rag        [label="top-3" color="#7b1fa2" style=dashed]
    rag       -> kb
    kb        -> sent
    sent      -> comp
    comp      -> esc
    esc       -> summ
    summ      -> claude     [color="#9c27b0" style=dashed]
    summ      -> gpt4       [color="#9c27b0" style=dashed]
    summ      -> gemini     [color="#9c27b0" style=dashed]
    qa        -> claude     [color="#9c27b0" style=dashed]
    qa        -> gpt4       [color="#9c27b0" style=dashed]
    qa        -> gemini     [color="#9c27b0" style=dashed]
    summ      -> tags
    tags      -> qa
    qa        -> coach
    coach     -> anom
    anom      -> end_n
    end_n     -> cache      [label="save" color="#ff6f00"]
    end_n     -> chroma     [label="store\\nembedding" color="#7b1fa2"]
    end_n     -> jsonl      [label="call history" color="#ff6f00"]
    end_n     -> ui         [label="result" color="#4caf50"]
    transcr   -> whisper    [label="audio" color="#9c27b0" style=dashed]
    transcr   -> errh       [style=dashed color="#f57c00" label="errors"]
    errh      -> summ       [style=dashed color="#f57c00"]
    cust_prof -> jsonl      [label="reads history" color="#ff6f00" style=dashed]
}
"""
    st.graphviz_chart(_arch_dot)

    st.markdown("---")

    # ── Version History ────────────────────────────────────────────────────────
    st.subheader("🗂️ Version History")
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        _v1_label = "**⬅️ Active — Version 1 — Baseline**" if _pv7 == "V1" else "**Version 1 — Baseline**"
        st.markdown(_v1_label)
        if _pv7 == "V1":
            st.info("""
- 5 agents: intake, transcription, summarization, quality_score, routing
- LangGraph state machine with conditional routing
- 3 LLMs: Claude, GPT-4o, Gemini 2.5 Flash
- File-based SHA-256 cache
- JSONL call history (memory layer)
- Streamlit UI: 7 tabs
- Docker + docker-compose
- OpenAI Whisper audio transcription
        """)
        else:
            st.markdown("""
- 5 agents: intake, transcription, summarization, quality_score, routing
- LangGraph state machine with conditional routing
- 3 LLMs: Claude, GPT-4o, Gemini 2.5 Flash
- File-based SHA-256 cache
- JSONL call history (memory layer)
- Streamlit UI: 7 tabs
- Docker + docker-compose
- OpenAI Whisper audio transcription
        """)
    with v_col2:
        st.markdown("**Version 2 — Production RAG**")
        st.markdown("""
- ✅ **PIIRedactionAgent** — masks PII before LLMs
- ✅ **RAGRetrievalAgent** — ChromaDB semantic search
- ✅ **SentimentAgent** — per-turn + escalation risk
- ✅ **ComplianceCheckerAgent** — HIPAA/GDPR/PCI
- ✅ **EscalationPredictionAgent** — risk scoring
- ✅ 11-node LangGraph pipeline
- ✅ RAG context injected into all LLM prompts
- ✅ Architecture tab added
        """)
    with v_col3:
        _v3_label = "**➡️ Active — Version 3 — Full AI Suite**" if _pv7 == "V3" else "**Version 3 — Full AI Suite**"
        st.markdown(_v3_label)
        if _pv7 == "V3":
            st.info("""
- ✅ **CustomerProfileAgent** — cross-call journey
- ✅ **KnowledgeBaseAgent** — SOP compliance check
- ✅ **AutoTaggingAgent** — multi-label routing tags
- ✅ **CallCoachingAgent** — personalised coaching
- ✅ **AnomalyDetectionAgent** — outlier flagging
- ✅ **FeedbackLoopAgent** — coaching effectiveness
- ✅ 17-node LangGraph pipeline
- ✅ Zero-cost agents (no extra LLM calls for 3 of 6)
        """)
        else:
            st.markdown("""
- ✅ **CustomerProfileAgent** — cross-call journey
- ✅ **KnowledgeBaseAgent** — SOP compliance check
- ✅ **AutoTaggingAgent** — multi-label routing tags
- ✅ **CallCoachingAgent** — personalised coaching
- ✅ **AnomalyDetectionAgent** — outlier flagging
- ✅ **FeedbackLoopAgent** — coaching effectiveness
- ✅ 17-node LangGraph pipeline
- ✅ Zero-cost agents (no extra LLM calls for 3 of 6)
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
    st.subheader("🚀 Agent Roadmap — ALL 8 AGENTS COMPLETE ✅")
    st.caption("All 8 agents from the original roadmap are now implemented across V2 and V3.")

    road_col1, road_col2 = st.columns(2)
    with road_col1:
        st.markdown("**All 8 agents — implemented**")
        st.markdown("""
| Agent | Version | Business Value |
|-------|---------|---------------|
| ✅ `ComplianceCheckerAgent` | V2 | HIPAA/GDPR/PCI violation scanning |
| ✅ `EscalationPredictionAgent` | V2 | Real-time supervisor alert |
| ✅ `CallCoachingAgent` | V3 | Per-agent personalised coaching |
| ✅ `KnowledgeBaseAgent` | V3 | SOP compliance + KB retrieval |
| ✅ `CustomerProfileAgent` | V3 | Cross-call customer journey |
| ✅ `AutoTaggingAgent` | V3 | Multi-label routing + analytics |
| ✅ `AnomalyDetectionAgent` | V3 | Outlier flagging for QA review |
| ✅ `FeedbackLoopAgent` | V3 | Closed-loop coaching effectiveness |
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

    # ── Evaluation & Robustness ─────────────────────────────────────────────────
    st.subheader("🛡️ Evaluation & Robustness Architecture")
    ev_col1, ev_col2, ev_col3 = st.columns(3, gap="large")
    with ev_col1:
        st.markdown("**Input Guardrails**")
        st.markdown("""
| Check | Tool |
|---|---|
| Transcript length (10–50K chars) | `utils/validation.py` |
| Null bytes / encoding | `sanitize_transcript()` |
| Audio format + size (max 100MB) | `validate_audio_file()` |
| Call ID format (3–64 alphanum) | `validate_call_id()` |
| PII masking before LLM (V3) | `PIIRedactionAgent` regex |
""")
        st.markdown("**Output Guardrails**")
        st.markdown("""
| Check | Tool |
|---|---|
| Schema enforcement | Pydantic v2 on all outputs |
| Score ranges (0–25 / 0–100) | `QAScore` field constraints |
| Enum validation | `ResolutionStatus`, compliance levels |
| Parse failure fallback | Each agent returns safe defaults |
| State error list | `errors[]` never silently dropped |
""")
    with ev_col2:
        st.markdown("**LLM-as-Judge / Evaluation**")
        st.markdown("""
| Evaluator | Mechanism |
|---|---|
| **Cross-model benchmark** | Claude vs GPT-4o vs Gemini on same transcript — agreement = confidence |
| **Schema validator** | Pydantic rejects malformed LLM output before it enters state |
| **Anomaly detector** (V3) | Z-score vs history ± 2σ; flags outliers for human QA review |
| **Compliance checker** (V3) | LLM judges agent behaviour against HIPAA/GDPR/PCI/TCPA rules at `temp=0.1` |
| **Feedback loop** (V3) | Compares agent scores across calls — measures coaching effectiveness |
| **LangSmith traces** | Token count, latency, and full prompt/response logged per node |
""")
    with ev_col3:
        st.markdown("**Temperature Calibration by Task**")
        st.markdown("""
| Agent | Temp | Reason |
|---|---|---|
| Compliance | `0.1` | Needs consistency — low variance |
| Escalation | `0.2` | Conservative — avoid false positives |
| Sentiment | `0.3` | Some variation acceptable |
| Summarization | `0.7` | Balanced natural language |
| QA Scoring | `0.7` | Balanced scoring |
""")
        st.markdown("**Error Recovery Strategy**")
        st.markdown("""
- Every node wrapped in `try/except`
- Errors appended to `state["errors"]`
- `error_handler` routes to earliest incomplete step
- Pipeline **always produces a result** — no silent failures
- Graceful degradation: enrichment nodes (PII/RAG/KB) failures don't block summarization
""")

    st.markdown("---")

    # ── Production Checklist ───────────────────────────────────────────────────
    st.subheader("✅ Production Readiness Checklist")
    check_col1, check_col2 = st.columns(2)
    with check_col1:
        st.markdown("""
**Implemented (V1-V3)**
- ✅ PII redaction before LLM calls (GDPR / HIPAA)
- ✅ File-based caching (zero duplicate API spend)
- ✅ ChromaDB vector store (semantic memory)
- ✅ Error recovery (conditional routing)
- ✅ LangSmith observability tracing
- ✅ Multi-LLM fallback support
- ✅ Docker containerization
- ✅ AWS EC2 deploy scripts + SSM secrets
- ✅ Sentiment + escalation risk scoring
- ✅ Compliance checking (6 regulation categories)
- ✅ Customer profile & journey tracking
- ✅ Knowledge base SOP compliance
- ✅ Multi-label auto-tagging & routing
- ✅ Personalised agent coaching
- ✅ Statistical anomaly detection (z-score ± 2σ)
- ✅ Coaching feedback loop
- ✅ Pydantic schema validation on all outputs
- ✅ Input sanitization (length, encoding, format)
- ✅ Cross-model LLM benchmark (LLM-as-judge)
- ✅ Temperature calibration per agent task type
- ✅ Graceful degradation (enrichment failures don't block core output)
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


# TAB 8: V1→V3 Gains
with tab8:
    st.header("📈 V1 → V3 Processing Gains")

    if st.session_state.pipeline_version == "V1":
        st.info("Switch to **V3** in the sidebar to see how processing improves from V1 to V3 for the selected transcript.")
    elif not st.session_state.call_result:
        st.info("👈 Process a call with **V3** selected in the sidebar — this tab will show the side-by-side comparison automatically.")
    elif not st.session_state.v1_comparison_result:
        st.info("V1 baseline result not available. Re-process the current transcript with V3 selected to generate the comparison.")
    else:
        v1r = st.session_state.v1_comparison_result
        v3r = st.session_state.call_result
        extras = st.session_state.v2_extras

        st.caption(
            f"Comparing **V1 (5-node baseline)** vs **V3 (17-node full suite)** "
            f"for call `{v3r.call_id}` · LLM: `{st.session_state.active_llm}`"
        )

        # ── SECTION 1: Pipeline Overview ──────────────────────────────────────
        st.subheader("🔁 Pipeline Overview")
        ov1, ov2, ov3, ov4 = st.columns(4)
        with ov1:
            st.metric("V1 Nodes", "5", help="intake → transcription → summarization → quality_score → end")
        with ov2:
            st.metric("V3 Nodes", "17", delta="+12 agents", help="Full AI suite including PII, RAG, KB, sentiment, compliance, coaching, anomaly")
        with ov3:
            st.metric("V1 LLM Calls", "2", help="summarization + quality_score")
        with ov4:
            st.metric("V3 LLM Calls", "7", delta="+5 enrichments", help="sentiment + compliance + escalation + summarization + auto_tagging + quality_score + call_coaching")

        st.markdown("---")

        # ── SECTION 2: QA Score Comparison ────────────────────────────────────
        st.subheader("⭐ QA Score — V1 vs V3")
        st.caption("V3 scores are informed by RAG context from similar past calls and KB articles injected into the prompt.")

        v1_qa = v1r.qa_score
        v3_qa = v3r.qa_score

        if v1_qa and v3_qa:
            q1, q2, q3 = st.columns(3)
            v1_overall = v1_qa.overall_score
            v3_overall = v3_qa.overall_score
            delta_overall = v3_overall - v1_overall
            with q1:
                st.metric("V1 Overall Score", f"{v1_overall:.1f}/100")
            with q2:
                st.metric("V3 Overall Score", f"{v3_overall:.1f}/100",
                          delta=f"{delta_overall:+.1f} pts")
            with q3:
                context_gain = "✅ Context-enriched" if extras.get("rag_context") or extras.get("kb_context") else "ℹ️ No prior history yet"
                st.metric("RAG/KB Context", context_gain)

            # Per-dimension breakdown
            st.markdown("**Per-dimension score breakdown:**")
            import pandas as pd
            dims = {
                "Empathy":         (v1_qa.empathy_score,         v3_qa.empathy_score),
                "Professionalism": (v1_qa.professionalism_score, v3_qa.professionalism_score),
                "Resolution":      (v1_qa.resolution_score,      v3_qa.resolution_score),
                "Compliance":      (v1_qa.compliance_score,       v3_qa.compliance_score),
            }
            rows = []
            for dim, (s1, s3) in dims.items():
                rows.append({"Dimension": dim, "V1": s1, "V3": s3, "Delta": s3 - s1})
            df_dims = pd.DataFrame(rows).set_index("Dimension")
            dc1, dc2 = st.columns(2)
            with dc1:
                st.dataframe(
                    df_dims.style.format({"V1": "{:.1f}", "V3": "{:.1f}", "Delta": "{:+.1f}"}),
                    use_container_width=True,
                )
            with dc2:
                st.bar_chart(df_dims[["V1", "V3"]])
        else:
            st.warning("QA scores not available for comparison.")

        st.markdown("---")

        # ── SECTION 3: Summary Quality Comparison ─────────────────────────────
        st.subheader("📝 Summary Quality — V1 vs V3")
        st.caption("V3 summary is generated with RAG context (similar past calls) and KB articles injected into the prompt.")

        v1_sum = v1r.summary
        v3_sum = v3r.summary

        if v1_sum and v3_sum:
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**V1 Summary** _(no context)_")
                st.info(v1_sum.summary)
                s1c1, s1c2, s1c3 = st.columns(3)
                with s1c1:
                    st.metric("Key Points", len(v1_sum.key_points))
                with s1c2:
                    st.metric("Action Items", len(v1_sum.action_items))
                with s1c3:
                    st.metric("Resolution", v1_sum.resolution_status.value.title())
            with sc2:
                st.markdown("**V3 Summary** _(RAG + KB context injected)_")
                st.success(v3_sum.summary)
                s2c1, s2c2, s2c3 = st.columns(3)
                with s2c1:
                    kp_delta = len(v3_sum.key_points) - len(v1_sum.key_points)
                    st.metric("Key Points", len(v3_sum.key_points), delta=f"{kp_delta:+d}" if kp_delta else None)
                with s2c2:
                    ai_delta = len(v3_sum.action_items) - len(v1_sum.action_items)
                    st.metric("Action Items", len(v3_sum.action_items), delta=f"{ai_delta:+d}" if ai_delta else None)
                with s2c3:
                    st.metric("Resolution", v3_sum.resolution_status.value.title())

            # Key points side by side
            if v1_sum.key_points or v3_sum.key_points:
                with st.expander("📋 Key Points — V1 vs V3", expanded=False):
                    kp1, kp2 = st.columns(2)
                    with kp1:
                        st.markdown("**V1 Key Points**")
                        for kp in v1_sum.key_points:
                            st.markdown(f"• {kp}")
                    with kp2:
                        st.markdown("**V3 Key Points**")
                        for kp in v3_sum.key_points:
                            st.markdown(f"• {kp}")

            if v1_sum.action_items or v3_sum.action_items:
                with st.expander("✅ Action Items — V1 vs V3", expanded=False):
                    ai1, ai2 = st.columns(2)
                    with ai1:
                        st.markdown("**V1 Action Items**")
                        for ai in v1_sum.action_items:
                            st.markdown(f"• {ai}")
                    with ai2:
                        st.markdown("**V3 Action Items**")
                        for ai in v3_sum.action_items:
                            st.markdown(f"• {ai}")
        else:
            st.warning("Summaries not available for comparison.")

        st.markdown("---")

        # ── SECTION 4: V3-Exclusive Enrichments ───────────────────────────────
        st.subheader("🚀 V3-Exclusive Enrichments — Not Available in V1")
        st.caption("These insights are produced by the 12 additional agents in V3. V1 produces none of them.")

        enrichments_found = []

        # PII Redaction
        pii = extras.get("pii_summary", {})
        total_pii = sum(pii.values()) if pii else 0
        with st.expander(f"🔒 PII Redaction — {'**' + str(total_pii) + ' item(s) masked** before any LLM saw the transcript' if total_pii else 'No PII detected'}", expanded=total_pii > 0):
            st.markdown("**What V1 did:** Sent raw transcript directly to the LLM — PII exposed.")
            st.markdown("**What V3 does:** Regex-masks phone numbers, emails, SSNs, card numbers, and ZIP codes before any LLM call.")
            if total_pii:
                enrichments_found.append(f"PII: {total_pii} item(s) redacted")
                for field, count in pii.items():
                    st.caption(f"  • {field.replace('_', ' ').title()}: {count}")
            else:
                st.success("✅ No PII detected in this transcript — transcript was clean.")

        # RAG Context
        rag_ctx = extras.get("rag_context", "")
        with st.expander(f"🔍 RAG Retrieval — {'**' + str(len(rag_ctx)) + ' chars** of similar-call context injected into summarization + QA prompts' if rag_ctx else 'No prior calls yet (first run)'}", expanded=bool(rag_ctx)):
            st.markdown("**What V1 did:** Summarized and scored each call in isolation — no historical context.")
            st.markdown("**What V3 does:** Embeds the transcript, retrieves top-3 semantically similar past calls from ChromaDB, and injects them as context into every LLM prompt — so scores are calibrated against real history.")
            if rag_ctx:
                enrichments_found.append("RAG: similar-call context injected")
                with st.container():
                    st.caption(rag_ctx[:500] + ("..." if len(rag_ctx) > 500 else ""))
            else:
                st.info("Process more calls to build up the vector store — RAG context will appear on subsequent calls.")

        # KB Context
        kb_analysis = extras.get("kb_analysis")
        sop_score = kb_analysis.get("sop_compliance_score", 100.0) if kb_analysis else None
        sop_icon = ("🟢" if sop_score >= 90 else "🟡" if sop_score >= 70 else "🔴") if sop_score is not None else "⚪"
        with st.expander(f"📚 Knowledge Base — SOP Compliance {sop_icon} {sop_score:.0f}%" if sop_score is not None else "📚 Knowledge Base — not run", expanded=(sop_score is not None and sop_score < 80)):
            st.markdown("**What V1 did:** No SOP awareness — agent could deviate from procedures without detection.")
            st.markdown("**What V3 does:** Retrieves relevant SOPs and product articles, audits agent adherence, and flags knowledge gaps.")
            if kb_analysis:
                enrichments_found.append(f"KB: SOP compliance {sop_score:.0f}%")
                missed = kb_analysis.get("missed_knowledge_opportunities", [])
                if missed and missed != ["No major missed KB opportunities detected"]:
                    st.warning("Missed KB opportunities detected:")
                    for m in missed:
                        st.caption(f"→ {m}")
                else:
                    st.success("✅ No major knowledge gaps detected.")

        # Sentiment
        sentiment = extras.get("sentiment")
        if sentiment:
            risk = sentiment.get("escalation_risk", "unknown")
            risk_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk, "⚪")
            with st.expander(f"🎭 Sentiment Analysis — {risk_icon} Escalation risk: **{risk.upper()}**", expanded=risk == "high"):
                st.markdown("**What V1 did:** No sentiment tracking — agent/customer tone was invisible.")
                st.markdown("**What V3 does:** Scores every conversation turn individually, tracks sentiment trend, identifies escalation risk level and reason.")
                enrichments_found.append(f"Sentiment: escalation risk {risk}")
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Overall Sentiment", sentiment.get("overall_customer_sentiment", "N/A").title())
                with sc2:
                    st.metric("Trend", sentiment.get("customer_sentiment_trend", "N/A").title())
                with sc3:
                    st.metric("Escalation Risk", f"{risk_icon} {risk.title()}")
                if sentiment.get("escalation_risk_reason"):
                    st.caption(f"Reason: {sentiment['escalation_risk_reason']}")

        # Compliance
        compliance = extras.get("compliance")
        if compliance:
            violations = compliance.get("violations", [])
            comp_score = compliance.get("compliance_score", 100)
            comp_icon = "🟢" if comp_score >= 90 else "🟡" if comp_score >= 70 else "🔴"
            with st.expander(f"⚖️ Compliance Check — {comp_icon} Score: **{comp_score:.0f}/100**, {len(violations)} violation(s)", expanded=len(violations) > 0):
                st.markdown("**What V1 did:** No compliance scanning — HIPAA/GDPR/PCI/TCPA violations went undetected.")
                st.markdown("**What V3 does:** Scans every call for regulatory violations across HIPAA, GDPR, PCI-DSS, TCPA, and financial regulations.")
                enrichments_found.append(f"Compliance: {len(violations)} violation(s), score {comp_score:.0f}/100")
                if violations:
                    for v in violations:
                        sev_icon = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(v.get("severity", ""), "⚪")
                        st.caption(f"{sev_icon} [{v.get('category')}] {v.get('description', '')}")
                else:
                    st.success("✅ No compliance violations detected.")

        # Escalation
        escalation = extras.get("escalation")
        if escalation:
            risk_score = escalation.get("risk_score", 0)
            risk_level = escalation.get("risk_level", "unknown")
            esc_icon = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(risk_level, "⚪")
            with st.expander(f"🚨 Escalation Prediction — {esc_icon} Risk: **{risk_level.upper()}** ({risk_score:.0f}/100)", expanded=risk_level in ("high", "critical")):
                st.markdown("**What V1 did:** No escalation prediction — supervisors had no advance warning.")
                st.markdown("**What V3 does:** Combines sentiment, compliance, and transcript signals to predict escalation risk with trigger moments and recommended interventions.")
                enrichments_found.append(f"Escalation: {risk_level} ({risk_score:.0f}/100)")
                st.metric("Risk Score", f"{risk_score:.0f}/100")
                if escalation.get("recommended_intervention"):
                    st.markdown(f"**Recommended action:** {escalation['recommended_intervention']}")

        # Customer Profile
        customer_profile = extras.get("customer_profile")
        if customer_profile:
            risk_tier = customer_profile.get("risk_tier", "regular")
            tier_icon = {"vip": "⭐", "at_risk": "⚠️", "churning": "🚨", "regular": "👤"}.get(risk_tier, "👤")
            total_calls = customer_profile.get("total_calls_in_history", 0)
            with st.expander(f"👤 Customer Profile — {tier_icon} {risk_tier.upper().replace('_', ' ')} ({total_calls} prior call(s))", expanded=risk_tier in ("at_risk", "churning")):
                st.markdown("**What V1 did:** Treated every call as independent — no customer history awareness.")
                st.markdown("**What V3 does:** Loads cross-call history, assigns a risk tier (VIP / at_risk / churning / regular), and surfaces trends for this customer.")
                if total_calls > 0:
                    enrichments_found.append(f"Customer profile: {risk_tier} tier, {total_calls} prior calls")
                else:
                    enrichments_found.append("Customer profile: new customer (first call)")

        # Auto Tags
        tags = extras.get("tags")
        if tags and tags.get("primary_category"):
            cat = tags.get("primary_category", "").replace("_", " ").title()
            conf = tags.get("confidence_score", 0.0)
            with st.expander(f"🏷️ Auto Tagging — Category: **{cat}** (confidence {conf:.0%})", expanded=False):
                st.markdown("**What V1 did:** No automatic classification — routing and analytics required manual tagging.")
                st.markdown("**What V3 does:** Assigns 5-taxonomy labels (category, sub-category, intent, routing, product, sentiment tags) for downstream routing and analytics.")
                enrichments_found.append(f"Auto tags: {cat}")
                t1, t2 = st.columns(2)
                with t1:
                    st.markdown(f"**Category:** `{tags.get('primary_category', 'N/A')}`")
                    st.markdown(f"**Sub-category:** `{tags.get('sub_category', 'N/A')}`")
                with t2:
                    intents = tags.get("intent_tags", [])
                    if intents:
                        st.markdown("**Intent:** " + " ".join(f"`{i}`" for i in intents))

        # Coaching
        coaching = extras.get("coaching")
        if coaching and coaching.get("coaching_tips") is not None:
            priority = coaching.get("overall_coaching_priority", "low")
            n_tips = len(coaching.get("coaching_tips", []))
            p_icon = {"immediate": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            with st.expander(f"🎓 Agent Coaching — {p_icon} {priority.upper()} priority, {n_tips} tip(s)", expanded=priority in ("immediate", "high")):
                st.markdown("**What V1 did:** No coaching — QA scores were produced but no actionable feedback was generated.")
                st.markdown("**What V3 does:** Generates personalised coaching tips with example scripts for each weak QA dimension, prioritised by impact.")
                enrichments_found.append(f"Coaching: {n_tips} tips ({priority} priority)")
                strengths = coaching.get("agent_strengths", [])
                if strengths:
                    st.markdown("**Strengths identified:**")
                    for s in strengths:
                        st.success(f"✓ {s}")
                tips = coaching.get("coaching_tips", [])
                for tip in tips[:3]:  # show top 3
                    tip_icon = {"immediate": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(tip.get("priority", ""), "⚪")
                    st.caption(f"{tip_icon} [{tip.get('dimension', '').replace('_', ' ').title()}] {tip.get('what_to_do_instead', '')[:120]}")

        # Anomaly
        anomaly = extras.get("anomaly")
        if anomaly:
            anomaly_level = anomaly.get("anomaly_level", "normal")
            anomaly_score = anomaly.get("anomaly_score", 0.0)
            a_icon = {"normal": "✅", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(anomaly_level, "⚪")
            with st.expander(f"🔬 Anomaly Detection — {a_icon} {anomaly_level.upper()} (score {anomaly_score:.0f}/100)", expanded=anomaly.get("requires_review", False)):
                st.markdown("**What V1 did:** No anomaly detection — outlier calls were indistinguishable from normal calls.")
                st.markdown("**What V3 does:** Z-scores the call's QA score, duration, PII count, and compliance score against historical averages to flag outlier calls for QA review.")
                enrichments_found.append(f"Anomaly: {anomaly_level} (score {anomaly_score:.0f}/100)")
                if anomaly.get("requires_review"):
                    st.error("⚠️ This call has been flagged for QA review.")
                else:
                    st.success("✅ Call within normal parameters — no anomaly flagged.")

        st.markdown("---")

        # ── SECTION 5: Summary scorecard ──────────────────────────────────────
        st.subheader("🏆 Improvement Scorecard")
        st.caption("What this specific transcript gained by running through V3 instead of V1.")

        if enrichments_found:
            for item in enrichments_found:
                st.success(f"✅ {item}")
        else:
            st.info("Process more calls with V3 to build history — RAG context and customer profiles improve with each run.")

        st.markdown(f"""
**Bottom line for this transcript:**
- V1 produced a summary and a QA score in **2 LLM calls** with no context.
- V3 produced the same outputs **plus {len(enrichments_found)} enrichments** across 17 pipeline nodes.
- The LLM prompts in V3 included RAG context from similar past calls and KB articles — leading to better-calibrated scores.
- V3 surfaced actionable insights (PII exposure risk, compliance gaps, escalation signals, coaching tips) that V1 cannot produce.
        """)


# TAB 9: Pitch Deck
with tab9:

    # Pull live metrics from session state where available
    _v3_qa   = st.session_state.call_result.qa_score   if st.session_state.call_result and st.session_state.call_result.qa_score   else None
    _v1_qa   = st.session_state.v1_comparison_result.qa_score if st.session_state.v1_comparison_result and st.session_state.v1_comparison_result.qa_score else None
    _extras  = st.session_state.v2_extras or {}
    _has_live = _v3_qa and _v1_qa

    # ── slide helper ──────────────────────────────────────────────────────────
    def slide(title: str, subtitle: str = "", accent: str = "#1976d2"):
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{accent}18 0%,#ffffff 100%);
                        border-left:6px solid {accent};border-radius:8px;
                        padding:18px 24px 8px 24px;margin-bottom:8px;">
              <h2 style="margin:0 0 2px 0;color:{accent};">{title}</h2>
              {"<p style='margin:0;color:#555;font-size:0.95rem;'>" + subtitle + "</p>" if subtitle else ""}
            </div>""",
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 1 — Title
    # ════════════════════════════════════════════════════════════════════════
    slide("📞 Call Center AI", "V1 Baseline → V3 Full AI Suite  ·  A Business Case for Intelligent Call Analysis", accent="#1976d2")
    st.markdown("")
    s1c1, s1c2, s1c3, s1c4 = st.columns(4)
    with s1c1:
        st.metric("Pipeline Versions", "V1 → V3", help="5-node baseline evolved to 17-node full AI suite")
    with s1c2:
        st.metric("Agents", "5 → 17", delta="+12 AI agents")
    with s1c3:
        st.metric("Output Dimensions", "2 → 10+", delta="+8 new signals")
    with s1c4:
        st.metric("LLM Calls / Call", "2 → 7", delta="+5 enrichments")
    st.caption("Technology stack: LangGraph · LangChain · Claude Sonnet · GPT-4o · Gemini 2.5 Flash · ChromaDB · OpenAI Whisper")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 2 — Industry Problem & Business Objective
    # ════════════════════════════════════════════════════════════════════════
    slide("🏭 The Industry Problem", "Why call centers need AI-powered quality assurance", accent="#c62828")
    st.markdown("")

    p1, p2 = st.columns([1, 1], gap="large")
    with p1:
        st.markdown("#### 📊 Call Center Industry at a Glance")
        st.markdown("""
| Metric | Industry Benchmark |
|--------|-------------------|
| Avg cost per inbound call | **$5 – $12** |
| Agent annual turnover rate | **35 – 45 %** |
| Cost to hire + train one agent | **$10,000 – $20,000** |
| Calls escalated to supervisor | **15 – 20 %** |
| HIPAA violation fine (per incident) | **$100 – $50,000** |
| Avg PII data-breach cost | **$4.45 M** _(IBM 2023)_ |
| Calls QA-reviewed by humans | **< 5 %** _(manual limit)_ |
| Agent coaching frequency | **Monthly at best** |
""")
    with p2:
        st.markdown("#### ❌ What Goes Wrong Without AI")
        st.error("**Compliance blind spots** — < 5 % of calls are manually reviewed; violations go undetected for weeks")
        st.error("**Reactive escalation** — supervisors intervene after the customer hangs up, not during the call")
        st.error("**Generic coaching** — agents receive monthly feedback on aggregate scores, not the specific call that went wrong")
        st.error("**PII exposure** — transcripts sent to LLMs or stored with raw personal data — GDPR / HIPAA risk")
        st.warning("**Business objective:** Analyse **100 % of calls** in real time, flag issues before they escalate, coach agents on each individual call — at < $0.30 / call AI cost")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 3 — V1 Baseline
    # ════════════════════════════════════════════════════════════════════════
    slide("🔵 Version 1 — Baseline Pipeline", "What V1 delivered: automated summarisation and QA scoring", accent="#1565c0")
    st.markdown("")

    v1c1, v1c2 = st.columns([1, 1], gap="large")
    with v1c1:
        st.markdown("#### ✅ V1 Capabilities (5 nodes)")
        st.success("**Automated transcription** — OpenAI Whisper converts audio to text automatically")
        st.success("**LLM summarisation** — key points, action items, resolution status per call")
        st.success("**4-dimension QA scoring** — empathy, professionalism, resolution, compliance (0–100)")
        st.success("**3 LLM choices** — Claude Sonnet, GPT-4o, Gemini 2.5 Flash selectable per run")
        st.success("**Call history** — JSONL memory layer for audit trail")
        st.success("**SHA-256 cache** — no duplicate API spend on identical transcripts")
    with v1c2:
        st.markdown("#### ⚠️ V1 Limitations")
        st.error("No PII protection — raw names, phones, emails sent to LLMs")
        st.error("No historical context — each call scored in isolation, no RAG")
        st.error("No compliance scanning — HIPAA / GDPR gaps undetected")
        st.error("No sentiment tracking — escalation risk invisible until too late")
        st.error("No coaching output — QA numbers produced but no actionable next steps")
        st.error("No customer journey — repeat customers treated as strangers each call")
        st.markdown("")
        st.markdown("#### V1 Economics")
        cols = st.columns(3)
        cols[0].metric("LLM calls / call", "2")
        cols[1].metric("Est. LLM cost", "~$0.04–0.08")
        cols[2].metric("Coverage", "Summary + QA only")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 4 — V3 Full AI Suite
    # ════════════════════════════════════════════════════════════════════════
    slide("🚀 Version 3 — Full AI Suite", "12 new agents, 17-node pipeline, 10+ output dimensions per call", accent="#2e7d32")
    st.markdown("")

    v3c1, v3c2 = st.columns([1, 1], gap="large")
    with v3c1:
        st.markdown("#### New Agents Added in V3 (+12)")
        st.markdown("""
| Agent | What it does |
|-------|-------------|
| 🔒 **PIIRedactionAgent** | Masks PII before any LLM sees it |
| 🔍 **RAGRetrievalAgent** | Top-3 similar past calls injected as context |
| 📚 **KnowledgeBaseAgent** | SOP compliance audit + knowledge gaps |
| 🎭 **SentimentAgent** | Per-turn sentiment + escalation risk signal |
| ⚖️ **ComplianceCheckerAgent** | HIPAA / GDPR / PCI-DSS / TCPA scanning |
| 🚨 **EscalationPredictionAgent** | Risk score 0–100 + recommended action |
| 👤 **CustomerProfileAgent** | Cross-call journey, risk tier, churn signal |
| 🏷️ **AutoTaggingAgent** | 5-taxonomy routing + analytics labels |
| 🎓 **CallCoachingAgent** | Personalised tips + example scripts per agent |
| 🔬 **AnomalyDetectionAgent** | Z-score outlier flagging, no LLM cost |
| 🔄 **FeedbackLoopAgent** | Tracks whether coaching was adopted |
| 🗃️ **CustomerProfile** | Risk tiers: VIP / at_risk / churning / regular |
""")
    with v3c2:
        st.markdown("#### V3 Architecture Highlights")
        st.info("**17-node LangGraph pipeline** — each node is an independent, testable agent with explicit state transitions")
        st.info("**ChromaDB vector store** — embeddings persist across sessions; RAG context improves with every call processed")
        st.info("**3 zero-cost agents** — CustomerProfile, AnomalyDetection, FeedbackLoop use pure Python logic, no LLM tokens spent")
        st.info("**PII-first design** — transcript is redacted before reaching any LLM; raw text never stored in vector DB")
        st.info("**LangSmith tracing** — every agent transition and LLM call logged for observability and debugging")
        st.markdown("")
        v3_cols = st.columns(3)
        v3_cols[0].metric("LLM calls / call", "7", delta="+5 vs V1")
        v3_cols[1].metric("Est. LLM cost", "~$0.15–0.30", delta="+$0.11–0.22")
        v3_cols[2].metric("Insights produced", "10+", delta="+8 vs V1")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 5 — V1 → V3 Measurable Improvements
    # ════════════════════════════════════════════════════════════════════════
    slide("📊 V1 → V3 Measurable Improvements", "Live data from this session where available, industry benchmarks otherwise", accent="#6a1b9a")
    st.markdown("")

    if _has_live:
        st.success(f"✅ Live comparison available for call `{st.session_state.call_result.call_id}`")
        im1, im2, im3, im4 = st.columns(4)
        delta_overall = _v3_qa.overall_score - _v1_qa.overall_score
        im1.metric("V1 QA Score",   f"{_v1_qa.overall_score:.1f}/100")
        im2.metric("V3 QA Score",   f"{_v3_qa.overall_score:.1f}/100", delta=f"{delta_overall:+.1f} pts")
        pii_total = sum(_extras.get("pii_summary", {}).values())
        im3.metric("PII Items Masked", str(pii_total) if pii_total else "0", help="Items redacted before LLM")
        compliance = _extras.get("compliance", {})
        im4.metric("Compliance Score", f"{compliance.get('compliance_score', 'N/A')}" + ("/100" if compliance else ""), help="V1 had no compliance scanning")
        st.markdown("")
        import pandas as pd
        dims = {
            "Empathy":         (_v1_qa.empathy_score,         _v3_qa.empathy_score),
            "Professionalism": (_v1_qa.professionalism_score, _v3_qa.professionalism_score),
            "Resolution":      (_v1_qa.resolution_score,      _v3_qa.resolution_score),
            "Compliance":      (_v1_qa.compliance_score,       _v3_qa.compliance_score),
        }
        rows = [{"Dimension": k, "V1": v1, "V3": v3, "Δ": v3 - v1} for k, (v1, v3) in dims.items()]
        df_live = pd.DataFrame(rows).set_index("Dimension")
        lc1, lc2 = st.columns(2)
        with lc1:
            st.dataframe(df_live.style.format({"V1": "{:.1f}", "V3": "{:.1f}", "Δ": "{:+.1f}"}), use_container_width=True)
        with lc2:
            st.bar_chart(df_live[["V1", "V3"]])
    else:
        st.info("Process a call with V3 selected to see live score comparison here. Showing industry benchmarks below.")

    st.markdown("#### Industry Benchmark Improvements (AI-assisted QA vs manual-only)")
    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("QA Coverage",         "< 5 %  →  100 %",  delta="20× more calls reviewed")
    bm2.metric("Compliance Detection", "Reactive  →  Real-time", delta="Days → seconds")
    bm3.metric("Agent Score (coached)","Baseline  →  +20–30 %", delta="Per IBM/Gartner research")
    bm4.metric("Escalation Prevention","After hang-up  →  During call", delta="Supervisor alert in < 1s")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 6 — Cost-Benefit & ROI
    # ════════════════════════════════════════════════════════════════════════
    slide("💰 Cost-Benefit Analysis & ROI", "Where the $0.11–0.22 extra AI cost per call pays for itself", accent="#e65100")
    st.markdown("")

    cb1, cb2 = st.columns([1, 1], gap="large")
    with cb1:
        st.markdown("#### AI Cost per Call")
        st.markdown("""
| Pipeline | LLM calls | Est. cost / call | Cost / 1,000 calls |
|----------|-----------|-----------------|-------------------|
| **V1** | 2 | ~$0.04 – $0.08 | ~$40 – $80 |
| **V3** | 7 | ~$0.15 – $0.30 | ~$150 – $300 |
| **Δ extra spend** | +5 | **+$0.11 – $0.22** | **+$110 – $220** |
""")
        st.markdown("#### What That Extra $0.11–0.22 Prevents")
        st.markdown("""
| Risk Prevented | Per-Incident Cost | Calls / Month Exposure |
|---------------|------------------|----------------------|
| HIPAA violation | $100 – $50,000 | 1 violation averted pays **months** of V3 cost |
| PII breach (avg) | $4.45 M | 1 breach = **2M+ calls** of V3 AI cost |
| Supervisor escalation avoided | $15 – $40 extra handle time | 15 % of 10K calls = **$22,500 / month saved** |
| Agent churn (coaching reduces) | $10,000 – $20,000 / hire | 1 retained agent = **50,000 V3 calls** |
""")
    with cb2:
        st.markdown("#### 📈 ROI Scenario — 10,000 Calls / Month")
        st.markdown("""
| Cost Item | V1 | V3 | Delta |
|-----------|----|----|-------|
| AI cost (LLM) | $400–$800 | $1,500–$3,000 | +$1,100–$2,200 |
| Compliance fines avoided | $0 | $5,000–$50,000 | **+$5K–$50K saved** |
| Escalations prevented (15 % → 10 %) | $0 | $7,500 saved | **+$7,500** |
| Agent retention (2 % churn reduction) | $0 | $20,000 saved | **+$20,000** |
| **Net monthly benefit** | — | — | **+$30K – $75K** |
| **ROI on extra AI spend** | — | — | **14× – 34×** |
""")
        st.success("**Break-even:** V3's extra AI cost ($1,100–$2,200/month) is recovered by preventing **1 escalation per week** or **1 compliance incident per quarter**.")
        st.info("**Zero-cost agents:** CustomerProfile, AnomalyDetection, FeedbackLoop add insights at **$0 additional LLM cost**.")
    st.markdown("")
    st.markdown("#### 🧪 Precaching Cost — 33 Sample Transcripts × 3 Models × 4 Cache Types")
    pc1, pc2 = st.columns([1, 1], gap="large")
    with pc1:
        st.markdown("""
**What was precached (396 cache entries total):**

| Cache Type | Description | LLM Calls |
|---|---|---|
| `workflow` (V3) | Full 17-node pipeline | 6 LLM agents |
| `workflow_v1` (V1) | Baseline 5-node pipeline | 2 LLM agents |
| `v1_comparison` | Gains tab baseline | 0 (reused V1 cache) |
| `benchmark` | Summarization + QA only | 2 LLM agents |

**Avg transcript:** ~75 tokens · **Avg output:** ~200 tokens/call
""")
    with pc2:
        st.markdown("""
**Cost breakdown (est.) — 33 files × 10 LLM calls/model:**

| Model | Input cost | Output cost | **Total** |
|---|---|---|---|
| Claude Sonnet 4.5 | $0.49 | $1.04 | **~$1.53** |
| GPT-4o | $0.41 | $0.69 | **~$1.10** |
| Gemini 2.5 Flash | $0.02 | $0.04 | **~$0.07** |
| OpenAI Embeddings (RAG) | <$0.01 | — | **~$0.00** |
| **Grand total** | | | **~$2.70 – $4.50** |
""")
        st.success("**Gemini 2.5 Flash** is **22× cheaper** than Claude and **16× cheaper** than GPT-4o for the same workload — ideal for high-volume production.")
        st.info("All 396 entries are now cached — subsequent app loads cost **$0 in API fees** until the cache is cleared.")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 7 — Summary & Next Steps
    # ════════════════════════════════════════════════════════════════════════
    slide("🏆 Summary & Roadmap", "Key takeaways and what comes next", accent="#00695c")
    st.markdown("")

    sm1, sm2 = st.columns([1, 1], gap="large")
    with sm1:
        st.markdown("#### ✅ What V1 → V3 Delivers")
        st.success("**100 % call coverage** — every call analysed, not just a 5 % manual sample")
        st.success("**PII-safe by design** — data masked before any LLM or vector DB touch")
        st.success("**Real-time compliance** — HIPAA / GDPR / PCI scanned on every call")
        st.success("**Predictive escalation** — supervisors alerted during the call, not after")
        st.success("**Per-call coaching** — agents get specific feedback with example scripts")
        st.success("**Context-calibrated scores** — RAG + KB context makes QA scores reflect actual industry standards, not just the isolated call")
        st.success("**Customer intelligence** — risk tier, churn signal, cross-call journey built automatically")
        st.success("**14× – 34× ROI** at 10,000 calls/month vs extra AI cost")
    with sm2:
        st.markdown("#### 🗺️ Roadmap (Production Upgrades)")
        st.markdown("""
| Upgrade | Business Value |
|---------|---------------|
| **Real-time streaming** | Supervisors see alerts as call unfolds |
| **FastAPI REST layer** | Integrate with existing CRM / ticketing |
| **Redis cache** | Multi-instance deployment, shared cache |
| **Pinecone / Weaviate** | Managed vector DB — scales to millions of calls |
| **Async pipeline** | Parallel LLM calls reduce latency 40–60 % |
| **Role-based access** | Per-team / per-agent dashboards |
| **Webhook push** | Auto-post QA results to Slack / ServiceNow |
""")
        st.markdown("")
        st.markdown("#### 📌 Active Pipeline")
        active_pv = st.session_state.pipeline_version
        if active_pv == "V3":
            st.success(f"**{active_pv}** is selected — switch to the **📈 V1→V3 Gains** tab after processing a transcript to see live comparison numbers.")
        else:
            st.info(f"**{active_pv}** is currently selected. Switch to **V3** in the sidebar to unlock all enrichments shown in this deck.")

    st.markdown("---")
    st.caption("Sources: IBM Cost of a Data Breach 2023 · Gartner Call Center Benchmarks 2023 · SHRM Agent Turnover Report · HHS HIPAA Penalty Schedule · Deloitte Contact Centre Insights 2023")


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
