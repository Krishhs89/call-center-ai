# Session Log — Call Center AI

Persistent record of all development sessions. Stored in the repo so it survives logouts and is version-controlled on GitHub.

---

## Session 3 — 2026-03-21

### Topics covered
- Discussed vector databases and where they fit in the pipeline
- Implemented Version 2 with RAG + Cache combined flow
- Added 3 new agents: PIIRedactionAgent, RAGRetrievalAgent, SentimentAgent
- Added Architecture tab (tab 7) to Streamlit UI
- Pushed V2 to GitHub → auto-deployed to Streamlit Cloud

### What was built

#### New agents
| File | Purpose |
|------|---------|
| `agents/pii_redaction_agent.py` | Regex masking of phone, email, SSN, card#, zip before LLM |
| `agents/rag_retrieval_agent.py` | Queries ChromaDB for top-3 similar past calls, formats context |
| `agents/sentiment_agent.py` | Per-turn sentiment, overall trend, escalation risk, agent tone |
| `utils/vector_store.py` | ChromaDB wrapper: store + retrieve + stats |

#### Workflow V2 (9 nodes)
```
intake → transcription → pii_redaction → rag_retrieval → sentiment
       → summarization → quality_score → end
Error path: transcription errors → error_handler → recovery
```

#### Combined Cache + RAG flow
```
New transcript arrives
       │
[1] SHA-256 cache check (exact match) → HIT: return immediately
       │ MISS
[2] RAG: ChromaDB semantic search → top-3 similar past calls
       │
[3] PII-safe redacted transcript + RAG context → LLM call
       │
[4] Save result to SHA-256 cache + ChromaDB embedding
```

#### UI changes
- Results tab: sentiment metrics (4 columns), turn-by-turn chart, PII audit panel, RAG indicator
- Workflow tab: updated to 9-node diagram
- Architecture tab (NEW): full system diagram, version history, tech stack, agent roadmap, production checklist
- Sidebar: Vector DB status (ChromaDB embedding count)

### Key decisions
- Use chromadb 1.5.5 (installed in venv)
- Embeddings via OpenAI `text-embedding-3-small` (reuses existing OPENAI_API_KEY)
- PII redaction is regex-only (zero API cost, zero latency)
- Sentiment uses a single LLM call for the full transcript (not per-turn calls)
- RAG context injected into both summarization AND quality_score prompts
- Vector embeddings stored using the PII-redacted transcript (not raw)
- `_v2_extras` attribute attached to CallResult to pass sentiment/PII/RAG data to UI
- `.env` not modified — MOCK_LLM overridden via env var when launching

### Git commit
`535d70e` — "Version 2: RAG + PII Redaction + Sentiment Analysis + Architecture tab"

### App status
- Local: running at http://localhost:8501 (PID ~55924), MOCK_LLM=false
- GitHub: https://github.com/Krishhs89/call-center-ai (master, up to date)
- Streamlit Cloud: auto-deploying (~60s after push)

### Agent roadmap discussed (Version 3+)
| Priority | Agent | Value |
|----------|-------|-------|
| High | ComplianceCheckerAgent | HIPAA/GDPR/PCI violation scanner |
| High | EscalationPredictionAgent | Real-time supervisor alert |
| High | KnowledgeBaseAgent | RAG against internal SOPs/product docs |
| Medium | CallCoachingAgent | Per-agent personalised coaching |
| Medium | CustomerProfileAgent | Cross-call customer journey |
| Medium | AutoTaggingAgent | Multi-label classification for routing |
| Low | AnomalyDetectionAgent | Flags outlier calls |
| Low | FeedbackLoopAgent | Tracks if coaching feedback improved scores |

---

## Session 2 — 2026-03-19

### Topics covered
- Fixed Streamlit Cloud deployment (sys.path + removed ragas dependency)
- Added PRODUCTION.md with full setup and functional flow documentation
- Fixed LangGraph diagram to show audio transcript panel
- Improved cache messaging in UI
- Added Dev Container folder

### Key fixes
- `sys.path.insert(0, str(_repo_root))` added to streamlit_app.py for Streamlit Cloud
- Removed ragas from requirements.txt (was causing import errors on Cloud)
- LangGraph diagram updated to match actual compiled graph
- Audio transcript panel now shows extracted text after Whisper transcription

### Git commits
- `183d252` — Fix LangGraph diagram, show audio transcript panel, improve cache messaging
- `13d668f` — Add PRODUCTION.md: full setup and functional flow documentation
- `af84ec8` — Added Dev Container Folder

---

## Session 1 — 2026-03-15 (estimated)

### What was built (Version 1 baseline)
- 5-agent LangGraph pipeline: intake → transcription → summarization → quality_score → end
- Multi-LLM support: Claude Sonnet 4.5, GPT-4o, Gemini 2.5 Flash
- Streamlit UI with 6 tabs: Upload, Results, QA Score, Benchmark, Workflow, Call History
- File-based SHA-256 cache at `data/cache/`
- JSONL call history at `data/call_history.jsonl`
- OpenAI Whisper audio transcription
- Pydantic schemas for all agent outputs
- LangSmith tracing integration
- Docker + docker-compose
- AWS deployment scripts (CloudFormation, ECR, SSM)
- config/mcp.yaml (Model Control Plane)

### Key technical decisions
- Gemini needs `method="json_mode"` for `with_structured_output()` (list[str] fields fail with function calling)
- Settings class uses @property with `load_dotenv(override=True)` for live key rotation
- CallInput.category lives in metadata dict: `r.input_data.metadata.get("category", "N/A")`
- `st.write(text)` renders markdown — use `_safe()` to escape `$` signs before markdown calls

---

## Quick Reference

### Run locally (production mode)
```bash
cd "/Users/krishnakumar/Documents/Krishna/Interview Kickstart Agentic AI/Project/Slide decks & Project Resources/call_center_ai"
MOCK_LLM=false venv/bin/python -m streamlit run ui/streamlit_app.py --server.port 8501
```

### Run in mock mode (no API calls)
```bash
venv/bin/python -m streamlit run ui/streamlit_app.py --server.port 8501
```

### Push to GitHub (triggers Streamlit Cloud deploy)
```bash
git add -A && git commit -m "message" && git push origin master
```

### Key env vars (.env)
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIza...
DEFAULT_LLM=claude
CLAUDE_MODEL=claude-sonnet-4-5-20250929
GPT4_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.5-flash
MOCK_LLM=true   # override with MOCK_LLM=false when launching
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=call-center-ai
```
