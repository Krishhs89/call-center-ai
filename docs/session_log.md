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

## Session 5 — 2026-03-21 (continued)

### Version 3: Full AI Agent Suite — All 8 Agents Complete

#### New agents implemented (6 of 8 remaining from roadmap)

| # | Agent | File | Key output |
|---|-------|------|------------|
| 3 | CallCoachingAgent | `agents/call_coaching_agent.py` | agent_strengths, coaching_tips (with example_scripts), next_call_focus, estimated_improvement |
| 4 | KnowledgeBaseAgent | `agents/knowledge_base_agent.py` | relevant_articles, sop_compliance_score, missed_knowledge_opportunities, recommended_training_articles |
| 5 | CustomerProfileAgent | `agents/customer_profile_agent.py` | risk_tier (VIP/at_risk/churning/regular), total_calls, escalation_count, sentiment_trend — **no LLM call** |
| 6 | AutoTaggingAgent | `agents/auto_tagging_agent.py` | primary_category, sub_category, intent_tags, routing_tags, product_tags, confidence_score |
| 7 | AnomalyDetectionAgent | `agents/anomaly_detection_agent.py` | anomaly_score 0-100, flags list, requires_review, statistical z-score vs history — **no LLM call** |
| 8 | FeedbackLoopAgent | `agents/feedback_loop_agent.py` | improvement_status, score_delta, coaching_adoption_rate, improved/regressed dimensions — **no LLM call** |

#### Pipeline: 11 → 17 nodes
```
intake → customer_profile → transcription → pii_redaction → rag_retrieval
       → kb_retrieval → sentiment → compliance_check → escalation_prediction
       → summarization → auto_tagging → quality_score → call_coaching
       → anomaly_detection → end
(+ FeedbackLoopAgent runs post-workflow in run_workflow())
(+ error_handler branch)
```

#### Key design decisions
- CustomerProfileAgent, AnomalyDetectionAgent, FeedbackLoopAgent: **zero LLM cost** — pure data/stats
- KnowledgeBaseAgent: dual mode — `retrieve_context()` for prompt injection + `process()` for SOP audit
- AutoTaggingAgent: runs after summarization (has summary available to enrich tagging accuracy)
- CallCoachingAgent: runs after quality_score (consumes QA dimension_scores to prioritise weakest dims)
- FeedbackLoopAgent: post-workflow (needs assembled CallResult to compare against JSONL history)
- KB articles seeded as DEFAULT_KB_ARTICLES in knowledge_base_agent.py (7 articles covering refund, verification, escalation, billing, tech support, recording consent, shipping)
- `_v2_extras` dict now carries 13 keys: all V2 + customer_profile, kb_context, kb_analysis, tags, coaching, anomaly, feedback_loop

#### Smoke test result (MOCK_LLM=true)
- QA: 74.0/100
- Customer Profile: regular (first call)
- KB SOP: 80.0%
- Tags: billing | routing: [route_to_billing, route_to_supervisor]
- Coaching: HIGH priority, 1 tip
- Anomaly: medium (35/100)
- Escalation: critical (100/100)

#### Git commit
`f7f00fc` — "Version 3: Full AI Agent Suite — all 8 roadmap agents implemented"

#### Streamlit UI: 8 new panels added to Results tab
1. Customer Profile (risk tier, history stats, top issues)
2. Auto Tags (primary category, intent, routing, products)
3. Knowledge Base (SOP compliance, article compliance cards, training recommendations)
4. Call Coaching (strengths, prioritised tips with example scripts)
5. Anomaly Detection (anomaly flags, statistical context)
6. Feedback Loop (score delta, dimension improvements, coaching adoption)
7. V3 Processing Details (PII + RAG + KB context)

Workflow diagram, Architecture diagram, Version History, Roadmap all updated to V3.

---

## Session 4 — 2026-03-21 (continued)

### Agent Roadmap Implementation (1 of 8 started)

#### Agent 1/8: ComplianceCheckerAgent ✅ (commit `06223ff`)
**Pipeline position:** after `sentiment`, before `escalation_prediction`
**What it checks:** HIPAA, GDPR, PCI-DSS, TCPA, Financial, General (6 categories)
**Output:** violations list (severity + evidence + remediation), compliance_score 0-100, requires_immediate_review flag
**Business value:** replaces manual auditing — flags 100% of calls vs 2% sample; avoids GDPR fines up to €20M
**Real-world frequency:** 100% of calls in banking/healthcare/insurance; 20% spot-check elsewhere
**Mock mode:** pattern-based (CVV, recording consent, identity verification keywords)

#### Agent 2/8: EscalationPredictionAgent ✅ (commit `4efd841`)
**Pipeline position:** after `compliance_check`, before `summarization`
**What it produces:** risk_score 0-100, risk_level (low/medium/high/critical), trigger moments, recommended_intervention
**Consumes:** sentiment + compliance signals (cross-agent enrichment)
**Business value:** supervisor intervenes 2-3 turns early → 5 min AHT reduction; ~$2.1M/year in 500-seat center
**Real-world frequency:** 100% scored; supervisor alert fires on ~5-8%
**Fix logged:** Python 3.9 incompatibility with `dict | None` syntax → use `Optional[dict]`

#### Agents still to implement (6 remaining):
3. CallCoachingAgent
4. KnowledgeBaseAgent
5. CustomerProfileAgent
6. AutoTaggingAgent
7. AnomalyDetectionAgent
8. FeedbackLoopAgent

### Pipeline as of this session
```
intake → transcription → pii_redaction → rag_retrieval → sentiment
       → compliance_check → escalation_prediction → summarization → quality_score → end
```
(11 nodes total)

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
