# Call Center AI Assistant

A production-quality AI-powered call center analysis system built with **LangGraph multi-agent orchestration**, **17-node pipeline**, and **8 specialized AI agents**. Processes audio and text call transcripts end-to-end with compliance checking, coaching, RAG retrieval, and real-time escalation prediction.

**Live Demo:** http://52.55.75.19:8501 (AWS EC2)
**GitHub:** https://github.com/Krishhs89/call-center-ai

---

## What It Does

Upload a call recording (WAV) or paste a transcript → the system automatically:

1. Transcribes audio via OpenAI Whisper
2. Detects PII and redacts before any LLM sees it
3. Retrieves similar past calls from ChromaDB (RAG)
4. Checks relevant SOPs from the Knowledge Base
5. Scores sentiment turn-by-turn
6. Flags HIPAA / GDPR / PCI-DSS / TCPA compliance violations
7. Predicts escalation risk in real-time
8. Generates call summary + QA score (0–100)
9. Tags the call across 5 classification taxonomies
10. Produces personalised agent coaching with example scripts
11. Detects anomalies vs historical call statistics
12. Tracks whether coaching from previous calls was adopted

---

## Architecture — V3 (17-Node LangGraph Pipeline)

```
START
  │
  ├─[intake]              Validate input, extract metadata
  ├─[customer_profile]    Load customer history, assign risk tier (zero LLM)
  ├─[transcription]       Whisper audio → text / normalize transcript
  ├─[pii_redaction]       Regex mask phone/email/SSN/card before LLM
  ├─[rag_retrieval]       ChromaDB top-3 similar past calls
  ├─[kb_retrieval]        Knowledge base SOP context injection
  ├─[sentiment]           Turn-by-turn sentiment + escalation risk signal
  ├─[compliance_check]    HIPAA/GDPR/PCI-DSS/TCPA violation scanner
  ├─[escalation_prediction] Real-time supervisor alert (cross-agent signals)
  ├─[summarization]       Summary, key points, action items, resolution status
  ├─[auto_tagging]        Multi-label: category/intent/routing/product tags
  ├─[quality_score]       0–100 QA score across 6 dimensions
  ├─[call_coaching]       Personalised tips + example scripts per weak dimension
  ├─[anomaly_detection]   Z-score vs history, flags outlier calls (zero LLM)
  └─[end]                 Assemble CallResult → FeedbackLoop post-workflow
```

---

## Agent Inventory (17 agents total)

### Core Pipeline Agents
| Agent | File | LLM | Purpose |
|-------|------|-----|---------|
| IntakeAgent | `agents/intake_agent.py` | No | Input validation, metadata extraction |
| TranscriptionAgent | `agents/transcription_agent.py` | Whisper | Audio → text, speaker normalisation |
| SummarizationAgent | `agents/summarization_agent.py` | Yes | Summary, key points, action items |
| QualityScoreAgent | `agents/quality_score_agent.py` | Yes | 6-dimension QA scoring 0–100 |
| RoutingAgent | `agents/routing_agent.py` | No | State transition logging |

### V2 Agents
| Agent | File | LLM | Purpose |
|-------|------|-----|---------|
| PIIRedactionAgent | `agents/pii_redaction_agent.py` | No | Regex mask before LLM |
| RAGRetrievalAgent | `agents/rag_retrieval_agent.py` | Embeddings | ChromaDB top-3 similar calls |
| SentimentAgent | `agents/sentiment_agent.py` | Yes | Turn sentiment + trend + escalation signal |

### V3 Agents — Full AI Suite
| Agent | File | LLM | Purpose |
|-------|------|-----|---------|
| ComplianceCheckerAgent | `agents/compliance_checker_agent.py` | Yes | HIPAA/GDPR/PCI-DSS/TCPA/Financial violations |
| EscalationPredictionAgent | `agents/escalation_prediction_agent.py` | Yes | Risk score 0–100, trigger moments, recommended intervention |
| CallCoachingAgent | `agents/call_coaching_agent.py` | Yes | Strengths, tips with example scripts, next_call_focus |
| KnowledgeBaseAgent | `agents/knowledge_base_agent.py` | Yes | SOP compliance audit, missed knowledge opportunities |
| CustomerProfileAgent | `agents/customer_profile_agent.py` | **No** | Risk tier (VIP/at_risk/churning/regular), history stats |
| AutoTaggingAgent | `agents/auto_tagging_agent.py` | Yes | 5-taxonomy classification: category/intent/routing/product |
| AnomalyDetectionAgent | `agents/anomaly_detection_agent.py` | **No** | Z-score vs history, anomaly score 0–100 |
| FeedbackLoopAgent | `agents/feedback_loop_agent.py` | **No** | Tracks coaching adoption, score delta vs last 3 calls |

**Zero-cost agents** (CustomerProfile, AnomalyDetection, FeedbackLoop) make no LLM calls — pure Python data processing against JSONL history.

---

## Project Structure

```
call_center_ai/
├── agents/                          # 17 agent modules
│   ├── intake_agent.py
│   ├── transcription_agent.py
│   ├── summarization_agent.py
│   ├── quality_score_agent.py
│   ├── routing_agent.py
│   ├── pii_redaction_agent.py       # V2
│   ├── rag_retrieval_agent.py       # V2
│   ├── sentiment_agent.py           # V2
│   ├── compliance_checker_agent.py  # V3
│   ├── escalation_prediction_agent.py # V3
│   ├── call_coaching_agent.py       # V3
│   ├── knowledge_base_agent.py      # V3
│   ├── customer_profile_agent.py    # V3 zero-cost
│   ├── auto_tagging_agent.py        # V3
│   ├── anomaly_detection_agent.py   # V3 zero-cost
│   └── feedback_loop_agent.py       # V3 zero-cost
├── workflow/
│   └── langgraph_flow.py            # 17-node LangGraph state machine
├── ui/
│   └── streamlit_app.py             # Streamlit dashboard (8 tabs)
├── utils/
│   ├── schemas.py                   # Pydantic models
│   ├── cache.py                     # SHA-256 file cache
│   ├── vector_store.py              # ChromaDB wrapper
│   ├── memory.py                    # Conversation memory
│   └── validation.py                # Input validation
├── config/
│   └── settings.py                  # Environment config
├── evaluation/
│   └── benchmark.py                 # Multi-LLM comparison
├── scripts/
│   ├── generate_sample_audio.py     # Generate 6 test WAV files
│   └── precache_all.py              # Pre-warm SHA-256 cache
├── deploy/
│   ├── cloudformation-ec2-python.yaml  # AWS EC2 no-docker deployment
│   ├── cloudformation.yaml          # AWS Docker deployment
│   ├── deploy-to-aws.sh             # Docker-based deploy script
│   └── deploy-to-aws-nodcker.sh     # No-docker deploy script
├── data/
│   ├── sample_audio/                # 6 generated WAV test files
│   └── sample_transcripts/          # 33 JSON call transcripts
├── docs/
│   └── session_log.md               # Full development session history
├── tests/
│   └── test_agents.py               # Unit + integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── QUICKSTART.md
├── PRODUCTION.md
└── README.md
```

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Krishhs89/call-center-ai.git
cd call-center-ai
```

### 2. Create virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIza...
DEFAULT_LLM=claude
CLAUDE_MODEL=claude-sonnet-4-5-20250929
GPT4_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.5-flash-preview-04-17
WHISPER_MODEL=whisper-1
MOCK_LLM=false
```

---

## Running the App

### Production mode (real LLM calls)

```bash
MOCK_LLM=false venv/bin/python -m streamlit run ui/streamlit_app.py --server.port 8501
```

### Mock mode (no API calls, instant responses)

```bash
venv/bin/python -m streamlit run ui/streamlit_app.py --server.port 8501
```

Opens at `http://localhost:8501`

---

## Sample Audio Files

Generate 6 test WAV files (requires `gtts`, `pydub`, `ffmpeg`):

```bash
venv/bin/python scripts/generate_sample_audio.py
```

Files created in `data/sample_audio/`:

| File | Tests |
|------|-------|
| `sample_audio_billing.wav` | AutoTagging=billing, Compliance=Financial, CustomerProfile |
| `sample_audio_escalation.wav` | EscalationPrediction=critical, Sentiment=negative |
| `sample_audio_tech_support.wav` | KB articles, AutoTagging=technical |
| `sample_audio_fraud.wav` | AutoTagging=fraud_security, Compliance=Financial |
| `sample_audio_complaint.wav` | Compliance violation (no recording consent), Anomaly |
| `sample_audio_account.wav` | KB=account verification, AutoTagging=account_management |

Use the sidebar "Sample Audio" panel in the UI to load and auto-transcribe any file.

---

## UI Tabs

| Tab | Content |
|-----|---------|
| Upload | Paste text, upload JSON/WAV, or load sample audio |
| Results | All 8 V3 agent panels: profile, tags, KB, coaching, anomaly, feedback |
| QA Score | 6-dimension radar chart, strengths, improvement areas |
| Benchmark | Run all 3 LLMs in parallel, compare timing/quality |
| Call History | All past calls with search + filter |
| Workflow | Live LangGraph diagram (17 nodes) |
| Architecture | Full system architecture diagram |

---

## Multi-LLM Support

| Model | Provider | Default |
|-------|----------|---------|
| `claude-sonnet-4-5-20250929` | Anthropic | Yes |
| `gpt-4o` | OpenAI | No |
| `gemini-2.5-flash-preview-04-17` | Google | No |

Switch via `DEFAULT_LLM=claude|gpt4|gemini` in `.env` or the UI dropdown.

---

## Caching

SHA-256 content-based cache at `data/cache/`. Identical transcripts return instantly from cache — no LLM call made. Cache hit shown with green indicator in UI.

Pre-warm all sample transcripts:
```bash
venv/bin/python scripts/precache_all.py
```

---

## LangSmith Tracing (Optional)

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=call-center-ai
```

---

## AWS Deployment

### CloudFormation (no Docker)

```bash
# Store API keys in SSM first
aws ssm put-parameter --name /call-center-ai/ANTHROPIC_API_KEY \
  --value "sk-ant-..." --type SecureString

# Deploy stack
aws cloudformation create-stack \
  --stack-name call-center-ai \
  --template-body file://deploy/cloudformation-ec2-python.yaml \
  --parameters \
    ParameterKey=InstanceType,ParameterValue=t2.medium \
    ParameterKey=KeyPairName,ParameterValue=your-key-pair \
    ParameterKey=DefaultLLM,ParameterValue=claude \
  --capabilities CAPABILITY_NAMED_IAM
```

Stack outputs: `AppURL` (Streamlit URL), `PublicIP`, `SSHCommand`.

See `PRODUCTION.md` for full setup guide.

---

## Docker

```bash
docker-compose up --build
```

App available at `http://localhost:8501`.

---

## Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=agents --cov=workflow

# Mock mode (no API keys needed)
MOCK_LLM=true pytest tests/ -v
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `No API keys found` | Check `.env` has at least one key |
| `ImportError: langgraph` | `pip install -r requirements.txt` |
| `Whisper failed` | Needs `OPENAI_API_KEY`; system falls back gracefully |
| `chromadb` errors | `pip install chromadb==1.5.5` |
| Port in use | `--server.port 8502` |

---

## Version History

| Version | Pipeline | Agents | Key Features |
|---------|----------|--------|--------------|
| V1 | 5 nodes | 4 | Intake, Transcription, Summarization, QA Score |
| V2 | 9 nodes | 7 | + PII Redaction, RAG (ChromaDB), Sentiment |
| V3 | 17 nodes | 15 | + Compliance, Escalation, Coaching, KB, CustomerProfile, AutoTagging, Anomaly, FeedbackLoop |

---

## Tech Stack

- **Orchestration**: LangGraph (StateGraph, 17 nodes)
- **LLMs**: Claude Sonnet 4.5, GPT-4o, Gemini 2.5 Flash
- **Audio**: OpenAI Whisper
- **Vector DB**: ChromaDB 1.5.5
- **Embeddings**: OpenAI text-embedding-3-small
- **UI**: Streamlit
- **Schemas**: Pydantic v2
- **Cache**: SHA-256 file cache
- **Tracing**: LangSmith
- **Deployment**: AWS EC2 (CloudFormation), Streamlit Cloud, Docker

---

## License

MIT License

## Authors

Built as a capstone project demonstrating production-quality Agentic AI engineering.

---

**Start with:** `MOCK_LLM=false venv/bin/python -m streamlit run ui/streamlit_app.py`
