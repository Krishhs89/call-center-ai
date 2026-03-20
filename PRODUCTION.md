# Call Center AI Assistant — Production & Functional Guide

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Functional Flow](#2-functional-flow)
3. [Agent Reference](#3-agent-reference)
4. [Data Schemas](#4-data-schemas)
5. [Configuration](#5-configuration)
6. [Local Setup](#6-local-setup)
7. [Streamlit Cloud Deployment](#7-streamlit-cloud-deployment)
8. [AWS Production Deployment](#8-aws-production-deployment)
9. [Updating the App](#9-updating-the-app)
10. [Monitoring & Observability](#10-monitoring--observability)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Overview

The Call Center AI Assistant is a multi-agent system that converts raw call transcripts (or audio) into structured summaries and quality assessments. It uses a LangGraph state machine to orchestrate five specialised agents, supports three LLMs, and exposes everything via a Streamlit web interface.

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
│            (Paste text / Upload JSON or audio)              │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Streamlit UI   │  ui/streamlit_app.py
                    │  (5 tabs)       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  LangGraph      │  workflow/langgraph_flow.py
                    │  Workflow       │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   Intake Agent    Transcription Agent   Summarization Agent
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    QualityScore Agent
                             │
                    Routing Agent (error handling)
                             │
                    ┌────────▼────────┐
                    │   CallResult    │  Pydantic model
                    │ (final output)  │
                    └─────────────────┘
```

**Tech Stack**

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM integration | LangChain (Anthropic / OpenAI / Google) |
| Language Models | Claude Sonnet, GPT-4o, Gemini 2.5 Flash |
| Audio transcription | OpenAI Whisper API |
| Structured output | Pydantic v2 + LangChain function calling |
| UI | Streamlit |
| Caching | File-based SHA-256 cache (`data/cache/`) |
| Memory | File-based JSONL call history (`data/call_history.jsonl`) |
| Model control plane | `config/mcp.yaml` |
| Container | Docker / docker-compose |
| Cloud hosting | Streamlit Community Cloud (live) / AWS EC2 (ready) |
| Source control | GitHub — https://github.com/Krishhs89/call-center-ai |

---

## 2. Functional Flow

### 2.1 End-to-End Pipeline

```
User submits transcript or audio
          │
          ▼
┌─────────────────────┐
│   1. INTAKE AGENT   │  Validates format, assigns call_id,
│   intake_agent.py   │  extracts metadata (category,
└──────────┬──────────┘  duration, timestamp) from JSON
           │
           ▼
┌──────────────────────────┐
│  2. TRANSCRIPTION AGENT  │  If audio: calls OpenAI Whisper API
│  transcription_agent.py  │  Normalises speaker labels to
└──────────┬───────────────┘  "Agent:" / "Customer:" format
           │
           │  ← conditional routing (LangGraph edge) →
           │  errors? → error_handler → tries to recover
           │  ok?    → summarization
           ▼
┌──────────────────────────┐
│  3. SUMMARIZATION AGENT  │  Sends transcript to chosen LLM
│  summarization_agent.py  │  with structured output prompt.
└──────────┬───────────────┘  Returns: summary, key_points,
           │                   action_items, customer_issue,
           │                   resolution_status
           ▼
┌──────────────────────────┐
│  4. QUALITY SCORE AGENT  │  Scores agent performance on
│  quality_score_agent.py  │  4 dimensions (0–25 each):
└──────────┬───────────────┘  • Empathy
           │                   • Professionalism
           │                   • Resolution effectiveness
           │                   • Compliance
           ▼
┌──────────────────────────┐
│   5. ROUTING AGENT       │  Manages state transitions,
│   routing_agent.py       │  logs every step, handles
└──────────┬───────────────┘  errors and fallback routing
           │
           ▼
┌──────────────────────────┐
│      CallResult          │  Pydantic model combining all
│      (final output)      │  agent outputs. Persisted to
└──────────┬───────────────┘  data/call_history.jsonl
           │
           ▼
     Streamlit UI
  displays results across
  Results / QA Score tabs
```

### 2.2 LangGraph State Machine

```
START
  │
  ▼
[intake] ──────────────────────────────────┐
  │                                         │
  ▼                                         │
[transcription]                             │
  │                                         │
  ├─ errors? ──► [error_handler] ──► tries to resume
  │                                         │
  ▼                                         │
[summarization] ◄────────────────────────── ┘
  │
  ▼
[quality_score]
  │
  ▼
[end]
  │
  ▼
END
```

**Conditional edges:**
- `transcription → error_handler` if any errors in state
- `error_handler → summarization` if no summary yet
- `error_handler → quality_score` if summary exists but no QA score
- `error_handler → end` if both exist

### 2.3 LLM Structured Output

Both Summarization and QualityScore agents use LangChain's `with_structured_output()` which:
1. Sends the transcript + instruction prompt to the LLM
2. Forces the response to conform to a Pydantic schema
3. Validates and deserialises the response automatically

For Gemini, `method="json_mode"` is used instead of function calling (Gemini rejects `list[str]` fields in function calling mode).

### 2.4 Multi-LLM Benchmark Mode

When the user runs a benchmark, all three models process the same transcript in parallel threads:

```
Transcript
    │
    ├──► Claude  ──► SummaryOutput + QAScore + timing
    ├──► GPT-4o  ──► SummaryOutput + QAScore + timing
    └──► Gemini  ──► SummaryOutput + QAScore + timing
                          │
                          ▼
                   BenchmarkResult
                (side-by-side comparison)
```

Results are cached per model per transcript (SHA-256 hash) so re-running benchmarks is instant.

### 2.5 Mock Mode

When `MOCK_LLM=true` (set in `.env`), all LLM calls are bypassed. The agents parse the transcript locally and return context-aware responses for: billing, technical, healthcare, travel, orders, compliments, and angry/escalated calls. Useful for UI testing without spending API credits.

---

## 3. Agent Reference

### IntakeAgent (`agents/intake_agent.py`)

| Input | Output |
|---|---|
| `file_path` (JSON/text) or `transcript_text` (string) | `CallInput` Pydantic model |

- Generates a unique `call_id` if not provided (`CALL_YYYYMMDD_HHMMSS_ffffff`)
- Parses JSON files with keys: `transcript`, `text`, `content`, `call_id`, `category`, `duration_seconds`
- Validates transcript is non-empty
- Sets `audio_path` if the file is `.mp3/.wav/.m4a/.webm`

---

### TranscriptionAgent (`agents/transcription_agent.py`)

| Input | Output |
|---|---|
| `call_id`, `audio_path` (optional), `transcript_text` (optional) | `TranscriptOutput` |

- If `audio_path` provided: calls **OpenAI Whisper API** (`whisper-1` model)
- Normalises speaker labels — maps `Rep`, `Representative`, `Support` → `Agent`; `Client`, `Caller`, `User` → `Customer`
- Falls back to raw text if Whisper fails

---

### SummarizationAgent (`agents/summarization_agent.py`)

| Input | Output |
|---|---|
| `call_id`, `transcript` (string) | `SummaryOutput` |

**Prompt extracts:**
- `summary` — 2–3 sentence overview
- `key_points` — list of 3–5 discussion points
- `action_items` — list of 2–5 follow-up tasks
- `customer_issue` — primary concern raised
- `resolution_status` — `resolved` / `unresolved` / `escalated`

Falls back to a minimal valid response if the LLM call fails.

---

### QualityScoreAgent (`agents/quality_score_agent.py`)

| Input | Output |
|---|---|
| `call_id`, `transcript` (string) | `QAScore` |

**Scores (0–25 each, total 0–100):**

| Dimension | What it measures |
|---|---|
| Empathy | Did the agent acknowledge emotions and show genuine concern? |
| Professionalism | Was communication clear, courteous, and on-brand? |
| Resolution | Did the agent effectively solve or address the issue? |
| Compliance | Were policies and procedures correctly followed? |

Also returns `tone` description, `strengths` list, and `improvements` list.

---

### RoutingAgent (`agents/routing_agent.py`)

Stateless helper used by the LangGraph nodes. Provides:
- `route_intake()`, `route_transcription()`, `route_summarization()`, `route_quality_score()`
- `handle_error()` — attempts recovery by routing to the next incomplete step
- `log_state_transition()` — LangSmith-compatible structured logging
- `end_workflow()` — marks state as `completed`, logs final summary

---

## 4. Data Schemas (`utils/schemas.py`)

```python
CallInput          # call_id, audio_path?, transcript_text?, metadata{}
TranscriptOutput   # call_id, transcript, speakers[], duration_seconds?
SummaryOutput      # call_id, summary, key_points[], action_items[],
                   # customer_issue, resolution_status (enum)
QAScore            # call_id, overall_score, empathy_score,
                   # professionalism_score, resolution_score,
                   # compliance_score, tone, strengths[], improvements[]
CallResult         # Combines all above + errors[], current_step
BenchmarkResult    # claude/gpt4/gemini summary+qa + timing{} + token_counts{}
```

---

## 5. Configuration

### Environment Variables (`.env`)

```env
# LLM API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Model selection
DEFAULT_LLM=claude              # claude | gpt4 | gemini
CLAUDE_MODEL=claude-sonnet-4-5-20250929
GPT4_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.5-flash
WHISPER_MODEL=whisper-1

# App behaviour
MOCK_LLM=false                  # true = skip all LLM calls (for testing)
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=100

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=call-center-ai
```

### Model Control Plane (`config/mcp.yaml`)

Controls routing, fallback, cost limits, and observability:

```yaml
models:
  primary: claude
  fallback_sequence: [claude, gpt4, gemini]

routing:
  max_retries: 3
  fallback_on_error: true
  fallback_on_quota: true

cost_controls:
  max_cost_per_call_usd: 0.10
  daily_budget_usd: 10.00

observability:
  trace_agent_transitions: true
  track_latency: true
  track_token_usage: true
```

---

## 6. Local Setup

```bash
# 1. Clone repo
git clone https://github.com/Krishhs89/call-center-ai.git
cd call-center-ai

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your API keys

# 5. Run the app
python -m streamlit run ui/streamlit_app.py --server.port 8501
# Open http://localhost:8501
```

**Minimum requirement:** At least one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` must be set. Set `MOCK_LLM=true` to run without any API keys.

---

## 7. Streamlit Cloud Deployment

**Current live deployment** — auto-deploys from GitHub on every push to `master`.

### Initial Setup (one-time)

1. Go to **https://share.streamlit.io** → sign in with GitHub
2. Click **New app**
3. Set:
   - Repository: `Krishhs89/call-center-ai`
   - Branch: `master`
   - Main file: `ui/streamlit_app.py`
4. Click **Advanced settings** → add secrets:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY = "sk-..."
GOOGLE_API_KEY = "AIza..."
DEFAULT_LLM = "claude"
MOCK_LLM = "false"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
GPT4_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.5-flash"
WHISPER_MODEL = "whisper-1"
```

5. Click **Deploy**

### Updating secrets

Streamlit Cloud dashboard → your app → **Settings** → **Secrets** → edit → **Save** → app restarts automatically.

---

## 8. AWS Production Deployment

All AWS infrastructure files are in `deploy/`. The Docker image is already pushed to ECR and API keys are stored in SSM.

### Prerequisites (already done)

- [x] AWS CLI installed and configured (`aws configure`)
- [x] Docker Desktop running
- [x] ECR repository created: `434854365027.dkr.ecr.us-east-1.amazonaws.com/call-center-ai:latest`
- [x] API keys stored in SSM: `/call-center-ai/ANTHROPIC_API_KEY`, etc.
- [x] EC2 Key Pair: `call-center-ai-key` (`~/.ssh/call-center-ai-key.pem`)

### Deploy (one command)

```bash
cd call-center-ai
./deploy/deploy-to-aws.sh
```

This script:
1. Builds Docker image (`linux/amd64`) and pushes to ECR
2. Stores API keys in SSM Parameter Store as `SecureString`
3. Deploys CloudFormation stack:
   - EC2 t3.small (Amazon Linux 2023)
   - Elastic IP (stable public address)
   - IAM role (ECR read + SSM read permissions)
   - Security group (ports 22, 80, 8501)
4. EC2 UserData: installs Docker, authenticates ECR, fetches keys from SSM, runs container
5. Outputs the public URL

### CloudFormation Resources

| Resource | Type | Purpose |
|---|---|---|
| `AppRole` | IAM Role | Allows EC2 to read ECR images and SSM secrets |
| `AppInstanceProfile` | IAM InstanceProfile | Attaches role to EC2 |
| `AppSecurityGroup` | EC2 Security Group | Opens ports 22, 80, 8501 |
| `AppEIP` | Elastic IP | Stable public IP that survives reboots |
| `AppInstance` | EC2 Instance | t3.small running the Docker container |

### SSH and Logs

```bash
# SSH into the instance
ssh -i ~/.ssh/call-center-ai-key.pem ec2-user@<elastic-ip>

# View container logs
docker logs -f call-center-ai

# Restart container
docker restart call-center-ai
```

### Re-deploy after code changes

```bash
# Push new image to ECR
export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin"
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    434854365027.dkr.ecr.us-east-1.amazonaws.com

docker build --platform linux/amd64 \
  -t 434854365027.dkr.ecr.us-east-1.amazonaws.com/call-center-ai:latest .
docker push 434854365027.dkr.ecr.us-east-1.amazonaws.com/call-center-ai:latest

# SSH in and pull latest
ssh -i ~/.ssh/call-center-ai-key.pem ec2-user@<elastic-ip>
docker pull 434854365027.dkr.ecr.us-east-1.amazonaws.com/call-center-ai:latest
docker restart call-center-ai
```

---

## 9. Updating the App

### Code changes (Streamlit Cloud auto-deploys)

```bash
# Make your changes, then:
git add .
git commit -m "describe change"
git push
# Streamlit Cloud redeploys automatically within ~60 seconds
```

### Rotating API keys

**Streamlit Cloud:** Dashboard → Settings → Secrets → update value → Save

**AWS SSM:**
```bash
aws ssm put-parameter \
  --name "/call-center-ai/ANTHROPIC_API_KEY" \
  --value "new-key-here" \
  --type SecureString \
  --overwrite \
  --region us-east-1
# Then restart the EC2 container to pick up the new key:
ssh -i ~/.ssh/call-center-ai-key.pem ec2-user@<elastic-ip> \
  "docker restart call-center-ai"
```

---

## 10. Monitoring & Observability

### Application Logs

Every agent logs structured messages at INFO/WARNING/ERROR level:

```
[NODE] intake: Processing call CALL_20260319_123456
[NODE] transcription: Transcript normalized
[NODE] summarization: Summary generated - resolved
[NODE] quality_score: QA score 87.0/100
[WORKFLOW_END] CALL_20260319_123456: Transcript: 1240 chars | Summary: resolved | QA Score: 87.0/100
```

### LangSmith Tracing (optional)

Add to `.env` or Streamlit secrets:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=call-center-ai
```

All LLM calls and agent transitions are then visible at **https://smith.langchain.com**.

### Call History

All processed calls are persisted to `data/call_history.jsonl` (one JSON object per line). Accessible via the Streamlit **Workflow** tab → State Inspector.

### Caching

Results are cached by SHA-256 hash of the transcript + LLM name in `data/cache/`. Cache TTL is 7 days (configurable in `config/mcp.yaml`). The Streamlit UI shows a cache hit indicator and allows manual cache clearing.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: agents` | Python path issue | Ensure `sys.path` insert is at top of `streamlit_app.py` (already fixed) |
| `ValueError: No API keys found` | Missing env vars | Set at least one LLM key in `.env` or Streamlit secrets |
| `ValueError: OPENAI_API_KEY not configured` | Trying to use GPT-4 without key | Set `OPENAI_API_KEY` or switch to `DEFAULT_LLM=claude` |
| Whisper transcription fails | No `OPENAI_API_KEY` | Set key or provide text transcript instead of audio |
| Gemini returns parse error | Structured output issue | Already handled via `json_mode` — if persists, check `GOOGLE_API_KEY` |
| Streamlit Cloud redeploy slow | Cold start | Normal — first deploy takes 3–5 min, subsequent ~60s |
| EC2 blocked (`account not verified`) | New AWS account | Wait for AWS verification email (up to 24h), then re-run deploy script |
| Port 8501 already in use (local) | Previous instance running | `pkill -f streamlit` or use `--server.port 8502` |
| Docker build fails on M1/M2 Mac | Architecture mismatch | Always use `--platform linux/amd64` flag for AWS deployments |
