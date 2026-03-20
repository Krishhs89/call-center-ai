# Call Center AI Assistant

A production-quality AI-powered call center analysis system that processes customer service transcripts with multi-agent architecture, LangGraph orchestration, and multi-LLM benchmarking capabilities.

## Features

- **Multi-Agent Architecture**: Specialized agents for intake, transcription, summarization, and quality scoring
- **Multi-LLM Support**: Compare Claude, GPT-4, and Google Gemini on the same transcripts
- **Structured Output**: Pydantic-based schemas for reliable, validated data structures
- **LangGraph Workflow**: Orchestrated workflow with conditional routing and error handling
- **Streamlit UI**: Interactive dashboard with upload, analysis, and benchmarking tabs
- **Quality Assessment**: Multi-dimensional scoring (empathy, professionalism, resolution, compliance)
- **Production Ready**: Type hints, logging, error handling, and comprehensive documentation

## Project Structure

```
call_center_ai/
├── agents/                          # Agent implementations
│   ├── intake_agent.py             # Input validation and processing
│   ├── transcription_agent.py       # Transcript normalization (Whisper support)
│   ├── summarization_agent.py       # Call summarization with configurable LLM
│   ├── quality_score_agent.py       # Multi-dimensional QA scoring
│   └── routing_agent.py             # Workflow orchestration and routing
├── workflow/
│   └── langgraph_flow.py           # LangGraph state machine and workflow
├── evaluation/
│   └── benchmark.py                 # Multi-LLM comparison benchmarking
├── ui/
│   └── streamlit_app.py            # Interactive dashboard
├── utils/
│   └── schemas.py                   # Pydantic models and data structures
├── config/
│   └── settings.py                  # Configuration and environment variables
├── data/
│   └── sample_transcripts/          # Sample call transcripts (JSON)
├── tests/
│   └── test_agents.py              # Unit and integration tests
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
└── README.md                        # This file
```

## Installation

### 1. Clone/Setup

```bash
cd call_center_ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
DEFAULT_LLM=claude
```

#### API Key Setup

**Anthropic (Claude):**
- Go to https://console.anthropic.com/
- Create an API key
- Set `ANTHROPIC_API_KEY` in `.env`

**OpenAI (GPT-4 & Whisper):**
- Go to https://platform.openai.com/api-keys
- Create an API key
- Set `OPENAI_API_KEY` in `.env`

**Google (Gemini):**
- Go to https://makersuite.google.com/app/apikey
- Create an API key
- Set `GOOGLE_API_KEY` in `.env`

## Usage

### Run Streamlit App

```bash
streamlit run ui/streamlit_app.py
```

The app will open at `http://localhost:8501`

### Use Cases

#### 1. Upload and Analyze
- Paste transcript text or upload JSON/text file
- Click "Process Call"
- View transcript, summary, and QA scores

#### 2. View Results
- See normalized transcript with speaker labels
- Review auto-generated summary
- Check key points and action items
- View resolution status (resolved/unresolved/escalated)

#### 3. Quality Assessment
- View overall QA score (0-100)
- See breakdown by dimension (empathy, professionalism, resolution, compliance)
- Review agent strengths and improvement areas

#### 4. Multi-LLM Benchmarking
- Compare Claude, GPT-4, and Gemini simultaneously
- Run summarization, QA scoring, or full benchmark
- View timing comparisons and token usage
- Identify fastest/best-performing model

### Python API Usage

```python
from agents.intake_agent import IntakeAgent
from workflow.langgraph_flow import create_workflow, run_workflow

# Process intake
intake = IntakeAgent()
call_input = intake.process(transcript_text="Agent: Hello. Customer: Hi.")

# Create and run workflow
workflow = create_workflow(llm_name="claude")
result = run_workflow(workflow, call_input)

# Access results
print(f"Summary: {result.summary.summary}")
print(f"QA Score: {result.qa_score.overall_score}/100")
```

### Using Benchmark

```python
from evaluation.benchmark import BenchmarkRunner

benchmark = BenchmarkRunner()
result = benchmark.run_full_benchmark(
    call_id="CALL_001",
    transcript="Agent: ... Customer: ...",
)

# Compare results
comparison = benchmark.compare_results(result)
print(f"Fastest: {comparison['fastest_model']}")
```

## Dataset Integration (Week 1)

### Option 1: Use Sample Transcripts

Sample transcripts are included:
- `sample_call_1.json` - Banking (fraud/billing issue)
- `sample_call_2.json` - Telecom (billing error)
- `sample_call_3.json` - Healthcare (appointment scheduling)

Load them in the UI's sidebar or programmatically:

```python
import json
with open('data/sample_transcripts/sample_call_1.json') as f:
    data = json.load(f)
    transcript = data['transcript']
```

### Option 2: Kaggle Datasets

Download call center transcripts from Kaggle:

1. Go to https://kaggle.com/datasets
2. Search for "call center" or "customer service transcripts"
3. Download datasets (CSV/JSON format)
4. Convert to JSON format matching schema in `data/sample_transcripts/`
5. Place in `data/` directory
6. Load via upload tab

**Expected format:**
```json
{
  "call_id": "CALL_001",
  "transcript": "Agent: ... Customer: ...",
  "category": "billing",
  "duration_seconds": 300,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_agents.py -v

# With coverage
pytest tests/ --cov=agents --cov=workflow --cov=evaluation
```

**Note:** Some tests require API keys. Tests will be skipped if keys are unavailable.

## Architecture

### Data Flow

```
User Input (Transcript/File)
    ↓
[IntakeAgent] - Validate & extract metadata
    ↓
[TranscriptionAgent] - Normalize speaker labels
    ↓
[SummarizationAgent] - Extract summary, key points, action items
    ↓
[QualityScoreAgent] - Score on 4 dimensions
    ↓
CallResult (combined output)
```

### LangGraph Workflow

The workflow uses conditional routing:

```
START
  ↓
[intake_node]
  ↓
[transcription_node]
  ├─→ (if error) [error_handler]
  │              ↓ (resume)
  ├─→ (if success) [summarization_node]
  └─────────────→ [quality_score_node]
                    ↓
                   [end_node]
                    ↓
                   END
```

### Agents

**IntakeAgent**
- Validates input format (text/JSON)
- Extracts metadata (call_id, category, duration)
- Handles both plain text and structured files

**TranscriptionAgent**
- Integrates OpenAI Whisper for audio (if provided)
- Normalizes speaker labels to "Agent" and "Customer"
- Handles multiple speaker formats

**SummarizationAgent**
- Uses configurable LLM (Claude/GPT-4/Gemini)
- Structured output via function calling
- Extracts: summary, key points, action items, resolution status

**QualityScoreAgent**
- Scores on 4 dimensions (0-25 each, total 0-100)
- Provides strengths and improvement suggestions
- Uses structured output for reliable scoring

**RoutingAgent**
- Manages state transitions
- Logs workflow progress for debugging/monitoring
- Handles error recovery

## Configuration

### Environment Variables

See `.env.example` for all options:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | GPT-4 and Whisper API key |
| `GOOGLE_API_KEY` | Gemini API key |
| `DEFAULT_LLM` | Default model (claude/gpt4/gemini) |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing (optional) |
| `LANGCHAIN_API_KEY` | LangSmith API key (optional) |

### Settings

Modify in `config/settings.py`:

```python
settings.CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
settings.GPT4_MODEL = "gpt-4-turbo"
settings.GEMINI_MODEL = "gemini-2.0-flash"
settings.WHISPER_MODEL = "whisper-1"
```

## LangSmith Integration (Optional)

Enable LangSmith tracing for monitoring:

```bash
# In .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=call-center-ai
```

All workflow transitions are logged with LangSmith-compatible format:
```
[ROUTE] Call CALL_001: intake -> transcription
[NODE] transcription: Processing call CALL_001
[STATE_TRANSITION] CALL_001: transcription -> summarization
```

## Benchmarking

### Benchmark Types

1. **Summarization**: Compare summary quality and timing
2. **QA Scoring**: Compare scoring consistency
3. **Full Benchmark**: Complete pipeline for all models

### Results

Benchmark results include:
- Processing time per model
- Estimated token counts
- Direct output comparison
- Error handling/fallbacks

```python
result = benchmark.run_full_benchmark("CALL_001", transcript)
print(result.timing)           # {'claude': 1.5, 'gpt4': 2.1, 'gemini': 1.8}
print(result.token_counts)     # Token usage per model
print(result.claude_qa)        # Claude QA score
```

## Troubleshooting

### API Key Issues

```
ValueError: No API keys found
```
Solution: Ensure `.env` file exists with at least one API key set.

### LangChain/LangGraph Import Errors

```
ImportError: No module named 'langgraph'
```
Solution: `pip install langgraph>=0.0.23`

### Whisper Transcription Failed

```
Whisper transcription failed: ...
```
Solution: Ensure OpenAI API key is set. Whisper is optional; system will skip with warning.

### LLM Model Not Available

```
ValueError: OPENAI_API_KEY not configured
```
Solution: Set required API keys in `.env` for the model you want to use.

### Streamlit Port Already in Use

```
streamlit run ui/streamlit_app.py --server.port 8502
```

## Performance Notes

- **Claude**: Generally fastest, excellent summarization quality
- **GPT-4**: More detailed analysis, longer processing time
- **Gemini**: Good balance of speed and quality
- **Parallel Benchmarking**: Full benchmark runs ~3 threads simultaneously

Typical processing time per call:
- Summarization: 1-3 seconds per model
- QA Scoring: 1-3 seconds per model
- Full Benchmark: 3-10 seconds total (parallel)

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "ui/streamlit_app.py"]
```

### API Endpoint

Wrap workflow in FastAPI:

```python
from fastapi import FastAPI
from workflow.langgraph_flow import create_workflow, run_workflow

app = FastAPI()
workflow = create_workflow()

@app.post("/analyze")
def analyze_call(transcript: str, call_id: str):
    # Process and return result
    pass
```

## Capstone Checklist

- ✓ Multi-agent architecture with specialized agents
- ✓ LangGraph orchestration with conditional routing
- ✓ Structured output with Pydantic validation
- ✓ Multi-LLM support (Claude, GPT-4, Gemini)
- ✓ Benchmark comparison functionality
- ✓ Interactive Streamlit UI
- ✓ Sample transcripts for Week 1
- ✓ Error handling and fallbacks
- ✓ Production-quality code (type hints, logging, docstrings)
- ✓ Comprehensive testing
- ✓ Complete documentation

## Support & Resources

- LangChain Docs: https://python.langchain.com/
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- Streamlit Docs: https://docs.streamlit.io/
- Anthropic Docs: https://docs.anthropic.com/
- OpenAI Docs: https://platform.openai.com/docs/
- Google Gemini: https://ai.google.dev/

## License

MIT License

## Authors

Built as a comprehensive capstone project demonstrating AI/ML engineering best practices.

---

**Ready to analyze customer service calls with AI! Start with `streamlit run ui/streamlit_app.py`**
