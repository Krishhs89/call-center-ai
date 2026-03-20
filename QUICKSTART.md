# Quick Start Guide - Call Center AI Assistant

Get up and running in 5 minutes!

## 1. Setup (2 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
```

## 2. Configure API Keys (2 minutes)

Edit `.env` file and add at least one API key:

```env
# Option A: Use Anthropic Claude (recommended for demo)
ANTHROPIC_API_KEY=sk-ant-...

# Option B: Use OpenAI GPT-4
OPENAI_API_KEY=sk-...

# Option C: Use Google Gemini
GOOGLE_API_KEY=...
```

Get API keys from:
- **Claude**: https://console.anthropic.com/
- **GPT-4**: https://platform.openai.com/api-keys
- **Gemini**: https://makersuite.google.com/app/apikey

## 3. Run the App (1 minute)

```bash
streamlit run ui/streamlit_app.py
```

Opens automatically at `http://localhost:8501`

## 4. Try It!

### Option A: Use Sample Transcripts

1. Go to sidebar → "Sample Transcripts"
2. Click "Load Banking Sample (Call 1)"
3. Click "Process Call" in the Upload tab
4. View results in other tabs

### Option B: Paste Your Own Transcript

1. Go to Upload tab
2. Paste transcript (or upload JSON file)
3. Click "Process Call"
4. View results

### Option C: Run Benchmark

1. Upload a call first
2. Go to "Benchmark" tab
3. Click "Run Benchmark"
4. Compare Claude vs GPT-4 vs Gemini

## File Upload Format

### Plain Text
```
Agent: Hello, how can I help you?
Customer: I'm calling about my billing.
Agent: Let me check your account...
```

### JSON Format
```json
{
  "call_id": "CALL_001",
  "transcript": "Agent: ... Customer: ...",
  "category": "billing",
  "duration_seconds": 300,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Python API (Without UI)

```python
from agents.intake_agent import IntakeAgent
from workflow.langgraph_flow import create_workflow, run_workflow

# Process input
intake = IntakeAgent()
call_input = intake.process(
    transcript_text="Agent: Hello. Customer: Hi, I need help."
)

# Run workflow
workflow = create_workflow(llm_name="claude")
result = run_workflow(workflow, call_input)

# Access outputs
print(result.summary.summary)
print(result.qa_score.overall_score)
```

## Benchmark Comparison

```python
from evaluation.benchmark import BenchmarkRunner

benchmark = BenchmarkRunner()

# Run full benchmark
result = benchmark.run_full_benchmark(
    call_id="CALL_001",
    transcript="Agent: ... Customer: ..."
)

# Compare results
print(f"Claude QA: {result.claude_qa.overall_score:.1f}")
print(f"GPT-4 QA: {result.gpt4_qa.overall_score:.1f}")
print(f"Gemini QA: {result.gemini_qa.overall_score:.1f}")
print(f"Timing: {result.timing}")
```

## Common Issues

**"No API keys found"**
- Edit `.env` and add at least one API key

**"Module not found"**
- Run `pip install -r requirements.txt`

**"Port 8501 already in use"**
- Use `streamlit run ui/streamlit_app.py --server.port 8502`

**"Whisper transcription failed"**
- This is normal if no audio is provided, or OpenAI key isn't set
- The system will use raw text as fallback

## What Each Tab Does

| Tab | Purpose |
|-----|---------|
| **Upload** | Paste/upload transcripts and process |
| **Results** | View transcript, summary, action items |
| **QA Score** | See multi-dimensional quality assessment |
| **Benchmark** | Compare Claude, GPT-4, Gemini |

## Key Features

✓ **Summarization**: Auto-generates summary, key points, action items
✓ **Quality Scoring**: Rates empathy, professionalism, resolution, compliance
✓ **Multi-LLM**: Compare Claude, GPT-4, Gemini side-by-side
✓ **Speaker Normalization**: Standardizes "Agent" and "Customer" labels
✓ **Error Handling**: Graceful fallbacks if one agent fails
✓ **Sample Data**: 3 realistic call transcripts included

## Next Steps

1. **Week 1**: Process sample transcripts, integrate Kaggle datasets
2. **Week 2**: Fine-tune prompts, add custom metrics
3. **Week 3**: Deploy with Docker, add API endpoints
4. **Deployment**: Host on Streamlit Cloud or AWS

## Documentation

- Full documentation: `README.md`
- Architecture details: `README.md` → "Architecture" section
- API reference: Docstrings in agent files
- Configuration: `.env.example`

---

**That's it! You're ready to analyze call center transcripts with AI.** 🚀

For questions, see README.md or check docstrings in Python files.
