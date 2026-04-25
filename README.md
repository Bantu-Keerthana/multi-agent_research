# 🔬 Multi-Agent Research & Automation System

A **Planner → Specialist → Aggregator** multi-agent pipeline built with [LangGraph](https://github.com/langchain-ai/langgraph) and powered by free-tier LLMs. Agents search the web, analyze data, summarize findings, and produce structured research reports — all orchestrated through a stateful graph with human-in-the-loop approval.

## Architecture

```
User Query
    ↓
┌──────────────────┐
│  Planner Agent   │  ← Groq Llama 3.3 70B (fast, free)
│  Decomposes into │
│  subtasks        │
└────────┬─────────┘
         ↓
┌─────────────────────────────────────────┐
│  LangGraph StateGraph — conditional routing  │
├──────────┬──────────┬──────────┬────────┤
│ Web      │ Data     │ Summary  │ Code   │
│ Search   │ Analysis │ Agent    │ Gen    │  ← Gemini 2.0 Flash (powerful, free)
│ Agent    │ Agent    │          │ Agent  │
└──────────┴──────────┴──────────┴────────┘
         ↓
┌──────────────────┐
│ Human Review     │  ← interrupt_before (optional gate)
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Aggregator Agent │  → Structured Markdown Report
└──────────────────┘
```

## Free Stack

| Component | Tool | Free Tier |
|-----------|------|-----------|
| Orchestration | LangGraph | Open source |
| Power LLM | Gemini 2.0 Flash | Free API |
| Fast LLM | Groq Llama 3.3 70B | Free tier |
| Web Search | DuckDuckGo | No API key needed |
| Page Scraping | Firecrawl | 500 pages/month |
| Observability | LangSmith | 5K traces/month |
| API | FastAPI | Open source |
| UI | Chainlit | Open source |
| Deploy | Cloud Run + HF Spaces | Free tiers |

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-research.git
cd multi-agent-research
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys (Gemini + Groq are free to obtain)
```

### 3. Run

```bash
# Single query
python main.py "What are the latest trends in AI agent frameworks?"

# Interactive mode
python main.py --interactive

# With human-in-the-loop (pauses for approval before final report)
python main.py --review "Validate this startup idea: AI tutoring for kids"
```

### 4. API Server

```bash
# Start the FastAPI server
uvicorn api.server:app --reload --port 8000

# SSE streaming endpoint
curl -N -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "AI trends 2025"}'

# Synchronous endpoint
curl -X POST http://localhost:8000/research/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "AI trends 2025"}'

# Human-in-the-loop: start review
curl -X POST http://localhost:8000/research/review \
  -d '{"query": "Startup validation"}'
# → returns session_id

# Approve and get report
curl -X POST http://localhost:8000/research/approve \
  -d '{"session_id": "abc123", "approved": true}'
```

### 5. Chainlit UI

```bash
chainlit run ui/app.py --port 8080
# Open http://localhost:8080
# Type /review before your query to enable human-in-the-loop
```

### 6. Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8080
```

### 7. LangSmith Observability

```bash
# Add to .env:
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multi-agent-research

# Every agent step is now traced and inspectable at smith.langchain.com
```

## Human-in-the-Loop

The system supports an enterprise-grade **human review gate**:

```
Planner → Specialists (parallel) → [PAUSE] → Human Review → Aggregator → Report
```

When enabled, the pipeline pauses after all specialist agents complete. The reviewer sees a summary of collected results and sources, then can approve, reject, or provide feedback. This pattern is critical for production systems where automated reports need human validation before delivery.

**CLI:** `python main.py --review "your query"`
**API:** `POST /research/review` → inspect → `POST /research/approve`
**UI:** Type `/review your query` in Chainlit

## Model Tiering Strategy

This project demonstrates a **cost-efficient model tiering** pattern:

| Task Type | Model | Why |
|-----------|-------|-----|
| Planning & Routing | Groq Llama 3.3 70B | Fast (~100ms), lightweight tasks |
| Web Search & Analysis | Gemini 2.0 Flash | Powerful reasoning, 1M context |
| Summarization | Gemini 2.0 Flash | Long-context capability |
| Aggregation | Gemini 2.0 Flash | Synthesis requires deep reasoning |

Every agent step logs which model was used and the latency, enabling performance comparison.

## Project Structure

```
multi-agent-research/
├── agents/
│   ├── planner.py        # Task decomposition (Groq Llama 3.3 70B)
│   ├── web_search.py     # DuckDuckGo + Wikipedia + Gemini synthesis
│   ├── analyzer.py       # Data analysis (Gemini 2.0 Flash)
│   ├── summarizer.py     # Long-context summarization (Gemini)
│   ├── code_gen.py       # Code generation (Gemini)
│   └── aggregator.py     # Report synthesis (Gemini)
├── core/
│   ├── state.py          # AgentState TypedDict schema
│   ├── config.py         # Model tiering & API key config
│   ├── llm.py            # LLM factory with traced invocations
│   ├── tracer.py         # Model tiering tracer & latency logger
│   └── graph.py          # LangGraph StateGraph + human review node
├── api/
│   └── server.py         # FastAPI + SSE streaming
├── ui/
│   └── app.py            # Chainlit real-time UI
├── tests/
│   └── test_graph.py     # Unit + integration tests
├── .github/
│   └── workflows/ci.yml  # GitHub Actions CI/CD
├── main.py               # CLI entry point
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Use Case: Startup Idea Validator

Enter a startup idea, and the system will:
1. **Search** Hacker News, GitHub, and recent news for related products
2. **Analyze** market size, competition landscape, and technical feasibility
3. **Summarize** findings into actionable insights
4. **Generate** a 1-page validation report with a feasibility score

## Roadmap

- [x] Milestone 1 — Graph skeleton + state design
- [x] Milestone 2 — Real LLM-powered agents (Gemini + Groq + DuckDuckGo)
- [x] Milestone 3 — Human-in-the-loop + model tiering logs
- [x] Milestone 4 — FastAPI SSE + Chainlit UI + LangSmith
- [ ] Milestone 5 — Docker + Cloud Run deploy + demo

## License

MIT
