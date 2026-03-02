# DeepLens — Product Requirements Document

## 1. Project Overview

**DeepLens** is a multi-agent research and data analysis system built with LangGraph. Users provide a research question in natural language, and multiple specialized AI agents collaborate to gather data from YouTube and the web, analyze it, and produce a structured insight report with visualizations.

### Why This Project Exists

This is a **learning project and portfolio piece**, not a production SaaS. It exists to:

1. **Learn** — Hands-on experience building multi-agent systems with LangGraph
2. **Demonstrate** — Proof of agent orchestration skills for AI Engineer job applications
3. **Complement Doctify** — Show breadth alongside Doctify's depth

**Portfolio Narrative:**
| Project | What It Proves |
|---------|---------------|
| Doctify | "I understand RAG deeply — built retrieval, generation, reranking from scratch" |
| DeepLens | "I can orchestrate multi-agent systems — agents decide their own workflow" |

### What Makes DeepLens Different From Existing Research Agents

Most LangGraph research agent tutorials do: web search then summarize.

DeepLens adds **structured data analysis**:
- YouTube API data (not just web articles) — real metrics, real comments
- Statistical analysis on collected data (pandas)
- Sentiment analysis on comments
- Chart/visualization generation
- The Supervisor decides when to loop back for more data

This is closer to a **data analyst agent** than a **search summarizer**.

---

## 2. Goals and Non-Goals

### Goals
- Build a multi-agent system using LangGraph from scratch
- Demonstrate Supervisor pattern (LLM decides agent routing)
- Integrate YouTube Data API and web search as agent tools
- Automated sentiment analysis on real comment data
- Generate reports with charts and citations
- Clean, well-documented code suitable for portfolio

### Non-Goals
- NOT a real-time monitoring/dashboard system
- NOT scraping platforms that prohibit API access
- NOT a production SaaS with auth, deployment, scaling
- NOT fine-tuning or training models
- NOT building a formal plugin interface in MVP (good code organization is enough)
- NOT building a REST API unless the UI layer needs it

---

## 3. Learning Path

### Before Writing Code

**LangGraph fundamentals (1-2 days):**
- LangGraph official docs: StateGraph, nodes, edges, conditional edges
- Understand: State, Node, Edge, Supervisor pattern, tool calling
- Run the official LangGraph multi-agent example notebook
- Resource: https://langchain-ai.github.io/langgraph/

**YouTube Data API (half day):**
- Get API key from Google Cloud Console
- Understand quota: 10,000 units/day, search costs 100 units each
- Test: search, channels, videos, comments endpoints
- Resource: https://developers.google.com/youtube/v3

**Reference repos (read, don't fork):**
- langchain-ai/open_deep_research — official LangGraph research agent
- tarun7r/deep-research-agent — multi-agent with citation scoring

### LangGraph Concepts You Must Understand

| Concept | What It Is | Where in DeepLens |
|---------|-----------|-------------------|
| StateGraph | Graph definition with typed state | graph.py |
| State (TypedDict) | Shared data object all agents read/write | state.py |
| Node | A function that takes State, returns updates | Each agent file |
| Edge | Connection between nodes | graph.py |
| Conditional Edge | LLM decides which node runs next | Supervisor routing |
| Tool Calling | LLM invokes external functions | YouTube API, web search |

---

## 4. Target Use Cases

### MVP Focus: Entity Research

DeepLens is an **entity research agent**, not a YouTube analytics tool. Given any entity (person, brand, group, topic), it gathers structured data from YouTube + web, analyzes it, and produces an insight report. The key differentiator: **different queries trigger different research paths** — the Supervisor adapts its strategy based on the entity type and available data.

### MVP Demo Queries (3 types that show Supervisor adaptability)

| Query | Entity Type | What Makes It Interesting |
|-------|-------------|--------------------------|
| "Research Baby Monster" | Artist/Group | Full tool chain: channel stats + MV metrics + fan sentiment + news. Supervisor uses **all tools**. |
| "Research Elon Musk" | Public Figure | No official YT channel → Supervisor **skips** youtube_channel, focuses on video search + comments + web. Different path from above. |
| "Research AI coding assistants" | Topic | No single channel or person → Supervisor relies **heavily on web_search**, uses YouTube for opinion/review videos. Third distinct path. |

### Why These 3 Queries Matter

The same system, the same code, produces **three different execution paths**:

| Tool | Baby Monster | Elon Musk | AI Coding Assistants |
|------|:-----------:|:---------:|:-------------------:|
| youtube_channel | Yes | **No** | **No** |
| youtube_search | MV + stages | Videos about him | Review/opinion videos |
| youtube_comments | Fan sentiment | Public opinion | User experiences |
| web_search | News supplement | **Primary source** | **Primary source** |
| Analysis focus | Engagement + fan loyalty | Public perception + controversy | Landscape + comparison |

This is the core demo: **"I run 3 queries, the Supervisor walks a different path each time. None of this is hardcoded."**

### Phase 2 Use Cases (not in MVP)
- Video Deep Dive: single video analysis with detailed metrics
- Competitive Analysis: compare multiple channels side by side
- Trend Tracking: time-series analysis of entity mentions

---

## 5. Architecture

### Agent Graph

```
User Query
    |
    v
[Supervisor] ---> decides who works next
    |
    +---> [Research Agent] ---> gathers data
    |         |
    |         +-- youtube_search
    |         +-- youtube_channel
    |         +-- youtube_comments
    |         +-- web_search
    |
    +---> [Analysis Agent] ---> processes data
    |         |
    |         +-- sentiment_analyzer
    |         +-- statistics (pandas)
    |
    +---> [Report Agent] ---> generates output
              |
              +-- chart_generator
              +-- report_writer

Supervisor can loop: Research -> Analysis -> "need more?" -> Research
Max iterations: 5 (safety limit)
```

### Extensibility (Honest Assessment)

**Current architecture is YouTube-centric by design.** The MVP optimizes for one data source done well, not premature generalization.

Adding a new platform (e.g., Reddit) would require:
1. A new tool file in `tools/reddit.py` (straightforward)
2. New typed models in `models.py` for Reddit data structures
3. Extending the State schema to hold Reddit data alongside YouTube data
4. Updating the `statistics` tool to handle Reddit metrics (different from YouTube engagement)
5. Registering new tools with the Research Agent

**What does NOT change:** Supervisor routing logic, Report Agent, sentiment analysis (comments are comments regardless of platform), chart generation.

This is an intentional MVP trade-off: depth on one platform first, generalize in Phase 2 when the patterns are validated.

### LangGraph Implementation Pattern

```python
from langgraph.graph import StateGraph, END

# Define the graph
graph = StateGraph(DeepLensState)

# Add nodes (each node = one agent)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("research", research_agent)
graph.add_node("analysis", analysis_agent)
graph.add_node("report", report_agent)

# Entry point
graph.set_entry_point("supervisor")

# Supervisor decides routing (conditional edge)
graph.add_conditional_edges("supervisor", route_decision, {
    "research": "research",
    "analysis": "analysis",
    "report": "report",
    "done": END,
})

# After each agent, return to supervisor for evaluation
graph.add_edge("research", "supervisor")
graph.add_edge("analysis", "supervisor")
graph.add_edge("report", END)
```

### State Schema (MVP)

All agent-shared data uses **typed structures** (not bare dicts) so the data contract between agents is explicit and debuggable.

```python
# --- Data Models (models.py) ---

class YouTubeChannelData(TypedDict):
    channel_id: str
    title: str
    subscriber_count: int
    view_count: int
    video_count: int

class YouTubeVideoData(TypedDict):
    video_id: str
    title: str
    view_count: int
    like_count: int
    comment_count: int
    published_at: str

class CommentData(TypedDict):
    text: str
    like_count: int
    author: str
    video_id: str

class WebResult(TypedDict):
    title: str
    url: str
    snippet: str
    score: float              # Tavily relevance score

class Source(TypedDict):
    url: str
    title: str
    source_type: str          # "youtube" | "web"

class SentimentResult(TypedDict):
    positive: float           # 0.0-1.0 ratio
    neutral: float
    negative: float
    total_analyzed: int
    sample_positive: list[str]  # representative comments per category
    sample_negative: list[str]

class VideoStatistics(TypedDict):
    avg_views: float
    avg_likes: float
    avg_engagement_rate: float  # (likes + comments) / views
    top_videos: list[YouTubeVideoData]
    upload_frequency_days: float | None

# --- State ---

class DeepLensState(TypedDict):
    # Input
    user_query: str
    research_plan: list[str]

    # Research data (typed, not bare dicts)
    channel_data: YouTubeChannelData | None   # None if entity has no channel
    videos: list[YouTubeVideoData]
    comments: list[CommentData]
    web_results: list[WebResult]
    sources: list[Source]

    # Analysis results
    statistics: VideoStatistics | None
    sentiment: SentimentResult | None

    # Report
    report_markdown: str
    charts: list[str]               # file paths to generated chart PNGs

    # Control flow
    next_agent: str
    iteration_count: int
    max_iterations: int             # default: 5
    errors: list[str]
```

**Design note:** `channel_data` and `statistics` are `| None` because not every entity has a YouTube channel (e.g., "Elon Musk"). The Supervisor and Analysis Agent must handle the None case — this is part of the adaptive behavior.

---

## 6. Agent Design

### 6.1 Supervisor Agent

**Role:** Orchestrator. Decides what to do next based on current State.

**How it works:**
- Receives full State as input
- Uses LLM (with structured output) to decide: which agent next, or done?
- Increments iteration_count each loop
- Terminates if max_iterations reached or research is complete

**Supervisor Decision Scenarios (why this isn't if-else):**

An if-else router would always go Research → Analysis → Report. The Supervisor's value is making **context-dependent judgments** that differ by query:

| Scenario | What Happens | Supervisor's Judgment |
|----------|-------------|----------------------|
| **Tool selection by entity type** | "Research Elon Musk" → Supervisor tells Research Agent to skip youtube_channel (no official channel) and focus on youtube_search + web_search | LLM must understand the entity and decide which tools are relevant |
| **Data sufficiency** | Research returns channel info but youtube_comments failed (quota/no comments enabled) → Supervisor decides whether to retry or proceed with available data | LLM evaluates "is the current data enough to answer the user's question?" |
| **Direction adjustment** | User asks "Research Baby Monster" → initial web_search reveals a recent controversy → Supervisor routes back to Research for targeted comment analysis on the controversy video | LLM identifies unexpected findings and adjusts research direction |
| **Contradiction detection** | Analysis shows comment sentiment is very negative but channel metrics are growing fast → Supervisor routes back to Research for more web context | LLM recognizes data contradictions and decides how to resolve them |

**Key design question (for interview):**
"How does the Supervisor decide?" — It is an LLM call with the current State summary as context, using function calling to output a structured routing decision. The routing is not a fixed sequence — it adapts based on entity type, data availability, and findings.

"How is this different from if-else?" — An if-else can check "is youtube_data empty?", but it cannot judge "does this data sufficiently answer the user's question?" or "should we investigate a contradiction between sentiment and metrics?" These require understanding the research context.

### 6.2 Research Agent

**Role:** Data collector using YouTube API and web search.

**MVP Tools (4 tools):**
| Tool | What It Does |
|------|-------------|
| youtube_search | Search videos by query, returns list with metrics |
| youtube_channel | Get channel stats: subscribers, views, video count |
| youtube_comments | Get comment threads for a video |
| web_search | Tavily API search for web articles/news |

**Not in MVP:** web_scraper (moved to Phase 2)

### 6.3 Analysis Agent

**Role:** Processes raw data into insights.

**MVP Tools (2 tools):**
| Tool | What It Does |
|------|-------------|
| sentiment_analyzer | LLM classifies comments as positive/negative/neutral with rationale |
| statistics | Python function that computes metrics from YouTube data using pandas (avg views, engagement rate, top videos, upload frequency) |

**Not in MVP:** code_runner sandbox, trend_analyzer, comparator (moved to Phase 2)

The MVP statistics tool is a **predefined Python function**, not an LLM-generated code sandbox. This is simpler and more reliable. The code sandbox is a Phase 2 enhancement.

**Why LLM-based sentiment instead of VADER/TextBlob?**
- YouTube comments are multilingual, use slang, emoji, and sarcasm ("Great video, I only fell asleep twice"). Traditional NLP tools fail on these.
- LLM understands context and can provide a rationale for each classification.
- **Cost mitigation:** Comments are sent in batches (10-20 per LLM call, not one-by-one) to reduce API costs and latency.
- **Trade-off acknowledged:** Slower and more expensive than VADER. Acceptable for a research tool that processes ~100 comments per query, not 100K.
- **Phase 2 consideration:** Hybrid approach — VADER for initial bulk filtering, LLM for nuanced edge cases.

### 6.4 Report Agent

**Role:** Generates the final output.

**MVP Tools (2 tools):**
| Tool | What It Does |
|------|-------------|
| chart_generator | Creates matplotlib charts from analysis data, saves as PNG |
| report_writer | LLM generates structured markdown report with embedded chart references |

---

## 7. Tech Stack

### Core
| Component | Technology | Reason |
|-----------|-----------|--------|
| Agent Framework | LangGraph | Target skill for portfolio |
| LLM Provider | OpenAI API (GPT-4o-mini for dev, GPT-4o for demo) | Cost control during development |
| Language | Python 3.11+ | LangGraph ecosystem |

### Data and Analysis
| Component | Technology |
|-----------|-----------|
| Data Processing | Pandas |
| Visualization | Matplotlib |
| Sentiment | LLM-based (via OpenAI, same provider) |

### Platform Integrations
| Platform | API | Free Tier | Cost Per Query |
|----------|-----|-----------|---------------|
| YouTube | YouTube Data API v3 | 10,000 units/day | ~200-500 units per research |
| Web Search | Tavily API | 1,000 searches/month | 1-5 searches per research |

### Infrastructure
| Component | Technology | Reason |
|-----------|-----------|--------|
| UI (Phase 1) | Streamlit | Fast Python-native UI for demo |
| CLI | Typer | Terminal interface |
| API (Phase 2) | FastAPI | REST API layer, decouples UI from agent logic |
| Storage | SQLite | Research history (optional) |
| Config | pydantic-settings + .env | API key management |

### UI Roadmap
| Phase | Interface | Notes |
|-------|-----------|-------|
| Phase 1 | Streamlit + CLI | Quick demo, fast iteration |
| Phase 2 | + FastAPI backend | API layer, Streamlit calls API instead of graph directly |
| Future | + React/Web UI (optional) | Full frontend if needed, API already exists |

---

## 8. Cost Estimation

### Development Phase
| Item | Cost |
|------|------|
| OpenAI API (GPT-4o-mini for dev) | ~$0.01-0.05 per research query |
| YouTube API | Free (10,000 units/day) |
| Tavily API | Free (1,000/month) |
| **Total for 100 test runs** | **~$1-5** |

### Demo Phase
| Item | Cost |
|------|------|
| OpenAI API (GPT-4o for demo quality) | ~$0.10-0.30 per research query |
| YouTube + Tavily | Free |
| **Cost per demo** | **~$0.10-0.30** |

**Strategy:** Use GPT-4o-mini during development, switch to GPT-4o only for final demos and screenshots.

---

## 9. Feature Phases

### Phase 1: MVP (target: working end-to-end)

**Scope: 3 agents + 8 tools + CLI + Streamlit**

| Feature | Description |
|---------|-------------|
| Supervisor Agent | Parse query, route agents, loop control, max_iterations |
| Research Agent | youtube_search, youtube_channel, youtube_comments, web_search |
| Analysis Agent | sentiment_analyzer, statistics (predefined pandas functions) |
| Report Agent | chart_generator (matplotlib), report_writer (LLM markdown) |
| CLI | `python -m deeplens "query here"` |
| Streamlit UI | Text input, progress display, report output |

**MVP demo strategy: 3 queries that show adaptive behavior**

The power of the demo is NOT one impressive report — it's showing that **the same system behaves differently for different queries**.

```
Demo 1: "Research Baby Monster"
  → Full pipeline: channel + videos + comments + web
  → Output: artist profile report with fan sentiment chart

Demo 2: "Research Elon Musk"
  → Adaptive: skips channel, focus on public opinion
  → Output: public perception report, Supervisor loops back for controversy context

Demo 3: "Research AI coding assistants"
  → Web-heavy: multiple web searches, YouTube for user reviews
  → Output: landscape analysis with tool comparison
```

The interview talking point: "Notice how Demo 2 triggered a second Research iteration that Demo 1 didn't need. The Supervisor detected polarized sentiment and decided to gather more context. That decision was made by the LLM, not by my code."

### Phase 2: Enhanced Analysis + API Layer

| Feature | Description |
|---------|-------------|
| FastAPI backend | REST API wrapping the LangGraph agent graph |
| Code runner sandbox | Agent writes and runs custom pandas code |
| Chart improvements | Plotly interactive charts in Streamlit |
| Trend analysis | Time-series views/likes over time |
| Comparative analysis | Compare multiple channels side by side |
| Web scraper tool | Extract content from specific URLs |

### Phase 3: Polish

| Feature | Description |
|---------|-------------|
| Conversation memory | Follow-up queries on same research |
| Report export | PDF / HTML output |
| Research history | SQLite storage of past sessions |
| Better error handling | Graceful quota/rate limit recovery |

### Future Considerations

These are not committed phases, but potential directions that the architecture supports:

| Feature | Description | Prerequisite |
|---------|-------------|-------------|
| React Web UI | Full frontend consuming the FastAPI backend | Phase 2 API |
| X (Twitter) Integration | If API access is viable and cost-effective | Platform tool interface |
| More platforms | TikTok, Reddit, etc. as additional data sources | Platform tool interface |
| Scheduled research | Periodic re-analysis with change detection | Phase 3 history |

---

## 10. Data Flow Examples

### Example A: "Research Baby Monster" (Full tool chain)

```
1. Supervisor: entity = artist/group, has official YT channel
   Plan: ["channel stats", "top MVs", "fan comments", "recent news"]
   -> routes to: Research

2. Research Agent:
   - youtube_channel("Baby Monster") -> subs, views, video count
   - youtube_search("Baby Monster MV", max=15) -> music videos + stages
   - youtube_comments(top_mv_id, max=100) -> fan comments
   - web_search("Baby Monster 2026") -> news, comeback info
   -> writes to State, returns to Supervisor

3. Supervisor: channel_data ✓, videos ✓, comments ✓, web ✓
   -> sufficient data -> routes to Analysis

4. Analysis Agent:
   - statistics(videos) -> avg views, engagement rate, upload frequency
   - sentiment(comments) -> 85% positive, 10% neutral, 5% negative
   -> returns to Supervisor

5. Supervisor: analysis complete -> routes to Report

6. Report: channel overview + MV performance chart + fan sentiment pie
```

### Example B: "Research Elon Musk" (Adaptive — skip channel)

```
1. Supervisor: entity = public figure, NO official YT channel
   Plan: ["search videos about him", "public comments", "web news"]
   -> routes to: Research (NOTE: no youtube_channel in plan)

2. Research Agent:
   - youtube_search("Elon Musk", max=15) -> interview clips, news coverage
   - youtube_comments(controversial_video_id, max=100) -> public opinion
   - web_search("Elon Musk 2026") -> current news, controversies
   -> channel_data = None, writes to State, returns to Supervisor

3. Supervisor: videos ✓, comments ✓, web ✓, channel_data = None (expected)
   -> sufficient data -> routes to Analysis

4. Analysis Agent:
   - statistics(videos) -> engagement on videos ABOUT him (not his own)
   - sentiment(comments) -> 40% positive, 20% neutral, 40% negative
   -> returns to Supervisor

5. Supervisor: sentiment is polarized, web results mention controversy
   -> routes BACK to Research for targeted follow-up  ← NON-LINEAR

6. Research Agent (iteration 2):
   - web_search("Elon Musk controversy 2026") -> specific context
   -> returns to Supervisor

7. Supervisor: now has context for the polarization -> routes to Report

8. Report: public perception analysis + sentiment chart + controversy context
```

### Example C: "Research AI coding assistants" (Web-heavy)

```
1. Supervisor: entity = topic, no single channel
   Plan: ["web landscape", "YouTube reviews", "user opinions"]
   -> routes to: Research

2. Research Agent:
   - web_search("AI coding assistants 2026 comparison") -> landscape articles
   - web_search("best AI coding assistant") -> rankings, reviews
   - youtube_search("AI coding assistant review", max=10) -> review videos
   - youtube_comments(review_video_id, max=100) -> user experiences
   -> channel_data = None, writes to State, returns to Supervisor

3. Supervisor: web_results are primary source, YouTube supplements
   -> routes to Analysis

4. Analysis Agent:
   - statistics(videos) -> which review videos got most engagement
   - sentiment(comments) -> user satisfaction across tools
   -> returns to Supervisor

5. Supervisor: analysis complete -> routes to Report

6. Report: landscape overview + tool comparison + user sentiment chart
```

**Key observation:** Same system, same code, three different execution paths. The Supervisor adapts tool selection, iteration count, and analysis focus based on the entity type.

---

## 11. Non-Functional Requirements

### Performance
- Typical query completion: under 2 minutes
- YouTube API: stay within 10,000 units/day
- Max agent iterations: 5 (configurable, prevents infinite loops)

### Error Handling
- API quota exceeded: return partial results with warning
- LLM failure: retry once, then fail gracefully with error in report
- Invalid query: Supervisor returns helpful error message
- Loop detection: hard stop at max_iterations

### Observability (Critical for Demo)
- Every Supervisor decision is logged: `[Supervisor] iter=2, decision=research, reason="sentiment is polarized, need web context for controversy"`
- LangSmith tracing enabled (LangGraph native integration) for full execution trace
- Streamlit UI displays agent execution path using `get_graph().draw_mermaid()`
- Token usage logged per agent per iteration

### Cost Control
- Default to GPT-4o-mini during development
- Configurable model selection via .env
- Log token usage per research query

---

## 12. Project Structure

```
deeplens/
  docs/
    PRD.md                    # This document
  src/
    deeplens/
      __init__.py
      main.py                 # CLI entry point (Typer)
      graph.py                # LangGraph graph definition
      state.py                # DeepLensState TypedDict
      agents/
        __init__.py
        supervisor.py         # Supervisor agent node
        research.py           # Research agent node
        analysis.py           # Analysis agent node
        report.py             # Report agent node
      tools/
        __init__.py
        youtube.py            # YouTube Data API tools
        web_search.py         # Tavily search tool
        sentiment.py          # LLM sentiment analysis
        statistics.py         # Pandas-based metrics
        chart.py              # Matplotlib chart generation
      config.py               # Pydantic settings
      models.py               # Data models (Source, Chart, etc.)
  app/
    streamlit_app.py          # Streamlit UI
  api/
    server.py                 # FastAPI server (Phase 2)
  tests/
    test_agents/
    test_tools/
    conftest.py
  output/                     # Generated reports and charts
  .env.example
  pyproject.toml
  CLAUDE.md
```

---

## 13. Success Criteria

### Portfolio-Ready (must have)
- [ ] End-to-end demo works for all 3 query types (artist, person, topic)
- [ ] YouTube data collection uses real API (not mock data)
- [ ] Supervisor demonstrably takes **different paths** for different queries (visible in logs + LangGraph trace)
- [ ] Supervisor demonstrates at least one **non-linear decision** (loop back for more data)
- [ ] Sentiment analysis produces meaningful, explainable results with batch processing
- [ ] At least 2 chart types generated (pie chart, bar/line chart)
- [ ] State schema uses typed models (not bare dicts)
- [ ] Code is clean and well-organized
- [ ] README with architecture diagram, demo screenshots of all 3 query types, and execution trace comparison
- [ ] GitHub repo has clear commit history

### Interview-Ready (must be able to explain)
- [ ] "Why LangGraph?" — Graph-based state management, conditional routing, built for multi-agent
- [ ] "Why not just chain LLM calls?" — Supervisor pattern allows dynamic routing and loops
- [ ] "How does the Supervisor decide?" — LLM with structured output, State as context. It evaluates what data exists, whether it's sufficient, and what to do next.
- [ ] **"How is this different from if-else routing?"** — An if-else can check "is data empty?", but cannot judge "is this data sufficient to answer the user's question?" or "should we investigate a contradiction between positive growth metrics and negative sentiment?" These require understanding research context.
- [ ] "How do you prevent infinite loops?" — max_iterations counter, hard stop
- [ ] "Why YouTube API specifically?" — Reliable free API with structured data (metrics, comments), enables real quantitative + qualitative analysis beyond web search summaries
- [ ] "Why LLM for sentiment instead of VADER?" — YouTube comments are multilingual, use sarcasm and slang. LLM handles context. Batch processing (10-20 per call) mitigates cost.
- [ ] "How is this different from Doctify?" — Doctify is deterministic RAG pipeline, DeepLens is autonomous agent orchestration with adaptive routing
- [ ] Can draw the agent graph on a whiteboard and explain the 3 demo paths

### Demo Script (rehearse this — the 3-query comparison)
```
1. Open Streamlit UI
2. Run: "Research Baby Monster"
   → Show: full pipeline, all tools used, fan sentiment chart
   → Point out: agent execution path in LangGraph visualization

3. Run: "Research Elon Musk"
   → Show: Supervisor SKIPPED youtube_channel (no official channel)
   → Show: Supervisor LOOPED BACK to Research after detecting polarized sentiment
   → Point out: "This second Research iteration didn't happen in Demo 1.
     The Supervisor decided it was needed based on the sentiment results."

4. Run: "Research AI coding assistants"
   → Show: web_search is the primary source, YouTube is supplementary
   → Show: completely different report structure (landscape, not profile)

5. Compare the 3 LangGraph execution traces side by side
   → "Same code, same agents, three different paths.
     The routing is the LLM's decision, not mine."
```
