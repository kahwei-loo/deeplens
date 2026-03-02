# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

DeepLens is a **web-first** multi-agent entity research system built with LangGraph (Python). Users provide a research query about any entity (person, brand, group, topic); specialized AI agents collaborate to gather web data, optionally enrich with YouTube API data, analyze it, and produce a structured markdown report with charts.

Key differentiators:
- **Web-first architecture** — works with just OpenAI + Tavily keys. YouTube API is optional enrichment.
- **Multi-angle search** (inspired by Perplexity) — generates 2-5 search queries from different angles per research session.
- **Deep extraction** (inspired by Manus) — extracts full article content from top URLs, not just search snippets.
- **Adaptive Supervisor** — LLM decides routing based on information completeness, not fixed pipeline.

This is a learning project / portfolio piece demonstrating agent orchestration skills.

## Tech Stack

- **Python 3.11+**, LangGraph for agent orchestration
- **LLM**: OpenAI API (GPT-4o-mini for dev, GPT-4o for demos)
- **Data**: Tavily API (web search + extract, primary), YouTube Data API v3 (optional enrichment)
- **Analysis**: pandas, matplotlib, LLM-based sentiment (batch processing)
- **UI**: Streamlit (Phase 1), FastAPI backend (Phase 2)
- **CLI**: Typer — `python -m deeplens "query here"`
- **Config**: pydantic-settings + `.env`

## Build & Run Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run via CLI (requires OPENAI_API_KEY + TAVILY_API_KEY in .env)
python -m deeplens "Research Baby Monster"
python -m deeplens "Research Elon Musk" --verbose

# Run Streamlit UI
streamlit run app/streamlit_app.py

# Run tests
pytest
pytest tests/test_tools/       # specific test directory

# Type checking / linting
ruff check .
mypy src/
```

## Architecture

### Agent Graph (LangGraph Supervisor Pattern)

```
User Query → [Supervisor] → routes to one of:
  ├── [Research Agent] → multi_query_search, extract_urls, (optional: youtube_*)
  ├── [Analysis Agent] → sentiment_analyzer, statistics (pandas)
  └── [Report Agent]   → chart_generator (matplotlib), report_writer (LLM)

After Research/Analysis → returns to Supervisor for re-evaluation.
After Report → END.
Supervisor can loop (Research → Analysis → Research) up to max_iterations (default: 5).
```

### Web-First Research Strategy

The Research agent follows a multi-phase approach:
1. **Plan** — LLM generates 2-5 search queries from different angles (overview, news, opinion, controversy)
2. **Search** — Multi-angle web search via Tavily, results deduplicated by URL
3. **Extract** — Full content extraction from top 3 URLs via Tavily extract API
4. **Enrich** (optional) — YouTube API for video metrics + comments (only if API key configured AND entity has YouTube presence)

### Supervisor Adaptive Behavior

The Supervisor is NOT a fixed pipeline. It makes context-dependent decisions:
- **First iteration**: Always routes to Research with entity-type-aware instructions
- **Information completeness**: Routes based on coverage breadth, not just data volume
- **Web-only path**: If no YouTube API, routes Research → Report (skipping Analysis if no comments)
- **Non-linear loops**: Routes back to Research if coverage is thin or contradictions found
- **YouTube enrichment**: Only suggested for entities with strong YouTube presence (artists, creators, brands)

### Key Files

| File | Purpose |
|------|---------|
| `src/deeplens/graph.py` | LangGraph StateGraph definition, node registration, conditional edges |
| `src/deeplens/state.py` | `DeepLensState` TypedDict — shared state all agents read/write |
| `src/deeplens/models.py` | Typed data models (WebArticle, WebResult, YouTubeVideoData, SentimentResult, etc.) |
| `src/deeplens/agents/supervisor.py` | LLM-based routing decisions using structured output |
| `src/deeplens/agents/research.py` | Web-first data collection with optional YouTube enrichment |
| `src/deeplens/agents/analysis.py` | Statistics + sentiment processing node |
| `src/deeplens/agents/report.py` | Chart generation + markdown report node |
| `src/deeplens/tools/web_search.py` | Tavily web search (multi-query, dedup, extract) — PRIMARY data source |
| `src/deeplens/tools/youtube.py` | YouTube Data API v3 (optional, graceful degradation) |
| `src/deeplens/tools/sentiment.py` | LLM-based batch sentiment analysis |
| `src/deeplens/tools/statistics.py` | Pandas video statistics computation |
| `src/deeplens/tools/chart.py` | Matplotlib chart generation |
| `src/deeplens/config.py` | Pydantic settings (API keys from .env, `youtube_available` property) |
| `src/deeplens/main.py` | Typer CLI entry point |
| `app/streamlit_app.py` | Streamlit demo UI |

### State Schema

The `DeepLensState` TypedDict is the central data contract. Key fields:

**Web data (primary):**
- `web_results: list[WebResult]` — search results with title, url, snippet, score
- `web_articles: list[WebArticle]` — full extracted content from top URLs
- `sources: list[Source]` — all research sources for citation

**YouTube enrichment (optional — empty when no API key):**
- `channel_data: YouTubeChannelData | None`
- `videos: list[YouTubeVideoData]`, `comments: list[CommentData]`

**Analysis & Report:**
- `statistics: VideoStatistics | None`, `sentiment: SentimentResult | None`
- `report_markdown: str`, `charts: list[str]`

**Control flow:** `next_agent`, `iteration_count`, `max_iterations`, `errors`

## API Keys

**Required** (system won't work without these):
- `OPENAI_API_KEY` — LLM calls for all agents
- `TAVILY_API_KEY` — web search and content extraction

**Optional** (system works without, with graceful degradation):
- `YOUTUBE_API_KEY` — YouTube video metrics and comments enrichment

## API Quotas

- **Tavily API**: 1,000 searches/month free tier. Budget 5-15 searches per research query (multi-angle).
- **YouTube Data API**: 10,000 units/day. Only consumed when API key is configured.
- **OpenAI**: Use GPT-4o-mini during dev (~$0.01-0.05/query). Switch to GPT-4o for demos.

## Development Notes

- Web search is the primary data source — the system produces useful reports without YouTube API
- Supervisor routing is an LLM call with structured output — adapts by entity type, not hardcoded
- Research agent generates multi-angle search queries (like Perplexity) and extracts full content (like Manus)
- YouTube tools return empty/None when `YOUTUBE_API_KEY` is not set — no errors, just skipped
- The statistics tool is a predefined pandas function, NOT an LLM-generated code sandbox
- Sentiment analysis is LLM-based with batch processing (10-20 comments per call)
- All TypedDict models in `models.py`, all Pydantic BaseModels for structured LLM output in agent files
- Generated charts and reports go to `output/` directory (configurable via `OUTPUT_DIR`)
- Full PRD with phased roadmap: `docs/PRD.md`
