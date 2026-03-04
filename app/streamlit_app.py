"""Streamlit UI for DeepLens -- web-first entity research system."""

import logging
import os
from pathlib import Path

import streamlit as st

from deeplens.config import get_settings
from deeplens.graph import build_initial_state, create_graph, stream_with_timeout


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    st.set_page_config(page_title="DeepLens - Entity Research", layout="wide")
    st.title("DeepLens")
    st.caption("Web-first multi-agent entity research powered by LangGraph")

    with st.sidebar:
        st.header("Settings")

        # Model configuration -- supports OpenRouter and custom API bases
        model = st.text_input(
            "LLM Model",
            value=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            help="Any OpenAI-compatible model name (e.g. gpt-4o-mini)",
        )
        api_base = st.text_input(
            "API Base URL (optional)",
            value=os.environ.get("OPENAI_API_BASE", ""),
            help="Custom base URL for OpenRouter, Azure, etc. Leave empty for OpenAI default.",
        )
        max_iterations = st.slider("Max Iterations", 1, 10, 5)

        st.divider()
        st.markdown("**Demo Queries** -- adaptive routing in action:")
        if st.button("Research Baby Monster"):
            st.session_state["query_input"] = "Research Baby Monster"
        if st.button("Research Elon Musk"):
            st.session_state["query_input"] = "Research Elon Musk"
        if st.button("Research AI coding assistants"):
            st.session_state["query_input"] = "Research AI coding assistants"

    query = st.text_input(
        "Enter your research query",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., Research Baby Monster",
    )

    if st.button("Start Research", type="primary", disabled=not query):
        _run_research(query, model, api_base, max_iterations)

    if "report_markdown" in st.session_state and not st.session_state.get("running"):
        _display_results()


def _run_research(
    query: str, model: str, api_base: str, max_iterations: int
) -> None:
    # Guard against double-clicks or re-runs while a session is active.
    if st.session_state.get("running"):
        st.warning("Research is already in progress. Please wait for it to complete.")
        return

    query = query.strip()
    if not query:
        st.warning("Please enter a research query.")
        return

    # Apply overrides via env vars so get_settings() picks them up
    os.environ["MODEL_NAME"] = model
    os.environ["MAX_ITERATIONS"] = str(max_iterations)
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    elif "OPENAI_API_BASE" in os.environ:
        # Clear if user emptied the field but env had a value
        del os.environ["OPENAI_API_BASE"]

    # Clear cached settings so overrides take effect
    get_settings.cache_clear()

    st.session_state["running"] = True
    st.session_state["execution_log"] = []
    setup_logging()

    try:
        graph = create_graph()
        initial_state = build_initial_state(query)
        settings = get_settings()

        final_state = None
        execution_log: list[str] = []

        with st.status("Running research agents...", expanded=True) as status_ui:
            for event in stream_with_timeout(graph, initial_state, settings.graph_timeout_seconds):
                final_state = event
                na = event.get("next_agent", "")
                it = event.get("iteration_count", 0)
                if na:
                    execution_log.append(f"**iter {it}** -> `{na}`")
                    status_ui.update(label=f"Running: {na} (iteration {it})")
                    st.markdown(f"iter {it} -> `{na}`")

            if final_state is None:
                status_ui.update(label="Research produced no results", state="error")
                st.error("Research produced no results.")
                st.session_state["running"] = False
                return

            n_iter = final_state.get("iteration_count", 0)
            n_charts = len(final_state.get("charts", []))
            n_sources = len(final_state.get("sources", []))
            status_ui.update(
                label=(
                    f"Research complete! {n_iter} iterations, "
                    f"{n_sources} sources, {n_charts} charts."
                ),
                state="complete",
            )

        # Store state fields for display
        _STATE_KEYS = [
            "report_markdown", "charts", "errors", "channel_data",
            "videos", "comments", "web_results", "web_articles",
            "statistics", "sentiment", "sources",
        ]
        _LIST_KEYS = {
            "charts", "errors", "videos", "comments",
            "web_results", "web_articles", "sources",
        }
        for key in _STATE_KEYS:
            default: object = (
                [] if key in _LIST_KEYS
                else ("" if key == "report_markdown" else None)
            )
            st.session_state[key] = final_state.get(key, default)

        st.session_state["iterations"] = final_state.get("iteration_count", 0)
        st.session_state["execution_log"] = execution_log
        st.session_state["running"] = False

        _display_results()

    except Exception as exc:
        st.error(f"Research failed: {exc}")
        st.session_state["running"] = False


def _display_results() -> None:
    st.divider()

    # Execution trace
    with st.expander("Agent Execution Trace", expanded=False):
        log = st.session_state.get("execution_log", [])
        if log:
            st.markdown("\n\n".join(log))
        st.metric("Total Iterations", st.session_state.get("iterations", 0))

    # Charts
    charts = st.session_state.get("charts", [])
    if charts:
        st.subheader("Charts")
        cols = st.columns(min(len(charts), 2))
        for idx, cp in enumerate(charts):
            p = Path(cp)
            if p.exists():
                cols[idx % 2].image(str(p), use_container_width=True)

    # Report
    report = st.session_state.get("report_markdown", "")
    if report:
        st.subheader("Research Report")
        st.markdown(report)

    # Data panels
    col1, col2 = st.columns(2)

    with col1:
        # Web articles (primary data)
        with st.expander("Web Articles", expanded=False):
            articles = st.session_state.get("web_articles", [])
            if articles:
                for a in articles:
                    st.markdown(
                        f"**[{a.get('source_domain', '?')}]** "
                        f"[{a.get('title', 'Untitled')}]({a.get('url', '')})"
                    )
                    content = a.get("content", "")
                    if content:
                        st.caption(
                            content[:200] + "..." if len(content) > 200 else content
                        )
                    st.divider()
            else:
                st.info("No articles extracted")

        # Web search results
        with st.expander("Web Search Results", expanded=False):
            web = st.session_state.get("web_results", [])
            if web:
                for w in web:
                    st.markdown(
                        f"[{w.get('title', 'Untitled')}]({w.get('url', '')})"
                    )
                    st.divider()
            else:
                st.info("No web results")

    with col2:
        # YouTube data (optional enrichment)
        with st.expander("YouTube Videos", expanded=False):
            videos = st.session_state.get("videos", [])
            if videos:
                for v in sorted(
                    videos, key=lambda x: x.get("view_count", 0), reverse=True
                )[:10]:
                    st.markdown(
                        f"**{v.get('title', '?')}** "
                        f"Views: {v.get('view_count', 0):,} | "
                        f"Likes: {v.get('like_count', 0):,}"
                    )
                    st.divider()
            else:
                st.info("No YouTube videos (API enrichment not used)")

        # Statistics & Sentiment
        with st.expander("Statistics & Sentiment", expanded=False):
            stats = st.session_state.get("statistics")
            if stats:
                st.metric("Avg Views", f"{stats.get('avg_views', 0):,.0f}")
                st.metric(
                    "Avg Engagement",
                    f"{stats.get('avg_engagement_rate', 0):.2%}",
                )

            sentiment = st.session_state.get("sentiment")
            if sentiment:
                c1, c2, c3 = st.columns(3)
                c1.metric("Positive", f"{sentiment.get('positive', 0):.0%}")
                c2.metric("Neutral", f"{sentiment.get('neutral', 0):.0%}")
                c3.metric("Negative", f"{sentiment.get('negative', 0):.0%}")

            if not stats and not sentiment:
                st.info("No YouTube statistics or sentiment analysis")

    # Errors
    errors = st.session_state.get("errors", [])
    if errors:
        with st.expander(f"Warnings ({len(errors)})", expanded=False):
            for e in errors:
                st.warning(e)


if __name__ == "__main__":
    main()
