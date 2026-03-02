"""LangGraph StateGraph definition — wires together all DeepLens agents.

The graph follows the Supervisor pattern:
  User Query → Supervisor → (Research | Analysis | Report | END)
After Research or Analysis, control returns to Supervisor for re-evaluation.
After Report, the graph terminates.
"""

from langgraph.graph import END, StateGraph

from deeplens.agents.analysis import analysis_agent
from deeplens.agents.report import report_agent
from deeplens.agents.research import research_agent
from deeplens.agents.supervisor import route_decision, supervisor_agent
from deeplens.config import get_settings
from deeplens.state import DeepLensState


def create_graph() -> StateGraph:
    """Build and compile the DeepLens agent graph.

    Returns a compiled LangGraph that can be invoked with an initial state.
    """
    graph = StateGraph(DeepLensState)

    # Register agent nodes
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("research", research_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("report", report_agent)

    # Entry point — every run starts with the Supervisor
    graph.set_entry_point("supervisor")

    # Supervisor decides which agent runs next (conditional edge)
    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "research": "research",
            "analysis": "analysis",
            "report": "report",
            "done": END,
        },
    )

    # After Research or Analysis, return to Supervisor for evaluation
    graph.add_edge("research", "supervisor")
    graph.add_edge("analysis", "supervisor")

    # After Report, the graph terminates
    graph.add_edge("report", END)

    return graph.compile()


def build_initial_state(query: str) -> DeepLensState:
    """Construct the initial state for a research run.

    Args:
        query: The user's research query (e.g. "Research Baby Monster").

    Returns:
        A fully initialized DeepLensState ready for graph invocation.
    """
    settings = get_settings()
    return DeepLensState(
        user_query=query.strip(),
        research_plan=[],
        # Web data (primary)
        web_results=[],
        web_articles=[],
        sources=[],
        # YouTube enrichment (optional)
        channel_data=None,
        videos=[],
        comments=[],
        # Analysis
        statistics=None,
        sentiment=None,
        web_analysis=None,
        # Report
        report_markdown="",
        charts=[],
        # Control flow
        next_agent="",
        iteration_count=0,
        max_iterations=settings.max_iterations,
        errors=[],
    )
