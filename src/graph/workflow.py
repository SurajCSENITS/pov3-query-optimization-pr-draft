"""
LangGraph workflow definition.

Constructs a linear StateGraph that chains the five agents:

  AnalysisAgent → OptimizationAgent → ValidationAgent → ReportAgent → PRAgent

Each node is a thin wrapper that instantiates the agent and calls `run()`.
The graph uses QueryOptimizationState as the shared state container.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.analysis import AnalysisAgent
from src.agents.optimization import OptimizationAgent
from src.agents.pr import PRAgent
from src.agents.report import ReportAgent
from src.agents.validation import ValidationAgent
from src.models.state import QueryOptimizationState

# ── Instantiate agents (stateless — safe to reuse) ──────────────

_analysis_agent = AnalysisAgent()
_optimization_agent = OptimizationAgent()
_validation_agent = ValidationAgent()
_report_agent = ReportAgent()
_pr_agent = PRAgent()


# ── Node functions (LangGraph expects `state -> state_patch`) ───

def analysis_node(state: QueryOptimizationState) -> dict:
    return _analysis_agent.run(state)


def optimization_node(state: QueryOptimizationState) -> dict:
    return _optimization_agent.run(state)


def validation_node(state: QueryOptimizationState) -> dict:
    return _validation_agent.run(state)


def report_node(state: QueryOptimizationState) -> dict:
    return _report_agent.run(state)


def pr_node(state: QueryOptimizationState) -> dict:
    return _pr_agent.run(state)


def route_after_validation(state: QueryOptimizationState) -> str:
    """
    Route to optimization if validation fails and max retries not reached.
    """
    validation = state.get("validation", {})
    decision = validation.get("decision", "APPROVED")
    
    if decision != "APPROVED":
        retry_count = state.get("retry_count", 0)
        if retry_count < 2:
            return "optimization"
            
    return "report"


# ── Graph construction ──────────────────────────────────────────

def build_workflow() -> StateGraph:
    """
    Build and compile the POV3 query optimization LangGraph.

    Returns a compiled graph ready to `.invoke()`.
    """
    graph = StateGraph(QueryOptimizationState)

    # Add nodes
    graph.add_node("analysis", analysis_node)
    graph.add_node("optimization", optimization_node)
    graph.add_node("validation", validation_node)
    graph.add_node("report", report_node)
    graph.add_node("pr", pr_node)

    # Define edges (linear pipeline)
    graph.set_entry_point("analysis")
    graph.add_edge("analysis", "optimization")
    graph.add_edge("optimization", "validation")
    graph.add_conditional_edges(
        "validation", 
        route_after_validation,
        {"optimization": "optimization", "report": "report"}
    )
    graph.add_edge("report", "pr")
    graph.add_edge("pr", END)

    return graph.compile()
