from typing import TypedDict, Annotated, List, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from agents import (
    create_supervisor_node,
    create_researcher_node,
    create_coder_node,
    create_writer_node,
    create_analyst_node,
    create_planner_node,
)
from config import FINISH_TOKEN


# ─────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_agent: str
    current_instruction: str
    supervisor_decision: dict
    task_history: List[dict]
    last_agent_output: str
    last_agent: str
    iteration_count: int


# ─────────────────────────────────────────────
# ROUTING FUNCTION
# ─────────────────────────────────────────────
def route_to_agent(state: AgentState) -> str:
    """Route to the appropriate agent based on supervisor decision."""
    next_agent = state.get("next_agent", FINISH_TOKEN)
    iteration_count = state.get("iteration_count", 0)

    # Safety: max iterations to prevent infinite loops
    if iteration_count >= 10:
        return END

    routing_map = {
        "Researcher": "researcher",
        "Coder": "coder",
        "Writer": "writer",
        "Analyst": "analyst",
        "Planner": "planner",
        FINISH_TOKEN: END,
    }

    return routing_map.get(next_agent, END)


def increment_iteration(state: AgentState) -> AgentState:
    """Increment iteration counter after each agent call."""
    return {**state, "iteration_count": state.get("iteration_count", 0) + 1}


def wrap_agent_with_counter(agent_fn):
    """Wrap agent function to increment iteration count."""
    def wrapped(state):
        result = agent_fn(state)
        result["iteration_count"] = state.get("iteration_count", 0) + 1
        return result
    return wrapped


# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────
def build_workflow() -> StateGraph:
    """Build and compile the multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", create_supervisor_node)
    workflow.add_node("researcher", wrap_agent_with_counter(create_researcher_node))
    workflow.add_node("coder", wrap_agent_with_counter(create_coder_node))
    workflow.add_node("writer", wrap_agent_with_counter(create_writer_node))
    workflow.add_node("analyst", wrap_agent_with_counter(create_analyst_node))
    workflow.add_node("planner", wrap_agent_with_counter(create_planner_node))

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor → conditional routing
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "analyst": "analyst",
            "planner": "planner",
            END: END,
        }
    )

    # All agents → back to supervisor
    for agent in ["researcher", "coder", "writer", "analyst", "planner"]:
        workflow.add_edge(agent, "supervisor")

    return workflow.compile()


# ─────────────────────────────────────────────
# RUNNER FUNCTION
# ─────────────────────────────────────────────
def run_agent_system(user_input: str, callback=None) -> dict:
    """
    Run the multi-agent system with a user input.
    callback: optional function(event_type, data) for streaming updates
    """
    from langchain_core.messages import HumanMessage

    app = build_workflow()

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "next_agent": "",
        "current_instruction": "",
        "supervisor_decision": {},
        "task_history": [],
        "last_agent_output": "",
        "last_agent": "",
        "iteration_count": 0,
    }

    final_state = None
    events = []

    for event in app.stream(initial_state):
        for node_name, node_output in event.items():
            events.append({"node": node_name, "output": node_output})
            if callback:
                callback(node_name, node_output)
        final_state = node_output

    return {
        "final_state": final_state,
        "events": events,
        "task_history": final_state.get("task_history", []) if final_state else [],
        "messages": final_state.get("messages", []) if final_state else [],
    }
