from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from config import MODEL_NAME, AGENT_NAMES, FINISH_TOKEN
from tools import get_search_tools, get_code_tools, get_utility_tools
import functools


def get_llm(temperature: float = 0.1):
    """Initialize the LLM."""
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature, streaming=True)


# ─────────────────────────────────────────────
# SUPERVISOR NODE
# ─────────────────────────────────────────────
SUPERVISOR_SYSTEM_PROMPT = """You are a Master Supervisor Agent orchestrating a team of specialized AI agents.

Your team consists of:
- **Researcher**: Searches the web, gathers information, finds facts and news
- **Coder**: Writes, executes, and debugs Python code and scripts
- **Writer**: Drafts content, reports, summaries, emails, and documents
- **Analyst**: Analyzes data, performs calculations, interprets results
- **Planner**: Breaks down complex multi-step tasks into subtasks

Your job:
1. Understand the user's request
2. Decide which specialist agent should handle it (or handle simple tasks yourself)
3. Review each agent's output
4. Coordinate multiple agents for complex tasks
5. Synthesize final results into a clear, complete response

ROUTING RULES:
- Questions needing current info → Researcher
- Any coding, scripting, execution → Coder  
- Writing documents, emails, content → Writer
- Data analysis, math, calculations → Analyst
- Complex multi-step tasks → Planner first, then route subtasks
- Simple factual questions you can answer → respond directly with FINISH

Always respond with JSON in this format:
{{
  "reasoning": "Why you chose this action",
  "next_agent": "Researcher|Coder|Writer|Analyst|Planner|FINISH",
  "instruction": "Specific instruction for the next agent (or final answer if FINISH)"
}}
"""


def create_supervisor_node(state: dict) -> dict:
    """Supervisor that routes tasks to appropriate agents."""
    llm = get_llm(temperature=0.1)
    messages = state.get("messages", [])
    task_history = state.get("task_history", [])

    # Build context from history
    history_context = ""
    if task_history:
        history_context = "\n\nWork completed so far:\n" + "\n".join(
            [f"- {h['agent']}: {h['summary']}" for h in task_history[-5:]]
        )

    supervisor_messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        *messages,
        HumanMessage(content=f"Review the conversation and decide the next action.{history_context}\n\nRespond with JSON only.")
    ]

    response = llm.invoke(supervisor_messages)

    import json, re
    raw = response.content.strip()

    # Extract JSON
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            decision = json.loads(json_match.group())
        except:
            decision = {"reasoning": raw, "next_agent": FINISH_TOKEN, "instruction": raw}
    else:
        decision = {"reasoning": raw, "next_agent": FINISH_TOKEN, "instruction": raw}

    return {
        **state,
        "supervisor_decision": decision,
        "next_agent": decision.get("next_agent", FINISH_TOKEN),
        "current_instruction": decision.get("instruction", ""),
    }


# ─────────────────────────────────────────────
# HELPER: Create a generic ReAct agent node
# ─────────────────────────────────────────────
def _make_agent_node(name: str, system_prompt: str, tools: list):
    """Factory to create a ReAct agent node."""
    def agent_node(state: dict) -> dict:
        llm = get_llm(temperature=0.2)
        instruction = state.get("current_instruction", "")
        messages = state.get("messages", [])

        # Create agent
        agent = create_react_agent(llm, tools)

        # Build input
        agent_input = {
            "messages": [
                SystemMessage(content=system_prompt),
                *messages[-6:],  # Keep last 6 messages for context
                HumanMessage(content=instruction or messages[-1].content if messages else "Help with the task.")
            ]
        }

        result = agent.invoke(agent_input)
        agent_messages = result.get("messages", [])
        final_message = agent_messages[-1] if agent_messages else AIMessage(content="No response generated.")

        # Update task history
        task_history = state.get("task_history", [])
        task_history.append({
            "agent": name,
            "instruction": instruction[:100] + "..." if len(instruction) > 100 else instruction,
            "summary": str(final_message.content)[:200] + "..." if len(str(final_message.content)) > 200 else str(final_message.content),
            "full_output": final_message.content,
        })

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"[{name}]: {final_message.content}")],
            "task_history": task_history,
            "last_agent_output": final_message.content,
            "last_agent": name,
        }

    return agent_node


# ─────────────────────────────────────────────
# SPECIALIZED AGENT NODES
# ─────────────────────────────────────────────

RESEARCHER_PROMPT = """You are a Research Specialist Agent. Your expertise:
- Searching the web for accurate, current information
- Finding facts, statistics, news, and research
- Cross-referencing multiple sources
- Providing well-cited, accurate information

Always cite your sources. If you can't find info, say so clearly.
Provide comprehensive, well-structured research results."""

CODER_PROMPT = """You are a Senior Software Engineer Agent. Your expertise:
- Writing clean, efficient Python code
- Debugging and fixing code errors
- Executing code to verify results
- Explaining code clearly with comments
- Following best practices and PEP8

Always test your code mentally before presenting it.
Provide complete, runnable solutions with clear explanations."""

WRITER_PROMPT = """You are a Professional Content Writer Agent. Your expertise:
- Writing clear, engaging content for any audience
- Drafting reports, emails, articles, and documentation
- Editing and improving existing text
- Adapting tone and style to context
- Structuring content logically

Produce polished, well-structured output that's ready to use."""

ANALYST_PROMPT = """You are a Data & Business Analyst Agent. Your expertise:
- Analyzing data and identifying patterns
- Performing mathematical calculations
- Interpreting statistics and metrics
- Creating structured reports and summaries
- Providing actionable insights

Be precise with numbers. Show your work. Provide clear conclusions."""

PLANNER_PROMPT = """You are a Strategic Planning Agent. Your expertise:
- Breaking complex tasks into clear subtasks
- Identifying dependencies and optimal sequence
- Estimating effort and complexity
- Matching tasks to the right specialized agents
- Creating actionable execution plans

Provide detailed, structured plans that the team can execute."""


def create_researcher_node(state: dict) -> dict:
    return _make_agent_node("Researcher", RESEARCHER_PROMPT, get_search_tools())(state)


def create_coder_node(state: dict) -> dict:
    return _make_agent_node("Coder", CODER_PROMPT, get_code_tools())(state)


def create_writer_node(state: dict) -> dict:
    return _make_agent_node("Writer", WRITER_PROMPT, get_utility_tools())(state)


def create_analyst_node(state: dict) -> dict:
    return _make_agent_node("Analyst", ANALYST_PROMPT, get_utility_tools())(state)


def create_planner_node(state: dict) -> dict:
    return _make_agent_node("Planner", PLANNER_PROMPT, get_utility_tools())(state)
