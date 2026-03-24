import streamlit as st
import os
import sys
import threading
import time
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 Multi-Task AI Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #f8f9fa;
        border-left: 4px solid;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .supervisor-card { border-color: #764ba2; background: #f3f0ff; }
    .researcher-card { border-color: #2196F3; background: #e3f2fd; }
    .coder-card      { border-color: #4CAF50; background: #e8f5e9; }
    .writer-card     { border-color: #FF9800; background: #fff3e0; }
    .analyst-card    { border-color: #F44336; background: #ffebee; }
    .planner-card    { border-color: #9C27B0; background: #f3e5f5; }

    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .status-active  { background: #c8e6c9; color: #2e7d32; }
    .status-done    { background: #bbdefb; color: #1565c0; }
    .status-waiting { background: #fff9c4; color: #f57f17; }

    .chat-message-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    .chat-message-assistant {
        background: #f0f2f6;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        margin-right: 20%;
    }
    .task-chip {
        display: inline-block;
        background: #e8eaf6;
        color: #3949ab;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        cursor: pointer;
    }
    .metric-box {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "messages": [],
        "agent_logs": [],
        "is_processing": False,
        "total_tasks": 0,
        "api_key_set": False,
        "current_agents": [],
        "task_history_all": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─────────────────────────────────────────────
# AGENT CONFIG
# ─────────────────────────────────────────────
AGENT_CONFIG = {
    "Supervisor":  {"icon": "🧠", "color": "#764ba2", "css": "supervisor-card", "desc": "Orchestrates all agents"},
    "Researcher":  {"icon": "🔍", "color": "#2196F3", "css": "researcher-card", "desc": "Web search & information"},
    "Coder":       {"icon": "💻", "color": "#4CAF50", "css": "coder-card",      "desc": "Write & execute code"},
    "Writer":      {"icon": "✍️", "color": "#FF9800", "css": "writer-card",     "desc": "Content & documents"},
    "Analyst":     {"icon": "📊", "color": "#F44336", "css": "analyst-card",    "desc": "Data & calculations"},
    "Planner":     {"icon": "🗺️", "color": "#9C27B0", "css": "planner-card",    "desc": "Task planning"},
}

SAMPLE_TASKS = [
    "🔍 Research latest AI trends and summarize",
    "💻 Write a Python web scraper for news",
    "📊 Analyze pros and cons of microservices",
    "✍️ Write a blog post about LangGraph",
    "🗺️ Plan a full-stack app development roadmap",
    "🔢 Calculate compound interest over 10 years",
    "📰 Search and summarize today's tech news",
    "🐛 Debug this code: for i in range(10) print(i)",
]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API Keys
    with st.expander("🔑 API Keys", expanded=not st.session_state.api_key_set):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            placeholder="sk-...",
            help="Required for all agents"
        )
        tavily_key = st.text_input(
            "Tavily API Key (Optional)",
            type="password",
            value=os.getenv("TAVILY_API_KEY", ""),
            placeholder="tvly-...",
            help="For better web search (falls back to DuckDuckGo)"
        )
        model_choice = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )

        if st.button("💾 Save Configuration"):
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                if tavily_key:
                    os.environ["TAVILY_API_KEY"] = tavily_key
                os.environ["MODEL_NAME"] = model_choice
                st.session_state.api_key_set = True
                st.success("✅ Configuration saved!")
            else:
                st.error("OpenAI API key is required!")

    st.markdown("---")

    # Agent Team Overview
    st.markdown("## 🤖 Agent Team")
    for name, cfg in AGENT_CONFIG.items():
        status = "🟢" if name in st.session_state.current_agents else "⚪"
        st.markdown(
            f"{status} {cfg['icon']} **{name}**  \n<small style='color:#666'>{cfg['desc']}</small>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Stats
    st.markdown("## 📈 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Tasks", st.session_state.total_tasks)
    with col2:
        st.metric("Agent Calls", len(st.session_state.agent_logs))

    st.markdown("---")

    # Clear
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.agent_logs = []
        st.session_state.current_agents = []
        st.session_state.task_history_all = []
        st.rerun()


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Multi-Task AI Agent System</h1>
    <p>Powered by LangGraph • LangChain • Specialized Agents</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Agent Logs", "ℹ️ Architecture"])

# ─────────────────────────────────────────────
# TAB 1: CHAT INTERFACE
# ─────────────────────────────────────────────
with tab1:
    # Sample tasks
    st.markdown("**⚡ Quick Tasks:**")
    cols = st.columns(4)
    for i, task in enumerate(SAMPLE_TASKS):
        with cols[i % 4]:
            if st.button(task, key=f"sample_{i}", use_container_width=True):
                st.session_state["pending_input"] = task.split(" ", 1)[1]  # Remove emoji prefix

    st.markdown("---")

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-message-user">👤 <strong>You:</strong> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message-assistant">🤖 <strong>Agent System:</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

    # Input area
    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            pending = st.session_state.pop("pending_input", "")
            user_input = st.text_area(
                "Your Task",
                value=pending,
                placeholder="Ask anything: research a topic, write code, analyze data, create content...",
                height=80,
                label_visibility="collapsed"
            )
        with col_btn:
            submit = st.form_submit_button("🚀 Run", use_container_width=True)

    # Process input
    if submit and user_input.strip():
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ Please set your OpenAI API key in the sidebar first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.total_tasks += 1

            # Process with agents
            with st.spinner("🤖 Agent team is working..."):
                progress_placeholder = st.empty()
                agent_status = st.empty()

                try:
                    sys.path.insert(0, "/home/claude/multi_agent_system")
                    from graph.workflow import run_agent_system

                    agent_updates = []
                    current_agents = []

                    def on_agent_update(node_name: str, node_output: dict):
                        """Callback for real-time agent updates."""
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        if node_name == "supervisor":
                            decision = node_output.get("supervisor_decision", {})
                            reasoning = decision.get("reasoning", "")
                            next_a = decision.get("next_agent", "")
                            update = {
                                "time": timestamp,
                                "agent": "Supervisor",
                                "action": f"Routing to: {next_a}",
                                "detail": reasoning,
                                "icon": "🧠"
                            }
                        else:
                            agent_name = node_name.capitalize()
                            output = node_output.get("last_agent_output", "")
                            update = {
                                "time": timestamp,
                                "agent": agent_name,
                                "action": "Task completed",
                                "detail": str(output)[:300] + "..." if len(str(output)) > 300 else str(output),
                                "icon": AGENT_CONFIG.get(agent_name, {}).get("icon", "🤖")
                            }
                            if agent_name not in current_agents:
                                current_agents.append(agent_name)

                        agent_updates.append(update)
                        st.session_state.agent_logs.append(update)

                        # Live update display
                        agent_status.markdown(
                            f"**{update['icon']} {update['agent']}** is working... `{update['action']}`"
                        )

                    result = run_agent_system(user_input, callback=on_agent_update)

                    # Extract final response
                    task_history = result.get("task_history", [])
                    messages = result.get("messages", [])

                    # Build a comprehensive response
                    if task_history:
                        final_response = ""
                        agents_used = [h["agent"] for h in task_history]

                        final_response += f"**🤖 Agents Used:** {' → '.join(agents_used)}\n\n"
                        final_response += "---\n\n"

                        # Last agent's output is typically the most complete
                        last_output = task_history[-1].get("full_output", "")
                        final_response += last_output

                        st.session_state.current_agents = list(set(agents_used))
                    else:
                        # Fallback: get from messages
                        ai_messages = [m for m in messages if hasattr(m, 'content') and str(type(m).__name__) == 'AIMessage']
                        final_response = ai_messages[-1].content if ai_messages else "Task completed. Check Agent Logs for details."

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_response
                    })

                    # Store task history
                    st.session_state.task_history_all.extend(task_history)

                    progress_placeholder.empty()
                    agent_status.empty()

                except ImportError as e:
                    st.error(f"Import error: {e}. Make sure all dependencies are installed: `pip install -r requirements.txt`")
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Error processing request: {str(e)}\n\nMake sure:\n1. OpenAI API key is valid\n2. All dependencies installed\n3. Check console for details"
                    })

            st.rerun()


# ─────────────────────────────────────────────
# TAB 2: AGENT LOGS
# ─────────────────────────────────────────────
with tab2:
    st.markdown("### 📋 Real-Time Agent Activity Logs")

    if not st.session_state.agent_logs:
        st.info("🕐 No agent activity yet. Run a task in the Chat tab!")
    else:
        # Filter by agent
        all_agents = list(set([log["agent"] for log in st.session_state.agent_logs]))
        selected_agent = st.multiselect(
            "Filter by Agent",
            options=["All"] + all_agents,
            default=["All"]
        )

        logs_to_show = st.session_state.agent_logs
        if "All" not in selected_agent and selected_agent:
            logs_to_show = [l for l in logs_to_show if l["agent"] in selected_agent]

        for log in reversed(logs_to_show):
            agent_cfg = AGENT_CONFIG.get(log["agent"], {"css": "supervisor-card", "color": "#666"})
            st.markdown(f"""
<div class="agent-card {agent_cfg['css']}">
    <strong>{log['icon']} {log['agent']}</strong>
    <span class="status-badge status-done">{log['time']}</span>
    <br><em>{log['action']}</em>
    <br><small>{log['detail'][:300]}...</small>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 3: ARCHITECTURE
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### 🏗️ System Architecture")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
```
┌─────────────────────────────────────────────┐
│              USER INPUT                     │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│          🧠 SUPERVISOR AGENT                │
│   • Analyzes task complexity                │
│   • Routes to best specialist               │
│   • Reviews & synthesizes output           │
│   • Coordinates multi-step tasks           │
└──┬──────┬──────┬──────┬──────┬─────────────┘
   ↓      ↓      ↓      ↓      ↓
 ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐
 │🔍 │  │💻 │  │✍️ │  │📊 │  │🗺️ │
 │Res│  │Cod│  │Wri│  │Ana│  │Pla│
 │ear│  │der│  │ter│  │lys│  │nne│
 │ch │  │   │  │   │  │t  │  │r  │
 └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘
   └───────┴──────┴───────┴──────┘
                   ↓
         Back to Supervisor
         (iterate if needed)
                   ↓
        ┌─────────────────┐
        │  FINAL RESPONSE │
        └─────────────────┘
```
        """)

    with col2:
        st.markdown("### 🛠️ Agent Tools")
        tools_info = {
            "🔍 Researcher": ["web_search", "news_search"],
            "💻 Coder": ["execute_python", "write_code", "debug_code", "run_bash_command"],
            "✍️ Writer": ["text_summarizer", "word_count_analyzer", "get_datetime"],
            "📊 Analyst": ["calculator", "format_data_as_table", "task_breakdown"],
            "🗺️ Planner": ["task_breakdown", "calculator", "utility_tools"],
        }

        for agent, tools in tools_info.items():
            with st.expander(agent):
                for tool in tools:
                    st.markdown(f"• `{tool}`")

    st.markdown("---")
    st.markdown("### 📦 Tech Stack")
    cols = st.columns(3)
    with cols[0]:
        st.info("**LangGraph**\n\nStateGraph orchestration, conditional routing, cycle detection")
    with cols[1]:
        st.info("**LangChain**\n\nReAct agents, tool calling, prompt templates, LLM integration")
    with cols[2]:
        st.info("**Streamlit**\n\nInteractive UI, real-time updates, session management")
