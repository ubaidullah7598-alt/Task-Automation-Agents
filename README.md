# 🤖 Multi-Task AI Agent System

A production-ready multi-agent automation system built with **LangGraph**, **LangChain**, and **Streamlit**.

## 🏗️ Architecture

```
User Input → Supervisor → [Researcher | Coder | Writer | Analyst | Planner] → Supervisor → Final Answer
```

The **Supervisor Agent** orchestrates 5 specialized agents in a LangGraph state machine:

| Agent | Icon | Responsibility | Tools |
|-------|------|----------------|-------|
| Supervisor | 🧠 | Routes tasks, synthesizes results | LLM reasoning |
| Researcher | 🔍 | Web search, news, fact-finding | web_search, news_search |
| Coder | 💻 | Write, execute, debug code | execute_python, write_code, debug_code |
| Writer | ✍️ | Content, reports, summaries | text_summarizer, word_count |
| Analyst | 📊 | Data analysis, calculations | calculator, format_data |
| Planner | 🗺️ | Task breakdown, planning | task_breakdown |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
cp .env.example .env
# Edit .env and add your keys
```

Or set via environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."  # Optional, falls back to DuckDuckGo
```

### 3. Run the App
```bash
streamlit run app.py
```

## 📂 Project Structure
```
multi_agent_system/
├── app.py                 # Streamlit UI
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── .env.example           # Environment template
├── agents/
│   ├── __init__.py
│   └── agent_nodes.py     # Supervisor + 5 specialized agents
├── graph/
│   ├── __init__.py
│   └── workflow.py        # LangGraph StateGraph definition
└── tools/
    ├── __init__.py
    ├── search_tools.py    # Web search tools
    ├── code_tools.py      # Code execution tools
    └── utility_tools.py   # Calculator, formatter, etc.
```

## 💡 Example Tasks

- `"Research the latest developments in quantum computing"`
- `"Write a Python script to scrape weather data"`
- `"Analyze the trade-offs between SQL and NoSQL databases"`
- `"Create a blog post about the future of AI agents"`
- `"Plan a microservices migration for a monolith app"`
- `"Calculate the ROI of investing $10,000 in the S&P 500 over 20 years"`

## 🔑 API Keys

| Key | Required | Source |
|-----|----------|--------|
| `OPENAI_API_KEY` | ✅ Yes | [platform.openai.com](https://platform.openai.com) |
| `TAVILY_API_KEY` | ⚡ Optional | [tavily.com](https://tavily.com) — better search quality |

## 🧩 LangGraph Flow

```python
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", create_supervisor_node)
workflow.add_node("researcher", create_researcher_node)
# ... other agents

# Supervisor conditionally routes to agents
workflow.add_conditional_edges("supervisor", route_to_agent, {...})

# Agents always return to supervisor
for agent in agents:
    workflow.add_edge(agent, "supervisor")
```

## 🛡️ Safety Features
- Max 10 iterations to prevent infinite loops
- Bash command whitelist (read-only commands only)
- Safe Python execution in sandboxed namespace
- API key validation before processing
