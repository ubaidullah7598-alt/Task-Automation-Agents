import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

AGENT_NAMES = {
    "SUPERVISOR": "Supervisor",
    "RESEARCHER": "Researcher",
    "CODER": "Coder",
    "WRITER": "Writer",
    "ANALYST": "Analyst",
    "PLANNER": "Planner",
}

FINISH_TOKEN = "FINISH"
