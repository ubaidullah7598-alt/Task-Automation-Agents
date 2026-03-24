from langchain_core.tools import tool
import json
import math
import re
from datetime import datetime


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Supports basic arithmetic, math functions."""
    try:
        # Allow safe math expressions only
        allowed = set("0123456789+-*/().% ")
        allowed_funcs = ["sqrt", "pow", "abs", "round", "log", "sin", "cos", "tan", "pi", "e"]

        safe_expr = expression
        for func in allowed_funcs:
            safe_expr = safe_expr.replace(func, f"math.{func}")

        result = eval(safe_expr, {"math": math, "__builtins__": {}})
        return f"Expression: {expression}\nResult: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def text_summarizer(text: str, max_words: int = 100) -> str:
    """Summarize long text into a concise version."""
    words = text.split()
    if len(words) <= max_words:
        return f"Text is already concise ({len(words)} words):\n{text}"
    return f"Original length: {len(words)} words\nRequest to summarize to ~{max_words} words:\n\n{text[:2000]}..."


@tool
def format_data_as_table(data: str) -> str:
    """Format JSON or structured data as a readable markdown table."""
    try:
        parsed = json.loads(data)
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], dict):
                headers = list(parsed[0].keys())
                table = "| " + " | ".join(headers) + " |\n"
                table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in parsed[:20]:  # Max 20 rows
                    table += "| " + " | ".join([str(row.get(h, "")) for h in headers]) + " |\n"
                return f"Formatted Table:\n\n{table}"
        return f"Data formatted:\n```json\n{json.dumps(parsed, indent=2)}\n```"
    except Exception:
        return f"Raw data (could not parse as JSON):\n{data}"


@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current Date & Time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


@tool
def word_count_analyzer(text: str) -> str:
    """Analyze text statistics: word count, sentence count, reading time."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    chars = len(text)
    reading_time = max(1, len(words) // 200)  # avg 200 wpm

    return (
        f"📊 Text Analysis:\n"
        f"- Words: {len(words)}\n"
        f"- Sentences: {len(sentences)}\n"
        f"- Characters: {chars}\n"
        f"- Estimated reading time: ~{reading_time} min\n"
        f"- Avg words per sentence: {len(words) // max(1, len(sentences))}"
    )


@tool
def task_breakdown(complex_task: str) -> str:
    """Break down a complex task into smaller, manageable subtasks."""
    return f"""
Task Breakdown Request:
Main Task: {complex_task}

Please analyze this task and provide:
1. Clear list of subtasks (numbered)
2. Dependencies between subtasks
3. Estimated complexity for each
4. Recommended execution order
5. Which agent type is best for each subtask (Researcher/Coder/Writer/Analyst)
"""


def get_utility_tools():
    return [calculator, text_summarizer, format_data_as_table, get_current_datetime, word_count_analyzer, task_breakdown]
