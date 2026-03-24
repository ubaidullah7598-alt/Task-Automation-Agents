from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from duckduckgo_search import DDGS
import os


@tool
def web_search(query: str) -> str:
    """Search the web for current information about any topic."""
    try:
        # Try Tavily first (better quality)
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_key and tavily_key != "your_tavily_api_key_here":
            search = TavilySearchResults(max_results=5)
            results = search.invoke(query)
            formatted = "\n\n".join(
                [f"**Source:** {r.get('url', 'N/A')}\n**Content:** {r.get('content', '')}" for r in results]
            )
            return f"Web Search Results for '{query}':\n\n{formatted}"
        else:
            # Fallback to DuckDuckGo
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append(f"**Title:** {r['title']}\n**URL:** {r['href']}\n**Snippet:** {r['body']}")
            return f"Search Results for '{query}':\n\n" + "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Search failed: {str(e)}. Please check your API keys."


@tool
def news_search(query: str) -> str:
    """Search for latest news on a specific topic."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=5):
                results.append(
                    f"**Title:** {r['title']}\n"
                    f"**Source:** {r.get('source', 'N/A')}\n"
                    f"**Published:** {r.get('date', 'N/A')}\n"
                    f"**Summary:** {r['body']}"
                )
        return f"Latest News for '{query}':\n\n" + "\n\n---\n\n".join(results)
    except Exception as e:
        return f"News search failed: {str(e)}"


def get_search_tools():
    return [web_search, news_search]
