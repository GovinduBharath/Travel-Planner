import os

from dotenv import load_dotenv
from tavily import TavilyClient


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)


# ============================================================
# CREATE TAVILY CLIENT
# ============================================================

if TAVILY_API_KEY:

    tavily_client = TavilyClient(
        api_key=TAVILY_API_KEY
    )

else:

    tavily_client = None


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query: str):

    if not TAVILY_API_KEY:

        raise ValueError(
            "TAVILY_API_KEY is missing. "
            "Add TAVILY_API_KEY to your .env file."
        )


    response = tavily_client.search(

        query=query,

        search_depth="basic",

        max_results=5,

        include_answer=True
    )


    # ========================================================
    # FORMAT RESULTS
    # ========================================================

    output = []


    # Tavily summary

    answer = response.get(
        "answer"
    )

    if answer:

        output.append(
            "SUMMARY:\n" + answer
        )


    # Search results

    results = response.get(
        "results",
        []
    )


    for result in results:

        title = result.get(
            "title",
            "Unknown"
        )

        url = result.get(
            "url",
            ""
        )

        content = result.get(
            "content",
            ""
        )


        output.append(

            f"""
TITLE:
{title}

URL:
{url}

INFORMATION:
{content}
"""
        )


    return "\n".join(
        output
    )
