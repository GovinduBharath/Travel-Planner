import os

from dotenv import load_dotenv
from google import genai

from tools import web_search


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# CHECK GOOGLE API KEY
# ============================================================

if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Add GOOGLE_API_KEY to your .env file or Render environment variables."
    )


# ============================================================
# GOOGLE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ============================================================
# GEMINI MODEL
# ============================================================
MODEL_NAME = "gemini-2.5-flash"


# ============================================================
# TRAVEL AGENT INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are an intelligent AI Travel Planner.

Your job is to help users create personalized travel plans.

You can help with:

- Destinations
- Itineraries
- Tourist attractions
- Hotels
- Restaurants
- Local food
- Transportation
- Travel budgets
- Activities
- Best time to visit
- Family trips
- Solo trips
- Couple trips
- Weekend trips
- International trips
- Local trips

IMPORTANT RULES:

1. Understand the user's destination.
2. Consider the number of days.
3. Consider the user's budget.
4. Consider the number of travelers.
5. Consider the user's interests.
6. Use USD ($) for estimated costs.
7. Give practical recommendations.
8. Keep answers clear and easy to understand.
9. Do not invent current prices.
10. Do not invent current opening hours.
11. Use web search information when current information is required.

When creating an itinerary, use this structure:

DESTINATION:
TRIP DURATION:
TRAVELERS:
ESTIMATED BUDGET:

DAY 1:
Morning:
Afternoon:
Evening:

DAY 2:
Morning:
Afternoon:
Evening:

Continue for all requested days.

Then provide:

HOTELS / AREAS TO STAY:
FOOD:
TRANSPORTATION:
ESTIMATED COST:
TRAVEL TIPS:

If the user asks for current information such as:

- Current hotel prices
- Current restaurant information
- Current attractions
- Current events
- Current travel restrictions
- Current transportation
- Current ticket prices
- Current opening hours

use the Tavily web search information supplied to you.
"""


# ============================================================
# CHECK WHETHER WEB SEARCH IS REQUIRED
# ============================================================

def needs_web_search(message: str):

    keywords = [

        "latest",
        "current",
        "today",
        "price",
        "prices",
        "hotel",
        "hotels",
        "restaurant",
        "restaurants",
        "opening hours",
        "open now",
        "ticket",
        "tickets",
        "flight",
        "flights",
        "train",
        "trains",
        "bus",
        "buses",
        "visa",
        "weather",
        "events",
        "things to do",
        "best places",
        "tourist places",
        "attractions"

    ]

    message = message.lower()

    return any(
        keyword in message
        for keyword in keywords
    )


# ============================================================
# MAIN TRAVEL AGENT
# ============================================================

def travel_agent(user_message: str):

    web_context = ""


    # ========================================================
    # TAVILY SEARCH
    # ========================================================

    if needs_web_search(user_message):

        try:

            search_results = web_search(
                user_message
            )

            web_context = f"""
CURRENT WEB SEARCH INFORMATION:

{search_results}

Use this information when relevant.
Do not invent information that conflicts with reliable
current search results.
"""

        except Exception as e:

            print(
                "Tavily Search Error:",
                str(e)
            )

            web_context = """
Current web search is unavailable.

Use your general knowledge and clearly avoid claiming
uncertain information as current.
"""


    # ========================================================
    # CREATE FINAL PROMPT
    # ========================================================

    prompt = f"""
{SYSTEM_PROMPT}

{web_context}

USER REQUEST:

{user_message}

Create the best possible travel plan for the user.
"""


    # ========================================================
    # GEMINI
    # ========================================================

    response = client.models.generate_content(

        model=MODEL_NAME,

        contents=prompt
    )


    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    if not response.text:

        return "Sorry, I could not generate a travel plan."

    return response.text
