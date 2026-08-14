import os
import time

from dotenv import load_dotenv
from google import genai

from tools import web_search


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# CHECK API KEY
# ============================================================

if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Add GOOGLE_API_KEY to your .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ============================================================
# GEMINI MODELS
# ============================================================

MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite"
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an intelligent AI Travel Planner.

Your job is to create personalized and practical travel plans.

You help users with:

- Destination planning
- Travel itineraries
- Tourist attractions
- Hotels
- Restaurants
- Local food
- Transportation
- Budget planning
- Activities
- Best time to visit
- Solo trips
- Family trips
- Couple trips
- Weekend trips
- International trips
- Local trips

IMPORTANT RULES:

1. Understand the destination.
2. Consider the number of days.
3. Consider the number of travelers.
4. Consider the user's budget.
5. Consider the user's interests.
6. Use USD ($) for estimated costs.
7. Keep the answer simple and practical.
8. Do not invent current prices.
9. Do not invent current opening hours.
10. Use web information when current information is required.

For itineraries use:

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

Then include:

PLACES TO STAY:
FOOD:
TRANSPORTATION:
ESTIMATED COST:
TRAVEL TIPS:

If current information is requested, use the Tavily search
results provided in the prompt.
"""


# ============================================================
# DETERMINE WHETHER TAVILY IS NEEDED
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
        "attractions",
        "near me"

    ]

    message = message.lower()

    return any(
        keyword in message
        for keyword in keywords
    )


# ============================================================
# CALL GEMINI WITH RETRY + FALLBACK
# ============================================================

def generate_with_fallback(prompt: str):

    last_error = None


    for model in MODELS:

        for attempt in range(2):

            try:

                print(
                    f"Trying Gemini model: {model} "
                    f"(attempt {attempt + 1})"
                )


                response = client.models.generate_content(

                    model=model,

                    contents=prompt
                )


                if response and response.text:

                    print(
                        f"Successfully used: {model}"
                    )

                    return response.text


            except Exception as e:

                last_error = e

                error_text = str(e).lower()

                print(
                    f"{model} failed: {e}"
                )


                # --------------------------------------------
                # RETRY TEMPORARY SERVER / RATE ERRORS
                # --------------------------------------------

                if (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "429" in error_text
                    or "resource exhausted" in error_text
                ):

                    if attempt == 0:

                        print(
                            f"Temporary error. "
                            f"Retrying {model}..."
                        )

                        time.sleep(2)

                        continue


                    # Move to next model

                    break


                # --------------------------------------------
                # MODEL NOT AVAILABLE
                # --------------------------------------------

                if (
                    "404" in error_text
                    or "not found" in error_text
                    or "no longer available" in error_text
                ):

                    print(
                        f"{model} is unavailable. "
                        f"Trying next model..."
                    )

                    break


                # --------------------------------------------
                # OTHER ERROR
                # --------------------------------------------

                raise


    raise RuntimeError(
        "All Gemini models are currently unavailable. "
        f"Last error: {last_error}"
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

            print(
                "Searching Tavily..."
            )


            search_results = web_search(
                user_message
            )


            web_context = f"""

CURRENT WEB INFORMATION:

{search_results}

Use this information when relevant.
Do not invent current information.
"""


        except Exception as e:

            print(
                "Tavily Error:",
                str(e)
            )


            web_context = """

Tavily web search was unavailable.

Use general knowledge, but do not claim uncertain
information is current.
"""


    # ========================================================
    # FINAL PROMPT
    # ========================================================

    prompt = f"""
{SYSTEM_PROMPT}

{web_context}

USER REQUEST:

{user_message}

Create the best possible travel plan.
"""


    # ========================================================
    # GEMINI
    # ========================================================

    return generate_with_fallback(
        prompt
    )
