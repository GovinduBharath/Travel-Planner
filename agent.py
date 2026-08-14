import json
import os
import requests

from tools import TOOL_FUNCTIONS


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ============================================================
# TOOL DECLARATIONS
# ============================================================

TOOL_DECLARATIONS = [

    # --------------------------------------------------------
    # FLIGHT TOOL
    # --------------------------------------------------------

    {
        "name": "search_flights",

        "description": """
        Search the web for flight options between
        the origin and destination.

        Find approximate prices, airlines,
        travel duration and useful flight information.

        Prices should be presented in USD when possible.
        """,

        "parameters": {

            "type": "object",

            "properties": {

                "origin": {
                    "type": "string",
                    "description": "Starting city"
                },

                "destination": {
                    "type": "string",
                    "description": "Destination city"
                },

                "duration": {
                    "type": "string",
                    "description": "Trip duration"
                },

                "budget": {
                    "type": "string",
                    "description": "Budget in USD"
                }
            },

            "required": [
                "origin",
                "destination",
                "duration",
                "budget"
            ]
        }
    },


    # --------------------------------------------------------
    # HOTEL TOOL
    # --------------------------------------------------------

    {
        "name": "search_hotels",

        "description": """
        Search the web for hotels and accommodation
        options at the destination.

        Consider the trip duration and user's
        USD budget.
        """,

        "parameters": {

            "type": "object",

            "properties": {

                "destination": {
                    "type": "string"
                },

                "duration": {
                    "type": "string"
                },

                "budget": {
                    "type": "string"
                }
            },

            "required": [
                "destination",
                "duration",
                "budget"
            ]
        }
    },


    # --------------------------------------------------------
    # PLACES TOOL
    # --------------------------------------------------------

    {
        "name": "search_places",

        "description": """
        Search the web for tourist attractions,
        restaurants, activities and interesting
        places at the destination.
        """,

        "parameters": {

            "type": "object",

            "properties": {

                "destination": {
                    "type": "string"
                },

                "duration": {
                    "type": "string"
                },

                "interests": {
                    "type": "string"
                }
            },

            "required": [
                "destination",
                "duration",
                "interests"
            ]
        }
    },


    # --------------------------------------------------------
    # WEATHER TOOL
    # --------------------------------------------------------

    {
        "name": "get_weather",

        "description": """
        Get current and forecast weather information
        for the destination.
        """,

        "parameters": {

            "type": "object",

            "properties": {

                "destination": {
                    "type": "string"
                }
            },

            "required": [
                "destination"
            ]
        }
    }
]


# ============================================================
# GEMINI API CALL
# ============================================================

def call_gemini(contents):

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {

        "contents": contents,

        "tools": [
            {
                "function_declarations":
                    TOOL_DECLARATIONS
            }
        ],

        "generationConfig": {

            "temperature": 0.2,

            "maxOutputTokens": 4000
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=90
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# EXTRACT RESPONSE PARTS
# ============================================================

def extract_parts(response):

    candidates = response.get(
        "candidates",
        []
    )

    if not candidates:

        raise RuntimeError(
            "Gemini returned no candidates."
        )

    return candidates[0].get(
        "content",
        {}
    ).get(
        "parts",
        []
    )


# ============================================================
# MAIN TRAVEL AGENT
# ============================================================

def run_travel_agent(user_input):

    prompt = f"""

You are an intelligent AI Travel Planning Agent.

USER REQUEST:

{json.dumps(
    user_input,
    indent=2
)}

============================================================
YOUR JOB
============================================================

Create a complete, useful and budget-friendly
travel plan for the user.

The user's budget is in USD ($).

============================================================
AVAILABLE TOOLS
============================================================

1. Flight Search
2. Hotel Search
3. Places Search
4. Weather Check

You can decide which tools are required.

For a complete travel request, normally use
all four tools.

You can call multiple tools.

You can also call another tool after receiving
the result of a previous tool.

============================================================
IMPORTANT RULES
============================================================

1. Consider the user's budget.

2. Consider the trip duration.

3. Consider the user's interests.

4. Do not invent flight availability.

5. Do not invent hotel availability.

6. Web prices are approximate.

7. Use USD ($) when presenting the budget.

8. Use weather information when planning
   outdoor activities.

9. Avoid exceeding the user's budget.

10. Clearly explain if exact live prices
    are unavailable.

============================================================
FINAL RESPONSE
============================================================

Return the final answer using this structure:

TRIP SUMMARY

FLIGHT OPTIONS

HOTEL OPTIONS

DAY-BY-DAY ITINERARY

WEATHER

ESTIMATED BUDGET

TRAVEL TIPS

USEFUL SOURCES

Make the answer easy to read.
"""


    contents = [

        {
            "role": "user",

            "parts": [

                {
                    "text": prompt
                }

            ]
        }

    ]


    tools_used = []


    # ========================================================
    # AGENT LOOP
    # ========================================================

    for _ in range(6):

        response = call_gemini(
            contents
        )

        parts = extract_parts(
            response
        )


        # Add Gemini response
        contents.append({

            "role": "model",

            "parts": parts

        })


        # ====================================================
        # FIND FUNCTION CALLS
        # ====================================================

        function_calls = [

            part.get(
                "functionCall"
            )

            for part in parts

            if part.get(
                "functionCall"
            )

        ]


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        if not function_calls:

            final_text = "\n".join(

                part.get(
                    "text",
                    ""
                )

                for part in parts

                if part.get(
                    "text"
                )

            ).strip()


            return {

                "status":
                    "success",

                "destination":
                    user_input[
                        "destination"
                    ],

                "tools_used":
                    tools_used,

                "plan":
                    final_text,

                "model":
                    GEMINI_MODEL
            }


        # ====================================================
        # EXECUTE TOOLS
        # ====================================================

        tool_results = []


        for function_call in function_calls:

            tool_name = function_call.get(
                "name"
            )

            arguments = function_call.get(
                "args",
                {}
            )


            # Check tool
            if tool_name not in TOOL_FUNCTIONS:

                result = {

                    "error":
                        f"Unknown tool: {tool_name}"
                }

            else:

                try:

                    # Execute tool
                    result = TOOL_FUNCTIONS[
                        tool_name
                    ](
                        **arguments
                    )

                except Exception as error:

                    result = {

                        "error":
                            str(error)
                    }


            # Store tool information
            tools_used.append({

                "tool":
                    tool_name,

                "arguments":
                    arguments,

                "status":
                    "completed"
                    if "error" not in result
                    else "failed"
            })


            # Return result to Gemini
            tool_results.append({

                "functionResponse": {

                    "name":
                        tool_name,

                    "response":
                        result
                }

            })


        # ====================================================
        # SEND TOOL RESULTS BACK TO GEMINI
        # ====================================================

        contents.append({

            "role": "user",

            "parts":
                tool_results

        })


    # ========================================================
    # MAX TOOL CALL ERROR
    # ========================================================

    raise RuntimeError(
        "Agent exceeded maximum tool-call rounds."
    )
