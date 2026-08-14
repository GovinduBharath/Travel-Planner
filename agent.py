import json
import os
from typing import Any

import requests

from tools import TOOL_FUNCTIONS


# ==========================================
# CONFIGURATION
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ==========================================
# TOOL DEFINITIONS
# ==========================================

TOOL_DECLARATIONS = [

    {
        "name": "search_flights",

        "description": """
        Search the web for flight options between the origin
        and destination. Return approximate flight prices and
        useful flight information.
        """,

        "parameters": {
            "type": "object",

            "properties": {

                "origin": {
                    "type": "string"
                },

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
                "origin",
                "destination",
                "duration",
                "budget"
            ]
        }
    },

    {
        "name": "search_hotels",

        "description": """
        Search the web for hotels and accommodation options
        at the destination based on duration and budget.
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

    {
        "name": "search_places",

        "description": """
        Search for tourist attractions, restaurants,
        activities and interesting places at the destination.
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


# ==========================================
# CALL GEMINI
# ==========================================

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
                "function_declarations": TOOL_DECLARATIONS
            }
        ],

        "generationConfig": {

            "temperature": 0.2,

            "maxOutputTokens": 3000
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=90
    )

    response.raise_for_status()

    return response.json()


# ==========================================
# EXTRACT GEMINI RESPONSE
# ==========================================

def extract_parts(response):

    candidates = response.get(
        "candidates",
        []
    )

    if not candidates:

        raise RuntimeError(
            "Gemini returned no response"
        )

    return candidates[0].get(
        "content",
        {}
    ).get(
        "parts",
        []
    )


# ==========================================
# MAIN AGENT
# ==========================================

def run_travel_agent(user_input):

    prompt = f"""
You are an intelligent AI Travel Planning Agent.

The user provided:

{json.dumps(user_input, indent=2)}

Your objective is to create the best possible travel itinerary.

You have access to these tools:

1. Flight Search
2. Hotel Search
3. Places Search
4. Weather Check

You should intelligently decide which tools are required.

For a complete travel plan, normally use:

- Flight Search
- Hotel Search
- Places Search
- Weather

You may call multiple tools.

You may also call another tool after receiving
the result of a previous tool.

Important rules:

- Consider the user's budget.
- Consider the trip duration.
- Consider their interests.
- Do not invent flight availability.
- Do not invent hotel availability.
- Clearly mention that web prices are approximate.
- Use weather information when creating the itinerary.
- Combine all tool results into one useful plan.

Final response must contain:

1. Trip Summary
2. Flight Suggestions
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Weather Information
6. Estimated Budget
7. Travel Tips
8. Useful Sources
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

    # Maximum 6 agent/tool rounds
    for _ in range(6):

        response = call_gemini(contents)

        parts = extract_parts(response)

        # Add Gemini response
        contents.append({

            "role": "model",

            "parts": parts
        })

        # Find tool calls
        function_calls = [

            part.get("functionCall")

            for part in parts

            if part.get("functionCall")
        ]

        # ======================================
        # FINAL ANSWER
        # ======================================

        if not function_calls:

            final_text = "\n".join(

                part.get("text", "")

                for part in parts

                if part.get("text")
            )

            return {

                "status": "success",

                "destination":
                    user_input["destination"],

                "tools_used":
                    tools_used,

                "plan":
                    final_text,

                "model":
                    GEMINI_MODEL
            }

        # ======================================
        # EXECUTE TOOLS
        # ======================================

        tool_results = []

        for function_call in function_calls:

            tool_name = function_call.get(
                "name"
            )

            arguments = function_call.get(
                "args",
                {}
            )

            # Check tool exists
            if tool_name not in TOOL_FUNCTIONS:

                result = {
                    "error":
                    f"Unknown tool: {tool_name}"
                }

            else:

                try:

                    # Execute selected tool
                    result = TOOL_FUNCTIONS[
                        tool_name
                    ](**arguments)

                except Exception as error:

                    result = {
                        "error": str(error)
                    }

            # Store trace
            tools_used.append({

                "tool": tool_name,

                "arguments": arguments,

                "status":
                    "completed"
                    if "error" not in result
                    else "failed"
            })

            # Send result back to Gemini
            tool_results.append({

                "functionResponse": {

                    "name": tool_name,

                    "response": result
                }
            })

        # ======================================
        # SEND TOOL RESULTS BACK TO GEMINI
        # ======================================

        contents.append({

            "role": "user",

            "parts": tool_results
        })

    raise RuntimeError(
        "Agent exceeded maximum tool calls."
    )
