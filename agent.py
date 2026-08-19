import json
import os
import requests

from tools import TOOL_FUNCTIONS


GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


TOOL_DECLARATIONS = [

    {
        "name": "search_flights",

        "description": """
        Search for flight options from the
        user's origin to destination.
        Include approximate prices, airlines
        and flight duration.
        Prices should be in USD.
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
        Search for hotels and accommodation
        at the destination.
        Consider duration and USD budget.
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
        Search for tourist attractions,
        restaurants, activities and places
        to visit.
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
        Get current and forecast weather
        information for the destination.
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


def call_gemini(contents):

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GOOGLE_API_KEY}"
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


def extract_parts(response):

    candidates = response.get(
        "candidates",
        []
    )

    if not candidates:

        raise RuntimeError(
            "Gemini returned no response."
        )

    return candidates[0].get(
        "content",
        {}
    ).get(
        "parts",
        []
    )


def run_travel_agent(user_input):

    prompt = f"""
You are an AI Travel Planner Agent.

User information:

Starting City:
{user_input["origin"]}

Destination:
{user_input["destination"]}

Budget:
{user_input["budget"]} USD

Duration:
{user_input["duration"]}

Interests:
{user_input["interests"]}

Your task is to create a complete,
personalized and budget-friendly travel plan.

Use the available tools to research:

1. Flights
2. Hotels
3. Places to visit
4. Weather

The user's budget is in USD.

Consider the user's budget, duration
and interests.

Do not invent exact flight or hotel
availability.

Web prices are approximate.

After collecting the information,
create a final travel plan.

Include:

TRIP SUMMARY

FLIGHT OPTIONS

HOTEL OPTIONS

DAY-BY-DAY ITINERARY

WEATHER

ESTIMATED BUDGET

TRAVEL TIPS

USEFUL SOURCES
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


    for _ in range(6):

        response = call_gemini(
            contents
        )

        parts = extract_parts(
            response
        )

        contents.append({

            "role": "model",

            "parts": parts

        })


        function_calls = [

            part.get("functionCall")

            for part in parts

            if part.get("functionCall")
        ]


        if not function_calls:

            final_text = "\n".join(

                part.get(
                    "text",
                    ""
                )

                for part in parts

                if part.get("text")

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


        tool_results = []


        for function_call in function_calls:

            tool_name = function_call.get(
                "name"
            )

            arguments = function_call.get(
                "args",
                {}
            )


            if tool_name not in TOOL_FUNCTIONS:

                result = {
                    "error":
                        f"Unknown tool: {tool_name}"
                }

            else:

                try:

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


            tools_used.append({

                "tool":
                    tool_name,

                "arguments":
                    arguments

            })


            tool_results.append({

                "functionResponse": {

                    "name":
                        tool_name,

                    "response":
                        result

                }

            })


        contents.append({

            "role": "user",

            "parts":
                tool_results

        })


    raise RuntimeError(
        "Agent exceeded maximum tool calls."
    )
