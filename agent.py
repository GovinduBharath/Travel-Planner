import os

from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from tools import TOOLS


# ============================================================
# API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)


# ============================================================
# GEMINI MODEL
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


llm = ChatGoogleGenerativeAI(

    model=MODEL_NAME,

    google_api_key=GOOGLE_API_KEY,

    temperature=0.2,

    max_retries=2
)


# ============================================================
# BIND TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(
    TOOLS
)


# ============================================================
# GRAPH STATE
# ============================================================

class AgentState(TypedDict):

    messages: list


# ============================================================
# AGENT NODE
# ============================================================

def agent_node(
    state: AgentState
):

    system_message = SystemMessage(

        content="""

You are an intelligent AI Travel Planner Agent.

Your job is to create a personalized,
budget-friendly travel itinerary.

You have access to these tools:

1. Flight Search
2. Hotel Search
3. Places Search
4. Weather Check

==================================================
AGENT BEHAVIOR
==================================================

Analyze the user's request.

Use the appropriate tools to gather
real-world information.

For a complete travel request,
normally use all four tools.

You can call multiple tools.

Do not invent flight availability.

Do not invent hotel availability.

Web search prices are approximate.

The user's budget is in USD.

Consider:

- Starting city
- Destination
- Budget
- Duration
- Interests
- Weather

Use the weather information to improve
the itinerary.

Try to keep the estimated trip cost
within the user's budget.

==================================================
FINAL RESPONSE
==================================================

After gathering information, create:

1. TRIP SUMMARY

2. FLIGHT OPTIONS

3. HOTEL OPTIONS

4. DAY-BY-DAY ITINERARY

5. WEATHER

6. ESTIMATED BUDGET

7. TRAVEL TIPS

8. USEFUL SOURCES

Make the final response clear,
organized and easy to understand.

"""

    )


    messages = [

        system_message

    ] + state["messages"]


    response = llm_with_tools.invoke(
        messages
    )


    return {

        "messages": [
            response
        ]

    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )


    # Agent node
    graph.add_node(
        "agent",
        agent_node
    )


    # Tool node
    graph.add_node(
        "tools",
        ToolNode(TOOLS)
    )


    # START → Agent
    graph.add_edge(
        START,
        "agent"
    )


    # Agent → Tools OR END
    graph.add_conditional_edges(

        "agent",

        tools_condition,

        {

            "tools":
                "tools",

            END:
                END

        }

    )


    # Tools → Agent
    graph.add_edge(
        "tools",
        "agent"
    )


    return graph.compile()


# ============================================================
# CREATE GRAPH
# ============================================================

travel_graph = build_graph()


# ============================================================
# RUN AGENT
# ============================================================

def run_travel_agent(
    user_input
):

    user_message = f"""

Create a travel plan using the
following user information:

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

Use your tools to research the trip
and then create the final optimized itinerary.

"""


    result = travel_graph.invoke(

        {

            "messages": [

                HumanMessage(
                    content=user_message
                )

            ]

        }

    )


    messages = result.get(
        "messages",
        []
    )


    # --------------------------------------------------------
    # GET FINAL MESSAGE
    # --------------------------------------------------------

    final_message = messages[-1]

    final_text = final_message.content


    # Some Gemini responses can contain
    # structured content blocks.
    if isinstance(
        final_text,
        list
    ):

        text_parts = []

        for part in final_text:

            if isinstance(
                part,
                dict
            ):

                if part.get("text"):

                    text_parts.append(
                        part["text"]
                    )

            elif isinstance(
                part,
                str
            ):

                text_parts.append(part)


        final_text = "\n".join(
            text_parts
        )


    # --------------------------------------------------------
    # FIND TOOLS USED
    # --------------------------------------------------------

    tools_used = []


    for message in messages:

        tool_calls = getattr(
            message,
            "tool_calls",
            []
        )


        for call in tool_calls:

            tools_used.append({

                "tool":
                    call.get(
                        "name"
                    ),

                "arguments":
                    call.get(
                        "args",
                        {}
                    )

            })


    return {

        "status":
            "success",

        "destination":
            user_input[
                "destination"
            ],

        "framework":
            "LangChain + LangGraph",

        "model":
            MODEL_NAME,

        "tools_used":
            tools_used,

        "plan":
            final_text

    }
