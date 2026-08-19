import os
import requests

from langchain_core.tools import tool
from tavily import TavilyClient


# ============================================================
# TAVILY
# ============================================================

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

tavily = None

if TAVILY_API_KEY:

    tavily = TavilyClient(
        api_key=TAVILY_API_KEY
    )


def check_tavily():

    if tavily is None:

        raise RuntimeError(
            "TAVILY_API_KEY is missing."
        )


# ============================================================
# CLEAN SEARCH RESULTS
# ============================================================

def clean_results(
    response,
    limit=5
):

    results = []

    for item in response.get(
        "results",
        []
    )[:limit]:

        results.append({

            "title":
                item.get("title"),

            "url":
                item.get("url"),

            "content":
                item.get(
                    "content",
                    ""
                )[:1200]
        })

    return results


# ============================================================
# FLIGHT SEARCH
# ============================================================

@tool
def search_flights(
    origin: str,
    destination: str,
    duration: str,
    budget: str
) -> dict:

    """
    Search the web for flight options from
    the user's starting city to the destination.
    """

    check_tavily()

    query = f"""
    Find flight options from {origin} to {destination}
    for a {duration} trip.

    User budget: {budget} USD.

    Find airlines, approximate prices,
    flight duration and direct flight options.

    Use USD prices where possible.
    """

    response = tavily.search(

        query=query,

        search_depth="advanced",

        max_results=5,

        include_answer=True
    )

    return {

        "tool": "Flight Search",

        "note":
            "Prices and availability are approximate.",

        "answer":
            response.get("answer"),

        "results":
            clean_results(response)
    }


# ============================================================
# HOTEL SEARCH
# ============================================================

@tool
def search_hotels(
    destination: str,
    duration: str,
    budget: str
) -> dict:

    """
    Search the web for hotels and accommodation
    at the destination.
    """

    check_tavily()

    query = f"""
    Find hotels in {destination}
    for {duration}.

    User travel budget:
    {budget} USD.

    Find budget-friendly and mid-range hotels,
    approximate prices, ratings, location and reviews.

    Use USD prices where possible.
    """

    response = tavily.search(

        query=query,

        search_depth="advanced",

        max_results=5,

        include_answer=True
    )

    return {

        "tool": "Hotel Search",

        "note":
            "Hotel prices and availability may change.",

        "answer":
            response.get("answer"),

        "results":
            clean_results(response)
    }


# ============================================================
# PLACES SEARCH
# ============================================================

@tool
def search_places(
    destination: str,
    duration: str,
    interests: str
) -> dict:

    """
    Search for tourist attractions,
    restaurants, activities and experiences.
    """

    check_tavily()

    query = f"""
    Find the best places to visit in {destination}
    for a {duration} trip.

    User interests:
    {interests}

    Find tourist attractions, restaurants,
    activities, local experiences and popular places.
    """

    response = tavily.search(

        query=query,

        search_depth="advanced",

        max_results=7,

        include_answer=True
    )

    return {

        "tool": "Places Search",

        "answer":
            response.get("answer"),

        "results":
            clean_results(
                response,
                7
            )
    }


# ============================================================
# WEATHER
# ============================================================

@tool
def get_weather(
    destination: str
) -> dict:

    """
    Get current and forecast weather
    for the destination.
    """

    # --------------------------------------------------------
    # GEOCODING
    # --------------------------------------------------------

    geocoding_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    geo_response = requests.get(

        geocoding_url,

        params={

            "name":
                destination,

            "count":
                1,

            "language":
                "en",

            "format":
                "json"
        },

        timeout=20
    )

    geo_response.raise_for_status()

    geo_data = geo_response.json()

    locations = geo_data.get(
        "results",
        []
    )

    if not locations:

        raise RuntimeError(
            f"Location not found: {destination}"
        )

    location = locations[0]

    latitude = location["latitude"]

    longitude = location["longitude"]


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    weather_response = requests.get(

        weather_url,

        params={

            "latitude":
                latitude,

            "longitude":
                longitude,

            "current":
                "temperature_2m,"
                "relative_humidity_2m,"
                "weather_code,"
                "wind_speed_10m",

            "daily":
                "weather_code,"
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_probability_max",

            "forecast_days":
                7,

            "timezone":
                "auto"
        },

        timeout=20
    )

    weather_response.raise_for_status()

    weather_data = weather_response.json()

    return {

        "tool":
            "Weather Check",

        "location": {

            "name":
                location.get("name"),

            "country":
                location.get("country"),

            "latitude":
                latitude,

            "longitude":
                longitude
        },

        "current":
            weather_data.get(
                "current",
                {}
            ),

        "daily":
            weather_data.get(
                "daily",
                {}
            )
    }


# ============================================================
# TOOL LIST
# ============================================================

TOOLS = [

    search_flights,
    search_hotels,
    search_places,
    get_weather

]
