import os
import requests

from tavily import TavilyClient


# ============================================================
# TAVILY CONFIGURATION
# ============================================================

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)


if TAVILY_API_KEY:

    tavily = TavilyClient(
        api_key=TAVILY_API_KEY
    )

else:

    tavily = None


# ============================================================
# CHECK TAVILY
# ============================================================

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
                item.get(
                    "title"
                ),

            "url":
                item.get(
                    "url"
                ),

            "content":
                item.get(
                    "content",
                    ""
                )[:1500],

            "score":
                item.get(
                    "score"
                )
        })


    return results


# ============================================================
# FLIGHT SEARCH
# ============================================================

def search_flights(
    origin,
    destination,
    duration,
    budget
):

    check_tavily()


    query = f"""

Find flight options from
{origin} to {destination}.

Trip duration:
{duration}

Travel budget:
{budget} USD

Find:

- Airlines
- Approximate flight prices
- Flight duration
- Direct flights if available
- Useful travel information

Use USD prices when possible.

"""


    response = tavily.search(

        query=query,

        search_depth="advanced",

        max_results=5,

        include_answer=True
    )


    return {

        "tool":
            "Flight Search",

        "note":
            "Flight prices and availability are approximate.",

        "answer":
            response.get(
                "answer"
            ),

        "results":
            clean_results(
                response
            )
    }


# ============================================================
# HOTEL SEARCH
# ============================================================

def search_hotels(
    destination,
    duration,
    budget
):

    check_tavily()


    query = f"""

Find hotels and accommodation
in {destination}.

Trip duration:
{duration}

Travel budget:
{budget} USD

Find:

- Budget hotels
- Hotel prices
- Ratings
- Reviews
- Location
- Important facilities

Use USD prices when possible.

"""


    response = tavily.search(

        query=query,

        search_depth="advanced",

        max_results=5,

        include_answer=True
    )


    return {

        "tool":
            "Hotel Search",

        "note":
            "Hotel prices and availability can change.",

        "answer":
            response.get(
                "answer"
            ),

        "results":
            clean_results(
                response
            )
    }


# ============================================================
# PLACES SEARCH
# ============================================================

def search_places(
    destination,
    duration,
    interests
):

    check_tavily()


    query = f"""

Find the best places to visit
in {destination}.

Trip duration:
{duration}

User interests:
{interests}

Find:

- Tourist attractions
- Restaurants
- Activities
- Local experiences
- Popular places
- Hidden gems

Create suggestions suitable
for the trip duration.

"""


    response = tavily.search(

        query=query,

        search_depth="advanced",

        max_results=7,

        include_answer=True
    )


    return {

        "tool":
            "Places Search",

        "answer":
            response.get(
                "answer"
            ),

        "results":
            clean_results(
                response,
                limit=7
            )
    }


# ============================================================
# WEATHER
# ============================================================

def get_weather(
    destination
):

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

            f"Could not find destination: "
            f"{destination}"
        )


    location = locations[0]


    latitude = location[
        "latitude"
    ]

    longitude = location[
        "longitude"
    ]


    # --------------------------------------------------------
    # WEATHER API
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


    weather_data = (
        weather_response.json()
    )


    return {

        "tool":
            "Weather Check",

        "location": {

            "name":
                location.get(
                    "name"
                ),

            "country":
                location.get(
                    "country"
                ),

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
# TOOL REGISTRY
# ============================================================

TOOL_FUNCTIONS = {

    "search_flights":
        search_flights,

    "search_hotels":
        search_hotels,

    "search_places":
        search_places,

    "get_weather":
        get_weather
}
