import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import run_travel_agent

load_dotenv()

app = FastAPI(
    title="AI Travel Planner Agent",
    description="AI-powered travel planning agent using Gemini and multiple tools",
    version="1.0.0"
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# USER INPUT MODEL
# ============================================================

class TravelRequest(BaseModel):

    origin: str = Field(
        ...,
        min_length=2,
        description="Starting city"
    )

    destination: str = Field(
        ...,
        min_length=2,
        description="Destination city or country"
    )

    budget: str = Field(
        ...,
        min_length=1,
        description="Travel budget in USD"
    )

    duration: str = Field(
        ...,
        min_length=1,
        description="Trip duration"
    )

    interests: str = Field(
        default="sightseeing, food, culture",
        description="Travel interests"
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "success",
        "message": "AI Travel Planner Agent is running"
    }


# ============================================================
# CREATE TRAVEL PLAN
# ============================================================

@app.post("/api/plan")
def create_travel_plan(
    request: TravelRequest
):

    # Check Gemini API key
    if not os.getenv("GEMINI_API_KEY"):

        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing"
        )

    # Check Tavily API key
    if not os.getenv("TAVILY_API_KEY"):

        raise HTTPException(
            status_code=500,
            detail="TAVILY_API_KEY is missing"
        )

    try:

        # Convert request into dictionary
        user_input = request.model_dump()

        # Send data to Agent Core
        result = run_travel_agent(
            user_input
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(error)}"
        )


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                8000
            )
        ),
        reload=True
    )
