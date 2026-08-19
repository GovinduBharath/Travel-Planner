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
    description="Travel Planner using LangChain, LangGraph and Gemini",
    version="2.0.0"
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
# REQUEST MODEL
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
        description="Destination"
    )

    budget: str = Field(
        ...,
        description="Budget in USD"
    )

    duration: str = Field(
        ...,
        description="Trip duration"
    )

    interests: str = Field(
        default="sightseeing, food, culture",
        description="Travel interests"
    )


# ============================================================
# HOME
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
        "message": "AI Travel Planner is running",
        "framework": "LangChain + LangGraph",
        "llm": "Google Gemini"
    }


# ============================================================
# TRAVEL PLAN
# ============================================================

@app.post("/api/plan")
def create_travel_plan(
    request: TravelRequest
):

    if not os.getenv("GOOGLE_API_KEY"):

        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is missing"
        )

    if not os.getenv("TAVILY_API_KEY"):

        raise HTTPException(
            status_code=500,
            detail="TAVILY_API_KEY is missing"
        )

    try:

        user_input = request.model_dump()

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
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", 8000)
        ),
        reload=True
    )
