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
    description="Agentic AI Travel Planner using Gemini and multiple tools",
    version="1.0.0"
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# USER INPUT MODEL
# ==========================================

class TravelRequest(BaseModel):

    origin: str = Field(
        default="Hyderabad",
        description="Starting city"
    )

    destination: str = Field(
        ...,
        description="Travel destination"
    )

    budget: str = Field(
        ...,
        description="Travel budget"
    )

    duration: str = Field(
        ...,
        description="Trip duration"
    )

    interests: str = Field(
        default="food, sightseeing, culture",
        description="Travel interests"
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.get("/")
def home():

    return FileResponse("static/index.html")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "success",
        "message": "AI Travel Planner Agent is running"
    }


# ==========================================
# TRAVEL PLANNER API
# ==========================================

@app.post("/api/plan")
def create_travel_plan(request: TravelRequest):

    # Check API keys
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing"
        )

    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="TAVILY_API_KEY is missing"
        )

    try:

        # Convert request to dictionary
        user_input = request.model_dump()

        # Send request to Agent Core
        result = run_travel_agent(user_input)

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# LOCAL DEVELOPMENT
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
