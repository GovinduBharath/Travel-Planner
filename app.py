import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import travel_agent


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ============================================================
# CHECK API KEYS
# ============================================================

if not GEMINI_API_KEY:
    print("WARNING: GOOGLE_API_KEY is missing")

if not TAVILY_API_KEY:
    print("WARNING: TAVILY_API_KEY is missing")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Travel Planner",
    description="Gemini + Tavily AI Travel Planning Agent",
    version="1.0.0"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home():

    return FileResponse(
        "static/index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "online",
        "gemini_key": bool(GEMINI_API_KEY),
        "tavily_key": bool(TAVILY_API_KEY)
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    try:

        if not GOOGLE_API_KEY:
            return {
                "success": False,
                "response": "Gemini API key is missing. Add GOOGLE_API_KEY to your environment variables."
            }

        response = travel_agent(
            request.message
        )

        return {
            "success": True,
            "response": response
        }

    except Exception as e:

        print("Agent Error:", str(e))

        return {
            "success": False,
            "response": f"Agent error: {str(e)}"
        }
