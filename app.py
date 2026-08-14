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
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Travel Planner",
    description="AI Travel Planner powered by Google Gemini and Tavily",
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
# HOME
# ============================================================

@app.get("/")
async def home():
    return FileResponse("static/index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "online",
        "google_key": bool(GOOGLE_API_KEY),
        "tavily_key": bool(TAVILY_API_KEY)
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    try:

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
