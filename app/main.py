from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psycopg
import os
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Slack bot integration
from app.slack.app import app as slack_app

# Start the Slack bot in a separate thread when FastAPI starts
@app.on_event("startup")
async def startup_event():
    # Check if Slack credentials are set
    if os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_SIGNING_SECRET"):
        def start_slack_app():
            slack_app.start(port=3000)
        
        # Start the Slack app in a separate thread
        threading.Thread(target=start_slack_app, daemon=True).start()

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
