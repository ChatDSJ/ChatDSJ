from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    if os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_SIGNING_TOKEN") and os.environ.get("SLACK_APP_TOKEN"):
        try:
            if not hasattr(slack_app, 'logger'):
                print("Using dummy Slack app, skipping Socket Mode initialization")
                return
                
            def start_slack_app():
                from slack_bolt.adapter.socket_mode import SocketModeHandler
                handler = SocketModeHandler(slack_app, os.environ.get("SLACK_APP_TOKEN"))
                handler.start()
            
            # Start the Slack app in a separate thread
            threading.Thread(target=start_slack_app, daemon=True).start()
            print("Slack bot started successfully")
        except Exception as e:
            print(f"Failed to start Slack bot: {e}")
    else:
        print("Slack credentials not set, skipping Slack bot initialization")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/test-chatgpt")
async def test_chatgpt():
    from app.slack.app import get_openai_response, format_conversation_history_for_openai
    test_prompt = "Can you help me with my Python code?"
    response_text, usage = get_openai_response([], test_prompt) # Pass empty history for simplicity
    response = {"response": response_text, "usage": usage}
    return {"response": response}
