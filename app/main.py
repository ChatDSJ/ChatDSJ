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
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    slack_signing_secret = os.environ.get("SLACK_SIGNING_SECRET")
    slack_app_token = os.environ.get("SLACK_APP_TOKEN")
    
    if slack_bot_token and slack_signing_secret and slack_app_token:
        try:
            if not hasattr(slack_app, 'logger'):
                print("Using dummy Slack app, skipping Socket Mode initialization")
                return
                
            def start_slack_app():
                from slack_bolt.adapter.socket_mode import SocketModeHandler
                print(f"Initializing Socket Mode with app token: {slack_app_token[:5]}...")
                handler = SocketModeHandler(slack_app, slack_app_token)
                handler.start()
            
            # Start the Slack app in a separate thread
            threading.Thread(target=start_slack_app, daemon=True).start()
            print("Slack bot started successfully")
        except Exception as e:
            print(f"Failed to start Slack bot: {e}")
    else:
        missing_vars = []
        if not slack_bot_token: missing_vars.append("SLACK_BOT_TOKEN")
        if not slack_signing_secret: missing_vars.append("SLACK_SIGNING_SECRET") 
        if not slack_app_token: missing_vars.append("SLACK_APP_TOKEN")
        print(f"Slack credentials not set: {', '.join(missing_vars)}, skipping Slack bot initialization")

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

@app.get("/test-openai")
async def test_openai():
    """Test endpoint to diagnose OpenAI API issues"""
    from app.slack.app import openai_client, logger
    
    if not openai_client:
        return {"status": "error", "message": "OpenAI client not initialized"}
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello world"}
            ],
            max_tokens=20
        )
        return {
            "status": "success", 
            "response": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage.model_dump() if hasattr(response, "usage") else None
        }
    except Exception as e:
        error_message = str(e)
        logger.error(f"OpenAI API test error: {error_message}")
        return {"status": "error", "message": error_message}
