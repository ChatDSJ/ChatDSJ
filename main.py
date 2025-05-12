import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re
from loguru import logger

from config.settings import get_settings
from handler.openai_service import OpenAIService
from services.cached_notion_service import CachedNotionService
from services.slack_service import SlackService
from actions.action_framework import ServiceContainer, ActionRequest, ActionRouter

# Configure settings and logging
settings = get_settings()
logger.remove()
logger.add(
    "logs/chatdsj.log", 
    rotation="10 MB",
    level=settings.log_level
)
logger.add(
    lambda msg: print(msg), 
    level=settings.log_level
)

logger.info(f"Starting application in {settings.environment} mode")

# Initialize FastAPI app
app = FastAPI(title="ChatDSJ", description="A Slack bot with enhanced architecture")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
openai_service = OpenAIService()
notion_service = CachedNotionService(
    notion_api_token=settings.notion_api_token.get_secret_value() if settings.notion_api_token else None,
    notion_user_db_id=settings.notion_user_db_id or "",
    cache_ttl=settings.cache_ttl,
    cache_max_size=settings.cache_max_size
)
slack_service = SlackService(
    bot_token=settings.slack_bot_token.get_secret_value() if settings.slack_bot_token else None,
    signing_secret=settings.slack_signing_secret.get_secret_value() if settings.slack_signing_secret else None,
    app_token=settings.slack_app_token.get_secret_value() if settings.slack_app_token else None
)

# Create service container
services = ServiceContainer(
    slack_service=slack_service,
    notion_service=notion_service,
    openai_service=openai_service
)

# Slack event handler function
async def handle_mention(event, say, client):
    """Handle app_mention events."""
    try:
        channel_id = event["channel"]
        user_id = event["user"]
        message_ts = event["ts"]
        text = event.get("text", "")
        thread_ts = event.get("thread_ts", message_ts)
        
        # Clean prompt text
        prompt = slack_service.clean_prompt_text(text)
        
        # Send ephemeral acknowledgment
        slack_service.send_ephemeral_message(
            channel_id, 
            user_id, 
            "I heard you! I'm working on a response... 🧠"
        )
        
        # Check for nickname command first
        nickname_response, nickname_success = await asyncio.to_thread(
            notion_service.handle_nickname_command,
            prompt, 
            user_id,
            slack_service.get_user_display_name(user_id)
        )
        
        if nickname_response:
            # If this was a nickname command, send response and return
            response = slack_service.send_message(channel_id, nickname_response, thread_ts)
            slack_service.update_channel_stats(channel_id, user_id, message_ts)
            return response
        
        # Create action request
        request = ActionRequest(
            channel_id=channel_id,
            user_id=user_id,
            message_ts=message_ts,
            thread_ts=thread_ts,
            text=text,
            prompt=prompt
        )
        
        # Route to appropriate action
        router = ActionRouter(services)
        action_response = await router.route_action(request)
        
        # Send response to Slack
        if action_response.success and action_response.message:
            response = slack_service.send_message(
                channel_id, 
                action_response.message,
                action_response.thread_ts or thread_ts
            )
        else:
            error_msg = action_response.error or "Unknown error"
            logger.error(f"Action failed: {error_msg}")
            
            response = slack_service.send_message(
                channel_id,
                action_response.message or "I encountered an error processing your request.",
                thread_ts
            )
        
        # Update channel stats
        slack_service.update_channel_stats(channel_id, user_id, message_ts)
        
        return response
    except Exception as e:
        logger.error(f"Error handling mention: {e}", exc_info=True)
        say(
            text="I encountered an unexpected error. Please try again later.",
            thread_ts=event.get("thread_ts", event.get("ts"))
        )

# Register Slack event handlers
@slack_service.app.event("app_mention")
def app_mention_handler(event, say, client):
    """Handle app_mention events by running the async handler in a thread."""
    asyncio.run(handle_mention(event, say, client))

# Start Slack bot in socket mode
@app.on_event("startup")
async def startup_event():
    """Start the Slack bot in socket mode on startup."""
    if slack_service.is_available():
        slack_service.start_socket_mode()
        logger.info("Slack bot started in socket mode.")
    else:
        logger.warning("Slack bot could not be started in socket mode.")

# Health check endpoint
@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "services": {
            "slack": slack_service.is_available(),
            "notion": notion_service.is_available(),
            "openai": openai_service.is_available()
        }
    }

# Test OpenAI endpoint
@app.get("/test-openai")
async def test_openai():
    """Test endpoint to diagnose OpenAI API issues."""
    if not openai_service.is_available():
        return {"status": "error", "message": "OpenAI client not initialized"}
    
    try:
        content, usage = await openai_service.get_completion_async("Say hello world")
        return {
            "status": "success",
            "response": content,
            "model": openai_service.model,
            "usage": usage
        }
    except Exception as e:
        logger.error(f"OpenAI API test error: {e}")
        return {"status": "error", "message": str(e)}

# Main entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)