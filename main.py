import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re
from loguru import logger
import os

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

@slack_service.app.command("/name")
def handle_name_command(ack, respond, command):
    """Handle /name slash command."""
    ack()  # Acknowledge the command immediately
    
    try:
        user_id = command["user_id"]
        name = command["text"].strip()
        
        # Call the handler synchronously 
        if not name:
            respond("❌ Please provide a name. Usage: `/name John`")
            return
            
        if len(name) > 50:
            respond("❌ Name is too long. Please use a shorter name.")
            return
        
        logger.info(f"Processing /name command for user {user_id}: '{name}'")
        
        # Store the nickname synchronously
        success = notion_service.store_user_nickname(user_id, name, None)
        
        if success:
            respond(f"✅ Got it! I'll call you **{name}** from now on.")
        else:
            respond("❌ Sorry, I had trouble saving that name. Please try again.")
        
    except Exception as e:
        logger.error(f"Error handling /name command: {e}")
        respond("❌ I encountered an error. Please try again.")

@slack_service.app.command("/fact")
def handle_fact_command(ack, respond, command):
    """Handle /fact slash command."""
    ack()  # Acknowledge the command immediately
    
    try:
        user_id = command["user_id"]
        fact = command["text"].strip()
        
        # Validate fact
        if not fact:
            respond("❌ Please provide a fact. Usage: `/fact I like coffee`")
            return
            
        if len(fact) > 500:
            respond("❌ Fact is too long. Please use a shorter fact.")
            return
        
        logger.info(f"Processing /fact command for user {user_id}: '{fact}'")
        
        # Add fact to Notion synchronously
        success = add_fact_to_notion(user_id, fact)
        
        if success:
            respond(f"✅ Remembered: **{fact}**")
        else:
            respond("❌ Sorry, I couldn't save that fact. Please try again.")
        
    except Exception as e:
        logger.error(f"Error handling /fact command: {e}")
        respond("❌ I encountered an error. Please try again.")

@slack_service.app.command("/web")
def handle_web_command(ack, respond, command):
    """Handle /web slash command for web search."""
    ack()  # Acknowledge the command immediately
    
    try:
        user_id = command["user_id"]
        query = command["text"].strip()
        
        # Validate query
        if not query:
            respond("❌ Please provide a search query. Usage: `/web What's the latest news about AI?`")
            return
            
        if len(query) > 500:
            respond("❌ Search query is too long. Please use a shorter query.")
            return
        
        logger.info(f"Processing /web command for user {user_id}: '{query}'")
        
        # Run the web search asynchronously
        async def run_web_search():
            try:
                # Get user context for personalization (optional)
                try:
                    from utils.context_builder import get_enhanced_user_context
                    user_specific_context = get_enhanced_user_context(
                        notion_service,
                        user_id,
                        ""
                    )
                except Exception as context_error:
                    logger.warning(f"Could not get user context for web search: {context_error}")
                    user_specific_context = ""
                
                # Add context to the search query if available
                enhanced_query = query
                if user_specific_context:
                    enhanced_query = f"Context about the user: {user_specific_context[:200]}...\n\nUser's question: {query}"
                
                # Perform web search
                response_text, usage = await openai_service.get_web_search_completion_async(
                    enhanced_query,
                    slack_user_id=user_id,
                    notion_service=notion_service
                )
                
                if response_text:
                    # Format the response
                    formatted_response = f"🔍 **Web Search Results for:** {query}\n\n{response_text}"
                    respond(formatted_response)
                    logger.info(f"Web search completed successfully for user {user_id}")
                else:
                    respond("❌ I couldn't find any results for that search. Please try a different query.")
                    
            except Exception as e:
                logger.error(f"Error in web search: {e}")
                respond("❌ I encountered an error during the web search. Please try again.")
        
        # Run the async function
        asyncio.create_task(run_web_search())
        
        # Immediate response to user
        respond(f"🔍 Searching the web for: **{query}**\nResults coming up...")
        
    except Exception as e:
        logger.error(f"Error handling /web command: {e}")
        respond("❌ I encountered an error. Please try again.")

def add_fact_to_notion(slack_user_id: str, fact_text: str) -> bool:
    """Add a fact to the user's Notion page."""
    try:
        # Get or create user page
        page_id = notion_service.get_user_page_id(slack_user_id)
        if not page_id:
            # Create a new user page if it doesn't exist
            success = notion_service.store_user_nickname(slack_user_id, slack_user_id, None)
            if not success:
                return False
            
            page_id = notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                return False
        
        # Create a new fact bullet point
        new_fact_block = {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": fact_text}
                }]
            }
        }
        
        # Append to the page
        result = notion_service.client.blocks.children.append(
            block_id=page_id,
            children=[new_fact_block]
        )
        
        if result and result.get("results"):
            # Invalidate cache
            notion_service.invalidate_user_cache(slack_user_id)
            logger.info(f"Successfully added fact for user {slack_user_id}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error adding fact for user {slack_user_id}: {e}", exc_info=True)
        return False
    
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
        
        # Add "thinking" reaction to the original message
        slack_service.add_reaction(channel_id, message_ts, "thinking_face")
        
        # 3. Create action request for general processing
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
        
        # Handle the response based on success status
        if action_response.success:
            # STEP 1: Update reactions FIRST (before showing response)
            try:
                slack_service.remove_reaction(channel_id, message_ts, "thinking_face")
                slack_service.add_reaction(channel_id, message_ts, "white_check_mark")
                logger.debug("Successfully updated reactions before sending response")
            except Exception as e:
                logger.warning(f"Could not update reactions before response: {e}")
            
            # STEP 2: THEN send the response message
            if action_response.message:
                logger.info(f"Sending action response message (length: {len(action_response.message)})")
                response = slack_service.send_message(
                    channel_id, 
                    action_response.message,
                    action_response.thread_ts or thread_ts
                )
            else:
                logger.info("Action handled message sending directly (e.g., rich blocks)")
                response = {"ok": True, "ts": "handled_by_action"}
                    
        else:
            # STEP 1: Update reactions FIRST for errors too
            try:
                slack_service.add_reaction(channel_id, message_ts, "warning")
                slack_service.remove_reaction(channel_id, message_ts, "thinking_face")
                logger.debug("Successfully updated reactions before sending error response")
            except Exception as e:
                logger.warning(f"Could not update reactions before error response: {e}")
            
            # STEP 2: THEN send error message
            error_msg = action_response.error or "Unknown error"
            logger.error(f"Action failed: {error_msg}")
            
            error_message = action_response.message or "I encountered an error processing your request."
            response = slack_service.send_message(channel_id, error_message, thread_ts)
            
            # Update reactions for error response
            try:
                slack_service.add_reaction(channel_id, message_ts, "warning")
                slack_service.remove_reaction(channel_id, message_ts, "thinking_face")
                logger.debug("Successfully updated reactions for error response")
            except Exception as e:
                logger.warning(f"Could not update reactions after error response: {e}")
        
        # Update channel stats
        slack_service.update_channel_stats(channel_id, user_id, message_ts)
        
        return response
    except Exception as e:
        logger.error(f"Error handling mention: {e}", exc_info=True)
        # Remove thinking reaction and add error indicator
        try:
            slack_service.remove_reaction(channel_id, message_ts, "thinking_face")
            slack_service.add_reaction(channel_id, message_ts, "x")
        except:
            pass
        
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
        # Test regular completion
        content, usage = await openai_service.get_completion_async("Say hello world")
        
        # Test web search completion
        web_content, web_usage = await openai_service.get_web_search_completion_async("Current weather in San Francisco")
        
        return {
            "status": "success",
            "regular_response": content,
            "web_search_response": web_content,
            "model": openai_service.model,
            "regular_usage": usage,
            "web_search_usage": web_usage
        }
    except Exception as e:
        logger.error(f"OpenAI API test error: {e}")
        return {"status": "error", "message": str(e)}
    
# Main entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)