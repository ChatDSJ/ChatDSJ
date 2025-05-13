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
    
@app.get("/api/diagnose-user-context/{slack_user_id}")
async def diagnose_user_context(slack_user_id: str):
    """Diagnostic endpoint to see what context would be generated."""
    try:
        # Get the raw data
        user_page_content = await asyncio.to_thread(
            notion_service.get_user_page_content,
            slack_user_id
        )
        
        user_properties = await asyncio.to_thread(
            notion_service.get_user_page_properties,
            slack_user_id
        )
        
        preferred_name = await asyncio.to_thread(
            notion_service.get_user_preferred_name,
            slack_user_id
        )
        
        # Generate context
        from utils.context_builder import get_user_context_for_llm
        
        context = await asyncio.to_thread(
            get_user_context_for_llm,
            notion_service,
            slack_user_id
        )
        
        # Prepare diagnostic info
        property_fields = []
        if user_properties:
            for key, prop in user_properties.items():
                prop_type = prop.get("type", "unknown")
                if prop_type == "rich_text":
                    value = prop.get("rich_text", [])
                    if value:
                        text = value[0].get("plain_text", "")
                        property_fields.append(f"{key}: {text} ({prop_type})")
                elif prop_type == "select":
                    value = prop.get("select", {}).get("name", "")
                    property_fields.append(f"{key}: {value} ({prop_type})")
        
        # Extract sections from page content
        sections = {}
        current_section = "General"
        if user_page_content:
            for line in user_page_content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                # Check for section headers
                if line in ["Projects", "Preferences", "Known Facts", "Instructions"]:
                    current_section = line
                    sections[current_section] = []
                elif current_section in sections:
                    sections[current_section].append(line)
        
        # Return diagnostic info
        return {
            "user_id": slack_user_id,
            "preferred_name": preferred_name,
            "properties_count": len(user_properties) if user_properties else 0,
            "property_fields": property_fields,
            "content_length": len(user_page_content) if user_page_content else 0,
            "sections_found": list(sections.keys()),
            "section_items": {k: len(v) for k, v in sections.items()},
            "context_length": len(context),
            "context_preview": context[:500] + "..." if len(context) > 500 else context,
            "content_inclusion_ratio": len(context) / len(user_page_content) if user_page_content else 0
        }
    except Exception as e:
        logger.error(f"Error in diagnostic endpoint: {e}", exc_info=True)
        return {"error": str(e)}
    
@app.get("/api/test-search/{topic}")
async def test_search(topic: str):
    """
    Test the search functionality for a given topic.
    
    Args:
        topic: Topic to search for
        
    Returns:
        Search results
    """
    try:
        from utils.history_manager import HistoryManager
        manager = HistoryManager()
        
        # Test topic extraction
        test_query = f"Has anyone discussed {topic}?"
        extracted_topic = manager.extract_search_topic(test_query)
        
        # Get a random channel to test with
        channels_response = await asyncio.to_thread(
            slack_service.app.client.conversations_list,
            limit=5
        )
        
        if not channels_response.get("channels"):
            return {"error": "No channels available for testing"}
            
        test_channel = channels_response["channels"][0]["id"]
        
        # Get channel history
        messages = await asyncio.to_thread(
            slack_service.fetch_channel_history,
            test_channel,
            100
        )
        
        # Find messages containing the topic
        matching_messages = manager.find_messages_containing_topic(messages, topic)
        
        # Get user display names for formatting
        user_display_names = {}
        for msg in matching_messages:
            user_id = msg.get("user") or msg.get("bot_id", "unknown")
            if user_id not in user_display_names:
                user_display_names[user_id] = await asyncio.to_thread(
                    slack_service.get_user_display_name, 
                    user_id
                )
        
        # Format search results
        formatted_results = manager._format_search_results(
            matching_messages,
            topic,
            user_display_names
        )
        
        return {
            "query": test_query,
            "extracted_topic": extracted_topic,
            "test_channel": test_channel,
            "total_messages": len(messages),
            "matching_messages": len(matching_messages),
            "matching_message_texts": [msg.get("text", "") for msg in matching_messages[:5]],
            "formatted_results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Error testing search: {e}", exc_info=True)
        return {"error": str(e)}

@app.get("/api/test-deep-search/{topic}")
async def test_deep_search(topic: str):
    """
    Test the deep historical and multi-thread search functionality.
    
    Args:
        topic: Topic to search for
        
    Returns:
        Comprehensive search results
    """
    try:
        from utils.history_manager import HistoryManager
        manager = HistoryManager()
        
        # Test topic extraction
        test_query = f"Has anyone discussed {topic}?"
        query_params = manager.analyze_query(test_query)
        
        # Get a random channel to test with
        channels_response = await asyncio.to_thread(
            slack_service.app.client.conversations_list,
            limit=5
        )
        
        if not channels_response.get("channels"):
            return {"error": "No channels available for testing"}
            
        test_channel = channels_response["channels"][0]["id"]
        channel_name = channels_response["channels"][0].get("name", "unknown")
        
        # Perform deep search
        filtered_messages, search_params = await manager.retrieve_and_filter_history(
            slack_service,
            test_channel,
            None,  # No thread context
            test_query
        )
        
        # Get user display names for formatting
        user_display_names = {}
        for msg in filtered_messages:
            user_id = msg.get("user") or msg.get("bot_id", "unknown")
            if user_id not in user_display_names:
                user_display_names[user_id] = await asyncio.to_thread(
                    slack_service.get_user_display_name, 
                    user_id
                )
        
        # Format search results
        formatted_results = manager.format_history_for_prompt(
            filtered_messages,
            query_params,
            user_display_names,
            slack_service.bot_user_id
        )
        
        # Group messages by thread for analysis
        threads = defaultdict(list)
        for msg in filtered_messages:
            thread_key = msg.get("thread_ts") or msg.get("ts")
            threads[thread_key].append(msg)
        
        # Prepare thread summary
        thread_summary = []
        for thread_key, messages in threads.items():
            first_msg = min(messages, key=lambda m: float(m.get("ts", "0")))
            first_user = first_msg.get("user") or first_msg.get("bot_id", "unknown")
            first_user_name = user_display_names.get(first_user, f"User {first_user}")
            
            thread_summary.append({
                "thread_ts": thread_key,
                "message_count": len(messages),
                "started_by": first_user_name,
                "first_message": first_msg.get("text", "")[:100] + "..." if len(first_msg.get("text", "")) > 100 else first_msg.get("text", ""),
                "contains_topic": any(topic.lower() in msg.get("text", "").lower() for msg in messages)
            })
        
        return {
            "query": test_query,
            "channel": {
                "id": test_channel,
                "name": channel_name
            },
            "search_params": query_params,
            "results": {
                "total_messages": len(filtered_messages),
                "thread_count": len(threads),
                "thread_summary": thread_summary,
                "user_count": len(user_display_names),
                "users": list(user_display_names.values()),
                "sample_messages": [msg.get("text", "") for msg in filtered_messages[:5]]
            },
            "formatted_output_preview": formatted_results[:1000] + "..." if len(formatted_results) > 1000 else formatted_results
        }
        
    except Exception as e:
        logger.error(f"Error testing deep search: {e}", exc_info=True)
        return {"error": str(e)}
    
@app.get("/api/test-message-search/{topic}")
async def test_message_search(topic: str):
    """
    Test message content extraction and search for a given topic.
    
    Args:
        topic: Topic to search for in messages
        
    Returns:
        Detailed analysis of messages and search results
    """
    try:
        from utils.history_manager import HistoryManager
        manager = HistoryManager()
        
        # Log the search topic
        logger.info(f"Testing message search for topic: '{topic}'")
        
        # Get a random channel to test with
        channels_response = await asyncio.to_thread(
            slack_service.app.client.conversations_list,
            limit=5
        )
        
        if not channels_response.get("channels"):
            return {"error": "No channels available for testing"}
            
        test_channel = channels_response["channels"][0]["id"]
        channel_name = channels_response["channels"][0].get("name", "unknown")
        
        # Get sample messages from channel
        messages = await asyncio.to_thread(
            slack_service.fetch_channel_history,
            test_channel,
            100  # Get a reasonable number of messages
        )
        
        # Analyze message structure of the first few messages
        message_structures = []
        for i, msg in enumerate(messages[:5]):
            # Extract text content using our method
            extracted_text = manager.extract_full_message_text(msg)
            
            # Prepare simplified structure for analysis
            structure = {
                "text_field": msg.get("text", "")[:100] + "..." if len(msg.get("text", "")) > 100 else msg.get("text", ""),
                "has_blocks": "blocks" in msg,
                "block_count": len(msg.get("blocks", [])),
                "has_attachments": "attachments" in msg,
                "attachment_count": len(msg.get("attachments", [])),
                "extracted_full_text": extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text,
                "contains_topic": topic.lower() in extracted_text.lower(),
                "contains_topic_in_text_field": topic.lower() in msg.get("text", "").lower()
            }
            message_structures.append(structure)
        
        # Find messages containing the topic
        matching_messages = manager.find_messages_containing_topic(messages, topic)
        
        # Analyze matches
        match_analysis = []
        for msg in matching_messages[:5]:  # Limit to 5 for readability
            extracted_text = manager.extract_full_message_text(msg)
            analysis = {
                "text_field": msg.get("text", "")[:100] + "..." if len(msg.get("text", "")) > 100 else msg.get("text", ""),
                "full_extracted_text": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                "found_in_text_field": topic.lower() in msg.get("text", "").lower(),
                "found_in_blocks": any(
                    topic.lower() in manager.extract_full_message_text({"blocks": [block]}).lower()
                    for block in msg.get("blocks", [])
                ),
                "found_in_attachments": any(
                    topic.lower() in manager.extract_full_message_text({"attachments": [attachment]}).lower()
                    for attachment in msg.get("attachments", [])
                )
            }
            match_analysis.append(analysis)
        
        # Return comprehensive analysis
        return {
            "topic": topic,
            "channel": {
                "id": test_channel,
                "name": channel_name
            },
            "message_count": len(messages),
            "matching_count": len(matching_messages),
            "message_structure_samples": message_structures,
            "match_analysis": match_analysis,
            "summary": {
                "messages_with_blocks": sum(1 for msg in messages if "blocks" in msg),
                "messages_with_attachments": sum(1 for msg in messages if "attachments" in msg),
                "matches_only_in_blocks": sum(
                    1 for msg in matching_messages 
                    if topic.lower() not in msg.get("text", "").lower() and 
                    topic.lower() in manager.extract_full_message_text(msg).lower()
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing message search: {e}", exc_info=True)
        return {"error": str(e)}

# Main entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)