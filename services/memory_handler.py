import re
import logging
from typing import Optional
from services.cached_notion_service import CachedNotionService

logger = logging.getLogger(__name__)

def classify_memory_instruction(text: str) -> str:
    lowered = text.lower().strip()
    if "call me" in lowered or "my name is" in lowered or "i am" in lowered:
        return "nickname"
    elif "my new project is" in lowered:
        return "project_replace"
    elif "add project" in lowered:
        return "project_add"
    elif "remember that" in lowered or re.search(r"\bi (work|live|am from|was born in|reside in|moved to)\b", lowered):
        return "known_fact"
    elif "i prefer" in lowered or "i like" in lowered or "my preference" in lowered:
        return "preference"
    return "unknown"

def handle_memory_instruction(
    slack_user_id: str,
    text: str,
    notion_service: CachedNotionService
) -> Optional[str]:
    """
    Process a memory instruction and store it using the given Notion service.
    
    Args:
        slack_user_id: The Slack user ID
        text: The message content
        notion_service: The injected CachedNotionService instance
    
    Returns:
        A user-friendly message indicating success or failure
    """
    memory_type = classify_memory_instruction(text)
    logger.info(f"Classified memory instruction: {memory_type}")

    if memory_type == "known_fact":
        success = notion_service.append_fact_to_user_page(slack_user_id, text.strip())
        logger.info(f"Stored known_fact for user {slack_user_id}: {text.strip()}")
        return "Got it. I’ve added that to what I know about you." if success else "Sorry, I couldn’t remember that right now."

    elif memory_type == "preference":
        success = notion_service.append_preference_to_user_page(slack_user_id, text.strip())
        logger.info(f"Stored preference for user {slack_user_id}: {text.strip()}")
        return "Understood. I’ll keep that in mind for future responses." if success else "Sorry, I couldn’t save that preference."

    return None  # Let the fallback action (e.g., context response) handle it
