from typing import Dict, Any, Optional, List
from loguru import logger
from handler.memory_handler import MemoryHandler, PropertyType

def get_enhanced_user_context(notion_service, slack_user_id: str, base_prompt: str = "") -> str:
    """
    Get enhanced user context with strong preference enforcement and language preference.
    
    Args:
        notion_service: The Notion service
        slack_user_id: The Slack user ID
        base_prompt: Optional base prompt to include
        
    Returns:
        Formatted context string with language preference prominently featured
    """
    # Get user data from Notion
    user_page_content = notion_service.get_user_page_content(slack_user_id)
    user_properties = notion_service.get_user_page_properties(slack_user_id)
    preferred_name = notion_service.get_user_preferred_name(slack_user_id)
    
    logger.info(f"Building enhanced context for user {slack_user_id}")
    if preferred_name:
        logger.info(f"Using preferred name: {preferred_name}")
    
    if user_page_content:
        logger.info(f"Page content length: {len(user_page_content)} chars")
    else:
        logger.warning("No page content found")
    
    # Get language preference from database properties (NEW)
    language_preference = get_user_language_preference(user_properties)
    logger.info(f"Language preference: {language_preference}")
    
    # Extract structured fields
    structured_facts = extract_structured_fields(user_properties)
    logger.debug(f"Extracted {len(structured_facts)} structured fields")
    
    # Extract user preferences from content (but NOT language preferences)
    user_preferences = extract_user_preferences(user_page_content, exclude_language=True)
    
    # Build the enhanced context
    components = []
    
    # CRITICAL: Add language preference at the very beginning with maximum emphasis
    components.append("=" * 80)
    components.append("🌍 CRITICAL LANGUAGE INSTRUCTION 🌍")
    components.append(f"ALWAYS RESPOND IN: {language_preference.upper()}")
    components.append(f"USER'S LANGUAGE PREFERENCE: {language_preference}")
    components.append("This is MANDATORY - all responses must be in this language.")
    components.append("=" * 80)
    components.append("")
    
    # Add all important directives at the beginning for emphasis
    if user_preferences:
        components.append("=== IMPORTANT USER PREFERENCES - FOLLOW THESE EXACTLY ===")
        for pref in user_preferences:
            components.append(f"* {pref}")
        components.append("=== END USER PREFERENCES ===\n")
    
    # Add base prompt if provided
    if base_prompt:
        components.append(base_prompt)
    
    # Add user database properties (excluding language preference since it's already prominently featured)
    if structured_facts:
        components.append("=== USER DATABASE PROPERTIES ===")
        for fact in structured_facts:
            # Skip language preference here since it's already prominently featured above
            if not fact.startswith("LanguagePreference:"):
                components.append(f"* {fact}")
        components.append("=== END USER DATABASE PROPERTIES ===\n")
    
    # Add preferred name with emphasis
    if preferred_name:
        components.append(f"⭐ This user's preferred name is: {preferred_name} ⭐\n")
    
    # Add page content with section structure preserved
    if user_page_content:
        components.append("=== USER PROFILE CONTENT ===")
        
        # Process content preserving section structure
        current_section = None
        for line in user_page_content.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a section header
            if line in ["Projects", "Preferences", "Known Facts", "Instructions"]:
                current_section = line
                components.append(f"\n## {current_section}")
            elif line:
                # Add the line, preserving bullet points
                components.append(line)
        
        components.append("=== END USER PROFILE CONTENT ===\n")
    
    # Add a final reminder about language preference and user preferences
    components.append("FINAL REMINDERS:")
    components.append(f"1. 🌍 ALWAYS respond in {language_preference} - this is MANDATORY")
    components.append("2. Always respect this user's stated preferences in your responses.")
    components.append("3. Use the user's preferred name when addressing them.")
    
    # Build the final context string
    context_string = "\n".join(components)
    logger.info(f"Generated enhanced context length: {len(context_string)} chars")
    logger.info(f"Language preference featured prominently: {language_preference}")
    
    return context_string

def get_user_language_preference(db_properties: Optional[Dict[str, Any]]) -> str:
    """
    Get the user's language preference from database properties with English as default.
    
    Args:
        db_properties: Dictionary of database properties
        
    Returns:
        The language preference (default: "English")
    """
    if not db_properties:
        logger.debug("No database properties provided, defaulting to English")
        return "English"
    
    # Look for LanguagePreference property
    language_pref_prop = db_properties.get("LanguagePreference")
    
    if language_pref_prop and language_pref_prop.get("type") == "rich_text":
        rich_text_array = language_pref_prop.get("rich_text", [])
        if rich_text_array and len(rich_text_array) > 0:
            language = rich_text_array[0].get("plain_text", "").strip()
            if language:
                logger.debug(f"Found language preference: {language}")
                return language
    
    logger.debug("No language preference found, defaulting to English")
    return "English"

def extract_user_preferences(page_content: Optional[str], exclude_language: bool = True) -> List[str]:
    """
    Extract user preferences from page content, optionally excluding language preferences.
    
    Args:
        page_content: The raw page content
        exclude_language: Whether to exclude language-related preferences
        
    Returns:
        List of preference strings (excluding language preferences if requested)
    """
    if not page_content:
        return []
        
    preferences = []
    in_preferences_section = False
    
    for line in page_content.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        # Check for preferences section
        if line == "Preferences":
            in_preferences_section = True
            continue
            
        # Extract preferences from section
        if in_preferences_section:
            # Check if we've moved to a new section
            if line in ["Projects", "Known Facts", "Instructions"]:
                in_preferences_section = False
                continue
                
            # Extract the preference, removing bullet markers
            if line.startswith("*") or line.startswith("-") or line.startswith("•"):
                pref = line[1:].strip()
            else:
                pref = line
            
            # NEW: Skip language-related preferences if exclude_language is True
            if exclude_language:
                pref_lower = pref.lower()
                language_keywords = ["respond", "answer", "communicate", "talk", "speak", "reply", "language"]
                
                # Check if this preference is about language
                if any(keyword in pref_lower for keyword in language_keywords):
                    logger.debug(f"Excluding language preference: '{pref}'")
                    continue
            
            preferences.append(pref)
    
    logger.debug(f"Extracted user preferences (exclude_language={exclude_language}): {preferences}")
    return preferences

def extract_structured_fields(db_properties: Optional[Dict[str, Any]]) -> List[str]:
    """
    Extract structured fields from database properties.
    
    Args:
        db_properties: Dictionary of database properties
        
    Returns:
        List of formatted field strings
    """
    structured_facts = []
    if not db_properties:
        logger.debug("No database properties provided for structured fields")
        return structured_facts
        
    for field_name, prop in db_properties.items():
        field_type = prop.get("type")
        if field_type == "rich_text":
            texts = prop.get("rich_text", [])
            if texts:
                value = texts[0].get("plain_text", "").strip()
                if value:
                    # Skip preferred name to avoid duplication
                    if field_name == "PreferredName":
                        logger.debug(f"Skipping PreferredName in structured fields to avoid duplication")
                        continue
                    structured_facts.append(f"{field_name}: {value}")
                    logger.debug(f"Added rich_text field: {field_name} = {value}")
        elif field_type == "title":
            titles = prop.get("title", [])
            if titles:
                value = titles[0].get("plain_text", "").strip()
                if value:
                    structured_facts.append(f"{field_name}: {value}")
                    logger.debug(f"Added title field: {field_name} = {value}")
        elif field_type == "select":
            value = prop.get("select", {}).get("name", "")
            if value:
                structured_facts.append(f"{field_name}: {value}")
                logger.debug(f"Added select field: {field_name} = {value}")
        elif field_type == "multi_select":
            values = [v.get("name", "") for v in prop.get("multi_select", [])]
            if values:
                structured_facts.append(f"{field_name}: {', '.join(values)}")
                logger.debug(f"Added multi_select field: {field_name} = {', '.join(values)}")
    
    logger.debug(f"Structured fields included in prompt: {structured_facts}")
    return structured_facts

def get_user_context_for_llm(notion_service, slack_user_id: str, base_prompt: str = "") -> str:
    """
    Get user context formatted for an LLM prompt with language preference support.
    
    Args:
        notion_service: The Notion service
        slack_user_id: The Slack user ID
        base_prompt: Optional base prompt to include
        
    Returns:
        Formatted context string with language preference prominently featured
    """
    # Use the enhanced context builder that includes language preference
    return get_enhanced_user_context(notion_service, slack_user_id, base_prompt)

