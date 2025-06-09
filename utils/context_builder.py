from typing import Dict, Any, Optional, List
from loguru import logger

def get_enhanced_user_context(notion_service, slack_user_id: str, base_prompt: str = "") -> str:
    # Get user data from Notion (SAME AS BEFORE)
    user_page_content = notion_service.get_user_page_content(slack_user_id)
    user_properties = notion_service.get_user_page_properties(slack_user_id)
    preferred_name = notion_service.get_user_preferred_name(slack_user_id)
    
    logger.info(f"Building context for user {slack_user_id}")
    if preferred_name:
        logger.info(f"Using preferred name: {preferred_name}")
    
    if user_page_content:
        logger.info(f"Page content length: {len(user_page_content)} chars")
    else:
        logger.warning("No page content found")
    
    # SIMPLIFIED: Extract only useful structured fields
    structured_facts = extract_structured_fields(user_properties)
    logger.debug(f"Extracted {len(structured_facts)} structured fields")
    
    # Build the context (SAME STRUCTURE AS BEFORE)
    components = []
    
    # Add base prompt if provided
    if base_prompt:
        components.append(base_prompt)
    
    # Add user database properties (SIMPLIFIED)
    if structured_facts:
        components.append("=== USER DATABASE PROPERTIES ===")
        for fact in structured_facts:
            components.append(f"* {fact}")
        components.append("=== END USER DATABASE PROPERTIES ===\n")
    
    # Add preferred name with emphasis (SAME AS BEFORE)
    if preferred_name:
        components.append(f"⭐ This user's preferred name is: {preferred_name} ⭐\n")
    
    # Add page content (SAME AS BEFORE)
    if user_page_content:
        components.append("=== USER FACTS ===")
        components.append(user_page_content)
        components.append("=== END USER FACTS ===\n")
    
    # Build the final context string (SAME AS BEFORE)
    context_string = "\n".join(components)
    logger.info(f"Generated context length: {len(context_string)} chars")
    
    return context_string

def extract_structured_fields(db_properties: Optional[Dict[str, Any]]) -> List[str]:
    structured_facts = []
    if not db_properties:
        return structured_facts
        
    # SIMPLIFIED: Only handle the properties we actually use
    simple_properties = {
        'SlackDisplayName': 'Slack Display Name',
        # Remove: LanguagePreference, WorkLocation, HomeLocation, Role, Timezone, FavoriteEmoji
    }
    
    for prop_key, display_name in simple_properties.items():
        if prop_key not in db_properties:
            continue
            
        prop = db_properties[prop_key]
        prop_type = prop.get('type')
        
        # Only handle rich_text (keep it simple)
        if prop_type == 'rich_text':
            texts = prop.get('rich_text', [])
            if texts and texts[0].get('plain_text', '').strip():
                value = texts[0]['plain_text'].strip()
                structured_facts.append(f"{display_name}: {value}")
    
    return structured_facts

def get_user_context_for_llm(notion_service, slack_user_id: str, base_prompt: str = "") -> str:
    """
    Get user context formatted for an LLM prompt.
    
    Args:
        notion_service: The Notion service
        slack_user_id: The Slack user ID
        base_prompt: Optional base prompt to include
        
    Returns:
        Formatted context string
    """
    return get_enhanced_user_context(notion_service, slack_user_id, base_prompt)