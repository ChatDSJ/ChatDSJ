from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
import re
from loguru import logger

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
    # Get user data from Notion
    user_page_content = notion_service.get_user_page_content(slack_user_id)
    user_properties = notion_service.get_user_page_properties(slack_user_id)
    preferred_name = notion_service.get_user_preferred_name(slack_user_id)
    
    # Create context manager
    context_manager = NotionContextManager()
    
    # Build context string
    context_string = context_manager.build_openai_system_prompt(
        base_prompt=base_prompt,
        notion_content=user_page_content or "",
        preferred_name=preferred_name,
        db_properties=user_properties
    )
    
    return context_string

class InstructionType(Enum):
    """Classification of different types of instructions in the Notion profile."""
    USER_PROFILE = "user_profile"  # Facts about the user
    USER_PREFERENCE = "user_preference"  # How the bot should respond
    APP_INSTRUCTION = "app_instruction"  # How the app should handle commands

@dataclass
class ParsedInstruction:
    """Structured representation of a parsed instruction."""
    type: InstructionType
    content: str
    original_text: str
    section: str
    priority: int = 0  # Higher number = higher priority

class NotionContextManager:
    """Manager for processing Notion context for OpenAI."""
    
    def parse_notion_content(self, content: str) -> Dict[str, Any]:
        """Parse structured Notion content by sections."""
        instructions = []
        current_section = "General"
        raw_sections = {}
        
        # Split into lines and process each line
        lines = content.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers (like "Projects", "Preferences", etc.)
            if (line.endswith(":") or 
                line in ["Projects", "Preferences", "Known Facts", "Instructions"] or
                (i > 0 and lines[i-1].strip() == "")):  # Section header might not have colon
                
                # If it looks like a main header (capitalized, no bullet)
                if not line.startswith("*") and (line[0].isupper() if line else False):
                    current_section = line.rstrip(":")
                    if current_section not in raw_sections:
                        raw_sections[current_section] = []
                    continue
            
            # Add to current section's raw content
            if current_section in raw_sections:
                raw_sections[current_section].append(line)
            
            # Process bullet points
            if line.startswith("*") or line.startswith("-") or line.startswith("•"):
                item_text = line[1:].strip()
                raw_sections.setdefault(current_section, []).append(item_text)
                instruction = self._classify_instruction(item_text, current_section)
                if instruction:
                    instructions.append(instruction)
        
        # Group by type
        profile_data = []
        preferences = []
        app_instructions = []
        
        for instruction in instructions:
            if instruction.type == InstructionType.USER_PROFILE:
                profile_data.append(instruction)
            elif instruction.type == InstructionType.USER_PREFERENCE:
                preferences.append(instruction)
            elif instruction.type == InstructionType.APP_INSTRUCTION:
                app_instructions.append(instruction)
        
        # Sort preferences by priority
        preferences.sort(key=lambda x: x.priority, reverse=True)
        
        return {
            "profile_data": profile_data,
            "preferences": preferences,
            "app_instructions": app_instructions,
            "has_verse_preference": any("rhymed verse" in p.content.lower() for p in preferences),
            "has_format_preference": any(("bullet" in p.content.lower() or "concise" in p.content.lower()) for p in preferences),
            "raw_sections": raw_sections
        }

    def format_original_sections(self, raw_sections: Dict[str, List[str]]) -> str:
        """Format the raw sections to preserve original structure."""
        formatted = []
        
        for section_name, lines in raw_sections.items():
            if not lines:
                continue
                
            formatted.append(f"## {section_name}")
            for line in lines:
                # Preserve bullet points
                if line.startswith("*") or line.startswith("-") or line.startswith("•"):
                    formatted.append(line)
                else:
                    formatted.append(f"* {line}")
            formatted.append("")  # Empty line between sections
        
        return "\n".join(formatted)

    def generate_profile_context(self, profile_data: List[ParsedInstruction]) -> str:
        """
        Generate user profile context string.
        
        Args:
            profile_data: List of profile data instructions
            
        Returns:
            Formatted profile context string
        """
        if not profile_data:
            return ""
        
        sections = {}
        for item in profile_data:
            if item.section not in sections:
                sections[item.section] = []
            sections[item.section].append(item.content)
        
        # Build context string with sections
        parts = []
        for section, items in sections.items():
            parts.append(f"{section}:")
            for item in items:
                parts.append(f"* {item}")
            parts.append("")  # Add blank line between sections
        
        return "\n".join(parts)
    
    def generate_preference_directives(self, preferences: List[ParsedInstruction]) -> str:
        """
        Generate user preference directives string.
        
        Args:
            preferences: List of preference instructions
            
        Returns:
            Formatted preference directives string
        """
        if not preferences:
            return ""
        
        # Start with high priority directive
        parts = ["YOU MUST FOLLOW THESE USER PREFERENCES:"]
        
        # Add each preference as a directive
        for pref in preferences:
            parts.append(f"* {pref.content}")
        
        return "\n".join(parts)
    
    def extract_structured_fields(self, db_properties: Optional[Dict[str, Any]]) -> List[str]:
        structured_facts = []
        if not db_properties:
            return structured_facts
            
        # SIMPLIFIED: Only handle SlackDisplayName (removed all other properties)
        if 'SlackDisplayName' in db_properties:
            prop = db_properties['SlackDisplayName']
            if prop.get("type") == "rich_text":
                texts = prop.get("rich_text", [])
                if texts and texts[0].get("plain_text", "").strip():
                    value = texts[0]["plain_text"].strip()
                    structured_facts.append(f"SlackDisplayName: {value}")
                    logger.debug(f"Added SlackDisplayName: {value}")
        
        return structured_facts

    def process_notion_content(self, content: str) -> Dict[str, Any]:
        if not content:
            return {
                "profile_data": [],
                "preferences": [],
                "app_instructions": [],
                "raw_sections": {}
            }
        
        # SIMPLIFIED: Just split by basic section headers
        raw_sections = {}
        current_section = "General"
        
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for main section headers (SIMPLE detection)
            if line in ["Projects", "Preferences", "Known Facts", "Instructions"]:
                current_section = line
                raw_sections[current_section] = []
                continue
            
            # Add to current section
            if current_section not in raw_sections:
                raw_sections[current_section] = []
            raw_sections[current_section].append(line)
        
        # Return simple structure (for backward compatibility)
        return {
            "profile_data": [],  # Simplified - no complex classification
            "preferences": [],   # Simplified - no complex classification
            "app_instructions": [],  # Simplified - no complex classification
            "raw_sections": raw_sections
        }

    def build_openai_system_prompt(
        self, 
        base_prompt: str, 
        notion_content: str,
        preferred_name: Optional[str] = None,
        db_properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build a complete system prompt with ALL Notion content."""
        # Process the Notion content
        processed = self.process_notion_content(notion_content)
        
        # Extract structured fields from DB properties
        structured_facts = self.extract_structured_fields(db_properties)
        
        # Build the prompt components
        components = [base_prompt]
        
        # Add structured database properties
        if structured_facts:
            components.append("\n\n=== USER DATABASE PROPERTIES ===")
            for fact in structured_facts:
                components.append(f"* {fact}")
            components.append("=== END USER DATABASE PROPERTIES ===")
        
        # Add preferred name prominently
        if preferred_name:
            components.append(f"\n⭐ This user's preferred name is: {preferred_name} ⭐")
        
        # Add all the content sections with their original structure
        if processed.get("raw_sections"):
            components.append("\n=== USER PROFILE CONTENT ===")
            
            # First add the sections we specifically care about in a specific order
            priority_sections = ["Preferences", "Projects", "Known Facts", "Instructions"]
            
            for section in priority_sections:
                if section in processed["raw_sections"]:
                    components.append(f"\n## {section}")
                    for line in processed["raw_sections"][section]:
                        if not line.startswith("*") and not line.startswith("-") and not line.startswith("•"):
                            components.append(f"* {line}")
                        else:
                            components.append(line)
            
            # Then add any remaining sections
            for section, lines in processed["raw_sections"].items():
                if section not in priority_sections and lines:
                    components.append(f"\n## {section}")
                    for line in lines:
                        if not line.startswith("*") and not line.startswith("-") and not line.startswith("•"):
                            components.append(f"* {line}")
                        else:
                            components.append(line)
                            
            components.append("=== END USER PROFILE CONTENT ===")
        
        # If we somehow missed content, include it as a fallback
        raw_content_present = bool(processed.get("raw_sections"))
        generated_content_size = sum(len(component) for component in components)
        
        if notion_content and not raw_content_present and len(notion_content) > 100:
            if generated_content_size < len(notion_content) * 0.5:
                logger.warning(f"Content parsing may have failed. Including raw content as fallback.")
                components.append("\n=== RAW USER CONTENT (FALLBACK) ===")
                components.append(notion_content)
                components.append("=== END RAW USER CONTENT ===")
        
        # Complete prompt
        final_prompt = "\n".join(components)
        logger.debug(f"Final prompt length: {len(final_prompt)} characters")
        
        # Log sections found for debugging
        if processed.get("raw_sections"):
            section_names = list(processed["raw_sections"].keys())
            logger.debug(f"Content sections found: {section_names}")
        
        return final_prompt