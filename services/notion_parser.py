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

class NotionProfileParser:
    """Parser for structured Notion user profiles."""
    
    def __init__(self):
        """Initialize the parser."""
        # Define regex patterns for matching different instruction types
        self.preference_patterns = [
            r"(?:always|please)\s+(.+)",
            r"respond\s+(.+)",
            r"use\s+(.+)",
            r"format\s+(.+)",
            r"write\s+(.+)"
        ]
        
        self.app_instruction_patterns = [
            r"when I say \"([^\"]+)\"",
            r"when I ask \"([^\"]+)\"",
            r"when I (?:mention|state) \"([^\"]+)\""
        ]
    
    def parse_notion_content(self, content: str) -> List[ParsedInstruction]:
        """
        Parse structured Notion content into classified instructions.
        
        Args:
            content: The raw Notion page content
            
        Returns:
            List of parsed and classified instructions
        """
        instructions = []
        current_section = "Unknown"
        
        # Split into lines and process each line
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if line.endswith(":") and not line.startswith("*"):
                current_section = line.rstrip(":")
                continue
            
            # Process bullet points
            if line.startswith("*"):
                item_text = line[1:].strip()
                instruction = self._classify_instruction(item_text, current_section)
                if instruction:
                    instructions.append(instruction)
        
        return instructions
    
    def _classify_instruction(self, text: str, section: str) -> Optional[ParsedInstruction]:
        """
        Classify an instruction based on its content and section.
        
        Args:
            text: The instruction text
            section: The section where the instruction appears
            
        Returns:
            A ParsedInstruction object, or None if not classifiable
        """
        # Normalize section name
        normalized_section = section.lower()
        
        # Classify based on section
        if "basic information" in normalized_section or "location information" in normalized_section:
            # This is user profile data
            return ParsedInstruction(
                type=InstructionType.USER_PROFILE,
                content=text,
                original_text=text,
                section=section
            )
        
        elif "preferences" in normalized_section:
            # This is a user preference
            return ParsedInstruction(
                type=InstructionType.USER_PREFERENCE,
                content=text,
                original_text=text,
                section=section,
                priority=self._calculate_preference_priority(text)
            )
        
        elif "instructions" in normalized_section:
            # Check if this is an app instruction or a user preference
            for pattern in self.app_instruction_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return ParsedInstruction(
                        type=InstructionType.APP_INSTRUCTION,
                        content=text,
                        original_text=text,
                        section=section
                    )
            
            # If no app instruction patterns match, it might be a preference
            for pattern in self.preference_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return ParsedInstruction(
                        type=InstructionType.USER_PREFERENCE,
                        content=text,
                        original_text=text,
                        section=section,
                        priority=self._calculate_preference_priority(text)
                    )
            
            # Default to app instruction if in Instructions section
            return ParsedInstruction(
                type=InstructionType.APP_INSTRUCTION,
                content=text,
                original_text=text,
                section=section
            )
        
        elif "known facts" in normalized_section:
            # These are facts about the user
            return ParsedInstruction(
                type=InstructionType.USER_PROFILE,
                content=text,
                original_text=text,
                section=section
            )
        
        elif "projects" in normalized_section:
            # Project information is part of user profile
            return ParsedInstruction(
                type=InstructionType.USER_PROFILE,
                content=text,
                original_text=text,
                section=section
            )
        
        # Default case - if we can't classify, return None
        return None
    
    def _calculate_preference_priority(self, text: str) -> int:
        """
        Calculate a priority score for a preference.
        Higher scores indicate higher priority.
        
        Args:
            text: The preference text
            
        Returns:
            Priority score (0-100)
        """
        # Start with a base priority
        priority = 50
        
        # Specific formats get high priority
        if "rhymed verse" in text.lower():
            priority += 30
        elif "bullet" in text.lower() or "list" in text.lower():
            priority += 20
        
        # Strong modifiers increase priority
        if "always" in text.lower():
            priority += 10
        if "must" in text.lower():
            priority += 15
        
        # Length penalty for very long preferences
        if len(text) > 100:
            priority -= 10
        
        # Cap at 0-100
        return max(0, min(100, priority))

class NotionContextManager:
    """Manager for processing Notion context for OpenAI."""
    
    def __init__(self):
        """Initialize the context manager."""
        self.parser = NotionProfileParser()
    
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

    def process_notion_content(self, content: str) -> Dict[str, Any]:
        """
        Process raw Notion content into structured components for an LLM prompt.
        
        Args:
            content: The raw Notion page content
            
        Returns:
            Dictionary with structured components
        """
        if not content:
            return {
                "profile_data": [],
                "preferences": [],
                "app_instructions": [],
                "raw_sections": {}
            }
        
        # Extract sections
        raw_sections = {}
        current_section = "General"
        
        # Split into lines and process each line
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for main section headers
            if line in ["Projects", "Preferences", "Known Facts", "Instructions"]:
                current_section = line
                raw_sections[current_section] = []
                continue
            
            # Add to current section
            if current_section not in raw_sections:
                raw_sections[current_section] = []
            
            raw_sections[current_section].append(line)
        
        # For backward compatibility, simulate the old profile_data and preferences structure
        profile_data = []
        preferences = []
        
        # Extract any preferences from the Preferences section
        if "Preferences" in raw_sections:
            for line in raw_sections["Preferences"]:
                if line.startswith("*") or line.startswith("-") or line.startswith("•"):
                    pref_text = line[1:].strip()
                    preferences.append(
                        ParsedInstruction(
                            type=InstructionType.USER_PREFERENCE,
                            content=pref_text,
                            original_text=line,
                            section="Preferences",
                            priority=50  # Give all preferences equal priority
                        )
                    )
                else:
                    preferences.append(
                        ParsedInstruction(
                            type=InstructionType.USER_PREFERENCE,
                            content=line,
                            original_text=line,
                            section="Preferences",
                            priority=50
                        )
                    )
        
        # Extract any instructions for app behavior
        app_instructions = []
        if "Instructions" in raw_sections:
            for line in raw_sections["Instructions"]:
                app_instructions.append(
                    ParsedInstruction(
                        type=InstructionType.APP_INSTRUCTION,
                        content=line,
                        original_text=line,
                        section="Instructions"
                    )
                )
        
        # Return processed content
        return {
            "profile_data": profile_data,
            "preferences": preferences,
            "app_instructions": app_instructions,
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