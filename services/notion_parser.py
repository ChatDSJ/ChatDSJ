from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
import re

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
    
    def process_notion_content(self, content: str) -> Dict[str, Any]:
        """
        Process raw Notion content into structured components for an LLM prompt.
        
        Args:
            content: The raw Notion page content
            
        Returns:
            Dictionary with structured components
        """
        # Parse the content
        instructions = self.parser.parse_notion_content(content)
        
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
        
        # Sort preferences by priority (highest first)
        preferences.sort(key=lambda x: x.priority, reverse=True)
        
        # Build structured components
        result = {
            "profile_data": profile_data,
            "preferences": preferences,
            "app_instructions": app_instructions,
            "has_verse_preference": any("rhymed verse" in p.content.lower() for p in preferences),
            "has_format_preference": any(("bullet" in p.content.lower() or "concise" in p.content.lower()) for p in preferences)
        }
        
        return result
    
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
    
    def build_openai_system_prompt(
        self, 
        base_prompt: str, 
        notion_content: str,
        preferred_name: Optional[str] = None,
        db_properties: Optional[Dict[str, Any]] = None  # ⬅️ NEW ARG
    ) -> str:
        """
        Build a complete system prompt with structured Notion content.
        
        Args:
            base_prompt: The base system prompt
            notion_content: Raw Notion content
            preferred_name: User's preferred name
            
        Returns:
            Complete system prompt
        """
        # Process the Notion content
        processed = self.process_notion_content(notion_content)
        
        # Build the prompt components
        components = [base_prompt]

        if structured_facts:
            components.append("\n\n--- STRUCTURED USER FIELDS FROM NOTION DATABASE ---")
            for fact in structured_facts:
                components.append(f"* {fact}")
            components.append("--- END STRUCTURED USER FIELDS ---")

        
        # Add user profile information
        profile_context = self.generate_profile_context(processed["profile_data"])
        if profile_context:
            components.append("\n\n--- USER PROFILE INFORMATION ---")
            if preferred_name:
                components.append(f"This user's preferred name is: {preferred_name}")
            components.append(profile_context)
            components.append("--- END USER PROFILE INFORMATION ---")
        
        # Add user preferences with strong emphasis
        if processed["preferences"]:
            preference_directives = self.generate_preference_directives(processed["preferences"])
            components.append("\n\n!!! USER PREFERENCE DIRECTIVES - FOLLOW THESE EXACTLY !!!")
            components.append(preference_directives)
            components.append("!!! END USER PREFERENCE DIRECTIVES !!!")
            
            # Add special emphasis for verse preference if present
            if processed["has_verse_preference"]:
                components.append("\nIMPORTANT: YOU MUST WRITE YOUR RESPONSES IN RHYMED VERSE.\n")
        
        # Extract structured fields from DB properties
        structured_facts = []
        if db_properties:
            for field_name, prop in db_properties.items():
                field_type = prop.get("type")
                if field_type == "rich_text":
                    texts = prop.get("rich_text", [])
                    if texts:
                        value = texts[0].get("plain_text", "").strip()
                        if value:
                            structured_facts.append(f"{field_name}: {value}")
                elif field_type == "title":
                    titles = prop.get("title", [])
                    if titles:
                        value = titles[0].get("plain_text", "").strip()
                        if value:
                            structured_facts.append(f"{field_name}: {value}")
                elif field_type == "select":
                    value = prop.get("select", {}).get("name", "")
                    if value:
                        structured_facts.append(f"{field_name}: {value}")
                elif field_type == "multi_select":
                    values = [v.get("name", "") for v in prop.get("multi_select", [])]
                    if values:
                        structured_facts.append(f"{field_name}: {', '.join(values)}")

        # Complete prompt
        return "\n".join(components)