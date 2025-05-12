import re
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from loguru import logger

class SectionType(Enum):
    """Enum for the different sections in the user profile."""
    BASIC_INFO = "Basic Information"
    LOCATION_INFO = "Location Information"
    PROJECTS = "Projects"
    PREFERENCES = "Preferences"
    KNOWN_FACTS = "Known Facts"
    INSTRUCTIONS = "Instructions"

class PropertyType(Enum):
    """Enum for the database properties."""
    PREFERRED_NAME = "PreferredName"
    WORK_LOCATION = "WorkLocation"
    HOME_LOCATION = "HomeLocation"
    ROLE = "Role"
    TIMEZONE = "Timezone"
    FAVORITE_EMOJI = "FavoriteEmoji"
    LANGUAGE_PREFERENCE = "LanguagePreference"
    SLACK_DISPLAY_NAME = "SlackDisplayName"
    SLACK_USER_ID = "SlackUserID"

class MemoryHandler:
    """Enhanced memory handler for structured user profiles."""
    
    def __init__(self, notion_service):
        """Initialize with a reference to the Notion service."""
        self.notion_service = notion_service
    
    def handle_memory_instruction(self, slack_user_id: str, text: str) -> Optional[str]:
        """
        Process a memory instruction and store it appropriately.
        
        Args:
            slack_user_id: The Slack user ID
            text: The message content
        
        Returns:
            A user-friendly message indicating success or failure, or None if not a memory instruction
        """
        # Classify the memory instruction and get cleaned content
        memory_type, cleaned_content = self.classify_memory_instruction(text)
        
        # If not a memory instruction, return None
        if memory_type == "unknown" or cleaned_content is None:
            return None
        
        # Handle different types of memory instructions
        if memory_type == "nickname":
            success = self.update_user_name(slack_user_id, cleaned_content)
            return f"Got it! I'll call you {cleaned_content} from now on." if success else "Sorry, I couldn't update your name right now."
        
        elif memory_type == "work_location":
            success = self.update_location(slack_user_id, "work", cleaned_content)
            return f"Updated! I've noted that you work from {cleaned_content}." if success else "Sorry, I couldn't update your work location right now."
        
        elif memory_type == "home_location":
            success = self.update_location(slack_user_id, "home", cleaned_content)
            return f"Updated! I've noted that you're based in {cleaned_content}." if success else "Sorry, I couldn't update your home location right now."
        
        elif memory_type == "known_fact":
            success = self.add_known_fact(slack_user_id, cleaned_content)
            return "Got it. I've added that to what I know about you." if success else "Sorry, I couldn't remember that right now."
        
        elif memory_type == "preference":
            success = self.add_preference(slack_user_id, cleaned_content)
            return "Understood. I'll keep that in mind for future responses." if success else "Sorry, I couldn't save that preference right now."
        
        elif memory_type == "project_replace":
            success = self.replace_projects(slack_user_id, cleaned_content)
            return "Got it. I've updated your project list." if success else "Sorry, I couldn't update your projects right now."
        
        elif memory_type == "project_add":
            success = self.add_project(slack_user_id, cleaned_content)
            return "Got it. I've added that to your project list." if success else "Sorry, I couldn't add that project right now."
        
        elif memory_type == "todo":
            success = self.add_todo(slack_user_id, cleaned_content)
            return "✅ Added to your TODO list." if success else "Sorry, I couldn't add that to your TODO list right now."
        
        return None
    
    def classify_memory_instruction(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Classify a message as a memory instruction type and extract the clean content.
        
        Args:
            text: The message content
            
        Returns:
            A tuple of (memory_type, cleaned_content) where cleaned_content is None 
            if the message is not a memory instruction
        """
        lowered = text.lower().strip()
        
        # First check if this is a question - if so, it's not a memory instruction
        question_patterns = [
            r"^(?:what|where|when|who|how|why|do|does|can|could|would|should|is|are|am)(?:\s|\?)",  # Starts with question word
            r"\?$",  # Ends with question mark
            r"^(?:tell me|do you know|can you tell)"  # Asking for information
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, lowered):
                return "unknown", None  # This is a question, not a memory instruction
        
        # Clean up common greetings and bot mentions
        cleaned_text = text.strip()
        greeting_patterns = [
            r"^(?:hey|hello|hi|greetings|yo|hiya)(?:\s+(?:there|you|everyone|all|team|folks))?\s+",
            r"^(?:good\s+(?:morning|afternoon|evening|day))(?:\s+(?:there|you|everyone|all|team|folks))?\s+"
        ]
        
        for pattern in greeting_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
        
        # Remove bot mentions
        bot_mention_pattern = r"<@[A-Z0-9]+>\s*"
        cleaned_text = re.sub(bot_mention_pattern, "", cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # TODO command has highest priority
        if re.search(r"^todo:", lowered, re.IGNORECASE) or re.search(r"\btodo:", lowered, re.IGNORECASE):
            todo_match = re.search(r"todo:(.*)", lowered, re.IGNORECASE)
            todo_text = todo_match.group(1).strip() if todo_match else cleaned_text
            return "todo", todo_text
        
        # Explicit statements that should be remembered
        if "remember that" in lowered or "note that" in lowered:
            fact_text = re.sub(r"^remember that\s+", "", cleaned_text, flags=re.IGNORECASE)
            fact_text = re.sub(r"^note that\s+", "", fact_text, flags=re.IGNORECASE)
            return "known_fact", fact_text
        
        # Nickname setting
        if (re.search(r"\bcall me\b", lowered) or 
            re.search(r"\bmy name is\b", lowered) or 
            re.search(r"\bi am called\b", lowered)):
            name_match = re.search(r"(?:call me|my name is|i am called)\s+['\"]?([^'\"]+)['\"]?", lowered)
            if name_match:
                return "nickname", name_match.group(1).strip()
            return "nickname", cleaned_text
        
        # Projects with explicit markers
        if "my new project is" in lowered:
            project_text = re.sub(r".*?my new project is\s+", "", cleaned_text, flags=re.IGNORECASE)
            return "project_replace", project_text
        
        if "add project" in lowered:
            project_text = re.sub(r".*?add project\s+", "", cleaned_text, flags=re.IGNORECASE)
            return "project_add", project_text
        
        # Facts with explicit first-person statements about location
        location_match = re.search(r"\bi (?:work|live|am from|was born in|reside in|moved to)\s+(.*?)(?:\.|\s*$)", lowered)
        if location_match and not re.search(r"\bwhere\s+(?:do|does)\s+i\b", lowered):
            location_type = "work_location" if "work" in location_match.group(0) else "home_location"
            location_text = location_match.group(1).strip()
            
            # Extract just the location by removing any "remotely from" that might be part of the extracted text
            if "work" in location_match.group(0):
                location_text = re.sub(r"^(?:remotely\s+from\s+|from\s+)", "", location_text)
                clean_fact = location_text
            else:
                clean_fact = location_text
                
            return location_type, clean_fact
        
        # Preferences with explicit first-person statements
        if (re.search(r"\bi prefer\b", lowered) or 
            re.search(r"\bi (?:like|love|enjoy|hate|dislike)\b", lowered) or 
            re.search(r"\bmy preference is\b", lowered)):
            return "preference", cleaned_text
        
        # General facts with first-person statements
        if re.search(r"\bi (?:[a-z]+)\b", lowered) and not re.search(r"\bi (?:want|need|would like|am looking for)\b", lowered):
            return "known_fact", cleaned_text
        
        # No pattern matched
        return "unknown", None
    
    def update_user_name(self, slack_user_id: str, name: str) -> bool:
        """
        Update a user's preferred name.
        
        Args:
            slack_user_id: The Slack user ID
            name: The preferred name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Update the property
            success = self.notion_service.store_user_nickname(slack_user_id, name)
            if not success:
                return False
            
            # 2. Get the page ID
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                return False
            
            # 3. Find and update the "Name: [PreferredName property]" in Basic Information section
            section_block = self._find_section_block(page_id, SectionType.BASIC_INFO.value)
            if not section_block:
                # Create the Basic Information section if it doesn't exist
                self._create_section(page_id, SectionType.BASIC_INFO.value)
                return True  # The property is already updated, so return True
            
            # Find the Name item in the Basic Information section
            name_block = self._find_text_block_in_section(
                page_id, 
                section_block.get("id"), 
                "Name:", 
                exact_match=False
            )
            
            if name_block:
                # Update the existing Name block
                self.notion_service.client.blocks.update(
                    block_id=name_block.get("id"),
                    bulleted_list_item={
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": f"Name: [{PropertyType.PREFERRED_NAME.value} property]"}
                        }]
                    }
                )
            else:
                # Add a new Name block
                self.notion_service.client.blocks.children.append(
                    block_id=section_block.get("id"),
                    children=[{
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": f"Name: [{PropertyType.PREFERRED_NAME.value} property]"}
                            }]
                        }
                    }]
                )
            
            # Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating user name for {slack_user_id}: {e}", exc_info=True)
            return False
    
    def update_location(self, slack_user_id: str, location_type: str, location: str) -> bool:
        """
        Update a user's location.
        
        Args:
            slack_user_id: The Slack user ID
            location_type: Either "work" or "home"
            location: The location value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Determine the property name
            property_name = PropertyType.WORK_LOCATION.value if location_type == "work" else PropertyType.HOME_LOCATION.value
            
            # 2. Get the page ID
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id)
                if not success:
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    return False
            
            # 3. Update the property
            properties_update = {
                property_name: {"rich_text": [{"type": "text", "text": {"content": location}}]}
            }
            self.notion_service.client.pages.update(page_id=page_id, properties=properties_update)
            
            # 4. Find and update the location in the Location Information section
            location_section_block = self._find_section_block(page_id, SectionType.LOCATION_INFO.value)
            if not location_section_block:
                # Create the Location Information section if it doesn't exist
                location_section_block = self._create_section(page_id, SectionType.LOCATION_INFO.value)
            
            # Find the location item in the Location Information section
            location_label = "Work Location:" if location_type == "work" else "Home Location:"
            location_block = self._find_text_block_in_section(
                page_id, 
                location_section_block.get("id"), 
                location_label, 
                exact_match=False
            )
            
            if location_block:
                # Update the existing location block
                self.notion_service.client.blocks.update(
                    block_id=location_block.get("id"),
                    bulleted_list_item={
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": f"{location_label} [{property_name} property]"}
                        }]
                    }
                )
            else:
                # Add a new location block
                self.notion_service.client.blocks.children.append(
                    block_id=location_section_block.get("id"),
                    children=[{
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": f"{location_label} [{property_name} property]"}
                            }]
                        }
                    }]
                )
            
            # Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating location for {slack_user_id}: {e}", exc_info=True)
            return False
    
    def add_known_fact(self, slack_user_id: str, fact: str) -> bool:
        """
        Add a fact to the Known Facts section.
        
        Args:
            slack_user_id: The Slack user ID
            fact: The fact to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Get the page ID
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id)
                if not success:
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    return False
            
            # 2. Find the Known Facts section
            known_facts_section_block = self._find_section_block(page_id, SectionType.KNOWN_FACTS.value)
            if not known_facts_section_block:
                # Create the Known Facts section if it doesn't exist
                known_facts_section_block = self._create_section(page_id, SectionType.KNOWN_FACTS.value)
            
            # 3. Add the new fact as a bullet point
            self.notion_service.client.blocks.children.append(
                block_id=known_facts_section_block.get("id"),
                children=[{
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": fact}
                        }]
                    }
                }]
            )
            
            # Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding known fact for {slack_user_id}: {e}", exc_info=True)
            return False
    
    def add_preference(self, slack_user_id: str, preference: str) -> bool:
        """
        Add a preference to the Preferences section.
        
        Args:
            slack_user_id: The Slack user ID
            preference: The preference to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Get the page ID
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id)
                if not success:
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    return False
            
            # 2. Find the Preferences section
            preferences_section_block = self._find_section_block(page_id, SectionType.PREFERENCES.value)
            if not preferences_section_block:
                # Create the Preferences section if it doesn't exist
                preferences_section_block = self._create_section(page_id, SectionType.PREFERENCES.value)
            
            # 3. Add the new preference as a bullet point
            self.notion_service.client.blocks.children.append(
                block_id=preferences_section_block.get("id"),
                children=[{
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": preference}
                        }]
                    }
                }]
            )
            
            # Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding preference for {slack_user_id}: {e}", exc_info=True)
            return False
    
    def replace_projects(self, slack_user_id: str, project: str) -> bool:
        """
        Replace the Projects section with a new project.
        
        Args:
            slack_user_id: The Slack user ID
            project: The new project
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Get the page ID
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id)
                if not success:
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    return False
            
            # 2. Find the Projects section
            projects_section_block = self._find_section_block(page_id, SectionType.PROJECTS.value)
            if not projects_section_block:
                # Create the Projects section if it doesn't exist
                projects_section_block = self._create_section(page_id, SectionType.PROJECTS.value)
                
                # Add the new project as a bullet point
                self.notion_service.client.blocks.children.append(
                    block_id=projects_section_block.get("id"),
                    children=[{
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": project}
                            }]
                        }
                    }]
                )
                
                # Invalidate cache
                self.notion_service.invalidate_user_cache(slack_user_id)
                
                return True
            
            # 3. Delete all existing projects
            children_response = self.notion_service.client.blocks.children.list(
                block_id=projects_section_block.get("id")
            )
            
            for block in children_response.get("results", []):
                if block.get("type") == "bulleted_list_item":
                    self.notion_service.client.blocks.delete(block_id=block.get("id"))
            
            # 4. Add the new project as a bullet point
            self.notion_service.client.blocks.children.append(
                block_id=projects_section_block.get("id"),
                children=[{
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": project}
                        }]
                    }
                }]
            )
            
            # Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error replacing projects for {slack_user_id}: {e}", exc_info=True)
            return False
    
    def add_project(self, slack_user_id: str, project: str) -> bool:
        """
        Add a project to the Projects section.
        
        Args:
            slack_user_id: The Slack user ID
            project: The project to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Get the page ID
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id)
                if not success:
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    return False
            
            # 2. Find the Projects section
            projects_section_block = self._find_section_block(page_id, SectionType.PROJECTS.value)
            if not projects_section_block:
                # Create the Projects section if it doesn't exist
                projects_section_block = self._create_section(page_id, SectionType.PROJECTS.value)
            
            # 3. Add the new project as a bullet point
            self.notion_service.client.blocks.children.append(
                block_id=projects_section_block.get("id"),
                children=[{
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": project}
                        }]
                    }
                }]
            )
            
            # Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding project for {slack_user_id}: {e}", exc_info=True)
            return False
    
    def add_todo(self, slack_user_id: str, todo: str) -> bool:
        """
        Add a TODO item.
        
        Args:
            slack_user_id: The Slack user ID
            todo: The TODO item
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use the existing add_todo_item method from the Notion service
            return self.notion_service.add_todo_item(slack_user_id, todo)
        
        except Exception as e:
            logger.error(f"Error adding TODO item for {slack_user_id}: {e}", exc_info=True)
            return False
    
    def _find_section_block(self, page_id: str, section_name: str) -> Optional[Dict[str, Any]]:
        """
        Find a section block by name.
        
        Args:
            page_id: The page ID
            section_name: The name of the section
            
        Returns:
            The section block if found, None otherwise
        """
        # Get all blocks
        blocks_response = self.notion_service.client.blocks.children.list(block_id=page_id)
        
        # Look for a heading block with the section name
        for block in blocks_response.get("results", []):
            block_type = block.get("type")
            if block_type in ["heading_1", "heading_2", "heading_3"]:
                text_content = ""
                rich_text = block.get(block_type, {}).get("rich_text", [])
                
                for text_item in rich_text:
                    text_content += text_item.get("plain_text", "")
                
                if text_content == section_name:
                    return block
        
        return None
    
    def _find_text_block_in_section(self, page_id: str, section_id: str, text_prefix: str, exact_match: bool = False) -> Optional[Dict[str, Any]]:
        """
        Find a text block in a section that starts with the given prefix.
        
        Args:
            page_id: The page ID
            section_id: The section block ID
            text_prefix: The text prefix to look for
            exact_match: Whether to require an exact match
            
        Returns:
            The text block if found, None otherwise
        """
        # Get all blocks in the section
        blocks_response = self.notion_service.client.blocks.children.list(block_id=section_id)
        
        # Look for a text block that starts with the given prefix
        for block in blocks_response.get("results", []):
            block_type = block.get("type")
            if block_type in ["paragraph", "bulleted_list_item"]:
                text_content = ""
                rich_text = block.get(block_type, {}).get("rich_text", [])
                
                for text_item in rich_text:
                    text_content += text_item.get("plain_text", "")
                
                if (exact_match and text_content == text_prefix) or (not exact_match and text_content.startswith(text_prefix)):
                    return block
        
        return None
    
    def _create_section(self, page_id: str, section_name: str) -> Dict[str, Any]:
        """
        Create a section block.
        
        Args:
            page_id: The page ID
            section_name: The name of the section
            
        Returns:
            The created section block
        """
        # Create a heading block
        result = self.notion_service.client.blocks.children.append(
            block_id=page_id,
            children=[{
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": section_name}
                    }]
                }
            }]
        )
        
        # Return the created block
        return result.get("results", [{}])[0]