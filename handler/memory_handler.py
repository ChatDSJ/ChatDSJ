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
        if memory_type == "unknown" or (cleaned_content is None and not memory_type.startswith("list_")):
            logger.debug(f"Not a memory instruction: '{text}'")
            return None
        
        logger.info(f"Processing memory instruction type: {memory_type}, content: '{cleaned_content}'")
        
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
            return "✅ Added to your Known Facts: \"" + cleaned_content + "\"" if success else "Sorry, I couldn't remember that right now."
        
        elif memory_type == "preference":
            success = self.add_preference(slack_user_id, cleaned_content)
            return "✅ Added to your Preferences: \"" + cleaned_content + "\"" if success else "Sorry, I couldn't save that preference right now."
        
        elif memory_type == "project_replace":
            success = self.replace_projects(slack_user_id, cleaned_content)
            return "✅ Updated your project list to: \"" + cleaned_content + "\"" if success else "Sorry, I couldn't update your projects right now."
        
        elif memory_type == "project_add":
            success = self.add_project(slack_user_id, cleaned_content)
            return "✅ Added to your Projects: \"" + cleaned_content + "\"" if success else "Sorry, I couldn't add that project right now."
        
        elif memory_type == "todo":
            success = self.add_todo(slack_user_id, cleaned_content)
            return "✅ Added to your TODO list: \"" + cleaned_content + "\"" if success else "Sorry, I couldn't add that to your TODO list right now."
        
        # New management commands
        elif memory_type == "list_facts":
            facts = self.get_known_facts(slack_user_id)
            if not facts:
                return "You don't have any facts stored yet. You can add facts by saying \"Remember [your fact]\" or \"Fact: [your fact]\"."
            facts_list = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])
            return f"Here are your stored facts:\n\n{facts_list}"
        
        elif memory_type == "list_preferences":
            preferences = self.get_preferences(slack_user_id)
            if not preferences:
                return "You don't have any preferences stored yet. You can add preferences by saying \"I prefer [your preference]\" or \"Preference: [your preference]\"."
            preferences_list = "\n".join([f"{i+1}. {pref}" for i, pref in enumerate(preferences)])
            return f"Here are your stored preferences:\n\n{preferences_list}"
        
        elif memory_type == "list_projects":
            projects = self.get_projects(slack_user_id)
            if not projects:
                return "You don't have any projects stored yet. You can add projects by saying \"Project: [your project]\" or \"Add project [your project]\"."
            projects_list = "\n".join([f"{i+1}. {proj}" for i, proj in enumerate(projects)])
            return f"Here are your stored projects:\n\n{projects_list}"
        
        elif memory_type == "delete_fact":
            # IMPROVED: More explicit feedback with detailed info about what was deleted or not found
            before_facts = self.get_known_facts(slack_user_id)
            success = self.delete_known_fact(slack_user_id, cleaned_content)
            
            if success:
                # Get facts after deletion to confirm what was deleted
                after_facts = self.get_known_facts(slack_user_id)
                
                # Find which fact was deleted by comparing before and after
                deleted_fact = None
                for fact in before_facts:
                    if fact not in after_facts and cleaned_content.lower() in fact.lower():
                        deleted_fact = fact
                        break
                
                if deleted_fact:
                    return f"✅ Successfully removed \"{deleted_fact}\" from your Known Facts."
                else:
                    return f"✅ Removed fact about \"{cleaned_content}\" from your Known Facts."
            else:
                # If deletion failed, check if there are any facts containing the fragment
                matching_facts = [fact for fact in before_facts if cleaned_content.lower() in fact.lower()]
                
                if matching_facts:
                    # Facts matching the fragment exist, but deletion failed
                    facts_list = "\n".join([f"• {fact}" for fact in matching_facts])
                    return f"❌ Error: I found facts matching \"{cleaned_content}\" but couldn't delete them. Technical issue.\n\nMatching facts:\n{facts_list}"
                else:
                    # No facts matching the fragment exist
                    return f"❓ I couldn't find any facts containing \"{cleaned_content}\" to remove."
        
        elif memory_type == "delete_preference":
            success = self.delete_preference(slack_user_id, cleaned_content)
            return f"✅ Removed preference about \"{cleaned_content}\" from your Preferences." if success else f"Sorry, I couldn't find a preference about \"{cleaned_content}\" to remove."
        
        elif memory_type == "delete_project":
            success = self.delete_project(slack_user_id, cleaned_content)
            return f"✅ Removed project \"{cleaned_content}\" from your Projects." if success else f"Sorry, I couldn't find a project \"{cleaned_content}\" to remove."
        
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
                logger.debug(f"Detected question pattern in: '{text}'")
                return "unknown", None  # This is a question, not a memory instruction
        
        # Direct command checking - these have priority over other patterns
        if lowered.startswith("fact:") or lowered.startswith("fact "):
            content = re.sub(r'^fact[:\s]\s*', '', text, flags=re.IGNORECASE).strip()
            logger.debug(f"Detected fact command: '{content}'")
            return "known_fact", content
            
        if lowered.startswith("project:") or lowered.startswith("project "):
            content = re.sub(r'^project[:\s]\s*', '', text, flags=re.IGNORECASE).strip()
            logger.debug(f"Detected project command: '{content}'")
            return "project_add", content
            
        if lowered.startswith("preference:") or lowered.startswith("preference "):
            content = re.sub(r'^preference[:\s]\s*', '', text, flags=re.IGNORECASE).strip()
            logger.debug(f"Detected preference command: '{content}'")
            return "preference", content
            
        # List/show commands
        if re.match(r'^(?:list|show)\s+(?:my\s+)?facts', lowered):
            logger.debug(f"Detected list facts command")
            return "list_facts", None
            
        if re.match(r'^(?:list|show)\s+(?:my\s+)?preferences', lowered):
            logger.debug(f"Detected list preferences command")
            return "list_preferences", None
            
        if re.match(r'^(?:list|show)\s+(?:my\s+)?projects', lowered):
            logger.debug(f"Detected list projects command")
            return "list_projects", None
        
        # Delete commands - FIX: Added "known" as an optional word
        if re.match(r'^(?:remove|delete)\s+(?:my\s+)?(?:known\s+)?fact', lowered):
            content = re.sub(r'^(?:remove|delete)\s+(?:my\s+)?(?:known\s+)?fact\s+(?:about\s+)?', '', text, flags=re.IGNORECASE).strip()
            logger.info(f"Detected delete fact command with content: '{content}'")
            return "delete_fact", content
            
        if re.match(r'^(?:remove|delete)\s+(?:my\s+)?preference', lowered):
            content = re.sub(r'^(?:remove|delete)\s+(?:my\s+)?preference\s+(?:about\s+)?', '', text, flags=re.IGNORECASE).strip()
            logger.debug(f"Detected delete preference command: '{content}'")
            return "delete_preference", content
            
        if re.match(r'^(?:remove|delete)\s+(?:my\s+)?project', lowered):
            content = re.sub(r'^(?:remove|delete)\s+(?:my\s+)?project\s+(?:about\s+)?', '', text, flags=re.IGNORECASE).strip()
            logger.debug(f"Detected delete project command: '{content}'")
            return "delete_project", content
        
        # Remove "remember that" or similar prefixes to get the core content for pattern matching
        core_content = text
        remember_prefixes = [
            r"^remember\s+that\s+",
            r"^remember\s+",
            r"^note\s+that\s+",
            r"^note\s+",
            r"^keep in mind\s+that\s+",
            r"^keep in mind\s+"
        ]
        for prefix in remember_prefixes:
            core_content = re.sub(prefix, "", core_content, flags=re.IGNORECASE)
        
        # Location patterns - check these before other patterns
        # These need to check both the original text and the content after "remember"
        location_patterns = [
            (r"\bi (?:work|am working)(?:\s+in|\s+at|\s+from|\s+remotely\s+from|\s+remotely\s+in)\s+(.*?)(?:\.|\s*$)", "work_location"),
            (r"\bi (?:live|reside|am living|am from|was born in|moved to)\s+(.*?)(?:\.|\s*$)", "home_location")
        ]
        
        for pattern, location_type in location_patterns:
            # Check in both original and clean content
            for check_text in [text, core_content]:
                match = re.search(pattern, check_text.lower())
                if match:
                    location_text = match.group(1).strip()
                    # Clean up location text
                    location_text = re.sub(r"^(?:remotely\s+from\s+|from\s+)", "", location_text)
                    return location_type, location_text
        
        # TODO command has highest priority among remaining types
        if re.search(r"^todo:", lowered, re.IGNORECASE) or re.search(r"\btodo:", lowered, re.IGNORECASE):
            todo_match = re.search(r"todo:(.*)", lowered, re.IGNORECASE)
            todo_text = todo_match.group(1).strip() if todo_match else text
            return "todo", todo_text
        
        # Enhanced project-related patterns
        if "my new project is" in lowered:
            project_text = re.sub(r".*?my new project is\s+", "", text, flags=re.IGNORECASE)
            return "project_replace", project_text
        
        if re.search(r"\badd project\b", lowered):
            project_text = re.sub(r".*?\badd project\b\s+", "", text, flags=re.IGNORECASE)
            return "project_add", project_text
        
        # If we made it this far and it's a "remember" statement, it's a known fact
        remember_patterns = [
            r"^remember\b",
            r"^note\b",
            r"^keep in mind\b"
        ]
        
        for pattern in remember_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "known_fact", core_content
        
        # Enhanced preference patterns
        preference_patterns = [
            r"\bi prefer\b",
            r"\bi (?:like|love|enjoy|hate|dislike)\b",
            r"\bmy preference is\b"
        ]
        
        for pattern in preference_patterns:
            if re.search(pattern, lowered):
                # Extract the preference content
                match = re.search(r"(?:prefer|like|love|enjoy|hate|dislike|preference is)\s+(.+)", lowered)
                if match:
                    return "preference", match.group(1).strip()
                return "preference", core_content
        
        # Simple statements that could be added as facts
        fact_patterns = [
            r"^i\s+(?:am|have|like|prefer|want|need)\s+(.+)"
        ]
        
        for pattern in fact_patterns:
            match = re.search(pattern, lowered)
            if match:
                return "known_fact", match.group(0)
        
        # No pattern matched
        logger.debug(f"No memory instruction pattern matched for: {text}")
        return "unknown", None

    def get_known_facts(self, slack_user_id: str) -> List[str]:
        """
        Get all known facts for a user.
        
        Args:
            slack_user_id: The Slack user ID
            
        Returns:
            List of known facts as strings
        """
        page_id = self.notion_service.get_user_page_id(slack_user_id)
        if not page_id:
            return []
        
        facts = []
        section_block = self._find_section_block(page_id, SectionType.KNOWN_FACTS.value)
        if not section_block:
            return []
        
        try:
            children_response = self.notion_service.client.blocks.children.list(
                block_id=section_block.get("id")
            )
            
            for block in children_response.get("results", []):
                if block.get("type") == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    facts.append(text_content)
            
            return facts
        except Exception as e:
            logger.error(f"Error getting known facts for user {slack_user_id}: {e}", exc_info=True)
            return []

    def get_preferences(self, slack_user_id: str) -> List[str]:
        """
        Get all preferences for a user.
        
        Args:
            slack_user_id: The Slack user ID
            
        Returns:
            List of preferences as strings
        """
        page_id = self.notion_service.get_user_page_id(slack_user_id)
        if not page_id:
            return []
        
        logger.info(f"Getting preferences for user {slack_user_id}, page ID: {page_id}")
        
        preferences = []
        
        try:
            # Get all blocks on the page first
            all_blocks = self.notion_service.client.blocks.children.list(block_id=page_id)
            blocks = all_blocks.get("results", [])
            
            # Find the Preferences section and collect all bulleted list items that follow it
            in_preferences_section = False
            
            for block in blocks:
                block_type = block.get("type")
                
                # Check if this is a heading (potentially the Preferences section)
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    heading_text = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    # Found Preferences section - now we'll collect items until the next section
                    if heading_text == "Preferences":
                        in_preferences_section = True
                        logger.debug(f"Found Preferences section")
                        continue
                    # If we find another heading and we were in Preferences section, exit the section
                    elif in_preferences_section:
                        in_preferences_section = False
                        logger.debug(f"Exiting Preferences section at heading: {heading_text}")
                
                # If we're in the Preferences section, collect bulleted list items
                elif in_preferences_section and block_type == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if text_content:
                        preferences.append(text_content)
                        logger.debug(f"Found preference: {text_content}")
                
            logger.info(f"Retrieved {len(preferences)} preferences for user {slack_user_id}")
            
            # If we didn't find any preferences using the section method, try a direct search
            if not preferences:
                logger.warning(f"No preferences found in section. Trying direct bullet scan.")
                preference_section_block = self._find_section_block(page_id, "Preferences")
                if preference_section_block:
                    # Find bullets after the preferences heading
                    section_index = -1
                    for i, block in enumerate(blocks):
                        if block.get("id") == preference_section_block.get("id"):
                            section_index = i
                            break
                    
                    if section_index >= 0:
                        # Look at the blocks after the section heading
                        for i in range(section_index + 1, len(blocks)):
                            block = blocks[i]
                            if block.get("type") == "heading_2":  # Next section
                                break
                            
                            if block.get("type") == "bulleted_list_item":
                                text_content = ""
                                rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                                
                                for text_item in rich_text:
                                    text_content += text_item.get("plain_text", "")
                                
                                if text_content:
                                    preferences.append(text_content)
                                    logger.debug(f"Found preference (direct scan): {text_content}")
            
            return preferences
        except Exception as e:
            logger.error(f"Error getting preferences for user {slack_user_id}: {e}", exc_info=True)
            return []
    
    def get_projects(self, slack_user_id: str) -> List[str]:
        """
        Get all projects for a user.
        
        Args:
            slack_user_id: The Slack user ID
            
        Returns:
            List of projects as strings
        """
        page_id = self.notion_service.get_user_page_id(slack_user_id)
        if not page_id:
            return []
        
        projects = []
        section_block = self._find_section_block(page_id, SectionType.PROJECTS.value)
        if not section_block:
            return []
        
        try:
            children_response = self.notion_service.client.blocks.children.list(
                block_id=section_block.get("id")
            )
            
            for block in children_response.get("results", []):
                if block.get("type") == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    projects.append(text_content)
            
            return projects
        except Exception as e:
            logger.error(f"Error getting projects for user {slack_user_id}: {e}", exc_info=True)
            return []

    def delete_known_fact(self, slack_user_id: str, fact_fragment: str) -> bool:
        """
        Delete a known fact containing the given fragment.
        
        Args:
            slack_user_id: The Slack user ID
            fact_fragment: A text fragment to identify the fact
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Attempting to delete fact containing '{fact_fragment}' for user {slack_user_id}")
        
        page_id = self.notion_service.get_user_page_id(slack_user_id)
        if not page_id:
            logger.error(f"No Notion page found for user {slack_user_id}")
            return False
        
        logger.debug(f"Found user page ID: {page_id}")
        
        # Try to find the Known Facts section
        section_block = self._find_section_block(page_id, SectionType.KNOWN_FACTS.value)
        if not section_block:
            logger.error(f"No 'Known Facts' section found in page {page_id}")
            return False
        
        logger.debug(f"Found 'Known Facts' section ID: {section_block.get('id')}")
        
        try:
            # Get all the blocks in the Known Facts section
            children_response = self.notion_service.client.blocks.children.list(
                block_id=section_block.get("id")
            )
            
            results = children_response.get("results", [])
            logger.debug(f"Found {len(results)} items in 'Known Facts' section")
            
            # Log all facts for debugging
            found_facts = []
            for block in results:
                if block.get("type") == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if text_content:
                        found_facts.append(text_content)
            
            logger.info(f"All facts in section: {found_facts}")
            
            # Track if we've successfully deleted anything
            successful_deletion = False
            deleted_facts = []
            
            # First look directly in the section block's children
            for block in results:
                if block.get("type") == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    logger.debug(f"Checking fact item: '{text_content}'")
                    
                    # Check if this fact contains our search fragment
                    if fact_fragment.lower() in text_content.lower():
                        logger.info(f"Found matching fact: '{text_content}' contains '{fact_fragment}'")
                        
                        try:
                            self.notion_service.client.blocks.delete(block_id=block.get("id"))
                            logger.info(f"Successfully deleted block with ID {block.get('id')}")
                            successful_deletion = True
                            deleted_facts.append(text_content)
                            
                        except Exception as delete_e:
                            logger.error(f"Error deleting block {block.get('id')}: {delete_e}")
            
            # If we haven't found anything yet, look more broadly in the page
            if not successful_deletion:
                logger.info(f"No facts found in section children. Checking the entire page...")
                
                # Get all blocks in the page
                all_blocks_response = self.notion_service.client.blocks.children.list(block_id=page_id)
                all_blocks = all_blocks_response.get("results", [])
                
                # Find the Known Facts section to know where to stop
                facts_section_idx = -1
                next_section_idx = len(all_blocks)
                
                for i, block in enumerate(all_blocks):
                    block_type = block.get("type")
                    if block_type in ["heading_1", "heading_2", "heading_3"]:
                        text_content = ""
                        rich_text = block.get(block_type, {}).get("rich_text", [])
                        
                        for text_item in rich_text:
                            text_content += text_item.get("plain_text", "")
                        
                        if "Known Facts" in text_content:
                            facts_section_idx = i
                        elif facts_section_idx >= 0 and i > facts_section_idx:
                            # This is the next section after Known Facts
                            next_section_idx = i
                            break
                
                # If we found the Known Facts section
                if facts_section_idx >= 0:
                    logger.info(f"Found Known Facts at index {facts_section_idx}, next section at {next_section_idx}")
                    
                    # Look at all blocks between Facts section and next section
                    for i in range(facts_section_idx + 1, next_section_idx):
                        block = all_blocks[i]
                        
                        if block.get("type") == "bulleted_list_item":
                            text_content = ""
                            rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                            
                            for text_item in rich_text:
                                text_content += text_item.get("plain_text", "")
                            
                            logger.debug(f"Checking fact in page: '{text_content}'")
                            
                            if fact_fragment.lower() in text_content.lower():
                                logger.info(f"Found matching fact in page: '{text_content}'")
                                
                                try:
                                    self.notion_service.client.blocks.delete(block_id=block.get("id"))
                                    logger.info(f"Successfully deleted block with ID {block.get('id')}")
                                    successful_deletion = True
                                    deleted_facts.append(text_content)
                                    
                                except Exception as delete_e:
                                    logger.error(f"Error deleting block {block.get('id')}: {delete_e}")
            
            # Invalidate cache if we deleted anything
            if successful_deletion:
                self.notion_service.invalidate_user_cache(slack_user_id)
                logger.info(f"Successfully deleted facts: {deleted_facts}")
            
            return successful_deletion
            
        except Exception as e:
            logger.error(f"Error searching for facts to delete: {e}", exc_info=True)
            return False
    
    def delete_preference(self, slack_user_id: str, preference_fragment: str) -> bool:
        """
        Delete a preference containing the given fragment.
        
        Args:
            slack_user_id: The Slack user ID
            preference_fragment: A text fragment to identify the preference
            
        Returns:
            True if successful, False otherwise
        """
        page_id = self.notion_service.get_user_page_id(slack_user_id)
        if not page_id:
            return False
        
        section_block = self._find_section_block(page_id, SectionType.PREFERENCES.value)
        if not section_block:
            return False
        
        try:
            children_response = self.notion_service.client.blocks.children.list(
                block_id=section_block.get("id")
            )
            
            for block in children_response.get("results", []):
                if block.get("type") == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if preference_fragment.lower() in text_content.lower():
                        self.notion_service.client.blocks.delete(block_id=block.get("id"))
                        
                        # Invalidate cache
                        self.notion_service.invalidate_user_cache(slack_user_id)
                        
                        return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting preference for user {slack_user_id}: {e}", exc_info=True)
            return False

    def delete_project(self, slack_user_id: str, project_fragment: str) -> bool:
        """
        Delete a project containing the given fragment.
        
        Args:
            slack_user_id: The Slack user ID
            project_fragment: A text fragment to identify the project
            
        Returns:
            True if successful, False otherwise
        """
        page_id = self.notion_service.get_user_page_id(slack_user_id)
        if not page_id:
            return False
        
        section_block = self._find_section_block(page_id, SectionType.PROJECTS.value)
        if not section_block:
            return False
        
        try:
            children_response = self.notion_service.client.blocks.children.list(
                block_id=section_block.get("id")
            )
            
            for block in children_response.get("results", []):
                if block.get("type") == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if project_fragment.lower() in text_content.lower():
                        self.notion_service.client.blocks.delete(block_id=block.get("id"))
                        
                        # Invalidate cache
                        self.notion_service.invalidate_user_cache(slack_user_id)
                        
                        return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting project for user {slack_user_id}: {e}", exc_info=True)
            return False
    
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
            
            # 3. Update the property in the database
            properties_update = {
                property_name: {"rich_text": [{"type": "text", "text": {"content": location}}]}
            }
            self.notion_service.client.pages.update(page_id=page_id, properties=properties_update)
            
            # 4. Find or create the Location Information section
            location_section_block = self.ensure_section_exists(page_id, SectionType.LOCATION_INFO.value)
            if not location_section_block:
                logger.error(f"Failed to create Location Information section for user {slack_user_id}")
                return False
            
            # 5. Prepare the location text for display
            location_label = "Work Location:" if location_type == "work" else "Home Location:"
            location_bullet = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": f"{location_label} [{property_name} property] {location}"}
                    }]
                }
            }
            
            # 6. Find existing location bullets
            existing_location_bullets = []
            try:
                # Get all blocks on the page
                page_blocks = self.notion_service.client.blocks.children.list(block_id=page_id)
                
                # Find the Location Information section to get its position
                section_index = -1
                for i, block in enumerate(page_blocks.get("results", [])):
                    if block.get("id") == location_section_block.get("id"):
                        section_index = i
                        break
                
                if section_index >= 0:
                    # Look for location bullets after the section header
                    for i in range(section_index + 1, len(page_blocks.get("results", []))):
                        block = page_blocks.get("results")[i]
                        if block.get("type") == "heading_2":  # Found next section
                            break
                        
                        if block.get("type") == "bulleted_list_item":
                            text_content = ""
                            rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                            for text_item in rich_text:
                                text_content += text_item.get("plain_text", "")
                            
                            # Check if this is our location type
                            current_label = "Work Location:" if location_type == "work" else "Home Location:"
                            if current_label in text_content:
                                # Found a matching location bullet
                                existing_location_bullets.append((i, block))
            except Exception as e:
                logger.warning(f"Error finding existing location bullets: {e}")
            
            # 7. Delete existing location bullets for this type
            for _, block in existing_location_bullets:
                try:
                    self.notion_service.client.blocks.delete(block_id=block.get("id"))
                except Exception as e:
                    logger.warning(f"Error deleting existing location bullet: {e}")
            
            # 8. Add the new location bullet after the section heading
            try:
                # Insert after the section heading
                self.notion_service.client.blocks.children.append(
                    block_id=page_id,  # Append to the page
                    children=[location_bullet],
                    after=location_section_block.get("id")  # After the section heading
                )
            except Exception as e:
                logger.warning(f"Error adding location bullet after section: {e}")
                
                # Fallback - try adding at the page level without positioning
                try:
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[location_bullet]
                    )
                except Exception as e2:
                    logger.error(f"Error adding location bullet to page: {e2}")
                    # The property was still updated, so don't return False here
            
            # Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            # Return success if we at least updated the property
            return True
        
        except Exception as e:
            logger.error(f"Error updating location for user {slack_user_id}: {e}", exc_info=True)
            return False
    
    def add_known_fact(self, slack_user_id: str, fact_text: str) -> bool:
        """
        Add a fact to the Known Facts section.
        
        Args:
            slack_user_id: The Slack user ID
            fact_text: The fact to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or create user page
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
            
            # Ensure the Known Facts section exists
            known_facts_section = self.ensure_section_exists(page_id, SectionType.KNOWN_FACTS.value)
            if not known_facts_section:
                logger.error(f"Failed to create Known Facts section for user {slack_user_id}")
                return False
                
            # Create a new bullet for the fact
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
            
            # Try to insert the fact after the heading
            try:
                page_blocks = self.notion_service.client.blocks.children.list(block_id=page_id)
                insert_after_id = None
                
                for i, block in enumerate(page_blocks.get("results", [])):
                    if block.get("id") == known_facts_section.get("id"):
                        insert_after_id = block.get("id")
                        break
                
                if insert_after_id:
                    # Insert directly after the heading in the page
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_fact_block],
                        after=insert_after_id
                    )
                else:
                    # Fallback if we can't find the section
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_fact_block]
                    )
            except Exception as e:
                logger.warning(f"Failed to append fact after section: {e}")
                
                # Fallback - try adding directly to the page without positioning
                try:
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_fact_block]
                    )
                except Exception as e2:
                    logger.error(f"Failed to add fact to page: {e2}")
                    return False
            
            # Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            return True
        except Exception as e:
            logger.error(f"Error adding known fact for user {slack_user_id}: {e}", exc_info=True)
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
            
            # 2. Ensure the Preferences section exists
            preferences_section_block = self.ensure_section_exists(page_id, SectionType.PREFERENCES.value)
            if not preferences_section_block:
                logger.error(f"Failed to create Preferences section for user {slack_user_id}")
                return False
            
            # 3. Create the new preference bullet point
            new_preference_block = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": preference}
                    }]
                }
            }
            
            # 4. Try to insert the preference after the section heading
            try:
                page_blocks = self.notion_service.client.blocks.children.list(block_id=page_id)
                insert_after_id = None
                
                for i, block in enumerate(page_blocks.get("results", [])):
                    if block.get("id") == preferences_section_block.get("id"):
                        insert_after_id = block.get("id")
                        break
                
                if insert_after_id:
                    # Insert directly after the heading in the page
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,  # Insert at page level, not heading level
                        children=[new_preference_block],
                        after=insert_after_id  # Insert after the heading
                    )
                else:
                    # Fallback if we can't find the section
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_preference_block]
                    )
            except Exception as e:
                logger.warning(f"Failed to append preference after section: {e}")
                
                # Fallback - try adding directly to the page without positioning
                try:
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_preference_block]
                    )
                except Exception as e2:
                    logger.error(f"Failed to add preference to page: {e2}")
                    return False
            
            # 5. Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding preference for {slack_user_id}: {e}", exc_info=True)
            return False
    
    def replace_projects(self, slack_user_id: str, project_name: str) -> bool:
        """
        Replace the projects section with a new project.
        
        Args:
            slack_user_id: The Slack user ID
            project_name: The name of the new project
            
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
            
            # If no Projects section exists, create it
            if not projects_section_block:
                # Create a new Projects section with a header and bullet point
                new_blocks = [
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": SectionType.PROJECTS.value}
                            }]
                        }
                    },
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": project_name}
                            }]
                        }
                    }
                ]
                
                # Add the new section directly to the page
                self.notion_service.client.blocks.children.append(
                    block_id=page_id,
                    children=new_blocks
                )
                
                # Invalidate cache
                self.notion_service.invalidate_user_cache(slack_user_id)
                
                return True
            
            # 3. If the Projects section exists, get its children
            try:
                children_response = self.notion_service.client.blocks.children.list(
                    block_id=projects_section_block.get("id")
                )
                
                # 4. Delete all existing project bullet points
                for block in children_response.get("results", []):
                    if block.get("type") == "bulleted_list_item":
                        self.notion_service.client.blocks.delete(block_id=block.get("id"))
            except Exception as e:
                # If we can't get children of the section (which is the issue from the error),
                # we need a different approach
                logger.warning(f"Cannot get children of Projects section: {e}")
                
                # Try to delete the section and recreate it
                try:
                    self.notion_service.client.blocks.delete(block_id=projects_section_block.get("id"))
                    
                    # Add new section after deletion
                    new_blocks = [
                        {
                            "object": "block",
                            "type": "heading_2",
                            "heading_2": {
                                "rich_text": [{
                                    "type": "text",
                                    "text": {"content": SectionType.PROJECTS.value}
                                }]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{
                                    "type": "text",
                                    "text": {"content": project_name}
                                }]
                            }
                        }
                    ]
                    
                    # Add the new section directly to the page
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=new_blocks
                    )
                    
                    # Invalidate cache
                    self.notion_service.invalidate_user_cache(slack_user_id)
                    
                    return True
                except Exception as delete_error:
                    logger.error(f"Failed to delete and recreate Projects section: {delete_error}")
                    return False
            
            # 5. Add the new project as a bullet point directly after the heading
            # First find where to insert the new bullet
            try:
                page_blocks = self.notion_service.client.blocks.children.list(block_id=page_id)
                insert_after_id = None
                
                for i, block in enumerate(page_blocks.get("results", [])):
                    if block.get("id") == projects_section_block.get("id"):
                        insert_after_id = block.get("id")
                        break
                
                if insert_after_id:
                    # Insert the new bullet point directly after the heading in the page
                    new_bullet = {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": project_name}
                            }]
                        }
                    }
                    
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,  # Insert directly in the page, not the heading
                        children=[new_bullet],
                        after=insert_after_id  # Insert after the heading
                    )
                else:
                    # Fallback if we can't find the section in the page blocks
                    logger.warning("Couldn't locate projects section in page blocks")
                    
                    # Add new bullet directly to the page
                    new_bullet = {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": project_name}
                            }]
                        }
                    }
                    
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_bullet]
                    )
            except Exception as e:
                logger.error(f"Failed to add new project bullet: {e}")
                return False
            
            # Invalidate cache
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            return True
        except Exception as e:
            logger.error(f"Error replacing projects for user {slack_user_id}: {e}", exc_info=True)
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
    
    def ensure_section_exists(self, page_id: str, section_name: str) -> Dict[str, Any]:
        """
        Ensure a section exists, creating it if necessary.
        
        Args:
            page_id: The page ID
            section_name: The name of the section
            
        Returns:
            The section block
        """
        # First check if the section exists
        section_block = self._find_section_block(page_id, section_name)
        if section_block:
            return section_block
        
        # Create a new section
        try:
            # Add a new heading for the section
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
            
            # Return the newly created section
            if result.get("results"):
                return result["results"][0]
            
            # If we can't get the result directly, try to find it again
            return self._find_section_block(page_id, section_name)
        except Exception as e:
            logger.error(f"Error creating section {section_name}: {e}", exc_info=True)
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
    
    def get_help_text(self) -> str:
        """
        Get help text explaining how to use memory commands.
        
        Returns:
            Formatted help text
        """
        help_text = """
    Here's how you can manage your information with me:

    **Adding Information:**
    - "Remember I like coffee" - Add to Known Facts
    - "Fact: I prefer tea" - Add to Known Facts
    - "I work in New York" - Add location information
    - "My new project is Website Redesign" - Replace all projects
    - "Project: Mobile App Development" - Add a project
    - "I prefer short answers" - Add to Preferences
    - "Preference: Use bullet points" - Add to Preferences
    - "TODO: Call John tomorrow" - Add a TODO item

    **Viewing Information:**
    - "Show my facts" - List all Known Facts
    - "List my preferences" - List all Preferences
    - "Show my projects" - List all Projects

    **Deleting Information:**
    - "Delete fact about coffee" - Remove a fact
    - "Remove preference about bullet points" - Remove a preference
    - "Delete project Website Redesign" - Remove a project

    I'll remember this information and use it to personalize our conversations.
    """
        return help_text
    
    def get_example_for_command(self, attempted_command: str) -> str:
        """
        Get a helpful example based on what the user attempted to do.
        
        Args:
            attempted_command: The command the user attempted
            
        Returns:
            Example text
        """
        lowered = attempted_command.lower()
        
        # Project examples
        if "project" in lowered:
            return "Here are some examples:\n• \"Project: Mobile App Development\"\n• \"My new project is Website Redesign\"\n• \"Add project Database Migration\""
        
        # Fact examples
        if any(word in lowered for word in ["fact", "remember", "know"]):
            return "Here are some examples:\n• \"Remember I drink coffee\"\n• \"Fact: I have a cat named Max\"\n• \"I live in New York\""
        
        # Preference examples
        if any(word in lowered for word in ["prefer", "like", "don't like", "hate"]):
            return "Here are some examples:\n• \"I prefer short answers\"\n• \"Preference: Use bullet points\"\n• \"I like technical explanations\""
        
        # TODO examples
        if "todo" in lowered:
            return "Here are some examples:\n• \"TODO: Call John tomorrow\"\n• \"TODO: Finish report by Friday\""
        
        # Deletion examples
        if any(word in lowered for word in ["delete", "remove"]):
            return "Here are some examples:\n• \"Delete fact about coffee\"\n• \"Remove preference about bullet points\"\n• \"Delete project Website Redesign\""
        
        # List examples
        if any(word in lowered for word in ["list", "show", "view"]):
            return "Here are some examples:\n• \"Show my facts\"\n• \"List my preferences\"\n• \"Show my projects\""
        
        # Default example
        return "Try saying \"help\" to see all available commands."