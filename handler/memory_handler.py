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
    """Enhanced memory handler for structured user profiles with language preference support."""
    
    def __init__(self, notion_service):
        """Initialize with a reference to the Notion service."""
        self.notion_service = notion_service
    
    def handle_memory_instruction(self, slack_user_id: str, text: str) -> Optional[str]:
        """
        Process a memory instruction and store it appropriately with enhanced error handling.
        
        Args:
            slack_user_id: The Slack user ID
            text: The message content
        
        Returns:
            A user-friendly message indicating success or failure, or None if not a memory instruction
        """
        try:
            if not self._is_likely_memory_command(text):
                logger.debug(f"Text doesn't appear to be a memory command: '{text}'")
                return None
            
            # Classify the memory instruction and get cleaned content
            memory_type, cleaned_content = self.classify_memory_instruction(text)
            
            # If not a memory instruction, return None
            if memory_type == "unknown" or (cleaned_content is None and not memory_type.startswith("list_")):
                logger.debug(f"Not a memory instruction: '{text}'")
                return None
            
            logger.info(f"Processing memory instruction - Type: {memory_type}, Content: '{cleaned_content}', User: {slack_user_id}")
            
            # Handle language preference instructions
            if memory_type == "language_preference":
                logger.info(f"Processing language preference: '{cleaned_content}'")
                success = self.set_language_preference(slack_user_id, cleaned_content)
                
                if success:
                    # Verify the language was stored
                    stored_language = self.get_language_preference(slack_user_id)
                    if stored_language and stored_language.lower() == cleaned_content.lower():
                        logger.info(f"✅ Language preference successfully stored for user {slack_user_id}")
                        return f"✅ I've set your language preference to {cleaned_content}. I will now respond in {cleaned_content}."
                    else:
                        logger.error(f"❌ Language preference verification failed for user {slack_user_id}")
                        return f"❌ I tried to set your language preference, but verification failed. Please try again."
                else:
                    logger.error(f"❌ Failed to store language preference for user {slack_user_id}")
                    return "❌ Sorry, I couldn't store your language preference right now. Please try again later."
            
            # Handle different types of memory instructions (existing code)
            elif memory_type == "known_fact":
                logger.info(f"Processing known fact: '{cleaned_content}'")

                # Check for and remove similar existing facts before adding new one
                try:
                    existing_facts = self.get_known_facts(slack_user_id)
                    for existing_fact in existing_facts:
                        if self._facts_are_similar(cleaned_content, existing_fact):
                            logger.info(f"Found similar fact to replace: '{existing_fact}' → '{cleaned_content}'")
                            # Extract a key word to use for deletion (first non-common word)
                            words = existing_fact.lower().split()
                            key_word = next((word for word in words if word not in ['my', 'i', 'am', 'is', 'the', 'a', 'an']), words[0] if words else "fact")
                            
                            delete_success = self.delete_known_fact(slack_user_id, key_word)
                            if delete_success:
                                logger.info(f"Successfully removed similar fact: '{existing_fact}'")
                            else:
                                logger.warning(f"Failed to remove similar fact: '{existing_fact}'")
                            break  # Only replace one similar fact
                            
                except Exception as e:
                    logger.warning(f"Error checking for similar facts: {e}")
                    # Continue anyway - we'll just add the new fact
                    
                success = self.add_known_fact(slack_user_id, cleaned_content)
                
                if success:
                    # ENHANCED: Double verification with retry
                    verification_attempts = 3
                    verification_success = False
                    
                    for attempt in range(verification_attempts):
                        logger.info(f"Verification attempt {attempt + 1}/{verification_attempts}")
                        
                        if self.verify_fact_stored(slack_user_id, cleaned_content):
                            verification_success = True
                            break
                        else:
                            logger.warning(f"Verification attempt {attempt + 1} failed, waiting before retry...")
                            import time
                            time.sleep(2)  # Wait before retry
                    
                    if verification_success:
                        logger.info(f"✅ Fact successfully stored and verified for user {slack_user_id}")
                        return f"✅ Added to your Known Facts: \"{cleaned_content}\""
                    else:
                        logger.error(f"❌ Fact storage verification failed after {verification_attempts} attempts")
                        
                        # Try to diagnose the issue
                        diagnostic_info = self._diagnose_storage_issue(slack_user_id, cleaned_content)
                        logger.error(f"Storage diagnostic: {diagnostic_info}")
                        
                        return f"❌ I tried to store that fact, but couldn't verify it was saved properly. Please try again. (Debug: {diagnostic_info})"
                else:
                    logger.error(f"❌ Failed to store fact for user {slack_user_id}")
                    return "❌ Sorry, I couldn't store that fact right now. Please try again later."
            
            # List commands with enhanced error handling
            elif memory_type == "list_facts":
                logger.info(f"Processing list facts request for user {slack_user_id}")
                
                try:
                    facts = self.get_known_facts(slack_user_id)
                    
                    if not facts:
                        # Try direct read as fallback
                        facts = self._get_facts_direct_from_notion(slack_user_id)
                        
                    if facts:
                        facts_list = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])
                        return f"Here are your stored facts:\n\n{facts_list}\n\n📝 Retrieved from Notion database"
                    else:
                        return "You don't have any facts stored yet. You can add facts by saying \"known fact: [your fact]\" or \"remember [your fact]\"."
                        
                except Exception as list_error:
                    logger.error(f"Error listing facts: {list_error}", exc_info=True)
                    return "❌ I had trouble retrieving your facts from Notion. Please try again later."
            
            elif memory_type == "list_preferences":
                logger.info(f"Processing list preferences request for user {slack_user_id}")
                
                try:
                    preferences = self.get_preferences(slack_user_id)
                    
                    if preferences:
                        preferences_list = "\n".join([f"{i+1}. {pref}" for i, pref in enumerate(preferences)])
                        return f"Here are your stored preferences:\n\n{preferences_list}\n\n📝 Retrieved from Notion database"
                    else:
                        return "You don't have any preferences stored yet. You can add preferences by saying \"preference: [your preference]\" or \"I prefer [your preference]\"."
                        
                except Exception as list_error:
                    logger.error(f"Error listing preferences: {list_error}", exc_info=True)
                    return "❌ I had trouble retrieving your preferences from Notion. Please try again later."
            
            elif memory_type == "list_projects":
                logger.info(f"Processing list projects request for user {slack_user_id}")
                
                try:
                    projects = self.get_projects(slack_user_id)
                    
                    if projects:
                        projects_list = "\n".join([f"{i+1}. {proj}" for i, proj in enumerate(projects)])
                        return f"Here are your stored projects:\n\n{projects_list}\n\n📝 Retrieved from Notion database"
                    else:
                        return "You don't have any projects stored yet. You can add projects by saying \"project: [your project]\" or \"add project [your project]\"."
                        
                except Exception as list_error:
                    logger.error(f"Error listing projects: {list_error}", exc_info=True)
                    return "❌ I had trouble retrieving your projects from Notion. Please try again later."
            
            # Handle other memory types (preferences, projects, etc.) - keeping existing logic
            elif memory_type == "preference":
                success = self.add_preference(slack_user_id, cleaned_content)
                if success:
                    # ENHANCED: Double verification with retry
                    verification_attempts = 3
                    verification_success = False
                    
                    for attempt in range(verification_attempts):
                        logger.info(f"Verification attempt {attempt + 1}/{verification_attempts}")
                        
                        if self.verify_preference_stored(slack_user_id, cleaned_content):
                            verification_success = True
                            break
                        else:
                            logger.warning(f"Verification attempt {attempt + 1} failed, waiting before retry...")
                            import time
                            time.sleep(2)
                    
                    if verification_success:
                        logger.info(f"✅ Preference successfully stored and verified for user {slack_user_id}")
                        return f"✅ Added to your Preferences: \"{cleaned_content}\""
                    else:
                        logger.error(f"❌ Preference storage verification failed after {verification_attempts} attempts")
                        return f"❌ I tried to store that preference, but couldn't verify it was saved properly. Please try again."
                else:
                    logger.error(f"❌ Failed to store preference for user {slack_user_id}")
                    return "❌ Sorry, I couldn't store that preference right now. Please try again later."
    
            elif memory_type == "delete_fact":
                # Get current facts for better error reporting
                before_facts = self.get_known_facts(slack_user_id)
                success = self.delete_known_fact(slack_user_id, cleaned_content)
                
                if success:
                    # Verify deletion worked
                    after_facts = self.get_known_facts(slack_user_id)
                    
                    if len(after_facts) < len(before_facts):
                        return f"✅ Successfully removed fact about \"{cleaned_content}\" from your Known Facts."
                    else:
                        return f"❌ Deletion verification failed. The fact may still exist."
                else:
                    return f"❓ I couldn't find any facts containing \"{cleaned_content}\" to remove."
            
            elif memory_type == "project_add":
                logger.info(f"Processing project add command: '{cleaned_content}'")
                success = self.add_project(slack_user_id, cleaned_content)
                
                if success:
                    # ENHANCED: Double verification with retry
                    verification_attempts = 3
                    verification_success = False
                    
                    for attempt in range(verification_attempts):
                        logger.info(f"Verification attempt {attempt + 1}/{verification_attempts}")
                        
                        if self.verify_project_stored(slack_user_id, cleaned_content):
                            verification_success = True
                            break
                        else:
                            logger.warning(f"Verification attempt {attempt + 1} failed, waiting before retry...")
                            import time
                            time.sleep(2)
                    
                    if verification_success:
                        logger.info(f"✅ Project successfully stored and verified for user {slack_user_id}")
                        return f"✅ Added to your Projects: \"{cleaned_content}\""
                    else:
                        logger.error(f"❌ Project storage verification failed after {verification_attempts} attempts")
                        return f"❌ I tried to store that project, but couldn't verify it was saved properly. Please try again."
                else:
                    logger.error(f"❌ Failed to store project for user {slack_user_id}")
                    return "❌ Sorry, I couldn't save that project right now. Please try again later."
    
            elif memory_type == "project_replace":
                logger.info(f"Processing project replace command: '{cleaned_content}'")
                success = self.replace_projects(slack_user_id, cleaned_content)
                
                if success:
                    return f"✅ Replaced your projects with: \"{cleaned_content}\""
                else:
                    return "❌ Sorry, I couldn't update your projects right now. Please try again later."

            elif memory_type == "delete_preference":
                logger.info(f"Processing delete preference command: '{cleaned_content}'")
                success = self.delete_preference(slack_user_id, cleaned_content)
                
                if success:
                    return f"✅ Successfully removed preference about \"{cleaned_content}\" from your preferences."
                else:
                    return f"❓ I couldn't find any preferences containing \"{cleaned_content}\" to remove."

            elif memory_type == "delete_project":
                logger.info(f"Processing delete project command: '{cleaned_content}'")
                success = self.delete_project(slack_user_id, cleaned_content)
                
                if success:
                    return f"✅ Successfully removed project about \"{cleaned_content}\" from your projects."
                else:
                    return f"❓ I couldn't find any projects containing \"{cleaned_content}\" to remove."

            elif memory_type == "todo":
                logger.info(f"Processing TODO command: '{cleaned_content}'")
                success = self.add_todo(slack_user_id, cleaned_content)
                
                if success:
                    return f"✅ Added to your TODO list: \"{cleaned_content}\""
                else:
                    return "❌ Sorry, I couldn't add that TODO item right now. Please try again later."

            elif memory_type == "work_location":
                # Clean up prepositions from the location
                clean_location = re.sub(r'^(?:in|at|from)\s+', '', cleaned_content.strip())
                logger.info(f"Processing work location command: '{clean_location}'")
                
                # Update ONLY the database property, not Known Facts
                success = self._update_location_property(slack_user_id, "WorkLocation", clean_location)
                
                if success:
                    return f"✅ Updated your work location to: {clean_location}"
                else:
                    return "❌ Sorry, I couldn't update your work location right now. Please try again later."

            elif memory_type == "home_location":
                # Clean up prepositions from the location  
                clean_location = re.sub(r'^(?:in|at|from)\s+', '', cleaned_content.strip())
                logger.info(f"Processing home location command: '{clean_location}'")
                
                # Update ONLY the database property, not Known Facts
                success = self._update_location_property(slack_user_id, "HomeLocation", clean_location)
                
                if success:
                    return f"✅ Updated your home location to: {clean_location}"
                else:
                    return "❌ Sorry, I couldn't update your home location right now. Please try again later."
            
            logger.warning(f"Unhandled memory instruction type: {memory_type}")
            return None
            
        except Exception as e:
            logger.error(f"Error in handle_memory_instruction: {e}", exc_info=True)
            return f"❌ I encountered an error while processing that instruction. Please try again. (Error: {str(e)})"

    def _is_likely_memory_command(self, text: str) -> bool:
        """Quick check if text is likely a memory command before full processing."""
        text_lower = text.lower()
        
        # Explicit memory keywords
        explicit_keywords = [
            "remember", "something to remember", "fact:", "preference:", "project:", "todo:", 
            "my preference is", "i prefer", "store that", "note that",
            "remove", "delete", "add project", "new project", "delete project"
        ]
        
        # If it contains explicit memory keywords, it's likely a memory command
        if any(keyword in text_lower for keyword in explicit_keywords):
            return True
        
        language_preference_patterns = [
            r"(?:always\s+)?(?:respond|answer|communicate|talk|speak|reply)\s+(?:to me\s+)?(?:in\s+)\w+",
            r"(?:my\s+)?(?:preferred\s+)?language\s+(?:is\s+|preference\s+is\s+)\w+",
            r"(?:set\s+)?(?:my\s+)?language\s+(?:preference\s+)?(?:to\s+)\w+",
            r"(?:use\s+|switch\s+to\s+)\w+(?:\s+language)?",
        ]
        
        for pattern in language_preference_patterns:
            if re.search(pattern, text_lower):
                logger.debug(f"Detected language preference pattern in: '{text}'")
                return True

        location_patterns = [
            r"\bi (?:work|am working)(?:\s+in|\s+at|\s+from|\s+remotely\s+from|\s+remotely\s+in)\s+",
            r"\bi (?:live|reside|am living|am from|was born in|moved to)\s+"
        ]
        
        for pattern in location_patterns:
            if re.search(pattern, text_lower):
                logger.debug(f"Detected location pattern in: '{text}'")
                return True
        
        # If it looks like a summarization request, it's NOT a memory command
        summarization_indicators = [
            "summarize", "summary", "tldr", "brief", "recap", "overview"
        ]
        
        if any(indicator in text_lower for indicator in summarization_indicators):
            return False
        
        # Default to not a memory command for ambiguous cases
        return False

    def _diagnose_storage_issue(self, slack_user_id: str, fact_text: str) -> str:
        """
        Diagnose why a fact storage might have failed.
        
        Args:
            slack_user_id: The Slack user ID
            fact_text: The fact that failed to store
            
        Returns:
            Diagnostic information string
        """
        try:
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            
            if not page_id:
                return "No Notion page found for user"
            
            # Check if page is accessible
            try:
                page_info = self.notion_service.client.pages.retrieve(page_id=page_id)
                if not page_info:
                    return "Page not accessible"
            except Exception as e:
                return f"Page access error: {str(e)}"
            
            # Check if Known Facts section exists
            try:
                blocks = self.notion_service.client.blocks.children.list(block_id=page_id)
                has_known_facts = any(
                    "Known Facts" in str(block) for block in blocks.get("results", [])
                )
                
                if not has_known_facts:
                    return "No Known Facts section found"
                else:
                    return "Known Facts section exists but fact not found"
                    
            except Exception as e:
                return f"Block access error: {str(e)}"
            
        except Exception as e:
            return f"Diagnostic error: {str(e)}"
    
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
        
        # ENHANCED: Check for language preference commands first (highest priority) - RESTRICTIVE
            language_patterns = [
                r"(?:preference:|my preference is|preference is|i prefer|prefer)\s+(?:to\s+)?(?:always\s+)?(?:respond|answer|communicate|talk|speak|reply)\s+(?:in\s+|to me in\s+)(\w+)",
                r"(?:always\s+)?(?:respond|answer|communicate|talk|speak|reply)\s+(?:to me\s+)?(?:in\s+)(\w+)",
                r"(?:my\s+)?(?:preferred\s+)?language\s+(?:is\s+|preference\s+is\s+)(\w+)",
                r"(?:set\s+)?(?:my\s+)?language\s+(?:preference\s+)?(?:to\s+)(\w+)",
                r"(?:use\s+|switch\s+to\s+)(\w+)\s+language",  # Only if explicitly followed by "language"
                r"(?:use\s+|switch\s+to\s+)(\w+)\s+(?:when\s+)?(?:responding|answering)",  # Only if followed by responding/answering
                r"(?:i\s+)?(?:want|need|would like)\s+(?:you\s+to\s+)?(?:respond|answer|communicate|talk|speak|reply)\s+(?:to me\s+)?(?:in\s+)(\w+)"
            ]

            # Known languages list
            known_languages = ["english", "spanish", "french", "german", "italian", "portuguese", "chinese", "japanese", "korean", "russian", "arabic", "hindi", "dutch", "swedish", "norwegian", "finnish"]

            # Additional validation: reject common non-language words
            non_language_words = ["bullet", "points", "format", "style", "verse", "rhyme", "short", "long", "brief", "detailed"]

            for pattern in language_patterns:
                match = re.search(pattern, lowered)
                if match:
                    potential_language = match.group(1).strip().title()  # Capitalize first letter
                    
                    # Only consider it a language if it's a known language name AND not a style word
                    if (potential_language.lower() in known_languages and 
                        potential_language.lower() not in non_language_words):
                        logger.info(f"Detected language preference command: '{potential_language}'")
                        return "language_preference", potential_language
                    else:
                        logger.debug(f"Rejected potential language '{potential_language}' - not a valid language or is a style word")
                        # Continue to check other patterns instead of returning
                        continue

            # If we get here, no valid language preference was detected in the early patterns
        
        # Direct command checking - these have priority over other patterns
        if (lowered.startswith("fact:") or lowered.startswith("fact ") or 
            lowered.startswith("known fact:") or lowered.startswith("known fact ")):
            # Handle both "fact:" and "known fact:" patterns
            content = re.sub(r'^(?:known\s+)?fact[:\s]\s*', '', text, flags=re.IGNORECASE).strip()
            logger.info(f"Detected fact command: '{content}'")
            return "known_fact", content
            
        if (lowered.startswith("project:") or lowered.startswith("project ") or
            lowered.startswith("add project") or lowered.startswith("new project:")):
            content = re.sub(r'^(?:add\s+|new\s+)?project[:\s]\s*', '', text, flags=re.IGNORECASE).strip()
            logger.info(f"Detected project command: '{content}'")
            return "project_add", content

        # MODIFIED: Exclude language preferences from regular preferences
        if (lowered.startswith("preference:") or lowered.startswith("preference ") or
            lowered.startswith("my preference:") or lowered.startswith("my preference ")):
            content = re.sub(r'^(?:my\s+)?preference[:\s]\s*', '', text, flags=re.IGNORECASE).strip()
            
            # Check if this is actually a language preference that we missed
            # But be MUCH more specific - only if it explicitly mentions responding IN a language
            content_lower = content.lower()
            
            # Only detect as language preference if it explicitly mentions responding/answering IN a specific language
            language_indicators = [
                r"(?:respond|answer|communicate|talk|speak|reply).*?(?:in\s+)(\w+)",
                r"(?:always\s+)?(?:respond|answer|communicate|talk|speak|reply)\s+(?:to me\s+)?(?:in\s+)(\w+)",
                r"(?:use\s+|switch\s+to\s+)(\w+)\s+language",  # Only if explicitly followed by "language"
                r"(?:use\s+|switch\s+to\s+)(\w+)\s+(?:when\s+)?(?:responding|answering)",  # Only if followed by responding/answering
            ]

            is_language_pref = False
            extracted_language = None

            # Known languages list
            known_languages = ["english", "spanish", "french", "german", "italian", "portuguese", "chinese", "japanese", "korean", "russian", "arabic", "hindi", "dutch", "swedish", "norwegian", "finnish"]

            for pattern in language_indicators:
                match = re.search(pattern, content_lower)
                if match:
                    potential_language = match.group(1).strip()
                    # Only consider it a language if it's a known language name AND the context is clearly about language
                    if potential_language.lower() in known_languages:
                        is_language_pref = True
                        extracted_language = potential_language.title()
                        logger.info(f"Detected valid language preference: '{extracted_language}' from pattern: {pattern}")
                        break
                    else:
                        logger.debug(f"Rejected potential language '{potential_language}' - not in known languages list")

            # Additional validation: reject common non-language words even if they match patterns
            non_language_words = ["bullet", "points", "format", "style", "verse", "rhyme", "short", "long", "brief", "detailed"]
            if is_language_pref and extracted_language and extracted_language.lower() in non_language_words:
                logger.info(f"Rejected language preference '{extracted_language}' - appears to be a style preference, not language")
                is_language_pref = False
                extracted_language = None
            
            if is_language_pref and extracted_language:
                logger.info(f"Detected language preference in preference command: '{extracted_language}'")
                return "language_preference", extracted_language
            
            logger.info(f"Detected preference command: '{content}'")
            return "preference", content

        if re.match(r'^(?:list|show|display)\s+(?:my\s+)?(?:known\s+)?facts?$', lowered):
            logger.info(f"Detected list facts command")
            return "list_facts", None
            
        if re.match(r'^(?:list|show)\s+(?:my\s+)?preferences', lowered):
            logger.debug(f"Detected list preferences command")
            return "list_preferences", None
            
        if re.match(r'^(?:list|show)\s+(?:my\s+)?projects', lowered):
            logger.debug(f"Detected list projects command")
            return "list_projects", None
        
        # Delete commands
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
        
        # Enhanced preference patterns (but exclude language preferences)
        preference_patterns = [
            r"\bi prefer\b",
            r"\bi (?:like|love|enjoy|hate|dislike)\b",
            r"\bmy preference is\b"
        ]
        
        for pattern in preference_patterns:
            if re.search(pattern, lowered):
                # Check if this is actually a language preference
                if any(word in lowered for word in ["respond", "answer", "communicate", "talk", "speak", "reply", "language"]):
                    # This is likely a language preference, try to extract it
                    language_match = re.search(r"(?:respond|answer|communicate|talk|speak|reply).*?(?:in\s+)(\w+)", lowered)
                    if language_match:
                        return "language_preference", language_match.group(1).title()
                
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

    def set_language_preference(self, slack_user_id: str, language: str) -> bool:
        """
        Set the user's language preference in the Notion database.
        
        Args:
            slack_user_id: The Slack user ID
            language: The preferred language (e.g., "Italian", "Spanish")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Setting language preference for user {slack_user_id}: {language}")
            
            # Get or create user page
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id, None)
                if not success:
                    logger.error(f"Failed to create user page for {slack_user_id}")
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    logger.error(f"Failed to get page ID after creation for {slack_user_id}")
                    return False
            
            # Update the LanguagePreference property
            properties_to_update = {
                PropertyType.LANGUAGE_PREFERENCE.value: {
                    "rich_text": [{"type": "text", "text": {"content": language}}]
                }
            }
            
            # Update the page properties
            result = self.notion_service.client.pages.update(
                page_id=page_id,
                properties=properties_to_update
            )
            
            if result:
                # Clear cache to ensure fresh reads
                self.notion_service.invalidate_user_cache(slack_user_id)
                logger.info(f"Successfully set language preference to {language} for user {slack_user_id}")
                return True
            else:
                logger.error(f"Failed to update language preference for user {slack_user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting language preference for user {slack_user_id}: {e}", exc_info=True)
            return False

    def get_language_preference(self, slack_user_id: str) -> Optional[str]:
        """
        Get the user's language preference from the Notion database.
        
        Args:
            slack_user_id: The Slack user ID
            
        Returns:
            The language preference if found, None otherwise
        """
        try:
            # Get user properties (this is cached)
            properties = self.notion_service.get_user_page_properties(slack_user_id)
            if not properties:
                return None
            
            # Extract language preference from properties
            language_pref_prop = properties.get(PropertyType.LANGUAGE_PREFERENCE.value)
            
            if language_pref_prop and language_pref_prop.get("type") == "rich_text":
                rich_text_array = language_pref_prop.get("rich_text", [])
                if rich_text_array and len(rich_text_array) > 0:
                    language = rich_text_array[0].get("plain_text", "").strip()
                    if language:
                        logger.debug(f"Retrieved language preference for user {slack_user_id}: {language}")
                        return language
            
            logger.debug(f"No language preference found for user {slack_user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting language preference for user {slack_user_id}: {e}", exc_info=True)
            return None
        
    def verify_fact_stored(self, slack_user_id: str, fact_text: str) -> bool:
        """
        Verify that a fact was actually stored in Notion with enhanced reliability.
        
        Args:
            slack_user_id: The Slack user ID
            fact_text: The fact text to verify
            
        Returns:
            True if the fact is found in Notion, False otherwise
        """
        try:
            logger.info(f"Verifying fact storage for user {slack_user_id}: '{fact_text}'")
            
            # Clear cache to force fresh read from Notion
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            # Wait a moment for Notion's eventual consistency
            import time
            time.sleep(1)
            
            # Get fresh facts from Notion using multiple methods
            facts_from_get_method = self.get_known_facts(slack_user_id)
            facts_from_direct_read = self._get_facts_direct_from_notion(slack_user_id)
            
            logger.info(f"Facts from get_method ({len(facts_from_get_method)}): {facts_from_get_method}")
            logger.info(f"Facts from direct_read ({len(facts_from_direct_read)}): {facts_from_direct_read}")
            
            # Check both methods for the fact
            fact_lower = fact_text.lower().strip()
            
            all_facts = set(facts_from_get_method + facts_from_direct_read)
            
            for stored_fact in all_facts:
                stored_fact_lower = stored_fact.lower().strip()
                
                # More flexible matching
                if (fact_lower in stored_fact_lower or 
                    stored_fact_lower in fact_lower or
                    self._facts_are_similar(fact_lower, stored_fact_lower)):
                    
                    logger.info(f"✅ Fact verification successful: '{fact_text}' found as '{stored_fact}'")
                    return True
            
            logger.warning(f"❌ Fact verification failed: '{fact_text}' not found")
            logger.warning(f"Available facts: {list(all_facts)}")
            return False
            
        except Exception as e:
            logger.error(f"Error verifying fact storage: {e}", exc_info=True)
            return False

    def _get_facts_direct_from_notion(self, slack_user_id: str) -> List[str]:
        """
        Get facts directly from Notion without using the cache, as a verification method.
        
        Args:
            slack_user_id: The Slack user ID
            
        Returns:
            List of facts found directly in Notion
        """
        try:
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                logger.warning(f"No page ID found for user {slack_user_id}")
                return []
            
            # Get all blocks directly from Notion API
            all_blocks = self.notion_service.client.blocks.children.list(
                block_id=page_id,
                page_size=100  # Get more blocks at once
            )
            blocks = all_blocks.get("results", [])
            
            logger.info(f"Direct API call returned {len(blocks)} blocks")
            
            facts = []
            in_known_facts_section = False
            
            for i, block in enumerate(blocks):
                block_type = block.get("type")
                block_id = block.get("id")
                
                logger.debug(f"Block {i}: type={block_type}, id={block_id}")
                
                # Check for section headings
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    heading_text = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    if heading_text == "Known Facts":
                        in_known_facts_section = True
                        logger.debug(f"Entered Known Facts section at block {i}")
                        continue
                    elif in_known_facts_section and heading_text:
                        in_known_facts_section = False
                        logger.debug(f"Exited Known Facts section at block {i}")
                
                # Collect facts from the Known Facts section
                elif in_known_facts_section and block_type == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if text_content:
                        facts.append(text_content)
                        logger.debug(f"Found fact in direct read: '{text_content}'")
            
            logger.info(f"Direct read found {len(facts)} facts")
            return facts
            
        except Exception as e:
            logger.error(f"Error in direct facts read: {e}", exc_info=True)
            return []

    def _get_preferences_direct_from_notion(self, slack_user_id: str) -> List[str]:
        """
        Get preferences directly from Notion without using the cache, as a verification method.
        """
        try:
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                logger.warning(f"No page ID found for user {slack_user_id}")
                return []
            
            # Get all blocks directly from Notion API
            all_blocks = self.notion_service.client.blocks.children.list(
                block_id=page_id,
                page_size=100
            )
            blocks = all_blocks.get("results", [])
            
            logger.info(f"Direct API call returned {len(blocks)} blocks")
            
            preferences = []
            in_preferences_section = False
            
            for i, block in enumerate(blocks):
                block_type = block.get("type")
                block_id = block.get("id")
                
                logger.debug(f"Block {i}: type={block_type}, id={block_id}")
                
                # Check for section headings
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    heading_text = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    if heading_text == "Preferences":
                        in_preferences_section = True
                        logger.debug(f"Entered Preferences section at block {i}")
                        continue
                    elif in_preferences_section and heading_text:
                        in_preferences_section = False
                        logger.debug(f"Exited Preferences section at block {i}")
                
                # Collect preferences from the Preferences section
                elif in_preferences_section and block_type == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if text_content:
                        preferences.append(text_content)
                        logger.debug(f"Found preference in direct read: '{text_content}'")
            
            logger.info(f"Direct read found {len(preferences)} preferences")
            return preferences
            
        except Exception as e:
            logger.error(f"Error in direct preferences read: {e}", exc_info=True)
            return []

    def _get_projects_direct_from_notion(self, slack_user_id: str) -> List[str]:
        """
        Get projects directly from Notion without using the cache, as a verification method.
        """
        try:
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                logger.warning(f"No page ID found for user {slack_user_id}")
                return []
            
            # Get all blocks directly from Notion API
            all_blocks = self.notion_service.client.blocks.children.list(
                block_id=page_id,
                page_size=100
            )
            blocks = all_blocks.get("results", [])
            
            logger.info(f"Direct API call returned {len(blocks)} blocks")
            
            projects = []
            in_projects_section = False
            
            for i, block in enumerate(blocks):
                block_type = block.get("type")
                block_id = block.get("id")
                
                logger.debug(f"Block {i}: type={block_type}, id={block_id}")
                
                # Check for section headings
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    heading_text = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    if heading_text == "Projects":
                        in_projects_section = True
                        logger.debug(f"Entered Projects section at block {i}")
                        continue
                    elif in_projects_section and heading_text:
                        in_projects_section = False
                        logger.debug(f"Exited Projects section at block {i}")
                
                # Collect projects from the Projects section
                elif in_projects_section and block_type == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if text_content:
                        projects.append(text_content)
                        logger.debug(f"Found project in direct read: '{text_content}'")
            
            logger.info(f"Direct read found {len(projects)} projects")
            return projects
            
        except Exception as e:
            logger.error(f"Error in direct projects read: {e}", exc_info=True)
            return []

    def _facts_are_similar(self, fact1: str, fact2: str) -> bool:
        """
        Check if two facts are similar enough to be considered the same.
        
        Args:
            fact1: First fact (normalized)
            fact2: Second fact (normalized)
            
        Returns:
            True if facts are similar, False otherwise
        """
        # Remove common variations
        def normalize_fact(fact: str) -> str:
            fact = fact.lower().strip()
            # Remove articles and common prefixes
            fact = re.sub(r'^(?:the|a|an|my|your|his|her|their)\s+', '', fact)
            # Remove punctuation
            fact = re.sub(r'[.,!?;:]', '', fact)
            return fact
        
        norm1 = normalize_fact(fact1)
        norm2 = normalize_fact(fact2)
        
        # Check for substantial overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1.intersection(words2))
            min_length = min(len(words1), len(words2))
            
            # If more than 70% of words overlap, consider them similar
            return overlap / min_length > 0.7
        
        return False

    def _preferences_are_similar(self, pref1: str, pref2: str) -> bool:
        """
        Check if two preferences are similar enough to be considered the same.
        """
        # Remove common variations
        def normalize_preference(pref: str) -> str:
            pref = pref.lower().strip()
            # Remove articles and common prefixes
            pref = re.sub(r'^(?:i\s+prefer\s+|my\s+preference\s+is\s+|preference\s+is\s+)', '', pref)
            # Remove punctuation
            pref = re.sub(r'[.,!?;:]', '', pref)
            return pref
        
        norm1 = normalize_preference(pref1)
        norm2 = normalize_preference(pref2)
        
        # Check for substantial overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1.intersection(words2))
            min_length = min(len(words1), len(words2))
            
            # If more than 70% of words overlap, consider them similar
            return overlap / min_length > 0.7
        
        return False

    def _projects_are_similar(self, proj1: str, proj2: str) -> bool:
        """
        Check if two projects are similar enough to be considered the same.
        """
        # Remove common variations
        def normalize_project(proj: str) -> str:
            proj = proj.lower().strip()
            # Remove articles and common prefixes
            proj = re.sub(r'^(?:the\s+|a\s+|an\s+|my\s+|our\s+)', '', proj)
            # Remove punctuation
            proj = re.sub(r'[.,!?;:]', '', proj)
            return proj
        
        norm1 = normalize_project(proj1)
        norm2 = normalize_project(proj2)
        
        # Check for substantial overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1.intersection(words2))
            min_length = min(len(words1), len(words2))
            
            # If more than 70% of words overlap, consider them similar
            return overlap / min_length > 0.7
        
        return False

    def verify_preference_stored(self, slack_user_id: str, preference_text: str) -> bool:
        """
        Verify that a preference was actually stored in Notion with enhanced reliability.
        """
        try:
            logger.info(f"Verifying preference storage for user {slack_user_id}: '{preference_text}'")
            
            # Clear cache to force fresh read from Notion
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            # Wait a moment for Notion's eventual consistency
            import time
            time.sleep(1)
            
            # Get fresh preferences from Notion using multiple methods
            preferences_from_get_method = self.get_preferences(slack_user_id)
            preferences_from_direct_read = self._get_preferences_direct_from_notion(slack_user_id)
            
            logger.info(f"Preferences from get_method ({len(preferences_from_get_method)}): {preferences_from_get_method}")
            logger.info(f"Preferences from direct_read ({len(preferences_from_direct_read)}): {preferences_from_direct_read}")
            
            # Check both methods for the preference
            pref_lower = preference_text.lower().strip()
            
            all_preferences = set(preferences_from_get_method + preferences_from_direct_read)
            
            for stored_pref in all_preferences:
                stored_pref_lower = stored_pref.lower().strip()
                
                # More flexible matching
                if (pref_lower in stored_pref_lower or 
                    stored_pref_lower in pref_lower or
                    self._preferences_are_similar(pref_lower, stored_pref_lower)):
                    
                    logger.info(f"✅ Preference verification successful: '{preference_text}' found as '{stored_pref}'")
                    return True
            
            logger.warning(f"❌ Preference verification failed: '{preference_text}' not found")
            logger.warning(f"Available preferences: {list(all_preferences)}")
            return False
            
        except Exception as e:
            logger.error(f"Error verifying preference storage: {e}", exc_info=True)
            return False
    
    def verify_project_stored(self, slack_user_id: str, project_text: str) -> bool:
        """
        Verify that a project was actually stored in Notion with enhanced reliability.
        """
        try:
            logger.info(f"Verifying project storage for user {slack_user_id}: '{project_text}'")
            
            # Clear cache to force fresh read from Notion
            self.notion_service.invalidate_user_cache(slack_user_id)
            
            # Wait a moment for Notion's eventual consistency
            import time
            time.sleep(1)
            
            # Get fresh projects from Notion using multiple methods
            projects_from_get_method = self.get_projects(slack_user_id)
            projects_from_direct_read = self._get_projects_direct_from_notion(slack_user_id)
            
            logger.info(f"Projects from get_method ({len(projects_from_get_method)}): {projects_from_get_method}")
            logger.info(f"Projects from direct_read ({len(projects_from_direct_read)}): {projects_from_direct_read}")
            
            # Check both methods for the project
            project_lower = project_text.lower().strip()
            
            all_projects = set(projects_from_get_method + projects_from_direct_read)
            
            for stored_project in all_projects:
                stored_project_lower = stored_project.lower().strip()
                
                # More flexible matching
                if (project_lower in stored_project_lower or 
                    stored_project_lower in project_lower or
                    self._projects_are_similar(project_lower, stored_project_lower)):
                    
                    logger.info(f"✅ Project verification successful: '{project_text}' found as '{stored_project}'")
                    return True
            
            logger.warning(f"❌ Project verification failed: '{project_text}' not found")
            logger.warning(f"Available projects: {list(all_projects)}")
            return False
            
        except Exception as e:
            logger.error(f"Error verifying project storage: {e}", exc_info=True)
            return False

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
        
        try:
            # Get all blocks on the page first
            all_blocks = self.notion_service.client.blocks.children.list(block_id=page_id)
            blocks = all_blocks.get("results", [])
            
            # Find the Known Facts section and collect all bulleted list items that follow it
            in_known_facts_section = False
            
            for block in blocks:
                block_type = block.get("type")
                
                # Check if this is a heading (potentially the Known Facts section)
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    heading_text = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    # Found Known Facts section - now we'll collect items until the next section
                    if heading_text == "Known Facts":
                        in_known_facts_section = True
                        logger.debug(f"Found Known Facts section")
                        continue
                    # If we find another heading and we were in Known Facts section, exit the section
                    elif in_known_facts_section:
                        in_known_facts_section = False
                        logger.debug(f"Exiting Known Facts section at heading: {heading_text}")
                
                # If we're in the Known Facts section, collect bulleted list items
                elif in_known_facts_section and block_type == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if text_content:
                        facts.append(text_content)
                        logger.debug(f"Found fact: {text_content}")
            
            logger.info(f"Retrieved {len(facts)} facts for user {slack_user_id}")
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
        
        try:
            # Get all blocks in the page to find Known Facts section
            all_blocks_response = self.notion_service.client.blocks.children.list(block_id=page_id)
            all_blocks = all_blocks_response.get("results", [])
            
            # Find the Known Facts section and items under it
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
            
            if facts_section_idx < 0:
                logger.error(f"No 'Known Facts' section found in page {page_id}")
                return False
            
            logger.info(f"Found Known Facts at index {facts_section_idx}, next section at {next_section_idx}")
            
            # Track deletions
            successful_deletion = False
            deleted_facts = []
            
            # Look at all blocks between Facts section and next section
            for i in range(facts_section_idx + 1, next_section_idx):
                block = all_blocks[i]
                
                if block.get("type") == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    logger.debug(f"Checking fact: '{text_content}'")
                    
                    if fact_fragment.lower() in text_content.lower():
                        logger.info(f"Found matching fact: '{text_content}'")
                        
                        try:
                            self.notion_service.client.blocks.delete(block_id=block.get("id"))
                            logger.info(f"Successfully deleted block with ID {block.get('id')}")
                            successful_deletion = True
                            deleted_facts.append(text_content)
                            
                        except Exception as delete_e:
                            logger.error(f"Error deleting block {block.get('id')}: {delete_e}")
                            return False
            
            # FIXED: Only invalidate cache after successful deletion AND verification
            if successful_deletion:
                # Verify the deletion actually worked
                updated_facts = self.get_known_facts(slack_user_id)
                deletion_verified = True
                for deleted_fact in deleted_facts:
                    if any(deleted_fact.lower() in existing_fact.lower() for existing_fact in updated_facts):
                        logger.error(f"Deletion verification failed: '{deleted_fact}' still exists")
                        deletion_verified = False
                
                if deletion_verified:
                    self.notion_service.invalidate_user_cache(slack_user_id)
                    logger.info(f"Successfully deleted and verified facts: {deleted_facts}")
                    return True
                else:
                    logger.error(f"Deletion verification failed, not invalidating cache")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error searching for facts to delete: {e}", exc_info=True)
            return False
    
    def delete_preference(self, slack_user_id: str, preference_fragment: str) -> bool:
        """
        Delete a preference containing the given fragment.
        """
        logger.info(f"Attempting to delete preference containing '{preference_fragment}' for user {slack_user_id}")
        
        page_id = self.notion_service.get_user_page_id(slack_user_id)
        if not page_id:
            logger.error(f"No Notion page found for user {slack_user_id}")
            return False
        
        logger.debug(f"Found user page ID: {page_id}")
        
        try:
            # Get all blocks in the page to find Preferences section
            all_blocks_response = self.notion_service.client.blocks.children.list(block_id=page_id)
            all_blocks = all_blocks_response.get("results", [])
            
            # Find the Preferences section and items under it
            preferences_section_idx = -1
            next_section_idx = len(all_blocks)
            
            for i, block in enumerate(all_blocks):
                block_type = block.get("type")
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    text_content = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if "Preferences" in text_content:
                        preferences_section_idx = i
                    elif preferences_section_idx >= 0 and i > preferences_section_idx:
                        # This is the next section after Preferences
                        next_section_idx = i
                        break
            
            if preferences_section_idx < 0:
                logger.error(f"No 'Preferences' section found in page {page_id}")
                return False
            
            logger.info(f"Found Preferences at index {preferences_section_idx}, next section at {next_section_idx}")
            
            # Track deletions
            successful_deletion = False
            deleted_preferences = []
            
            # Look at all blocks between Preferences section and next section
            for i in range(preferences_section_idx + 1, next_section_idx):
                block = all_blocks[i]
                
                if block.get("type") == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    logger.debug(f"Checking preference: '{text_content}'")
                    
                    if preference_fragment.lower() in text_content.lower():
                        logger.info(f"Found matching preference: '{text_content}'")
                        
                        try:
                            self.notion_service.client.blocks.delete(block_id=block.get("id"))
                            logger.info(f"Successfully deleted block with ID {block.get('id')}")
                            successful_deletion = True
                            deleted_preferences.append(text_content)
                            
                        except Exception as delete_e:
                            logger.error(f"Error deleting block {block.get('id')}: {delete_e}")
                            return False
            
            # Only invalidate cache after successful deletion AND verification
            if successful_deletion:
                # Verify the deletion actually worked
                updated_preferences = self.get_preferences(slack_user_id)
                deletion_verified = True
                for deleted_pref in deleted_preferences:
                    if any(deleted_pref.lower() in existing_pref.lower() for existing_pref in updated_preferences):
                        logger.error(f"Deletion verification failed: '{deleted_pref}' still exists")
                        deletion_verified = False
                
                if deletion_verified:
                    self.notion_service.invalidate_user_cache(slack_user_id)
                    logger.info(f"Successfully deleted and verified preferences: {deleted_preferences}")
                    return True
                else:
                    logger.error(f"Deletion verification failed, not invalidating cache")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error searching for preferences to delete: {e}", exc_info=True)
            return False

    def delete_project(self, slack_user_id: str, project_fragment: str) -> bool:
        """
        Delete a project containing the given fragment.
        """
        logger.info(f"Attempting to delete project containing '{project_fragment}' for user {slack_user_id}")
        
        page_id = self.notion_service.get_user_page_id(slack_user_id)
        if not page_id:
            logger.error(f"No Notion page found for user {slack_user_id}")
            return False
        
        logger.debug(f"Found user page ID: {page_id}")
        
        try:
            # Get all blocks in the page to find Projects section
            all_blocks_response = self.notion_service.client.blocks.children.list(block_id=page_id)
            all_blocks = all_blocks_response.get("results", [])
            
            # Find the Projects section and items under it
            projects_section_idx = -1
            next_section_idx = len(all_blocks)
            
            for i, block in enumerate(all_blocks):
                block_type = block.get("type")
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    text_content = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    if "Projects" in text_content:
                        projects_section_idx = i
                    elif projects_section_idx >= 0 and i > projects_section_idx:
                        # This is the next section after Projects
                        next_section_idx = i
                        break
            
            if projects_section_idx < 0:
                logger.error(f"No 'Projects' section found in page {page_id}")
                return False
            
            logger.info(f"Found Projects at index {projects_section_idx}, next section at {next_section_idx}")
            
            # Track deletions
            successful_deletion = False
            deleted_projects = []
            
            # Look at all blocks between Projects section and next section
            for i in range(projects_section_idx + 1, next_section_idx):
                block = all_blocks[i]
                
                if block.get("type") == "bulleted_list_item":
                    text_content = ""
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        text_content += text_item.get("plain_text", "")
                    
                    logger.debug(f"Checking project: '{text_content}'")
                    
                    if project_fragment.lower() in text_content.lower():
                        logger.info(f"Found matching project: '{text_content}'")
                        
                        try:
                            self.notion_service.client.blocks.delete(block_id=block.get("id"))
                            logger.info(f"Successfully deleted block with ID {block.get('id')}")
                            successful_deletion = True
                            deleted_projects.append(text_content)
                            
                        except Exception as delete_e:
                            logger.error(f"Error deleting block {block.get('id')}: {delete_e}")
                            return False
            
            # Only invalidate cache after successful deletion AND verification
            if successful_deletion:
                # Verify the deletion actually worked
                updated_projects = self.get_projects(slack_user_id)
                deletion_verified = True
                for deleted_proj in deleted_projects:
                    if any(deleted_proj.lower() in existing_proj.lower() for existing_proj in updated_projects):
                        logger.error(f"Deletion verification failed: '{deleted_proj}' still exists")
                        deletion_verified = False
                
                if deletion_verified:
                    self.notion_service.invalidate_user_cache(slack_user_id)
                    logger.info(f"Successfully deleted and verified projects: {deleted_projects}")
                    return True
                else:
                    logger.error(f"Deletion verification failed, not invalidating cache")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error searching for projects to delete: {e}", exc_info=True)
            return False

    def add_known_fact(self, slack_user_id: str, fact_text: str) -> bool:
        """
        Add a fact to the Known Facts section with improved error handling.
        
        Args:
            slack_user_id: The Slack user ID
            fact_text: The fact to add
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Adding known fact for user {slack_user_id}: '{fact_text}'")
        
        try:
            # Get or create user page
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id, None)
                if not success:
                    logger.error(f"Failed to create user page for {slack_user_id}")
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    logger.error(f"Failed to get page ID after creation for {slack_user_id}")
                    return False
            
            # Ensure the Known Facts section exists
            known_facts_section_id = self._ensure_known_facts_section_exists(page_id)
            if not known_facts_section_id:
                logger.error(f"Failed to create/find Known Facts section for user {slack_user_id}")
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
            
            # FIXED: More robust insertion logic
            try:
                # Strategy 1: Try to append directly after the Known Facts heading
                logger.info(f"Attempting to append fact block after Known Facts section")
                result = self.notion_service.client.blocks.children.append(
                    block_id=page_id,
                    children=[new_fact_block],
                    after=known_facts_section_id
                )
                
                if result and result.get("results"):
                    new_block_id = result["results"][0]["id"]
                    logger.info(f"Successfully added fact block with ID: {new_block_id}")
                    
                    # FIXED: Verify the block was actually created
                    verification_success = self._verify_block_exists(page_id, new_block_id)
                    if verification_success:
                        logger.info(f"Fact storage verified for user {slack_user_id}")
                        return True
                    else:
                        logger.error(f"Block verification failed for user {slack_user_id}")
                        return False
                else:
                    logger.error(f"Notion API returned unexpected result: {result}")
                    return False
                    
            except Exception as insert_error:
                logger.error(f"Failed to insert after Known Facts section: {insert_error}")
                
                # Strategy 2: Fallback - append to end of page
                try:
                    logger.info(f"Trying fallback: append to end of page")
                    result = self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_fact_block]
                    )
                    
                    if result and result.get("results"):
                        new_block_id = result["results"][0]["id"]
                        logger.info(f"Fallback successful: added fact block with ID: {new_block_id}")
                        return True
                    else:
                        logger.error(f"Fallback also failed: {result}")
                        return False
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback insertion also failed: {fallback_error}")
                    return False
                
        except Exception as e:
            logger.error(f"Error adding known fact for user {slack_user_id}: {e}", exc_info=True)
            return False
    
    def _ensure_known_facts_section_exists(self, page_id: str) -> Optional[str]:
        """
        Ensure the Known Facts section exists and return its block ID.
        
        Args:
            page_id: The Notion page ID
            
        Returns:
            The Known Facts section block ID if successful, None otherwise
        """
        try:
            # Get all blocks on the page
            blocks_response = self.notion_service.client.blocks.children.list(block_id=page_id)
            blocks = blocks_response.get("results", [])
            
            # Look for existing Known Facts section
            for block in blocks:
                block_type = block.get("type")
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    heading_text = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    if heading_text == "Known Facts":
                        logger.info(f"Found existing Known Facts section: {block.get('id')}")
                        return block.get("id")
            
            # Section doesn't exist, create it
            logger.info(f"Known Facts section not found, creating it")
            new_section_block = {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": "Known Facts"}
                    }]
                }
            }
            
            result = self.notion_service.client.blocks.children.append(
                block_id=page_id,
                children=[new_section_block]
            )
            
            if result and result.get("results"):
                section_id = result["results"][0]["id"]
                logger.info(f"Created Known Facts section with ID: {section_id}")
                return section_id
            else:
                logger.error(f"Failed to create Known Facts section: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error ensuring Known Facts section exists: {e}", exc_info=True)
            return None

    def _verify_block_exists(self, page_id: str, block_id: str) -> bool:
        """
        Verify that a block exists on the page.
        
        Args:
            page_id: The Notion page ID
            block_id: The block ID to verify
            
        Returns:
            True if the block exists, False otherwise
        """
        try:
            # Try to retrieve the specific block
            block = self.notion_service.client.blocks.retrieve(block_id=block_id)
            if block and block.get("id") == block_id:
                logger.debug(f"Block verification successful: {block_id}")
                return True
            else:
                logger.warning(f"Block verification failed: {block_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying block {block_id}: {e}")
            return False

    def add_preference(self, slack_user_id: str, preference: str) -> bool:
        """
        Add a preference to the Preferences section with improved error handling.
        
        Args:
            slack_user_id: The Slack user ID
            preference: The preference to add
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Adding preference for user {slack_user_id}: '{preference}'")
        
        try:
            # Get the page ID
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id, None)
                if not success:
                    logger.error(f"Failed to create user page for {slack_user_id}")
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    logger.error(f"Failed to get page ID after creation for {slack_user_id}")
                    return False
            
            # Ensure the Preferences section exists
            preferences_section_block = self.ensure_section_exists(page_id, SectionType.PREFERENCES.value)
            if not preferences_section_block:
                logger.error(f"Failed to create Preferences section for user {slack_user_id}")
                return False
            
            # Create the new preference bullet point
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
            
            # Insert the preference after the section heading
            try:
                page_blocks = self.notion_service.client.blocks.children.list(block_id=page_id)
                
                preferences_index = -1
                for i, block in enumerate(page_blocks.get("results", [])):
                    if block.get("id") == preferences_section_block.get("id"):
                        preferences_index = i
                        break
                
                if preferences_index >= 0:
                    # Insert after the heading
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_preference_block],
                        after=preferences_section_block.get("id")
                    )
                    logger.info(f"Successfully inserted preference after Preferences heading")
                else:
                    # Fallback
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_preference_block]
                    )
                    logger.warning(f"Used fallback insertion method for preference")
                
                logger.info(f"Successfully added preference to Notion for user {slack_user_id}")
                return True
                
            except Exception as insert_error:
                logger.error(f"Error inserting preference block: {insert_error}")
                
                # Fallback
                try:
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_preference_block]
                    )
                    logger.info(f"Used fallback insertion for preference")
                    return True
                except Exception as final_error:
                    logger.error(f"Fallback insertion failed: {final_error}")
                    return False
            
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
            # Get the page ID
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id, None)
                if not success:
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    return False
            
            # Find the Projects section
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
                
                # Only invalidate cache after successful operation
                self.notion_service.invalidate_user_cache(slack_user_id)
                
                return True
            
            # Delete existing project section and recreate
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
                
                # Only invalidate cache after successful operation
                self.notion_service.invalidate_user_cache(slack_user_id)
                
                return True
            except Exception as delete_error:
                logger.error(f"Failed to delete and recreate Projects section: {delete_error}")
                return False
            
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
            # Get the page ID
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new user page if it doesn't exist
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id, None)
                if not success:
                    return False
                
                # Get the new page ID
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    return False
            
            # Ensure the Projects section exists
            projects_section_block = self.ensure_section_exists(page_id, SectionType.PROJECTS.value)
            if not projects_section_block:
                logger.error(f"Failed to create Projects section for user {slack_user_id}")
                return False
            
            # Create the new project bullet point
            new_project_block = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": project}
                    }]
                }
            }
            
            # FIXED: Append to the page after the section heading, not to the heading itself
            try:
                self.notion_service.client.blocks.children.append(
                    block_id=page_id,  # Append to PAGE, not to the heading block
                    children=[new_project_block],
                    after=projects_section_block.get("id")  # Position after the heading
                )
                logger.info(f"Successfully added project to Notion for user {slack_user_id}")
                
                # Invalidate cache after successful operation
                self.notion_service.invalidate_user_cache(slack_user_id)
                return True
                
            except Exception as insert_error:
                logger.error(f"Error inserting project block: {insert_error}")
                
                # Fallback: append to end of page if positioning fails
                try:
                    self.notion_service.client.blocks.children.append(
                        block_id=page_id,
                        children=[new_project_block]
                    )
                    logger.info(f"Used fallback insertion for project")
                    self.notion_service.invalidate_user_cache(slack_user_id)
                    return True
                except Exception as fallback_error:
                    logger.error(f"Fallback insertion failed: {fallback_error}")
                    return False
            
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
    
    def _update_location_property(self, slack_user_id: str, property_name: str, location: str) -> bool:
        """
        Update only the database property for location, not Known Facts.
        
        Args:
            slack_user_id: The Slack user ID
            property_name: "WorkLocation" or "HomeLocation" 
            location: The clean location name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or create user page
            page_id = self.notion_service.get_user_page_id(slack_user_id)
            if not page_id:
                success = self.notion_service.store_user_nickname(slack_user_id, slack_user_id, None)
                if not success:
                    return False
                page_id = self.notion_service.get_user_page_id(slack_user_id)
                if not page_id:
                    return False
            
            # Update only the database property
            properties_update = {
                property_name: {"rich_text": [{"type": "text", "text": {"content": location}}]}
            }
            
            result = self.notion_service.client.pages.update(page_id=page_id, properties=properties_update)
            
            if result:
                # Invalidate cache
                self.notion_service.invalidate_user_cache(slack_user_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating {property_name} for user {slack_user_id}: {e}", exc_info=True)
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
    - "fact: I prefer tea" - Add to Known Facts
    - "Known Fact: I prefer tea" - Add to Known Facts
    - "known fact: I prefer tea" - Add to Known Facts
    - "I work in New York" - Add location information
    - "My new project is Website Redesign" - Replace all projects
    - "Project: Mobile App Development" - Add a project
    - "I prefer short answers" - Add to Preferences
    - "Preference: Use bullet points" - Add to Preferences
    - "TODO: Call John tomorrow" - Add a TODO item

    **Viewing Information:**
    - "Show my facts" - List all Known Facts
    - "Show my known facts" - List all Known Facts
    - "List my preferences" - List all Preferences
    - "Show my projects" - List all Projects

    **Deleting Information:**
    - "Delete fact about coffee" - Remove a fact
    - "Delete known fact about coffee" - Remove a fact
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