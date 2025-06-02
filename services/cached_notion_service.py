import time
import threading
import asyncio
from datetime import datetime
import logging
from loguru import logger
from notion_client import Client, AsyncClient, APIResponseError
from cachetools import TTLCache, cached
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from handler.memory_handler import MemoryHandler

class CachedNotionService:
    """
    Thread-safe cached wrapper for Notion API interactions.
    
    This service provides:
    - In-memory caching with TTL for Notion API calls
    - Thread safety with locking
    - Both synchronous and asynchronous methods
    - Automatic retry on transient failures
    """
    
    def __init__(
        self,
        notion_api_token: str,
        notion_user_db_id: str,
        cache_ttl: int = 300,  # 5 minutes default
        cache_max_size: int = 1000,
        enable_async: bool = True
    ):
        """
        Initialize the Cached Notion Service.
        
        Args:
            notion_api_token: Notion API token
            notion_user_db_id: ID of the user database in Notion
            cache_ttl: Cache time-to-live in seconds
            cache_max_size: Maximum number of items in the cache
            enable_async: Whether to initialize async client
        """
        self.api_token = notion_api_token
        self.user_db_id = notion_user_db_id
        self.cache_ttl = cache_ttl
        
        # Initialize clients
        self.client = Client(auth=notion_api_token) if notion_api_token else None
        self.async_client = AsyncClient(auth=notion_api_token) if notion_api_token and enable_async else None
        
        # Initialize the memory handler
        self.memory_handler = MemoryHandler(self)

        # Create caches with TTL
        self.cache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl)
        
        # Cache for page content (separate as it's larger)
        self.page_content_cache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl)
        
        # Thread locks for cache access
        self.cache_lock = threading.RLock()
        self.page_content_lock = threading.RLock()
        
        # Track cache stats
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "last_cleared": datetime.now().isoformat()
        }
        
        logger.info(
            f"CachedNotionService initialized with TTL={cache_ttl}s, "
            f"max_size={cache_max_size}"
        )

    def is_available(self) -> bool:
        """Check if the Notion service is available."""
        return self.client is not None and self.user_db_id is not None

    def clear_cache(self) -> None:
        """Clear all caches."""
        with self.cache_lock:
            self.cache.clear()
        
        with self.page_content_lock:
            self.page_content_cache.clear()
        
        self.cache_stats["last_cleared"] = datetime.now().isoformat()
        logger.info("Notion service caches cleared")

    def invalidate_user_cache(self, slack_user_id: str) -> None:
        """
        Invalidate cache entries for a specific user.
        
        Args:
            slack_user_id: The Slack user ID to invalidate cache for
        """
        user_cache_keys = [
            f"user_page_id_{slack_user_id}",
            f"user_properties_{slack_user_id}",
            f"user_name_{slack_user_id}",
            f"user_page_content_{slack_user_id}"
        ]
        
        with self.cache_lock:
            for key in user_cache_keys:
                if key in self.cache:
                    del self.cache[key]
        
        with self.page_content_lock:
            page_content_key = f"user_page_content_{slack_user_id}"
            if page_content_key in self.page_content_cache:
                del self.page_content_cache[page_content_key]
        
        logger.info(f"Cache invalidated for user {slack_user_id}")

    def get_user_page_id(self, slack_user_id: str) -> Optional[str]:
        """
        Find a user's Notion page ID by Slack User ID with caching.
        
        Args:
            slack_user_id: The Slack user ID to look up
            
        Returns:
            The Notion page ID if found, None otherwise
        """
        cache_key = f"user_page_id_{slack_user_id}"
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache_stats["hits"] += 1
                return self.cache[cache_key]
        
        if not self.is_available():
            return None
        
        try:
            # Query Notion for the user page
            response = self.client.databases.query(
                database_id=self.user_db_id,
                filter={
                    "property": "UserID",  # Must match property name in Notion
                    "title": {
                        "equals": slack_user_id
                    }
                }
            )
            
            page_id = response["results"][0]["id"] if response.get("results") else None
            
            # Cache the result (even if None)
            with self.cache_lock:
                self.cache[cache_key] = page_id
                self.cache_stats["misses"] += 1
                
            return page_id
            
        except Exception as e:
            logger.error(f"Error fetching Notion page ID for user {slack_user_id}: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def get_user_page_id_async(self, slack_user_id: str) -> Optional[str]:
        """
        Asynchronously find a user's Notion page ID with caching.
        
        Args:
            slack_user_id: The Slack user ID to look up
            
        Returns:
            The Notion page ID if found, None otherwise
        """
        cache_key = f"user_page_id_{slack_user_id}"
        
        # Check cache first (using sync lock)
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache_stats["hits"] += 1
                return self.cache[cache_key]
        
        if not self.async_client:
            return await asyncio.to_thread(self.get_user_page_id, slack_user_id)
        
        try:
            # Query Notion for the user page
            response = await self.async_client.databases.query(
                database_id=self.user_db_id,
                filter={
                    "property": "UserID",
                    "title": {
                        "equals": slack_user_id
                    }
                }
            )
            
            page_id = response["results"][0]["id"] if response.get("results") else None
            
            # Cache the result (even if None)
            with self.cache_lock:
                self.cache[cache_key] = page_id
                self.cache_stats["misses"] += 1
                
            return page_id
            
        except Exception as e:
            logger.error(f"Error fetching Notion page ID for user {slack_user_id}: {e}")
            self.cache_stats["errors"] += 1
            return None

    def get_user_page_properties(self, slack_user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve all properties of a user's Notion page with caching.
        
        Args:
            slack_user_id: The Slack user ID to look up
            
        Returns:
            Dictionary of page properties if found, None otherwise
        """
        cache_key = f"user_properties_{slack_user_id}"
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache_stats["hits"] += 1
                return self.cache[cache_key]
        
        if not self.is_available():
            return None
        
        # Get page ID (this is also cached)
        page_id = self.get_user_page_id(slack_user_id)
        if not page_id:
            return None
        
        try:
            # Query Notion for the page properties
            page_object = self.client.pages.retrieve(page_id=page_id)
            properties = page_object.get("properties")
            
            # Cache the result
            with self.cache_lock:
                self.cache[cache_key] = properties
                self.cache_stats["misses"] += 1
                
            return properties
            
        except Exception as e:
            logger.error(f"Error fetching Notion properties for user {slack_user_id}: {e}")
            self.cache_stats["errors"] += 1
            return None

    def get_user_preferred_name(self, slack_user_id: str) -> Optional[str]:
        """
        Get a user's preferred name from their Notion page with caching.
        
        Args:
            slack_user_id: The Slack user ID to look up
            
        Returns:
            The user's preferred name if found, None otherwise
        """
        cache_key = f"user_name_{slack_user_id}"
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache_stats["hits"] += 1
                return self.cache[cache_key]
        
        # Get page properties (this is also cached)
        properties = self.get_user_page_properties(slack_user_id)
        if not properties:
            return None
        
        # Extract preferred name from properties
        preferred_name = None
        preferred_name_prop = properties.get("PreferredName")
        
        if preferred_name_prop and preferred_name_prop.get("type") == "rich_text":
            rich_text_array = preferred_name_prop.get("rich_text", [])
            if rich_text_array and len(rich_text_array) > 0:
                preferred_name = rich_text_array[0].get("plain_text")
        
        # Cache the result (even if None)
        with self.cache_lock:
            self.cache[cache_key] = preferred_name
            self.cache_stats["misses"] += 1
            
        return preferred_name

    def get_user_page_content(self, slack_user_id: str) -> Optional[str]:
        """
        Retrieve the concatenated text content from a user's Notion page with caching.
        
        Args:
            slack_user_id: The Slack user ID to look up
            
        Returns:
            The page content if found, empty string if page exists but is empty,
            None if page doesn't exist or error occurs
        """
        cache_key = f"user_page_content_{slack_user_id}"
        
        # Check cache first
        with self.page_content_lock:
            if cache_key in self.page_content_cache:
                self.cache_stats["hits"] += 1
                logger.info(f"Cache hit for user page content: {slack_user_id}")
                return self.page_content_cache[cache_key]
        
        if not self.is_available():
            logger.warning("Notion client not available. Cannot get user page content.")
            return None
        
        # Get page ID (this is also cached)
        page_id = self.get_user_page_id(slack_user_id)
        if not page_id:
            logger.info(f"No Notion page found for Slack User ID: {slack_user_id}")
            return None
        
        try:
            # Fetch all blocks from the page
            all_text_parts = []
            has_more = True
            next_cursor = None
            
            logger.debug(f"Fetching blocks for Notion page {page_id}")
            
            while has_more:
                blocks_response = self.client.blocks.children.list(
                    block_id=page_id,
                    start_cursor=next_cursor
                )
                
                results = blocks_response.get("results", [])
                logger.debug(f"Fetched {len(results)} blocks from Notion page {page_id}")
                
                # Process each block
                for block in results:
                    block_type = block.get("type")
                    
                    # Handle text-bearing blocks
                    if block_type in [
                        "paragraph", "heading_1", "heading_2", "heading_3",
                        "bulleted_list_item", "numbered_list_item", "code"
                    ]:
                        text_element = block.get(block_type, {})
                        current_block_texts = []
                        
                        rich_text_list = text_element.get("rich_text", [])
                        for rich_text_item in rich_text_list:
                            plain_text = rich_text_item.get("plain_text", "")
                            if plain_text:
                                current_block_texts.append(plain_text)
                        
                        if current_block_texts:
                            block_full_text = "".join(current_block_texts)
                            all_text_parts.append(block_full_text)
                
                has_more = blocks_response.get("has_more", False)
                next_cursor = blocks_response.get("next_cursor")
            
            # Join all text parts
            full_content = "\n".join(filter(None, all_text_parts)).strip()
            
            # Cache the result
            with self.page_content_lock:
                self.page_content_cache[cache_key] = full_content if full_content else ""
                self.cache_stats["misses"] += 1
            
            if full_content:
                logger.info(f"Retrieved Notion page content for user {slack_user_id} (length: {len(full_content)})")
                logger.debug(f"First 200 chars of content: {full_content[:200]}...")
                return full_content
            else:
                logger.info(f"Notion page for user {slack_user_id} exists but is empty")
                return ""  # Return empty string for empty page
                
        except Exception as e:
            logger.error(f"Error fetching Notion content for user {slack_user_id}: {e}", exc_info=True)
            self.cache_stats["errors"] += 1
            return None
    
    def store_user_nickname(
        self, 
        slack_user_id: str, 
        nickname: str, 
        slack_display_name: Optional[str] = None
    ) -> bool:
        """Store a user's nickname in their Notion page using exact property names from database."""
        if not self.is_available():
            logger.warning("Notion client not available. Cannot store nickname.")
            return False
        
        # Get the page ID (could be cached)
        page_id = self.get_user_page_id(slack_user_id)
        
        # Prepare properties to update - only use safe, common properties
        properties_to_update = {
            "PreferredName": {"rich_text": [{"type": "text", "text": {"content": nickname}}]}
        }
        
        if slack_display_name:
            properties_to_update["SlackDisplayName"] = {
                "rich_text": [{"type": "text", "text": {"content": slack_display_name}}]
            }

        try:
            if page_id:
                # Update existing page
                logger.info(f"Updating Notion page for user {slack_user_id} with nickname: {nickname}")
                self.client.pages.update(page_id=page_id, properties=properties_to_update)
            else:
                # Create new page - get EXACT property names from database
                logger.info(f"Creating new Notion page for user {slack_user_id} with nickname: {nickname}")
                
                # Get the database schema to see exact property names
                database_info = self.client.databases.retrieve(database_id=self.user_db_id)
                available_properties = database_info.get("properties", {})
                
                logger.info(f"Available properties in database: {list(available_properties.keys())}")
                
                # Build properties dictionary with EXACT property names from database
                new_page_properties = {
                    # Primary field (title) - this should always exist
                    "UserID": {"title": [{"type": "text", "text": {"content": slack_user_id}}]},
                    "PreferredName": {"rich_text": [{"type": "text", "text": {"content": nickname}}]}
                }
                
                # Map of what we want to set -> what to look for in database
                desired_properties = {
                    "SlackUserID": slack_user_id,
                    "SlackDisplayName": slack_display_name or "",
                    "FavoriteEmoji": "",
                    "LanguagePreference": "",
                    "WorkLocation": "",
                    "HomeLocation": "",
                    "Role": "",
                    "Timezone": "",
                    "User Notes (Admin)": "",
                    "User TODO Page Link": None,
                }
                
                # Check for LastBotInteraction vs "Last Bot Interaction"
                if "LastBotInteraction" in available_properties:
                    desired_properties["LastBotInteraction"] = datetime.now().isoformat()
                elif "Last Bot Interaction" in available_properties:
                    desired_properties["Last Bot Interaction"] = datetime.now().isoformat()
                
                # Only add properties that actually exist in the database with exact names
                for prop_name in list(desired_properties.keys()):
                    if prop_name in available_properties:
                        prop_type = available_properties[prop_name].get("type")
                        prop_value = desired_properties[prop_name]
                        
                        logger.debug(f"Setting property {prop_name} (type: {prop_type}) = {prop_value}")
                        
                        if prop_type == "rich_text":
                            if prop_value:  # Only set non-empty values
                                new_page_properties[prop_name] = {
                                    "rich_text": [{"type": "text", "text": {"content": str(prop_value)}}]
                                }
                            else:
                                new_page_properties[prop_name] = {"rich_text": []}
                        elif prop_type == "date":
                            if prop_value:
                                new_page_properties[prop_name] = {"date": {"start": prop_value}}
                            # Don't set empty dates
                        elif prop_type == "url":
                            if prop_value:
                                new_page_properties[prop_name] = {"url": prop_value}
                            # Don't set empty URLs
                        else:
                            logger.debug(f"Skipping property {prop_name} with type {prop_type} (not handled)")
                    else:
                        logger.debug(f"Property {prop_name} does not exist in database, skipping")
                
                logger.info(f"Creating page with properties: {list(new_page_properties.keys())}")
                
                # Log the exact properties being sent for debugging
                for prop_name, prop_data in new_page_properties.items():
                    logger.debug(f"Property: {prop_name} = {prop_data}")
                
                # Initial page content with proper structured sections
                initial_body_content = [
                    # Projects section
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Projects"}}]}
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": "(Add your projects here.)"}}]}
                    },
                    # Preferences section
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Preferences"}}]}
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": "(Add your preferences here.)"}}]}
                    },
                    # Known Facts section
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Known Facts"}}]}
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": "(Add facts about yourself here.)"}}]}
                    },
                    # Instructions section
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Instructions"}}]}
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": "(Add instructions for the bot here.)"}}]}
                    }
                ]
                
                # Create the page
                try:
                    result = self.client.pages.create(
                        parent={"database_id": self.user_db_id},
                        properties=new_page_properties,
                        children=initial_body_content
                    )
                    
                    if result and result.get("id"):
                        logger.info(f"Successfully created page: {result.get('id')}")
                    else:
                        logger.error(f"Page creation returned unexpected result: {result}")
                        return False
                        
                except Exception as create_error:
                    logger.error(f"Notion API error during page creation: {create_error}")
                    
                    # Try with minimal properties if full creation failed
                    logger.info("Trying with minimal properties...")
                    minimal_properties = {
                        "UserID": {"title": [{"type": "text", "text": {"content": slack_user_id}}]},
                        "PreferredName": {"rich_text": [{"type": "text", "text": {"content": nickname}}]}
                    }
                    
                    try:
                        result = self.client.pages.create(
                            parent={"database_id": self.user_db_id},
                            properties=minimal_properties,
                            children=initial_body_content
                        )
                        
                        if result and result.get("id"):
                            logger.info(f"Successfully created page with minimal properties: {result.get('id')}")
                        else:
                            logger.error(f"Minimal page creation also failed: {result}")
                            return False
                            
                    except Exception as minimal_error:
                        logger.error(f"Even minimal page creation failed: {minimal_error}")
                        return False
            
            # Verify the nickname was actually stored
            try:
                # Small delay to account for Notion's eventual consistency
                import time
                time.sleep(0.5)
                
                # Clear cache first
                self.invalidate_user_cache(slack_user_id)
                
                stored_name = self.get_user_preferred_name(slack_user_id)
                if stored_name != nickname:
                    logger.warning(f"Nickname verification: Expected '{nickname}', Got '{stored_name}'")
                    # Don't fail for verification mismatches, might be timing
            except Exception as verification_error:
                logger.warning(f"Could not verify nickname storage: {verification_error}")
            
            # Invalidate cache after successful operation
            self.invalidate_user_cache(slack_user_id)
            logger.info(f"Successfully processed nickname '{nickname}' for user {slack_user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing nickname for user {slack_user_id}: {e}", exc_info=True)
            self.cache_stats["errors"] += 1
            return False
    
    def handle_nickname_command(
        self, 
        prompt_text: str, 
        slack_user_id: str, 
        slack_display_name: Optional[str] = None
    ) -> Tuple[Optional[str], bool]:
        """
        Process a nickname command and update Notion.
        
        FIXED: More specific patterns to avoid conflicts with memory commands.
        
        Args:
            prompt_text: The user's message text
            slack_user_id: The Slack user ID
            slack_display_name: Optional Slack display name
            
        Returns:
            A tuple of (response_message, success_boolean)
        """
        # FIXED: Check if this is likely a memory command first
        # If it starts with memory keywords, don't treat as nickname
        memory_indicators = [
            "remember that", "remember:", "fact:", "fact that", 
            "store that", "note that", "keep in mind that"
        ]
        
        prompt_lower = prompt_text.lower()
        for indicator in memory_indicators:
            if indicator in prompt_lower:
                logger.debug(f"Detected memory indicator '{indicator}' in text, not treating as nickname command")
                return None, False
        
        # Extract nickname from text using FIXED more specific regex patterns
        nickname = None
        
        # FIXED Pattern 1: Explicit "call me" commands with quotes
        quoted_pattern = r'(?:call me|please call me|you can call me)\s+[\'"]([a-zA-Z0-9\s\'-]+)[\'"]'
        match = re.search(quoted_pattern, prompt_text, re.IGNORECASE)
        if match:
            nickname = match.group(1).strip()
            logger.debug(f"Matched quoted call me pattern: {nickname}")
        
        # FIXED Pattern 2: Explicit "call me" commands without quotes
        if not nickname:
            call_me_pattern = r'(?:call me|please call me|you can call me)\s+([a-zA-Z0-9\s\'-]+?)(?:\.|$|,|\!|\?)'
            match = re.search(call_me_pattern, prompt_text, re.IGNORECASE)
            if match:
                nickname = match.group(1).strip()
                # Clean up potential trailing punctuation
                nickname = re.sub(r'[.,!?]$', '', nickname)
                logger.debug(f"Matched call me pattern: {nickname}")
        
        # FIXED Pattern 3: "My name is" but ONLY if it's a simple declaration, not part of a memory statement
        if not nickname:
            # Only match if it's at the start of the sentence or after punctuation
            name_pattern = r'(?:^|[.!?]\s+)(?:my name is|i\'m called|i go by)\s+([a-zA-Z0-9\s\'-]+?)(?:\.|$|,|\!|\?)'
            match = re.search(name_pattern, prompt_text, re.IGNORECASE)
            if match:
                nickname = match.group(1).strip()
                nickname = re.sub(r'[.,!?]$', '', nickname)
                logger.debug(f"Matched name declaration pattern: {nickname}")
        
        # FIXED: Do NOT match "I am" patterns as they're too likely to be memory statements
        
        # If no nickname found, return None
        if not nickname:
            logger.debug("No nickname pattern matched")
            return None, False
        
        # Additional validation: nickname should be reasonably short
        if len(nickname) > 50:
            logger.debug(f"Nickname too long ({len(nickname)} chars), likely not a nickname command")
            return None, False
        
        # Additional validation: avoid common memory statement fragments
        memory_words = ["that", "how", "when", "where", "what", "why", "because", "since", "as", "like"]
        if any(word in nickname.lower() for word in memory_words):
            logger.debug(f"Nickname contains memory-like words, likely not a nickname command: {nickname}")
            return None, False
        
        logger.info(f"Processing nickname command for user {slack_user_id}: '{nickname}'")
        
        # Store the nickname with verification
        success = self.store_user_nickname(slack_user_id, nickname, slack_display_name)
        
        if success:
            return f"Got it! I'll call you {nickname} from now on. 👍", True
        else:
            return "Hmm, I had a little trouble remembering that nickname. Please try again later.", False

    def get_notion_page_content(self, page_id: str) -> Optional[str]:
        """
        Retrieve content from any Notion page by ID with caching.
        
        Args:
            page_id: The Notion page ID
            
        Returns:
            The page content if found, empty string if page exists but is empty,
            None if error occurs
        """
        cache_key = f"page_content_{page_id}"
        
        # Check cache first
        with self.page_content_lock:
            if cache_key in self.page_content_cache:
                self.cache_stats["hits"] += 1
                return self.page_content_cache[cache_key]
        
        if not self.is_available():
            return None
        
        try:
            # Fetch all blocks from the page
            all_text_parts = []
            has_more = True
            next_cursor = None
            
            logger.debug(f"Fetching blocks for Notion page {page_id}")
            
            while has_more:
                blocks_response = self.client.blocks.children.list(
                    block_id=page_id,
                    start_cursor=next_cursor
                )
                
                results = blocks_response.get("results", [])
                
                # Process each block
                for block in results:
                    block_type = block.get("type")
                    
                    # Handle text-bearing blocks
                    if block_type in [
                        "paragraph", "heading_1", "heading_2", "heading_3",
                        "bulleted_list_item", "numbered_list_item", "code"
                    ]:
                        text_element = block.get(block_type, {})
                        current_block_texts = []
                        
                        rich_text_list = text_element.get("rich_text", [])
                        for rich_text_item in rich_text_list:
                            plain_text = rich_text_item.get("plain_text", "")
                            if plain_text:
                                current_block_texts.append(plain_text)
                        
                        if current_block_texts:
                            block_full_text = "".join(current_block_texts)
                            all_text_parts.append(block_full_text)
                
                has_more = blocks_response.get("has_more", False)
                next_cursor = blocks_response.get("next_cursor")
            
            # Join all text parts
            full_content = "\n".join(filter(None, all_text_parts)).strip()
            
            # Cache the result
            with self.page_content_lock:
                self.page_content_cache[cache_key] = full_content if full_content else ""
                self.cache_stats["misses"] += 1
            
            if full_content:
                logger.info(f"Retrieved Notion page content for page {page_id} (length: {len(full_content)})")
                return full_content
            else:
                logger.info(f"Notion page {page_id} exists but is empty")
                return ""  # Return empty string for empty page
                
        except Exception as e:
            logger.error(f"Error fetching Notion content for page {page_id}: {e}", exc_info=True)
            self.cache_stats["errors"] += 1
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.cache_lock:
            stats = self.cache_stats.copy()
            stats["cache_size"] = len(self.cache)
            stats["page_content_cache_size"] = len(self.page_content_cache)
            stats["hit_ratio"] = (
                stats["hits"] / (stats["hits"] + stats["misses"]) 
                if (stats["hits"] + stats["misses"]) > 0 else 0
            )
            return stats
        
    def append_fact_to_user_page(self, slack_user_id: str, fact_text: str) -> bool:
        """
        Append a known fact to a user's Notion page.
        Uses the structured memory handler.
        
        Args:
            slack_user_id: The Slack user ID
            fact_text: The fact to append
            
        Returns:
            True if successful, False otherwise
        """
        return self.memory_handler.add_known_fact(slack_user_id, fact_text)
    
    def append_preference_to_user_page(self, slack_user_id: str, preference_text: str) -> bool:
        """
        Append a preference to a user's Notion page as a bulleted list item.
        Uses the structured memory handler.
        
        Args:
            slack_user_id: The Slack user ID
            preference_text: The preference to append
            
        Returns:
            True if successful, False otherwise
        """
        return self.memory_handler.add_preference(slack_user_id, preference_text)
    
    def add_todo_item(self, slack_user_id: str, todo_text: str) -> bool:
        """
        Add a TODO item to a user's Notion page.
        
        Args:
            slack_user_id: The Slack user ID
            todo_text: The TODO item to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or create user page
            page_id = self.get_user_page_id(slack_user_id)
            if not page_id:
                # Create a new page with basic user info
                success = self.store_user_nickname(slack_user_id, slack_user_id, None)
                if not success:
                    return False
                
                # Get the new page ID
                page_id = self.get_user_page_id(slack_user_id)
                if not page_id:
                    return False
            
            # Create a new to-do block
            new_todo_block = {
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": todo_text}
                    }],
                    "checked": False
                }
            }
            
            # First, try to find an Instructions section to add the TODO
            instructions_section = self.find_section_block(page_id, "Instructions")
            if instructions_section:
                # Append to the Instructions section
                self.client.blocks.children.append(
                    block_id=instructions_section.get("id"),
                    children=[new_todo_block]
                )
            else:
                # Append directly to the page
                self.client.blocks.children.append(
                    block_id=page_id, 
                    children=[new_todo_block]
                )
            
            # Invalidate cache
            self.invalidate_user_cache(slack_user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding TODO item for user {slack_user_id}: {e}", exc_info=True)
            return False
        
    def handle_memory_instruction(self, slack_user_id: str, text: str) -> Optional[str]:
        """
        Process a memory instruction and store it appropriately.
        
        Args:
            slack_user_id: The Slack user ID
            text: The message content
        
        Returns:
            A user-friendly message indicating success or failure, or None if not a memory instruction
        """
        return self.memory_handler.handle_memory_instruction(slack_user_id, text)
    
    def _update_location_fact(self, slack_user_id: str, location_type: str, new_location_fact: str) -> bool:
        """
        Update a location fact in a structured format.
        
        Args:
            slack_user_id: The Slack user ID
            location_type: Either "work_location" or "home_location"
            new_location_fact: The new location fact text
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Notion client not available. Cannot update location fact.")
            return False
        
        # Extract just the location from the fact
        location_match = re.search(r"(work|live|am from|was born in|reside in|moved to)\s+(.*?)$", new_location_fact)
        if location_match:
            location_only = location_match.group(2).strip()
        else:
            location_only = new_location_fact
        
        # Get the page ID
        page_id = self.get_user_page_id(slack_user_id)
        if not page_id:
            # If no page exists, create one and then add the fact
            success = self.store_user_nickname(slack_user_id, slack_user_id, None)
            if not success:
                return False
            
            # Get the new page ID
            page_id = self.get_user_page_id(slack_user_id)
            if not page_id:
                return False
        
        try:
            # First update the property in the database (if applicable)
            property_name = "WorkLocation" if location_type == "work_location" else "HomeLocation"
            try:
                # Try to update the property if it exists
                properties_update = {
                    property_name: {"rich_text": [{"type": "text", "text": {"content": location_only}}]}
                }
                self.client.pages.update(page_id=page_id, properties=properties_update)
            except Exception as e:
                # If the property doesn't exist, we'll continue with the page content update
                logger.warning(f"Could not update property {property_name}: {e}")
            
            # Get all blocks from the page
            blocks_response = self.client.blocks.children.list(block_id=page_id)
            blocks = blocks_response.get("results", [])
            
            # Look for location section
            location_section_heading = None
            location_section_blocks = []
            
            # First look for a dedicated Location Information section
            for i, block in enumerate(blocks):
                block_type = block.get("type")
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    heading_text = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    if "location" in heading_text.lower():
                        location_section_heading = block
                        location_section_start_index = i
                        break
            
            # If we found a location section, find blocks until next heading
            if location_section_heading:
                # Collect all list items under this heading until we hit another heading
                for i in range(location_section_start_index + 1, len(blocks)):
                    block = blocks[i]
                    block_type = block.get("type")
                    
                    if block_type in ["heading_1", "heading_2", "heading_3"]:
                        # Found next heading, stop collecting
                        break
                    
                    if block_type in ["bulleted_list_item", "paragraph"]:
                        text_content = ""
                        rich_text = block.get(block_type, {}).get("rich_text", [])
                        
                        for text_item in rich_text:
                            text_content += text_item.get("plain_text", "")
                        
                        # Check if this is a location block matching our type
                        is_work_location = "work" in text_content.lower() and location_type == "work_location"
                        is_home_location = any(x in text_content.lower() for x in ["live", "reside", "home"]) and location_type == "home_location"
                        
                        if is_work_location or is_home_location:
                            location_section_blocks.append((i, block))
            
            # If we found a matching location block, update it
            if location_section_blocks:
                # Update the first matching location block
                index, block = location_section_blocks[0]
                block_id = block.get("id")
                block_type = block.get("type")
                
                # Prepare updated text based on location type
                if location_type == "work_location":
                    updated_text = f"Work Location: {location_only}"
                else:
                    updated_text = f"Home Location: {location_only}"
                
                # Create the update data
                updated_block = {
                    block_type: {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": updated_text}
                        }]
                    }
                }
                
                # Update the block
                result = self.client.blocks.update(block_id=block_id, **updated_block)
                
                # Invalidate cache
                self.invalidate_user_cache(slack_user_id)
                
                return result is not None
            
            # If no location section or no matching block, look in Known Facts section
            known_facts_heading = None
            known_facts_blocks = []
            
            for i, block in enumerate(blocks):
                block_type = block.get("type")
                if block_type in ["heading_1", "heading_2", "heading_3"]:
                    heading_text = ""
                    rich_text = block.get(block_type, {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    if "known facts" in heading_text.lower() or "facts" in heading_text.lower():
                        known_facts_heading = block
                        known_facts_start_index = i
                        break
            
            # If we found a Known Facts section, find blocks until next heading
            if known_facts_heading:
                # Collect all list items under this heading until we hit another heading
                for i in range(known_facts_start_index + 1, len(blocks)):
                    block = blocks[i]
                    block_type = block.get("type")
                    
                    if block_type in ["heading_1", "heading_2", "heading_3"]:
                        # Found next heading, stop collecting
                        break
                    
                    if block_type in ["bulleted_list_item", "paragraph"]:
                        text_content = ""
                        rich_text = block.get(block_type, {}).get("rich_text", [])
                        
                        for text_item in rich_text:
                            text_content += text_item.get("plain_text", "")
                        
                        # Check if this is a location block matching our type
                        is_work_location = "work" in text_content.lower() and location_type == "work_location"
                        is_home_location = any(x in text_content.lower() for x in ["live", "reside", "home"]) and location_type == "home_location"
                        
                        if is_work_location or is_home_location:
                            known_facts_blocks.append((i, block))
            
            # If we found a matching fact in Known Facts, update it
            if known_facts_blocks:
                # Update the first matching fact
                index, block = known_facts_blocks[0]
                block_id = block.get("id")
                block_type = block.get("type")
                
                # Create the update data for bullet point (keep the bullet format)
                updated_block = {
                    block_type: {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": new_location_fact}
                        }]
                    }
                }
                
                # Update the block
                result = self.client.blocks.update(block_id=block_id, **updated_block)
                
                # Invalidate cache
                self.invalidate_user_cache(slack_user_id)
                
                return result is not None
            
            # If no appropriate section found, create or append to Known Facts section
            if known_facts_heading:
                # Append to existing Known Facts section
                new_fact_block = {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": new_location_fact}
                        }]
                    }
                }
                
                # Append the new block after the heading
                result = self.client.blocks.children.append(
                    block_id=known_facts_heading.get("id"),
                    children=[new_fact_block]
                )
            else:
                # Create a new Known Facts section with this fact
                new_blocks = [
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": "Known Facts:"}
                            }]
                        }
                    },
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": new_location_fact}
                            }]
                        }
                    }
                ]
                
                # Add the new Known Facts section
                result = self.client.blocks.children.append(block_id=page_id, children=new_blocks)
            
            # Invalidate cache
            self.invalidate_user_cache(slack_user_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating location for user {slack_user_id}: {e}", exc_info=True)
            return False
    
    def _replace_projects_section(self, slack_user_id: str, project_name: str) -> bool:
        """
        Replace the projects section with a new project.
        
        Args:
            slack_user_id: The Slack user ID
            project_name: The name of the new project
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Notion client not available. Cannot replace projects.")
            return False
        
        # Get the page ID
        page_id = self.get_user_page_id(slack_user_id)
        if not page_id:
            # If no page exists, create one and then add the projects section
            success = self.store_user_nickname(slack_user_id, slack_user_id, None)
            if not success:
                return False
            
            # Get the new page ID
            page_id = self.get_user_page_id(slack_user_id)
            if not page_id:
                return False
        
        try:
            # Create heading and bullet for project
            new_blocks = [
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": "Projects:"}
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
            
            # Get all blocks from the page
            blocks_response = self.client.blocks.children.list(block_id=page_id)
            blocks = blocks_response.get("results", [])
            
            # Find existing projects section
            projects_section_start = None
            projects_section_blocks = []
            
            for i, block in enumerate(blocks):
                block_type = block.get("type")
                if block_type == "heading_2":
                    heading_text = ""
                    rich_text = block.get("heading_2", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    if "projects" in heading_text.lower() or "project" in heading_text.lower():
                        projects_section_start = i
                        projects_section_blocks.append(block)
                elif projects_section_start is not None and (
                    block_type == "bulleted_list_item" or 
                    (block_type == "paragraph" and i == projects_section_start + 1)
                ):
                    # This is a project item or paragraph right after the projects heading
                    projects_section_blocks.append(block)
                elif projects_section_start is not None and block_type == "heading_2":
                    # We've reached the next section
                    break
            
            # If we found a projects section, delete all its blocks
            if projects_section_blocks:
                for block in projects_section_blocks:
                    self.client.blocks.delete(block_id=block.get("id"))
            
            # Add the new projects section
            self.client.blocks.children.append(block_id=page_id, children=new_blocks)
            
            # Invalidate cache
            self.invalidate_user_cache(slack_user_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error replacing projects for user {slack_user_id}: {e}", exc_info=True)
            return False

    def _add_to_projects_section(self, slack_user_id: str, project_name: str) -> bool:
        """
        Add a project to the projects section.
        
        Args:
            slack_user_id: The Slack user ID
            project_name: The name of the project to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Notion client not available. Cannot add project.")
            return False
        
        # Get the page ID
        page_id = self.get_user_page_id(slack_user_id)
        if not page_id:
            # If no page exists, create one and then add the projects section
            success = self.store_user_nickname(slack_user_id, slack_user_id, None)
            if not success:
                return False
            
            # Get the new page ID
            page_id = self.get_user_page_id(slack_user_id)
            if not page_id:
                return False
        
        try:
            # Get all blocks from the page
            blocks_response = self.client.blocks.children.list(block_id=page_id)
            blocks = blocks_response.get("results", [])
            
            # Find existing projects section
            projects_section_block = None
            
            for block in blocks:
                block_type = block.get("type")
                if block_type == "heading_2":
                    heading_text = ""
                    rich_text = block.get("heading_2", {}).get("rich_text", [])
                    
                    for text_item in rich_text:
                        heading_text += text_item.get("plain_text", "")
                    
                    if "projects" in heading_text.lower() or "project" in heading_text.lower():
                        projects_section_block = block
                        break
            
            # Create a new project bullet
            new_project_block = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": project_name}
                    }]
                }
            }
            
            if projects_section_block:
                # Add the new project to the existing section
                self.client.blocks.children.append(
                    block_id=projects_section_block.get("id"),
                    children=[new_project_block]
                )
            else:
                # Create a new projects section with this project
                new_blocks = [
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": "Projects:"}
                            }]
                        }
                    },
                    new_project_block
                ]
                
                # Add the new projects section
                self.client.blocks.children.append(block_id=page_id, children=new_blocks)
            
            # Invalidate cache
            self.invalidate_user_cache(slack_user_id)
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding project for user {slack_user_id}: {e}", exc_info=True)
            return False
    
    def _classify_memory_instruction(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Classify a message as a memory instruction type and extract the clean content.
        Delegates to the memory handler.
        
        Args:
            text: The message content
            
        Returns:
            A tuple of (memory_type, cleaned_content) where cleaned_content is None 
            if the message is not a memory instruction
        """
        return self.memory_handler.classify_memory_instruction(text)
    
    def find_section_block(self, page_id: str, section_name: str) -> Optional[Dict[str, Any]]:
        """
        Find a section block by name.
        
        Args:
            page_id: The page ID
            section_name: The name of the section
            
        Returns:
            The section block if found, None otherwise
        """
        return self.memory_handler._find_section_block(page_id, section_name)
    
    def create_section(self, page_id: str, section_name: str) -> Dict[str, Any]:
        """
        Create a section block.
        
        Args:
            page_id: The page ID
            section_name: The name of the section
            
        Returns:
            The created section block
        """
        return self.memory_handler._create_section(page_id, section_name)
    
    def find_text_block_in_section(self, page_id: str, section_id: str, text_prefix: str, exact_match: bool = False) -> Optional[Dict[str, Any]]:
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
        return self.memory_handler._find_text_block_in_section(page_id, section_id, text_prefix, exact_match)
    
    def debug_user_notion_page(self, slack_user_id: str) -> Dict[str, Any]:
        """
        Generate a detailed diagnostic report of a user's Notion page.
        Useful for debugging issues with memory operations.
        
        Args:
            slack_user_id: The Slack user ID
            
        Returns:
            A dictionary with diagnostic information
        """
        logger.info(f"Running Notion page diagnostics for user {slack_user_id}")
        diagnostics = {
            "slack_user_id": slack_user_id,
            "timestamp": datetime.now().isoformat(),
            "errors": [],
            "page_exists": False,
            "sections": {},
            "cache_status": {}
        }
        
        try:
            # Check if user page exists
            page_id = self.get_user_page_id(slack_user_id)
            if not page_id:
                diagnostics["errors"].append("User page not found")
                return diagnostics
            
            diagnostics["page_exists"] = True
            diagnostics["page_id"] = page_id
            
            # Check cache status
            cache_key_page = f"user_page_id_{slack_user_id}"
            cache_key_content = f"user_page_content_{slack_user_id}"
            
            with self.cache_lock:
                diagnostics["cache_status"]["page_id_in_cache"] = cache_key_page in self.cache
                if cache_key_page in self.cache:
                    diagnostics["cache_status"]["cached_page_id"] = self.cache[cache_key_page]
            
            with self.page_content_lock:
                diagnostics["cache_status"]["content_in_cache"] = cache_key_content in self.page_content_cache
            
            # Get all blocks to analyze page structure
            try:
                blocks_response = self.client.blocks.children.list(block_id=page_id)
                results = blocks_response.get("results", [])
                diagnostics["total_blocks"] = len(results)
                
                # Track sections
                current_section = None
                for block in results:
                    block_type = block.get("type")
                    block_id = block.get("id")
                    
                    # Identify section headings
                    if block_type in ["heading_1", "heading_2", "heading_3"]:
                        text_content = ""
                        rich_text = block.get(block_type, {}).get("rich_text", [])
                        
                        for text_item in rich_text:
                            text_content += text_item.get("plain_text", "")
                        
                        current_section = text_content
                        diagnostics["sections"][current_section] = {
                            "id": block_id,
                            "type": block_type,
                            "items": []
                        }
                    
                    # Track items under current section
                    elif current_section and block_type in ["paragraph", "bulleted_list_item", "numbered_list_item", "to_do"]:
                        text_content = ""
                        rich_text = block.get(block_type, {}).get("rich_text", [])
                        
                        for text_item in rich_text:
                            text_content += text_item.get("plain_text", "")
                        
                        diagnostics["sections"][current_section]["items"].append({
                            "id": block_id,
                            "type": block_type,
                            "content": text_content
                        })
            
            except Exception as e:
                diagnostics["errors"].append(f"Error listing blocks: {str(e)}")
            
            # Get & summarize "Known Facts" section specifically
            # This is particularly relevant for the deletion issue
            if "Known Facts" in diagnostics["sections"]:
                known_facts = diagnostics["sections"]["Known Facts"]
                known_facts["item_count"] = len(known_facts["items"])
                
                # Extract just the text content for easy viewing
                known_facts["text_items"] = [item["content"] for item in known_facts["items"]]
                
                # Check for items containing "Meatloaf" as this was the reported issue
                meatloaf_items = [
                    {"content": item["content"], "id": item["id"]} 
                    for item in known_facts["items"] 
                    if "meatloaf" in item["content"].lower()
                ]
                
                if meatloaf_items:
                    known_facts["meatloaf_items"] = meatloaf_items
            
            return diagnostics
        
        except Exception as e:
            logger.error(f"Error in debug_user_notion_page: {e}", exc_info=True)
            diagnostics["errors"].append(f"Unexpected error: {str(e)}")
            return diagnostics
    