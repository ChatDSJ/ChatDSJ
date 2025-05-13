from typing import Dict, List, Any, Optional, Tuple
import re
from loguru import logger
import asyncio
from collections import defaultdict

class HistoryManager:
    """
    Manages retrieval, filtering and processing of channel history.
    Includes deep historical search and cross-thread searching capability.
    """
    
    def __init__(self):
        # Common patterns to identify search intent in user queries
        self.search_patterns = [
            r"(?:has|have|did)\s+anyone\s+(?:talk|discuss|mention)(?:ed)?\s+(?:about\s+)?([^?\.]+)",
            r"(?:was|were)\s+([^?\.]+)\s+(?:discussed|mentioned|talked about)",
            r"(?:any|are there)\s+discussions?\s+(?:about|on|regarding)\s+([^?\.]+)"
        ]
        
        # Constants for search limits
        self.SEARCH_DEPTH_LIMIT = 2000  # Increased to 2000 messages for deep search
        self.THREAD_SEARCH_LIMIT = 100  # Messages per thread
        self.MAX_THREADS_TO_SEARCH = 20  # Maximum number of threads to search
    
    async def retrieve_and_filter_history(self,
                                        slack_service,
                                        channel_id: str,
                                        thread_ts: Optional[str],
                                        prompt: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Extract the search topic (existing code)
        search_topic = self.extract_search_topic(prompt)
        is_search_query = search_topic is not None
        
        # Set up query parameters (existing code)
        query_params = {
            "limit": self.SEARCH_DEPTH_LIMIT if is_search_query else 100,
            "is_search": is_search_query,
            "search_topic": search_topic,
            "description": f"Deep search for: {search_topic}" if is_search_query else "Recent context"
        }
        
        # Check if this is a thread summary request
        is_thread_summary = (thread_ts is not None and 
                            ("summarize" in prompt.lower() or 
                            "summary" in prompt.lower() or
                            "recap" in prompt.lower()))
        
        # HANDLE SEARCH QUERY CASE (Search Query + Deep Search)
        if is_search_query and search_topic:
            # Existing search query handling code... (from your example, should still function correctly)
            logger.info(f"Performing deep historical search for topic: '{search_topic}'")

            # Step 1: Fetch deep channel history
            main_channel_history = await asyncio.to_thread(
                slack_service.fetch_channel_history,
                channel_id,
                self.SEARCH_DEPTH_LIMIT
            )

            logger.info(f"Retrieved {len(main_channel_history)} main channel messages")

            # Step 2: Identify potential thread parent messages
            thread_parent_candidates = [
                msg for msg in main_channel_history
                if msg.get("reply_count", 0) > 0 or msg.get("thread_ts")
            ]

            # Prioritize threads that might contain the search topic
            if search_topic:
                # Check thread parent messages for the search topic
                relevant_thread_parents = []
                for msg in thread_parent_candidates:
                    # Get text from message or blocks
                    msg_text = self.extract_full_message_text(msg).lower()
                    if search_topic.lower() in msg_text:
                        relevant_thread_parents.append(msg)

                # Also include threads with high reply counts
                high_activity_threads = sorted(
                    [msg for msg in thread_parent_candidates if msg.get("reply_count", 0) > 3],
                    key=lambda x: x.get("reply_count", 0),
                    reverse=True
                )[:10]  # Top 10 most active threads

                # Combine and deduplicate
                thread_parents = []
                seen_ts = set()
                for msg in relevant_thread_parents + high_activity_threads:
                    ts = msg.get("thread_ts") or msg.get("ts")
                    if ts and ts not in seen_ts:
                        thread_parents.append(msg)
                        seen_ts.add(ts)

                # Limit to reasonable number
                thread_parents = thread_parents[:self.MAX_THREADS_TO_SEARCH]
            else:
                # Without a search topic, just use most recent threads
                thread_parents = thread_parent_candidates[:self.MAX_THREADS_TO_SEARCH]

            logger.info(f"Identified {len(thread_parents)} thread parent messages to search")

            # Step 3: Fetch thread replies for each parent
            all_thread_messages = []
            for parent in thread_parents:
                parent_ts = parent.get("thread_ts") or parent.get("ts")
                if not parent_ts:
                    continue

                try:
                    thread_replies = await asyncio.to_thread(
                        slack_service.fetch_thread_history,
                        channel_id,
                        parent_ts,
                        self.THREAD_SEARCH_LIMIT
                    )
                    all_thread_messages.extend(thread_replies)
                    logger.debug(f"Retrieved {len(thread_replies)} messages from thread {parent_ts}")
                except Exception as e:
                    logger.error(f"Error fetching thread {parent_ts}: {e}")

            logger.info(f"Retrieved {len(all_thread_messages)} total messages from {len(thread_parents)} threads")

            # Step 4: Combine main channel and thread messages, removing duplicates
            merged_messages = []
            seen_ts = set()

            # Add main channel messages
            for msg in main_channel_history:
                ts = msg.get("ts")
                if ts and ts not in seen_ts:
                    merged_messages.append(msg)
                    seen_ts.add(ts)

            # Add thread messages
            for msg in all_thread_messages:
                ts = msg.get("ts")
                if ts and ts not in seen_ts:
                    merged_messages.append(msg)
                    seen_ts.add(ts)

            logger.info(f"Combined into {len(merged_messages)} unique messages after deduplication")

            # Step 5: Filter by search topic if applicable
            if search_topic:
                # Log that we're searching for the topic
                logger.info(f"Filtering messages containing topic: '{search_topic}'")

                # Before filtering
                filtered_messages = self.find_messages_containing_topic(merged_messages, search_topic)

                # Log filtering results
                logger.info(f"FOUND {len(filtered_messages)} MESSAGES CONTAINING TOPIC '{search_topic}'")
                logger.info(f"Search ratio: {len(filtered_messages)}/{len(merged_messages)} messages matched")

                # Log sample results for debugging
                if filtered_messages:
                    logger.info(f"Sample matching message: {filtered_messages[0].get('text', '')[:100]}")

                # If no matches, try a simpler approach
                if len(filtered_messages) == 0:
                    logger.warning(f"No matches found for topic '{search_topic}'. Trying looser matching...")
                    # Try common variations
                    variations = [
                        search_topic.lower(),
                        search_topic.title(),
                        search_topic.upper()
                    ]
                    # Try each variation
                    for variation in variations:
                        logger.info(f"Trying variation: '{variation}'")
                        variation_matches = [
                            msg for msg in merged_messages
                            if variation in self.extract_full_message_text(msg)
                        ]
                        if variation_matches:
                            logger.info(f"Found {len(variation_matches)} matches using variation '{variation}'")
                            filtered_messages = variation_matches
                            break
            else:
                # For non-search, use recency bias
                sorted_messages = sorted(
                    merged_messages,
                    key=lambda m: float(m.get("ts", "0")),
                    reverse=True
                )
                filtered_messages = sorted_messages[:100]  # Most recent 100

            # Final sort chronologically for presentation
            filtered_messages = sorted(
                filtered_messages,
                key=lambda m: float(m.get("ts", "0"))
            )

        # Handle regular requests for history (non-search and thread summary)
        else: # Not a search query
            limit = query_params.get("limit", 100)

            # Fetch thread-specific history if in a thread
            if thread_ts: # Now inside the else for NON-search
                thread_history = await asyncio.to_thread(
                    slack_service.fetch_thread_history,
                    channel_id,
                    thread_ts,
                    limit
                )

                # For thread summary requests, ONLY use thread history
                if is_thread_summary:
                    query_params["description"] = f"Thread summary (thread_ts: {thread_ts})"
                    filtered_messages = thread_history # THREAD HISTORY ONLY for summaries
                else:
                    # For regular thread context, still get some channel history
                    channel_history = await asyncio.to_thread(
                        slack_service.fetch_channel_history,
                        channel_id,
                        50  # Limited channel context for threads
                    )

                    # Merge without duplicates
                    thread_timestamps = {msg["ts"] for msg in thread_history}
                    merged_messages = thread_history + [
                        msg for msg in channel_history if msg["ts"] not in thread_timestamps
                    ]
                    # Remove duplicates
                    unique_messages = {}
                    for msg in merged_messages:
                        ts = msg.get("ts")
                        if ts and ts not in unique_messages:
                            unique_messages[ts] = msg
            
                    # Sort chronologically
                    filtered_messages = sorted(
                        unique_messages.values(),
                        key=lambda m: float(m.get("ts", "0"))
                    )
            # Not in thread fetch main
            else:
                # Not in a thread, just get channel history
                merged_messages = await asyncio.to_thread(
                    slack_service.fetch_channel_history,
                    channel_id,
                    limit
                )
        
                # Remove duplicates
                unique_messages = {}
                for msg in merged_messages:
                    ts = msg.get("ts")
                    if ts and ts not in unique_messages:
                        unique_messages[ts] = msg
                
                # Sort chronologically
                filtered_messages = sorted(
                    unique_messages.values(),
                    key=lambda m: float(m.get("ts", "0"))
                )

        return filtered_messages, query_params

    def extract_search_topic(self, prompt: str) -> Optional[str]:
        logger.info(f"Extracting search topic from prompt: {prompt}")
        prompt_lower = prompt.lower()
        
        # First: pattern-based extraction
        for pattern in self.search_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                topic = match.group(1).strip()
                topic = re.sub(r'[.,;:!?]+$', '', topic)
                logger.info(f"Extracted topic from pattern: '{topic}'")
                return topic
                
        # Second: direct simple pattern extraction
        direct_patterns = [
            r'about\s+([A-Za-z0-9\s]+)(?:[,\.?!]|$)',
            r'([A-Za-z0-9\s]+)\s+was\s+(?:discussed|mentioned|talked about)',
        ]
        for pattern in direct_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                topic = match.group(1).strip()
                topic = re.sub(r'[.,;:!?]+$', '', topic)
                logger.info(f"Extracted topic from direct pattern: '{topic}'")
                return topic
                
        # Third: fallback to capitalized words (look for adjacent first)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', prompt)
        for i in range(len(capitalized) - 1):
            two_word = f"{capitalized[i]} {capitalized[i+1]}"
            if two_word in prompt:
                logger.info(f"Extracted multi-word capitalized topic: '{two_word}'")
                return two_word
        if capitalized:
            topic = capitalized[-1]
            logger.info(f"Extracted capitalized topic: '{topic}'")
            return topic
            
        # Fourth: quoted phrases
        quotes = re.findall(r'"([^"]+)"', prompt)
        if quotes:
            topic = quotes[0].strip()
            logger.info(f"Extracted quoted topic: '{topic}'")
            return topic
            
        logger.info("No topic could be extracted from the prompt")
        return None
    
    def find_messages_containing_topic(self, 
                                    messages: List[Dict[str, Any]], 
                                    topic: str,
                                    current_user_id: Optional[str] = None) -> List[Dict[str, Any]]:

        # Correct logic would be to first find relevant messages and then filter self-referential messages from those

        topic_lower = topic.lower()
        matched_messages = []
        self_reference_patterns = [
            r"\bhas\s+anyone\s+(?:talk|discuss|mention)(?:ed)?\s+(?:about\s+)?",
            r"\bdid\s+anyone\s+(?:talk|discuss|mention)(?:ed)?\s+(?:about\s+)?",
            r"\bwas\s+[\w\s]+\s+(?:discussed|mentioned|talked about)",
            r"\bhas\s+[\w\s]+\s+been\s+(?:discussed|mentioned|talked about)"
        ]
        
        # Log a sample of messages to check structure
        if messages and len(messages) > 0:
            sample_msg = messages[0]
            has_blocks = "blocks" in sample_msg
            logger.debug(f"Message structure sample - has blocks: {has_blocks}, keys: {list(sample_msg.keys())}")
        
        logger.info(f"Searching for topic '{topic}' in {len(messages)} messages")
        
        match_count = 0
        text_field_only_count = 0
        blocks_only_count = 0
        attachments_only_count = 0
        self_referential_count = 0
        
        for msg in messages:
            # Extract all text from message (including blocks)
            full_text = self.extract_full_message_text(msg).lower()
            text_field = msg.get("text", "").lower()
            
            # Check for matches in different parts
            match_in_text_field = topic_lower in text_field
            match_in_full_text = topic_lower in full_text
            
            # If match found in full text
            if match_in_full_text:
                
                # Skip self-referential messages where someone is asking about the topic
                # rather than actually discussing the topic
                is_self_referential = False
                
                # If this is from the current user asking about the topic, it's self-referential
                if current_user_id and msg.get("user") == current_user_id:
                    # Check if it's a question about the topic rather than discussing the topic
                    for pattern in self_reference_patterns:
                        if re.search(pattern + re.escape(topic_lower), full_text):
                            is_self_referential = True
                            self_referential_count += 1
                            break

                # If it's not self-referential
                if not is_self_referential:
                    matched_messages.append(msg)
                    match_count += 1
                    
                    # Track where the match was found
                    if match_in_text_field:
                        text_field_only_count += 1
                    
                    # Check if match was only in blocks or attachments
                    if match_in_full_text and not match_in_text_field:
                        # Check blocks
                        if "blocks" in msg:
                            blocks_text = self.extract_text_from_blocks(msg).lower()
                            if topic_lower in blocks_text:
                                blocks_only_count += 1
                        
                        # Check attachments
                        if "attachments" in msg:
                            attachments_text = self.extract_text_from_attachments(msg).lower()
                            if topic_lower in attachments_text:
                                attachments_only_count += 1
        
        # Log detailed match statistics
        logger.info(f"📊 Search results for topic '{topic}':")
        logger.info(f"  - Total matches: {match_count}")
        logger.info(f"  - Skipped self-referential messages: {self_referential_count}")
        logger.info(f"  - Matches in text field: {text_field_only_count}")
        logger.info(f"  - Matches only in blocks: {blocks_only_count}")
        logger.info(f"  - Matches only in attachments: {attachments_only_count}")
        
        return matched_messages
    
    def extract_full_message_text(self, msg: Dict[str, Any]) -> str:
        """
        Extract all text content from a message, including blocks.
        
        Args:
            msg: The message object
            
        Returns:
            All extracted text content
        """
        # Start with the main text field
        full_text = msg.get("text", "")
        
        # Add text from blocks
        if "blocks" in msg:
            full_text += " " + self.extract_text_from_blocks(msg)
        
        # Add text from attachments
        if "attachments" in msg:
            full_text += " " + self.extract_text_from_attachments(msg)
        
        return full_text
    
    def extract_text_from_blocks(self, msg: Dict[str, Any]) -> str:
        """Extract text content from message blocks."""
        blocks_text = []

        for block in msg.get("blocks", []):
            for element in block.get("elements", []):
                if isinstance(element, dict):
                    # Direct text
                    if "text" in element and isinstance(element["text"], str):
                        blocks_text.append(element["text"])
                    # Nested elements
                    elif "elements" in element and isinstance(element["elements"], list):
                        for nested in element["elements"]:
                            if isinstance(nested, dict):
                                if "text" in nested and isinstance(nested["text"], str):
                                    blocks_text.append(nested["text"])
                                elif "url" in nested and isinstance(nested["url"], str):
                                    blocks_text.append(nested["url"])

        return " ".join(blocks_text)

    def extract_text_from_attachments(self, msg: Dict[str, Any]) -> str:
        """Extract text content from message attachments."""
        attachments_text = []

        for attachment in msg.get("attachments", []):
            if isinstance(attachment, dict):
                for key in ["text", "title", "fallback"]:
                    value = attachment.get(key)
                    if isinstance(value, str):
                        attachments_text.append(value)

        return " ".join(attachments_text)

    def format_history_for_prompt(self,
                                filtered_messages: List[Dict[str, Any]],
                                query_params: Dict[str, Any],
                                user_display_names: Dict[str, str],
                                bot_user_id: str) -> str:
            """
            Format filtered channel history for inclusion in the prompt.
            
            Args:
                filtered_messages: Filtered message list
                query_params: Parameters from analyze_query
                user_display_names: Mapping of user IDs to display names
                bot_user_id: The bot's user ID
                
            Returns:
                Formatted history string
            """
            # If no messages, return empty context
            if not filtered_messages:
                return "No relevant messages found in channel history."
            
            # Group messages by thread for better context
            thread_groups = defaultdict(list)
            
            # First pass: separate into threads
            for msg in filtered_messages:
                thread_key = msg.get("thread_ts") or msg.get("ts")
                thread_groups[thread_key].append(msg)
            
            # Format with thread grouping for better context
            formatted_parts = []
            
            # For search queries, add a contextual header
            if query_params.get("is_search") and query_params.get("search_topic"):
                topic = query_params["search_topic"]
                thread_count = len(thread_groups)
                message_count = len(filtered_messages)
                
                if message_count > 0:
                    header = (f"YES, I found {message_count} message(s) in {thread_count} conversation(s) "
                            f"that mention '{topic}'. Here are the relevant discussions:") # EXPLICIT positive confirmation
                    formatted_parts.append(header)
                else:
                    formatted_parts.append(f"I did not find any messages that mention '{topic}'.")

            # For thread summaries, add a contextual header
            if "Thread summary" in query_params.get("description", ""):
                message_count = len(filtered_messages)
                thread_ts = query_params.get("description").split("thread_ts: ")[1].strip(")")
                
                if message_count > 0:
                    header = f"Here is a summary of THIS SPECIFIC THREAD with {message_count} messages:"
                    formatted_parts.append(header)
            
            # Add each thread group
            for thread_key, messages in thread_groups.items():
                # Sort messages within thread chronologically
                sorted_msgs = sorted(messages, key=lambda m: float(m.get("ts", "0")))
                
                # Format thread messages
                thread_lines = []
                for msg in sorted_msgs:
                    user_id = msg.get("user") or msg.get("bot_id", "unknown")
                    
                    # Get text from message and blocks
                    text = self.extract_full_message_text(msg).strip()
                    
                    # Skip empty messages
                    if not text:
                        continue
                    
                    # Get user display name
                    display_name = user_display_names.get(user_id, f"User {user_id}")
                    
                    # Format: [Name] Text
                    thread_lines.append(f"[{display_name}] {text}")
                
                # Add thread separator if needed
                if thread_lines and len(thread_groups) > 1:
                    formatted_parts.append("---")
                
                # Add thread messages
                formatted_parts.extend(thread_lines)
            
            # Log total formatted length
            formatted_result = "\n".join(formatted_parts)
            logger.info(f"Formatted message history: {len(formatted_result)} characters")
            
            return formatted_result