from typing import Dict, List, Any, Optional, Tuple
import re
from loguru import logger
import string
from collections import Counter
import nltk

class HistoryManager:
    """
    Manages retrieval, filtering and processing of channel history.
    Provides different retrieval strategies based on query type.
    """
    
    def __init__(self):
        # Keywords for different query types
        self.summarize_keywords = [
            "summarize", "summary", "recap", "highlights", "sum up",
            "summarization", "overview", "brief", "bullet points"
        ]
        
        self.search_keywords = [
            "search", "find", "locate", "mentioned", "talked about", 
            "discussed", "said about", "did anyone", "has anyone",
            "was there", "were there", "remember", "recall", "look up"
        ]
        
        self.history_keywords = [
            "history", "past", "previously", "before", "earlier",
            "last time", "last week", "yesterday", "earlier today",
            "what happened", "went down", "transpired"
        ]
        
        # Regular expressions for search term extraction
        self.search_term_patterns = [
            r"(?:search|find|locate|mentioned|talked about|discussed|said about)(?:\s+about)?\s+(?:the\s+)?(?:topic\s+)?['\"]?([^'\"]+)['\"]?",
            r"(?:did|has|have)\s+(?:anyone|somebody|someone)(?:\s+ever)?\s+(?:talk|mention|discuss|say)\s+(?:about|regarding)?\s+['\"]?([^'\"]+)['\"]?",
            r"(?:was|were)\s+(?:there|any)(?:\s+ever)?\s+(?:any\s+)?(?:discussion|mention|talk)\s+(?:about|regarding|on)?\s+['\"]?([^'\"]+)['\"]?",
        ]
        
        # Regular expressions for time frame extraction
        self.time_frame_patterns = [
            r"(?:in|during|over|for|since|from|within|throughout)\s+(?:the\s+)?(?:last|past|previous)?\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)"
        ]
    
    def determine_history_depth(self, prompt: str) -> Dict[str, Any]:
        """
        Determine the appropriate history retrieval strategy based on the query.
        
        Args:
            prompt: The user's query
            
        Returns:
            Dictionary with retrieval parameters
        """
        prompt_lower = prompt.lower()
        
        # Check for summarization request
        if any(keyword in prompt_lower for keyword in self.summarize_keywords):
            return {
                "limit": 1000,
                "mode": "summarize",
                "description": "Channel summarization"
            }
        
        # Check for search request
        if any(keyword in prompt_lower for keyword in self.search_keywords):
            search_terms = self.extract_search_terms(prompt)
            return {
                "limit": 1000,
                "mode": "search",
                "search_terms": search_terms,
                "description": f"Search for terms: {', '.join(search_terms)}"
            }
        
        # Check for history inquiry
        if any(keyword in prompt_lower for keyword in self.history_keywords):
            time_frame = self.extract_time_frame(prompt)
            return {
                "limit": 500,
                "mode": "history",
                "time_frame": time_frame,
                "description": f"Historical retrieval ({time_frame or 'general'})"
            }
        
        # Default to recent context
        return {
            "limit": 20,
            "mode": "recent",
            "description": "Recent context only"
        }
    
    def extract_search_terms(self, prompt: str) -> List[str]:
        """
        Extracts key search terms from a prompt using NLTK NER and fallback strategies.
        Returns a deduplicated, lowercased list of terms.
        """
        logger.info(f"[extract_search_terms] Starting search term extraction for prompt: {prompt}")
        cleaned = prompt.translate(str.maketrans("", "", string.punctuation)).strip()

        stop_words = {
            "the", "a", "an", "and", "or", "but", "if", "then", 
            "is", "are", "was", "were", "be", "been", "being",
            "this", "that", "these", "those", "do", "does", "did",
            "has", "have", "had", "about", "for", "in", "to", "from",
            "with", "without", "by", "at", "on", "off"
        }

        named_entities = []
        try:
            tokens = nltk.word_tokenize(cleaned)
            pos_tags = nltk.pos_tag(tokens)
            tree = nltk.ne_chunk(pos_tags, binary=False)
            for subtree in tree:
                if hasattr(subtree, 'label'):
                    entity = " ".join(word for word, _ in subtree.leaves())
                    named_entities.append(entity.lower())
            logger.info(f"[extract_search_terms] Named entities found: {named_entities}")
        except Exception as e:
            logger.warning(f"[extract_search_terms] NLTK entity extraction failed: {e}")

        fallback_terms = []
        try:
            words = cleaned.lower().split()
            filtered = [w for w in words if w not in stop_words and len(w) > 3]
            word_counts = Counter(filtered)
            fallback_terms = [word for word, _ in word_counts.most_common(5)]
            logger.info(f"[extract_search_terms] Fallback keyword terms: {fallback_terms}")
        except Exception as e:
            logger.warning(f"[extract_search_terms] Fallback keyword extraction failed: {e}")

        combined = set()
        for term in named_entities + fallback_terms:
            combined.update(term.lower().split())

        final_terms = sorted(combined)
        logger.info(f"[extract_search_terms] Final search terms: {final_terms}")
        return final_terms

    def extract_time_frame(self, prompt: str) -> Optional[str]:
        """
        Extract time frame information from the prompt.
        
        Args:
            prompt: The user's query
            
        Returns:
            Time frame string or None
        """
        for pattern in self.time_frame_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                number = match.group(1)
                unit = match.group(2)
                return f"{number} {unit}"
        
        return None
    
    def filter_relevant_messages(self, 
                                 messages: List[Dict[str, Any]], 
                                 query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter messages based on relevance to the query.
        
        Args:
            messages: List of message objects
            query_params: Parameters from determine_history_depth
            
        Returns:
            Filtered list of messages
        """
        mode = query_params.get("mode", "recent")
        
        if mode == "summarize":
            # For summarization, we want a representative sample
            # Include 30% of messages evenly distributed through time
            total_messages = len(messages)
            sample_size = max(min(int(total_messages * 0.3), 200), 50)  # At least 50, at most 200
            
            # Select messages evenly spaced
            if total_messages <= sample_size:
                return messages
            
            step = total_messages / sample_size
            indices = [min(int(i * step), total_messages - 1) for i in range(sample_size)]
            return [messages[i] for i in indices]
        
        elif mode == "search":
            search_terms = query_params.get("search_terms", [])
            if not search_terms:
                return messages[:50]  # Fall back to recent if no search terms
            
            # Score messages by relevance to search terms
            scored_messages = []
            for msg in messages:
                text = msg.get("text", "").lower()
                score = sum(term.lower() in text for term in search_terms)
                if score > 0:
                    logger.debug(f"Matched message: {text} with score {score}")
                    scored_messages.append((msg, score))
            
            # Sort by relevance score
            sorted_messages = [msg for msg, _ in sorted(scored_messages, 
                                                       key=lambda x: x[1], 
                                                       reverse=True)]
            
            # Include up to 50 relevant messages
            return sorted_messages[:50]
        
        elif mode == "history":
            # For history mode, include a larger set of messages
            # but still limit to avoid token explosion
            return messages[:100]
        
        else:  # "recent" mode
            # Just the most recent messages
            return messages[:20]
    
    async def retrieve_and_filter_history(self,
                                         slack_service,
                                         channel_id: str,
                                         thread_ts: Optional[str],
                                         prompt: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve and filter channel history based on query type.
        
        Args:
            slack_service: The Slack service instance
            channel_id: Slack channel ID
            thread_ts: Thread timestamp if in a thread
            prompt: The user's query
            
        Returns:
            Tuple of (filtered_messages, query_params)
        """
        import asyncio
        
        # Determine history depth
        query_params = self.determine_history_depth(prompt)
        limit = query_params.get("limit", 20)
        
        logger.info(f"History retrieval strategy: {query_params.get('description', 'default')}")
        
        # Retrieve history based on context
        if thread_ts:
            # In a thread, get both thread and channel history
            thread_history = await asyncio.to_thread(
                slack_service.fetch_thread_history,
                channel_id,
                thread_ts,
                limit
            )
            
            channel_history = await asyncio.to_thread(
                slack_service.fetch_channel_history,
                channel_id,
                min(limit, 50)  # Limit channel context in threads
            )
            
            # Merge thread and channel history
            thread_ts_set = {msg["ts"] for msg in thread_history}
            merged_messages = thread_history + [
                msg for msg in channel_history if msg["ts"] not in thread_ts_set
            ]
        else:
            # Not in a thread, just get channel history
            merged_messages = await asyncio.to_thread(
                slack_service.fetch_channel_history,
                channel_id,
                limit
            )
        
        # Filter for relevance
        filtered_messages = self.filter_relevant_messages(merged_messages, query_params)
        
        logger.info(f"Retrieved {len(merged_messages)} messages, filtered to {len(filtered_messages)}")
        
        return filtered_messages, query_params
    
    def format_history_for_prompt(self,
                                 filtered_messages: List[Dict[str, Any]],
                                 query_params: Dict[str, Any],
                                 user_display_names: Dict[str, str],
                                 bot_user_id: str) -> str:
        """
        Format filtered history for inclusion in the prompt.
        
        Args:
            filtered_messages: Filtered message list
            query_params: Parameters from determine_history_depth
            user_display_names: Mapping of user IDs to display names
            bot_user_id: The bot's user ID
            
        Returns:
            Formatted history string
        """
        mode = query_params.get("mode", "recent")
        
        if mode == "summarize":
            # Create a condensed summary
            return self._format_summary(filtered_messages, user_display_names)
        
        elif mode == "search":
            # Format with search term highlighting
            search_terms = query_params.get("search_terms", [])
            return self._format_search_results(filtered_messages, search_terms, user_display_names)
        
        else:  # "recent" or "history" mode
            # Standard chronological format
            return self._format_chronological(filtered_messages, user_display_names, bot_user_id)
    
    def _format_summary(self, 
                       messages: List[Dict[str, Any]], 
                       user_display_names: Dict[str, str]) -> str:
        """Format messages as a condensed summary."""
        # Group messages by user
        user_messages = {}
        for msg in messages:
            user_id = msg.get("user") or msg.get("bot_id", "unknown")
            text = msg.get("text", "").strip()
            if not text:
                continue
                
            display_name = user_display_names.get(user_id, f"User {user_id}")
            
            if display_name not in user_messages:
                user_messages[display_name] = []
            
            user_messages[display_name].append(text)
        
        # Format as a summary
        parts = ["=== CHANNEL SUMMARY ==="]
        
        for user, texts in user_messages.items():
            # Limit to 3 messages per user in summary
            sample = texts[:3]
            parts.append(f"{user} has sent {len(texts)} messages, including:")
            for text in sample:
                # Truncate long messages
                if len(text) > 100:
                    text = text[:97] + "..."
                parts.append(f"- \"{text}\"")
        
        parts.append("=== END CHANNEL SUMMARY ===")
        
        return "\n".join(parts)
    
    def _format_search_results(self,
                              messages: List[Dict[str, Any]],
                              search_terms: List[str],
                              user_display_names: Dict[str, str]) -> str:
        """Format messages as search results with term highlighting."""
        if not messages:
            return "=== NO RELEVANT MESSAGES FOUND ==="
        
        parts = ["=== RELEVANT MESSAGE MATCHES ==="]
        
        for msg in messages:
            user_id = msg.get("user") or msg.get("bot_id", "unknown")
            text = msg.get("text", "").strip()
            display_name = user_display_names.get(user_id, f"User {user_id}")
            
            # Skip empty messages
            if not text:
                continue
            
            # Highlight search terms (simplified)
            for term in search_terms:
                if term.lower() in text.lower():
                    # Find the term with original casing
                    start = text.lower().find(term.lower())
                    end = start + len(term)
                    term_original_case = text[start:end]
                    
                    # Replace with uppercase (not actually displayed to user)
                    text = text.replace(term_original_case, f"[{term_original_case}]")
            
            parts.append(f"{display_name}: {text}")
        
        parts.append("=== END RELEVANT MESSAGE MATCHES ===")
        
        return "\n".join(parts)
    
    def _format_chronological(self,
                             messages: List[Dict[str, Any]],
                             user_display_names: Dict[str, str],
                             bot_user_id: str) -> str:
        """Format messages in chronological order."""
        parts = ["=== MATCHED MESSAGES FROM CHANNEL HISTORY ==="]
        
        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda m: float(m.get("ts", "0")))
        
        for msg in sorted_messages:
            user_id = msg.get("user") or msg.get("bot_id", "unknown")
            text = msg.get("text", "").strip()
            
            # Skip empty messages
            if not text:
                continue
                
            display_name = user_display_names.get(user_id, f"User {user_id}")
            
            # Format differently for bot vs user
            if user_id == bot_user_id:
                parts.append(f"Bot: {text}")
            else:
                parts.append(f"{display_name}: {text}")
        
        parts.append("=== END CONVERSATION HISTORY ===")
        
        return "\n".join(parts)