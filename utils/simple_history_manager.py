from typing import Dict, List, Any, Optional
import asyncio
from loguru import logger

class SimpleHistoryManager:
    """
    Simplified history manager that just fetches recent channel/thread history.
    NO SEARCH LOGIC, NO FILTERING, NO KEYWORD MATCHING.
    """
    
    def __init__(self):
        # Simple constants
        self.DEFAULT_LIMIT = 200
        self.THREAD_LIMIT = 100
    
    async def get_recent_history(self,
                            slack_service,
                            channel_id: str,
                            thread_ts: Optional[str],
                            limit: int = None) -> List[Dict[str, Any]]:
        """
        UPDATED: Get more comprehensive history, especially for threads.
        """
        if limit is None:
            # INCREASED LIMITS: Get more messages for better context
            limit = 1000 if thread_ts else 500  # More for threads, they're conversations
            
        logger.info(f"Fetching history for channel {channel_id}, thread_ts: {thread_ts}, limit: {limit}")
        
        all_messages = []
        
        # Get channel history (but less if we're in a thread)
        try:
            channel_limit = 200 if thread_ts else limit  # Fewer channel messages if in thread
            channel_messages = await asyncio.to_thread(
                slack_service.fetch_channel_history,
                channel_id,
                channel_limit
            )
            all_messages.extend(channel_messages)
            logger.info(f"Retrieved {len(channel_messages)} channel messages")
        except Exception as e:
            logger.error(f"Error fetching channel history: {e}")
        
        # If in a thread, get ALL thread messages (threads are conversations)
        if thread_ts:
            try:
                thread_messages = await asyncio.to_thread(
                    slack_service.fetch_thread_history,
                    channel_id,
                    thread_ts,
                    500  # Get lots of thread messages - it's a conversation!
                )
                
                # Add thread messages that aren't already in channel messages
                channel_timestamps = {msg.get("ts") for msg in all_messages}
                new_thread_messages = 0
                for msg in thread_messages:
                    if msg.get("ts") not in channel_timestamps:
                        all_messages.append(msg)
                        new_thread_messages += 1
                
                logger.info(f"Added {len(thread_messages)} thread messages ({new_thread_messages} new)")
            except Exception as e:
                logger.error(f"Error fetching thread history: {e}")
        
        # Remove duplicates and sort chronologically
        unique_messages = {}
        for msg in all_messages:
            ts = msg.get("ts")
            if ts and ts not in unique_messages:
                unique_messages[ts] = msg
        
        # Sort by timestamp
        sorted_messages = sorted(
            unique_messages.values(),
            key=lambda m: float(m.get("ts", "0"))
        )
        
        logger.info(f"Returning {len(sorted_messages)} total unique messages")
        return sorted_messages