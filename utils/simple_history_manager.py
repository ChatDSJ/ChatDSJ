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
        Get recent history from channel and thread (if applicable).
        Simple, no filtering, no search logic.
        
        Args:
            slack_service: Slack service instance
            channel_id: Channel ID
            thread_ts: Thread timestamp (if in a thread)
            limit: Maximum messages to fetch
            
        Returns:
            List of messages sorted chronologically
        """
        if limit is None:
            limit = self.DEFAULT_LIMIT
            
        logger.info(f"Fetching recent history for channel {channel_id}, thread_ts: {thread_ts}, limit: {limit}")
        
        all_messages = []
        
        # Get channel history
        try:
            channel_messages = await asyncio.to_thread(
                slack_service.fetch_channel_history,
                channel_id,
                limit
            )
            all_messages.extend(channel_messages)
            logger.info(f"Retrieved {len(channel_messages)} channel messages")
        except Exception as e:
            logger.error(f"Error fetching channel history: {e}")
        
        # If in a thread, also get thread history
        if thread_ts:
            try:
                thread_messages = await asyncio.to_thread(
                    slack_service.fetch_thread_history,
                    channel_id,
                    thread_ts,
                    self.THREAD_LIMIT
                )
                
                # Add thread messages that aren't already in channel messages
                channel_timestamps = {msg.get("ts") for msg in all_messages}
                for msg in thread_messages:
                    if msg.get("ts") not in channel_timestamps:
                        all_messages.append(msg)
                
                logger.info(f"Added {len(thread_messages)} thread messages ({len([m for m in thread_messages if m.get('ts') not in channel_timestamps])} new)")
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