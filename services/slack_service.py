from typing import Dict, Any, List, Optional, Callable
import re
import threading
from collections import defaultdict
from datetime import datetime
from loguru import logger
from slack_bolt import App
from slack_sdk.errors import SlackApiError

class SlackService:
    """Service for handling all Slack API interactions."""
    
    def __init__(self, bot_token: str, signing_secret: str, app_token: str):
        """Initialize the Slack service."""
        self.bot_token = bot_token
        self.signing_secret = signing_secret
        self.app_token = app_token
        
        self.is_dummy = False
        self.app = None
        self.bot_user_id = None
        
        # Shared state
        self.channel_data = {}
        self.emoji_tally = defaultdict(int)
        self.bot_message_timestamps = set()
        self.user_info_cache = {}
        
        # Initialize the app
        self._initialize_app()
    
    def _initialize_app(self):
        """Initialize the Slack Bolt app."""
        if not self.bot_token or not self.signing_secret:
            logger.warning("Missing Slack credentials — using DummyApp.")
            self.is_dummy = True
            
            class DummyApp:
                def __init__(self): 
                    self.client = None
                def event(self, *args, **kwargs): 
                    return lambda f: f
                def error(self, f): 
                    return f
                def message(self, *args, **kwargs): 
                    return lambda f: f
                def reaction_added(self, *args, **kwargs): 
                    return lambda f: f
            
            self.app = DummyApp()
            return
        
        try:
            self.app = App(token=self.bot_token, signing_secret=self.signing_secret)
            logger.info("Slack app initialized successfully.")
            
            # Get bot user ID
            auth_test = self.app.client.auth_test()
            self.bot_user_id = auth_test.get("user_id")
            logger.info(f"Bot User ID: {self.bot_user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Slack app: {e}")
            self.is_dummy = True
            self.app = None
    
    def is_available(self) -> bool:
        """Check if the Slack service is available."""
        return not self.is_dummy and self.app is not None
    
    def start_socket_mode(self):
        """Start the Slack app in Socket Mode."""
        if self.is_dummy or not self.app or not self.app_token:
            logger.warning("Cannot start Socket Mode with dummy app or missing app token.")
            return
            
        def socket_mode_runner():
            from slack_bolt.adapter.socket_mode import SocketModeHandler
            logger.info("Starting Slack SocketModeHandler...")
            handler = SocketModeHandler(self.app, self.app_token)
            handler.start()
        
        # Start in a new thread
        thread = threading.Thread(target=socket_mode_runner, daemon=True)
        thread.start()
        logger.info("Slack Socket Mode thread started.")
    
    def clean_prompt_text(self, text: str) -> str:
        """Remove bot mention from the text."""
        if self.bot_user_id:
            bot_mention_pattern = f"<@{self.bot_user_id}>"
            return re.sub(bot_mention_pattern, "", text).strip()
        return text.strip()
    
    def add_reaction(self, channel_id: str, timestamp: str, emoji: str) -> bool:
        """Add a reaction to a message.
        
        Args:
            channel_id: The channel ID
            timestamp: The message timestamp
            emoji: The emoji name (without colons)
            
        Returns:
            True if successful, False otherwise
        """
        if self.is_dummy:
            logger.info(f"Would add reaction :{emoji}: to message {timestamp} in {channel_id}")
            return True
            
        try:
            self.app.client.reactions_add(
                channel=channel_id,
                timestamp=timestamp,
                name=emoji
            )
            logger.debug(f"Added reaction :{emoji}: to message {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Failed to add reaction :{emoji}: to message {timestamp}: {e}")
            return False

    def remove_reaction(self, channel_id: str, timestamp: str, emoji: str) -> bool:
        """Remove a reaction from a message.
        
        Args:
            channel_id: The channel ID
            timestamp: The message timestamp
            emoji: The emoji name (without colons)
            
        Returns:
            True if successful, False otherwise
        """
        if self.is_dummy:
            logger.info(f"Would remove reaction :{emoji}: from message {timestamp} in {channel_id}")
            return True
            
        try:
            self.app.client.reactions_remove(
                channel=channel_id,
                timestamp=timestamp,
                name=emoji
            )
            logger.debug(f"Removed reaction :{emoji}: from message {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove reaction :{emoji}: from message {timestamp}: {e}")
            return False

    def send_ephemeral_message(self, channel_id: str, user_id: str, text: str) -> bool:
        """Send an ephemeral message visible only to the user."""
        if self.is_dummy:
            logger.info(f"Would send ephemeral message to {user_id} in {channel_id}: {text}")
            return True
            
        try:
            self.app.client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=text
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send ephemeral message: {e}")
            return False
    
    def send_message(self, channel_id: str, text: str, thread_ts: Optional[str] = None) -> Dict[str, Any]:
        """Send a message to a channel or thread."""
        if self.is_dummy:
            logger.info(f"Would send message to {channel_id}: {text}")
            return {"ok": True, "ts": "dummy_ts"}
            
        try:
            logger.info(f"[Slack] Sending message to channel {channel_id} (length: {len(text)} chars):\n{text}")

            response = self.app.client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts
            )

            logger.info(f"[Slack] Slack API response: {response}")

            # Track bot message
            if response.get("ok") and response.get("ts"):
                self.bot_message_timestamps.add(response["ts"])
            
            return response
        except Exception as e:
            logger.error(f"[Slack POST] Channel: {channel_id}, Text Length: {len(text)}, Text Preview: {text[:200]}")
            logger.error(f"Failed to send message: {e}")
            return {"ok": False, "error": str(e)}
    
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get information about a Slack user."""
        if user_id in self.user_info_cache:
            return self.user_info_cache[user_id]
            
        if self.is_dummy:
            dummy_info = {"user": {"id": user_id, "real_name": f"User {user_id}", "profile": {"display_name": f"User {user_id}"}}}
            self.user_info_cache[user_id] = dummy_info
            return dummy_info
            
        try:
            user_info = self.app.client.users_info(user=user_id)
            self.user_info_cache[user_id] = user_info
            return user_info
        except Exception as e:
            logger.error(f"Failed to get user info for {user_id}: {e}")
            # Return a fallback
            fallback = {"user": {"id": user_id, "real_name": f"User {user_id}", "profile": {"display_name": f"User {user_id}"}}}
            self.user_info_cache[user_id] = fallback
            return fallback
    
    def get_user_display_name(self, user_id: str) -> str:
        """Get a user's display name for inclusion in conversation history."""
        user_info = self.get_user_info(user_id)
        return (
            user_info.get("user", {}).get("profile", {}).get("display_name") or
            user_info.get("user", {}).get("real_name") or
            f"User {user_id}"
        )
    
    def fetch_channel_history(self, channel_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch the message history of a channel."""
        if self.is_dummy:
            logger.info(f"Would fetch channel history for {channel_id}")
            return []
            
        all_messages = []
        cursor = None
        
        logger.info(f"Fetching channel history for {channel_id} (limit: {limit})...")
        
        while True:
            try:
                result = self.app.client.conversations_history(
                    channel=channel_id,
                    limit=min(200, limit - len(all_messages)),
                    cursor=cursor
                )
                
                messages = result.get("messages", [])
                all_messages.extend(messages)
                
                cursor = result.get("response_metadata", {}).get("next_cursor")
                if not cursor or len(all_messages) >= limit:
                    break
            except Exception as e:
                logger.error(f"Failed to fetch channel history for {channel_id}: {e}")
                break
        
        logger.info(f"Fetched {len(all_messages)} messages from channel {channel_id}.")
        return all_messages[:limit]
    
    def fetch_thread_history(self, channel_id: str, thread_ts: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch the message history of a thread."""
        if self.is_dummy:
            logger.info(f"Would fetch thread history for {channel_id}, ts: {thread_ts}")
            return []
            
        all_replies = []
        cursor = None
        
        logger.info(f"Fetching thread history for {channel_id}, ts: {thread_ts} (limit: {limit})...")
        
        while True:
            try:
                result = self.app.client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    limit=min(200, limit - len(all_replies)),
                    cursor=cursor
                )
                
                replies = result.get("messages", [])
                all_replies.extend(replies)
                
                cursor = result.get("response_metadata", {}).get("next_cursor")
                if not cursor or len(all_replies) >= limit:
                    break
            except Exception as e:
                logger.error(f"Failed to fetch thread history for {thread_ts}: {e}")
                break
        
        logger.info(f"Fetched {len(all_replies)} messages from thread {thread_ts}.")
        return all_replies[:limit]
    
    def update_channel_stats(self, channel_id: str, user_id: str, message_ts: str) -> None:
        """Update statistics for a channel."""
        if channel_id not in self.channel_data:
            self.channel_data[channel_id] = {
                "message_count": 0,
                "participants": set(),
                "last_updated": datetime.now()
            }
        
        self.channel_data[channel_id]["message_count"] += 1
        self.channel_data[channel_id]["participants"].add(user_id)
        self.channel_data[channel_id]["last_updated"] = datetime.now()
    
    def get_channel_stats(self, channel_id: str) -> Dict[str, Any]:
        """Get statistics for a channel."""
        if channel_id not in self.channel_data:
            return {
                "message_count": 0,
                "participants": set(),
                "last_updated": datetime.now()
            }
        return self.channel_data[channel_id]
    
    def get_team_info(self) -> Dict[str, Any]:
        """Get team information including domain for permalink generation."""
        if self.is_dummy:
            return {"team": {"domain": "dummy-team"}}
        
        try:
            return self.app.client.team_info()
        except Exception as e:
            logger.error(f"Failed to get team info: {e}")
            return {"team": {"domain": "unknown-team"}}

    def generate_message_permalink(self, channel_id: str, message_ts: str) -> str:
        """
        Generate a proper Slack permalink for a message.
        
        Args:
            channel_id: The channel ID
            message_ts: The message timestamp
            
        Returns:
            Slack permalink URL
        """
        if self.is_dummy:
            return f"https://dummy-team.slack.com/archives/{channel_id}/p{message_ts.replace('.', '')}"
        
        try:
            # Use Slack's official permalink API
            response = self.app.client.chat_getPermalink(
                channel=channel_id,
                message_ts=message_ts
            )
            
            if response.get("ok") and response.get("permalink"):
                return response["permalink"]
            else:
                logger.warning(f"Failed to get permalink from Slack API: {response}")
                # Fallback to manual construction
                return self._construct_permalink_fallback(channel_id, message_ts)
                
        except Exception as e:
            logger.error(f"Error generating permalink: {e}")
            return self._construct_permalink_fallback(channel_id, message_ts)

    def _construct_permalink_fallback(self, channel_id: str, message_ts: str) -> str:
        """Fallback method to construct permalink when API fails."""
        team_info = self.get_team_info()
        team_domain = team_info.get("team", {}).get("domain", "your-team")
        ts_for_url = message_ts.replace('.', '')
        return f"https://{team_domain}.slack.com/archives/{channel_id}/p{ts_for_url}"

    def get_channel_name(self, channel_id: str) -> str:
        """Get the human-readable channel name."""
        if self.is_dummy:
            return f"channel-{channel_id}"
        
        try:
            response = self.app.client.conversations_info(channel=channel_id)
            if response.get("ok") and response.get("channel"):
                return response["channel"].get("name", channel_id)
        except Exception as e:
            logger.error(f"Failed to get channel name for {channel_id}: {e}")
        
        return channel_id

    def send_rich_message(self, channel_id: str, blocks: List[Dict[str, Any]], text: str, thread_ts: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a rich message with blocks for better formatting.
        
        Args:
            channel_id: The channel ID
            blocks: Slack blocks for rich formatting
            text: Fallback text for notifications
            thread_ts: Optional thread timestamp
            
        Returns:
            Slack API response
        """
        if self.is_dummy:
            logger.info(f"Would send rich message to {channel_id}: {text}")
            return {"ok": True, "ts": "dummy_ts"}
        
        try:
            response = self.app.client.chat_postMessage(
                channel=channel_id,
                blocks=blocks,
                text=text,  # Fallback text for notifications
                thread_ts=thread_ts
            )
            
            if response.get("ok") and response.get("ts"):
                self.bot_message_timestamps.add(response["ts"])
            
            return response
        except Exception as e:
            logger.error(f"Failed to send rich message: {e}")
            # Fallback to plain text
            return self.send_message(channel_id, text, thread_ts)