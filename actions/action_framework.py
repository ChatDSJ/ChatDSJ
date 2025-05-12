from typing import Dict, Any, List, Optional, Tuple, Union, Protocol
import re
import asyncio
from abc import ABC, abstractmethod
from loguru import logger
from pydantic import BaseModel, Field

class ServiceContainer:
    """
    Container for all service dependencies required by actions.
    
    This class provides a central place to access all services needed by the
    various actions in the system, ensuring consistent access and initialization.
    """
    
    def __init__(
        self,
        slack_service=None,
        notion_service=None,
        openai_service=None,
        web_service=None,
        youtube_service=None
    ):
        """
        Initialize the service container with available services.
        
        Args:
            slack_service: Service for Slack interactions
            notion_service: Service for Notion interactions
            openai_service: Service for OpenAI interactions
            web_service: Service for web content retrieval
            youtube_service: Service for YouTube content retrieval
        """
        self.slack_service = slack_service
        self.notion_service = notion_service
        self.openai_service = openai_service
        self.web_service = web_service
        self.youtube_service = youtube_service
        
        logger.info("ServiceContainer initialized with available services")
    
    def validate_required_services(self, service_names: List[str]) -> bool:
        """
        Validate that required services are available.
        
        Args:
            service_names: List of service names to validate
            
        Returns:
            True if all required services are available, False otherwise
        """
        missing_services = []
        
        for name in service_names:
            service = getattr(self, f"{name}_service", None)
            if service is None:
                missing_services.append(name)
                continue
                
            # If the service has an is_available method, check it
            if hasattr(service, "is_available") and callable(service.is_available):
                if not service.is_available():
                    missing_services.append(name)
        
        if missing_services:
            logger.warning(f"Required services not available: {', '.join(missing_services)}")
            return False
            
        return True

class ActionRequest(BaseModel):
    """
    Base model for action requests with common fields.
    
    This defines the common structure for all action requests, which can be
    extended by specific action types with additional fields.
    """
    channel_id: str = Field(..., description="Slack channel ID")
    user_id: str = Field(..., description="Slack user ID")
    message_ts: str = Field(..., description="Slack message timestamp")
    thread_ts: Optional[str] = Field(None, description="Slack thread timestamp")
    text: str = Field(..., description="Full message text")
    prompt: str = Field(..., description="Cleaned message text (without mentions)")

class ActionResponse(BaseModel):
    """
    Base model for action responses with common fields.
    
    This defines the common structure for all action responses, which can be
    extended by specific action types with additional fields.
    """
    success: bool = Field(..., description="Whether the action was successful")
    message: Optional[str] = Field(None, description="Response message to send to Slack")
    thread_ts: Optional[str] = Field(None, description="Thread timestamp for the response")
    error: Optional[str] = Field(None, description="Error message if not successful")

class Action(ABC):
    """
    Base class for all actions in the system.
    
    Actions are the core building blocks of the system's behavior, encapsulating
    specific tasks like responding to messages, processing URLs, etc.
    """
    
    def __init__(self, services: ServiceContainer):
        """
        Initialize the action with service dependencies.
        
        Args:
            services: Container with all available services
        """
        self.services = services
        self.name = self.__class__.__name__
        logger.debug(f"Action {self.name} initialized")
    
    @abstractmethod
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """
        Execute the action asynchronously.
        
        Args:
            request: The action request parameters
            
        Returns:
            Response from the action execution
        """
        pass
    
    @abstractmethod
    def can_handle(self, text: str) -> bool:
        """
        Check if this action can handle the given text.
        
        Args:
            text: The message text to check
            
        Returns:
            True if this action can handle the text, False otherwise
        """
        pass
    
    def get_required_services(self) -> List[str]:
        """
        Get the list of services required by this action.
        
        Returns:
            List of service names required by this action
        """
        return ["slack"]  # By default, all actions need the slack service

class ContextResponseAction(Action):
    """
    Action for generating a contextual response to a user message.
    
    This is the default action that processes regular chat messages and
    generates responses based on conversation history and user context.
    """
    
    def get_required_services(self) -> List[str]:
        """Get required services for context response."""
        return ["slack", "openai", "notion"]
    
    def can_handle(self, text: str) -> bool:
        """
        This is the default action, so it can handle any text.
        
        Args:
            text: The message text
            
        Returns:
            Always True, as this is the fallback action
        """
        return True
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """
        Execute the context response action.
        
        Args:
            request: The action request
            
        Returns:
            Action response with the generated message
        """
        try:
            # Validate required services
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Determine if this is a new channel question or in a thread
            is_new_main_channel_question = request.thread_ts is None
            
            # Fetch appropriate history based on the context
            if is_new_main_channel_question:
                # Check if asking about past history
                history_query_keywords = [
                    "discussed", "discussion", "mentioned", "talked about", "said about",
                    "summarize", "summary", "what was said", "history of", "previously on"
                ]
                is_querying_history = any(keyword.lower() in request.prompt.lower() for keyword in history_query_keywords)
                
                # If asking about history, fetch more messages
                limit = 1000 if is_querying_history else 50
                channel_history = await asyncio.to_thread(
                    self.services.slack_service.fetch_channel_history,
                    request.channel_id, 
                    limit
                )
                thread_history = []
            else:
                # This is in a thread
                channel_history = await asyncio.to_thread(
                    self.services.slack_service.fetch_channel_history,
                    request.channel_id, 
                    1000
                )
                thread_history = await asyncio.to_thread(
                    self.services.slack_service.fetch_thread_history,
                    request.channel_id, 
                    request.thread_ts,
                    1000
                )
            
            # Merge and deduplicate history
            merged_messages = []
            if thread_history:
                thread_message_timestamps = {msg["ts"] for msg in thread_history}
                merged_messages.extend(thread_history)
                # Add channel messages not already in the thread history
                for msg in channel_history:
                    if msg["ts"] not in thread_message_timestamps:
                        merged_messages.append(msg)
            else:
                merged_messages = channel_history
            
            # Build user display names dictionary for formatting
            user_display_names = {}
            for msg in merged_messages:
                user_id = msg.get("user") or msg.get("bot_id")
                if user_id and user_id not in user_display_names:
                    user_display_names[user_id] = await asyncio.to_thread(
                        self.services.slack_service.get_user_display_name,
                        user_id
                    )
            
            # Format history for OpenAI
            formatted_history = await asyncio.to_thread(
                self.services.openai_service._format_conversation_for_openai,
                merged_messages,
                user_display_names,
                self.services.slack_service.bot_user_id
            )
            
            # Get user-specific context from Notion
            user_page_content = await asyncio.to_thread(
                self.services.notion_service.get_user_page_content,
                request.user_id
            )
            user_preferred_name = await asyncio.to_thread(
                self.services.notion_service.get_user_preferred_name,
                request.user_id
            )
            
            # Construct user context
            user_context_parts = []
            if user_preferred_name:
                user_context_parts.append(f"The user's preferred name is: {user_preferred_name}.")
            
            if user_page_content and user_page_content.strip():
                user_context_parts.append(
                    f"Other known facts and preferences for this user:\n{user_page_content.strip()}"
                )
            
            user_specific_context = "\n".join(user_context_parts) if user_context_parts else None
            
            # Try to classify and store memory if applicable
            from services.memory_handler import handle_memory_instruction

            memory_response = await asyncio.to_thread(
                handle_memory_instruction,
                request.user_id,
                request.prompt,
                self.services.notion_service  # Make sure this is the CachedNotionService instance
            )

            if memory_response:
                return ActionResponse(
                    success=True,
                    message=memory_response,
                    thread_ts=request.thread_ts
                )

            # Generate response
            response_text, usage = await self.services.openai_service.get_completion_async(
                prompt=request.prompt,
                conversation_history=formatted_history,
                user_specific_context=user_specific_context
            )
            
            if not response_text:
                return ActionResponse(
                    success=False,
                    message="I'm sorry, I couldn't generate a response for that.",
                    thread_ts=request.thread_ts
                )
            
            return ActionResponse(
                success=True,
                message=response_text,
                thread_ts=request.thread_ts
            )
            
        except Exception as e:
            logger.error(f"Error in ContextResponseAction: {e}", exc_info=True)
            return ActionResponse(
                success=False,
                error=str(e),
                message="I encountered an error while processing your request.",
                thread_ts=request.thread_ts
            )

class RetrieveSummarizeAction(Action):
    """
    Action for retrieving and summarizing web content.
    
    This action handles URLs in messages, fetches the content, and generates
    a summary that is stored in Notion for future reference.
    """
    
    def get_required_services(self) -> List[str]:
        """Get required services for web content retrieval and summarization."""
        return ["slack", "openai", "notion", "web"]
    
    def can_handle(self, text: str) -> bool:
        """
        Check if the text contains a web URL.
        
        Args:
            text: The message text
            
        Returns:
            True if the text contains a web URL, False otherwise
        """
        # URL detection pattern (simple version)
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return bool(re.search(url_pattern, text)) and "youtube.com" not in text and "youtu.be" not in text
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """
        Execute the retrieve and summarize action.
        
        Args:
            request: The action request
            
        Returns:
            Action response with the summary
        """
        try:
            # Validate required services
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Extract URL from text
            url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
            match = re.search(url_pattern, request.text)
            
            if not match:
                return ActionResponse(
                    success=False,
                    error="No URL found in message",
                    message="I couldn't find a valid URL in your message.",
                    thread_ts=request.thread_ts
                )
            
            url = match.group(0)
            
            # Fetch web content
            content = await self.services.web_service.fetch_content(url)
            
            if not content:
                return ActionResponse(
                    success=False,
                    error=f"Failed to fetch content from {url}",
                    message=f"I couldn't fetch content from {url}. The site may be unavailable or blocking access.",
                    thread_ts=request.thread_ts
                )
            
            # Generate summary
            summary_prompt = f"Please summarize the following web content from {url}:\n\n{content[:10000]}..."
            summary, _ = await self.services.openai_service.get_completion_async(
                prompt=summary_prompt,
                max_tokens=500
            )
            
            if not summary:
                return ActionResponse(
                    success=False,
                    error="Failed to generate summary",
                    message="I was able to fetch the content, but couldn't generate a summary.",
                    thread_ts=request.thread_ts
                )
            
            # Create a Notion page with the summary
            notion_page_id = await asyncio.to_thread(
                self.services.notion_service.create_content_page,
                f"Summary of {url}",
                f"# Summary of {url}\n\n{summary}\n\n## Full Content\n\n{content[:20000]}..."
            )
            
            # Generate a shorter summary for Slack
            mini_summary_prompt = f"Please create a very short summary (2-3 sentences) of this content from {url}:\n\n{summary}"
            mini_summary, _ = await self.services.openai_service.get_completion_async(
                prompt=mini_summary_prompt,
                max_tokens=100
            )
            
            # Construct response message
            response_message = (
                f"*Summary of {url}*\n\n{mini_summary}\n\n"
                f"I've saved a more detailed summary in Notion: {self.services.notion_service.get_page_url(notion_page_id)}"
            )
            
            return ActionResponse(
                success=True,
                message=response_message,
                thread_ts=request.thread_ts
            )
            
        except Exception as e:
            logger.error(f"Error in RetrieveSummarizeAction: {e}", exc_info=True)
            return ActionResponse(
                success=False,
                error=str(e),
                message="I encountered an error while trying to retrieve and summarize that URL.",
                thread_ts=request.thread_ts
            )

class YoutubeSummarizeAction(Action):
    """
    Action for retrieving and summarizing YouTube videos.
    
    This action handles YouTube URLs, fetches the transcript, and generates
    a summary that is stored in Notion for future reference.
    """
    
    def get_required_services(self) -> List[str]:
        """Get required services for YouTube content retrieval and summarization."""
        return ["slack", "openai", "notion", "youtube"]
    
    def can_handle(self, text: str) -> bool:
        """
        Check if the text contains a YouTube URL.
        
        Args:
            text: The message text
            
        Returns:
            True if the text contains a YouTube URL, False otherwise
        """
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://(?:www\.)?youtu\.be/[\w-]+'
        ]
        return any(bool(re.search(pattern, text)) for pattern in youtube_patterns)
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """
        Execute the YouTube summarize action.
        
        Args:
            request: The action request
            
        Returns:
            Action response with the summary
        """
        try:
            # Validate required services
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Extract YouTube URL from text
            youtube_patterns = [
                r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
                r'https?://(?:www\.)?youtu\.be/[\w-]+'
            ]
            
            youtube_url = None
            for pattern in youtube_patterns:
                match = re.search(pattern, request.text)
                if match:
                    youtube_url = match.group(0)
                    break
            
            if not youtube_url:
                return ActionResponse(
                    success=False,
                    error="No YouTube URL found in message",
                    message="I couldn't find a valid YouTube URL in your message.",
                    thread_ts=request.thread_ts
                )
            
            # Fetch video info and transcript
            video_info = await self.services.youtube_service.get_video_info(youtube_url)
            transcript = await self.services.youtube_service.get_transcript(youtube_url)
            
            if not transcript:
                return ActionResponse(
                    success=False,
                    error=f"Failed to fetch transcript for {youtube_url}",
                    message=f"I couldn't fetch a transcript for {youtube_url}. The video may not have captions available.",
                    thread_ts=request.thread_ts
                )
            
            # Generate summary
            summary_prompt = (
                f"Please summarize the following YouTube video transcript:\n\n"
                f"Title: {video_info.get('title', 'Unknown')}\n"
                f"Channel: {video_info.get('channel', 'Unknown')}\n"
                f"Transcript:\n{transcript[:10000]}..."
            )
            
            summary, _ = await self.services.openai_service.get_completion_async(
                prompt=summary_prompt,
                max_tokens=500
            )
            
            if not summary:
                return ActionResponse(
                    success=False,
                    error="Failed to generate summary",
                    message="I was able to fetch the transcript, but couldn't generate a summary.",
                    thread_ts=request.thread_ts
                )
            
            # Create a Notion page with the summary and transcript
            notion_page_id = await asyncio.to_thread(
                self.services.notion_service.create_content_page,
                f"Summary of YouTube: {video_info.get('title', youtube_url)}",
                (
                    f"# Summary of YouTube Video\n\n"
                    f"**Title:** {video_info.get('title', 'Unknown')}\n"
                    f"**Channel:** {video_info.get('channel', 'Unknown')}\n"
                    f"**URL:** {youtube_url}\n\n"
                    f"## Summary\n\n{summary}\n\n"
                    f"## Full Transcript\n\n{transcript[:20000]}..."
                )
            )
            
            # Generate a shorter summary for Slack
            mini_summary_prompt = (
                f"Please create a very short summary (2-3 sentences) of this YouTube video:\n\n"
                f"Title: {video_info.get('title', 'Unknown')}\n"
                f"Summary: {summary}"
            )
            
            mini_summary, _ = await self.services.openai_service.get_completion_async(
                prompt=mini_summary_prompt,
                max_tokens=100
            )
            
            # Construct response message
            response_message = (
                f"*Summary of YouTube Video: {video_info.get('title', 'Unknown')}*\n\n"
                f"{mini_summary}\n\n"
                f"I've saved the full transcript and a detailed summary in Notion: "
                f"{self.services.notion_service.get_page_url(notion_page_id)}"
            )
            
            return ActionResponse(
                success=True,
                message=response_message,
                thread_ts=request.thread_ts
            )
            
        except Exception as e:
            logger.error(f"Error in YoutubeSummarizeAction: {e}", exc_info=True)
            return ActionResponse(
                success=False,
                error=str(e),
                message="I encountered an error while trying to process that YouTube video.",
                thread_ts=request.thread_ts
            )

class TodoAction(Action):
    """
    Action for adding TODO items to a user's Notion page.
    
    This action handles messages with TODO: prefixes and adds the content
    to the user's TODO list in Notion.
    """
    
    def get_required_services(self) -> List[str]:
        """Get required services for TODO action."""
        return ["slack", "notion"]
    
    def can_handle(self, text: str) -> bool:
        """
        Check if the text contains a TODO command.
        
        Args:
            text: The message text
            
        Returns:
            True if the text contains a TODO command, False otherwise
        """
        return bool(re.search(r'^TODO:', text, re.IGNORECASE)) or bool(re.search(r'\bTODO:', text, re.IGNORECASE))
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """
        Execute the TODO action.
        
        Args:
            request: The action request
            
        Returns:
            Action response with confirmation message
        """
        try:
            # Validate required services
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Extract TODO content
            todo_match = re.search(r'(?:^|\b)TODO:\s*(.*?)(?:$|$)', request.text, re.IGNORECASE)
            
            if not todo_match:
                return ActionResponse(
                    success=False,
                    error="No TODO content found",
                    message="I couldn't find any TODO content in your message.",
                    thread_ts=request.thread_ts
                )
            
            todo_text = todo_match.group(1).strip()
            
            if not todo_text:
                return ActionResponse(
                    success=False,
                    error="Empty TODO content",
                    message="I found a TODO marker, but there was no content after it.",
                    thread_ts=request.thread_ts
                )
            
            # Add to user's TODO list in Notion
            success = await asyncio.to_thread(
                self.services.notion_service.add_todo_item,
                request.user_id,
                todo_text
            )
            
            if not success:
                return ActionResponse(
                    success=False,
                    error="Failed to add TODO item",
                    message="I couldn't add that TODO item to your list. Please try again later.",
                    thread_ts=request.thread_ts
                )
            
            return ActionResponse(
                success=True,
                message=f"✅ Added to your TODO list: {todo_text}",
                thread_ts=request.thread_ts
            )
            
        except Exception as e:
            logger.error(f"Error in TodoAction: {e}", exc_info=True)
            return ActionResponse(
                success=False,
                error=str(e),
                message="I encountered an error while trying to add that TODO item.",
                thread_ts=request.thread_ts
            )

class ActionRouter:
    """
    Router for determining and executing the appropriate action for a message.
    
    This class is responsible for analyzing messages, determining which action
    should handle them, and delegating to the appropriate action handler.
    """
    
    def __init__(self, services: ServiceContainer):
        """
        Initialize the action router with services and actions.
        
        Args:
            services: Container with all available services
        """
        self.services = services
        
        # Register all available actions
        self.actions = [
            TodoAction(services),
            YoutubeSummarizeAction(services),
            RetrieveSummarizeAction(services),
            ContextResponseAction(services)  # Always last as fallback
        ]
        
        logger.info(f"ActionRouter initialized with {len(self.actions)} actions")
    
    def _get_action_for_text(self, text: str) -> Action:
        """
        Determine which action should handle the given text.
        
        Args:
            text: The message text
            
        Returns:
            The appropriate action handler
        """
        for action in self.actions:
            if action.can_handle(text):
                logger.info(f"Selected action {action.name} for message: {text[:50]}...")
                return action
        
        # Should never reach here because ContextResponseAction handles everything
        # But just in case, return the last action (ContextResponseAction)
        return self.actions[-1]
    
    async def route_action(self, request: ActionRequest) -> ActionResponse:
        """
        Route the request to the appropriate action handler.
        
        Args:
            request: The action request
            
        Returns:
            Response from the action handler
        """
        # Check for nickname command as a special case
        if hasattr(self.services.notion_service, "handle_nickname_command"):
            nickname_response, nickname_success = await asyncio.to_thread(
                self.services.notion_service.handle_nickname_command,
                request.prompt,
                request.user_id,
                await asyncio.to_thread(
                    self.services.slack_service.get_user_display_name,
                    request.user_id
                )
            )
            
            if nickname_response:
                return ActionResponse(
                    success=nickname_success,
                    message=nickname_response,
                    thread_ts=request.thread_ts
                )
        
        # Determine the appropriate action and execute it
        action = self._get_action_for_text(request.text)
        return await action.execute(request)
    
    def get_available_actions(self) -> List[str]:
        """
        Get a list of available action names.
        
        Returns:
            List of action names
        """
        return [action.name for action in self.actions]

# Example of how to use the action framework
async def process_slack_mention(event: Dict[str, Any], services: ServiceContainer) -> Dict[str, Any]:
    """
    Process a Slack mention event using the action framework.
    
    Args:
        event: The Slack event object
        services: Service container
        
    Returns:
        The result of sending the message to Slack
    """
    # Extract key information
    channel_id = event["channel"]
    user_id = event["user"]
    message_ts = event["ts"]
    text = event.get("text", "")
    thread_ts = event.get("thread_ts")
    
    # Clean the prompt text (remove bot mention)
    prompt = services.slack_service.clean_prompt_text(text)
    
    # Send ephemeral acknowledgment
    await asyncio.to_thread(
        services.slack_service.send_ephemeral_message,
        channel_id,
        user_id,
        "I heard you! I'm working on a response... 🧠"
    )
    
    # Create action request
    request = ActionRequest(
        channel_id=channel_id,
        user_id=user_id,
        message_ts=message_ts,
        thread_ts=thread_ts,
        text=text,
        prompt=prompt
    )
    
    # Route and execute the appropriate action
    router = ActionRouter(services)
    response = await router.route_action(request)
    
    # Send the response to Slack
    if response.message:
        result = await asyncio.to_thread(
            services.slack_service.send_message,
            channel_id,
            response.message,
            response.thread_ts or thread_ts
        )
    else:
        result = await asyncio.to_thread(
            services.slack_service.send_message,
            channel_id,
            "I'm sorry, I couldn't process that request.",
            thread_ts
        )
    
    # Update channel stats
    await asyncio.to_thread(
        services.slack_service.update_channel_stats,
        channel_id,
        user_id,
        message_ts
    )
    
    return result