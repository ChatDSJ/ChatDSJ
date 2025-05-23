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
        
        # Inject the NotionContextManager into OpenAIService if both services are available
        if self.notion_service and self.openai_service and not hasattr(self.openai_service, 'notion_context_manager'):
            from services.notion_parser import NotionContextManager
            self.openai_service.notion_context_manager = NotionContextManager()
        
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
        Execute the context response action with direct Notion context inclusion.
        Uses an approach similar to the previously working implementation.
        
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
            
            # Try to handle memory instruction first
            memory_response = await asyncio.to_thread(
                self.services.notion_service.handle_memory_instruction,
                request.user_id,
                request.prompt
            )

            if memory_response:
                return ActionResponse(
                    success=True,
                    message=memory_response,
                    thread_ts=request.thread_ts
                )
            
            # Import and initialize history manager
            from utils.history_manager import HistoryManager
            history_manager = HistoryManager()
            
            # Retrieve and filter history based on query type
            filtered_messages, query_params = await history_manager.retrieve_and_filter_history(
                self.services.slack_service,
                request.channel_id,
                request.thread_ts,
                request.prompt
            )
            
            # Build user display names dictionary
            user_display_names = {}
            for msg in filtered_messages:
                user_id = msg.get("user") or msg.get("bot_id")
                if user_id and user_id not in user_display_names:
                    user_display_names[user_id] = await asyncio.to_thread(
                        self.services.slack_service.get_user_display_name,
                        user_id
                    )
            
            # Format the filtered messages based on query type
            formatted_history_text = history_manager.format_history_for_prompt(
                filtered_messages,
                query_params,
                user_display_names,
                self.services.slack_service.bot_user_id
            )
            
            # Format history for OpenAI using standard method
            formatted_history = await asyncio.to_thread(
                self.services.openai_service._format_conversation_for_openai,
                filtered_messages,
                user_display_names,
                self.services.slack_service.bot_user_id
            )
            
            # Get user-specific context from Notion
            logger.info(f"Fetching Notion context for user {request.user_id}")
            
            # Get both page content and properties
            user_page_content = await asyncio.to_thread(
                self.services.notion_service.get_user_page_content,
                request.user_id
            )
            
            user_page_properties = await asyncio.to_thread(
                self.services.notion_service.get_user_page_properties,
                request.user_id
            )
            
            user_preferred_name = await asyncio.to_thread(
                self.services.notion_service.get_user_preferred_name,
                request.user_id
            )
            
            # Use enhanced context builder to emphasize preferences
            from utils.context_builder import get_enhanced_user_context
            user_specific_context = await asyncio.to_thread(
                get_enhanced_user_context,
                self.services.notion_service,
                request.user_id,
                ""  # No base prompt needed, it will be added in get_completion_async
            )
            
            # Log summary of chat history processing
            logger.info(f"History retrieval mode: {query_params.get('mode', 'unknown')}")
            logger.info(f"Retrieved {len(filtered_messages)} relevant messages")
            
            # Generate response using direct approach
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
            
            # Try to handle as a memory instruction first
            memory_response = await asyncio.to_thread(
                self.services.notion_service.handle_memory_instruction,
                request.user_id,
                request.prompt
            )
            
            if memory_response:
                return ActionResponse(
                    success=True,
                    message=memory_response,
                    thread_ts=request.thread_ts
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
        
        # Register all available actions - ORDER MATTERS!
        self.actions = [
            TodoAction(services),
            YoutubeSummarizeAction(services),
            RetrieveSummarizeAction(services),
            HistoricalSearchAction(services),
            SearchFollowUpAction(services),    
            ContextResponseAction(services)
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
        Route the request to the appropriate action handler with improved memory command handling.
        
        Args:
            request: The action request
            
        Returns:
            Response from the action handler
        """
        try:
            # First, try to identify and directly handle any memory operations
            if hasattr(self.services.notion_service, "memory_handler"):
                # Use the memory handler's classify method to check if this is a memory command
                memory_type, content = self.services.notion_service.memory_handler.classify_memory_instruction(request.prompt)
                
                if memory_type != "unknown":
                    logger.info(f"Identified memory command of type '{memory_type}' with content: '{content}'")
                    
                    # Handle deletion commands directly (bypassing OpenAI)
                    if memory_type == "delete_fact":
                        # Direct deletion execution
                        success = self.services.notion_service.memory_handler.delete_known_fact(request.user_id, content)
                        
                        # Get current facts to see what's still there
                        remaining_facts = self.services.notion_service.memory_handler.get_known_facts(request.user_id)
                        
                        if success:
                            # Check if any matching facts remain
                            matching_remaining = [f for f in remaining_facts if content.lower() in f.lower()]
                            
                            if matching_remaining:
                                # We thought we deleted it but matching facts still exist
                                facts_list = "\n".join([f"• {f}" for f in matching_remaining])
                                return ActionResponse(
                                    success=True,
                                    message=f"I deleted a fact containing '{content}', but I found these similar facts still remaining:\n\n{facts_list}\n\nWould you like me to delete these as well?",
                                    thread_ts=request.thread_ts
                                )
                            else:
                                # Success - no more matching facts
                                return ActionResponse(
                                    success=True,
                                    message=f"✅ Successfully removed the fact containing '{content}' from your Known Facts.",
                                    thread_ts=request.thread_ts
                                )
                        else:
                            # Failed to delete
                            if not remaining_facts:
                                return ActionResponse(
                                    success=False,
                                    message=f"You don't have any facts stored yet. Add facts by saying \"Remember [your fact]\" or \"Fact: [your fact]\".",
                                    thread_ts=request.thread_ts
                                )
                                
                            # Check if any matching facts exist at all
                            matching_facts = [f for f in remaining_facts if content.lower() in f.lower()]
                            
                            if matching_facts:
                                # We have matching facts but couldn't delete them
                                facts_list = "\n".join([f"• {f}" for f in matching_facts])
                                return ActionResponse(
                                    success=False,
                                    message=f"I found facts containing '{content}' but couldn't delete them. Please try again with a more specific text:\n\n{facts_list}",
                                    thread_ts=request.thread_ts
                                )
                            else:
                                # No matching facts found
                                return ActionResponse(
                                    success=False,
                                    message=f"I couldn't find any facts containing '{content}' to remove.",
                                    thread_ts=request.thread_ts
                                )
                    
                    # Handle other deletion commands similarly
                    elif memory_type == "delete_preference":
                        success = self.services.notion_service.memory_handler.delete_preference(request.user_id, content)
                        return ActionResponse(
                            success=success,
                            message=f"✅ Removed preference about '{content}'." if success else f"I couldn't find a preference about '{content}' to remove.",
                            thread_ts=request.thread_ts
                        )
                        
                    elif memory_type == "delete_project":
                        success = self.services.notion_service.memory_handler.delete_project(request.user_id, content)
                        return ActionResponse(
                            success=success,
                            message=f"✅ Removed project '{content}'." if success else f"I couldn't find a project '{content}' to remove.",
                            thread_ts=request.thread_ts
                        )
                    
                    # For listing operations, bypass OpenAI as well
                    elif memory_type in ["list_facts", "list_preferences", "list_projects"]:
                        response = self.services.notion_service.memory_handler.handle_memory_instruction(
                            request.user_id, request.prompt
                        )
                        if response:
                            return ActionResponse(
                                success=True,
                                message=response,
                                thread_ts=request.thread_ts
                            )
                    
                    # For other memory instructions, we can use the regular handler
                    else:
                        response = self.services.notion_service.memory_handler.handle_memory_instruction(
                            request.user_id, request.prompt
                        )
                        if response:
                            return ActionResponse(
                                success=True,
                                message=response,
                                thread_ts=request.thread_ts
                            )
            
            # If not handled by direct memory operations, continue with regular flow...
            
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
            
            # Check for memory commands
            if hasattr(self.services.notion_service, "memory_handler"):
                memory_response = await asyncio.to_thread(
                    self.services.notion_service.handle_memory_instruction,
                    request.user_id,
                    request.prompt
                )
                
                if memory_response:
                    return ActionResponse(
                        success=True,
                        message=memory_response,
                        thread_ts=request.thread_ts
                    )
            
            # Determine the appropriate action and execute it
            action = self._get_action_for_text(request.text)
            response = await action.execute(request)
            
            # If there was an error and it seems to be a memory-related command,
            # provide a helpful suggestion
            if not response.success and response.error:
                # Check if this looks like an attempted memory command
                memory_keywords = ["remember", "fact", "project", "preference", "todo", "delete", "remove", "list", "show"]
                
                if any(keyword in request.prompt.lower() for keyword in memory_keywords):
                    examples = await asyncio.to_thread(
                        self.services.notion_service.memory_handler.get_example_for_command,
                        request.prompt
                    )
                    
                    # Update the error message with the example
                    response.message = f"{response.message}\n\n{examples}"
            
            return response
        
        except Exception as e:
            logger.error(f"Error in route_action: {e}", exc_info=True)
            return ActionResponse(
                success=False,
                message=f"I encountered an error while processing your request. Please try again.",
                thread_ts=request.thread_ts
            )
    
class HistoricalSearchAction(Action):
    """
    Enhanced action for historical search queries with rich thread display.
    """
    
    def get_required_services(self) -> List[str]:
        """Get required services for historical search."""
        return ["slack", "openai"]
    
    def can_handle(self, text: str) -> bool:
        """
        Check if this is a historical search query.
        
        Args:
            text: The message text
            
        Returns:
            True if this is a search query, False otherwise
        """
        search_patterns = [
            r"(?:has|have|did)\s+anyone\s+(?:talk|discuss|mention)(?:ed)?\s+(?:about\s+)?",
            r"(?:was|were)\s+.*?\s+(?:discussed|mentioned|talked about)",
            r"(?:any|are there)\s+discussions?\s+(?:about|on|regarding)\s+",
            r"(?:find|search|show).*?(?:discussions?|conversations?|messages?)",
            r"when\s+(?:did|was).*?(?:discussed|mentioned)"
        ]
        
        for pattern in search_patterns:
            if re.search(pattern, text.lower()):
                return True
        return False
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """
        Execute enhanced historical search with rich thread display.
        
        Args:
            request: The action request
            
        Returns:
            Action response with rich search results
        """
        try:
            # Validate required services
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Import and initialize history manager
            from utils.history_manager import HistoryManager
            history_manager = HistoryManager()
            
            # Perform the search
            filtered_messages, query_params = await history_manager.retrieve_and_filter_history(
                self.services.slack_service,
                request.channel_id,
                request.thread_ts,
                request.prompt
            )
            
            # Build user display names dictionary
            user_display_names = {}
            for msg in filtered_messages:
                user_id = msg.get("user") or msg.get("bot_id")
                if user_id and user_id not in user_display_names:
                    user_display_names[user_id] = await asyncio.to_thread(
                        self.services.slack_service.get_user_display_name,
                        user_id
                    )
            
            # Format search results with thread information
            search_results = history_manager.format_search_results_with_threads(
                filtered_messages,
                query_params,
                user_display_names,
                self.services.slack_service.bot_user_id,
                request.channel_id,
                self.services.slack_service
            )
            
            # Build rich response message
            response_message = self.build_rich_search_response(search_results)
            
            return ActionResponse(
                success=True,
                message=response_message,
                thread_ts=request.thread_ts
            )
            
        except Exception as e:
            logger.error(f"Error in HistoricalSearchAction: {e}", exc_info=True)
            return ActionResponse(
                success=False,
                error=str(e),
                message="I encountered an error while searching through the message history.",
                thread_ts=request.thread_ts
            )
    
    def build_rich_search_response(self, search_results: Dict[str, Any]) -> str:
        """
        Build a rich response message with thread links and excerpts.
        
        Args:
            search_results: Formatted search results
            
        Returns:
            Rich response message
        """
        if not search_results.get("show_details"):
            return search_results.get("summary", "No results found.")
        
        message_parts = [search_results["summary"]]
        threads = search_results.get("threads", [])
        
        # Add thread summaries with links
        for i, thread in enumerate(threads[:5], 1):  # Limit to 5 most relevant
            excerpt = thread["excerpt"]
            link = thread["link"]
            starter = thread["starter"]
            msg_count = thread["message_count"]
            
            thread_summary = (
                f"\n**{i}. Thread started by {starter}** ({msg_count} message{'s' if msg_count != 1 else ''})\n"
                f"💬 {excerpt}\n"
                f"🔗 [Jump to conversation]({link})"
            )
            message_parts.append(thread_summary)
        
        # Add footer with options
        if len(threads) > 5:
            message_parts.append(f"\n_Showing 5 of {len(threads)} conversations. Ask for more details if needed._")
        
        # Add interactive options
        topic = search_results.get("topic", "this topic")
        message_parts.append(f"\n💡 *Want more details? Try:*\n• \"Show me more conversations about {topic}\"\n• \"Summarize the {topic} discussions\"")
        
        return "\n".join(message_parts)

class SearchFollowUpAction(Action):
    """
    Action for handling follow-up requests to search results.
    """
    
    def get_required_services(self) -> List[str]:
        return ["slack"]
    
    def can_handle(self, text: str) -> bool:
        """Check if this is a follow-up to a search query."""
        followup_patterns = [
            r"show me more conversations?",
            r"more details",
            r"see the actual (?:messages|conversations|threads)",
            r"take me to (?:the )?(?:thread|conversation)s?",
            r"(?:show|display) (?:the )?(?:full )?(?:thread|conversation)s?",
            r"summarize (?:the )?.*?discussions?",
            r"what exactly (?:did|was) (?:said|discussed)"
        ]
        
        for pattern in followup_patterns:
            if re.search(pattern, text.lower()):
                return True
        return False
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Handle follow-up requests with enhanced detail."""
        # Implementation would depend on storing context from previous searches
        # This could use thread_ts to maintain conversation state
        
        return ActionResponse(
            success=True,
            message="I can help you explore those conversations in more detail. Which specific thread would you like to see?",
            thread_ts=request.thread_ts
        )

# Helper function to create Slack blocks for rich formatting
def create_search_result_blocks(search_results: Dict[str, Any], channel_name: str) -> List[Dict[str, Any]]:
    """
    Create Slack blocks for rich search result display.
    
    Args:
        search_results: Formatted search results
        channel_name: Human-readable channel name
        
    Returns:
        List of Slack blocks
    """
    blocks = []
    
    # Header section
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": search_results["summary"]
        }
    })
    
    # Divider
    if search_results.get("show_details"):
        blocks.append({"type": "divider"})
    
    # Thread results
    threads = search_results.get("threads", [])
    for i, thread in enumerate(threads[:3], 1):  # Show top 3 in blocks
        excerpt = thread["excerpt"]
        link = thread["link"]
        starter = thread["starter"]
        msg_count = thread["message_count"]
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{i}. Thread by {starter}* ({msg_count} messages)\n{excerpt}"
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Jump to Thread"
                },
                "url": link,
                "action_id": f"jump_to_thread_{thread['thread_ts']}"
            }
        })
    
    # Footer with additional options
    if len(threads) > 3:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Showing 3 of {len(threads)} conversations. Ask for more details if needed."
                }
            ]
        })
    
    return blocks

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