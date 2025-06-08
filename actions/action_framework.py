from typing import Dict, Any, List, Optional, Tuple, Union, Protocol
import re
import asyncio
from abc import ABC, abstractmethod
from loguru import logger
from pydantic import BaseModel, Field

class ServiceContainer:
    """Container for all service dependencies required by actions."""
    
    def __init__(
        self,
        slack_service=None,
        notion_service=None,
        openai_service=None,
        web_service=None,
        youtube_service=None
    ):
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
        """Validate that required services are available."""
        missing_services = []
        
        for name in service_names:
            service = getattr(self, f"{name}_service", None)
            if service is None:
                missing_services.append(name)
                continue
                
            if hasattr(service, "is_available") and callable(service.is_available):
                if not service.is_available():
                    missing_services.append(name)
        
        if missing_services:
            logger.warning(f"Required services not available: {', '.join(missing_services)}")
            return False
            
        return True

class ActionRequest(BaseModel):
    """Base model for action requests with common fields."""
    channel_id: str = Field(..., description="Slack channel ID")
    user_id: str = Field(..., description="Slack user ID")
    message_ts: str = Field(..., description="Slack message timestamp")
    thread_ts: Optional[str] = Field(None, description="Slack thread timestamp")
    text: str = Field(..., description="Full message text")
    prompt: str = Field(..., description="Cleaned message text (without mentions)")

class ActionResponse(BaseModel):
    """Base model for action responses with common fields."""
    success: bool = Field(..., description="Whether the action was successful")
    message: Optional[str] = Field(None, description="Response message to send to Slack")
    thread_ts: Optional[str] = Field(None, description="Thread timestamp for the response")
    error: Optional[str] = Field(None, description="Error message if not successful")

class Action(ABC):
    """Base class for all actions in the system."""
    
    def __init__(self, services: ServiceContainer):
        self.services = services
        self.name = self.__class__.__name__
        logger.debug(f"Action {self.name} initialized")
    
    @abstractmethod
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute the action asynchronously."""
        pass
    
    @abstractmethod
    def can_handle(self, text: str) -> bool:
        """Check if this action can handle the given text."""
        pass
    
    def get_required_services(self) -> List[str]:
        """Get the list of services required by this action."""
        return ["slack"]

class ContextResponseAction(Action):
    """
    Default action for all user messages.
    Gathers full context (channel/thread history + Notion) and sends to LLM.
    """
    
    def get_required_services(self) -> List[str]:
        return ["slack", "openai", "notion"]
    
    def can_handle(self, text: str) -> bool:
        """This is the default action, handles everything except URLs."""
        # Don't handle URLs - let specialized actions handle those
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        has_url = bool(re.search(url_pattern, text))
        return not has_url
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute with full context gathering - no filtering or search logic."""
        try:
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Simple history gathering - NO FILTERING OR SEARCH LOGIC
            from utils.simple_history_manager import SimpleHistoryManager
            history_manager = SimpleHistoryManager()
            
            # Get recent channel and thread history
            all_messages = await history_manager.get_recent_history(
                self.services.slack_service,
                request.channel_id,
                request.thread_ts,
                limit=200  # Simple limit, no complex logic
            )
            
            # Build user display names
            user_display_names = {}
            for msg in all_messages:
                user_id = msg.get("user") or msg.get("bot_id")
                if user_id and user_id not in user_display_names:
                    try:
                        user_display_names[user_id] = await asyncio.to_thread(
                            self.services.slack_service.get_user_display_name,
                            user_id
                        )
                    except Exception as name_error:
                        logger.warning(f"Failed to get display name for user {user_id}: {name_error}")
                        user_display_names[user_id] = f"User {user_id}"
            
            # Format history for OpenAI
            formatted_history = await asyncio.to_thread(
                self.services.openai_service._format_conversation_for_openai,
                all_messages,
                user_display_names,
                self.services.slack_service.bot_user_id
            )
            
            # Get user context from Notion
            try:
                from utils.context_builder import get_enhanced_user_context
                user_specific_context = await asyncio.to_thread(
                    get_enhanced_user_context,
                    self.services.notion_service,
                    request.user_id,
                    ""
                )
            except Exception as context_error:
                logger.error(f"Error building user context: {context_error}")
                user_specific_context = ""
            
            # Send to LLM with full context - no special handling
            try:
                response_text, usage = await self.services.openai_service.get_completion_async(
                    prompt=request.prompt,
                    conversation_history=formatted_history,
                    user_specific_context=user_specific_context,
                    slack_user_id=request.user_id,
                    notion_service=self.services.notion_service
                )
            except Exception as openai_error:
                logger.error(f"OpenAI API error: {openai_error}")
                return ActionResponse(
                    success=False,
                    error=f"OpenAI API error: {str(openai_error)}",
                    message="I'm having trouble connecting to my AI service. Please try again.",
                    thread_ts=request.thread_ts
                )
            
            if not response_text:
                return ActionResponse(
                    success=False,
                    message="I couldn't generate a response for that.",
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
                message="I encountered an error processing your request. Please try again.",
                thread_ts=request.thread_ts
            )

class RetrieveSummarizeAction(Action):
    """Action for retrieving and summarizing web content."""
    
    def get_required_services(self) -> List[str]:
        return ["slack", "openai", "notion", "web"]
    
    def can_handle(self, text: str) -> bool:
        """Check if the text contains a web URL (excluding YouTube)."""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return bool(re.search(url_pattern, text)) and "youtube.com" not in text and "youtu.be" not in text
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute web content retrieval and summarization."""
        try:
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Extract URL
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
            try:
                content = await self.services.web_service.fetch_content(url)
            except Exception as web_error:
                logger.error(f"Web service error: {web_error}")
                return ActionResponse(
                    success=False,
                    error=f"Web fetch error: {str(web_error)}",
                    message=f"I couldn't fetch content from {url}. The site may be unavailable.",
                    thread_ts=request.thread_ts
                )
            
            if not content:
                return ActionResponse(
                    success=False,
                    error=f"Failed to fetch content from {url}",
                    message=f"I couldn't fetch content from {url}.",
                    thread_ts=request.thread_ts
                )
            
            # Generate summary
            try:
                summary_prompt = f"Please summarize the following web content from {url}:\n\n{content[:10000]}..."
                summary, _ = await self.services.openai_service.get_completion_async(
                    prompt=summary_prompt,
                    max_tokens=500,
                    slack_user_id=request.user_id,
                    notion_service=self.services.notion_service
                )
            except Exception as openai_error:
                logger.error(f"OpenAI summarization error: {openai_error}")
                return ActionResponse(
                    success=False,
                    error=f"Summarization error: {str(openai_error)}",
                    message="I fetched the content but couldn't generate a summary.",
                    thread_ts=request.thread_ts
                )
            
            if not summary:
                return ActionResponse(
                    success=False,
                    error="Failed to generate summary",
                    message="I fetched the content but couldn't generate a summary.",
                    thread_ts=request.thread_ts
                )
            
            # Store in Notion
            try:
                notion_page_id = await asyncio.to_thread(
                    self.services.notion_service.create_content_page,
                    f"Summary of {url}",
                    f"# Summary of {url}\n\n{summary}\n\n## Full Content\n\n{content[:20000]}..."
                )
            except Exception as notion_error:
                logger.error(f"Notion page creation error: {notion_error}")
                notion_page_id = None
            
            # Construct response
            if notion_page_id:
                response_message = (
                    f"*Summary of {url}*\n\n{summary}\n\n"
                    f"Full summary saved in Notion: {self.services.notion_service.get_page_url(notion_page_id)}"
                )
            else:
                response_message = f"*Summary of {url}*\n\n{summary}"
            
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
                message="I encountered an error retrieving and summarizing that URL.",
                thread_ts=request.thread_ts
            )

class YoutubeSummarizeAction(Action):
    """Action for retrieving and summarizing YouTube videos."""
    
    def get_required_services(self) -> List[str]:
        return ["slack", "openai", "notion", "youtube"]
    
    def can_handle(self, text: str) -> bool:
        """Check if the text contains a YouTube URL."""
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://(?:www\.)?youtu\.be/[\w-]+'
        ]
        return any(bool(re.search(pattern, text)) for pattern in youtube_patterns)
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute YouTube video summarization."""
        try:
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Extract YouTube URL
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
                    error="No YouTube URL found",
                    message="I couldn't find a valid YouTube URL in your message.",
                    thread_ts=request.thread_ts
                )
            
            # Fetch video info and transcript
            try:
                video_info = await self.services.youtube_service.get_video_info(youtube_url)
                transcript = await self.services.youtube_service.get_transcript(youtube_url)
            except Exception as youtube_error:
                logger.error(f"YouTube service error: {youtube_error}")
                return ActionResponse(
                    success=False,
                    error=f"YouTube fetch error: {str(youtube_error)}",
                    message=f"I couldn't fetch transcript for {youtube_url}. The video may not have captions available.",
                    thread_ts=request.thread_ts
                )
            
            if not transcript:
                return ActionResponse(
                    success=False,
                    error=f"Failed to fetch transcript for {youtube_url}",
                    message=f"I couldn't fetch a transcript for {youtube_url}. The video may not have captions available.",
                    thread_ts=request.thread_ts
                )
            
            # Generate summary
            try:
                summary_prompt = (
                    f"Please summarize the following YouTube video transcript:\n\n"
                    f"Title: {video_info.get('title', 'Unknown')}\n"
                    f"Channel: {video_info.get('channel', 'Unknown')}\n"
                    f"Transcript:\n{transcript[:10000]}..."
                )
                
                summary, _ = await self.services.openai_service.get_completion_async(
                    prompt=summary_prompt,
                    max_tokens=500,
                    slack_user_id=request.user_id,
                    notion_service=self.services.notion_service
                )
            except Exception as openai_error:
                logger.error(f"OpenAI summarization error: {openai_error}")
                return ActionResponse(
                    success=False,
                    error=f"Summarization error: {str(openai_error)}",
                    message="I fetched the transcript but couldn't generate a summary.",
                    thread_ts=request.thread_ts
                )
            
            if not summary:
                return ActionResponse(
                    success=False,
                    error="Failed to generate summary",
                    message="I fetched the transcript but couldn't generate a summary.",
                    thread_ts=request.thread_ts
                )
            
            # Store in Notion
            try:
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
            except Exception as notion_error:
                logger.error(f"Notion page creation error: {notion_error}")
                notion_page_id = None
            
            # Construct response
            if notion_page_id:
                response_message = (
                    f"*Summary of YouTube Video: {video_info.get('title', 'Unknown')}*\n\n"
                    f"{summary}\n\n"
                    f"Full transcript and summary saved in Notion: "
                    f"{self.services.notion_service.get_page_url(notion_page_id)}"
                )
            else:
                response_message = (
                    f"*Summary of YouTube Video: {video_info.get('title', 'Unknown')}*\n\n"
                    f"{summary}"
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
                message="I encountered an error processing that YouTube video.",
                thread_ts=request.thread_ts
            )

class ActionRouter:
    """Simplified router for determining and executing actions."""
    
    def __init__(self, services: ServiceContainer):
        self.services = services
        
        # Simple action list - order matters!
        self.actions = [
            YoutubeSummarizeAction(services),    # YouTube URLs first
            RetrieveSummarizeAction(services),   # Other URLs second  
            ContextResponseAction(services)      # Everything else (default)
        ]
        
        logger.info(f"ActionRouter initialized with {len(self.actions)} actions")
    
    def _get_action_for_text(self, text: str) -> Action:
        """Determine which action should handle the given text."""
        for action in self.actions:
            if action.can_handle(text):
                logger.info(f"Selected action {action.name} for message: {text[:50]}...")
                return action
        
        # Should never reach here because ContextResponseAction handles everything
        return self.actions[-1]
    
    async def route_action(self, request: ActionRequest) -> ActionResponse:
        """Route the request to the appropriate action handler."""
        try:
            # Determine and execute action
            action = self._get_action_for_text(request.text)
            
            try:
                response = await action.execute(request)
            except Exception as action_error:
                logger.error(f"Action execution error in {action.name}: {action_error}", exc_info=True)
                return ActionResponse(
                    success=False,
                    error=f"Action execution failed: {str(action_error)}",
                    message="I encountered an error processing your request. Please try again.",
                    thread_ts=request.thread_ts
                )
            
            if not isinstance(response, ActionResponse):
                logger.error(f"Action {action.name} returned invalid response type")
                return ActionResponse(
                    success=False,
                    error="Invalid action response",
                    message="I encountered an internal error. Please try again.",
                    thread_ts=request.thread_ts
                )
            
            return response
        
        except Exception as e:
            logger.error(f"Error in route_action: {e}", exc_info=True)
            return ActionResponse(
                success=False,
                error=f"Router error: {str(e)}",
                message="I encountered an error processing your request. Please try again.",
                thread_ts=request.thread_ts
            )