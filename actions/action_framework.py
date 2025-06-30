from typing import Dict, Any, List, Optional, Tuple, Union, Protocol
import re
import asyncio
from abc import ABC, abstractmethod
from loguru import logger
from pydantic import BaseModel, Field
from config.settings import get_settings
from utils.token_management import count_messages_tokens

class ServiceContainer:
    """Container for all service dependencies required by actions."""
    
    def __init__(
        self,
        slack_service=None,
        notion_service=None,
        openai_service=None,
        web_service=None,  # ADD THIS
        youtube_service=None
    ):
        self.slack_service = slack_service
        self.notion_service = notion_service
        self.openai_service = openai_service
        self.web_service = web_service  # ADD THIS
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
        # Only exclude specific thread summary requests
        thread_summary_patterns = [
            r"summarize\s+(?:this\s+)?thread",
            r"(?:give\s+me\s+a\s+)?summary\s+of\s+(?:this\s+)?thread", 
            r"thread\s+summary",
            r"what\s+(?:is|was)\s+(?:this\s+)?thread\s+about",
            r"recap\s+(?:this\s+)?thread"
        ]
        
        is_thread_summary = any(re.search(pattern, text, re.IGNORECASE) for pattern in thread_summary_patterns)
        
        # Handle everything except thread summaries (which need special handling)
        return not is_thread_summary
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute with structured thread context awareness."""
        try:
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Get conversation history
            from utils.simple_history_manager import SimpleHistoryManager
            history_manager = SimpleHistoryManager()
            
            all_messages = await history_manager.get_recent_history(
                self.services.slack_service,
                request.channel_id,
                request.thread_ts,
                limit=500
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
            
            # DEBUG: Force a log to confirm we reach this point
            logger.info("🔍 DEBUG: About to start Notion link extraction")

            # NEW: Extract and fetch Notion page content
            linked_notion_content = ""
            notion_page_info = []  # For enhanced logging

            try:
                logger.info("🔍 DEBUG: Attempting import of notion_link_extractor")
                from utils.notion_link_extractor import extract_notion_links, format_notion_page_id, extract_page_title_from_content
                logger.info("🔍 DEBUG: Import successful, about to extract links")
                
                notion_page_ids = extract_notion_links(all_messages)
                logger.info(f"🔍 DEBUG: extract_notion_links returned: {notion_page_ids}")

                # TEMPORARY DEBUG - Remove after testing
                logger.info(f"🔍 DEBUG: Sample message text: {all_messages[0].get('text', 'NO TEXT') if all_messages else 'NO MESSAGES'}")
                for i, msg in enumerate(all_messages[:3]):
                    text = msg.get('text', '')
                    if 'notion.so' in text:
                        logger.info(f"🔍 DEBUG: Message {i} contains notion.so: {text}")
                
                if notion_page_ids:
                    logger.info(f"Found {len(notion_page_ids)} Notion page links in conversation")
                    
                    notion_contents = []
                    
                    # Limit to 10 pages to prevent token explosion
                    for page_id in list(notion_page_ids)[:10]:
                        formatted_id = format_notion_page_id(page_id)
                        
                        try:
                            content = await asyncio.to_thread(
                                self.services.notion_service.get_notion_page_content,
                                formatted_id
                            )
                            
                            if content and content.strip():
                                # Extract title for logging
                                page_title = extract_page_title_from_content(content, page_id)
                                content_length = len(content)
                                
                                # Format content with metadata for LLM
                                page_section = (
                                    f"=== NOTION PAGE: {page_title} (ID: {formatted_id}) ===\n"
                                    f"{content.strip()}\n"
                                    f"=== END NOTION PAGE ===\n"
                                )
                                notion_contents.append(page_section)
                                
                                # Track for logging
                                notion_page_info.append({
                                    'title': page_title,
                                    'id': formatted_id,
                                    'length': content_length
                                })
                                
                                logger.debug(f"Retrieved Notion page: {page_title} ({content_length} chars)")
                            else:
                                logger.warning(f"Empty or inaccessible Notion page: {formatted_id}")
                                
                        except Exception as page_error:
                            logger.error(f"Error fetching Notion page {formatted_id}: {page_error}")
                            continue  # Skip broken pages, continue processing
                    
                    if notion_contents:
                        linked_notion_content = "\n".join(notion_contents)
                        
                        # Enhanced logging as requested
                        total_chars = sum(info['length'] for info in notion_page_info)
                        logger.info(f"Included {len(notion_page_info)} Notion pages in prompt (total {total_chars:,} characters)")
                        for info in notion_page_info:
                            logger.info(f"  - \"{info['title']}\" ({info['length']:,} chars)")
                
            except Exception as notion_error:
                logger.error(f"🔍 DEBUG: Exception in Notion extraction: {notion_error}")
                logger.error(f"🔍 DEBUG: Exception type: {type(notion_error)}")
                import traceback
                logger.error(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
                linked_notion_content = "" 

            # NEW: Structure context based on thread vs channel
            thread_context = None
            conversation_history = None
            
            if request.thread_ts:
                # IN A THREAD: Separate channel and thread context
                channel_messages = []
                thread_messages = []
                
                for msg in all_messages:
                    if msg.get("thread_ts") == request.thread_ts:
                        thread_messages.append(msg)
                    else:
                        channel_messages.append(msg)
                
                logger.info(f"Thread mode: {len(thread_messages)} thread, {len(channel_messages)} channel messages")
                
                # Format both contexts separately
                formatted_channel = await asyncio.to_thread(
                    self.services.openai_service._format_conversation_for_openai,
                    channel_messages[-50:],  # Recent channel context
                    user_display_names,
                    self.services.slack_service.bot_user_id
                )
                
                formatted_thread = await asyncio.to_thread(
                    self.services.openai_service._format_conversation_for_openai,
                    thread_messages,  # ALL thread messages
                    user_display_names,
                    self.services.slack_service.bot_user_id
                )
                
                # Create structured thread context
                thread_context = {
                    'channel_messages': formatted_channel,
                    'thread_messages': formatted_thread
                }
                
            else:
                # REGULAR CHANNEL: Use normal conversation history
                conversation_history = await asyncio.to_thread(
                    self.services.openai_service._format_conversation_for_openai,
                    all_messages,
                    user_display_names,
                    self.services.slack_service.bot_user_id
                )
                logger.info(f"Channel mode: {len(conversation_history)} total messages")
            
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
            
            # NEW: Use the structured message preparation
            try:
                # Build messages with thread context structure
                messages = self.services.openai_service._prepare_messages(
                    prompt=request.prompt,
                    conversation_history=conversation_history,
                    user_specific_context=user_specific_context,
                    linked_notion_content=linked_notion_content,  # ADD THIS LINE
                    slack_user_id=request.user_id,
                    notion_service=self.services.notion_service,
                    thread_context=thread_context
                )
                
                # Send directly to OpenAI (messages are already prepared)
                from utils.token_management import count_messages_tokens
                token_count = count_messages_tokens(messages, self.services.openai_service.model)
                logger.info(f"Sending {token_count} tokens to OpenAI with structured thread context")
                
                # Track usage stats
                self.services.openai_service.usage_stats["request_count"] += 1

                prompt_text = self.services.openai_service._extract_prompt_for_logging(messages)
                logger.info(f"📤 REGULAR LLM CALL - Model: {self.services.openai_service.model}")
                logger.info(f"📝 INITIAL PROMPT ({len(prompt_text)} chars):")
                logger.info(prompt_text)
                
                response = await self.services.openai_service.async_client.chat.completions.create(
                    model=self.services.openai_service.model,
                    messages=messages,
                    max_tokens=self.services.openai_service.max_tokens,
                )
                
                response_text = response.choices[0].message.content
                usage = response.usage.model_dump() if hasattr(response, "usage") else None
                
                # Add tracking and logging that was missing
                if usage:
                    self.services.openai_service._update_usage_tracking(usage)
                    logger.info(f"✅ RECEIVED RESPONSE - Length: {len(response_text) if response_text else 0} chars")
                else:
                    logger.warning("⚠️ No usage data returned by OpenAI for non-web call")
                    self.services.openai_service.usage_stats["error_count"] += 1
                
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
    
class ThreadSummaryAction(Action):
    """Action specifically for summarizing thread conversations."""
    
    def get_required_services(self) -> List[str]:
        return ["slack", "openai", "notion"]
    
    def can_handle(self, text: str) -> bool:
        """Check if this is a thread summarization request."""
        # Look for explicit thread summary requests - comprehensive patterns
        summary_patterns = [
            # Direct summarization requests
            r"summarize\s+(?:this\s+)?thread",
            r"(?:give\s+me\s+a\s+)?summary\s+of\s+(?:this\s+)?thread",
            r"thread\s+summary",
            r"sum\s+up\s+(?:this\s+)?thread",
            r"recap\s+(?:this\s+)?thread",
            r"(?:thread\s+)?recap",
            
            # Question-based requests
            r"what\s+(?:is|was)\s+(?:this\s+)?thread\s+about",
            r"what's\s+(?:this\s+)?thread\s+about",
            r"what\s+(?:happened|was\s+discussed|went\s+on)\s+in\s+(?:this\s+)?thread",
            r"what\s+(?:happened|was\s+discussed|went\s+on)\s+here",
            r"what\s+did\s+(?:we|you\s+(?:all|guys))\s+discuss\s+(?:in\s+)?(?:this\s+)?thread",
            
            # Can/could variations
            r"can\s+you\s+summarize\s+(?:this\s+)?thread",
            r"could\s+you\s+(?:summarize|sum\s+up)\s+(?:this\s+)?thread",
            
            # Catch-all for "thread" + summary-related words
            r"thread.*(?:summary|recap|overview|rundown)",
            r"(?:summary|recap|overview|rundown).*thread"
        ]
        
        for pattern in summary_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute thread summarization with thread-only context."""
        try:
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Check if we're actually in a thread
            if not request.thread_ts:
                return ActionResponse(
                    success=False,
                    message="I can only summarize threads when you're actually in a thread conversation.",
                    thread_ts=request.thread_ts
                )
            
            # Fetch ONLY thread messages - no channel history
            try:
                thread_messages = await asyncio.to_thread(
                    self.services.slack_service.fetch_thread_history,
                    request.channel_id,
                    request.thread_ts,
                    limit=100  # Reasonable limit for thread summary
                )
                
                if not thread_messages:
                    return ActionResponse(
                        success=False,
                        message="I couldn't find any messages in this thread to summarize.",
                        thread_ts=request.thread_ts
                    )
                
                # Filter to just the main conversation (exclude bot messages, system messages)
                conversation_messages = []
                for msg in thread_messages:
                    if (msg.get("type") == "message" and 
                        not msg.get("subtype") and 
                        msg.get("text") and
                        msg.get("user") != self.services.slack_service.bot_user_id):
                        conversation_messages.append(msg)
                
                if not conversation_messages:
                    return ActionResponse(
                        success=False,
                        message="This thread doesn't seem to have any user messages to summarize.",
                        thread_ts=request.thread_ts
                    )
                
                logger.info(f"Summarizing {len(conversation_messages)} thread messages (filtered from {len(thread_messages)} total)")
                
            except Exception as fetch_error:
                logger.error(f"Error fetching thread messages: {fetch_error}")
                return ActionResponse(
                    success=False,
                    error=f"Thread fetch error: {str(fetch_error)}",
                    message="I couldn't retrieve the thread messages to summarize.",
                    thread_ts=request.thread_ts
                )
            
            # Build user display names for thread participants
            user_display_names = {}
            for msg in conversation_messages:
                user_id = msg.get("user")
                if user_id and user_id not in user_display_names:
                    try:
                        user_display_names[user_id] = await asyncio.to_thread(
                            self.services.slack_service.get_user_display_name,
                            user_id
                        )
                    except Exception as name_error:
                        logger.warning(f"Failed to get display name for user {user_id}: {name_error}")
                        user_display_names[user_id] = f"User {user_id}"
            
            # Format thread conversation for summarization
            formatted_conversation = []
            for msg in conversation_messages:
                user_id = msg.get("user")
                text = msg.get("text", "")
                username = user_display_names.get(user_id, f"User {user_id}")
                
                # Clean up text (remove mentions, etc.)
                clean_text = self.services.slack_service.clean_prompt_text(text)
                if clean_text:
                    formatted_conversation.append(f"{username}: {clean_text}")
            
            if not formatted_conversation:
                return ActionResponse(
                    success=False,
                    message="After filtering, there are no meaningful messages to summarize in this thread.",
                    thread_ts=request.thread_ts
                )
            
            # Create summarization prompt
            thread_content = "\n".join(formatted_conversation)
            summary_prompt = (
                f"Please provide a concise summary of this thread conversation. "
                f"Focus on the main topics discussed, key decisions made, and important outcomes.\n\n"
                f"Thread conversation:\n{thread_content}"
            )
            
            # Get user context for personalization (optional)
            try:
                from utils.context_builder import get_enhanced_user_context
                user_specific_context = await asyncio.to_thread(
                    get_enhanced_user_context,
                    self.services.notion_service,
                    request.user_id,
                    ""
                )
            except Exception as context_error:
                logger.warning(f"Could not get user context for summary: {context_error}")
                user_specific_context = ""
            
            # Generate summary using OpenAI
            try:
                summary, usage = await self.services.openai_service.get_completion_async(
                    prompt=summary_prompt,
                    conversation_history=None,  # No conversation history needed for summary
                    user_specific_context=user_specific_context,
                    max_tokens=500,  # Reasonable limit for summaries
                    slack_user_id=request.user_id,
                    notion_service=self.services.notion_service
                )
            except Exception as openai_error:
                logger.error(f"OpenAI summarization error: {openai_error}")
                return ActionResponse(
                    success=False,
                    error=f"Summarization error: {str(openai_error)}",
                    message="I encountered an error while generating the summary. Please try again.",
                    thread_ts=request.thread_ts
                )
            
            if not summary:
                return ActionResponse(
                    success=False,
                    message="I couldn't generate a summary for this thread. Please try again.",
                    thread_ts=request.thread_ts
                )
            
            # Format the response
            participant_list = ", ".join(set(user_display_names.values()))
            message_count = len(conversation_messages)
            
            response_message = (
                f"**Thread Summary** ({message_count} messages, participants: {participant_list})\n\n"
                f"{summary}"
            )
            
            return ActionResponse(
                success=True,
                message=response_message,
                thread_ts=request.thread_ts
            )
            
        except Exception as e:
            logger.error(f"Error in ThreadSummaryAction: {e}", exc_info=True)
            return ActionResponse(
                success=False,
                error=str(e),
                message="I encountered an error while summarizing this thread. Please try again.",
                thread_ts=request.thread_ts
            )

class RetrieveSummarizeAction(Action):
    """Action for retrieving and summarizing web content."""
    def get_required_services(self) -> List[str]:
        return ["slack", "openai", "notion", "web"]
    
    def can_handle(self, text: str) -> bool:
        """
        MUCH MORE RESTRICTIVE: Only handle direct URL fetch requests.
        Don't handle text summarization that happens to contain URLs.
        """
        # Must contain a URL
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        has_url = bool(re.search(url_pattern, text))
        
        if not has_url:
            return False
        
        # Don't handle YouTube URLs
        if "youtube.com" in text or "youtu.be" in text:
            return False
        
        # RESTRICTIVE: Only handle if this looks like a direct URL request
        # Examples that should be handled:
        # - "summarize https://example.com"
        # - "https://example.com"
        # - "check out https://example.com"
        
        # Examples that should NOT be handled (go to ContextResponseAction):
        # - "summarize this: [long pasted text that contains URLs]"
        # - "analyze this document: [text with embedded URLs]"
        
        url_match = re.search(url_pattern, text)
        if not url_match:
            return False
        
        url = url_match.group(0)
        
        # If the message is mostly just the URL (with minimal other text), handle it
        text_without_url = re.sub(url_pattern, '', text).strip()
        text_without_url = re.sub(r'<@\w+>', '', text_without_url).strip()  # Remove mentions
        
        # If what's left is very short, this is probably a URL request
        if len(text_without_url) < 30:
            return True
        
        # Check for explicit URL fetch requests
        fetch_patterns = [
            f"summarize {url}",
            f"summary of {url}",
            f"analyze {url}",
            f"what does {url} say",
            f"check {url}",
            f"look at {url}"
        ]
        
        for pattern in fetch_patterns:
            if pattern.lower() in text.lower():
                return True
        
        # Otherwise, let ContextResponseAction handle it
        return False
    
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
            
            # Store in Notion with dual summary approach
            notion_page_id = None
            try:
                # Generate shorter summary for Slack
                if summary:
                    short_summary_prompt = f"Create a 2-3 sentence summary of this summary:\n\n{summary}"
                    short_summary, _ = await self.services.openai_service.get_completion_async(
                        prompt=short_summary_prompt,
                        max_tokens=100,
                        slack_user_id=request.user_id,
                        notion_service=self.services.notion_service
                    )
                else:
                    short_summary = "Could not generate summary."
                
                # Create Notion page with full content
                page_title = f"Article Summary: {url}"
                page_content = f"# {page_title}\n\n**URL:** {url}\n\n## Summary\n\n{summary}\n\n## Full Content\n\n{content[:10000]}"
                
                notion_page_id = await asyncio.to_thread(
                    self.services.notion_service.create_content_page,
                    page_title,
                    page_content
                )
            except Exception as notion_error:
                logger.error(f"Notion page creation error: {notion_error}")
            
            # Construct response with short summary + Notion link
            if notion_page_id:
                notion_url = self.services.notion_service.get_page_url(notion_page_id)
                response_message = (
                    f"📄 **Article Summary**\n\n"
                    f"{short_summary or summary[:200]+'...'}\n\n"
                    f"📝 [View full summary in Notion]({notion_url})"
                )
            else:
                response_message = (
                    f"📄 **Article Summary**\n\n"
                    f"{short_summary or summary}"
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

class WebSearchAction(Action):
    """Action for performing web searches when explicitly requested."""
    
    def get_required_services(self) -> List[str]:
        return ["slack", "openai", "notion"]
    
    def can_handle(self, text: str) -> bool:
        """Check if this is a web search request."""
        search_patterns = [
            r"search\s+(?:the\s+)?web",
            r"web\s+search", 
            r"google\s+(?:this|that|it)",
            r"look\s+(?:this|that|it)\s+up\s+(?:on\s+)?(?:the\s+)?(?:web|internet)",
            r"find\s+(?:on\s+)?(?:the\s+)?(?:web|internet)"
        ]
        
        for pattern in search_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _extract_search_query(self, text: str) -> str:
        """Extract the search query from the message, removing trigger phrases."""
        search_patterns = [
            r"search\s+(?:the\s+)?web\s+(?:for\s+)?",
            r"web\s+search\s+(?:for\s+)?",
            r"google\s+",
            r"look\s+(?:this|that|it)\s+up\s+(?:on\s+)?(?:the\s+)?(?:web|internet)\s*:?\s*",
            r"find\s+(?:on\s+)?(?:the\s+)?(?:web|internet)\s+"
        ]
        
        clean_text = text
        for pattern in search_patterns:
            clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE)
        
        # Remove mentions and clean up
        clean_text = re.sub(r'<@\w+>', '', clean_text).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text if clean_text else text
    
    def _build_conversation_context_string(self, formatted_messages: List[Dict[str, str]]) -> str:
        """Convert formatted messages to context string for web search."""
        context_parts = []
        for msg in formatted_messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            context_parts.append(f"[{role}]: {content}")
        return "\n".join(context_parts)
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute web search with appropriate context based on channel vs thread."""
        try:
            if not self.services.validate_required_services(self.get_required_services()):
                return ActionResponse(
                    success=False,
                    error="Required services not available"
                )
            
            # Extract search query
            search_query = self._extract_search_query(request.prompt)
            if not search_query:
                return ActionResponse(
                    success=False,
                    message="I need a search query. Try 'search the web for <your question>'",
                    thread_ts=request.thread_ts
                )
            
            # Context handling: Thread-specific vs comprehensive (following existing patterns)
            conversation_context = ""
            
            if request.thread_ts:
                # THREAD CONTEXT: Like ThreadSummaryAction - thread messages only
                try:
                    thread_messages = await asyncio.to_thread(
                        self.services.slack_service.fetch_thread_history,
                        request.channel_id,
                        request.thread_ts,
                        limit=100
                    )
                    
                    if not thread_messages:
                        conversation_context = "No thread conversation history available."
                    else:
                        # Build user display names for thread participants
                        user_display_names = {}
                        for msg in thread_messages:
                            user_id = msg.get("user")
                            if user_id and user_id not in user_display_names:
                                try:
                                    user_display_names[user_id] = await asyncio.to_thread(
                                        self.services.slack_service.get_user_display_name,
                                        user_id
                                    )
                                except Exception as name_error:
                                    logger.warning(f"Failed to get display name for user {user_id}: {name_error}")
                                    user_display_names[user_id] = f"User {user_id}"
                        
                        # Format thread conversation
                        formatted_thread = await asyncio.to_thread(
                            self.services.openai_service._format_conversation_for_openai,
                            thread_messages,
                            user_display_names,
                            self.services.slack_service.bot_user_id
                        )
                        
                        conversation_context = f"=== THREAD CONVERSATION ===\n{self._build_conversation_context_string(formatted_thread)}"
                        
                except Exception as thread_error:
                    logger.error(f"Error fetching thread context: {thread_error}")
                    conversation_context = "Error retrieving thread conversation."
            
            else:
                # CHANNEL CONTEXT: Like ContextResponseAction - comprehensive history
                try:
                    from utils.simple_history_manager import SimpleHistoryManager
                    history_manager = SimpleHistoryManager()
                    
                    all_messages = await history_manager.get_recent_history(
                        self.services.slack_service,
                        request.channel_id,
                        None,  # No thread_ts for channel context
                        limit=500
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
                    
                    # Format conversation history
                    conversation_history = await asyncio.to_thread(
                        self.services.openai_service._format_conversation_for_openai,
                        all_messages,
                        user_display_names,
                        self.services.slack_service.bot_user_id
                    )
                    
                    conversation_context = f"=== CHANNEL CONVERSATION HISTORY ===\n{self._build_conversation_context_string(conversation_history)}"
                    
                except Exception as channel_error:
                    logger.error(f"Error fetching channel context: {channel_error}")
                    conversation_context = "Error retrieving channel conversation."
            
            # Get user context from Notion (same as existing actions)
            try:
                from utils.context_builder import get_enhanced_user_context
                notion_context = await asyncio.to_thread(
                    get_enhanced_user_context,
                    self.services.notion_service,
                    request.user_id,
                    ""
                )
            except Exception as context_error:
                logger.error(f"Error building user context: {context_error}")
                notion_context = ""
            
            # Combine contexts for web search
            full_context = ""
            if notion_context:
                full_context += notion_context + "\n\n"
            if conversation_context:
                full_context += conversation_context
            
            # Perform web search using existing method
            try:
                response_text, usage = await self.services.openai_service.get_web_search_completion_async(
                    prompt=search_query,
                    user_specific_context=full_context,
                    timeout=60.0,
                    slack_user_id=request.user_id,
                    notion_service=self.services.notion_service
                )
            except Exception as search_error:
                logger.error(f"Web search error: {search_error}")
                return ActionResponse(
                    success=False,
                    error=f"Web search error: {str(search_error)}",
                    message="I encountered an error during the web search. Please try again.",
                    thread_ts=request.thread_ts
                )
            
            if not response_text:
                return ActionResponse(
                    success=False,
                    message="I couldn't find any results for that search. Please try a different query.",
                    thread_ts=request.thread_ts
                )
            
            # Format response with search query context
            formatted_response = f"🔍 **Web Search Results for:** {search_query}\n\n{response_text}"
            
            return ActionResponse(
                success=True,
                message=formatted_response,
                thread_ts=request.thread_ts
            )
            
        except Exception as e:
            logger.error(f"Error in WebSearchAction: {e}", exc_info=True)
            return ActionResponse(
                success=False,
                error=str(e),
                message="I encountered an error during the web search. Please try again.",
                thread_ts=request.thread_ts
            )

class ActionRouter:
    """Simplified router for determining and executing actions."""
    
    def __init__(self, services: ServiceContainer):
        self.services = services
        
        # Simple action list - order matters!
        self.actions = [
            ThreadSummaryAction(services),       # Thread summaries first
            YoutubeSummarizeAction(services),    # YouTube URLs first
            RetrieveSummarizeAction(services),   # Other URLs second
            WebSearchAction(services),           # Web search requests third
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