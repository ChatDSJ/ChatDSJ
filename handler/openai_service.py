import time
import asyncio
from services.notion_parser import NotionContextManager
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import get_settings
from utils.token_management import ensure_messages_within_limit, count_messages_tokens

class OpenAIService:
    """
    Service for handling all OpenAI API interactions with language preference support and proper error handling.
    """
    
    def __init__(self):
        """Initialize the OpenAI service with API key from settings."""
        settings = get_settings()
        self.api_key = settings.openai_api_key.get_secret_value()
        self.model = settings.openai_model
        self.max_tokens = settings.max_tokens_response
        
        # Initialize both sync and async clients
        self._init_clients()

        # Initialize the Notion context manager
        self.notion_context_manager = NotionContextManager()
        
        # Model pricing for usage tracking
        self.model_pricing = {
            "gpt-4o": {"prompt": 5.00, "completion": 15.00},
            "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
            "gpt-3.5-turbo-0125": {"prompt": 0.50, "completion": 1.50},
        }
        
        # Usage statistics
        self.usage_stats = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "request_count": 0,
            "error_count": 0,
        }
        
        logger.info(f"OpenAI service initialized with model {self.model}")
    
    def _init_clients(self):
        """Initialize both synchronous and asynchronous OpenAI clients."""
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
            logger.info("OpenAI clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI clients: {e}")
            self.client = None
            self.async_client = None
    
    def is_available(self) -> bool:
        """Check if the OpenAI service is available."""
        return self.client is not None and self.async_client is not None

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    
    async def get_completion_async(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_specific_context: Optional[str] = None,
        linked_notion_content: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 30.0, 
        slack_user_id: Optional[str] = None,
        notion_service=None
        # REMOVE: task_type: str = "general"  # DELETE THIS PARAMETER
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Asynchronously get a completion from OpenAI with full context injection and language preference support.
        """
        if not self.is_available():
            logger.error("OpenAI async client not initialized")
            return None, None

        try:
            # Construct full message list using the context-aware helper with language preference
            messages = self._prepare_messages(
                prompt=prompt,
                conversation_history=conversation_history,
                user_specific_context=user_specific_context,
                linked_notion_content=linked_notion_content,
                system_prompt=system_prompt,
                slack_user_id=slack_user_id,
                notion_service=notion_service
                # REMOVE: task_type=task_type  # DELETE THIS LINE
            )

            # Log payload being sent to OpenAI (for debugging only)
            logger.debug("Full OpenAI prompt:\n" + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages))

            # Track usage stats
            self.usage_stats["request_count"] += 1
            token_count = count_messages_tokens(messages, self.model)
            logger.info(f"Sending {token_count} tokens to OpenAI async. Model: {self.model}")

            for attempt in range(3):
                try:
                    # Use asyncio.wait_for to implement timeout
                    response = await asyncio.wait_for(
                        self.async_client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=max_tokens or self.max_tokens,
                        ),
                        timeout=timeout
                    )
                    
                    content = response.choices[0].message.content
                    usage = response.usage.model_dump() if hasattr(response, "usage") else None

                    if usage:
                        self._update_usage_tracking(usage)

                    logger.info(f"OpenAI response received successfully in {attempt+1} attempt(s)")
                    return content, usage

                except asyncio.TimeoutError:
                    logger.error(f"OpenAI request timed out after {timeout} seconds (attempt {attempt+1}/3)")
                    if attempt == 2:  # Last attempt
                        return "I'm sorry, but I timed out while processing your request. Please try again.", None
                    # Otherwise retry with longer timeout
                    timeout *= 1.5  # Increase timeout for next attempt
                    continue
                    
                except (TimeoutError, ConnectionError) as e:
                    if attempt < 2:
                        wait_time = 2 ** attempt
                        logger.warning(f"Retrying after error: {e}. Attempt {attempt+1}/3. Waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed after 3 attempts: {e}")
                        raise

        except Exception as e:
            self.usage_stats["error_count"] += 1
            logger.error(f"Error getting async OpenAI response: {e}", exc_info=True)
            return "I'm sorry, I encountered an error when processing your request. Please try again.", None

    async def get_web_search_completion_async(
        self,
        prompt: str,
        user_specific_context: Optional[str] = None,
        timeout: float = 60.0,  # Web search takes longer
        slack_user_id: Optional[str] = None,
        notion_service=None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Get a completion with web search using OpenAI's responses API with full context.
        """
        if not self.is_available():
            logger.error("OpenAI async client not initialized")
            return None, None

        try:
            # Build comprehensive prompt using similar structure to _prepare_messages
            settings = get_settings()
            
            # Build system prompt for web search
            web_search_system_prompt = (
                "You are an assistant with web search capabilities embedded in a Slack channel. "
                "You have access to current information from the web and stored user information. "
                
                "CRITICAL: When answering questions, you MUST synthesize information from ALL sources: "
                "1. Current web search results (for up-to-date information) "
                "2. Stored user profile information (their persistent facts and preferences) "
                
                "Both sources are equally important. Use web search for current/factual information, "
                "but incorporate user preferences and context to personalize your response. "
                
                "Your primary job is to answer the user's question directly and concisely while "
                "incorporating relevant information from all available sources."
            )
            
            # Build the comprehensive prompt
            full_prompt = "=== SYSTEM INSTRUCTIONS ===\n" + web_search_system_prompt + "\n\n"
            
            # Add user context if available
            if user_specific_context:
                full_prompt += "=== USER PROFILE INFORMATION ===\n"
                full_prompt += user_specific_context + "\n\n"
            
            # Add the current user question
            full_prompt += "=== USER'S CURRENT QUESTION (WITH WEB SEARCH) ===\n" + prompt + "\n\n"
            
            # Add web search specific instructions
            full_prompt += """Please search the web for current information to answer this question, then provide a comprehensive response that:
    - Uses the most up-to-date information from your web search
    - Incorporates relevant user preferences/context from their profile
    - Is direct, helpful, and personalized to this specific user
    - Includes citations or sources when referencing specific web results"""

            # Track usage stats
            self.usage_stats["request_count"] += 1
            logger.info(f"Sending web search request to OpenAI. Model: {self.model}")
            logger.debug(f"Web search prompt length: {len(full_prompt)} characters")

            # Use the responses.create method with web search tool
            kwargs = {
                "model": self.model,
                "input": full_prompt  # Use the comprehensive prompt instead of raw query
            }
            
            # Only add web search for supported models
            if self.model in ("gpt-4o", "gpt-4.1"):
                kwargs["tools"] = [{"type": "web_search"}]
            
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.async_client.responses.create,
                    **kwargs
                ),
                timeout=timeout
            )
            
            # Extract response text
            response_text = ""
            if hasattr(response, 'output_text'):
                response_text = response.output_text
            elif hasattr(response, 'output'):
                response_text = str(response.output)
            
            # Create usage dict (simplified since responses API may not provide detailed usage)
            usage = {
                "prompt_tokens": self._estimate_tokens(full_prompt),
                "completion_tokens": self._estimate_tokens(response_text),
                "total_tokens": self._estimate_tokens(full_prompt) + self._estimate_tokens(response_text)
            }
            
            if usage:
                self._update_usage_tracking(usage)

            logger.info(f"Web search response received successfully")
            return response_text, usage

        except asyncio.TimeoutError:
            logger.error(f"Web search request timed out after {timeout} seconds")
            return "I'm sorry, the web search took too long. Please try again.", None
        except Exception as e:
            self.usage_stats["error_count"] += 1
            logger.error(f"Error getting web search response: {e}", exc_info=True)
            return "I'm sorry, I encountered an error during the web search. Please try again.", None
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens for usage tracking when not provided by API."""
        from utils.token_management import count_tokens
        return count_tokens(text or "", self.model)

    def _prepare_messages(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_specific_context: Optional[str] = None,
        linked_notion_content: Optional[str] = None,
        system_prompt: Optional[str] = None,
        slack_user_id: Optional[str] = None,
        notion_service=None,
        thread_context: Optional[Dict] = None
    ) -> List[Dict[str, str]]:
        """
        UPDATED: Enhanced context synthesis for natural information combination.
        """
        settings = get_settings()
        max_context_tokens = settings.max_context_tokens_general - settings.max_tokens_response

        # Build system prompt
        system_prompt_content = system_prompt if system_prompt else settings.openai_system_prompt

        # UPDATED: Enhanced structure for natural synthesis
        full_prompt = "=== SYSTEM INSTRUCTIONS ===\n" + system_prompt_content + "\n\n"

        # UPDATED: Present user context as one of multiple information sources (not authoritative)
        if user_specific_context:
            full_prompt += "=== USER PROFILE INFORMATION ===\n"
            full_prompt += user_specific_context + "\n\n"

        # PRESERVE CRITICAL THREAD LOGIC - Don't change this working code!
        if thread_context:
            full_prompt += "=== CONTEXT HIERARCHY ===\n"
            full_prompt += "You are participating in a Slack thread conversation.\n\n"
            
            # Channel context (background)
            if thread_context.get('channel_messages'):
                full_prompt += "BACKGROUND CHANNEL CONTEXT (for general reference):\n"
                for msg in thread_context['channel_messages']:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    full_prompt += f"[{role.upper()}]: {content}\n"
                full_prompt += "\n"
            
            # Thread context (current conversation)
            if thread_context.get('thread_messages'):
                full_prompt += "CURRENT THREAD CONVERSATION (your immediate context):\n"
                full_prompt += "This is the specific thread conversation you are participating in.\n"
                full_prompt += "When the user refers to anything from our conversation, they are referring to content in THIS thread.\n\n"
                for msg in thread_context['thread_messages']:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    full_prompt += f"[THREAD {role.upper()}]: {content}\n"
                full_prompt += "\n"
            
            full_prompt += "=== END CONTEXT HIERARCHY ===\n\n"
        else:
            # Regular conversation history (non-thread)
            if conversation_history:
                full_prompt += "=== CONVERSATION HISTORY ===\n"
                for msg in conversation_history:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    full_prompt += f"[{role.upper()}]: {content}\n"
                full_prompt += "=== END CONVERSATION HISTORY ===\n\n"

        # Add the current user question
        full_prompt += "=== USER'S CURRENT QUESTION ===\n" + prompt + "\n\n"

        # ENHANCED: Preserve thread logic + add information synthesis
        if thread_context:
            full_prompt += """Please respond based on the context hierarchy above. When the user references anything from our conversation, look to the CURRENT THREAD CONVERSATION section.

    IMPORTANT: When answering questions about user preferences, facts, or interests, naturally combine information from:
    - User background information (their stored profile)
    - Current thread conversation (what they've said in THIS thread)
    Both sources are equally valid - synthesize them naturally."""
        else:
            full_prompt += """Please respond by synthesizing information from ALL sources above - both your stored profile information and recent conversation history.

    IMPORTANT: When answering questions about user preferences, facts, or interests, naturally combine information from:
    - User background information (their stored profile)  
    - Recent conversation activity (what they've mentioned anywhere recently, including in threads)
    Both sources are equally valid - synthesize them naturally."""
        
        # Create single message
        messages = [{"role": "user", "content": full_prompt}]

        # Apply token limits
        messages = ensure_messages_within_limit(messages, self.model, max_context_tokens)

        return messages

    def _update_usage_tracking(self, usage: Dict[str, int]) -> None:
        """Update usage statistics for monitoring."""
        if not usage:
            return
            
        # Update token counts
        self.usage_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.usage_stats["completion_tokens"] += usage.get("completion_tokens", 0)
        self.usage_stats["total_tokens"] += usage.get("total_tokens", 0)
        
        # Calculate and update cost
        cost = self._calculate_cost(usage)
        self.usage_stats["total_cost"] += cost
        
        logger.debug(
            f"OpenAI request: {usage.get('prompt_tokens', 0)} prompt tokens, "
            f"{usage.get('completion_tokens', 0)} completion tokens. "
            f"Cost: ${cost:.6f}. Total cost: ${self.usage_stats['total_cost']:.6f}"
        )
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate the cost of an API request."""
        base_model = self.model.split('-preview')[0] if '-preview' in self.model else self.model
        pricing = self.model_pricing.get(base_model)
        
        if not pricing or not usage:
            return 0.0
            
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        cost = (
            (prompt_tokens / 1_000_000) * pricing["prompt"] +
            (completion_tokens / 1_000_000) * pricing["completion"]
        )
        
        return cost

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats.copy()
        
    def _format_conversation_for_openai(
        self,
        messages: list, 
        user_display_names: Dict[str, str],
        bot_user_id: str
    ) -> list:
        """Format conversation history for OpenAI."""
        formatted = []
        
        for msg in messages:
            if msg.get("type") != "message" or msg.get("subtype") or not msg.get("text"):
                continue

            user_id = msg.get("user") or msg.get("bot_id")
            text = msg.get("text", "")
            
            if not text:
                continue
                
            if user_id == bot_user_id:
                # Bot's own messages
                formatted.append({"role": "assistant", "content": text})
            else:
                # User messages - include username for context
                username = user_display_names.get(user_id, f"User {user_id}")
                formatted.append({"role": "user", "content": f"{username}: {text}"})
        
        return formatted