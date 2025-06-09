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

    def _prepare_messages(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_specific_context: Optional[str] = None,
        linked_notion_content: Optional[str] = None,
        system_prompt: Optional[str] = None,
        slack_user_id: Optional[str] = None,
        notion_service=None
    ) -> List[Dict[str, str]]:
        
        settings = get_settings()
        max_context_tokens = settings.max_context_tokens_general - settings.max_tokens_response

        # Load the SYSTEM INSTRUCTIONS
        system_prompt_content = system_prompt if system_prompt else settings.openai_system_prompt if settings.openai_system_prompt else ""

        # 1. Add the SYSTEM INSTRUCTIONS
        full_prompt = "=== SYSTEM INSTRUCTIONS ===\n" + system_prompt_content + "\n\n"

        # 2. Add the USER'S CURRENT QUESTION
        full_prompt += "=== USER'S QUESTION ===\n" + prompt + "\n\n"

        # 3. Create unified knowledge pool from all available context
        full_prompt += "=== KNOWLEDGE POOL ===\n"
        full_prompt += "Use the following information to answer the user's question:\n\n"
        
        # Add user's personal facts from Notion
        if user_specific_context:
            full_prompt += "**User's Personal Facts:**\n"
            full_prompt += user_specific_context + "\n\n"
        
        # Add linked Notion content
        if linked_notion_content:
            full_prompt += "**Additional User Documents:**\n"
            full_prompt += linked_notion_content + "\n\n"
        
        # Add conversation history
        if conversation_history:
            full_prompt += "**Recent Conversation History:**\n"
            for msg in conversation_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                full_prompt += f"[{role.upper()}]: {content}\n"
            full_prompt += "\n"
        
        full_prompt += "=== END KNOWLEDGE POOL ===\n\n"

        # Simple instruction
        full_prompt += "Please answer the user's question based on the knowledge pool above."

        # Create a SINGLE MESSAGE with role "user" containing the FULL PROMPT
        messages = [{"role": "user", "content": full_prompt}]

        logger.debug(f"FULL OpenAI prompt (first 500 chars): {full_prompt[:500]}...")

        # Truncate if necessary
        messages = ensure_messages_within_limit(messages, self.model, max_context_tokens)

        return messages

    def _prepare_structured_messages(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_specific_context: Optional[str] = None,
        linked_notion_content: Optional[str] = None,
        system_prompt: Optional[str] = None,
        preferred_name: Optional[str] = None,
        slack_user_id: Optional[str] = None,
        notion_service=None
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for the OpenAI API with structured Notion context and language preference.
        
        NOTE: This method is now simplified. Consider using _prepare_messages directly 
        for most use cases as it handles everything in a single user message.
        """
        # Get settings for default system prompt
        settings = get_settings()
        
        # Use provided system prompt or default
        base_system_prompt = system_prompt if system_prompt else settings.openai_system_prompt
        
        # Get user's language preference
        language_preference = "English"  # Default
        if slack_user_id and notion_service and hasattr(notion_service, 'get_user_language_preference'):
            try:
                language_preference = notion_service.get_user_language_preference(slack_user_id)
                logger.info(f"Retrieved language preference for user {slack_user_id}: {language_preference}")
            except Exception as e:
                logger.warning(f"Could not get language preference for user {slack_user_id}: {e}")

        # Process Notion content if available
        if user_specific_context:
            # Build a structured system prompt with the Notion context and language preference
            system_prompt_content = self.notion_context_manager.build_openai_system_prompt(
                base_prompt=base_system_prompt,
                notion_content=user_specific_context,
                preferred_name=preferred_name
            )
            logger.info("Built structured system prompt with Notion context and language preference")
        else:
            # Even without user context, include language preference
            system_prompt_content = "=" * 80 + "\n"
            system_prompt_content += "🌍 CRITICAL LANGUAGE INSTRUCTION 🌍\n"
            system_prompt_content += f"ALWAYS RESPOND IN: {language_preference.upper()}\n"
            system_prompt_content += f"USER'S LANGUAGE PREFERENCE: {language_preference}\n"
            system_prompt_content += "This is MANDATORY - all responses must be in this language.\n"
            system_prompt_content += "=" * 80 + "\n\n"
            system_prompt_content += base_system_prompt
        
        # Add linked Notion content if available
        if linked_notion_content:
            system_prompt_content += (
                f"\n\n--- REFERENCED NOTION PAGES CONTENT ---\n"
                f"{linked_notion_content.strip()}\n"
                f"--- END REFERENCED NOTION PAGES CONTENT ---"
            )
        
        # SIMPLIFIED: Just create a system message and user message
        messages = [
            {"role": "system", "content": system_prompt_content}
        ]
        
        # Add conversation history if available
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add the user's prompt
        user_instruction = (
            f"Use the context above to respond to the user's message: \"{prompt}\".\n"
            f"Provide a concise, helpful response as if participating in the same Slack conversation.\n"
            f"IMPORTANT: Respond in {language_preference}."
        )
        messages.append({"role": "user", "content": user_instruction})
        
        # Log the system prompt (first 500 chars) to help with debugging
        logger.debug(f"System prompt (first 500 chars): {system_prompt_content[:500]}...")
        logger.info(f"Language preference prominently featured: {language_preference}")
        
        # SIMPLIFIED: Use default token limits
        max_context_tokens = settings.max_context_tokens_general - settings.max_tokens_response
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