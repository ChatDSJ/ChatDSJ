import tiktoken
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger

def get_encoder_for_model(model: str) -> tiktoken.Encoding:
    """
    Get the appropriate token encoder for a given model.
    FIXED: Use correct encoding for GPT-4o models.
    
    Args:
        model: The model name (e.g., 'gpt-4o', 'gpt-4-turbo')
        
    Returns:
        The appropriate tiktoken encoder
    """
    # Handle GPT-4o models specifically
    if model.startswith("gpt-4o"):
        return tiktoken.get_encoding("o200k_base")
    
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back based on model type
        if model.startswith("gpt-4"):
            logger.warning(f"Model {model} not found in tiktoken. Using cl100k_base for GPT-4.")
            return tiktoken.get_encoding("cl100k_base")
        else:
            logger.warning(f"Model {model} not found in tiktoken. Falling back to cl100k_base.")
            return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in the text for the specified model.
    
    Args:
        text: The text to count tokens for
        model: The model name to use for token counting
        
    Returns:
        The number of tokens in the text
    """
    if not text:
        return 0
        
    encoder = get_encoder_for_model(model)
    return len(encoder.encode(text))

def count_messages_tokens(messages: List[Dict[str, Any]], model: str = "gpt-4o") -> int:
    """
    Count the total tokens in a messages array for OpenAI.
    
    This follows OpenAI's token counting approach, accounting for message formatting.
    
    Args:
        messages: List of message objects (role, content)
        model: The model name to use for token counting
        
    Returns:
        The total number of tokens used by the messages
    """
    if not messages:
        return 0
        
    encoder = get_encoder_for_model(model)
    token_count = 0
    
    # Every reply is primed with <|start|>assistant<|message|>
    token_count += 3
    
    for message in messages:
        # Every message follows <|start|>{role}<|message|>{content}<|end|>
        # That's 4 tokens for the formatting
        token_count += 4
        
        for key, value in message.items():
            if key == "role":
                # Count tokens for the role (typically 1)
                token_count += 1
            elif key == "content" and value:
                # Count tokens for the content
                token_count += len(encoder.encode(value))
            elif key == "name" and value:
                # Count tokens for the name (if present)
                token_count += len(encoder.encode(value))
            # Additional handling for function/tool calls would go here
    
    return token_count

def truncate_text_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """
    Truncate text to fit within a token limit.
    
    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed
        model: The model name to use for token counting
        
    Returns:
        Truncated text that fits within the token limit
    """
    if not text:
        return ""
        
    encoder = get_encoder_for_model(model)
    tokens = encoder.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens and decode
    truncated_tokens = tokens[:max_tokens]
    return encoder.decode(truncated_tokens)

def ensure_messages_within_limit(
    messages: List[Dict[str, Any]], 
    model: str = "gpt-4o", 
    max_tokens: int = 100000,
    preserve_system_message: bool = True,
    preserve_latest_messages: int = 4
) -> List[Dict[str, Any]]:
    """
    Ensure messages fit within the token limit by trimming older messages.
    
    This function preserves the system message and the most recent exchanges,
    trimming older messages as needed to fit within the token limit.
    
    Args:
        messages: List of message objects (role, content)
        model: The model name to use for token counting
        max_tokens: Maximum number of tokens allowed
        preserve_system_message: Whether to always keep the system message
        preserve_latest_messages: Minimum number of recent messages to preserve
        
    Returns:
        A list of messages that fits within the token limit
    """
    if not messages:
        return messages
    
    # Get current token count
    current_tokens = count_messages_tokens(messages, model)
    
    # If already under the limit, return as-is
    if current_tokens <= max_tokens:
        return messages
    
    logger.warning(
        f"Messages exceed token limit: {current_tokens}/{max_tokens}. Trimming older messages."
    )
    
    # Extract system messages and regular messages
    system_messages = []
    regular_messages = []
    
    for msg in messages:
        if msg.get("role") == "system" and preserve_system_message:
            system_messages.append(msg)
        else:
            regular_messages.append(msg)
    
    # Calculate tokens for system messages
    system_tokens = count_messages_tokens(system_messages, model)
    
    # If system messages alone exceed the limit, we have a problem
    if system_tokens > max_tokens:
        logger.error(
            f"System messages alone exceed token limit: {system_tokens}/{max_tokens}. "
            "Consider reducing system message size."
        )
        # Try to preserve at least some of the system message
        if preserve_system_message and system_messages:
            # Truncate the system message content
            for i, msg in enumerate(system_messages):
                if msg.get("content"):
                    # Leave some room for message formatting
                    content_max_tokens = max_tokens - 10
                    msg["content"] = truncate_text_to_token_limit(
                        msg["content"], content_max_tokens, model
                    )
                    system_messages[i] = msg
            
            return system_messages
        return []  # Cannot fit anything
    
    # Start with system messages
    result = system_messages.copy()
    available_tokens = max_tokens - system_tokens
    
    # Ensure we keep the most recent messages (from end of list)
    recent_messages = regular_messages[-preserve_latest_messages:] if preserve_latest_messages > 0 else []
    older_messages = regular_messages[:-preserve_latest_messages] if preserve_latest_messages > 0 else regular_messages
    
    # Calculate tokens for recent messages
    recent_tokens = count_messages_tokens(recent_messages, model)
    
    # If recent messages alone exceed available tokens, we need to drop some
    if recent_tokens > available_tokens:
        logger.warning(
            f"Recent messages exceed available tokens: {recent_tokens}/{available_tokens}. "
            "Dropping oldest recent messages."
        )
        # Keep as many recent messages as possible, starting from the newest
        temp_result = []
        temp_tokens = 0
        
        for msg in reversed(recent_messages):
            msg_tokens = count_messages_tokens([msg], model)
            if temp_tokens + msg_tokens <= available_tokens:
                temp_result.insert(0, msg)
                temp_tokens += msg_tokens
            else:
                break
        
        return result + temp_result
    
    # We can keep all recent messages
    result.extend(recent_messages)
    available_tokens -= recent_tokens
    
    # Now add as many older messages as we can fit, starting from the newest
    for msg in reversed(older_messages):
        msg_tokens = count_messages_tokens([msg], model)
        if available_tokens - msg_tokens >= 0:
            result.insert(len(system_messages), msg)  # Insert after system messages
            available_tokens -= msg_tokens
        else:
            break
    
    # Log the results of truncation
    original_count = len(messages)
    final_count = len(result)
    dropped_count = original_count - final_count
    
    if dropped_count > 0:
        logger.info(
            f"Dropped {dropped_count} older messages to fit within token limit. "
            f"Kept {final_count}/{original_count} messages."
        )
    
    return result

def calculate_max_tokens_for_completion(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o",
    max_total_tokens: int = 8192,
    safety_margin: int = 50
) -> int:
    """
    Calculate the maximum tokens available for completion based on input messages.
    
    Args:
        messages: The input messages to count tokens for
        model: The model name to use for token counting
        max_total_tokens: The maximum total tokens for the model
        safety_margin: Additional tokens to reserve as safety margin
        
    Returns:
        The maximum number of tokens available for the completion
    """
    input_tokens = count_messages_tokens(messages, model)
    available_tokens = max_total_tokens - input_tokens - safety_margin
    
    # Ensure we have a reasonable minimum
    if available_tokens < 100:
        logger.warning(
            f"Very limited tokens available for completion: {available_tokens}. "
            "Consider reducing input size."
        )
        available_tokens = 100  # Set a minimum
    
    return max(0, available_tokens)

def split_text_into_chunks(text: str, max_tokens_per_chunk: int, model: str = "gpt-4o") -> List[str]:
    """
    Split a large text into chunks that fit within a token limit.
    
    Args:
        text: The text to split
        max_tokens_per_chunk: Maximum tokens per chunk
        model: The model name to use for token counting
        
    Returns:
        List of text chunks that each fit within the token limit
    """
    if not text:
        return []
    
    encoder = get_encoder_for_model(model)
    tokens = encoder.encode(text)
    
    if len(tokens) <= max_tokens_per_chunk:
        return [text]
    
    # Split into chunks
    chunks = []
    for i in range(0, len(tokens), max_tokens_per_chunk):
        chunk_tokens = tokens[i:i + max_tokens_per_chunk]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks