import pytest
from utils.token_management import (
    count_tokens, 
    count_messages_tokens, 
    ensure_messages_within_limit,
    truncate_text_to_token_limit,
    calculate_max_tokens_for_completion,
    split_text_into_chunks
)

def test_count_tokens():
    """Test the token counting function."""
    # Basic token counting
    assert count_tokens("Hello world") == 2
    assert count_tokens("") == 0
    
    # More complex text
    long_text = "This is a longer piece of text that should have more tokens."
    assert count_tokens(long_text) > 10

def test_count_messages_tokens():
    """Test counting tokens in a messages array."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "How can I help you today?"}
    ]
    
    # Count tokens
    token_count = count_messages_tokens(messages)
    
    # Should be greater than sum of individual components due to formatting
    individual_sum = (
        count_tokens(messages[0]["content"]) +
        count_tokens(messages[1]["content"]) +
        count_tokens(messages[2]["content"])
    )
    
    assert token_count > individual_sum

def test_ensure_messages_within_limit():
    """Test limiting messages to a token budget."""
    # Create a long conversation
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Add 10 exchanges (20 messages)
    for i in range(10):
        messages.append({"role": "user", "content": f"This is message {i} from the user with some extra text to use tokens."})
        messages.append({"role": "assistant", "content": f"This is response {i} from the assistant with additional text."})
    
    # Limit to a small token budget
    max_tokens = 100
    limited_messages = ensure_messages_within_limit(messages, max_tokens=max_tokens)
    
    # Count tokens in the limited messages
    token_count = count_messages_tokens(limited_messages)
    
    # Should be less than or equal to the limit
    assert token_count <= max_tokens
    
    # System message should be preserved
    assert limited_messages[0]["role"] == "system"
    
    # Most recent messages should be preserved when preserve_latest_messages is enabled
    if len(limited_messages) > 1:
        last_original = messages[-1]["content"]
        assert any(msg["content"] == last_original for msg in limited_messages)

def test_calculate_max_tokens_for_completion():
    """Test calculating the maximum tokens available for completion."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello world"},
    ]
    
    # Calculate max completion tokens
    max_tokens = calculate_max_tokens_for_completion(
        messages,
        max_total_tokens=4096
    )
    
    # Should be positive and less than max_total_tokens
    assert max_tokens > 0
    assert max_tokens < 4096
    
    # Adding more messages should reduce available tokens
    more_messages = messages + [
        {"role": "assistant", "content": "How can I help you today?"},
        {"role": "user", "content": "I need assistance with a project."}
    ]
    
    max_tokens_less = calculate_max_tokens_for_completion(
        more_messages,
        max_total_tokens=4096
    )
    
    assert max_tokens_less < max_tokens

def test_truncate_text_to_token_limit():
    """Test truncating text to fit within token limits."""
    text = "This is a sample text that will be truncated to a smaller token count."
    truncated = truncate_text_to_token_limit(text, max_tokens=5)
    
    # Check that truncated text has fewer tokens
    assert count_tokens(truncated) <= 5
    # Check that truncated text is a prefix of original
    assert text.startswith(truncated)
    
    # Test with text that's already within limits
    short_text = "Short text"
    result = truncate_text_to_token_limit(short_text, max_tokens=10)
    assert result == short_text

def test_split_text_into_chunks():
    """Test splitting text into token-sized chunks."""
    # Create a longer text
    long_text = " ".join(["chunk"] * 100)  # Will be more than 100 tokens
    
    # Split into chunks of ~10 tokens
    chunks = split_text_into_chunks(long_text, max_tokens_per_chunk=10)
    
    # Verify each chunk is within limits
    for chunk in chunks:
        assert count_tokens(chunk) <= 10
    
    # Verify recombining chunks gives original (minus possible whitespace differences)
    recombined = " ".join(chunks)
    assert recombined.replace("  ", " ").strip() == long_text.replace("  ", " ").strip()
    
    # Test with text that fits in one chunk
    short_text = "This is a short text"
    chunks = split_text_into_chunks(short_text, max_tokens_per_chunk=10)
    assert len(chunks) == 1
    assert chunks[0] == short_text