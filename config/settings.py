from typing import Optional
from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Add this to ignore extra fields and fix the validation error
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8", 
        case_sensitive=False,
        extra='ignore'  # This fixes your validation error
    )
    
    # Slack Configuration
    slack_bot_token: SecretStr = Field(..., description="Slack Bot User OAuth Token")
    slack_signing_secret: SecretStr = Field(..., description="Slack Signing Secret")
    slack_app_token: SecretStr = Field(..., description="Slack App-Level Token for Socket Mode")

    # OpenAI Configuration
    openai_api_key: SecretStr = Field(..., description="OpenAI API Key")
    openai_model: str = Field("gpt-4o", description="OpenAI Model to use")
    
    # NEW: Multi-LLM Configuration
    anthropic_api_key: Optional[SecretStr] = Field(None, description="Anthropic API Key")
    google_api_key: Optional[SecretStr] = Field(None, description="Google API Key")
    
    # NEW: Model preferences
    anthropic_model: str = Field("claude-3-5-sonnet-20241022", description="Anthropic Model to use")
    google_model: str = Field("gemini-2.0-flash-exp", description="Google Model to use")

    openai_system_prompt: str = Field(
        "You are an assistant embedded in a Slack channel with access to both stored user information and conversation history. "
        
        "CRITICAL: When answering questions about user preferences, interests, facts, or what they like/love/enjoy, you MUST synthesize information from ALL sources: "
        "1. Stored user profile information (their persistent facts and preferences) "
        "2. Recent conversation messages (what they've mentioned in any recent messages) "
        
        "Both sources are equally important. Recent conversation adds to stored information - never ignore something the user recently told you. "
        
        "Your primary job is to answer the user's most recent question directly and concisely while incorporating relevant information from all available sources.",
        description="System prompt for OpenAI",
    )

    # Notion Configuration
    notion_api_token: Optional[SecretStr] = Field(None, description="Notion API Token")
    notion_user_db_id: Optional[str] = Field(None, description="Notion User Database ID")

    # Application Configuration
    log_level: str = Field("INFO", description="Log level")
    environment: str = Field("development", description="Environment (development, production)")
    
    # ENHANCED: Task-specific token limits
    max_tokens_response: int = Field(1500, description="Maximum tokens for response generation")
    max_context_tokens_general: int = Field(100000, description="Max context tokens for general tasks")
    
    max_message_history: int = Field(1000, description="Maximum number of messages to fetch from history")
    
    # Caching Configuration
    cache_ttl: int = Field(300, description="Cache time-to-live in seconds")
    cache_max_size: int = Field(1000, description="Maximum number of items in cache")

@lru_cache()
def get_settings():
    """Get application settings, cached for performance."""
    return Settings()