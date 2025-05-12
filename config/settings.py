from typing import Optional
from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Slack Configuration
    slack_bot_token: SecretStr = Field(..., description="Slack Bot User OAuth Token")
    slack_signing_secret: SecretStr = Field(..., description="Slack Signing Secret")
    slack_app_token: SecretStr = Field(..., description="Slack App-Level Token for Socket Mode")

    # OpenAI Configuration
    openai_api_key: SecretStr = Field(..., description="OpenAI API Key")
    openai_model: str = Field("gpt-4o", description="OpenAI Model to use")
    openai_system_prompt: str = Field(
        "You are an assistant embedded in a Slack channel. Your primary job is to answer "
        "the user's most recent question directly and concisely. "
        "Review the provided message history to understand the immediate context of the "
        "user's question. "
        "If the user's question is *explicitly about past discussions* in the channel "
        "(e.g., 'was X discussed before?', 'what did Y say about Z?'), "
        "then you should thoroughly search the history to answer.",
        description="System prompt for OpenAI",
    )

    # Notion Configuration
    notion_api_token: Optional[SecretStr] = Field(None, description="Notion API Token")
    notion_user_db_id: Optional[str] = Field(None, description="Notion User Database ID")

    # Application Configuration
    log_level: str = Field("INFO", description="Log level")
    environment: str = Field("development", description="Environment (development, production)")
    max_tokens_response: int = Field(1500, description="Maximum tokens for response generation")
    max_message_history: int = Field(1000, description="Maximum number of messages to fetch from history")
    
    # Caching Configuration
    cache_ttl: int = Field(300, description="Cache time-to-live in seconds")
    cache_max_size: int = Field(1000, description="Maximum number of items in cache")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings():
    """Get application settings, cached for performance."""
    return Settings()