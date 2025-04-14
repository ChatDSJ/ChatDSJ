import os
import logging
from unittest.mock import MagicMock
from dotenv import load_dotenv
from app.slack.app import handle_mention, get_openai_response, get_channel_history, format_conversation_history_for_openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def test_bot_response_without_slack():
    """Test the bot's response generation without sending actual Slack messages"""
    event = {
        "channel": "C12345",
        "user": "U12345",
        "ts": "1617984000.000100",
        "text": "<@U08N3EFH6SE> what day is it?"
    }
    
    mock_client = MagicMock()
    mock_say = MagicMock()
    
    mock_messages = [
        {"user": "U12345", "text": "Hello everyone", "ts": "1617983900.000100"},
        {"user": "U67890", "text": "How's it going?", "ts": "1617983950.000100"},
        {"user": "U12345", "text": "<@U08N3EFH6SE> what day is it?", "ts": "1617983980.000100"}
    ]
    
    mock_user_info = {
        "user": {
            "real_name": "Test User"
        }
    }
    
    mock_client.conversations_history.return_value = {"messages": mock_messages}
    mock_client.users_info.return_value = mock_user_info
    
    handle_mention(event, mock_say, mock_client, logger)
    
    call_kwargs = mock_say.call_args[1] if mock_say.call_args and mock_say.call_args[1] else {}
    call_text = call_kwargs.get('text', '')
    logger.info(f"Bot response to 'what day is it?': {call_text}")
    
    event["text"] = "<@U08N3EFH6SE> What was a positive news story from today?"
    mock_messages[-1]["text"] = "<@U08N3EFH6SE> What was a positive news story from today?"
    mock_client.conversations_history.return_value = {"messages": mock_messages}
    
    handle_mention(event, mock_say, mock_client, logger)
    
    call_kwargs = mock_say.call_args[1] if mock_say.call_args and mock_say.call_args[1] else {}
    call_text = call_kwargs.get('text', '')
    logger.info(f"Bot response to 'What was a positive news story from today?': {call_text}")

if __name__ == "__main__":
    test_bot_response_without_slack()
