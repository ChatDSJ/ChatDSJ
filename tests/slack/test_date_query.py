import unittest
import os
import logging
import re
from unittest.mock import MagicMock, patch
from app.slack.app import handle_mention, get_openai_response, get_channel_history, format_conversation_history_for_openai, bot_user_id

class TestDateQuery(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.event = {
            "channel": "C12345",
            "user": "U12345",
            "ts": "1617984000.000100",
            "text": "<@U08N3EFH6SE> what day is it?"
        }
        
        self.mock_client = MagicMock()
        self.mock_say = MagicMock()
        
        self.mock_messages = [
            {"user": "U12345", "text": "Hello everyone", "ts": "1617983900.000100"},
            {"user": "U67890", "text": "How's it going?", "ts": "1617983950.000100"},
            {"user": "U12345", "text": "<@U08N3EFH6SE> what day is it?", "ts": "1617983980.000100"}
        ]
        
        self.mock_user_info = {
            "user": {
                "real_name": "Test User"
            }
        }
    
    def test_date_query_response(self):
        """Test the bot's response to 'what day is it?' query"""
        self.mock_client.conversations_history.return_value = {"messages": self.mock_messages}
        self.mock_client.users_info.return_value = self.mock_user_info
        
        messages = get_channel_history(self.mock_client, "C12345", limit=1000)
        
        conversation_history = format_conversation_history_for_openai(messages, self.mock_client)
        
        prompt = "what day is it?"
        response_text, usage = get_openai_response(conversation_history, prompt, web_search=True)
        
        self.logger.info(f"OpenAI Response to 'what day is it?': {response_text}")
        
        self.assertEqual(response_text, "I don't have a specific answer for that right now.", "Expected fallback message for 'what day is it?' query")
        
    def test_date_query_full_flow(self):
        """Test the entire flow from receiving a 'what day is it?' mention to sending a response"""
        self.mock_client.conversations_history.return_value = {"messages": self.mock_messages}
        self.mock_client.users_info.return_value = self.mock_user_info
        
        handle_mention(self.event, self.mock_say, self.mock_client, self.logger)
        
        self.mock_say.assert_called_once()
        call_kwargs = self.mock_say.call_args[1] if self.mock_say.call_args and self.mock_say.call_args[1] else {}
        call_text = call_kwargs.get('text', '')
        self.logger.info(f"Bot response to 'what day is it?': {call_text}")
        
        self.assertEqual(call_text, "I don't have a specific answer for that right now.")

if __name__ == '__main__':
    unittest.main()
