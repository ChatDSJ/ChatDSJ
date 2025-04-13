import unittest
import os
import logging
import re
from unittest.mock import MagicMock, patch
from app.slack.app import handle_mention, get_openai_response, get_channel_history, format_conversation_history_for_openai, bot_user_id

class TestResponsesAPI(unittest.TestCase):
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
    
    def test_responses_api_date_query(self):
        """Test the responses API with 'what day is it?' query"""
        self.mock_client.conversations_history.return_value = {"messages": self.mock_messages}
        self.mock_client.users_info.return_value = self.mock_user_info
        
        messages = get_channel_history(self.mock_client, "C12345", limit=1000)
        
        conversation_history = format_conversation_history_for_openai(messages, self.mock_client)
        
        prompt = "what day is it?"
        response_text, usage = get_openai_response(conversation_history, prompt, web_search=True)
        
        self.logger.info(f"OpenAI Response to 'what day is it?': {response_text}")
        
        self.assertIsNotNone(response_text)
        self.assertTrue(len(response_text) > 0)
        self.assertNotEqual(response_text, "I'm having trouble thinking right now. Please try again later.")
        
    def test_responses_api_news_query(self):
        """Test the responses API with 'What was a positive news story from today?' query"""
        self.mock_client.conversations_history.return_value = {"messages": self.mock_messages}
        self.mock_client.users_info.return_value = self.mock_user_info
        
        messages = get_channel_history(self.mock_client, "C12345", limit=1000)
        
        conversation_history = format_conversation_history_for_openai(messages, self.mock_client)
        
        prompt = "What was a positive news story from today?"
        response_text, usage = get_openai_response(conversation_history, prompt, web_search=True)
        
        self.logger.info(f"OpenAI Response to 'What was a positive news story from today?': {response_text}")
        
        self.assertIsNotNone(response_text)
        self.assertTrue(len(response_text) > 0)
        self.assertNotEqual(response_text, "I'm having trouble thinking right now. Please try again later.")
        
    def test_responses_api_full_flow(self):
        """Test the entire flow from receiving a mention to sending a response using the responses API"""
        self.mock_client.conversations_history.return_value = {"messages": self.mock_messages}
        self.mock_client.users_info.return_value = self.mock_user_info
        
        with patch('app.slack.app.get_openai_response') as mock_get_openai_response:
            mock_get_openai_response.return_value = ("Today is Sunday, April 13, 2025.", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
            
            handle_mention(self.event, self.mock_say, self.mock_client, self.logger)
            
            self.mock_say.assert_called_once()
            call_kwargs = self.mock_say.call_args[1] if self.mock_say.call_args and self.mock_say.call_args[1] else {}
            call_text = call_kwargs.get('text', '')
            self.logger.info(f"Bot response to 'what day is it?': {call_text}")
            
            self.assertEqual(call_text, "Today is Sunday, April 13, 2025.")

if __name__ == '__main__':
    unittest.main()
