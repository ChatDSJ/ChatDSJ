import unittest
import os
import logging
from unittest.mock import MagicMock, patch
from app.slack.app import get_chatgpt_response, openai_client, SYSTEM_PROMPT

class TestChatGPTResponse(unittest.TestCase):
    def setUp(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Test data
        self.conversation_history = "User1: Hello everyone\nUser2: How's it going?\nUser1: I'm working on a project"
        self.current_message = "@ChatDSJ Can you help me with my Python code?"

    def test_openai_client_initialization(self):
        """Test that the OpenAI client is properly initialized"""
        self.assertIsNotNone(openai_client, "OpenAI client should be initialized")
        
    def test_get_chatgpt_response_with_web_search_tool(self):
        """Test the actual functionality that's failing in production - using web_search tool"""
        # This test reproduces the exact error happening in production
        response = get_chatgpt_response(self.conversation_history, self.current_message)
        self.logger.info(f"Response: {response}")
        self.assertNotEqual(response, "I'm having trouble thinking right now. Please try again later.",
                        "Should get a successful response now that the tools parameter is fixed")
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 0)
        
    def test_get_chatgpt_response_without_tools(self):
        """Test without the problematic tools parameter"""
        with patch('app.slack.app.openai_client.chat.completions.create') as mock_create:
            # Mock a successful response without using the problematic tools parameter
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "This is a test response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_create.return_value = mock_response
            
            # Call the function
            response = get_chatgpt_response(self.conversation_history, self.current_message)
            
            # Verify the response
            self.assertEqual(response, "This is a test response", 
                           "Should get successful response when tools parameter is not used")
            
            # Verify the call parameters
            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            self.assertEqual(call_args['model'], 'gpt-4o')
            self.assertEqual(call_args['messages'][0]['role'], 'system')
            self.assertEqual(call_args['messages'][0]['content'], SYSTEM_PROMPT)
            self.assertEqual(call_args['messages'][1]['role'], 'user')
            self.assertTrue('Conversation history' in call_args['messages'][1]['content'])
            self.assertTrue('Current message' in call_args['messages'][1]['content'])
            
    @patch('app.slack.app.openai_client')
    def test_get_chatgpt_response_exception(self, mock_openai_client):
        """Test that exceptions are properly handled"""
        # Setup the mock to raise an exception
        mock_chat = MagicMock()
        mock_openai_client.chat = mock_chat
        mock_completions = MagicMock()
        mock_chat.completions = mock_completions
        mock_completions.create.side_effect = Exception("Test exception")
        
        # Call the function
        response = get_chatgpt_response(self.conversation_history, self.current_message)
        
        # Verify the response
        self.assertEqual(response, "I'm having trouble thinking right now. Please try again later.",
                        "Should get error message when exception occurs")

if __name__ == '__main__':
    unittest.main()
