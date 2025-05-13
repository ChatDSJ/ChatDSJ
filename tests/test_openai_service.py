import unittest
from unittest.mock import MagicMock, patch
from handler.openai_service import OpenAIService

class TestOpenAIService(unittest.TestCase):
    def setUp(self):
        # Patch config to avoid actual API calls
        self.config_patch = patch('handler.openai_service.get_settings')
        self.mock_config = self.config_patch.start()
        
        # Setup mock settings
        self.mock_settings = MagicMock()
        self.mock_settings.openai_api_key.get_secret_value.return_value = "dummy-api-key"
        self.mock_settings.openai_model = "gpt-4o"
        self.mock_settings.max_tokens_response = 1500
        
        self.mock_config.return_value = self.mock_settings
        
        # Create OpenAIService instance
        self.service = OpenAIService()
        self.service.client = MagicMock()
        self.service.async_client = MagicMock()
    
    def tearDown(self):
        self.config_patch.stop()
    
    def test_initialization(self):
        """Test service initialization with proper config."""
        self.assertEqual(self.service.api_key, "dummy-api-key")
        self.assertEqual(self.service.model, "gpt-4o")
        self.assertEqual(self.service.max_tokens, 1500)
        self.assertIsNotNone(self.service.notion_context_manager)
    
    def test_is_available(self):
        """Test service availability check."""
        # Should be available with mock clients
        self.assertTrue(self.service.is_available())
        
        # Set clients to None to test unavailability
        self.service.client = None
        self.assertFalse(self.service.is_available())
        
        # Reset client but set async_client to None
        self.service.client = MagicMock()
        self.service.async_client = None
        self.assertFalse(self.service.is_available())
    
    def test_prepare_messages(self):
        """Test message preparation for API calls."""
        prompt = "How are you?"
        conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        user_context = "This user prefers short responses."
        
        # Test with all components
        messages = self.service._prepare_messages(
            prompt=prompt,
            conversation_history=conversation_history,
            user_specific_context=user_context
        )
        
        # Should be a single message with all content combined
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        
        # Content should contain all components
        content = messages[0]["content"]
        self.assertIn(prompt, content)
        self.assertIn("short responses", content)
        
        # Test without optional components
        basic_messages = self.service._prepare_messages(prompt="Just a prompt")
        self.assertEqual(len(basic_messages), 1)
        self.assertIn("Just a prompt", basic_messages[0]["content"])
    
    @patch('handler.openai_service.ensure_messages_within_limit')
    def test_message_truncation(self, mock_ensure_limit):
        """Test that messages are truncated if needed."""
        # Setup mock to return the same messages
        mock_ensure_limit.return_value = [{"role": "user", "content": "Hello"}]
        
        messages = self.service._prepare_messages("Hello")
        
        # Verify truncation was attempted
        mock_ensure_limit.assert_called_once()
    
    def test_calculate_cost(self):
        """Test cost calculation logic."""
        # Test with gpt-4o model
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        
        cost = self.service._calculate_cost(usage)
        
        # Expected cost is (10/1M * $5) + (20/1M * $15)
        expected_cost = (10 / 1_000_000 * 5.0) + (20 / 1_000_000 * 15.0)
        self.assertEqual(cost, expected_cost)
        
        # Test with empty usage
        self.assertEqual(self.service._calculate_cost({}), 0.0)
        
        # Test with unknown model
        self.service.model = "unknown-model"
        self.assertEqual(self.service._calculate_cost(usage), 0.0)
    
    def test_update_usage_tracking(self):
        """Test updating usage statistics."""
        # Initial stats
        self.assertEqual(self.service.usage_stats["prompt_tokens"], 0)
        
        # Update with usage
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
        
        self.service._update_usage_tracking(usage)
        
        # Verify stats were updated
        self.assertEqual(self.service.usage_stats["prompt_tokens"], 100)
        self.assertEqual(self.service.usage_stats["completion_tokens"], 50)
        self.assertEqual(self.service.usage_stats["total_tokens"], 150)
        
        # Cost should also be updated
        self.assertGreater(self.service.usage_stats["total_cost"], 0)
    
    @patch('handler.openai_service.OpenAIService.get_completion')
    def test_get_completion_success(self, mock_get_completion):
        """Test successful completion generation."""
        mock_get_completion.return_value = ("This is a test response", {"total_tokens": 150})
        
        response, usage = self.service.get_completion("Test prompt")
        
        # Verify response and usage
        self.assertEqual(response, "This is a test response")
        self.assertEqual(usage["total_tokens"], 150)
        
        # Verify correct arguments
        mock_get_completion.assert_called_once_with(
            prompt="Test prompt",
            conversation_history=None,
            user_specific_context=None,
            linked_notion_content=None,
            system_prompt=None,
            max_tokens=None
        )
    
    @patch('handler.openai_service.OpenAIService._prepare_messages')
    def test_get_completion_with_error(self, mock_prepare):
        """Test handling of errors during completion."""
        # Setup client to raise exception
        mock_prepare.return_value = [{"role": "user", "content": "Test"}]
        self.service.client.chat.completions.create.side_effect = Exception("API error")
        
        # Call should raise the exception since it has @retry
        with self.assertRaises(Exception):
            self.service.get_completion("Test prompt")
        
        # Error count should be incremented
        self.assertEqual(self.service.usage_stats["error_count"], 1)

if __name__ == '__main__':
    unittest.main()