import unittest
from unittest.mock import MagicMock, patch, AsyncMock
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
        self.mock_settings.openai_system_prompt = "You are a helpful assistant."
        
        self.mock_config.return_value = self.mock_settings
        
        # Patch the OpenAI clients
        self.openai_patch = patch('handler.openai_service.OpenAI')
        self.async_openai_patch = patch('handler.openai_service.AsyncOpenAI')
        
        self.mock_openai = self.openai_patch.start()
        self.mock_async_openai = self.async_openai_patch.start()
        
        # Create mock client instances
        self.mock_client = MagicMock()
        self.mock_async_client = MagicMock()
        
        self.mock_openai.return_value = self.mock_client
        self.mock_async_openai.return_value = self.mock_async_client
        
        # Create OpenAIService instance
        self.service = OpenAIService()

    def tearDown(self):
        self.config_patch.stop()
        self.openai_patch.stop()
        self.async_openai_patch.stop()

    def test_initialization(self):
        """Test service initialization with proper config."""
        self.assertEqual(self.service.api_key, "dummy-api-key")
        self.assertEqual(self.service.model, "gpt-4o")
        self.assertEqual(self.service.max_tokens, 1500)
        self.assertIsNotNone(self.service.notion_context_manager)
        self.assertIsNotNone(self.service.client)
        self.assertIsNotNone(self.service.async_client)

    def test_is_available(self):
        """Test service availability check."""
        # Should be available with mock clients
        self.assertTrue(self.service.is_available())
        
        # Set clients to None to test unavailability
        self.service.client = None
        self.assertFalse(self.service.is_available())
        
        # Reset client but set async_client to None
        self.service.client = self.mock_client
        self.service.async_client = None
        self.assertFalse(self.service.is_available())

    @patch('handler.openai_service.count_messages_tokens')
    @patch('handler.openai_service.ensure_messages_within_limit')
    def test_prepare_messages(self, mock_ensure_limit, mock_count_tokens):
        """Test message preparation for API calls."""
        # Setup mocks
        mock_count_tokens.return_value = 100
        mock_ensure_limit.return_value = [{"role": "user", "content": "test"}]
        
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
        self.assertIn("SYSTEM INSTRUCTIONS", content)
        
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

    @patch('handler.openai_service.count_messages_tokens')
    def test_get_completion_success(self, mock_count_tokens):
        """Test successful completion generation."""
        # Setup mocks
        mock_count_tokens.return_value = 50
        
        # Mock the client response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test response"
        mock_response.usage.model_dump.return_value = {"total_tokens": 150}
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Call get_completion
        response, usage = self.service.get_completion("Test prompt")
        
        # Verify response and usage
        self.assertEqual(response, "This is a test response")
        self.assertEqual(usage["total_tokens"], 150)
        
        # Verify client was called
        self.mock_client.chat.completions.create.assert_called_once()


class TestOpenAIServiceAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Patch config
        self.config_patch = patch('handler.openai_service.get_settings')
        self.mock_config = self.config_patch.start()
        
        # Setup mock settings
        self.mock_settings = MagicMock()
        self.mock_settings.openai_api_key.get_secret_value.return_value = "dummy-api-key"
        self.mock_settings.openai_model = "gpt-4o"
        self.mock_settings.max_tokens_response = 1500
        self.mock_settings.openai_system_prompt = "You are a helpful assistant."
        
        self.mock_config.return_value = self.mock_settings
        
        # Patch the OpenAI clients
        self.openai_patch = patch('handler.openai_service.OpenAI')
        self.async_openai_patch = patch('handler.openai_service.AsyncOpenAI')
        
        self.mock_openai = self.openai_patch.start()
        self.mock_async_openai = self.async_openai_patch.start()
        
        # Create mock client instances
        self.mock_client = MagicMock()
        self.mock_async_client = AsyncMock()
        
        self.mock_openai.return_value = self.mock_client
        self.mock_async_openai.return_value = self.mock_async_client
        
        # Create OpenAIService instance
        self.service = OpenAIService()

    async def asyncTearDown(self):
        self.config_patch.stop()
        self.openai_patch.stop()
        self.async_openai_patch.stop()

    @patch('handler.openai_service.count_messages_tokens')
    async def test_get_completion_async_success(self, mock_count_tokens):
        """Test successful async completion generation."""
        # Setup mocks
        mock_count_tokens.return_value = 50
        
        # Mock the async client response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is an async test response"
        mock_response.usage.model_dump.return_value = {"total_tokens": 150}
        
        self.mock_async_client.chat.completions.create.return_value = mock_response
        
        # Call get_completion_async
        response, usage = await self.service.get_completion_async("Test prompt")
        
        # Verify response and usage
        self.assertEqual(response, "This is an async test response")
        self.assertEqual(usage["total_tokens"], 150)
        
        # Verify async client was called
        self.mock_async_client.chat.completions.create.assert_called_once()

    @patch('handler.openai_service.count_messages_tokens')
    async def test_get_completion_async_timeout(self, mock_count_tokens):
        """Test async completion with timeout."""
        # Setup mocks
        mock_count_tokens.return_value = 50
        
        # Mock the async client to raise TimeoutError
        self.mock_async_client.chat.completions.create.side_effect = asyncio.TimeoutError()
        
        # Call get_completion_async
        response, usage = await self.service.get_completion_async("Test prompt", timeout=1.0)
        
        # Should get timeout message
        self.assertIn("timed out", response)
        self.assertIsNone(usage)

    @patch('handler.openai_service.count_messages_tokens')
    async def test_get_completion_async_error(self, mock_count_tokens):
        """Test async completion with general error."""
        # Setup mocks
        mock_count_tokens.return_value = 50
        
        # Mock the async client to raise an exception
        self.mock_async_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Call get_completion_async
        response, usage = await self.service.get_completion_async("Test prompt")
        
        # Should get error message
        self.assertIn("encountered an error", response)
        self.assertIsNone(usage)


if __name__ == '__main__':
    unittest.main()
