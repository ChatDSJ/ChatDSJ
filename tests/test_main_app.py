import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from fastapi.testclient import TestClient
from main import app, handle_mention, services

class TestMainApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Sample Slack event
        self.sample_event = {
            "channel": "C12345",
            "user": "U12345",
            "ts": "1617984000.000100",
            "text": "<@BOT123> how are you?",
            "type": "app_mention"
        }
    
    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/healthz")
        
        # Should return 200 OK
        self.assertEqual(response.status_code, 200)
        
        # Should include service statuses
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("services", data)
        self.assertIn("slack", data["services"])
        self.assertIn("notion", data["services"])
        self.assertIn("openai", data["services"])
    
    @patch('services.slack_service.SlackService.is_available')
    @patch('services.cached_notion_service.CachedNotionService.is_available')
    @patch('handler.openai_service.OpenAIService.is_available')
    def test_health_check_with_service_status(self, mock_openai, mock_notion, mock_slack):
        """Test health check reflects service availability."""
        # Configure mock services
        mock_slack.return_value = True
        mock_notion.return_value = False  # Notion unavailable
        mock_openai.return_value = True
        
        # Make request
        response = self.client.get("/healthz")
        data = response.json()
        
        # Should show correct service statuses
        self.assertEqual(data["services"]["slack"], True)
        self.assertEqual(data["services"]["notion"], False)
        self.assertEqual(data["services"]["openai"], True)
    
    def test_test_openai_endpoint(self):
        """Test the OpenAI test endpoint."""
        # Mock OpenAI service
        with patch('handler.openai_service.OpenAIService.get_completion_async') as mock_completion:
            mock_completion.return_value = ("Hello, world!", {"total_tokens": 10})
            
            # Make request
            response = self.client.get("/test-openai")
            
            # Should return 200 OK
            self.assertEqual(response.status_code, 200)
            
            # Should include the response
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["response"], "Hello, world!")
            self.assertIn("usage", data)
    
    def test_test_openai_endpoint_error(self):
        """Test OpenAI test endpoint with error."""
        # Mock OpenAI service to be unavailable
        with patch('handler.openai_service.OpenAIService.is_available', return_value=False):
            # Make request
            response = self.client.get("/test-openai")
            
            # Should return 200 OK (endpoint itself works)
            self.assertEqual(response.status_code, 200)
            
            # Should report error
            data = response.json()
            self.assertEqual(data["status"], "error")
            self.assertIn("message", data)
            self.assertIn("not initialized", data["message"])

class TestHandleMention(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock services
        self.mock_slack = MagicMock()
        self.mock_slack.clean_prompt_text = MagicMock(return_value="how are you?")
        self.mock_slack.send_ephemeral_message = MagicMock(return_value=True)
        self.mock_slack.send_message = MagicMock(return_value={"ok": True, "ts": "123456.789"})
        self.mock_slack.update_channel_stats = MagicMock()
        self.mock_slack.get_user_display_name = AsyncMock(return_value="Test User")
        
        self.mock_notion = MagicMock()
        self.mock_notion.handle_nickname_command = AsyncMock(return_value=(None, False))
        
        self.mock_router = MagicMock()
        self.mock_router.route_action = AsyncMock(return_value=MagicMock(
            success=True,
            message="I'm doing well, thank you!",
            thread_ts=None
        ))
        
        # Patch ActionRouter
        self.router_patch = patch('main.ActionRouter', return_value=self.mock_router)
        self.mock_router_class = self.router_patch.start()
        
        # Sample event
        self.event = {
            "channel": "C12345",
            "user": "U12345",
            "ts": "1617984000.000100",
            "text": "<@BOT123> how are you?",
            "type": "app_mention"
        }
        
        # Mock say and client
        self.mock_say = MagicMock()
        self.mock_client = MagicMock()
    
    async def asyncTearDown(self):
        self.router_patch.stop()
    
    async def test_handle_mention_normal_flow(self):
        """Test the normal handle_mention flow."""
        # Call handle_mention
        response = await handle_mention(self.event, self.mock_say, self.mock_client)
        
        # Verify ephemeral message was sent
        self.mock_slack.send_ephemeral_message.assert_called_once_with(
            "C12345",
            "U12345",
            "I heard you! I'm working on a response... 🧠"
        )
        
        # Verify nickname command was checked
        self.mock_notion.handle_nickname_command.assert_called_once()
        
        # Verify action was routed
        self.mock_router.route_action.assert_called_once()
        
        # Verify message was sent
        self.mock_slack.send_message.assert_called_once_with(
            "C12345",
            "I'm doing well, thank you!",
            None  # thread_ts is None in this test
        )
        
        # Verify channel stats were updated
        self.mock_slack.update_channel_stats.assert_called_once_with(
            "C12345",
            "U12345",
            "1617984000.000100"
        )
    
    async def test_handle_mention_nickname_command(self):
        """Test handle_mention with nickname command."""
        # Configure nickname mock to respond
        self.mock_notion.handle_nickname_command.return_value = ("I'll call you Test User!", True)
        
        # Call handle_mention
        response = await handle_mention(self.event, self.mock_say, self.mock_client)
        
        # Verify nickname command was checked
        self.mock_notion.handle_nickname_command.assert_called_once()
        
        # Verify action was NOT routed (since nickname handled it)
        self.mock_router.route_action.assert_not_called()
        
        # Verify nickname response was sent
        self.mock_slack.send_message.assert_called_once_with(
            "C12345",
            "I'll call you Test User!",
            None  # thread_ts
        )
        
        # Verify channel stats were updated
        self.mock_slack.update_channel_stats.assert_called_once()
    
    async def test_handle_mention_with_error(self):
        """Test handle_mention with error during processing."""
        # Configure router to raise exception
        self.mock_router.route_action.side_effect = Exception("Test error")
        
        # Call handle_mention
        with self.assertLogs(level='ERROR') as cm:
            response = await handle_mention(self.event, self.mock_say, self.mock_client)
        
        # Verify error was logged
        self.assertIn("Error handling mention", cm.output[0])
        
        # Verify error message was sent
        self.mock_say.assert_called_once()
        call_args = self.mock_say.call_args[1]
        self.assertIn("unexpected error", call_args["text"].lower())

if __name__ == '__main__':
    unittest.main()