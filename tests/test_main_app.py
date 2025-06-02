import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from fastapi.testclient import TestClient

# Import the app and handle_mention function
from main import app, handle_mention


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

    def test_test_openai_endpoint(self):
        """Test the OpenAI test endpoint."""
        with patch('main.openai_service') as mock_openai:
            mock_openai.is_available.return_value = True
            mock_openai.get_completion_async = AsyncMock(return_value=("Hello, world!", {"total_tokens": 10}))
            
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
        with patch('main.openai_service') as mock_openai:
            mock_openai.is_available.return_value = False
            
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
        # Mock the services used in main.py
        self.slack_service_patch = patch('main.slack_service')
        self.notion_service_patch = patch('main.notion_service')
        self.action_router_patch = patch('main.ActionRouter')
        
        self.mock_slack = self.slack_service_patch.start()
        self.mock_notion = self.notion_service_patch.start()
        self.mock_router_class = self.action_router_patch.start()
        
        # Configure mock slack service
        self.mock_slack.clean_prompt_text.return_value = "how are you?"
        self.mock_slack.send_ephemeral_message = MagicMock(return_value=True)
        self.mock_slack.send_message = MagicMock(return_value={"ok": True, "ts": "123456.789"})
        self.mock_slack.add_reaction = MagicMock(return_value=True)
        self.mock_slack.remove_reaction = MagicMock(return_value=True)
        self.mock_slack.update_channel_stats = MagicMock()
        
        # Configure mock notion service
        self.mock_notion.memory_handler = MagicMock()
        self.mock_notion.handle_memory_instruction = MagicMock(return_value=None)
        self.mock_notion.handle_nickname_command = MagicMock(return_value=(None, False))
        
        # Configure mock router
        self.mock_router = MagicMock()
        self.mock_router.route_action = AsyncMock(return_value=MagicMock(
            success=True,
            message="I'm doing well, thank you!",
            thread_ts=None
        ))
        self.mock_router_class.return_value = self.mock_router
        
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
        self.slack_service_patch.stop()
        self.notion_service_patch.stop()
        self.action_router_patch.stop()

    async def test_handle_mention_normal_flow(self):
        """Test the normal handle_mention flow."""
        # Call handle_mention
        response = await handle_mention(self.event, self.mock_say, self.mock_client)
        
        # Verify thinking reaction was added
        self.mock_slack.add_reaction.assert_any_call("C12345", "1617984000.000100", "thinking_face")
        
        # Verify memory instruction was checked
        self.mock_notion.handle_memory_instruction.assert_called_once()
        
        # Verify nickname command was checked
        self.mock_notion.handle_nickname_command.assert_called_once()
        
        # Verify action was routed
        self.mock_router.route_action.assert_called_once()
        
        # Verify success reaction was added and thinking removed
        self.mock_slack.remove_reaction.assert_any_call("C12345", "1617984000.000100", "thinking_face")
        self.mock_slack.add_reaction.assert_any_call("C12345", "1617984000.000100", "white_check_mark")
        
        # Verify message was sent
        self.mock_slack.send_message.assert_called_once_with(
            "C12345",
            "I'm doing well, thank you!",
            None
        )
        
        # Verify channel stats were updated
        self.mock_slack.update_channel_stats.assert_called_once_with(
            "C12345",
            "U12345",
            "1617984000.000100"
        )

    async def test_handle_mention_memory_command(self):
        """Test handle_mention with memory command."""
        # Configure memory handler to respond
        self.mock_notion.handle_memory_instruction.return_value = "I've stored that fact for you."
        
        # Call handle_mention
        response = await handle_mention(self.event, self.mock_say, self.mock_client)
        
        # Verify memory command was processed
        self.mock_notion.handle_memory_instruction.assert_called_once()
        
        # Verify action was NOT routed (since memory handled it)
        self.mock_router.route_action.assert_not_called()
        
        # Verify memory response was sent
        self.mock_slack.send_message.assert_called_once_with(
            "C12345",
            "I've stored that fact for you.",
            None
        )
        
        # Verify success reaction
        self.mock_slack.add_reaction.assert_any_call("C12345", "1617984000.000100", "white_check_mark")

    async def test_handle_mention_nickname_command(self):
        """Test handle_mention with nickname command."""
        # Configure nickname handler to respond
        self.mock_notion.handle_nickname_command.return_value = ("I'll call you Test User!", True)
        
        # Call handle_mention
        response = await handle_mention(self.event, self.mock_say, self.mock_client)
        
        # Verify nickname command was processed
        self.mock_notion.handle_nickname_command.assert_called_once()
        
        # Verify action was NOT routed (since nickname handled it)
        self.mock_router.route_action.assert_not_called()
        
        # Verify nickname response was sent
        self.mock_slack.send_message.assert_called_once_with(
            "C12345",
            "I'll call you Test User!",
            None
        )

    async def test_handle_mention_action_failure(self):
        """Test handle_mention with action failure."""
        # Configure router to return failure
        self.mock_router.route_action.return_value = MagicMock(
            success=False,
            error="Test error",
            message="I encountered an error."
        )
        
        # Call handle_mention
        response = await handle_mention(self.event, self.mock_say, self.mock_client)
        
        # Verify warning reaction was added
        self.mock_slack.add_reaction.assert_any_call("C12345", "1617984000.000100", "warning")
        
        # Verify error message was sent
        self.mock_slack.send_message.assert_called_once_with(
            "C12345",
            "I encountered an error.",
            None
        )

    async def test_handle_mention_with_exception(self):
        """Test handle_mention with exception during processing."""
        # Configure router to raise exception
        self.mock_router.route_action.side_effect = Exception("Test error")
        
        # Call handle_mention
        response = await handle_mention(self.event, self.mock_say, self.mock_client)
        
        # Verify error reaction was added
        self.mock_slack.add_reaction.assert_any_call("C12345", "1617984000.000100", "x")
        
        # Verify error was sent via say
        self.mock_say.assert_called_once()
        call_args = self.mock_say.call_args[1]
        self.assertIn("unexpected error", call_args["text"].lower())


if __name__ == '__main__':
    unittest.main()
