import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from actions.action_framework import (
    ServiceContainer, 
    ActionRequest, 
    ActionResponse, 
    Action,
    ContextResponseAction,
    TodoAction,
    RetrieveSummarizeAction,
    YoutubeSummarizeAction,
    HistoricalSearchAction,
    SearchFollowUpAction,
    ActionRouter
)
from utils.simple_history_manager import SimpleHistoryManager

class TestActionFramework(unittest.TestCase):
    def setUp(self):
        # Create mock services
        self.mock_slack = MagicMock()
        self.mock_slack.bot_user_id = "BOT123"
        self.mock_notion = MagicMock()
        self.mock_openai = MagicMock()
        self.mock_web = MagicMock()
        self.mock_youtube = MagicMock()
        
        # Create service container
        self.services = ServiceContainer(
            slack_service=self.mock_slack,
            notion_service=self.mock_notion,
            openai_service=self.mock_openai,
            web_service=self.mock_web,
            youtube_service=self.mock_youtube
        )
        
        # Sample request
        self.request = ActionRequest(
            channel_id="C12345",
            user_id="U12345",
            message_ts="123456.789",
            thread_ts=None,
            text="Hello bot, how are you?",
            prompt="how are you?"
        )

    def test_service_container_initialization(self):
        """Test service container initialization."""
        container = ServiceContainer(
            slack_service=self.mock_slack,
            notion_service=self.mock_notion,
            openai_service=self.mock_openai
        )
        
        # Check services are set correctly
        self.assertEqual(container.slack_service, self.mock_slack)
        self.assertEqual(container.notion_service, self.mock_notion)
        self.assertEqual(container.openai_service, self.mock_openai)
        self.assertIsNone(container.web_service)
        self.assertIsNone(container.youtube_service)

    def test_service_validation(self):
        """Test service validation logic."""
        # Test with all required services available
        self.mock_slack.is_available.return_value = True
        self.mock_notion.is_available.return_value = True
        
        result = self.services.validate_required_services(["slack", "notion"])
        self.assertTrue(result)
        
        # Test with missing service
        self.mock_slack.is_available.return_value = False
        
        result = self.services.validate_required_services(["slack", "notion"])
        self.assertFalse(result)


class TestContextResponseAction(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create mock services
        self.mock_slack = MagicMock()
        self.mock_slack.bot_user_id = "BOT123"
        self.mock_slack.get_user_display_name = AsyncMock(return_value="Test User")
        
        self.mock_notion = MagicMock()
        self.mock_notion.get_user_page_content = MagicMock(return_value="Test content")
        self.mock_notion.get_user_page_properties = MagicMock(return_value={})
        self.mock_notion.get_user_preferred_name = MagicMock(return_value="TestUser")
        
        self.mock_openai = MagicMock()
        self.mock_openai.get_completion_async = AsyncMock(return_value=("I'm doing well, thank you!", {"total_tokens": 10}))
        
        # Create service container
        self.services = ServiceContainer(
            slack_service=self.mock_slack,
            notion_service=self.mock_notion,
            openai_service=self.mock_openai
        )
        
        # Create action
        self.action = ContextResponseAction(self.services)
        
        # Sample request
        self.request = ActionRequest(
            channel_id="C12345",
            user_id="U12345",
            message_ts="123456.789",
            thread_ts=None,
            text="Hello bot, how are you?",
            prompt="how are you?"
        )

    async def test_can_handle(self):
        """Test that the action can handle any text."""
        self.assertTrue(self.action.can_handle("any text"))
        self.assertTrue(self.action.can_handle(""))

    async def test_get_required_services(self):
        """Test required services."""
        required = self.action.get_required_services()
        self.assertIn("slack", required)
        self.assertIn("openai", required)
        self.assertIn("notion", required)

    @patch('utils.history_manager.HistoryManager')
    @patch('utils.context_builder.get_enhanced_user_context')
    async def test_execute_normal_response(self, mock_context_builder, mock_history_manager):
        """Test normal execution path."""
        # Mock service validation
        self.services.validate_required_services = MagicMock(return_value=True)
        
        # Mock history manager
        mock_history_instance = MagicMock()
        mock_history_manager.return_value = mock_history_instance
        mock_history_instance.retrieve_and_filter_history = AsyncMock(return_value=([], {}))
        mock_history_instance.format_history_for_prompt = MagicMock(return_value="")
        
        # Mock context builder
        mock_context_builder.return_value = "User context"
        
        # Execute action
        response = await self.action.execute(self.request)
        
        # Verify response
        self.assertTrue(response.success)
        self.assertEqual(response.message, "I'm doing well, thank you!")
        self.assertEqual(response.thread_ts, self.request.thread_ts)
        
        # Verify openai service was called
        self.mock_openai.get_completion_async.assert_called_once()


class TestHistoricalSearchAction(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create mock services
        self.mock_slack = MagicMock()
        self.mock_slack.bot_user_id = "BOT123"
        self.mock_slack.get_user_display_name = AsyncMock(return_value="Test User")
        
        self.mock_openai = MagicMock()
        
        # Create service container
        self.services = ServiceContainer(
            slack_service=self.mock_slack,
            openai_service=self.mock_openai
        )
        
        # Create action
        self.action = HistoricalSearchAction(self.services)

    async def test_can_handle_search_patterns(self):
        """Test that the action can identify search queries."""
        # Should handle search patterns
        self.assertTrue(self.action.can_handle("has anyone discussed Python?"))
        self.assertTrue(self.action.can_handle("was Miami mentioned?"))
        self.assertTrue(self.action.can_handle("did anyone talk about the project?"))
        
        # Should not handle regular questions
        self.assertFalse(self.action.can_handle("How are you today?"))
        self.assertFalse(self.action.can_handle("What time is it?"))

    @patch('utils.history_manager.HistoryManager')
    async def test_execute_search(self, mock_history_manager):
        """Test search execution."""
        # Mock service validation
        self.services.validate_required_services = MagicMock(return_value=True)
        
        # Mock history manager
        mock_history_instance = MagicMock()
        mock_history_manager.return_value = mock_history_instance
        
        sample_messages = [
            {"ts": "123", "user": "U123", "text": "I love Python programming"}
        ]
        mock_history_instance.retrieve_and_filter_history = AsyncMock(
            return_value=(sample_messages, {"search_topic": "Python", "is_search": True})
        )
        mock_history_instance.format_search_results_with_threads = MagicMock(
            return_value={
                "summary": "Found 1 message about Python",
                "threads": [],
                "show_details": True
            }
        )
        
        # Create search request
        request = ActionRequest(
            channel_id="C12345",
            user_id="U12345",
            message_ts="123456.789",
            text="has anyone discussed Python?",
            prompt="has anyone discussed Python?"
        )
        
        # Execute action
        response = await self.action.execute(request)
        
        # Verify response
        self.assertTrue(response.success)
        self.assertIn("Python", response.message)


class TestTodoAction(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create mock services
        self.mock_slack = MagicMock()
        self.mock_notion = MagicMock()
        self.mock_notion.add_todo_item = AsyncMock(return_value=True)
        
        # Create service container
        self.services = ServiceContainer(
            slack_service=self.mock_slack,
            notion_service=self.mock_notion
        )
        
        # Create action
        self.action = TodoAction(self.services)

    async def test_can_handle(self):
        """Test text patterns the action can handle."""
        # Should handle TODO prefixes
        self.assertTrue(self.action.can_handle("TODO: Buy milk"))
        self.assertTrue(self.action.can_handle("Remember to TODO: Call mom"))
        
        # Should not handle other text
        self.assertFalse(self.action.can_handle("Hello there"))
        self.assertFalse(self.action.can_handle("Remember to buy milk"))

    async def test_execute_success(self):
        """Test successful TODO execution."""
        # Setup request with TODO
        request = ActionRequest(
            channel_id="C12345",
            user_id="U12345",
            message_ts="123456.789",
            text="TODO: Buy milk",
            prompt="TODO: Buy milk"
        )
        
        # Mock service validation
        self.services.validate_required_services = MagicMock(return_value=True)
        
        # Execute action
        response = await self.action.execute(request)
        
        # Verify response
        self.assertTrue(response.success)
        self.assertIn("Added to your TODO list", response.message)
        
        # Verify notion service was called
        self.mock_notion.add_todo_item.assert_called_once_with("U12345", "Buy milk")


class TestActionRouter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create mock services
        self.mock_services = MagicMock()
        
        # Create router
        self.router = ActionRouter(self.mock_services)
        
        # Sample request
        self.request = ActionRequest(
            channel_id="C12345",
            user_id="U12345",
            message_ts="123456.789",
            text="Hello",
            prompt="Hello"
        )

    async def test_get_action_for_text(self):
        """Test action selection based on text."""
        # Test TODO action selection
        action = self.router._get_action_for_text("TODO: Buy milk")
        self.assertEqual(action.__class__.__name__, "TodoAction")
        
        # Test search action selection
        action = self.router._get_action_for_text("has anyone discussed Python?")
        self.assertEqual(action.__class__.__name__, "HistoricalSearchAction")
        
        # Test default (context) action
        action = self.router._get_action_for_text("Hello world")
        self.assertEqual(action.__class__.__name__, "ContextResponseAction")

    async def test_route_action(self):
        """Test routing to the appropriate action."""
        # Mock all actions to return success
        for action in self.router.actions:
            action.execute = AsyncMock(return_value=ActionResponse(
                success=True, 
                message="Test response"
            ))
        
        # Route action
        response = await self.router.route_action(self.request)
        
        # Verify result
        self.assertTrue(response.success)
        self.assertEqual(response.message, "Test response")


if __name__ == '__main__':
    unittest.main()
