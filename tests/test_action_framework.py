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
    ActionRouter
)

class TestActionFramework(unittest.TestCase):
    def setUp(self):
        # Create mock services
        self.mock_slack = MagicMock()
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
        self.assertIsNone(container.web_service)  # Not provided
        self.assertIsNone(container.youtube_service)  # Not provided
    
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
        
        # Test with unavailable service
        self.mock_slack.is_available.return_value = True
        self.services.web_service = None
        
        result = self.services.validate_required_services(["slack", "web"])
        self.assertFalse(result)
    
    def test_action_request_model(self):
        """Test ActionRequest model validation."""
        # Valid request
        request = ActionRequest(
            channel_id="C12345",
            user_id="U12345",
            message_ts="123456.789",
            text="Hello",
            prompt="Hello"
        )
        
        self.assertEqual(request.channel_id, "C12345")
        self.assertEqual(request.user_id, "U12345")
        self.assertEqual(request.message_ts, "123456.789")
        self.assertIsNone(request.thread_ts)  # Default None
        self.assertEqual(request.text, "Hello")
        self.assertEqual(request.prompt, "Hello")
        
        # With thread_ts
        request = ActionRequest(
            channel_id="C12345",
            user_id="U12345",
            message_ts="123456.789",
            thread_ts="123456.700",
            text="Hello",
            prompt="Hello"
        )
        
        self.assertEqual(request.thread_ts, "123456.700")
    
    def test_action_response_model(self):
        """Test ActionResponse model."""
        # Success response
        response = ActionResponse(
            success=True,
            message="Hello there!",
            thread_ts="123456.789"
        )
        
        self.assertTrue(response.success)
        self.assertEqual(response.message, "Hello there!")
        self.assertEqual(response.thread_ts, "123456.789")
        self.assertIsNone(response.error)  # Default None
        
        # Error response
        response = ActionResponse(
            success=False,
            error="Something went wrong",
            message="Error occurred"
        )
        
        self.assertFalse(response.success)
        self.assertEqual(response.error, "Something went wrong")
        self.assertEqual(response.message, "Error occurred")
        self.assertIsNone(response.thread_ts)  # Default None

class TestContextResponseAction(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create mock services
        self.mock_slack = MagicMock()
        self.mock_slack.bot_user_id = "BOT123"
        self.mock_slack.clean_prompt_text = MagicMock(return_value="cleaned prompt")
        
        self.mock_notion = MagicMock()
        self.mock_notion.handle_memory_instruction = MagicMock(return_value=None)
        
        self.mock_openai = AsyncMock()
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
    
    async def test_execute_normal_response(self):
        """Test normal execution path."""
        # Mock service validation
        self.services.validate_required_services = MagicMock(return_value=True)
        
        # Execute action
        response = await self.action.execute(self.request)
        
        # Verify response
        self.assertTrue(response.success)
        self.assertEqual(response.message, "I'm doing well, thank you!")
        self.assertEqual(response.thread_ts, self.request.thread_ts)
        
        # Verify openai service was called
        self.mock_openai.get_completion_async.assert_called_once()
        
        # Notion memory check should have been attempted
        self.mock_notion.handle_memory_instruction.assert_called_once()
    
    async def test_execute_memory_response(self):
        """Test execution with memory instruction."""
        # Setup mock to return a memory response
        self.mock_notion.handle_memory_instruction = MagicMock(return_value="I've stored that fact for you.")
        
        # Mock service validation
        self.services.validate_required_services = MagicMock(return_value=True)
        
        # Execute action
        response = await self.action.execute(self.request)
        
        # Verify response
        self.assertTrue(response.success)
        self.assertEqual(response.message, "I've stored that fact for you.")
        
        # OpenAI should not be called
        self.mock_openai.get_completion_async.assert_not_called()
    
    async def test_execute_service_unavailable(self):
        """Test execution with unavailable services."""
        # Mock service validation to fail
        self.services.validate_required_services = MagicMock(return_value=False)
        
        # Execute action
        response = await self.action.execute(self.request)
        
        # Verify error response
        self.assertFalse(response.success)
        self.assertIn("Required services not available", response.error)
        
        # Services should not be called
        self.mock_openai.get_completion_async.assert_not_called()

class TestTodoAction(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create mock services
        self.mock_slack = MagicMock()
        self.mock_notion = MagicMock()
        self.mock_notion.add_todo_item = MagicMock(return_value=True)
        
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
    
    async def test_execute_no_content(self):
        """Test execution with empty TODO."""
        # Setup request with empty TODO
        request = ActionRequest(
            channel_id="C12345",
            user_id="U12345",
            message_ts="123456.789",
            text="TODO:",
            prompt="TODO:"
        )
        
        # Mock service validation
        self.services.validate_required_services = MagicMock(return_value=True)
        
        # Execute action
        response = await self.action.execute(request)
        
        # Verify error response
        self.assertFalse(response.success)
        self.assertIn("Empty TODO content", response.error)
        
        # Notion service should not be called
        self.mock_notion.add_todo_item.assert_not_called()

class TestActionRouter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create mock services
        self.mock_services = MagicMock()
        
        # Create mock actions
        self.mock_todo_action = MagicMock()
        self.mock_todo_action.name = "TodoAction"
        self.mock_todo_action.can_handle = MagicMock(return_value=False)
        self.mock_todo_action.execute = AsyncMock()
        
        self.mock_context_action = MagicMock()
        self.mock_context_action.name = "ContextResponseAction"
        self.mock_context_action.can_handle = MagicMock(return_value=True)  # Default fallback
        self.mock_context_action.execute = AsyncMock()
        
        # Create router with mock actions
        self.router = ActionRouter(self.mock_services)
        self.router.actions = [
            self.mock_todo_action,
            self.mock_context_action  # Last action should be fallback
        ]
        
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
        # Default should select context action (fallback)
        action = self.router._get_action_for_text("Hello")
        self.assertEqual(action, self.mock_context_action)
        
        # Setup TODO action to handle TODO text
        self.mock_todo_action.can_handle = MagicMock(return_value=lambda x: "TODO" in x)
        
        # Should select TODO action
        self.mock_todo_action.can_handle.return_value = True
        action = self.router._get_action_for_text("TODO: Buy milk")
        self.assertEqual(action, self.mock_todo_action)
    
    async def test_route_action(self):
        """Test routing to the appropriate action."""
        # Setup mock response from action
        mock_response = ActionResponse(success=True, message="Hello there!")
        self.mock_context_action.execute.return_value = mock_response
        
        # Route action
        response = await self.router.route_action(self.request)
        
        # Verify result
        self.assertEqual(response, mock_response)
        
        # Context action should be called
        self.mock_context_action.execute.assert_called_once_with(self.request)
        
        # TODO action should not be called
        self.mock_todo_action.execute.assert_not_called()
    
    async def test_nickname_command_handling(self):
        """Test nickname command handling in router."""
        # Mock notion service with nickname handler
        self.mock_services.notion_service.handle_nickname_command = AsyncMock(
            return_value=("I'll call you TestUser!", True)
        )
        self.mock_services.slack_service.get_user_display_name = AsyncMock(
            return_value="Original Name"
        )
        
        # Route action
        response = await self.router.route_action(self.request)
        
        # Verify nickname response
        self.assertTrue(response.success)
        self.assertEqual(response.message, "I'll call you TestUser!")
        
        # Actions should not be called
        self.mock_context_action.execute.assert_not_called()
        self.mock_todo_action.execute.assert_not_called()
    
    async def test_get_available_actions(self):
        """Test getting available action names."""
        action_names = self.router.get_available_actions()
        self.assertEqual(action_names, ["TodoAction", "ContextResponseAction"])

if __name__ == '__main__':
    unittest.main()