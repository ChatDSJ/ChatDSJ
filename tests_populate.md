# Test Files

## Action Framework Tests

### `tests/test_action_framework.py`

```python
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

    @patch('actions.action_framework.HistoryManager')
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

    @patch('actions.action_framework.HistoryManager')
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
```

## Memory Handler Tests

### `tests/test_memory_handler.py`

```python
import unittest
from unittest.mock import MagicMock, patch
from handler.memory_handler import MemoryHandler, SectionType, PropertyType


class TestMemoryHandler(unittest.TestCase):
    def setUp(self):
        # Mock notion service
        self.mock_notion = MagicMock()
        
        # Create memory handler
        self.handler = MemoryHandler(self.mock_notion)

    def test_classify_memory_instruction_facts(self):
        """Test classifying fact memory instructions."""
        # Test explicit fact commands
        memory_type, content = self.handler.classify_memory_instruction("fact: I like coffee")
        self.assertEqual(memory_type, "known_fact")
        self.assertEqual(content, "I like coffee")
        
        memory_type, content = self.handler.classify_memory_instruction("known fact: I have a cat")
        self.assertEqual(memory_type, "known_fact")
        self.assertEqual(content, "I have a cat")
        
        # Test remember patterns
        memory_type, content = self.handler.classify_memory_instruction("remember that I like tea")
        self.assertEqual(memory_type, "known_fact")
        self.assertEqual(content, "I like tea")

    def test_classify_memory_instruction_locations(self):
        """Test classifying location memory instructions."""
        # Test work location
        memory_type, content = self.handler.classify_memory_instruction("I work from Seattle")
        self.assertEqual(memory_type, "work_location")
        self.assertEqual(content, "Seattle")
        
        memory_type, content = self.handler.classify_memory_instruction("I am working in New York")
        self.assertEqual(memory_type, "work_location")
        self.assertEqual(content, "New York")
        
        # Test home location
        memory_type, content = self.handler.classify_memory_instruction("I live in Boston")
        self.assertEqual(memory_type, "home_location")
        self.assertEqual(content, "Boston")
        
        memory_type, content = self.handler.classify_memory_instruction("I was born in Chicago")
        self.assertEqual(memory_type, "home_location")
        self.assertEqual(content, "Chicago")

    def test_classify_memory_instruction_preferences(self):
        """Test classifying preference memory instructions."""
        # Test explicit preference commands
        memory_type, content = self.handler.classify_memory_instruction("preference: Use bullet points")
        self.assertEqual(memory_type, "preference")
        self.assertEqual(content, "Use bullet points")
        
        # Test I prefer patterns
        memory_type, content = self.handler.classify_memory_instruction("I prefer short responses")
        self.assertEqual(memory_type, "preference")
        self.assertEqual(content, "short responses")

    def test_classify_memory_instruction_projects(self):
        """Test classifying project memory instructions."""
        # Test project add
        memory_type, content = self.handler.classify_memory_instruction("project: Building a chatbot")
        self.assertEqual(memory_type, "project_add")
        self.assertEqual(content, "Building a chatbot")
        
        # Test project replace
        memory_type, content = self.handler.classify_memory_instruction("my new project is AI research")
        self.assertEqual(memory_type, "project_replace")
        self.assertEqual(content, "AI research")

    def test_classify_memory_instruction_todos(self):
        """Test classifying TODO memory instructions."""
        memory_type, content = self.handler.classify_memory_instruction("TODO: Buy groceries")
        self.assertEqual(memory_type, "todo")
        self.assertEqual(content, "Buy groceries")

    def test_classify_memory_instruction_list_commands(self):
        """Test classifying list commands."""
        memory_type, content = self.handler.classify_memory_instruction("list my facts")
        self.assertEqual(memory_type, "list_facts")
        self.assertIsNone(content)
        
        memory_type, content = self.handler.classify_memory_instruction("show my preferences")
        self.assertEqual(memory_type, "list_preferences")
        self.assertIsNone(content)
        
        memory_type, content = self.handler.classify_memory_instruction("list my projects")
        self.assertEqual(memory_type, "list_projects")
        self.assertIsNone(content)

    def test_classify_memory_instruction_delete_commands(self):
        """Test classifying delete commands."""
        memory_type, content = self.handler.classify_memory_instruction("delete fact about coffee")
        self.assertEqual(memory_type, "delete_fact")
        self.assertEqual(content, "coffee")
        
        memory_type, content = self.handler.classify_memory_instruction("remove preference about bullets")
        self.assertEqual(memory_type, "delete_preference")
        self.assertEqual(content, "bullets")

    def test_classify_memory_instruction_questions(self):
        """Test that questions are not classified as memory instructions."""
        # Questions should return unknown
        memory_type, content = self.handler.classify_memory_instruction("what time is it?")
        self.assertEqual(memory_type, "unknown")
        self.assertIsNone(content)
        
        memory_type, content = self.handler.classify_memory_instruction("how are you?")
        self.assertEqual(memory_type, "unknown")
        self.assertIsNone(content)
        
        memory_type, content = self.handler.classify_memory_instruction("do you know about Python?")
        self.assertEqual(memory_type, "unknown")
        self.assertIsNone(content)

    def test_handle_memory_instruction_known_fact(self):
        """Test handling known fact memory instructions."""
        # Mock add_known_fact to succeed
        self.handler.add_known_fact = MagicMock(return_value=True)
        self.handler.verify_fact_stored = MagicMock(return_value=True)
        
        # Handle fact instruction
        response = self.handler.handle_memory_instruction(
            "U12345", "fact: I like coffee"
        )
        
        # Verify response
        self.assertIn("Added to your Known Facts", response)
        
        # Verify add_known_fact was called
        self.handler.add_known_fact.assert_called_once_with("U12345", "I like coffee")

    def test_handle_memory_instruction_preference(self):
        """Test handling preference memory instructions."""
        # Mock add_preference to succeed
        self.handler.add_preference = MagicMock(return_value=True)
        self.handler.verify_preference_stored = MagicMock(return_value=True)
        
        # Handle preference instruction
        response = self.handler.handle_memory_instruction(
            "U12345", "preference: Use bullet points"
        )
        
        # Verify response
        self.assertIn("Added to your Preferences", response)
        
        # Verify add_preference was called
        self.handler.add_preference.assert_called_once_with("U12345", "Use bullet points")

    def test_handle_memory_instruction_list_facts(self):
        """Test handling list facts instructions."""
        # Mock get_known_facts to return facts
        self.handler.get_known_facts = MagicMock(return_value=["I like coffee", "I have a cat"])
        
        # Handle list instruction
        response = self.handler.handle_memory_instruction("U12345", "list my facts")
        
        # Verify response
        self.assertIn("Here are your stored facts", response)
        self.assertIn("I like coffee", response)
        self.assertIn("I have a cat", response)

    def test_handle_memory_instruction_delete_fact(self):
        """Test handling delete fact instructions."""
        # Mock delete_known_fact to succeed
        self.handler.get_known_facts = MagicMock(side_effect=[
            ["I like coffee", "I like tea"],  # Before deletion
            ["I like tea"]  # After deletion
        ])
        self.handler.delete_known_fact = MagicMock(return_value=True)
        
        # Handle delete instruction
        response = self.handler.handle_memory_instruction("U12345", "delete fact about coffee")
        
        # Verify response
        self.assertIn("Successfully removed fact", response)
        
        # Verify delete was called
        self.handler.delete_known_fact.assert_called_once_with("U12345", "coffee")

    def test_get_known_facts(self):
        """Test retrieving known facts."""
        # Mock notion service
        self.mock_notion.get_user_page_id.return_value = "page_123"
        self.mock_notion.client.blocks.children.list.return_value = {
            "results": [
                {
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"plain_text": "Known Facts"}]}
                },
                {
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"plain_text": "I like coffee"}]}
                },
                {
                    "type": "bulleted_list_item", 
                    "bulleted_list_item": {"rich_text": [{"plain_text": "I have a cat"}]}
                }
            ]
        }
        
        # Get facts
        facts = self.handler.get_known_facts("U12345")
        
        # Verify facts
        self.assertEqual(len(facts), 2)
        self.assertIn("I like coffee", facts)
        self.assertIn("I have a cat", facts)

    def test_add_known_fact(self):
        """Test adding a known fact."""
        # Mock notion service
        self.mock_notion.get_user_page_id.return_value = "page_123"
        self.handler._ensure_known_facts_section_exists = MagicMock(return_value="section_123")
        self.mock_notion.client.blocks.children.append.return_value = {"results": [{"id": "new_fact"}]}
        self.handler._verify_block_exists = MagicMock(return_value=True)
        
        # Add fact
        result = self.handler.add_known_fact("U12345", "I like tea")
        
        # Verify result
        self.assertTrue(result)
        
        # Verify notion client was called
        self.mock_notion.client.blocks.children.append.assert_called_once()


if __name__ == '__main__':
    unittest.main()
```

## History Manager Tests

### `tests/test_history_manager.py`

```python
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from utils.simple_history_manager import SimpleHistoryManager


class TestHistoryManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = HistoryManager()
        
        # Mock Slack service
        self.mock_slack = MagicMock()
        self.mock_slack.bot_user_id = "BOT123"
        self.mock_slack.fetch_channel_history = AsyncMock()
        self.mock_slack.fetch_thread_history = AsyncMock()
        
        # Sample messages
        self.sample_messages = [
            {
                "ts": "1617983900.000100",
                "user": "U12345",
                "text": "Hello everyone",
                "type": "message"
            },
            {
                "ts": "1617983950.000100",
                "user": "U67890",
                "text": "Has anyone discussed Python lately?",
                "type": "message"
            },
            {
                "ts": "1617984000.000100",
                "user": "U12345",
                "text": "I was working on a Python project yesterday",
                "type": "message"
            }
        ]

    def test_extract_search_topic(self):
        """Test extracting search topics from different query patterns."""
        # Test pattern: "has anyone discussed X"
        topic = self.manager.extract_search_topic("has anyone discussed Python lately?")
        self.assertEqual(topic, "Python")
        
        # Test pattern: "was X discussed"
        topic = self.manager.extract_search_topic("was machine learning discussed in the meeting?")
        self.assertEqual(topic, "machine learning")
        
        # Test pattern: "any discussions about X"
        topic = self.manager.extract_search_topic("any discussions about the new API?")
        self.assertEqual(topic, "the new API")
        
        # Test direct pattern with "has X been discussed"
        topic = self.manager.extract_search_topic("has Miami been discussed?")
        self.assertEqual(topic, "Miami")
        
        # Test with capitalized words
        topic = self.manager.extract_search_topic("did we talk about Docker Containers?")
        self.assertEqual(topic, "Docker Containers")
        
        # Test with quoted phrases
        topic = self.manager.extract_search_topic('what do we know about "quantum computing"?')
        self.assertEqual(topic, "quantum computing")
        
        # Test non-search query
        topic = self.manager.extract_search_topic("how are you today?")
        self.assertIsNone(topic)

    def test_extract_full_message_text(self):
        """Test extracting all text from a message including blocks."""
        # Basic message
        basic_msg = {"text": "Hello world", "type": "message"}
        text = self.manager.extract_full_message_text(basic_msg)
        self.assertEqual(text, "Hello world")
        
        # Message with blocks
        msg_with_blocks = {
            "text": "Hello",
            "blocks": [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "elements": [
                                {"type": "text", "text": "Hello world"}
                            ]
                        }
                    ]
                }
            ]
        }
        text = self.manager.extract_full_message_text(msg_with_blocks)
        self.assertEqual(text, "Hello Hello world")
        
        # Message with attachments
        msg_with_attachments = {
            "text": "Check this out:",
            "attachments": [
                {
                    "text": "Important document",
                    "title": "Quarterly Report"
                }
            ]
        }
        text = self.manager.extract_full_message_text(msg_with_attachments)
        self.assertEqual(text, "Check this out: Important document Quarterly Report")

    def test_find_messages_containing_topic(self):
        """Test finding messages that contain a specific topic."""
        # Find messages containing "Python"
        matching_msgs = self.manager.find_messages_containing_topic(
            self.sample_messages, "Python"
        )
        
        # Should find 1 message (excluding the question about Python)
        self.assertEqual(len(matching_msgs), 1)
        self.assertEqual(matching_msgs[0]["ts"], "1617984000.000100")
        
        # Case insensitive search
        matching_msgs = self.manager.find_messages_containing_topic(
            self.sample_messages, "python"
        )
        self.assertEqual(len(matching_msgs), 1)
        
        # Search for non-existent topic
        matching_msgs = self.manager.find_messages_containing_topic(
            self.sample_messages, "JavaScript"
        )
        self.assertEqual(len(matching_msgs), 0)

    def test_format_history_for_prompt(self):
        """Test formatting message history for LLM prompt."""
        # Create user display names
        user_display_names = {
            "U12345": "Alice",
            "U67890": "Bob",
            "BOT123": "TestBot"
        }
        
        # Mock query params for regular context
        query_params = {
            "is_search": False,
            "description": "Recent context"
        }
        
        # Format regular history
        formatted = self.manager.format_history_for_prompt(
            self.sample_messages,
            query_params,
            user_display_names,
            "BOT123"
        )
        
        # Check formatting
        self.assertIn("[Alice]", formatted)
        self.assertIn("[Bob]", formatted)
        self.assertIn("Hello everyone", formatted)
        self.assertIn("Python project", formatted)
        
        # Format search results
        search_params = {
            "is_search": True,
            "search_topic": "Python",
            "description": "Deep search for: Python"
        }
        
        # Only include the message that matches the search
        python_msg = [msg for msg in self.sample_messages if "Python" in msg["text"] and "discussed" not in msg["text"]]
        
        formatted = self.manager.format_history_for_prompt(
            python_msg,
            search_params,
            user_display_names,
            "BOT123"
        )
        
        # Check search-specific formatting
        self.assertIn("YES, I found", formatted)
        self.assertIn("Python", formatted)
        self.assertIn("[Alice]", formatted)
        self.assertIn("Python project", formatted)

    async def test_retrieve_and_filter_history_regular(self):
        """Test retrieving and filtering for regular (non-search) queries."""
        # Setup mock to return sample messages
        self.mock_slack.fetch_channel_history.return_value = self.sample_messages
        
        # Non-search prompt
        prompt = "how are you today?"
        
        # Retrieve and filter
        messages, params = await self.manager.retrieve_and_filter_history(
            self.mock_slack,
            "C12345",
            None,  # No thread
            prompt
        )
        
        # Verify results
        self.assertEqual(len(messages), len(self.sample_messages))
        self.assertFalse(params["is_search"])
        self.assertIsNone(params["search_topic"])
        
        # Verify channel history was fetched
        self.mock_slack.fetch_channel_history.assert_called_once()
        self.mock_slack.fetch_thread_history.assert_not_called()

    async def test_retrieve_and_filter_history_search(self):
        """Test retrieving and filtering for search queries."""
        # Setup mocks to return messages
        self.mock_slack.fetch_channel_history.return_value = self.sample_messages
        self.mock_slack.fetch_thread_history.return_value = []
        
        # Search prompt
        prompt = "has anyone discussed Python?"
        
        # Retrieve and filter
        messages, params = await self.manager.retrieve_and_filter_history(
            self.mock_slack,
            "C12345",
            None,  # No thread
            prompt
        )
        
        # Verify search parameters
        self.assertTrue(params["is_search"])
        self.assertEqual(params["search_topic"], "Python")
        
        # Verify deep history search was performed
        self.mock_slack.fetch_channel_history.assert_called_once_with(
            "C12345",
            self.manager.SEARCH_DEPTH_LIMIT
        )

    async def test_retrieve_and_filter_history_thread_summary(self):
        """Test retrieving history for thread summaries."""
        # Setup mocks
        thread_messages = [
            {"ts": "1617984050.000100", "thread_ts": "1617984050.000100", "user": "U12345", "text": "Thread start"},
            {"ts": "1617984100.000100", "thread_ts": "1617984050.000100", "user": "U67890", "text": "Thread reply"}
        ]
        self.mock_slack.fetch_thread_history.return_value = thread_messages
        
        # Thread summary prompt
        prompt = "summarize this thread"
        thread_ts = "1617984050.000100"
        
        # Retrieve and filter
        messages, params = await self.manager.retrieve_and_filter_history(
            self.mock_slack,
            "C12345",
            thread_ts,
            prompt
        )
        
        # Verify thread-only results for thread summary
        self.assertEqual(len(messages), len(thread_messages))
        self.assertIn("Thread summary", params["description"])
        
        # Verify thread history was fetched
        self.mock_slack.fetch_thread_history.assert_called_once()

    async def test_retrieve_and_filter_history_channel_summary(self):
        """Test retrieving history for channel summaries."""
        # Setup mock
        self.mock_slack.fetch_channel_history.return_value = self.sample_messages
        
        # Channel summary prompt
        prompt = "summarize the channel"
        
        # Retrieve and filter
        messages, params = await self.manager.retrieve_and_filter_history(
            self.mock_slack,
            "C12345",
            None,
            prompt
        )
        
        # Verify results
        self.assertEqual(len(messages), len(self.sample_messages))
        self.assertIn("Channel summary", params["description"])
        
        # Verify channel history was fetched
        self.mock_slack.fetch_channel_history.assert_called_once()

    def test_format_search_results_with_threads(self):
        """Test formatting search results with thread information."""
        # Create user display names
        user_display_names = {
            "U12345": "Alice",
            "U67890": "Bob"
        }
        
        # Mock query params
        query_params = {
            "search_topic": "Python",
            "is_search": True
        }
        
        # Mock slack service with permalink generation
        mock_slack_service = MagicMock()
        mock_slack_service.generate_message_permalink.return_value = "https://test.slack.com/link"
        
        # Test with messages
        results = self.manager.format_search_results_with_threads(
            self.sample_messages,
            query_params,
            user_display_names,
            "BOT123",
            "C12345",
            mock_slack_service
        )
        
        # Verify results structure
        self.assertIn("summary", results)
        self.assertIn("threads", results)
        self.assertTrue(results["show_details"])
        self.assertEqual(results["topic"], "Python")
        
        # Test with no messages
        empty_results = self.manager.format_search_results_with_threads(
            [],
            query_params,
            user_display_names,
            "BOT123",
            "C12345",
            mock_slack_service
        )
        
        self.assertFalse(empty_results["show_details"])
        self.assertEqual(len(empty_results["threads"]), 0)


if __name__ == '__main__':
    unittest.main()
```

## Main App Tests

### `tests/test_main_app.py`

```python
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
```

## OpenAI Service Tests

### `tests/test_openai_service.py`

```python
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
```

## Context Builder Tests

### `tests/test_context_builder.py`

```python
import unittest
from unittest.mock import MagicMock, patch
import asyncio
from utils.context_builder import (
    get_enhanced_user_context,
    extract_user_preferences,
    extract_structured_fields
)


class TestContextBuilder(unittest.TestCase):
    def setUp(self):
        # Mock notion service
        self.mock_notion = MagicMock()
        
        # Sample content and properties
        self.sample_content = """Projects

Build a chatbot
Improve documentation

Preferences

Use bullet points
Keep responses concise
Write in a friendly tone

Known Facts

Working as a developer
Enjoys hiking on weekends
Based in Seattle"""

        self.sample_properties = {
            "UserID": {
                "type": "title",
                "title": [{"plain_text": "U12345"}]
            },
            "PreferredName": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "Test User"}]
            },
            "WorkLocation": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "Remote"}]
            },
            "Role": {
                "type": "select",
                "select": {"name": "Developer"}
            }
        }
        
        # Configure mock service
        self.mock_notion.get_user_page_content.return_value = self.sample_content
        self.mock_notion.get_user_page_properties.return_value = self.sample_properties
        self.mock_notion.get_user_preferred_name.return_value = "Test User"

    def test_extract_user_preferences(self):
        """Test extracting user preferences from page content."""
        preferences = extract_user_preferences(self.sample_content)
        
        # Should extract 3 preferences
        self.assertEqual(len(preferences), 3)
        self.assertIn("Use bullet points", preferences)
        self.assertIn("Keep responses concise", preferences)
        self.assertIn("Write in a friendly tone", preferences)
        
        # Test with no preferences section
        content_no_prefs = """Projects

Build a chatbot

Known Facts

Working as a developer"""
        preferences = extract_user_preferences(content_no_prefs)
        self.assertEqual(len(preferences), 0)
        
        # Test with empty content
        self.assertEqual(extract_user_preferences(""), [])
        self.assertEqual(extract_user_preferences(None), [])

    def test_extract_structured_fields(self):
        """Test extracting structured fields from properties."""
        fields = extract_structured_fields(self.sample_properties)
        
        # Should extract fields while skipping PreferredName
        self.assertEqual(len(fields), 3)  # UserID, WorkLocation, Role
        self.assertIn("UserID: U12345", fields)
        self.assertIn("WorkLocation: Remote", fields)
        self.assertIn("Role: Developer", fields)
        self.assertNotIn("PreferredName:", " ".join(fields))  # Should be skipped
        
        # Test with empty properties
        self.assertEqual(extract_structured_fields({}), [])
        self.assertEqual(extract_structured_fields(None), [])


class TestGetEnhancedUserContext(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock notion service
        self.mock_notion = MagicMock()
        
        # Sample content and properties
        self.sample_content = """Projects

Build a chatbot
Improve documentation

Preferences

Use bullet points
Keep responses concise

Known Facts

Working as a developer
Based in Seattle"""

        self.sample_properties = {
            "UserID": {
                "type": "title",
                "title": [{"plain_text": "U12345"}]
            },
            "WorkLocation": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "Remote"}]
            },
            "Role": {
                "type": "select",
                "select": {"name": "Developer"}
            }
        }
        
        # Configure mock service
        self.mock_notion.get_user_page_content.return_value = self.sample_content
        self.mock_notion.get_user_page_properties.return_value = self.sample_properties
        self.mock_notion.get_user_preferred_name.return_value = "Test User"

    async def test_get_enhanced_user_context_full(self):
        """Test building enhanced user context with all components."""
        # Get enhanced context
        context = await asyncio.to_thread(
            get_enhanced_user_context,
            self.mock_notion,
            "U12345",
            "Base prompt"
        )
        
        # Verify components are included
        self.assertIn("Base prompt", context)
        self.assertIn("USER PREFERENCES", context)
        self.assertIn("bullet points", context)
        self.assertIn("USER DATABASE PROPERTIES", context)
        self.assertIn("Remote", context)
        self.assertIn("Test User", context)
        self.assertIn("Projects", context)
        self.assertIn("Known Facts", context)
        
        # Verify preferences are at the top
        pref_index = context.find("USER PREFERENCES")
        base_index = context.find("Base prompt")
        self.assertLess(pref_index, base_index, "Preferences should come before base prompt")

    async def test_get_enhanced_user_context_minimal(self):
        """Test building context with missing components."""
        # Configure mock to return minimal data
        self.mock_notion.get_user_page_content.return_value = None
        self.mock_notion.get_user_preferred_name.return_value = None
        self.mock_notion.get_user_page_properties.return_value = {}
        
        # Get context
        context = await asyncio.to_thread(
            get_enhanced_user_context,
            self.mock_notion,
            "U12345",
            "Base prompt"
        )
        
        # Should still include base prompt
        self.assertIn("Base prompt", context)
        # Should not include sections for missing data
        self.assertNotIn("USER PREFERENCES", context)
        self.assertNotIn("USER DATABASE PROPERTIES", context)

    async def test_get_enhanced_user_context_preferences_priority(self):
        """Test that preferences are given priority in the context."""
        # Content with strong preferences
        content_with_strong_prefs = """Preferences

Always write your answers in rhymed verse
Use only emojis for punctuation
Keep responses under 50 words

Known Facts

Likes poetry
Works remotely"""

        self.mock_notion.get_user_page_content.return_value = content_with_strong_prefs
        
        # Get context
        context = await asyncio.to_thread(
            get_enhanced_user_context,
            self.mock_notion,
            "U12345",
            "You are a helpful assistant."
        )
        
        # Verify preferences are emphasized
        self.assertIn("IMPORTANT USER PREFERENCES", context)
        self.assertIn("rhymed verse", context)
        self.assertIn("emojis for punctuation", context)
        self.assertIn("REMINDER: Always respect", context)
        
        # Check that preferences appear before other content
        pref_index = context.find("IMPORTANT USER PREFERENCES")
        facts_index = context.find("Known Facts")
        self.assertLess(pref_index, facts_index)


if __name__ == '__main__':
    unittest.main()
```

## Cached Notion Service Tests

### `tests/test_cached_notion_service.py`

```python
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from services.cached_notion_service import CachedNotionService


class TestCachedNotionService(unittest.TestCase):
    def setUp(self):
        # Create service with mocked Notion client
        self.mock_client = MagicMock()
        with patch('services.cached_notion_service.Client', return_value=self.mock_client):
            self.service = CachedNotionService(
                notion_api_token="secret_dummy_token",
                notion_user_db_id="dummy_db_id",
                cache_ttl=60,
                cache_max_size=100
            )

    def test_initialization(self):
        """Test service initialization."""
        self.assertEqual(self.service.api_token, "secret_dummy_token")
        self.assertEqual(self.service.user_db_id, "dummy_db_id")
        self.assertEqual(self.service.cache_ttl, 60)
        self.assertIsNotNone(self.service.memory_handler)
        
        # Test caches are initialized
        self.assertEqual(len(self.service.cache), 0)
        self.assertEqual(len(self.service.page_content_cache), 0)

    def test_is_available(self):
        """Test availability check."""
        # Should be available with client and DB ID
        self.assertTrue(self.service.is_available())
        
        # Test without client
        self.service.client = None
        self.assertFalse(self.service.is_available())
        
        # Reset client
        self.service.client = self.mock_client
        
        # Test without DB ID
        self.service.user_db_id = None
        self.assertFalse(self.service.is_available())

    def test_clear_cache(self):
        """Test clearing the caches."""
        # Add items to caches
        with self.service.cache_lock:
            self.service.cache["test_key"] = "test_value"
        
        with self.service.page_content_lock:
            self.service.page_content_cache["test_page"] = "test_content"
        
        # Clear caches
        self.service.clear_cache()
        
        # Verify caches are empty
        self.assertEqual(len(self.service.cache), 0)
        self.assertEqual(len(self.service.page_content_cache), 0)

    def test_invalidate_user_cache(self):
        """Test invalidating cache for a specific user."""
        user_id = "U12345"
        
        # Add user-specific entries to caches
        with self.service.cache_lock:
            self.service.cache[f"user_page_id_{user_id}"] = "page_id_123"
            self.service.cache[f"user_properties_{user_id}"] = {"prop": "value"}
            self.service.cache["unrelated_key"] = "unrelated_value"
        
        with self.service.page_content_lock:
            self.service.page_content_cache[f"user_page_content_{user_id}"] = "page content"
        
        # Invalidate user cache
        self.service.invalidate_user_cache(user_id)
        
        # Verify user-specific entries are removed
        self.assertNotIn(f"user_page_id_{user_id}", self.service.cache)
        self.assertNotIn(f"user_properties_{user_id}", self.service.cache)
        self.assertNotIn(f"user_page_content_{user_id}", self.service.page_content_cache)
        
        # Verify unrelated entries are kept
        self.assertIn("unrelated_key", self.service.cache)

    def test_get_user_page_id_with_cache_hit(self):
        """Test retrieving user page ID with cache hit."""
        user_id = "U12345"
        expected_page_id = "page_123"
        
        # Add to cache
        with self.service.cache_lock:
            self.service.cache[f"user_page_id_{user_id}"] = expected_page_id
        
        # Get page ID
        page_id = self.service.get_user_page_id(user_id)
        
        # Verify result and cache hit
        self.assertEqual(page_id, expected_page_id)
        self.assertEqual(self.service.cache_stats["hits"], 1)
        self.assertEqual(self.service.cache_stats["misses"], 0)
        
        # Client should not be called
        self.mock_client.databases.query.assert_not_called()

    def test_get_user_page_id_with_cache_miss(self):
        """Test retrieving user page ID with cache miss."""
        user_id = "U12345"
        expected_page_id = "page_123"
        
        # Setup mock response
        mock_response = {
            "results": [{"id": expected_page_id}]
        }
        self.mock_client.databases.query.return_value = mock_response
        
        # Get page ID
        page_id = self.service.get_user_page_id(user_id)
        
        # Verify result and cache miss
        self.assertEqual(page_id, expected_page_id)
        self.assertEqual(self.service.cache_stats["hits"], 0)
        self.assertEqual(self.service.cache_stats["misses"], 1)
        
        # Verify client was called with correct parameters
        self.mock_client.databases.query.assert_called_once()
        call_args = self.mock_client.databases.query.call_args[1]
        self.assertEqual(call_args["database_id"], "dummy_db_id")
        self.assertEqual(call_args["filter"]["property"], "UserID")
        self.assertEqual(call_args["filter"]["title"]["equals"], user_id)
        
        # Check that result was cached
        with self.service.cache_lock:
            self.assertIn(f"user_page_id_{user_id}", self.service.cache)
            self.assertEqual(self.service.cache[f"user_page_id_{user_id}"], expected_page_id)

    def test_get_user_page_id_not_found(self):
        """Test retrieving user page ID when not found."""
        user_id = "U12345"
        
        # Setup mock response with no results
        mock_response = {"results": []}
        self.mock_client.databases.query.return_value = mock_response
        
        # Get page ID
        page_id = self.service.get_user_page_id(user_id)
        
        # Verify result
        self.assertIsNone(page_id)
        
        # Check that None was cached
        with self.service.cache_lock:
            self.assertIn(f"user_page_id_{user_id}", self.service.cache)
            self.assertIsNone(self.service.cache[f"user_page_id_{user_id}"])

    def test_store_user_nickname_new_user(self):
        """Test storing a nickname for a new user."""
        user_id = "U12345"
        nickname = "TestUser"
        
        # Mock get_user_page_id to return None (new user)
        with patch.object(self.service, 'get_user_page_id', return_value=None):
            # Mock database info for page creation
            self.mock_client.databases.retrieve.return_value = {
                "properties": {
                    "UserID": {"type": "title"},
                    "PreferredName": {"type": "rich_text"},
                    "SlackDisplayName": {"type": "rich_text"}
                }
            }
            
            # Mock page creation
            self.mock_client.pages.create.return_value = {"id": "new_page_123"}
            
            # Store nickname
            result = self.service.store_user_nickname(user_id, nickname)
            
            # Verify result
            self.assertTrue(result)
            
            # Verify client was called for page creation
            self.mock_client.pages.create.assert_called_once()
            call_args = self.mock_client.pages.create.call_args[1]
            self.assertEqual(call_args["parent"]["database_id"], "dummy_db_id")
            self.assertEqual(call_args["properties"]["UserID"]["title"][0]["text"]["content"], user_id)
            self.assertEqual(call_args["properties"]["PreferredName"]["rich_text"][0]["text"]["content"], nickname)

    def test_store_user_nickname_existing_user(self):
        """Test storing a nickname for an existing user."""
        user_id = "U12345"
        nickname = "NewNickname"
        
        # Mock get_user_page_id to return existing page
        with patch.object(self.service, 'get_user_page_id', return_value="existing_page_123"):
            # Store nickname
            result = self.service.store_user_nickname(user_id, nickname)
            
            # Verify result
            self.assertTrue(result)
            
            # Verify client was called for page update
            self.mock_client.pages.update.assert_called_once()
            call_args = self.mock_client.pages.update.call_args[1]
            self.assertEqual(call_args["page_id"], "existing_page_123")
            self.assertEqual(call_args["properties"]["PreferredName"]["rich_text"][0]["text"]["content"], nickname)

    def test_handle_nickname_command(self):
        """Test handling nickname commands from chat."""
        user_id = "U12345"
        
        # Mock store_user_nickname to succeed
        with patch.object(self.service, 'store_user_nickname', return_value=True):
            # Test explicit nickname command
            response, success = self.service.handle_nickname_command(
                "call me TestUser", user_id, "Display Name"
            )
            
            # Verify response
            self.assertTrue(success)
            self.assertIn("TestUser", response)
            
            # Test "my name is" format
            response, success = self.service.handle_nickname_command(
                "my name is John Doe", user_id, "Display Name"
            )
            
            # Verify response
            self.assertTrue(success)
            self.assertIn("John Doe", response)
            
            # Test quoted name
            response, success = self.service.handle_nickname_command(
                'call me "Captain Awesome"', user_id, "Display Name"
            )
            
            # Verify response
            self.assertTrue(success)
            self.assertIn("Captain Awesome", response)
        
        # Test with store failure
        with patch.object(self.service, 'store_user_nickname', return_value=False):
            response, success = self.service.handle_nickname_command(
                "call me TestUser", user_id, "Display Name"
            )
            
            # Verify response
            self.assertFalse(success)
            self.assertIn("trouble", response.lower())
        
        # Test non-nickname command
        response, success = self.service.handle_nickname_command(
            "what time is it?", user_id, "Display Name"
        )
        
        # Verify response
        self.assertFalse(success)
        self.assertIsNone(response)

    def test_get_user_preferred_name(self):
        """Test retrieving a user's preferred name."""
        user_id = "U12345"
        
        # Mock get_user_page_properties to return properties with preferred name
        properties = {
            "PreferredName": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "Test User"}]
            }
        }
        
        with patch.object(self.service, 'get_user_page_properties', return_value=properties):
            # Get preferred name
            name = self.service.get_user_preferred_name(user_id)
            
            # Verify result
            self.assertEqual(name, "Test User")
        
        # Test with empty rich_text array
        properties = {
            "PreferredName": {
                "type": "rich_text",
                "rich_text": []
            }
        }
        
        with patch.object(self.service, 'get_user_page_properties', return_value=properties):
            name = self.service.get_user_preferred_name(user_id)
            self.assertIsNone(name)
        
        # Test with no properties
        with patch.object(self.service, 'get_user_page_properties', return_value=None):
            name = self.service.get_user_preferred_name(user_id)
            self.assertIsNone(name)

    def test_handle_memory_instruction(self):
        """Test delegating memory instructions to memory handler."""
        user_id = "U12345"
        instruction = "fact: I like coffee"
        
        # Mock memory handler response
        self.service.memory_handler.handle_memory_instruction = MagicMock(
            return_value="✅ Added to your Known Facts: I like coffee"
        )
        
        # Handle memory instruction
        response = self.service.handle_memory_instruction(user_id, instruction)
        
        # Verify delegation
        self.service.memory_handler.handle_memory_instruction.assert_called_once_with(user_id, instruction)
        self.assertEqual(response, "✅ Added to your Known Facts: I like coffee")

    def test_add_todo_item(self):
        """Test adding a TODO item to a user's page."""
        user_id = "U12345"
        todo_text = "Finish tests"
        
        # Mock get_user_page_id for existing user
        with patch.object(self.service, 'get_user_page_id', return_value="page_123"):
            # Mock find_section_block to return None (no Instructions section)
            with patch.object(self.service, 'find_section_block', return_value=None):
                # Mock block append
                self.mock_client.blocks.children.append.return_value = {"results": [{"id": "block_123"}]}
                
                # Add TODO item
                result = self.service.add_todo_item(user_id, todo_text)
                
                # Verify result
                self.assertTrue(result)
                
                # Verify client was called correctly
                self.mock_client.blocks.children.append.assert_called_once()
                call_args = self.mock_client.blocks.children.append.call_args[1]
                self.assertEqual(call_args["block_id"], "page_123")
                
                # Verify the to_do block was created correctly
                todo_block = call_args["children"][0]
                self.assertEqual(todo_block["type"], "to_do")
                self.assertEqual(todo_block["to_do"]["rich_text"][0]["text"]["content"], todo_text)
                self.assertFalse(todo_block["to_do"]["checked"])


if __name__ == '__main__':
    unittest.main()
```

## Notion Parser Tests

### `tests/test_notion_parser.py`

```python
import unittest
from unittest.mock import MagicMock, patch
from services.notion_parser import NotionContextManager, get_user_context_for_llm
from utils.context_builder import extract_structured_fields


class TestNotionParser(unittest.TestCase):
    def setUp(self):
        self.context_manager = NotionContextManager()

    def test_extract_structured_fields(self):
        """Test extraction of structured fields from Notion properties."""
        properties = {
            "WorkLocation": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "New York"}]
            },
            "EmptyField": {
                "type": "rich_text",
                "rich_text": []
            },
            "PreferredName": {  # Should be skipped
                "type": "rich_text",
                "rich_text": [{"plain_text": "John"}]
            },
            "Role": {
                "type": "select",
                "select": {"name": "Developer"}
            },
            "Languages": {
                "type": "multi_select",
                "multi_select": [
                    {"name": "Python"},
                    {"name": "JavaScript"}
                ]
            },
            "UserID": {
                "type": "title",
                "title": [{"plain_text": "U12345"}]
            }
        }
        
        result = extract_structured_fields(properties)
        
        # Verify correct fields were extracted
        self.assertIn("WorkLocation: New York", result)
        self.assertIn("Role: Developer", result)
        self.assertIn("Languages: Python, JavaScript", result)
        self.assertIn("UserID: U12345", result)
        
        # Verify skipped fields
        self.assertNotIn("PreferredName: John", result)
        
        # Verify empty fields are skipped
        self.assertEqual(len([r for r in result if r.startswith("EmptyField")]), 0)

    def test_process_notion_content(self):
        """Test parsing Notion content with sections."""
        content = """Projects

Project A
Project B

Preferences

Always write your answers in rhymed verse
Keep responses concise

Known Facts

Background in tech
Lives in San Juan

Instructions

When I say "remember X", store it under Known Facts
When I ask "what do you know about me?", return all facts"""

        result = self.context_manager.process_notion_content(content)
        
        # Check if sections were parsed correctly
        self.assertIn("raw_sections", result)
        self.assertEqual(len(result["raw_sections"]), 4)
        self.assertIn("Projects", result["raw_sections"])
        self.assertIn("Preferences", result["raw_sections"])
        self.assertIn("Known Facts", result["raw_sections"])
        self.assertIn("Instructions", result["raw_sections"])
        
        # Check section content
        self.assertEqual(len(result["raw_sections"]["Projects"]), 2)
        self.assertIn("* Project A", result["raw_sections"]["Projects"])
        self.assertIn("* Project B", result["raw_sections"]["Projects"])
        
        # Check preference detection
        preference_contents = '\n'.join(result["raw_sections"]["Preferences"])
        self.assertIn("rhymed verse", preference_contents)
        self.assertIn("concise", preference_contents)

    def test_build_openai_system_prompt(self):
        """Test building the OpenAI system prompt with user context."""
        base_prompt = "You are a helpful assistant."
        notion_content = """Preferences

Use bullet points
Be concise

Known Facts

Works as a developer
Enjoys hiking"""
        preferred_name = "TestUser"
        
        # Test with all components
        result = self.context_manager.build_openai_system_prompt(
            base_prompt=base_prompt,
            notion_content=notion_content,
            preferred_name=preferred_name
        )
        
        # Verify all components are included
        self.assertIn(base_prompt, result)
        self.assertIn("TestUser", result)
        self.assertIn("Preferences", result)
        self.assertIn("bullet points", result)
        self.assertIn("Known Facts", result)
        self.assertIn("developer", result)
        
        # Test with minimal components
        minimal_result = self.context_manager.build_openai_system_prompt(
            base_prompt=base_prompt,
            notion_content="",
            preferred_name=None
        )
        
        self.assertIn(base_prompt, minimal_result)
        self.assertNotIn("Preferences", minimal_result)


class TestGetUserContextForLLM(unittest.TestCase):
    def setUp(self):
        # Mock notion service
        self.mock_notion = MagicMock()
        
        # Sample data
        self.sample_content = """Preferences

Use bullet points
Be concise

Known Facts

Developer
Likes coffee"""

        self.sample_properties = {
            "Role": {
                "type": "select",
                "select": {"name": "Developer"}
            }
        }
        
        # Configure mock
        self.mock_notion.get_user_page_content.return_value = self.sample_content
        self.mock_notion.get_user_page_properties.return_value = self.sample_properties
        self.mock_notion.get_user_preferred_name.return_value = "TestUser"

    def test_get_user_context_for_llm(self):
        """Test the main function for getting user context."""
        context = get_user_context_for_llm(
            self.mock_notion,
            "U12345",
            "Base prompt"
        )
        
        # Verify components are included
        self.assertIn("Base prompt", context)
        self.assertIn("TestUser", context)
        self.assertIn("bullet points", context)
        self.assertIn("Developer", context)
        
        # Verify service calls
        self.mock_notion.get_user_page_content.assert_called_once_with("U12345")
        self.mock_notion.get_user_page_properties.assert_called_once_with("U12345")
        self.mock_notion.get_user_preferred_name.assert_called_once_with("U12345")


if __name__ == "__main__":
    unittest.main()
```