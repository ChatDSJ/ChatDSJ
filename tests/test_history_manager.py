import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from utils.history_manager import HistoryManager


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
