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
        
        # Sample thread
        self.sample_thread = [
            {
                "ts": "1617984050.000100",
                "thread_ts": "1617984050.000100",
                "user": "U12345",
                "text": "Let's discuss AI frameworks",
                "type": "message",
                "reply_count": 2
            },
            {
                "ts": "1617984100.000100",
                "thread_ts": "1617984050.000100",
                "user": "U67890",
                "text": "I like TensorFlow for deep learning",
                "type": "message"
            },
            {
                "ts": "1617984150.000100",
                "thread_ts": "1617984050.000100",
                "user": "BOT123",
                "text": "TensorFlow is an open-source library developed by Google",
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
        
        # Test direct pattern
        topic = self.manager.extract_search_topic("tell me about JavaScript")
        self.assertEqual(topic, "JavaScript")
        
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
                            "type": "rich_text_section",
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
        
        # Message with both blocks and attachments
        complex_msg = {
            "text": "Hello",
            "blocks": [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {"type": "text", "text": "World"}
                            ]
                        }
                    ]
                }
            ],
            "attachments": [
                {
                    "text": "Nice day",
                    "title": "Weather"
                }
            ]
        }
        text = self.manager.extract_full_message_text(complex_msg)
        self.assertEqual(text, "Hello World Nice day Weather")
    
    def test_find_messages_containing_topic(self):
        """Test finding messages that contain a specific topic."""
        # Find messages containing "Python"
        matching_msgs = self.manager.find_messages_containing_topic(
            self.sample_messages, "Python"
        )
        
        # Should find 1 message
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
        
        # Search with blocks
        msg_with_block = {
            "ts": "1617984200.000100",
            "user": "U12345",
            "text": "Check out this:",
            "blocks": [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "elements": [
                                {"type": "text", "text": "React is awesome for frontend"}
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Should find the message by content in blocks
        matching_msgs = self.manager.find_messages_containing_topic(
            [msg_with_block], "React"
        )
        self.assertEqual(len(matching_msgs), 1)
    
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
        python_msg = [msg for msg in self.sample_messages if "Python" in msg["text"]]
        
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
        
        # Test empty results
        empty_formatted = self.manager.format_history_for_prompt(
            [],
            search_params,
            user_display_names,
            "BOT123"
        )
        
        self.assertEqual(empty_formatted, "No relevant messages found in channel history.")
    
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
    
    async def test_retrieve_and_filter_history_in_thread(self):
        """Test retrieving in thread context."""
        # Setup mocks to return thread and channel history
        self.mock_slack.fetch_thread_history.return_value = self.sample_thread
        self.mock_slack.fetch_channel_history.return_value = self.sample_messages
        
        # Non-search prompt in thread
        prompt = "what do you think about TensorFlow?"
        thread_ts = "1617984050.000100"
        
        # Retrieve and filter
        messages, params = await self.manager.retrieve_and_filter_history(
            self.mock_slack,
            "C12345",
            thread_ts,
            prompt
        )
        
        # Should include both thread and channel messages
        self.assertGreaterEqual(len(messages), len(self.sample_thread) + len(self.sample_messages))
        
        # Verify both thread and channel history were fetched
        self.mock_slack.fetch_thread_history.assert_called_once_with(
            "C12345",
            thread_ts,
            self.manager.THREAD_SEARCH_LIMIT
        )
        self.mock_slack.fetch_channel_history.assert_called_once()
    
    async def test_retrieve_and_filter_history_search(self):
        """Test retrieving and filtering for search queries."""
        # Setup mocks to return messages
        self.mock_slack.fetch_channel_history.return_value = self.sample_messages
        
        # Thread parent message that contains 'Python'
        thread_parent = {
            "ts": "1617984050.000100",
            "user": "U12345",
            "text": "Let's discuss Python frameworks",
            "reply_count": 2,
            "type": "message"
        }
        
        # Thread messages that mention Python
        python_thread = [
            thread_parent,
            {
                "ts": "1617984100.000100",
                "thread_ts": "1617984050.000100",
                "user": "U67890",
                "text": "I like Django for Python web development",
                "type": "message"
            }
        ]
        
        # Setup to return the thread when requested
        self.mock_slack.fetch_thread_history.return_value = python_thread
        
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

if __name__ == '__main__':
    unittest.main()