import unittest
from unittest.mock import MagicMock, patch
# Removed AsyncMock as it's not needed for the methods we are mocking in SlackService
# If other parts of your tests (not shown) mock truly async methods, you might need it.
import asyncio # Keep for unittest.IsolatedAsyncioTestCase

from utils.simple_history_manager import HistoryManager # Assuming this is the correct path

class TestHistoryManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = HistoryManager()

        # Mock Slack service
        self.mock_slack = MagicMock() # Top-level mock can be MagicMock
        self.mock_slack.bot_user_id = "BOT123"

        # THESE ARE SYNCHRONOUS METHODS in SlackService, so mock them with MagicMock
        self.mock_slack.fetch_channel_history = MagicMock()
        self.mock_slack.fetch_thread_history = MagicMock()
        # If generate_message_permalink is used and is synchronous in SlackService, mock it too
        self.mock_slack.generate_message_permalink = MagicMock(return_value="https://test.slack.com/link")


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
                "user": "U67890", # User asking the question
                "text": "Has anyone discussed Python lately?",
                "type": "message"
            },
            {
                "ts": "1617984000.000100",
                "user": "U12345", # User discussing the topic
                "text": "I was working on a Python project yesterday",
                "type": "message"
            },
            { # Another message for broader testing
                "ts": "1617984010.000100",
                "user": "UABCDE",
                "text": "Let's talk about our new API.",
                "type": "message"
            }
        ]
        self.user_display_names = {
            "U12345": "Alice",
            "U67890": "Bob",
            "UABCDE": "Charlie",
            "BOT123": "TestBot"
        }

    def test_extract_search_topic(self):
        """Test extracting search topics from different query patterns."""
        # Test pattern: "has anyone discussed X"
        topic = self.manager.extract_search_topic("has anyone discussed Python lately?")
        self.assertEqual(topic, "python") # EXPECTING LOWERCASE

        # Test pattern: "was X discussed"
        topic = self.manager.extract_search_topic("was machine learning discussed in the meeting?")
        self.assertEqual(topic, "machine learning") # EXPECTING LOWERCASE

        # Test pattern: "any discussions about X"
        # WITH ARTICLE REMOVAL in extract_search_topic
        topic = self.manager.extract_search_topic("any discussions about the new API?")
        self.assertEqual(topic, "new api") # EXPECTING LOWERCASE & NO "the"

        # Test direct pattern with "has X been discussed"
        topic = self.manager.extract_search_topic("has Miami been discussed?")
        self.assertEqual(topic, "miami") # EXPECTING LOWERCASE

        # Test with capitalized words
        topic = self.manager.extract_search_topic("did we talk about Docker Containers?")
        self.assertEqual(topic, "docker containers") # EXPECTING LOWERCASE

        # Test with quoted phrases
        topic = self.manager.extract_search_topic('what do we know about "quantum computing"?')
        self.assertEqual(topic, "quantum computing") # EXPECTING LOWERCASE

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
            "text": "Hello", # Fallback text
            "blocks": [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section", # Corrected type based on common Slack structure
                            "elements": [
                                {"type": "text", "text": "world from blocks"}
                            ]
                        }
                    ]
                }
            ]
        }
        # HistoryManager.extract_full_message_text concatenates text + " " + blocks_text
        expected_text_blocks = "Hello world from blocks" # Updated based on typical block extraction
        text = self.manager.extract_full_message_text(msg_with_blocks)
        self.assertEqual(text.strip(), "Hello world from blocks") # .strip() to handle potential leading/trailing spaces from concat

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
        # HistoryManager.extract_full_message_text concatenates text + " " + blocks_text + " " + attachments_text
        expected_text_attachments = "Check this out: Important document Quarterly Report"
        text = self.manager.extract_full_message_text(msg_with_attachments)
        self.assertEqual(text.strip(), expected_text_attachments.strip())


    def test_find_messages_containing_topic(self):
        """Test finding messages that contain a specific topic."""
        # Find messages containing "Python"
        # Assuming HistoryManager.find_messages_containing_topic is updated to generally exclude questions
        # and the `current_user_id` parameter is used to refine self-exclusion, not general question exclusion.
        matching_msgs = self.manager.find_messages_containing_topic(
            self.sample_messages, "Python" # Not passing current_user_id, general question filtering should apply
        )

        # Should find 1 message (excluding the question about Python)
        self.assertEqual(len(matching_msgs), 1, f"Expected 1 message, got {len(matching_msgs)}")
        if matching_msgs: # Avoid IndexError if test fails above
            self.assertEqual(matching_msgs[0]["ts"], "1617984000.000100") # The actual discussion

        # Case insensitive search (assuming HistoryManager handles this internally)
        matching_msgs_lower = self.manager.find_messages_containing_topic(
            self.sample_messages, "python"
        )
        self.assertEqual(len(matching_msgs_lower), 1, f"Expected 1 message for lowercase, got {len(matching_msgs_lower)}")

        # Search for non-existent topic
        matching_msgs_none = self.manager.find_messages_containing_topic(
            self.sample_messages, "JavaScript"
        )
        self.assertEqual(len(matching_msgs_none), 0)

        # Test with current_user_id to ensure *their own question* is excluded if they initiated the search
        # In this scenario, current_user_id is U67890 (Bob), who asked the question.
        # The result should still be the 1 message from Alice, as Bob's question is filtered.
        matching_msgs_user_asking = self.manager.find_messages_containing_topic(
            self.sample_messages, "Python", current_user_id="U67890"
        )
        self.assertEqual(len(matching_msgs_user_asking), 1)
        if matching_msgs_user_asking:
             self.assertEqual(matching_msgs_user_asking[0]["ts"], "1617984000.000100")

        # Test with current_user_id who is *not* asking the question.
        # Alice (U12345) is searching. Bob's question (U67890) should still be filtered by general question logic.
        matching_msgs_user_not_asking = self.manager.find_messages_containing_topic(
            self.sample_messages, "Python", current_user_id="U12345"
        )
        self.assertEqual(len(matching_msgs_user_not_asking), 1)
        if matching_msgs_user_not_asking:
            self.assertEqual(matching_msgs_user_not_asking[0]["ts"], "1617984000.000100")


    def test_format_history_for_prompt(self):
        """Test formatting message history for LLM prompt."""
        # Mock query params for regular context
        query_params_regular = {
            "is_search": False,
            "description": "Recent context"
        }

        # Format regular history
        formatted_regular = self.manager.format_history_for_prompt(
            self.sample_messages, # Use all sample messages for general context
            query_params_regular,
            self.user_display_names,
            "BOT123"
        )

        # Check formatting for regular context
        self.assertIn("[Alice]", formatted_regular) # U12345
        self.assertIn("[Bob]", formatted_regular)   # U67890
        self.assertIn("[Charlie]", formatted_regular) # UABCDE
        self.assertIn("Hello everyone", formatted_regular)
        self.assertIn("Python project", formatted_regular)
        self.assertIn("new API", formatted_regular)

        # Format search results
        query_params_search = {
            "is_search": True,
            "search_topic": "Python",
            "description": "Deep search for: Python"
        }

        # Only include the message that actually discusses Python (Alice's message)
        # This assumes find_messages_containing_topic (or upstream filtering) has already done its job.
        python_discussion_msg = [msg for msg in self.sample_messages if msg["ts"] == "1617984000.000100"]

        formatted_search = self.manager.format_history_for_prompt(
            python_discussion_msg,
            query_params_search,
            self.user_display_names,
            "BOT123"
        )

        # Check search-specific formatting
        self.assertIn("YES, I found 1 message(s)", formatted_search) # Adjust count based on input
        self.assertIn("that mention 'Python'", formatted_search)
        self.assertIn("[Alice]", formatted_search)
        self.assertIn("Python project", formatted_search)
        self.assertNotIn("[Bob]", formatted_search) # Bob's question should not be in the *formatted search results*
                                                    # if python_discussion_msg only contains Alice's message.

    async def test_retrieve_and_filter_history_regular(self):
        """Test retrieving and filtering for regular (non-search) queries."""
        # Setup mock to return sample messages for channel history
        self.mock_slack.fetch_channel_history.return_value = self.sample_messages
        self.mock_slack.fetch_thread_history.return_value = [] # No thread context for this test

        prompt = "how are you today?"
        channel_id = "C12345"

        messages, params = await self.manager.retrieve_and_filter_history(
            self.mock_slack,
            channel_id,
            None,  # No thread_ts
            prompt
        )

        # Verify results
        self.assertEqual(len(messages), len(self.sample_messages)) # Expects all messages for regular, non-thread context
        self.assertFalse(params["is_search"])
        self.assertIsNone(params["search_topic"])

        # Verify channel history was fetched with the correct limit (default 100 for non-search)
        self.mock_slack.fetch_channel_history.assert_called_once_with(
            channel_id,
            100 # Default limit for non-search in HistoryManager
        )
        self.mock_slack.fetch_thread_history.assert_not_called()

    async def test_retrieve_and_filter_history_search(self):
        """Test retrieving and filtering for search queries."""
        # Setup mocks
        # For deep search, fetch_channel_history will be called with SEARCH_DEPTH_LIMIT
        self.mock_slack.fetch_channel_history.return_value = self.sample_messages
        # Assume some threads are found and their history is fetched.
        # Let's say one thread from sample_messages (Bob's question) is identified as a parent.
        thread_parent_ts = "1617983950.000100" # Bob's message ts
        bobs_thread_replies = [
            {"ts": "1617983955.000100", "thread_ts": thread_parent_ts, "user": "U12345", "text": "I think Alice knows about Python."}
        ]
        # Configure fetch_thread_history to be called and return these replies for Bob's thread.
        # Since it might be called for multiple threads, we can use side_effect if needed,
        # or ensure our sample_messages leads to only one fetch_thread_history call.
        # For simplicity, let's assume Bob's message is the only one with replies or high activity.
        self.mock_slack.fetch_thread_history.return_value = bobs_thread_replies

        prompt = "has anyone discussed Python?"
        channel_id = "C12345"

        # Expected messages after filtering:
        # 1. Alice's discussion "I was working on a Python project yesterday" (from main channel)
        # 2. Alice's reply in Bob's thread "I think Alice knows about Python." (from thread)
        # Bob's original question "Has anyone discussed Python lately?" should be filtered out by find_messages_containing_topic
        expected_search_results_count = 2

        messages, params = await self.manager.retrieve_and_filter_history(
            self.mock_slack,
            channel_id,
            None,
            prompt
        )

        self.assertTrue(params["is_search"])
        self.assertEqual(params["search_topic"], "python") # EXPECTING LOWERCASE
        self.assertEqual(len(messages), expected_search_results_count, f"Search results: {[m['text'] for m in messages]}")

        self.mock_slack.fetch_channel_history.assert_called_once_with(
            channel_id,
            self.manager.SEARCH_DEPTH_LIMIT # 2000
        )
        # Assert fetch_thread_history was called for Bob's thread
        # This depends on how HistoryManager identifies thread parents.
        # If Bob's message (ts: 1617983950.000100) has "reply_count" > 0 or "thread_ts" in your sample_messages (or if code makes it so)
        # then fetch_thread_history will be called for it.
        # Let's assume it is called for Bob's thread:
        self.mock_slack.fetch_thread_history.assert_any_call( # Use assert_any_call if multiple threads could be checked
            channel_id,
            thread_parent_ts, # Bob's message ts
            self.manager.THREAD_SEARCH_LIMIT # 100
        )


    async def test_retrieve_and_filter_history_thread_summary(self):
        """Test retrieving history for thread summaries."""
        thread_ts = "1617984050.000100"
        thread_messages = [
            {"ts": thread_ts, "thread_ts": thread_ts, "user": "U12345", "text": "Thread start about topic X"},
            {"ts": "1617984100.000100", "thread_ts": thread_ts, "user": "U67890", "text": "Thread reply to X"}
        ]
        self.mock_slack.fetch_thread_history.return_value = thread_messages
        # fetch_channel_history should NOT be called for a pure thread summary
        self.mock_slack.fetch_channel_history.return_value = []


        prompt = "summarize this thread"
        channel_id = "C12345"

        messages, params = await self.manager.retrieve_and_filter_history(
            self.mock_slack,
            channel_id,
            thread_ts,
            prompt
        )

        self.assertEqual(len(messages), len(thread_messages))
        self.assertIn("Thread summary", params["description"])

        self.mock_slack.fetch_thread_history.assert_called_once_with(
            channel_id,
            thread_ts,
            100 # Default limit for non-search, which summary falls under initially
        )
        self.mock_slack.fetch_channel_history.assert_not_called() # Crucial for thread summary

    async def test_retrieve_and_filter_history_channel_summary(self):
        """Test retrieving history for channel summaries."""
        self.mock_slack.fetch_channel_history.return_value = self.sample_messages
        # fetch_thread_history should not be called for a pure channel summary
        self.mock_slack.fetch_thread_history.return_value = []

        prompt = "summarize the channel"
        channel_id = "C12345"

        messages, params = await self.manager.retrieve_and_filter_history(
            self.mock_slack,
            channel_id,
            None, # No thread_ts
            prompt
        )

        self.assertEqual(len(messages), len(self.sample_messages))
        self.assertIn("Channel summary", params["description"])

        self.mock_slack.fetch_channel_history.assert_called_once_with(
            channel_id,
            100 # Default limit for non-search (which summary is initially)
        )
        self.mock_slack.fetch_thread_history.assert_not_called()

    def test_format_search_results_with_threads(self):
        """Test formatting search results with thread information."""
        query_params = {
            "search_topic": "Python",
            "is_search": True
        }
        # Messages that would be passed after find_messages_containing_topic
        # Let's say only Alice's message and a reply in a thread about Python
        python_related_messages = [
            self.sample_messages[2], # Alice: "I was working on a Python project yesterday" (ts: 1617984000.000100)
            {
                "ts": "1617983955.000100",
                "thread_ts": "1617983950.000100", # Bob's original question thread
                "user": "U12345",
                "text": "Yes, I know about Python. We used it for project Y."
            }
        ]

        # Mock slack_service directly, not self.mock_slack for this specific method call if it's passed in
        # Or, if HistoryManager uses its own internal slack_service, mock that.
        # For this test, HistoryManager.format_search_results_with_threads takes slack_service as an arg
        mock_slack_for_permalink = MagicMock()
        mock_slack_for_permalink.generate_message_permalink.return_value = "https://test.slack.com/permalink/specific"

        results = self.manager.format_search_results_with_threads(
            python_related_messages,
            query_params,
            self.user_display_names,
            "BOT123",
            "C12345",
            mock_slack_for_permalink # Pass the specifically configured mock
        )

        self.assertIn("summary", results)
        self.assertTrue(results["show_details"])
        self.assertEqual(results["topic"], "Python")
        self.assertIn("✅ Yes! I found 2 message(s) about 'Python'", results["summary"])
        self.assertEqual(len(results["threads"]), 2) # One main message, one thread
        # Further checks on thread content if necessary
        self.assertEqual(results["threads"][0]["link"], "https://test.slack.com/permalink/specific")
        self.assertEqual(results["threads"][1]["link"], "https://test.slack.com/permalink/specific")


        # Test with no messages
        empty_results = self.manager.format_search_results_with_threads(
            [],
            query_params,
            self.user_display_names,
            "BOT123",
            "C12345",
            mock_slack_for_permalink
        )

        self.assertFalse(empty_results["show_details"])
        self.assertEqual(len(empty_results["threads"]), 0)
        self.assertIn("No discussions found", empty_results["summary"])


if __name__ == '__main__':
    unittest.main()