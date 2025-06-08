import unittest
from unittest.mock import MagicMock, patch
from services.slack_service import SlackService

class TestSlackService(unittest.TestCase):
    def setUp(self):
        # Create a dummy version for testing
        self.service = SlackService(
            bot_token="xoxb-dummy-token",
            signing_secret="dummy-secret",
            app_token="xapp-dummy-token"
        )
        self.service.is_dummy = True  # Force dummy mode for testing
        
        # Sample test data
        self.sample_message = {
            "user": "U12345",
            "text": "Hello <@BOT_ID> how are you?",
            "ts": "1617984000.000100",
            "blocks": [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {"type": "text", "text": "Hello "},
                                {"type": "user", "user_id": "BOT_ID"},
                                {"type": "text", "text": " how are you?"}
                            ]
                        }
                    ]
                }
            ]
        }
    
    def test_initialization(self):
        """Test service initialization."""
        # Regular initialization
        service = SlackService("token", "secret", "app-token")
        self.assertIsNotNone(service)
        
        # Missing credentials should result in dummy mode
        dummy_service = SlackService(None, None, None)
        self.assertTrue(dummy_service.is_dummy)
    
    def test_clean_prompt_text(self):
        """Test removing bot mention from text."""
        # Set bot user ID
        self.service.bot_user_id = "U123BOT"
        
        # Test regular cleaning
        cleaned = self.service.clean_prompt_text("Hello <@U123BOT> how are you?")
        self.assertEqual(cleaned, "Hello how are you?")
        
        # Test with multiple mentions
        cleaned = self.service.clean_prompt_text("<@U123BOT> hi <@U123BOT> there")
        self.assertEqual(cleaned, "hi there")
        
        # Test with no mentions
        cleaned = self.service.clean_prompt_text("Just a regular message")
        self.assertEqual(cleaned, "Just a regular message")
    
    def test_send_message(self):
        """Test sending messages in dummy mode."""
        result = self.service.send_message("C12345", "Test message")
        self.assertTrue(result["ok"])
        self.assertIn("ts", result)
        self.assertIn(result["ts"], self.service.bot_message_timestamps)
        
        # Test with thread_ts
        result = self.service.send_message("C12345", "Thread reply", "123456.789")
        self.assertTrue(result["ok"])
    
    @patch('services.slack_service.SlackService.app', create=True)
    def test_send_message_with_real_client(self, mock_app):
        """Test sending messages with mocked Slack client."""
        # Create service with mocked app
        service = SlackService("token", "secret", "app-token")
        service.is_dummy = False
        
        # Mock the client
        mock_client = MagicMock()
        mock_app.client = mock_client
        mock_client.chat_postMessage.return_value = {"ok": True, "ts": "123456.789"}
        
        # Test sending a message
        result = service.send_message("C12345", "Hello world")
        
        # Verify the client was called correctly
        mock_client.chat_postMessage.assert_called_once_with(
            channel="C12345",
            text="Hello world",
            thread_ts=None
        )
        
        # Verify the result
        self.assertTrue(result["ok"])
        self.assertEqual(result["ts"], "123456.789")
        self.assertIn("123456.789", service.bot_message_timestamps)
    
    def test_get_user_display_name(self):
        """Test retrieving user display names."""
        # Prepare the mock user info
        self.service.user_info_cache = {
            "U12345": {"user": {"id": "U12345", "real_name": "Test User", "profile": {"display_name": "test.user"}}}
        }
        
        # Get display name from cache
        name = self.service.get_user_display_name("U12345")
        self.assertEqual(name, "test.user")  # Should use display_name
        
        # Get display name for unknown user
        name = self.service.get_user_display_name("UNKNOWN")
        self.assertEqual(name, "User UNKNOWN")  # Should use fallback
    
    def test_update_channel_stats(self):
        """Test updating and retrieving channel statistics."""
        channel_id = "C12345"
        user_id = "U12345"
        
        # Initial stats should be empty
        initial_stats = self.service.get_channel_stats(channel_id)
        self.assertEqual(initial_stats["message_count"], 0)
        self.assertEqual(len(initial_stats["participants"]), 0)
        
        # Update stats
        self.service.update_channel_stats(channel_id, user_id, "123456.789")
        
        # Check updated stats
        updated_stats = self.service.get_channel_stats(channel_id)
        self.assertEqual(updated_stats["message_count"], 1)
        self.assertIn(user_id, updated_stats["participants"])
        
        # Update with a second user
        self.service.update_channel_stats(channel_id, "U67890", "123456.790")
        
        # Check stats again
        final_stats = self.service.get_channel_stats(channel_id)
        self.assertEqual(final_stats["message_count"], 2)
        self.assertEqual(len(final_stats["participants"]), 2)
        self.assertIn("U67890", final_stats["participants"])

if __name__ == '__main__':
    unittest.main()