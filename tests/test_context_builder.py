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
