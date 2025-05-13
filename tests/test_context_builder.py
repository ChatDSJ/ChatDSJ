import unittest
from unittest.mock import MagicMock, patch
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
* Build a chatbot
* Improve documentation

Preferences
* Use bullet points
* Keep responses concise
* Write in a friendly tone

Known Facts
* Working as a developer
* Enjoys hiking on weekends
* Based in Seattle"""
        
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
* Build a chatbot

Known Facts
* Working as a developer"""
        
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
    
    @patch('utils.context_builder.extract_structured_fields')
    @patch('utils.context_builder.extract_user_preferences')
    async def test_get_enhanced_user_context(self, mock_extract_preferences, mock_extract_fields):
        """Test building enhanced user context."""
        # Configure mocks
        mock_extract_preferences.return_value = [
            "Use bullet points",
            "Keep responses concise"
        ]
        
        mock_extract_fields.return_value = [
            "UserID: U12345",
            "WorkLocation: Remote",
            "Role: Developer"
        ]
        
        # Get enhanced context
        context = await get_enhanced_user_context(
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
        
        # Test with missing components
        self.mock_notion.get_user_page_content.return_value = None
        self.mock_notion.get_user_preferred_name.return_value = None
        
        mock_extract_preferences.return_value = []
        mock_extract_fields.return_value = []
        
        # Should still include base prompt
        context = await get_enhanced_user_context(
            self.mock_notion,
            "U12345",
            "Base prompt"
        )
        
        self.assertIn("Base prompt", context)
        self.assertNotIn("USER PREFERENCES", context)
        self.assertNotIn("USER DATABASE PROPERTIES", context)

if __name__ == '__main__':
    unittest.main()