import unittest
from unittest.mock import MagicMock, patch
from services.notion_parser import NotionContextManager
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
        
        # Test with empty properties
        self.assertEqual(extract_structured_fields({}), [])
        self.assertEqual(extract_structured_fields(None), [])

    def test_process_notion_content(self):
        """Test parsing Notion content with sections."""
        content = """Projects
* Project A
* Project B

Preferences
* Always write your answers in rhymed verse
* Keep responses concise

Known Facts
* Background in tech
* Lives in San Juan

Instructions
* When I say "remember X", store it under Known Facts
* When I ask "what do you know about me?", return all facts"""

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
* Use bullet points
* Be concise

Known Facts
* Works as a developer
* Enjoys hiking"""
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
        
        # Empty content should still return base prompt
        empty_result = self.context_manager.build_openai_system_prompt("Base", "", None)
        self.assertEqual(empty_result, "Base")

if __name__ == "__main__":
    unittest.main()