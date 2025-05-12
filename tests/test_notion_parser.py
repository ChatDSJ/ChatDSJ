import unittest
from unittest.mock import MagicMock, patch
from services.notion_parser import NotionContextManager
from utils.context_builder import extract_structured_fields

class TestNotionParser(unittest.TestCase):
    def setUp(self):
        self.context_manager = NotionContextManager()
        
    def test_extract_structured_fields_empty(self):
        """Test extraction with empty properties"""
        result = extract_structured_fields(None)
        self.assertEqual(result, [])
        
        result = extract_structured_fields({})
        self.assertEqual(result, [])
        
    def test_extract_structured_fields_rich_text(self):
        """Test extraction of rich_text fields"""
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
            }
        }
        
        result = extract_structured_fields(properties)
        self.assertEqual(len(result), 1)
        self.assertIn("WorkLocation: New York", result)
        self.assertNotIn("PreferredName: John", result)  # Should be skipped
        
    def test_extract_structured_fields_title(self):
        """Test extraction of title fields"""
        properties = {
            "UserID": {
                "type": "title",
                "title": [{"plain_text": "U12345"}]
            }
        }
        
        result = extract_structured_fields(properties)
        self.assertEqual(result, ["UserID: U12345"])
        
    def test_extract_structured_fields_select(self):
        """Test extraction of select fields"""
        properties = {
            "Role": {
                "type": "select",
                "select": {"name": "Developer"}
            },
            "EmptySelect": {
                "type": "select",
                "select": {}
            }
        }
        
        result = extract_structured_fields(properties)
        self.assertEqual(len(result), 1)
        self.assertIn("Role: Developer", result)
        
    def test_extract_structured_fields_multi_select(self):
        """Test extraction of multi_select fields"""
        properties = {
            "Languages": {
                "type": "multi_select",
                "multi_select": [
                    {"name": "Python"},
                    {"name": "JavaScript"},
                    {"name": "TypeScript"}
                ]
            },
            "EmptyMultiSelect": {
                "type": "multi_select",
                "multi_select": []
            }
        }
        
        result = extract_structured_fields(properties)
        self.assertEqual(len(result), 1)
        self.assertIn("Languages: Python, JavaScript, TypeScript", result)
        
    def test_extract_structured_fields_mixed(self):
        """Test extraction of mixed field types"""
        properties = {
            "UserID": {
                "type": "title",
                "title": [{"plain_text": "U12345"}]
            },
            "Role": {
                "type": "select",
                "select": {"name": "Developer"}
            },
            "WorkLocation": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "New York"}]
            },
            "Languages": {
                "type": "multi_select",
                "multi_select": [
                    {"name": "Python"},
                    {"name": "JavaScript"}
                ]
            }
        }
        
        result = extract_structured_fields(properties)
        self.assertEqual(len(result), 4)
        self.assertIn("UserID: U12345", result)
        self.assertIn("Role: Developer", result)
        self.assertIn("WorkLocation: New York", result)
        self.assertIn("Languages: Python, JavaScript", result)

    def test_process_notion_content(self):
        """Test parsing Notion content with sections"""
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
        self.assertTrue("raw_sections" in result)
        self.assertEqual(len(result["raw_sections"]), 4)
        self.assertTrue("Projects" in result["raw_sections"])
        self.assertTrue("Preferences" in result["raw_sections"])
        self.assertTrue("Known Facts" in result["raw_sections"])
        self.assertTrue("Instructions" in result["raw_sections"])
        
        # Check preference detection
        self.assertTrue(result["has_verse_preference"])
        self.assertTrue(result["has_format_preference"])
        
        # Check section content
        self.assertEqual(len(result["raw_sections"]["Projects"]), 2)
        self.assertEqual(result["raw_sections"]["Projects"][0], "* Project A")
        self.assertEqual(result["raw_sections"]["Projects"][1], "* Project B")

if __name__ == "__main__":
    unittest.main()