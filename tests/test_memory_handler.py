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
        self.assertEqual(content, "seattle")
        
        memory_type, content = self.handler.classify_memory_instruction("I am working in New York")
        self.assertEqual(memory_type, "work_location")
        self.assertEqual(content, "new york")
        
        # Test home location
        memory_type, content = self.handler.classify_memory_instruction("I live in Boston")
        self.assertEqual(memory_type, "home_location")
        self.assertEqual(content, "boston")
        
        memory_type, content = self.handler.classify_memory_instruction("I was born in Chicago")
        self.assertEqual(memory_type, "home_location")
        self.assertEqual(content, "chicago")

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
        self.assertEqual(content, "buy groceries")

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
