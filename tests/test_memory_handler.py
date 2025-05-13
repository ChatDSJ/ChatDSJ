import unittest
from unittest.mock import MagicMock, patch
from handler.memory_handler import MemoryHandler, SectionType, PropertyType

class TestMemoryHandler(unittest.TestCase):
    def setUp(self):
        # Mock notion service
        self.mock_notion = MagicMock()
        
        # Create memory handler
        self.handler = MemoryHandler(self.mock_notion)
    
    def test_classify_memory_instruction(self):
        """Test classifying different types of memory instructions."""
        # Test nickname
        memory_type, content = self.handler.classify_memory_instruction("call me John")
        self.assertEqual(memory_type, "known_fact")
        self.assertEqual(content, "call me John")
        
        # Test work location
        memory_type, content = self.handler.classify_memory_instruction("I work from Seattle")
        self.assertEqual(memory_type, "work_location")
        self.assertEqual(content, "Seattle")
        
        # Test home location
        memory_type, content = self.handler.classify_memory_instruction("I live in New York")
        self.assertEqual(memory_type, "home_location")
        self.assertEqual(content, "New York")
        
        # Test known fact
        memory_type, content = self.handler.classify_memory_instruction("remember that I like coffee")
        self.assertEqual(memory_type, "known_fact")
        self.assertEqual(content, "I like coffee")
        
        # Test preference
        memory_type, content = self.handler.classify_memory_instruction("I prefer short responses")
        self.assertEqual(memory_type, "preference")
        self.assertEqual(content, "short responses")
        
        # Test project
        memory_type, content = self.handler.classify_memory_instruction("add project Building a chatbot")
        self.assertEqual(memory_type, "project_add")
        self.assertEqual(content, "Building a chatbot")
        
        # Test replace projects
        memory_type, content = self.handler.classify_memory_instruction("my new project is AI research")
        self.assertEqual(memory_type, "project_replace")
        self.assertEqual(content, "AI research")
        
        # Test TODO
        memory_type, content = self.handler.classify_memory_instruction("TODO: Buy milk")
        self.assertEqual(memory_type, "todo")
        self.assertEqual(content, "Buy milk")
        
        # Test question (not a memory instruction)
        memory_type, content = self.handler.classify_memory_instruction("what time is it?")
        self.assertEqual(memory_type, "unknown")
        self.assertIsNone(content)
        
        # Test with bot mention
        memory_type, content = self.handler.classify_memory_instruction("<@BOT123> remember that I like tea")
        self.assertEqual(memory_type, "known_fact")
        self.assertEqual(content, "I like tea")
    
    def test_handle_memory_instruction_nickname(self):
        """Test handling nickname memory instructions."""
        # Mock update_user_name to succeed
        self.handler.update_user_name = MagicMock(return_value=True)
        
        # Handle nickname instruction
        response = self.handler.handle_memory_instruction(
            "U12345", "call me Test User"
        )
        
        # Verify response
        self.assertIn("I'll call you", response)
        
        # Verify update was called with correct parameters
        self.handler.update_user_name.assert_called_once_with(
            "U12345", "call me Test User"
        )
        
        # Test with failure
        self.handler.update_user_name.return_value = False
        
        response = self.handler.handle_memory_instruction(
            "U12345", "call me Test User"
        )
        
        self.assertIn("couldn't update", response.lower())
    
    def test_handle_memory_instruction_work_location(self):
        """Test handling work location memory instructions."""
        # Mock update_location to succeed
        self.handler.update_location = MagicMock(return_value=True)
        
        # Handle work location instruction
        response = self.handler.handle_memory_instruction(
            "U12345", "I work from Seattle"
        )
        
        # Verify response
        self.assertIn("I've noted that you work", response)
        
        # Verify update was called with correct parameters
        self.handler.update_location.assert_called_once_with(
            "U12345", "work", "Seattle"
        )
    
    def test_handle_memory_instruction_known_fact(self):
        """Test handling known fact memory instructions."""
        # Mock add_known_fact to succeed
        self.handler.add_known_fact = MagicMock(return_value=True)
        
        # Handle fact instruction
        response = self.handler.handle_memory_instruction(
            "U12345", "remember that I like coffee"
        )
        
        # Verify response
        self.assertIn("I've added that", response)
        
        # Verify add_known_fact was called
        self.handler.add_known_fact.assert_called_once_with(
            "U12345", "I like coffee"
        )
    
    def test_handle_memory_instruction_preference(self):
        """Test handling preference memory instructions."""
        # Mock add_preference to succeed
        self.handler.add_preference = MagicMock(return_value=True)
        
        # Handle preference instruction
        response = self.handler.handle_memory_instruction(
            "U12345", "I prefer short responses"
        )
        
        # Verify response
        self.assertIn("Understood", response)
        
        # Verify add_preference was called
        self.handler.add_preference.assert_called_once_with(
            "U12345", "short responses"
        )
    
    def test_handle_memory_instruction_todo(self):
        """Test handling TODO memory instructions."""
        # Mock add_todo to succeed
        self.handler.add_todo = MagicMock(return_value=True)
        
        # Handle TODO instruction
        response = self.handler.handle_memory_instruction(
            "U12345", "TODO: Buy milk"
        )
        
        # Verify response
        self.assertIn("✅", response)
        
        # Verify add_todo was called
        self.handler.add_todo.assert_called_once_with(
            "U12345", "Buy milk"
        )
    
    def test_find_section_block(self):
        """Test finding section blocks in Notion pages."""
        # Mock client blocks response
        self.mock_notion.client.blocks.children.list.return_value = {
            "results": [
                {
                    "id": "block_1",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"plain_text": "Preferences"}]
                    }
                },
                {
                    "id": "block_2",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"plain_text": "Known Facts"}]
                    }
                }
            ]
        }
        
        # Find Preferences section
        section = self.handler._find_section_block("page_123", "Preferences")
        
        # Verify correct section was found
        self.assertIsNotNone(section)
        self.assertEqual(section["id"], "block_1")
        
        # Find Known Facts section
        section = self.handler._find_section_block("page_123", "Known Facts")
        
        # Verify correct section was found
        self.assertIsNotNone(section)
        self.assertEqual(section["id"], "block_2")
        
        # Find non-existent section
        section = self.handler._find_section_block("page_123", "Projects")
        
        # Should return None
        self.assertIsNone(section)
    
    def test_create_section(self):
        """Test creating a new section in a Notion page."""
        # Mock client response
        self.mock_notion.client.blocks.children.append.return_value = {
            "results": [
                {
                    "id": "new_block_123",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"plain_text": "New Section"}]
                    }
                }
            ]
        }
        
        # Create new section
        section = self.handler._create_section("page_123", "New Section")
        
        # Verify client was called correctly
        self.mock_notion.client.blocks.children.append.assert_called_once()
        call_args = self.mock_notion.client.blocks.children.append.call_args[1]
        self.assertEqual(call_args["block_id"], "page_123")
        
        # Verify returned section
        self.assertEqual(section["id"], "new_block_123")
    
    def test_add_known_fact(self):
        """Test adding a known fact to a user's page."""
        # Mock page ID retrieval
        self.mock_notion.get_user_page_id = MagicMock(return_value="page_123")
        
        # Mock finding Known Facts section
        self.handler._find_section_block = MagicMock(return_value={"id": "section_123"})
        
        # Mock client
        self.mock_notion.client.blocks.children.append.return_value = {"results": [{"id": "new_fact_123"}]}
        
        # Add known fact
        result = self.handler.add_known_fact("U12345", "I like coffee")
        
        # Verify result
        self.assertTrue(result)
        
        # Verify section was checked
        self.handler._find_section_block.assert_called_once_with("page_123", "Known Facts")
        
        # Verify fact was added
        self.mock_notion.client.blocks.children.append.assert_called_once()
        call_args = self.mock_notion.client.blocks.children.append.call_args[1]
        self.assertEqual(call_args["block_id"], "section_123")
        
        # Test with creating new section
        self.handler._find_section_block.return_value = None
        self.handler._create_section = MagicMock(return_value={"id": "new_section_123"})
        
        # Reset mocks
        self.mock_notion.client.blocks.children.append.reset_mock()
        
        # Add known fact with new section
        result = self.handler.add_known_fact("U12345", "I like coffee")
        
        # Verify result
        self.assertTrue(result)
        
        # Verify section was created
        self.handler._create_section.assert_called_once_with("page_123", "Known Facts")
        
        # Verify fact was added to new section
        self.mock_notion.client.blocks.children.append.assert_called_once()
        call_args = self.mock_notion.client.blocks.children.append.call_args[1]
        self.assertEqual(call_args["block_id"], "new_section_123")

if __name__ == '__main__':
    unittest.main()