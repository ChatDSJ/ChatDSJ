import unittest
from unittest.mock import MagicMock, patch
from services.cached_notion_service import CachedNotionService

class TestCachedNotionService(unittest.TestCase):
    def setUp(self):
        # Create service with mocked Notion client
        self.mock_client = MagicMock()
        
        with patch('services.cached_notion_service.Client', return_value=self.mock_client):
            self.service = CachedNotionService(
                notion_api_token="secret_dummy_token",
                notion_user_db_id="dummy_db_id",
                cache_ttl=60,
                cache_max_size=100
            )
    
    def test_initialization(self):
        """Test service initialization."""
        self.assertEqual(self.service.api_token, "secret_dummy_token")
        self.assertEqual(self.service.user_db_id, "dummy_db_id")
        self.assertEqual(self.service.cache_ttl, 60)
        self.assertIsNotNone(self.service.memory_handler)
        
        # Test caches are initialized
        self.assertEqual(len(self.service.cache), 0)
        self.assertEqual(len(self.service.page_content_cache), 0)
    
    def test_is_available(self):
        """Test availability check."""
        # Should be available with client and DB ID
        self.assertTrue(self.service.is_available())
        
        # Test without client
        self.service.client = None
        self.assertFalse(self.service.is_available())
        
        # Reset client
        self.service.client = self.mock_client
        
        # Test without DB ID
        self.service.user_db_id = None
        self.assertFalse(self.service.is_available())
    
    def test_clear_cache(self):
        """Test clearing the caches."""
        # Add items to caches
        with self.service.cache_lock:
            self.service.cache["test_key"] = "test_value"
        
        with self.service.page_content_lock:
            self.service.page_content_cache["test_page"] = "test_content"
        
        # Clear caches
        self.service.clear_cache()
        
        # Verify caches are empty
        self.assertEqual(len(self.service.cache), 0)
        self.assertEqual(len(self.service.page_content_cache), 0)
    
    def test_invalidate_user_cache(self):
        """Test invalidating cache for a specific user."""
        user_id = "U12345"
        
        # Add user-specific entries to caches
        with self.service.cache_lock:
            self.service.cache[f"user_page_id_{user_id}"] = "page_id_123"
            self.service.cache[f"user_properties_{user_id}"] = {"prop": "value"}
            self.service.cache["unrelated_key"] = "unrelated_value"
        
        with self.service.page_content_lock:
            self.service.page_content_cache[f"user_page_content_{user_id}"] = "page content"
        
        # Invalidate user cache
        self.service.invalidate_user_cache(user_id)
        
        # Verify user-specific entries are removed
        self.assertNotIn(f"user_page_id_{user_id}", self.service.cache)
        self.assertNotIn(f"user_properties_{user_id}", self.service.cache)
        self.assertNotIn(f"user_page_content_{user_id}", self.service.page_content_cache)
        
        # Verify unrelated entries are kept
        self.assertIn("unrelated_key", self.service.cache)
    
    def test_get_user_page_id_with_cache_hit(self):
        """Test retrieving user page ID with cache hit."""
        user_id = "U12345"
        expected_page_id = "page_123"
        
        # Add to cache
        with self.service.cache_lock:
            self.service.cache[f"user_page_id_{user_id}"] = expected_page_id
        
        # Get page ID
        page_id = self.service.get_user_page_id(user_id)
        
        # Verify result and cache hit
        self.assertEqual(page_id, expected_page_id)
        self.assertEqual(self.service.cache_stats["hits"], 1)
        self.assertEqual(self.service.cache_stats["misses"], 0)
        
        # Client should not be called
        self.mock_client.databases.query.assert_not_called()
    
    def test_get_user_page_id_with_cache_miss(self):
        """Test retrieving user page ID with cache miss."""
        user_id = "U12345"
        expected_page_id = "page_123"
        
        # Setup mock response
        mock_response = {
            "results": [{"id": expected_page_id}]
        }
        self.mock_client.databases.query.return_value = mock_response
        
        # Get page ID
        page_id = self.service.get_user_page_id(user_id)
        
        # Verify result and cache miss
        self.assertEqual(page_id, expected_page_id)
        self.assertEqual(self.service.cache_stats["hits"], 0)
        self.assertEqual(self.service.cache_stats["misses"], 1)
        
        # Verify client was called with correct parameters
        self.mock_client.databases.query.assert_called_once()
        call_args = self.mock_client.databases.query.call_args[1]
        self.assertEqual(call_args["database_id"], "dummy_db_id")
        self.assertEqual(call_args["filter"]["property"], "UserID")
        self.assertEqual(call_args["filter"]["title"]["equals"], user_id)
        
        # Check that result was cached
        with self.service.cache_lock:
            self.assertIn(f"user_page_id_{user_id}", self.service.cache)
            self.assertEqual(self.service.cache[f"user_page_id_{user_id}"], expected_page_id)
    
    def test_get_user_page_id_not_found(self):
        """Test retrieving user page ID when not found."""
        user_id = "U12345"
        
        # Setup mock response with no results
        mock_response = {"results": []}
        self.mock_client.databases.query.return_value = mock_response
        
        # Get page ID
        page_id = self.service.get_user_page_id(user_id)
        
        # Verify result
        self.assertIsNone(page_id)
        
        # Check that None was cached
        with self.service.cache_lock:
            self.assertIn(f"user_page_id_{user_id}", self.service.cache)
            self.assertIsNone(self.service.cache[f"user_page_id_{user_id}"])
    
    def test_store_user_nickname(self):
        """Test storing a user's nickname."""
        user_id = "U12345"
        nickname = "TestUser"
        
        # Mock get_user_page_id to return None (new user)
        with patch.object(self.service, 'get_user_page_id', return_value=None):
            # Mock page creation
            self.mock_client.pages.create.return_value = {"id": "new_page_123"}
            
            # Store nickname
            result = self.service.store_user_nickname(user_id, nickname)
            
            # Verify result
            self.assertTrue(result)
            
            # Verify client was called correctly for page creation
            self.mock_client.pages.create.assert_called_once()
            call_args = self.mock_client.pages.create.call_args[1]
            self.assertEqual(call_args["parent"]["database_id"], "dummy_db_id")
            self.assertEqual(call_args["properties"]["UserID"]["title"][0]["text"]["content"], user_id)
            self.assertEqual(call_args["properties"]["PreferredName"]["rich_text"][0]["text"]["content"], nickname)
        
        # Test updating existing user
        with patch.object(self.service, 'get_user_page_id', return_value="existing_page_123"):
            # Reset mock
            self.mock_client.reset_mock()
            
            # Store nickname
            result = self.service.store_user_nickname(user_id, "NewNickname")
            
            # Verify result
            self.assertTrue(result)
            
            # Verify client was called correctly for page update
            self.mock_client.pages.update.assert_called_once()
            call_args = self.mock_client.pages.update.call_args[1]
            self.assertEqual(call_args["page_id"], "existing_page_123")
            self.assertEqual(call_args["properties"]["PreferredName"]["rich_text"][0]["text"]["content"], "NewNickname")
    
    def test_handle_nickname_command(self):
        """Test handling nickname commands from chat."""
        user_id = "U12345"
        
        # Mock store_user_nickname to succeed
        with patch.object(self.service, 'store_user_nickname', return_value=True):
            # Test explicit nickname command
            response, success = self.service.handle_nickname_command(
                "call me TestUser", user_id, "Display Name"
            )
            
            # Verify response
            self.assertTrue(success)
            self.assertIn("TestUser", response)
            
            # Test "my name is" format
            response, success = self.service.handle_nickname_command(
                "my name is John Doe", user_id, "Display Name"
            )
            
            # Verify response
            self.assertTrue(success)
            self.assertIn("John Doe", response)
            
            # Test quoted name
            response, success = self.service.handle_nickname_command(
                'call me "Captain Awesome"', user_id, "Display Name"
            )
            
            # Verify response
            self.assertTrue(success)
            self.assertIn("Captain Awesome", response)
        
        # Test with store failure
        with patch.object(self.service, 'store_user_nickname', return_value=False):
            response, success = self.service.handle_nickname_command(
                "call me TestUser", user_id, "Display Name"
            )
            
            # Verify response
            self.assertFalse(success)
            self.assertIn("trouble", response.lower())
        
        # Test non-nickname command
        response, success = self.service.handle_nickname_command(
            "what time is it?", user_id, "Display Name"
        )
        
        # Verify response
        self.assertFalse(success)
        self.assertIsNone(response)
    
    def test_get_user_preferred_name(self):
        """Test retrieving a user's preferred name."""
        user_id = "U12345"
        
        # Mock get_user_page_properties to return properties with preferred name
        properties = {
            "PreferredName": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "Test User"}]
            }
        }
        
        with patch.object(self.service, 'get_user_page_properties', return_value=properties):
            # Get preferred name
            name = self.service.get_user_preferred_name(user_id)
            
            # Verify result
            self.assertEqual(name, "Test User")
        
        # Test with empty rich_text array
        properties = {
            "PreferredName": {
                "type": "rich_text",
                "rich_text": []
            }
        }
        
        with patch.object(self.service, 'get_user_page_properties', return_value=properties):
            name = self.service.get_user_preferred_name(user_id)
            self.assertIsNone(name)
        
        # Test with no properties
        with patch.object(self.service, 'get_user_page_properties', return_value=None):
            name = self.service.get_user_preferred_name(user_id)
            self.assertIsNone(name)
    
    def test_add_todo_item(self):
        """Test adding a TODO item to a user's page."""
        user_id = "U12345"
        todo_text = "Finish tests"
        
        # Mock get_user_page_id for existing user
        with patch.object(self.service, 'get_user_page_id', return_value="page_123"):
            # Mock block append
            self.mock_client.blocks.children.append.return_value = {"results": [{"id": "block_123"}]}
            
            # Add TODO item
            result = self.service.add_todo_item(user_id, todo_text)
            
            # Verify result
            self.assertTrue(result)
            
            # Verify client was called correctly
            self.mock_client.blocks.children.append.assert_called_once()
            call_args = self.mock_client.blocks.children.append.call_args[1]
            self.assertEqual(call_args["block_id"], "page_123")
            
            # Verify the to_do block was created correctly
            todo_block = call_args["children"][0]
            self.assertEqual(todo_block["type"], "to_do")
            self.assertEqual(todo_block["to_do"]["rich_text"][0]["text"]["content"], todo_text)
            self.assertFalse(todo_block["to_do"]["checked"])
            
            # Verify cache was invalidated
            with self.service.cache_lock:
                self.assertNotIn(f"user_page_content_{user_id}", self.service.page_content_cache)
        
        # Test with new user (no page yet)
        with patch.object(self.service, 'get_user_page_id', side_effect=[None, "new_page_123"]):
            # Mock store_user_nickname for page creation
            with patch.object(self.service, 'store_user_nickname', return_value=True):
                # Reset mock
                self.mock_client.reset_mock()
                
                # Add TODO item for new user
                result = self.service.add_todo_item(user_id, todo_text)
                
                # Verify result
                self.assertTrue(result)
                
                # Verify page was created and then blocks were appended
                self.mock_client.blocks.children.append.assert_called_once()

if __name__ == '__main__':
    unittest.main()