import re
from typing import List, Dict, Any, Set, Tuple

# Handle loguru import gracefully
try:
    from loguru import logger
except ImportError:
    # Fallback to standard logging if loguru not available
    import logging
    logger = logging.getLogger(__name__)

def extract_notion_links(messages: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract unique Notion page IDs from Slack messages.
    IMPROVED: Handles all URL formats AND extracts multiple IDs per URL.
    """
    # Comprehensive patterns to catch ALL Notion URL formats
    notion_url_patterns = [
        # Standard notion.so URLs (plain and in angle brackets)
        r'<?https://(?:www\.)?notion\.so/[^\s<>]*>?',
        
        # Workspace URLs (company.notion.so)
        r'<?https://[^\/\s]+\.notion\.so/[^\s<>]*>?',
        
        # Legacy formats just in case
        r'<?https://notion\.site/[^\s<>]*>?',
    ]
    
    page_ids = set()
    
    for message in messages:
        text = message.get('text', '')
        if not text or 'notion.' not in text:
            continue
            
        logger.debug(f"Checking message for Notion links: {text}")
        
        # Extract all potential Notion URLs using all patterns
        all_notion_urls = set()
        for pattern in notion_url_patterns:
            urls = re.findall(pattern, text, re.IGNORECASE)
            all_notion_urls.update(urls)
        
        # From each URL, extract ALL 32-char hex strings
        for url in all_notion_urls:
            # Clean up angle brackets if present
            clean_url = url.strip('<>')
            
            # Extract ALL 32-character hex strings from this URL
            hex_ids = re.findall(r'[a-f0-9]{32}', clean_url, re.IGNORECASE)
            
            if hex_ids:
                logger.debug(f"Found Notion URL: {clean_url}")
                logger.debug(f"Extracted hex IDs: {hex_ids}")
                page_ids.update(hex_ids)
    
    logger.info(f"Extracted {len(page_ids)} unique Notion page IDs from {len(messages)} messages: {list(page_ids)}")
    return page_ids

def format_notion_page_id(page_id: str) -> str:
    """Format page ID with dashes for Notion API."""
    if len(page_id) == 32 and '-' not in page_id:
        return f"{page_id[:8]}-{page_id[8:12]}-{page_id[12:16]}-{page_id[16:20]}-{page_id[20:]}"
    return page_id

def extract_page_title_from_content(content: str, page_id: str) -> str:
    """Extract a reasonable page title from content, fallback to page ID."""
    if not content:
        return f"Page {page_id[:8]}"
    
    lines = content.strip().split('\n')
    for line in lines[:3]:  # Check first 3 lines
        line = line.strip()
        if line and not line.startswith('#') and len(line) < 100:
            # Use first substantial line as title
            return line[:50] + ('...' if len(line) > 50 else '')
    
    return f"Page {page_id[:8]}"

# TEST: Let's verify this handles all cases
def test_extraction():
    """Test function to verify all URL formats work"""
    test_cases = [
        "Plain: https://www.notion.so/Page-abc123def456789012345678901234567890",
        "Unfurled: <https://www.notion.so/Page-abc123def456789012345678901234567890>", 
        "Short: https://notion.so/Page-abc123def456789012345678901234567890",
        "Workspace: https://mycompany.notion.so/Page-abc123def456789012345678901234567890",
        "Double ID: https://notion.so/Page-9ed47817425c44cd8e2fe99f6dd42177-4476094c0cfe41e8ad5228281cfe5daa",
        "Mixed: Check <https://notion.so/Page-abc123def456789012345678901234567890> and text"
    ]
    
    for test_text in test_cases:
        fake_message = {"text": test_text}
        result = extract_notion_links([fake_message])
        print(f"Input: {test_text}")
        print(f"Extracted: {list(result)}")
        print("---")