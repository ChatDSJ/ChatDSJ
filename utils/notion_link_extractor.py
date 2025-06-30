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
    Handles both plain URLs and Slack's angle bracket format.
    """
    # Updated patterns to handle Slack's angle bracket unfurling: <https://notion.so/xyz>
    notion_patterns = [
        r'<https://www\.notion\.so/[^/>]+/([a-f0-9]{32})[^>]*>',  # Slack unfurled standard
        r'<https://notion\.so/[^/>]+/([a-f0-9]{32})[^>]*>',       # Slack unfurled short
        r'<https://www\.notion\.so/([a-f0-9]{32})[^>]*>',         # Slack unfurled direct
        r'<https://notion\.so/([a-f0-9]{32})[^>]*>',              # Slack unfurled direct short
        r'https://www\.notion\.so/[^/\s]+/([a-f0-9]{32})',        # Plain standard format
        r'https://notion\.so/[^/\s]+/([a-f0-9]{32})',             # Plain short format  
        r'https://www\.notion\.so/([a-f0-9]{32})',                # Plain direct page
        r'https://notion\.so/([a-f0-9]{32})'                      # Plain direct page short
    ]
    
    page_ids = set()
    
    for message in messages:
        text = message.get('text', '')
        if not text or 'notion.so' not in text:
            continue
            
        for pattern in notion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            page_ids.update(matches)
    
    logger.debug(f"Extracted {len(page_ids)} unique Notion page IDs from {len(messages)} messages")
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