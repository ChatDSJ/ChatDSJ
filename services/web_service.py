import asyncio
import requests
from bs4 import BeautifulSoup
from typing import Optional
from loguru import logger
from urllib.parse import urlparse

class WebService:
    """Improved web content fetching with better scraping and error handling."""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def is_available(self) -> bool:
        return True
    
    async def fetch_content(self, url: str) -> Optional[str]:
        """Fetch content from URL using improved scraping."""
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.error(f"Invalid URL format: {url}")
            return None
        
        logger.info(f"🌐 Fetching content from: {url}")
        
        # SIMPLIFIED: Only use scraping (more reliable than LLM for this use case)
        content = await self._try_scraping(url)
        if content:
            logger.info(f"✅ Successfully scraped {len(content)} characters from {url}")
            return content
        
        logger.error(f"❌ Failed to fetch content from {url}")
        return None
             
    async def _try_scraping(self, url: str) -> Optional[str]:
        """Enhanced web scraping with better error handling."""
        try:
            response = await asyncio.to_thread(self._fetch_url, url)
            if not response:
                return None
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                logger.warning(f"Non-HTML content type: {content_type}")
                return None
            
            return await asyncio.to_thread(self._extract_text, response.text, url)
            
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            return None
    
    def _fetch_url(self, url: str) -> Optional[requests.Response]:
        """Fetch URL with better error handling and retries."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching URL (attempt {attempt + 1}/{max_retries}): {url}")
                
                response = self.session.get(
                    url, 
                    timeout=30,  # Increased timeout
                    allow_redirects=True,
                    stream=False
                )
                
                response.raise_for_status()
                
                # Check response size (avoid huge responses)
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > 10_000_000:  # 10MB limit
                    logger.warning(f"Content too large: {content_length} bytes")
                    return None
                
                logger.debug(f"Successfully fetched {len(response.content)} bytes")
                return response
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
                if attempt == max_retries - 1:
                    return None
                continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed on attempt {attempt + 1} for {url}: {e}")
                if attempt == max_retries - 1:
                    return None
                continue
                
        return None
    
    def _extract_text(self, html: str, url: str) -> str:
        """Enhanced text extraction with better content selection."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements more aggressively
            unwanted_tags = [
                'script', 'style', 'nav', 'footer', 'header', 'aside',
                'form', 'button', 'input', 'select', 'textarea',
                'iframe', 'embed', 'object', 'applet',
                'meta', 'link', 'title', 'base'
            ]
            
            for tag in unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Remove unwanted classes and IDs (common patterns)
            unwanted_patterns = [
                'ad', 'advertisement', 'banner', 'popup', 'modal',
                'sidebar', 'menu', 'navigation', 'breadcrumb',
                'comment', 'social', 'share', 'related', 'recommended'
            ]
            
            for pattern in unwanted_patterns:
                for element in soup.find_all(attrs={'class': lambda x: x and pattern in ' '.join(x).lower()}):
                    element.decompose()
                for element in soup.find_all(attrs={'id': lambda x: x and pattern in x.lower()}):
                    element.decompose()
            
            # Try to find main content in order of preference
            main_content = None
            
            # Strategy 1: Look for article/main tags
            main_content = soup.find('article') or soup.find('main')
            
            # Strategy 2: Look for common content classes
            if not main_content:
                content_selectors = [
                    'div[class*="content"]',
                    'div[class*="article"]',
                    'div[class*="post"]',
                    'div[class*="story"]',
                    'div[class*="entry"]',
                    '[role="main"]'
                ]
                
                for selector in content_selectors:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
            
            # Strategy 3: Find the div with the most text content
            if not main_content:
                divs = soup.find_all('div')
                if divs:
                    main_content = max(divs, key=lambda d: len(d.get_text(strip=True)))
            
            # Fallback to body
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract text with better formatting
            text = main_content.get_text(separator='\n', strip=True)
            
            # Clean up the text
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line and len(line) > 3:  # Filter out very short lines
                    lines.append(line)
            
            # Remove duplicate consecutive lines
            cleaned_lines = []
            prev_line = None
            for line in lines:
                if line != prev_line:
                    cleaned_lines.append(line)
                prev_line = line
            
            result = '\n'.join(cleaned_lines)
            
            # Log extraction results
            logger.debug(f"Extracted {len(result)} characters from {len(html)} HTML characters")
            
            # Quality check - if we got very little text, something probably went wrong
            if len(result) < 100:
                logger.warning(f"Very little text extracted ({len(result)} chars) from {url}")
            
            return result
            
        except Exception as e:
            logger.error(f"Text extraction failed for {url}: {e}")
            # Fallback: just get all text
            try:
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text(separator='\n', strip=True)
            except:
                return ""