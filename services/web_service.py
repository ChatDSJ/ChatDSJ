import asyncio
import requests
from bs4 import BeautifulSoup
from typing import Optional
from loguru import logger
from urllib.parse import urlparse

class WebService:
    """Hybrid web content fetching: LLM-first with scraping fallback."""
    
    def __init__(self, llm_service=None):  # Changed parameter name
        self.llm_service = llm_service      # Use any LLM service
        self.anthropic_service = anthropic_service
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ChatDSJ Bot 1.0',
            'Accept': 'text/html,application/xhtml+xml'
        })
    
    def is_available(self) -> bool:
        return True
    
    async def fetch_content(self, url: str) -> Optional[str]:
        """Try LLM first, fallback to scraping."""
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None
        
        # Method 1: Try LLM web access
        content = await self._try_llm_access(url)
        if content:
            logger.info(f"LLM successfully fetched content from {url}")
            return content
        
        # Method 2: Fallback to traditional scraping
        content = await self._try_scraping(url)
        if content:
            logger.info(f"Scraping successfully fetched content from {url}")
            return content
        
        logger.error(f"Both LLM and scraping failed for {url}")
        return None
    
    async def _try_llm_access(self, url: str) -> Optional[str]:
        """Try accessing content via LLM."""
        if not self.llm_service:
            return None
        
        try:
            prompt = f"Please visit this URL and extract the main content: {url}\n\nReturn only the article text content, no commentary."
            
            # Use OpenAI's web search capability
            response, _ = await self.llm_service.get_web_search_completion_async(
                prompt=prompt,
                timeout=30.0
            )
            
            if response and len(response.strip()) > 100:
                return response.strip()
            
        except Exception as e:
            logger.warning(f"LLM web access failed for {url}: {e}")
        
        return None
             
    async def _try_scraping(self, url: str) -> Optional[str]:
        """Fallback traditional scraping."""
        try:
            response = await asyncio.to_thread(self._fetch_url, url)
            if not response:
                return None
            
            return await asyncio.to_thread(self._extract_text, response.text)
            
        except Exception as e:
            logger.warning(f"Scraping failed for {url}: {e}")
            return None
    
    def _fetch_url(self, url: str) -> Optional[requests.Response]:
        """Fetch URL with requests."""
        try:
            response = self.session.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            return response
        except:
            return None
    
    def _extract_text(self, html: str) -> str:
        """Extract text using BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Try to find main content
        main_content = (soup.find('main') or 
                       soup.find('article') or 
                       soup.find(class_='content') or 
                       soup.find('body') or soup)
        
        text = main_content.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)