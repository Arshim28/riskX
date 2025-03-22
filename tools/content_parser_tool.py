from typing import Dict, List, Any, Optional, Tuple
import asyncio
import aiohttp
from urllib.parse import urlparse
from datetime import datetime
import re
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


class ContentParserTool(BaseTool):
    name = "content_parser_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_article_content(self, url: str) -> Tuple[Optional[str], Optional[Dict]]:
        self.logger.info(f"Fetching content from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        metadata = {
            "source_domain": urlparse(url).netloc,
            "fetch_timestamp": datetime.now().isoformat(),
            "fetch_method": None,
            "content_size": 0,
            "extraction_success": False
        }
        
        try:
            # First try using Jina
            jina_url = "https://r.jina.ai/" + url
            async with aiohttp.ClientSession() as session:
                async with session.get(jina_url, timeout=self.config.get("content_fetch", {}).get("timeout", 30), headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        if len(content) > 500:
                            metadata["fetch_method"] = "jina"
                            metadata["content_size"] = len(content)
                            metadata["extraction_success"] = True
                            self.logger.info(f"Jina extraction successful for: {url}")
                            return content, metadata
        except Exception as e:
            self.logger.warning(f"Jina extraction failed for {url}: {str(e)[:100]}...")
        
        max_retries = self.config.get("content_fetch", {}).get("max_retries", 3)
        timeout = self.config.get("content_fetch", {}).get("timeout", 30)
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=timeout, headers=headers) as response:
                        if response.status == 200:
                            html_content = await response.text()
                            
                            soup = BeautifulSoup(html_content, 'html.parser')
                            
                            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                                tag.decompose()
                            
                            article_content = None
                            for selector in ['article', 'main', '.article-content', '.post-content', '#article', '#content', '.content']:
                                content = soup.select_one(selector)
                                if content and len(content.get_text(strip=True)) > 200:
                                    article_content = content
                                    break
                            
                            if not article_content:
                                article_content = soup.body
                            
                            if article_content:
                                # Convert to markdown
                                import markdownify
                                markdown_content = markdownify.markdownify(str(article_content))
                                
                                metadata["fetch_method"] = "requests_with_extraction"
                                metadata["content_size"] = len(markdown_content)
                                metadata["extraction_success"] = True
                                self.logger.info(f"Direct extraction successful for: {url}")
                                return markdown_content, metadata
                            else:
                                self.logger.warning(f"Failed to extract content from {url}")
                        else:
                            self.logger.warning(f"HTTP error {response.status} for {url}")
                            
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Error during attempt {attempt+1} for {url}: {str(e)[:100]}...")
                await asyncio.sleep(2)
        
        self.logger.error(f"All extraction methods failed for: {url}")
        return None, metadata
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def run(self, url: str, **kwargs) -> ToolResult[Dict[str, Any]]:
        try:
            self.logger.info(f"Parsing content from URL: {url}")
            
            content, metadata = await self.fetch_article_content(url)
            
            result = {
                "url": url,
                "content": content,
                "metadata": metadata,
                "success": content is not None
            }
            
            if not content:
                self.logger.warning(f"Failed to extract content from {url}")
            else:
                self.logger.info(f"Successfully extracted {metadata.get('content_size')} bytes from {url}")
            
            return ToolResult(success=content is not None, data=result)
            
        except Exception as e:
            self.logger.error(f"Error during content parsing: {str(e)}")
            return await self._handle_error(e)