from pydantic import BaseModel
from typing import Dict, List, Any, Optional

import asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import aiohttp
from langchain_community.utilities import SerpAPIWrapper

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


class SearchQuery(BaseModel):
    query: str
    engine: str = "google"
    location: str = "India"
    google_domain: str = "google.co.in"
    gl: str = "in"
    hl: str = "en"
    num: int = 15
    tbm: Optional[str] = "nws"
    safe: str = "off"


class SearchResult(BaseModel):
    title: str
    link: str
    snippet: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    category: Optional[str] = None
    is_quarterly_report: bool = False


class SearchError(Exception):
    """Base class for search-related errors."""
    pass


class SearchConnectionError(SearchError):
    """Error when connecting to search API."""
    pass


class SearchRateLimitError(SearchError):
    """Error when search API rate limit is exceeded."""
    pass


class SearchDataError(SearchError):
    """Error when parsing search data."""
    pass


class SearchTool(BaseTool):
    name = "search_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)

    async def _execute(self, query: str, **kwargs) -> ToolResult[List[SearchResult]]:
        return await self.run

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type(SearchConnectionError) | 
            retry_if_exception_type(SearchRateLimitError) |
            retry_if_exception_type(aiohttp.ClientError)
        )
    )
    async def run(self, query: str, **kwargs) -> ToolResult[List[SearchResult]]:
        try:
            search_params = {**self.config, **kwargs}
            search_query = SearchQuery(query=query, **search_params)
            
            self.logger.info(f"Executing search: {query}")
            
            params = search_query.model_dump()
            serp = SerpAPIWrapper(params=params)
            
            loop = asyncio.get_event_loop()
            try:
                raw_results = await loop.run_in_executor(None, lambda: serp.run(query))
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "too many requests" in error_msg:
                    raise SearchRateLimitError(f"Search API rate limit exceeded: {str(e)}")
                elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
                    raise SearchConnectionError(f"Connection error: {str(e)}")
                else:
                    raise SearchError(f"Search error: {str(e)}")
            
            parsed_results = await self._parse_results(raw_results, search_query.query)
            
            self.logger.info(f"Search returned {len(parsed_results)} results")
            
            return ToolResult(success=True, data=parsed_results)
        except SearchError as e:
            self.logger.error(f"Search error: {str(e)}")
            return await self._handle_error(e)
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return await self._handle_error(e)
    
    async def _parse_results(self, raw_results: Any, original_query: str) -> List[SearchResult]:
        results = []
        
        try:
            if isinstance(raw_results, str):
                import json
                try:
                    data = json.loads(raw_results)
                    if isinstance(data, list):
                        raw_results = data
                    elif isinstance(data, dict) and 'organic_results' in data:
                        raw_results = data['organic_results']
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON results: {str(e)}")
            
            if isinstance(raw_results, list):
                for i, item in enumerate(raw_results):
                    if isinstance(item, dict) and "title" in item and "link" in item:
                        title = item.get("title", "").strip()
                        result = SearchResult(
                            title=title,
                            link=item.get("link", "").strip(),
                            snippet=item.get("snippet", "").strip(),
                            source=item.get("source", "Unknown source").strip(),
                            date=item.get("date", "Unknown date").strip(),
                            category=item.get("category", "general"),
                            is_quarterly_report=self._is_quarterly_report(title, item.get("snippet", ""))
                        )
                        results.append(result)
            else:
                self.logger.warning(f"Unexpected results format: {type(raw_results)}")
                
        except Exception as e:
            self.logger.error(f"Error parsing results: {str(e)}")
            raise SearchDataError(f"Failed to parse search results: {str(e)}")
            
        return results
    
    def _is_quarterly_report(self, title: str, snippet: str = "") -> bool:
        title_lower = title.lower()
        snippet_lower = snippet.lower() if snippet else ""
        
        report_terms = [
            'quarterly report', 'q1 report', 'q2 report', 'q3 report', 'q4 report',
            'quarterly results', 'q1 results', 'q2 results', 'q3 results', 'q4 results',
            'quarterly earnings', 'annual report', 'annual results', 'financial results',
            'earnings report', 'quarterly financial', 'year-end results'
        ]
        
        for term in report_terms:
            if term in title_lower or term in snippet_lower:
                return True
        
        import re
        if (re.search(r'q[1-4]\s*20[0-9]{2}', title_lower) or 
            re.search(r'fy\s*20[0-9]{2}', title_lower) or
            re.search(r'q[1-4]\s*20[0-9]{2}', snippet_lower) or
            re.search(r'fy\s*20[0-9]{2}', snippet_lower)):
            return True
        
        if (re.search(r'report[s]?\s+\d+%', title_lower) or 
            re.search(r'revenue\s+of\s+[\$£€]', title_lower) or
            re.search(r'profit\s+of\s+[\$£€]', title_lower)):
            return True
        
        return False