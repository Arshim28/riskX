from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import os
import asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import aiohttp
import requests  # Added for direct HTTP requests
import json

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
        return await self.run(query, **kwargs)

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
            # Debug statements to check API key
            print(f"DEBUG SERPAPI: Config keys: {list(self.config.keys())}")
            print(f"DEBUG SERPAPI: API key present: {'api_key' in self.config}")
            print(f"DEBUG SERPAPI: API key first few chars: {self.config.get('api_key', 'MISSING')[:5]}..." if self.config.get('api_key') else "MISSING")
            
            # Prepare parameters for direct SerpAPI request
            params = search_query.model_dump()
            params['api_key'] = self.config.get('api_key')  # Add API key to params
            params['q'] = query  # Ensure query parameter is set correctly
            
            print(f"DEBUG SERPAPI: Making direct HTTP request to SerpAPI with params: {list(params.keys())}")

            try:
                # Make direct HTTP request to SerpAPI
                loop = asyncio.get_event_loop()
                raw_results = await loop.run_in_executor(
                    None, 
                    lambda: self._make_serpapi_request(params)
                )
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
            return ToolResult(success=False, error=str(e), data=[])
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return ToolResult(success=False, error=str(e), data=[])
    
    def _make_serpapi_request(self, params: Dict[str, Any]) -> Any:
        """Make a direct HTTP request to SerpAPI."""
        try:
            print(f"DEBUG SERPAPI: Sending request to SerpAPI with API key: {params.get('api_key', 'MISSING')[:5]}...")
            response = requests.get("https://serpapi.com/search", params=params)
            
            if response.status_code == 200:
                print(f"DEBUG SERPAPI: Got successful response from SerpAPI")
                return response.json()
            else:
                error_msg = f"SerpAPI error: {response.status_code} - {response.text}"
                print(f"DEBUG SERPAPI: {error_msg}")
                if "Invalid API key" in response.text:
                    raise SearchError(f"Invalid SerpAPI key. Please check your configuration.")
                else:
                    raise SearchError(f"SerpAPI request failed: {response.text}")
        except requests.RequestException as e:
            raise SearchConnectionError(f"Error connecting to SerpAPI: {str(e)}")
    
    async def _parse_results(self, raw_results: Any, original_query: str) -> List[SearchResult]:
        results = []
        
        try:
            # Check for API error first
            if isinstance(raw_results, dict):
                # Check for error in search_metadata
                if 'search_metadata' in raw_results:
                    metadata = raw_results.get('search_metadata', {})
                    status = metadata.get('status')
                    
                    if status == 'Error':
                        error_message = raw_results.get('error', 'Unknown error')
                        raise SearchDataError(f"SerpAPI search failed: {error_message}")
                    
                    print(f"DEBUG SERPAPI: Search status: {status}, Search ID: {metadata.get('id')}")
                
                # Check for direct error field
                if 'error' in raw_results:
                    raise SearchDataError(f"SerpAPI returned an error: {raw_results['error']}")
                
                # Process organic_results if available
                if 'organic_results' in raw_results:
                    print(f"DEBUG SERPAPI: Found {len(raw_results['organic_results'])} organic results")
                    for item in raw_results['organic_results']:
                        title = item.get("title", "").strip()
                        source = item.get("source", "")
                        date = item.get("date", "Unknown date")
                        
                        # If source is not in item, try to extract it from displayed_link
                        if not source and "displayed_link" in item:
                            # Extract domain from displayed link
                            displayed_link = item.get("displayed_link", "")
                            if displayed_link:
                                # Simple extraction, might need refinement
                                source = displayed_link.split('/')[0] if '/' in displayed_link else displayed_link
                        
                        result = SearchResult(
                            title=title,
                            link=item.get("link", "").strip(),
                            snippet=item.get("snippet", "").strip(),
                            source=source.strip() if source else "Unknown source",
                            date=date.strip() if date else "Unknown date",
                            category="general",
                            is_quarterly_report=self._is_quarterly_report(title, item.get("snippet", ""))
                        )
                        results.append(result)
                
                # Process news_results if available (for news searches)
                if 'news_results' in raw_results:
                    news_data = raw_results['news_results']
                    # Handle different structures
                    if isinstance(news_data, dict) and 'news_results' in news_data:
                        news_items = news_data['news_results']
                    elif isinstance(news_data, list):
                        news_items = news_data
                    else:
                        news_items = []
                    
                    print(f"DEBUG SERPAPI: Found {len(news_items)} news results")
                    for item in news_items:
                        title = item.get("title", "").strip()
                        result = SearchResult(
                            title=title,
                            link=item.get("link", "").strip(),
                            snippet=item.get("snippet", "").strip(),
                            source=item.get("source", "Unknown source").strip(),
                            date=item.get("date", "Unknown date").strip(),
                            category="news",
                            is_quarterly_report=self._is_quarterly_report(title, item.get("snippet", ""))
                        )
                        results.append(result)
                        
                # If neither organic nor news results were found but no error was reported
                if not results and 'error' not in raw_results:
                    print(f"DEBUG SERPAPI: No organic or news results found in response")
                    self.logger.warning(f"No results found in SerpAPI response. Response keys: {list(raw_results.keys())}")
            else:
                print(f"DEBUG SERPAPI: Unexpected response type: {type(raw_results)}")
                self.logger.warning(f"Unexpected results format: {type(raw_results)}")
                
        except SearchDataError as e:
            # Re-raise SearchDataError to be handled by the caller
            raise
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