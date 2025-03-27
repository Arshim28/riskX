from base.base_tools import BaseTool, ToolResult
from typing import Dict, List, Optional, Any, Tuple, Callable
from utils.logging import get_logger
from pydantic import BaseModel
import yaml
import aiohttp
from urllib.parse import quote
import json 
import brotli
import datetime
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

RETRY_LIMIT = 3
MULTIPLIER = 1
MIN_WAIT = 2
MAX_WAIT = 10

class NSEError(Exception):
    """Base class for NSE-related errors."""
    pass


class NSEConnectionError(NSEError):
    """Error when connecting to NSE API."""
    pass


class NSERateLimitError(NSEError):
    """Error when NSE API rate limit is exceeded."""
    pass


class NSEAuthenticationError(NSEError):
    """Error when authentication with NSE API fails."""
    pass


class NSEDataError(NSEError):
    """Error when parsing NSE data."""
    pass

class NSEToolConfig(BaseModel):
    base_url: str = "https://www.nseindia.com"
    refresh_interval: int = 25
    config_path: str = "assets/nse_config.yaml"
    headers_path: str = "assets/headers.yaml"
    cookie_path: str = "assets/cookies.yaml"
    schema_path: str = "assets/nse_schema.yaml"
    use_hardcoded_cookies: bool = False
    domain: str = "nseindia.com"
    company: str 
    symbol: str

class NSETool(BaseTool):
    name = "nse_tool"

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger(self.name)
        self.config = NSEToolConfig(**config)
        self.data_config = self._load_yaml(self.config.config_path)
        self.headers = self._load_yaml(self.config.headers_path)
        self.cookies = self._load_yaml(self.config.cookie_path) if self.config.use_hardcoded_cookies else None
        self.session = None
        self.fallback_referer = "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings"
        self.stream_processors = self._init_stream_processors()
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    
    def _log_error(self, error_message: str):
        try:
            self.logger.error(error_message)
            with open("errors.txt", "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {error_message}\n")
        except Exception as e:
            print(f"Error writing to error log: {e}")

    async def _create_new_session(self) -> aiohttp.ClientSession:
        session = aiohttp.ClientSession(headers=self.headers)
        
        if self.config.use_hardcoded_cookies:
            for name, value in self.cookies.items():
                session.cookie_jar.update_cookies({name: value}, self.config.base_url)
        else:
            await session.get(self.config.base_url)
            filings_url = f"{self.config.base_url}/companies-listing/corporate-filings-announcements"
            await session.get(filings_url)
            self.cookies = {cookie.key: cookie.value for cookie in session.cookie_jar}
            
        return session

    @retry(
    stop=stop_after_attempt(RETRY_LIMIT), 
    wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT),
    before_sleep=lambda retry_state: retry_state.outcome.exception() and print(f"Retrying _refresh_session due to {retry_state.outcome.exception().__class__.__name__}: {retry_state.outcome.exception()}, attempt {retry_state.attempt_number}"))
    async def _refresh_session(self, referer: str) -> None:
        self.logger.info(f"Updating referer: {referer}")
        
        if self.session is None:
            self.session = await self._create_new_session()
        
        headers = self.headers.copy()
        headers["Referer"] = referer
        
        try:
            async with self.session.get(referer, headers=headers, timeout=10) as response:
                pass
                
            if not self.config.use_hardcoded_cookies:
                self.cookies = {cookie.key: cookie.value for cookie in self.session.cookie_jar}
                
            cookies_dict = {cookie.key: cookie.value for cookie in self.session.cookie_jar}
            if 'nsit' not in cookies_dict and not self.config.use_hardcoded_cookies:
                raise Exception("Required nsit cookie not found")
        except Exception as e:
            self._log_error(f"Error refreshing session: {str(e)}")
            raise
    
    def get_available_data_streams(self) -> Dict[str, Dict[str, Any]]:
        active_streams = {}
        
        for key, config in self.data_config.items():
            if config.get('active', False):
                active_streams[key] = {
                    'params': config.get('params', {}),
                    'description': config.get('description', '')
                }
        
        return active_streams
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def make_request(self, url, referer):
        try: 
            await self._refresh_session(referer)
            
            async with self.session.get(url, headers=self.headers, timeout=60) as response:
                response.raise_for_status()
                
                content = await response.read()
                if not content:
                    self._log_error(f"Empty response received for URL: {url}")
                    return None

                if 'br' in response.headers.get('Content-Encoding', ''):
                    try:
                        decompressed_content = brotli.decompress(content)
                        json_text = decompressed_content.decode('utf-8')
                        return json.loads(json_text)
                    except Exception as e:
                        self._log_error(f"Error decompressing/decoding JSON: {str(e)} for URL: {url}")
                        return None
                else:
                    try:
                        text = await response.text()
                        return json.loads(text)
                    except json.JSONDecodeError as e:
                        self._log_error(f"Error decoding JSON: {str(e)} for URL: {url}")
                        return None
                
        except aiohttp.ClientError as e:
            self._log_error(f"Request error for URL {url}: {str(e)}")
            return None
        except Exception as e:
            self._log_error(f"Unexpected error for URL {url}: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def fetch_data_from_nse(self, stream: str, input_params: Dict[str, Any]) -> List:
        stream_config = self.data_config.get(stream)
        if not stream_config:
            raise ValueError(f"Stream not found: {stream}")
        
        params = stream_config.get('params', {}).copy() 
        if "issuer" in params:
            params["issuer"] = self.config.company
        if "symbol" in params:
            params["symbol"] = self.config.symbol
        params.update(input_params)

        url = self._construct_url(stream_config.get("endpoint"), params)
        print(url)
        result = await self.make_request(url, stream_config.get("referer", self.fallback_referer))
        print(result)
        if result:
            max_results = input_params.get("max_results", 20)
            
            if isinstance(result, list):
                data_list = result
                if max_results and len(data_list) > max_results:
                    data_list = data_list[:max_results]
                return data_list
            elif isinstance(result, dict):
                data_list = result.get("data", result)
                if isinstance(data_list, list):
                    if max_results and len(data_list) > max_results:
                        data_list = data_list[:max_results]
                    return data_list
                else:
                    return [result]
            return [result]
        else:
            self._log_error(f"Unable to retrieve data from: {url}")
            return []

    async def fetch_data_from_multiple_streams(self, stream_dict: Dict[str, Dict[str, Any]]) -> Dict[str, List]:
        resp_dict = {}
        tasks = []
        
        for key, config in stream_dict.items():
            if config.get("active", False):
                processor = self.stream_processors.get(key)
                if processor:
                    task = asyncio.create_task(
                        processor(config.get("input_params", {}), self._get_schema(key))
                    )
                    tasks.append((key, task))
                else:
                    task = asyncio.create_task(
                        self.fetch_data_from_nse(key, config.get("input_params", {}))
                    )
                    tasks.append((key, task))
        
        for key, task in tasks:
            try:
                resp_dict[key] = await task
            except Exception as e:
                self._log_error(f"Error fetching data for stream {key}: {str(e)}")
                resp_dict[key] = []
                
        return resp_dict

    def _construct_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        filtered_params = {k: v for k, v in params.items() if v is not None and v != ""}
        
        base_url = f"{self.config.base_url}/api/{endpoint}"
        
        if filtered_params:
            query_parts = []
            for key, value in filtered_params.items():
                encoded_value = quote(str(value))
                query_parts.append(f"{key}={encoded_value}")
            
            query_string = "&".join(query_parts)
            return f"{base_url}?{query_string}"
        
        return base_url
    
    async def close(self):
        if self.session:
            await self.session.close()

    def _filter_on_schema(self, data, schema):
        if not data or not isinstance(data, list):
            raise TypeError(f"Expected list recieved {type(data)}")
        return [{new_key: entry[old_key] for new_key, old_key in schema.items() if old_key in entry} for entry in data]
    
    def _get_schema(self, stream_name: str) -> Dict[str, str]:
        schema_data = self._load_yaml(self.config.schema_path)
        return schema_data.get(stream_name, {})
    
    async def _process_stream(self, stream: str, params: Dict[str, Any], schema: Dict[str, str]) -> List[Dict[str, Any]]:
        data = await self.fetch_data_from_nse(stream, params)
        filtered_data = self._filter_on_schema(data, schema)
        return filtered_data
    
    async def _process_announcements(self, params: Dict[str, Any], schema: Dict[str, str]) -> List[Dict[str, Any]]:
        return await self._process_stream("Announcements", params, schema)
    
    async def _process_ann_xbrl(self, params: Dict[str, Any], schema: Dict[str, str]) -> List[Dict[str, Any]]:
        data = await self.fetch_data_from_nse("AnnXBRL", params)
        filtered_data = self._filter_on_schema(data, schema)
        for entry in filtered_data:
            app_id = entry.get("appId")
            if app_id:
                entry["details"] = await self._get_announcement_details(app_id, params)
        return filtered_data
    
    async def _get_announcement_details(self, appId: str, params: Dict[str, Any]):
        try:
            data = await self.fetch_data_from_nse("AnnXBRLDetails", {"appId": appId, "type": params.get("type", "announcements")})
            return data[0] if data else {}
        except:
            self._log_error("Failed to get announcement details")
            return {}
    
    async def _process_annual_reports(self, params: Dict[str, Any], schema: Dict[str, str]) -> List[Dict[str, Any]]:
        return await self._process_stream("AnnualReports", params, schema)
    
    async def _process_esg_reports(self, params: Dict[str, Any], schema: Dict[str, str]) -> List[Dict[str, Any]]:
        return await self._process_stream("BussinessSustainabilitiyReport", params, schema)
    
    async def _process_board_meetings(self, params: Dict[str, Any], schema: Dict[str, str]) -> List[Dict[str, Any]]:
        return await self._process_stream("BoardMeetings", params, schema)
    
    async def _process_corporate_actions(self, params: Dict[str, Any], schema: Dict[str, str]) -> List[Dict[str, Any]]:
        return await self._process_stream("CorporateActions", params, schema)
    
    def _init_stream_processors(self) -> Dict[str, Callable]:
        return {
            "Announcements": self._process_announcements,
            "AnnXBRL": self._process_ann_xbrl,
            "AnnualReports": self._process_annual_reports,
            "BussinessSustainabilitiyReport": self._process_esg_reports,
            "BoardMeetings": self._process_board_meetings,
            "CorporateActions": self._process_corporate_actions,
        }
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def run(self, command: str, **kwargs) -> ToolResult:
        try:
            # Verify event loop state to catch issues early
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop in current thread, creating new one
                self.logger.warning("No event loop in current thread, creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Check if the loop is closed
            if loop.is_closed():
                self.logger.error("Event loop is closed, creating new loop")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            if command == "fetch_data":
                stream = kwargs.get("stream")
                input_params = kwargs.get("input_params", {})
                schema = kwargs.get("schema")
                
                if not stream:
                    return ToolResult(success=False, error="Stream parameter is required for fetch_data command")
                
                processor = self.stream_processors.get(stream)
                if processor and not schema:
                    schema = self._get_schema(stream)
                    data = await processor(input_params, schema)
                else:
                    data = await self.fetch_data_from_nse(stream, input_params)
                    if schema:
                        data = self._filter_on_schema(data, schema)
                    
                return ToolResult(success=True, data=data)
            
            elif command == "fetch_multiple":
                stream_dict = kwargs.get("stream_dict")
                
                if not stream_dict:
                    return ToolResult(success=False, error="Stream dictionary is required for fetch_multiple command")
                
                data = await self.fetch_data_from_multiple_streams(stream_dict)
                return ToolResult(success=True, data=data)
            
            elif command == "get_streams":
                streams = self.get_available_data_streams()
                return ToolResult(success=True, data=streams)
                
            elif command == "get_announcements":
                params = kwargs.get("params", {})
                schema = kwargs.get("schema")
                
                if not schema:
                    schema = self._get_schema("Announcements")
                
                data = await self._process_announcements(params, schema)
                return ToolResult(success=True, data=data)
                
            elif command == "get_announcements_xbrl":
                params = kwargs.get("params", {})
                schema = kwargs.get("schema")
                
                if not schema:
                    schema = self._get_schema("AnnXBRL")
                
                data = await self._process_ann_xbrl(params, schema)
                return ToolResult(success=True, data=data)
            
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
            
        except Exception as e:
            error_message = f"Error in NSE tool: {str(e)}"
            self.logger.error(error_message)
            return ToolResult(success=False, error=error_message)
    
    async def _execute(self, **kwargs) -> ToolResult:
        pass

if __name__ == "__main__":
    config = {
        "company": "Reliance Industries Limited",
        "symbol": "RELIANCE",
    } 
    date_params = {
        "from_date": "01-09-2024",
        "to_date": "23-03-2025"
    }
    date_params_2 = {
        "from_date": "24-02-2025",
        "to_date": "24-03-2025"
    }

    async def test_stream(tool, stream_name, params=None):
        try:
            params = params or {}
            print(f"\n=== Testing {stream_name} with params: {params} ===")
            result = await tool.fetch_data_from_nse(stream_name, params)
            if result:
                print(f"{stream_name} Results: {len(result)} items")
                if len(result) > 0:
                    print(f"First item keys: {list(result[0].keys())[:5]}")
            else:
                print(f"{stream_name} returned no results")
            return result
        except Exception as e:
            print(f"Error testing {stream_name}: {str(e)}")
            return []

    async def test():
        tool = None
        try:
            tool = NSETool(config)
            
            # Test individual streams with appropriate parameters
            streams_to_test = [
                ("CreditRating", {}),
                ("Announcements", date_params_2),
                ("BoardMeetings", date_params),
                ("CorporateActions", date_params),
                ("BussinessSustainabilitiyReport", {}),
                ("AnnualReports", {}),
                ("AnnXBRL", date_params)
            ]
            
            for stream_name, params in streams_to_test:
                await test_stream(tool, stream_name, params)
            
            # Test schema processing
            print("\n=== Testing schema processing for Announcements ===")
            try:
                schema = tool._get_schema("Announcements")
                print(f"Schema keys: {schema.keys()}")
                filtered = await tool._process_announcements(date_params_2, schema)
                print(f"Filtered Results: {len(filtered)} items")
                if filtered and len(filtered) > 0:
                    print(f"Filtered keys: {filtered[0].keys()}")
            except Exception as e:
                print(f"Error during schema processing: {str(e)}")
            
            # Test multiple streams with safe error handling
            print("\n=== Testing multiple streams fetch ===")
            try:
                stream_dict = {
                    "Announcements": {
                        "active": True,
                        "input_params": date_params_2
                    },
                    "BoardMeetings": {
                        "active": True,
                        "input_params": date_params
                    },
                    "CorporateActions": {
                        "active": True,
                        "input_params": date_params
                    }
                }
                
                multi_results = await tool.fetch_data_from_multiple_streams(stream_dict)
                print(f"Multiple streams results: {len(multi_results)} streams")
                for stream_name, stream_data in multi_results.items():
                    print(f"  {stream_name}: {len(stream_data)} items")
            except Exception as e:
                print(f"Error in multiple streams test: {str(e)}")
                
        except Exception as e:
            print(f"Global error in tests: {str(e)}")
        finally:
            if tool:
                await tool.close()
                print("\nConnection closed properly")
    
    asyncio.run(test())