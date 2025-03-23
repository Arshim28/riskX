import os
import json
import yaml
import requests
import brotli
import asyncio
import aiohttp
import aiofiles
import tempfile
import zipfile
from typing import Dict, List, Any, Optional, Union, Tuple
import datetime
import time
import random
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


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


class NSEToolConfig(Dict):
    """Configuration class for NSETool"""
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        
        # Set default values if not provided
        self.setdefault("base_url", "https://www.nseindia.com")
        self.setdefault("requests_before_refresh", 20)
        self.setdefault("config_path", "assets/nse_config.yaml")
        self.setdefault("headers_path", "assets/headers.yaml")
        self.setdefault("cookie_retry_attempts", 3)
        self.setdefault("retry_limit", 3)
        self.setdefault("multiplier", 1)
        self.setdefault("min_wait", 2)
        self.setdefault("max_wait", 10)


class NSETool(BaseTool):
    name = "nse_tool"
    
    def __init__(self, config: Dict[str, Any], company: str = "", symbol: str = ""):
        self.config_dict = NSEToolConfig(config)
        self.logger = get_logger(self.name)
        self.base_url = self.config_dict["base_url"]
        self.domain = "nseindia.com"
        self.request_count = 0
        self.requests_before_refresh = self.config_dict["requests_before_refresh"]
        self.cookie_retry_attempts = self.config_dict.get("cookie_retry_attempts", 3)
        
        self.company = company
        self.symbol = symbol
        
        config_path = self.config_dict["config_path"]
        headers_path = self.config_dict["headers_path"]
        
        # Load configurations
        self.endpoint_config = self._load_yaml_config(config_path)
        self.headers = self._load_headers(headers_path)
        
        # Initialize session with fresh cookies
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.cookies = self._get_fresh_cookies()
        
        # Add cookies to session
        for name, value in self.cookies.items():
            self.session.cookies.set(name, value, domain=self.domain)
    
    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(config_path):
                self.logger.warning(f"Config file not found: {config_path}")
                return {}
                
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return config_dict
        except Exception as e:
            error_msg = f"Error loading config: {str(e)}"
            self.logger.error(error_msg)
            raise NSEError(error_msg)
    
    def _load_headers(self, headers_path: str) -> Dict[str, str]:
        try:
            if os.path.exists(headers_path):
                with open(headers_path, "r") as f:
                    headers = yaml.safe_load(f)
                return headers
            else:
                self.logger.warning(f"Headers file not found: {headers_path}")
                # Provide default headers if file not found
                return {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Pragma": "no-cache",
                    "Cache-Control": "no-cache"
                }
        except Exception as e:
            error_msg = f"Error loading headers: {str(e)}"
            self.logger.error(error_msg)
            return {}
    
    def _log_error(self, error_message: str):
        try:
            self.logger.error(error_message)
            with open("errors.txt", "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {error_message}\n")
        except Exception as e:
            print(f"Error writing to error log: {e}")
    
    def _validate_cookies(self, cookies: Dict[str, str]) -> bool:
        """Verify if cookies have required tokens."""
        required_cookies = ['nsit', 'nseappid']
        is_valid = all(cookie in cookies for cookie in required_cookies)
        if not is_valid:
            self.logger.warning(f"Invalid cookies: missing required cookies {required_cookies}")
            return False
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type(NSEConnectionError) | 
            retry_if_exception_type(requests.RequestException)
        )
    )
    def _get_fresh_cookies(self) -> Dict[str, str]:
        """Get fresh cookies from NSE India website."""
        cookies = {}
        session = requests.Session()
        session.headers.update(self.headers)
        
        for attempt in range(self.cookie_retry_attempts):
            try:
                self.logger.info(f"Getting fresh cookies (attempt {attempt+1}/{self.cookie_retry_attempts})")
                
                # Visit main page first to get initial cookies
                main_page_url = "https://www.nseindia.com/"
                response = session.get(main_page_url, timeout=15)
                response.raise_for_status()
                
                # Then visit a relevant page to get additional cookies
                referer_url = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
                response = session.get(referer_url, timeout=15)
                response.raise_for_status()
                
                # Convert cookies to dictionary
                cookies = {cookie.name: cookie.value for cookie in session.cookies}
                
                # Validate essential cookies
                if self._validate_cookies(cookies):
                    self.logger.info("Successfully obtained fresh cookies")
                    return cookies
                else:
                    self.logger.warning(f"Attempt {attempt+1}: Missing required cookies")
                    time.sleep(2)
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Connection error on attempt {attempt+1}: {str(e)}")
                time.sleep(2)
            except Exception as e:
                self.logger.warning(f"Unexpected error on attempt {attempt+1}: {str(e)}")
                time.sleep(2)
        
        # If all attempts fail, raise an error
        raise NSEAuthenticationError("Failed to get valid cookies after multiple attempts")
    
    def _refresh_session_if_needed(self) -> None:
        """Refresh session cookies if the request count exceeds the threshold."""
        self.request_count += 1
        if self.request_count >= self.requests_before_refresh:
            self.logger.info("Refreshing cookies due to request count threshold")
            try:
                # Get fresh cookies
                new_cookies = self._get_fresh_cookies()
                
                # Update session cookies
                self.session.cookies.clear()
                for name, value in new_cookies.items():
                    self.session.cookies.set(name, value, domain=self.domain)
                
                self.cookies = new_cookies
                self.request_count = 0
            except Exception as e:
                self.logger.error(f"Error refreshing cookies: {str(e)}")
                # Continue with existing cookies in case of error
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type(NSEConnectionError) | 
            retry_if_exception_type(NSERateLimitError)
        )
    )
    async def _make_async_request(self, url: str, referer_url: str) -> Optional[Dict[str, Any]]:
        """Make an async request to NSE API with proper error handling."""
        try:
            self._refresh_session_if_needed()
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Update the session's referer
            self.session.headers.update({"Referer": referer_url})
            
            # Create new async session with cookies and headers
            async with aiohttp.ClientSession(cookies=self.cookies) as session:
                headers = dict(self.session.headers)
                
                async with session.get(url, headers=headers, timeout=60) as response:
                    if response.status == 429:
                        raise NSERateLimitError(f"Rate limit exceeded for URL: {url}")
                    if response.status == 401 or response.status == 403:
                        raise NSEAuthenticationError(f"Authentication error for URL: {url}")
                    if response.status >= 500:
                        raise NSEConnectionError(f"Server error {response.status} for URL: {url}")
                    
                    response.raise_for_status()
                    
                    content = await response.content.read()
                    if not content:
                        raise NSEDataError(f"Empty response received for URL: {url}")
                    
                    if 'br' in response.headers.get('Content-Encoding', ''):
                        try:
                            decompressed_content = brotli.decompress(content)
                            json_text = decompressed_content.decode('utf-8')
                            return json.loads(json_text)
                        except Exception as e:
                            raise NSEDataError(f"Error decompressing/decoding JSON: {str(e)} for URL: {url}")
                    else:
                        try:
                            return json.loads(content.decode('utf-8'))
                        except json.JSONDecodeError as e:
                            # Check if we received HTML instead of JSON
                            content_str = content.decode('utf-8', errors='ignore')
                            if '<html' in content_str.lower():
                                if 'login' in content_str.lower() or 'captcha' in content_str.lower():
                                    raise NSEAuthenticationError("Received login page instead of JSON - session may have expired")
                            raise NSEDataError(f"Error decoding JSON: {str(e)} for URL: {url}")
        except aiohttp.ClientError as e:
            raise NSEConnectionError(f"Connection error for URL {url}: {str(e)}")
        except NSEError:
            # Re-raise NSE-specific errors
            raise
        except Exception as e:
            raise NSEError(f"Request error for URL {url}: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type(NSEConnectionError) | 
            retry_if_exception_type(NSERateLimitError)
        )
    )
    async def download_file(self, url: str, file_type: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[bytes]]:
        """Download a file from NSE with improved error handling."""
        try:
            if not file_type:
                parsed_url = urlparse(url)
                path = parsed_url.path
                file_type = path.split('.')[-1].lower() if '.' in path else "unknown"
            
            self._refresh_session_if_needed()
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Create new async session with cookies and headers
            async with aiohttp.ClientSession(cookies=self.cookies) as session:
                headers = dict(self.session.headers)
                headers["Referer"] = f"{self.base_url}/companies-listing/corporate-filings"
                
                async with session.get(url, headers=headers, timeout=60) as response:
                    if response.status == 429:
                        raise NSERateLimitError(f"Rate limit exceeded for URL: {url}")
                    if response.status == 401 or response.status == 403:
                        raise NSEAuthenticationError(f"Authentication error for URL: {url}")
                    if response.status >= 500:
                        raise NSEConnectionError(f"Server error {response.status} for URL: {url}")
                    
                    response.raise_for_status()
                    content = await response.read()
                    
                    filename = None
                    if 'Content-Disposition' in response.headers:
                        content_disp = response.headers['Content-Disposition']
                        if 'filename=' in content_disp:
                            filename = content_disp.split('filename=')[1].strip('"\'')
                    
                    if not filename:
                        path = urlparse(url).path
                        filename = os.path.basename(path)
                        if not filename:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                            filename = f"nse_download_{timestamp}.{file_type}"
                    
                    temp_path = os.path.join(tempfile.gettempdir(), filename)
                    
                    async with aiofiles.open(temp_path, 'wb') as f:
                        await f.write(content)
                    
                    if file_type.lower() == 'zip':
                        try:
                            extract_dir = tempfile.mkdtemp()
                            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                            
                            os.remove(temp_path)
                            return True, extract_dir, None
                        except Exception as e:
                            raise NSEDataError(f"Error extracting ZIP file: {str(e)}")
                    
                    return True, temp_path, content
        except aiohttp.ClientError as e:
            raise NSEConnectionError(f"Connection error downloading file: {str(e)}")
        except NSEError:
            # Re-raise NSE-specific errors
            raise
        except Exception as e:
            raise NSEError(f"Error downloading file: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(NSEConnectionError)
    )
    async def detect_file_type(self, url: str) -> str:
        """Detect file type from URL or headers with improved error handling."""
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            if '.' in path:
                ext = path.split('.')[-1].lower()
                if ext in ['pdf', 'xml', 'zip', 'xbrl']:
                    return ext
            
            self._refresh_session_if_needed()
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.3, 1.0))
            
            # Create new async session with cookies and headers
            async with aiohttp.ClientSession(cookies=self.cookies) as session:
                headers = dict(self.session.headers)
                headers["Referer"] = f"{self.base_url}/companies-listing/corporate-filings"
                
                async with session.head(url, headers=headers, timeout=10) as response:
                    response.raise_for_status()
                    
                    if 'Content-Type' in response.headers:
                        content_type = response.headers['Content-Type']
                        if 'application/pdf' in content_type:
                            return 'pdf'
                        elif 'application/xml' in content_type or 'text/xml' in content_type:
                            return 'xml'
                        elif 'application/zip' in content_type:
                            return 'zip'
                        
                    if 'Content-Disposition' in response.headers:
                        content_disp = response.headers['Content-Disposition']
                        if 'filename=' in content_disp:
                            filename = content_disp.split('filename=')[1].strip('"\'')
                            if '.' in filename:
                                return filename.split('.')[-1].lower()
            
            return 'bin'
        except aiohttp.ClientError as e:
            raise NSEConnectionError(f"Connection error detecting file type: {str(e)}")
        except Exception as e:
            raise NSEError(f"Error detecting file type: {str(e)}")
    
    def get_available_streams(self) -> List[Dict[str, Any]]:
        """Get available data streams from configuration."""
        streams = []
        
        for key, config in self.endpoint_config.items():
            if isinstance(config, dict) and config.get("active", False):
                stream_info = {
                    "key": key,
                    "endpoint": config.get("endpoint", ""),
                    "description": config.get("description", ""),
                    "params": config.get("params", {})
                }
                streams.append(stream_info)
                
        return streams
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type(NSEConnectionError) | 
            retry_if_exception_type(NSERateLimitError)
        )
    )
    async def fetch_data(self, stream_key: str, **kwargs) -> List[Dict[str, Any]]:
        """Fetch data from a specific stream with improved error handling."""
        try:
            if stream_key not in self.endpoint_config:
                raise NSEError(f"Stream key not found: {stream_key}")
                
            endpoint_config = self.endpoint_config[stream_key]
            
            if not isinstance(endpoint_config, dict) or not endpoint_config.get("active", False):
                raise NSEError(f"Stream is inactive or invalid: {stream_key}")
                
            endpoint = endpoint_config.get("endpoint", "")
            
            if not endpoint:
                raise NSEError(f"Endpoint URL not found for: {stream_key}")
                
            referer_suffix = endpoint_config.get("referer", "corporate-filings-announcements")
            if referer_suffix.startswith("http"):
                referer_url = referer_suffix
            else:
                referer_url = f"{self.base_url}/companies-listing/{referer_suffix}"
            
            params = {}
            config_params = endpoint_config.get("params", {})
            
            if self.company and "issuer" in config_params:
                params["issuer"] = self.company
            
            if self.symbol and "symbol" in config_params:
                params["symbol"] = self.symbol
            
            for key, value in kwargs.items():
                if key in config_params and value is not None and value != "":
                    params[key] = value
            
            url_extension = endpoint + "?" + "&".join([f"{key}={value}" for key, value in params.items() if value])
            url = self.base_url + "/api/" + url_extension
            
            # Try up to 3 times with increasing delays
            for attempt in range(3):
                if attempt > 0:
                    # Refresh cookies between attempts
                    try:
                        new_cookies = self._get_fresh_cookies()
                        self.cookies = new_cookies
                        self.session.cookies.clear()
                        for name, value in new_cookies.items():
                            self.session.cookies.set(name, value, domain=self.domain)
                    except Exception as e:
                        self.logger.warning(f"Failed to refresh cookies: {str(e)}")
                    
                    # Increase wait time with each attempt
                    await asyncio.sleep(2 * attempt)
                
                result = await self._make_async_request(url, referer_url)
                
                if result:
                    data_list = result.get("data", result)
                    
                    if isinstance(data_list, list):
                        max_results = kwargs.get("max_results", 20)
                        if max_results and len(data_list) > max_results:
                            data_list = data_list[:max_results]
                        return data_list
                    else:
                        self.logger.warning(f"Invalid data format for {stream_key}: not a list")
                        continue  # Try again if format is invalid
            
            # If we get here, all attempts failed
            raise NSEDataError(f"Failed to fetch valid data after multiple attempts for stream: {stream_key}")
            
        except NSEError:
            # Re-raise NSE-specific errors
            raise
        except Exception as e:
            raise NSEError(f"Error fetching data for {stream_key}: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type(NSEConnectionError) | 
            retry_if_exception_type(NSERateLimitError)
        )
    )
    async def fetch_multiple(self, actions_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Fetch data from multiple streams with improved error handling."""
        results = {}
        errors = {}
        
        for stream_key, params in actions_params.items():
            try:
                if stream_key not in self.endpoint_config:
                    errors[stream_key] = f"Unknown stream: {stream_key}"
                    continue
                    
                if not isinstance(self.endpoint_config[stream_key], dict) or not self.endpoint_config[stream_key].get("active", False):
                    errors[stream_key] = f"Stream is inactive: {stream_key}"
                    continue
                    
                data = await self.fetch_data(stream_key, **params)
                results[stream_key] = data
                
                # Add a small delay between requests
                await asyncio.sleep(random.uniform(0.5, 2.0))
                
            except NSEError as e:
                self.logger.error(f"Error fetching {stream_key}: {str(e)}")
                errors[stream_key] = str(e)
        
        # Include errors in the result if any
        if errors:
            results["_errors"] = errors
            
        return results
    async def _execute(self, command: str, **kwargs) -> ToolResult:
        return await self.run(command, **kwargs)
        
    async def run(self, command: str, **kwargs) -> ToolResult:
        """Execute a command with standardized error handling."""
        try:
            if command in self.endpoint_config and isinstance(self.endpoint_config[command], dict) and self.endpoint_config[command].get("active", False):
                data = await self.fetch_data(command, **kwargs)
                return ToolResult(success=True, data=data)
                    
            elif command == "fetch_multiple":
                actions_params = kwargs.get("actions_params", {})
                if not actions_params:
                    return ToolResult(success=False, error="actions_params dictionary is required")
                    
                data = await self.fetch_multiple(actions_params)
                return ToolResult(success=True, data=data)
                    
            elif command == "download_file":
                url = kwargs.get("url")
                file_type = kwargs.get("file_type")
                
                if not url:
                    return ToolResult(success=False, error="URL is required for download_file command")
                    
                success, file_path, content = await self.download_file(url, file_type)
                
                if not success:
                    return ToolResult(success=False, error="Failed to download file")
                    
                return ToolResult(success=True, data={"file_path": file_path, "file_type": file_type})
                    
            elif command == "get_streams":
                streams = self.get_available_streams()
                return ToolResult(success=True, data=streams)
                    
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
                    
        except NSEConnectionError as e:
            error_message = f"Connection error: {str(e)}"
            self.logger.error(error_message)
            return ToolResult(success=False, error=error_message)
        except NSERateLimitError as e:
            error_message = f"Rate limit exceeded: {str(e)}"
            self.logger.error(error_message)
            return ToolResult(success=False, error=error_message)
        except NSEAuthenticationError as e:
            error_message = f"Authentication error: {str(e)}"
            self.logger.error(error_message)
            return ToolResult(success=False, error=error_message)
        except NSEDataError as e:
            error_message = f"Data error: {str(e)}"
            self.logger.error(error_message)
            return ToolResult(success=False, error=error_message)
        except NSEError as e:
            error_message = f"NSE error: {str(e)}"
            self.logger.error(error_message)
            return ToolResult(success=False, error=error_message)
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            self.logger.error(error_message)
            return await self._handle_error(e)