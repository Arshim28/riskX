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
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


# Default retry parameters
RETRY_LIMIT = 3
MULTIPLIER = 1
MIN_WAIT = 2
MAX_WAIT = 10


class NSEToolConfig(Dict):
    """Configuration class for NSETool"""
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        
        # Set default values if not provided
        self.setdefault("base_url", "https://www.nseindia.com")
        self.setdefault("requests_before_refresh", 20)
        self.setdefault("config_path", "assets/nse_config.yaml")
        self.setdefault("headers_path", "assets/headers.yaml")
        self.setdefault("cookies_path", "assets/cookies.yaml")
        self.setdefault("retry_limit", RETRY_LIMIT)
        self.setdefault("multiplier", MULTIPLIER)
        self.setdefault("min_wait", MIN_WAIT)
        self.setdefault("max_wait", MAX_WAIT)
        self.setdefault("use_hardcoded_cookies", True)  # Default to using hardcoded cookies


class NSETool(BaseTool):
    name = "nse_tool"
    
    def __init__(self, config: Dict[str, Any], company: str = "", symbol: str = ""):
        self.config_dict = NSEToolConfig(config)
        self.logger = get_logger(self.name)
        self.base_url = self.config_dict["base_url"]
        self.domain = "nseindia.com"
        self.request_count = 0
        self.requests_before_refresh = self.config_dict["requests_before_refresh"]
        self.use_hardcoded_cookies = self.config_dict.get("use_hardcoded_cookies", False)
        
        self.company = company
        self.symbol = symbol
        
        config_path = self.config_dict["config_path"]
        headers_path = self.config_dict["headers_path"]
        
        # Update retry parameters based on config
        self._update_retry_params()
        
        self.config_dict = self._load_yaml_config(config_path)
        self.headers = self._load_headers(headers_path)
        
        # Prepare session with cookies
        self.session = self._get_fresh_cookies()
        self.cookies = {cookie.name: cookie.value for cookie in self.session.cookies}
    
    def _update_retry_params(self):
        """Update global retry parameters from config"""
        global RETRY_LIMIT, MULTIPLIER, MIN_WAIT, MAX_WAIT
        RETRY_LIMIT = self.config_dict.get("retry_limit", RETRY_LIMIT)
        MULTIPLIER = self.config_dict.get("multiplier", MULTIPLIER)
        MIN_WAIT = self.config_dict.get("min_wait", MIN_WAIT)
        MAX_WAIT = self.config_dict.get("max_wait", MAX_WAIT)
    
    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return config_dict
        except Exception as e:
            self._log_error(f"Error loading config: {str(e)}")
            raise
    
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
                    "Accept-Encoding": "gzip, deflate, br, zstd",
                    "Connection": "keep-alive",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Pragma": "no-cache",
                    "Cache-Control": "no-cache"
                }
        except Exception as e:
            self._log_error(f"Error loading headers: {str(e)}")
            return {}
    
    def _get_hardcoded_cookies(self) -> Dict[str, str]:
        """Return hardcoded cookies for NSE India site"""
        return {
            "_ga": "GA1.1.1579713621.1719746123", 
            "_ga_E0LYHCLJY3": "GS1.1.1742679997.1.0.1742680004.0.0.0", 
            "nsit": "yUBOLwMh-t6f0dFy41LLe8sx", 
            "AKA_A2": "A", 
            "bm_mi": "F8D1610FCF3E761629643C11DF2FAEE3~YAAQ1jSVtKurY56VAQAALSBPwBvl57RZHKzP2A6qrpRc08nforypQ256OEAa8n1NfWQ2FZ6xvqHWDIUzUcoMWY5R1b4FZ2mKaCasxK8q6zRB9L4jqRBgdoD1oW/98zn//4t8e4ERYxmLgWu3ZkG1XKwOSFdRFRsgqHJNa00+gZmCs6+siPwn4Tm++GQc0ZX8Wcd2vdcIEZWzkEep65urDpjWQTd3fP6NgTX1oX4AHy/qJGYO8xuOMpC2EdUGNGNeARPGowm5mKA4MrtKXpExfWZWmLKxHg7p7q38Oy4HnvZmXwikCwA8OiUQPvzvK5gBZ3ujpoiYGm5pbF/2Ek0WzTAR+66KlnUW6F52sGQNFa05U0I9DsbkQyBoAoG/+xVU~1", 
            "_abck": "AA3410E049A1D067C5938EDFAFDCB03B~0~YAAQ1jSVtLWrY56VAQAAJidPwA09BUzz0O7ILXOKCIablMXFJbA0/tKHkzZwqZU/rV25ZdWBG+Q6LL7x6opeUYjNcvukffMehQuW19OProKXvUg4uGTBpyPQTnA1t6leBYPI3mhO8E1golbC1AG/2gvgJFVlrrhLMNEtCUqD300Osr29OlwzMndGRLPEJiVOD6GByiiYB4vEjK1uZVijytCLqW69TYnl86wXL3I0BWXvvRjzjyzo7zNfFD8nyg3Jch25R9Yqxr1pZechB8dg3SGRp24ObiM9zsaY6I4b1UnDUy1N8lnBvxV9MngFGiB0/eKz49Jq/gf+V0lXLR80750TLqz1OYoLvzlZA85KBuLO7Ojfgv1uj4XQcSgR1VzeT1hVbH4LI4quEq4jBjhhkeori4gscGuXOUNwgwB0JsZHNxUvuWv/51IclvpIbZVJ9PzRThvhlWWZfKrvf0yiAq0crO07VRjLoBKWz4mbzzWlO3uxBJogq6OCBHBISAdDLLqoWAlYt/gaICdn63Y+oovEBfp4u2Mh4lb1g3Z+gpEfcK+XmVcrtk2XG91vuLRvoRp4HruTTeDhyGGZDSz3Nrt1He35ZiIE9s0RjA==~-1~-1~-1", 
            "ak_bmsc": "5B896744328CAAC090786D6BDAF81A89~000000000000000000000000000000~YAAQ1jSVtMerY56VAQAAKi1PwBuktMF8yNfA4Ckq+CzozOWYeTpPOHJy3xUiD4PaDoH8dcQfjGDXsjTPm8kqHVUzPWZz1ev2ZE78Xy1S5cimZrkJwr/tLXwEycK//lOzqD3qx2+kwGTCM75GR/GGrAXGQjx+xKQuHIiw0xZ3S1XHQM4cMnxOT6osrtrkWAN1atFuHGhbeJyE9ICn+FbOIQFOmeQFAT9Oph6DNe0l93y+EA9vVtPZSkRHYNscxcPM8pIQtXJBC/bdoWcaLtTHpYm6C7tDIWUdh1vC0BSQhbuA8PkN8OfsNw3TyaEAHqpOil4U7J6Oy71p6RLOcYziN8gp7hAmIY8Gr/I2AJLDOgRTDTXeqU/aykmDfhgGG5VFA8NCXhU9S38BZz1Zj0czHAsJ77/JzR9T+5wHR96CiFTS3VGkhbt4uy4i4siE2skN7IiG1T/sGSoGlO0ai1HRKo+ObsNRIiDLKruy8JMwz/AkKwZBrhh0sA/a07knGm/p9cnNn+CxrdwvLwiSLGHjzqyz24ornlw75vKbezx1hl9F", 
            "nseappid": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcGkubnNlIiwiYXVkIjoiYXBpLm5zZSIsImlhdCI6MTc0MjY4OTQ1MCwiZXhwIjoxNzQyNjk2NjUwfQ.2gYGSy0Brw-atRx7XngSEByNWuSCNG7hKTTMifgGShY", 
            "bm_sz": "4D388B70A77EC639AAEEF24673DFCF13~YAAQ1TSVtHXsk62VAQAA4xhZwBs2xcI0LUa9+v5QjweCMj2MUryEtx6k7ZYzzGQa76PZEaq8CqbAxwNMKuBStXBmbOY58AaADdrLWpwXwHcMXsit18bULCMaSDwK+6x0w7vzJMEofQzq7Z3C8SqW5w+opz9EZmbq2yVC/3s3TpeYOtVcXAp1TLJCq6/z9Q3vE8PDtGn8HUCMlj1FyFB8L4vr8AAOyT3jqniOuImyNXcaog+inAynrzbo0nJM8GR2F32esCuVqSrRyByvdQSzPebztBxFCLWuiQLuwrhET+PnS+FIURVIzTH4dj4eHlXmLhB/901OADdtcAgLCPQ54N8DVgqPG38G1Q55DR1VW0zX59y1uRnvz1PAPoCojfLTcKlX5jQx1YXPXrBMsz0xH1NgjfpAW7fro2gUxRt21lNlbRpZBozN52L9F9+G2qoawYOz7gm+7J55Eh/HcvH5vBm/5jrYDFbUSzyVNJwErrnZIvQ+KNmPGt8oBQUC9gq6sTeUsKo1aurRlJAobuoE8p3MF1CL+H5Ml8+EwM6piye8~3617328~4405560", 
            "_ga_87M7PJ3R97": "GS1.1.1742688165.29.1.1742689452.59.0.0", 
            "_ga_WM2NSQKJEK": "GS1.1.1742688165.27.1.1742689452.0.0.0", 
            "bm_sv": "844F9BEEC42F2F49FABEB18876EF30FD~YAAQ1TSVtIzsk62VAQAA9StZwBsIAILdMesCJSQ1cv7hOmPUNhSuZx39RDx1FMyw1A7krj6BzBd1BGQcdv0/cSuSv+OjPX7L7yG2xt9CYy2k4yuiB/YQ3IO145ZLiESUzbHoYRsosGROyRhv46irIViOuE0Cfy8yuvQQvQPSa/pCNd18bP1ouJEj1FhPwAmf6mJNDYNDI/n3WEl1bL5Yq34DQzm3TDj1hycF9Fw9Kfz93HyNQlhSa7NPC+tUqmMB4zJpxg==~1", 
            "RT": "z=1&dm=nseindia.com&si=838f4b69-b726-4466-81c7-40846689419c&ss=m8kwaxxl&sl=0&se=8c&tt=0&bcn=%2F%2F684d0d49.akstat.io%2F&ld=dmct&nu=kpaxjfo&cl=37i"
        }
    
    def _apply_hardcoded_cookies(self, session: requests.Session) -> requests.Session:
        """Apply hardcoded cookies to session"""
        hardcoded_cookies = self._get_hardcoded_cookies()
        for name, value in hardcoded_cookies.items():
            session.cookies.set(name, value, domain=self.domain)
        self.logger.info("Applied hardcoded cookies to session")
        return session
    
    def _log_error(self, error_message: str):
        try:
            self.logger.error(error_message)
            with open("errors.txt", "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {error_message}\n")
        except Exception as e:
            print(f"Error writing to error log: {e}")
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    def _get_fresh_cookies(self, referer_url: Optional[str] = None, max_retries: int = 3) -> requests.Session:
        """
        Get a session with fresh cookies from NSE India website.
        Falls back to hardcoded cookies if fresh ones can't be obtained.
        """
        if self.use_hardcoded_cookies:
            self.logger.info("Using hardcoded cookies directly as configured")
            session = requests.Session()
            session.headers.update(self.headers)
            return self._apply_hardcoded_cookies(session)
            
        # Try to get fresh cookies
        if not referer_url:
            referer_url = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
            
        session = requests.Session()
        session.headers.update(self.headers)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempt {attempt+1}/{max_retries}: Getting fresh cookies")
                main_page_url = "https://www.nseindia.com/"
                response = session.get(main_page_url, timeout=15)
                response.raise_for_status()
                
                if not session.cookies or 'nsit' not in [c.name for c in session.cookies]:
                    self.logger.warning("Missing 'nsit' cookie, retrying...")
                    time.sleep(2)
                    continue
                    
                response = session.get(referer_url, timeout=15)
                response.raise_for_status()
                
                if 'nseappid' in [c.name for c in session.cookies]:
                    self.logger.info("Successfully obtained NSE cookies")
                    return session
                else:
                    self.logger.warning("Missing 'nseappid' cookie, retrying...")
                    time.sleep(2)
                    continue
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Error getting fresh cookies (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
                
        self.logger.warning("Could not obtain fresh cookies. Falling back to hardcoded cookies.")
        return self._apply_hardcoded_cookies(session)
    
    def _refresh_session_if_needed(self, referer_url: str):
        self.request_count += 1
        if self.request_count >= self.requests_before_refresh:
            self.logger.info("Refreshing cookies...")
            self.session = self._get_fresh_cookies(referer_url)
            self.request_count = 0
            self.cookies = {cookie.name: cookie.value for cookie in self.session.cookies}
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def _make_async_request(self, url: str, referer_url: str) -> Optional[Dict[str, Any]]:
        try:
            self._refresh_session_if_needed(referer_url)
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Update the session's referer
            self.session.get(referer_url, timeout=10)
            
            # Prepare request headers with referer
            request_headers = {**self.headers, "Referer": referer_url}
            
            async with aiohttp.ClientSession(cookies=self.cookies) as session:
                async with session.get(url, headers=request_headers, timeout=60) as response:
                    response.raise_for_status()
                    
                    content = await response.content.read()
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
                            if '<html' in content.decode('utf-8', errors='ignore').lower():
                                self._log_error("Received HTML instead of JSON - might be a captcha or login page")
                            return None
                    else:
                        try:
                            return json.loads(content.decode('utf-8'))
                        except json.JSONDecodeError as e:
                            self._log_error(f"Error decoding JSON: {str(e)} for URL: {url}")
                            if '<html' in content.decode('utf-8', errors='ignore').lower():
                                self._log_error("Received HTML instead of JSON - might be a captcha or login page")
                            return None
        except Exception as e:
            self._log_error(f"Request error for URL {url}: {str(e)}")
            return None
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def download_file(self, url: str, file_type: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[bytes]]:
        try:
            if not file_type:
                parsed_url = urlparse(url)
                path = parsed_url.path
                file_type = path.split('.')[-1].lower() if '.' in path else "unknown"
            
            referer_url = f"{self.base_url}/companies-listing/corporate-filings"
            self._refresh_session_if_needed(referer_url)
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            self.session.get(referer_url, timeout=10)
            
            async with aiohttp.ClientSession(cookies=self.cookies) as session:
                headers = {**self.headers, "Referer": referer_url}
                async with session.get(url, headers=headers, timeout=60) as response:
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
                            self._log_error(f"Error extracting ZIP file: {str(e)}")
                            return False, None, content
                    
                    return True, temp_path, content
        except Exception as e:
            self._log_error(f"Error downloading file: {str(e)}")
            return False, None, None
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def detect_file_type(self, url: str) -> str:
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            if '.' in path:
                ext = path.split('.')[-1].lower()
                if ext in ['pdf', 'xml', 'zip', 'xbrl']:
                    return ext
            
            referer_url = f"{self.base_url}/companies-listing/corporate-filings"
            self._refresh_session_if_needed(referer_url)
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.3, 1.0))
            
            self.session.get(referer_url, timeout=10)
            
            async with aiohttp.ClientSession(cookies=self.cookies) as session:
                headers = {**self.headers, "Referer": referer_url}
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
        except Exception as e:
            self._log_error(f"Error detecting file type: {str(e)}")
            return 'unknown'
    
    def get_available_streams(self) -> List[Dict[str, Any]]:
        streams = []
        
        for key, config in self.config_dict.items():
            if config.get("active", False):
                stream_info = {
                    "key": key,
                    "endpoint": config.get("endpoint", ""),
                    "description": config.get("description", ""),
                    "params": config.get("params", {})
                }
                streams.append(stream_info)
                
        return streams
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def fetch_data(self, stream_key: str, **kwargs) -> List[Dict[str, Any]]:
        try:
            if stream_key not in self.config_dict:
                self._log_error(f"Stream key not found: {stream_key}")
                return []
                
            endpoint_config = self.config_dict[stream_key]
            
            if not endpoint_config.get("active", False):
                self._log_error(f"Stream is inactive: {stream_key}")
                return []
                
            endpoint = endpoint_config.get("endpoint", "")
            
            if not endpoint:
                self._log_error(f"Endpoint URL not found for: {stream_key}")
                return []
                
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
                    self.session = self._get_fresh_cookies(referer_url)
                    self.cookies = {cookie.name: cookie.value for cookie in self.session.cookies}
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
                        self._log_error(f"Invalid data format for {stream_key}")
                        continue  # Try again if format is invalid
            
            return []
            
        except Exception as e:
            self._log_error(f"Error fetching data for {stream_key}: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def fetch_multiple(self, actions_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        results = {}
        
        for stream_key, params in actions_params.items():
            if stream_key not in self.config_dict:
                self._log_error(f"Unknown stream: {stream_key}")
                continue
                
            if not self.config_dict[stream_key].get("active", False):
                self._log_error(f"Stream is inactive: {stream_key}")
                continue
                
            data = await self.fetch_data(stream_key, **params)
            results[stream_key] = data
            
            # Add a small delay between requests
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        return results
    
    async def run(self, command: str, **kwargs) -> ToolResult:
        try:
            if command in self.config_dict and self.config_dict[command].get("active", False):
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
                    
        except Exception as e:
            error_message = f"Error in NSE tool: {str(e)}"
            self._log_error(error_message)
            return await self._handle_error(e)


if __name__ == "__main__":
    CONFIG_PATH = "assets"
    
    async def test_nse_tool():
        """
        Test function to demonstrate NSETool functionality
        """
        config = {
            "base_url": "https://www.nseindia.com",
            "config_path": f"{CONFIG_PATH}/nse_config.yaml",
            "headers_path": f"{CONFIG_PATH}/headers.yaml",
            "requests_before_refresh": 20,
            "retry_limit": 3,
            "multiplier": 1,
            "min_wait": 2,
            "max_wait": 10,
            "use_hardcoded_cookies": False  # Use hardcoded cookies by default
        }
        
        try:
            company_name = "Reliance Industries Limited"
            symbol = "RELIANCE"
            
            tool = NSETool(config, company=company_name, symbol=symbol)
            print(f"NSETool initialized for {company_name} ({symbol})")
            
            print("\nAvailable data streams:")
            streams = tool.get_available_streams()
            for i, stream in enumerate(streams[:5]):  # Show first 5 streams
                print(f"  {i+1}. {stream['key']}: {stream['endpoint']}")
            print(f"  ... and {len(streams) - 5} more")
            
            print(f"\n[TEST 1] Fetching announcements (no additional parameters)")
            announcements = await tool.fetch_data("Announcements")
            print(f"Fetched {len(announcements)} announcements")
            
            if announcements:
                print("\nSample announcement:")
                first_announcement = announcements[0]
                sample_data = {k: first_announcement[k] for k in list(first_announcement.keys())[:5]}
                print(json.dumps(sample_data, indent=2))
            
            print(f"\n[TEST 2] Demonstrating fetch_multiple with different parameters for each stream")
            
            # Dictionary mapping streams to their specific parameters
            actions_with_params = {
                "InsiderTrading": {
                    "from_date": "01-01-2024",
                    "to_date": "15-01-2024",
                    "max_results": 5
                },
                "BoardMeetings": {
                    "from_date": "01-05-2023", 
                    "to_date": "31-12-2023",
                    "max_results": 3
                },
                "FinancialResults": {
                    "period": "Quarterly",
                    "max_results": 2
                }
            }
            
            multi_results = await tool.fetch_multiple(actions_with_params)
            
            print("Results summary with different parameters for each stream:")
            for key, data in multi_results.items():
                print(f"  - {key}: {len(data)} records")
                
                # Show the specific parameters used for this stream
                params = actions_with_params[key]
                params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                print(f"    Parameters: {params_str}")
                
                # Show a small sample of the first record if available
                if data:
                    first_record = data[0]
                    sample_keys = list(first_record.keys())[:3]  # Just show first 3 keys
                    sample_data = {k: first_record[k] for k in sample_keys}
                    print(f"    Sample: {json.dumps(sample_data)}")
                    
                print()  # Add a blank line for readability
            
            print(f"\n[TEST 3] Fetching financial results with period parameter")
            financial_results = await tool.fetch_data("FinancialResults", period="Annual")
            print(f"Fetched {len(financial_results)} annual financial results")
            
            if financial_results:
                print("\nSample annual financial result:")
                first_result = financial_results[0]
                sample_data = {k: first_result[k] for k in list(first_result.keys())[:5]}
                print(json.dumps(sample_data, indent=2))
            
            # Test using the run method
            print("\n[TEST 4] Using the run method to fetch data")
            result = await tool.run("InsiderTrading", from_date="01-01-2024", to_date="31-01-2024")
            if result.success:
                print(f"Successfully fetched {len(result.data)} insider trading records")
            else:
                print(f"Failed to fetch insider trading records: {result.error}")
        
        except Exception as e:
            print(f"Error in test: {str(e)}")
    
    asyncio.run(test_nse_tool())