import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import logging
import json
import tempfile
import os
import zipfile
from typing import Dict, List, Any, Optional

from tools.nse_tool import NSETool, NSEToolConfig
from base.base_tools import ToolResult


# Mock data for tests
MOCK_CONFIG = {
    "base_url": "https://www.nseindia.com",
    "requests_before_refresh": 5,
    "config_path": "assets/nse_config.yaml",
    "headers_path": "assets/headers.yaml",
    "cookies_path": "assets/cookies.yaml",
    "retry_limit": 1,  # Low values for testing
    "multiplier": 0,
    "min_wait": 0,
    "max_wait": 0,
    "use_hardcoded_cookies": True
}

MOCK_COMPANY = "Test Company"
MOCK_SYMBOL = "TEST"

# Mock YAML configs
MOCK_CONFIG_DICT = {
    "insider_trading": {
        "active": True,
        "endpoint": "corporates/insiderTrading/data",
        "description": "Insider Trading Data",
        "referer": "corporate-filings-insider-trading",
        "params": {
            "symbol": True,
            "period": True,
            "max_results": True
        }
    },
    "corporate_announcements": {
        "active": True,
        "endpoint": "corporates/announcements/data",
        "description": "Corporate Announcements",
        "referer": "corporate-filings-announcements",
        "params": {
            "symbol": True,
            "fromDate": True,
            "toDate": True
        }
    },
    "inactive_stream": {
        "active": False,
        "endpoint": "inactive/endpoint",
        "description": "Inactive Stream",
        "params": {}
    }
}

MOCK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "sec-ch-ua": "\"Chromium\";v=\"134\", \"Not:A-Brand\";v=\"24\", \"Google Chrome\";v=\"134\"",
    "sec-ch-ua-platform": "\"macOS\""
}

MOCK_COOKIES = {
    "_ga": "GA1.1.1579713621.1719746123", 
    "nsit": "yUBOLwMh-t6f0dFy41LLe8sx", 
    "nseappid": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcGkubnNlIiwiYXVkIjoiYXBpLm5zZSIsImlhdCI6MTc0MjY4OTQ1MCwiZXhwIjoxNzQyNjk2NjUwfQ.2gYGSy0Brw-atRx7XngSEByNWuSCNG7hKTTMifgGShY"
}

# Mock response data
MOCK_INSIDER_TRADING_DATA = {
    "data": [
        {
            "symbol": "TEST",
            "name": "Test Person",
            "personCategory": "Director",
            "acqMode": "Market Purchase",
            "secType": "Equity Shares",
            "secAcq": 1000,
            "secSold": 0,
            "value": 100000,
            "acqFromDt": "2023-01-01",
            "acqToDt": "2023-01-02"
        },
        {
            "symbol": "TEST",
            "name": "Another Person",
            "personCategory": "Promoter",
            "acqMode": "Market Sale",
            "secType": "Equity Shares",
            "secAcq": 0,
            "secSold": 500,
            "value": 50000,
            "acqFromDt": "2023-01-03",
            "acqToDt": "2023-01-04"
        }
    ]
}

MOCK_ANNOUNCEMENTS_DATA = {
    "data": [
        {
            "symbol": "TEST",
            "name": "Test Company",
            "subject": "Test Announcement",
            "attachmentName": "test.pdf",
            "attachmentURL": "/download/test.pdf",
            "cs": "Test CS",
            "csURL": "/download/cs.pdf",
            "xbrl": "test.xml",
            "xbrlURL": "/download/test.xml",
            "capturedDt": "2023-01-01"
        }
    ]
}

# Mock file data
MOCK_FILE_URL = "https://www.nseindia.com/download/test.pdf"
MOCK_FILE_CONTENT = b"Mock PDF content"
MOCK_FILE_PATH = "/tmp/test.pdf"


class MockResponse:
    """Mock for aiohttp ClientResponse"""
    def __init__(self, status=200, content=None, headers=None, url=None, content_type="application/json"):
        self.status = status
        self._content = content or b""
        self.headers = headers or {}
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = content_type
        self._url = url or "https://www.nseindia.com/api/test"
        self.content = MagicMock()
        self.content.read = AsyncMock(return_value=self._content)

    def raise_for_status(self):
        if self.status >= 400:
            raise Exception(f"HTTP Error {self.status}")


class MockCookie:
    """Mock for requests.cookies.Cookie"""
    def __init__(self, name, value):
        self.name = name
        self.value = value


class MockSession:
    """Mock for requests Session"""
    def __init__(self, response=None):
        self.response = response or MockResponse(200)
        self.headers = {}
        self.cookies = []
        for name, value in MOCK_COOKIES.items():
            self.cookies.append(MockCookie(name, value))
        self.get = MagicMock(return_value=self.response)
        self.post = MagicMock(return_value=self.response)


class MockClientSession:
    """Mock for aiohttp ClientSession"""
    def __init__(self, response=None):
        self.response = response or MockResponse(200)
        self.headers = {}
        self.cookies = MOCK_COOKIES
        self.get = AsyncMock(return_value=self.response)
        self.post = AsyncMock(return_value=self.response)
        self.head = AsyncMock(return_value=self.response)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_logger():
    """Fixture for mocking the logger"""
    mock_log = MagicMock(spec=logging.Logger)
    
    with patch("utils.logging.get_logger") as mock_get_logger:
        mock_get_logger.return_value = mock_log
        yield mock_log


@pytest.fixture
def mock_yaml_configs():
    """Fixture for mocking YAML configurations"""
    with patch("yaml.safe_load") as mock_yaml_load:
        # Return different values based on call sequence
        def side_effect(file_obj):
            call_count = mock_yaml_load.call_count
            if call_count == 1:  # First call (config)
                return MOCK_CONFIG_DICT
            else:  # Second call (headers)
                return MOCK_HEADERS
        
        mock_yaml_load.side_effect = side_effect
        yield mock_yaml_load


@pytest.fixture
def mock_requests_session():
    """Fixture for mocking requests Session"""
    with patch("requests.Session") as mock_session_class:
        session = MockSession()
        mock_session_class.return_value = session
        yield session


@pytest.fixture
def mock_aiohttp_session():
    """Fixture for mocking aiohttp ClientSession"""
    with patch("aiohttp.ClientSession") as mock_session_class:
        session = MockClientSession()
        mock_session_class.return_value = session
        yield session


@pytest_asyncio.fixture
async def nse_tool(mock_logger, mock_yaml_configs, mock_requests_session):
    """Fixture to create a NSETool with mocked dependencies"""
    with patch("os.path.exists", return_value=True), \
         patch("open", MagicMock()), \
         patch.object(NSETool, "_update_retry_params"):  # Skip retry params setup
        
        tool = NSETool(MOCK_CONFIG, MOCK_COMPANY, MOCK_SYMBOL)
        # Replace configs directly for testing
        tool.config_dict = MOCK_CONFIG_DICT
        tool.logger = mock_logger
        
        yield tool


@pytest.mark.asyncio
async def test_get_fresh_cookies(nse_tool, mock_requests_session):
    """Test getting fresh cookies"""
    # Call the method
    session = nse_tool._get_fresh_cookies()
    
    # Verify that the session is properly configured
    assert session is mock_requests_session
    assert any(cookie.name == "nsit" for cookie in session.cookies)
    assert any(cookie.name == "nseappid" for cookie in session.cookies)
    

@pytest.mark.asyncio
async def test_get_available_streams(nse_tool):
    """Test getting available streams"""
    # Call the method
    streams = nse_tool.get_available_streams()
    
    # Verify the result
    assert len(streams) == 2  # Only active streams
    assert any(stream["key"] == "insider_trading" for stream in streams)
    assert any(stream["key"] == "corporate_announcements" for stream in streams)
    assert not any(stream["key"] == "inactive_stream" for stream in streams)


@pytest.mark.asyncio
async def test_apply_hardcoded_cookies(nse_tool, mock_requests_session):
    """Test applying hardcoded cookies to session"""
    # Call the method
    session = nse_tool._apply_hardcoded_cookies(mock_requests_session)
    
    # Verify cookies were applied
    assert session is mock_requests_session
    cookie_names = [cookie.name for cookie in session.cookies]
    assert "nsit" in cookie_names
    assert "nseappid" in cookie_names


@pytest.mark.asyncio
async def test_make_async_request(nse_tool, mock_aiohttp_session):
    """Test making an async request"""
    # Mock the response content
    json_content = json.dumps(MOCK_INSIDER_TRADING_DATA).encode('utf-8')
    mock_aiohttp_session.response._content = json_content
    
    # Call the method
    url = "https://www.nseindia.com/api/corporates/insiderTrading/data?symbol=TEST"
    referer_url = "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"
    result = await nse_tool._make_async_request(url, referer_url)
    
    # Verify the result
    assert result == MOCK_INSIDER_TRADING_DATA
    mock_aiohttp_session.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_make_async_request_with_brotli(nse_tool, mock_aiohttp_session):
    """Test making an async request with brotli compression"""
    # Skip if brotli is not available
    brotli_import_error = False
    try:
        import brotli
    except ImportError:
        brotli_import_error = True
    
    if brotli_import_error:
        pytest.skip("brotli library not available")
    
    # Mock the response content and headers for brotli
    with patch("brotli.decompress") as mock_decompress:
        json_str = json.dumps(MOCK_INSIDER_TRADING_DATA)
        mock_decompress.return_value = json_str.encode('utf-8')
        
        mock_aiohttp_session.response.headers["Content-Encoding"] = "br"
        mock_aiohttp_session.response._content = b"mock_compressed_content"
        
        # Call the method
        url = "https://www.nseindia.com/api/corporates/insiderTrading/data?symbol=TEST"
        referer_url = "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"
        result = await nse_tool._make_async_request(url, referer_url)
        
        # Verify the result
        assert result == MOCK_INSIDER_TRADING_DATA
        mock_decompress.assert_called_once_with(b"mock_compressed_content")


@pytest.mark.asyncio
async def test_make_async_request_error(nse_tool, mock_aiohttp_session):
    """Test error handling in async request"""
    # Setup the mock to raise an exception
    mock_aiohttp_session.get.side_effect = Exception("Connection error")
    
    # Call the method
    url = "https://www.nseindia.com/api/error"
    referer_url = "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"
    result = await nse_tool._make_async_request(url, referer_url)
    
    # Verify the result
    assert result is None
    nse_tool.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_download_file(nse_tool, mock_aiohttp_session):
    """Test downloading a file"""
    # Mock file operations
    with patch("aiofiles.open", AsyncMock()), \
         patch("os.path.basename", return_value="test.pdf"), \
         patch("tempfile.gettempdir", return_value="/tmp"):
        
        # Setup mock response
        mock_aiohttp_session.response._content = MOCK_FILE_CONTENT
        mock_aiohttp_session.response.headers["Content-Disposition"] = 'filename="test.pdf"'
        
        # Call the method
        success, file_path, content = await nse_tool.download_file(MOCK_FILE_URL, "pdf")
        
        # Verify the result
        assert success is True
        assert file_path == MOCK_FILE_PATH
        assert content == MOCK_FILE_CONTENT


@pytest.mark.asyncio
async def test_download_zip_file(nse_tool, mock_aiohttp_session):
    """Test downloading and extracting a ZIP file"""
    # Mock file operations
    with patch("aiofiles.open", AsyncMock()), \
         patch("tempfile.mkdtemp", return_value="/tmp/extract_dir"), \
         patch("tempfile.gettempdir", return_value="/tmp"), \
         patch("os.path.basename", return_value="test.zip"), \
         patch("zipfile.ZipFile"), \
         patch("os.remove"):
        
        # Setup mock response
        zip_content = b"mock zip content"
        mock_aiohttp_session.response._content = zip_content
        mock_aiohttp_session.response.headers["Content-Disposition"] = 'filename="test.zip"'
        
        # Call the method
        success, file_path, content = await nse_tool.download_file(MOCK_FILE_URL, "zip")
        
        # Verify the result
        assert success is True
        assert file_path == "/tmp/extract_dir"
        assert content is None


@pytest.mark.asyncio
async def test_detect_file_type(nse_tool, mock_aiohttp_session):
    """Test detecting file type from URL"""
    # Test with file extension in URL
    file_type = await nse_tool.detect_file_type("https://www.nseindia.com/download/document.pdf")
    assert file_type == "pdf"
    
    # Test with Content-Type header
    mock_aiohttp_session.response.headers["Content-Type"] = "application/pdf"
    file_type = await nse_tool.detect_file_type("https://www.nseindia.com/download/document")
    assert file_type == "pdf"
    
    # Test with Content-Disposition header
    mock_aiohttp_session.response.headers["Content-Type"] = "application/octet-stream"
    mock_aiohttp_session.response.headers["Content-Disposition"] = 'filename="document.xml"'
    file_type = await nse_tool.detect_file_type("https://www.nseindia.com/download/document")
    assert file_type == "xml"


@pytest.mark.asyncio
async def test_fetch_data(nse_tool):
    """Test fetching data from a stream"""
    # Mock the _make_async_request method
    with patch.object(nse_tool, "_make_async_request") as mock_request:
        mock_request.return_value = MOCK_INSIDER_TRADING_DATA
        
        # Call the method
        data = await nse_tool.fetch_data("insider_trading", period="1m", max_results=10)
        
        # Verify the result
        assert len(data) == 2
        assert data[0]["symbol"] == "TEST"
        assert data[0]["name"] == "Test Person"
        assert data[1]["personCategory"] == "Promoter"


@pytest.mark.asyncio
async def test_fetch_data_stream_not_found(nse_tool):
    """Test fetching data from a non-existent stream"""
    # Call the method with invalid stream key
    data = await nse_tool.fetch_data("non_existent_stream")
    
    # Verify the result is empty
    assert data == []
    nse_tool.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_data_inactive_stream(nse_tool):
    """Test fetching data from an inactive stream"""
    # Call the method with inactive stream key
    data = await nse_tool.fetch_data("inactive_stream")
    
    # Verify the result is empty
    assert data == []
    nse_tool.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_multiple(nse_tool):
    """Test fetching data from multiple streams"""
    # Mock the fetch_data method
    with patch.object(nse_tool, "fetch_data") as mock_fetch:
        def side_effect(stream_key, **kwargs):
            if stream_key == "insider_trading":
                return MOCK_INSIDER_TRADING_DATA["data"]
            elif stream_key == "corporate_announcements":
                return MOCK_ANNOUNCEMENTS_DATA["data"]
            return []
        
        mock_fetch.side_effect = side_effect
        
        # Call the method
        actions_params = {
            "insider_trading": {"period": "1m"},
            "corporate_announcements": {"fromDate": "2023-01-01", "toDate": "2023-01-31"}
        }
        results = await nse_tool.fetch_multiple(actions_params)
        
        # Verify the result
        assert "insider_trading" in results
        assert "corporate_announcements" in results
        assert len(results["insider_trading"]) == 2
        assert len(results["corporate_announcements"]) == 1


@pytest.mark.asyncio
async def test_run_fetch_data_command(nse_tool):
    """Test the run method with a data fetch command"""
    # Mock the fetch_data method
    with patch.object(nse_tool, "fetch_data") as mock_fetch:
        mock_fetch.return_value = MOCK_INSIDER_TRADING_DATA["data"]
        
        # Call the run method
        result = await nse_tool.run("insider_trading", period="1m")
        
        # Verify the result
        assert result.success is True
        assert result.data == MOCK_INSIDER_TRADING_DATA["data"]
        mock_fetch.assert_called_once_with("insider_trading", period="1m")


@pytest.mark.asyncio
async def test_run_fetch_multiple_command(nse_tool):
    """Test the run method with the fetch_multiple command"""
    # Mock the fetch_multiple method
    with patch.object(nse_tool, "fetch_multiple") as mock_fetch:
        mock_result = {
            "insider_trading": MOCK_INSIDER_TRADING_DATA["data"],
            "corporate_announcements": MOCK_ANNOUNCEMENTS_DATA["data"]
        }
        mock_fetch.return_value = mock_result
        
        # Call the run method
        actions_params = {
            "insider_trading": {"period": "1m"},
            "corporate_announcements": {"fromDate": "2023-01-01", "toDate": "2023-01-31"}
        }
        result = await nse_tool.run("fetch_multiple", actions_params=actions_params)
        
        # Verify the result
        assert result.success is True
        assert result.data == mock_result
        mock_fetch.assert_called_once_with(actions_params)


@pytest.mark.asyncio
async def test_run_download_file_command(nse_tool):
    """Test the run method with the download_file command"""
    # Mock the download_file method
    with patch.object(nse_tool, "download_file") as mock_download:
        mock_download.return_value = (True, MOCK_FILE_PATH, MOCK_FILE_CONTENT)
        
        # Call the run method
        result = await nse_tool.run("download_file", url=MOCK_FILE_URL, file_type="pdf")
        
        # Verify the result
        assert result.success is True
        assert result.data["file_path"] == MOCK_FILE_PATH
        assert result.data["file_type"] == "pdf"
        mock_download.assert_called_once_with(MOCK_FILE_URL, "pdf")


@pytest.mark.asyncio
async def test_run_get_streams_command(nse_tool):
    """Test the run method with the get_streams command"""
    # Mock the get_available_streams method
    with patch.object(nse_tool, "get_available_streams") as mock_streams:
        mock_streams.return_value = [
            {"key": "insider_trading", "endpoint": "endpoint1", "description": "desc1", "params": {}},
            {"key": "corporate_announcements", "endpoint": "endpoint2", "description": "desc2", "params": {}}
        ]
        
        # Call the run method
        result = await nse_tool.run("get_streams")
        
        # Verify the result
        assert result.success is True
        assert len(result.data) == 2
        mock_streams.assert_called_once()


@pytest.mark.asyncio
async def test_run_invalid_command(nse_tool):
    """Test the run method with an invalid command"""
    # Call the run method with invalid command
    result = await nse_tool.run("invalid_command")
    
    # Verify the result
    assert result.success is False
    assert result.error is not None
    assert "Unknown command" in result.error


@pytest.mark.asyncio
async def test_run_download_file_missing_url(nse_tool):
    """Test the run method with missing required parameter"""
    # Call the run method with missing url parameter
    result = await nse_tool.run("download_file")
    
    # Verify the result
    assert result.success is False
    assert result.error is not None
    assert "URL is required" in result.error


@pytest.mark.asyncio
async def test_run_fetch_multiple_missing_params(nse_tool):
    """Test the run method with missing actions_params"""
    # Call the run method with missing actions_params
    result = await nse_tool.run("fetch_multiple")
    
    # Verify the result
    assert result.success is False
    assert result.error is not None
    assert "actions_params dictionary is required" in result.error


@pytest.mark.asyncio
async def test_error_handling(nse_tool):
    """Test general error handling in the run method"""
    # Mock fetch_data to raise an exception
    with patch.object(nse_tool, "fetch_data") as mock_fetch:
        mock_fetch.side_effect = Exception("API Error")
        
        # Call the run method
        result = await nse_tool.run("insider_trading", period="1m")
        
        # Verify the result
        assert result.success is False
        assert result.error is not None
        assert "API Error" in result.error


@pytest.mark.asyncio
async def test_refresh_session_if_needed(nse_tool):
    """Test session refresh when request count exceeds limit"""
    # Save original session
    original_session = nse_tool.session
    
    # Mock _get_fresh_cookies
    with patch.object(nse_tool, "_get_fresh_cookies") as mock_cookies:
        new_session = MockSession()
        mock_cookies.return_value = new_session
        
        # Set request count to trigger refresh
        nse_tool.request_count = nse_tool.requests_before_refresh
        
        # Call the method
        referer_url = "https://www.nseindia.com/test"
        nse_tool._refresh_session_if_needed(referer_url)
        
        # Verify session was refreshed
        assert nse_tool.request_count == 0
        assert nse_tool.session is not original_session
        mock_cookies.assert_called_once_with(referer_url)