import pytest
import pytest_asyncio
import json
import brotli
import asyncio
import yaml
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import logging
from typing import Dict, List, Any, Optional

from tools.nse_tool import NSETool, NSEToolConfig, NSEError, NSEConnectionError, NSERateLimitError

# Mock data for testing
MOCK_COMPANY = "Reliance Industries Limited"
MOCK_SYMBOL = "RELIANCE"

# Mock responses for different API calls
MOCK_ANNOUNCEMENTS_RESPONSE = [
    {
        "exchdisstime": "01-Mar-2025 09:30:00",
        "attchmntFile": "announcement1.pdf",
        "attchmntText": "Quarterly results announcement",
        "desc": "Financial Results"
    },
    {
        "exchdisstime": "15-Mar-2025 14:15:00",
        "attchmntFile": "announcement2.pdf",
        "attchmntText": "Board meeting outcome",
        "desc": "Board Meeting"
    }
]

MOCK_BOARD_MEETINGS_RESPONSE = [
    {
        "bm_date": "10-Mar-2025",
        "bm_purpose": "RELIANCE",
        "bm_desc": "To consider and approve quarterly results",
        "attachment": "meeting_notice.pdf"
    },
    {
        "bm_date": "25-Mar-2025",
        "bm_purpose": "RELIANCE",
        "bm_desc": "To consider dividend declaration",
        "attachment": "dividend_notice.pdf"
    }
]

MOCK_CORPORATE_ACTIONS_RESPONSE = [
    {
        "exDate": "20-Mar-2025",
        "subject": "Dividend Payment"
    },
    {
        "exDate": "05-Apr-2025",
        "subject": "Rights Issue"
    }
]

MOCK_ANNUAL_REPORTS_RESPONSE = [
    {
        "fromYr": "2023",
        "toYr": "2024",
        "fileName": "annual_report_2024.pdf"
    }
]

MOCK_BUSINESS_SUSTAINABILITY_RESPONSE = [
    {
        "fyFrom": "2023",
        "fyTo": "2024",
        "attachmentFile": "sustainability_report_2024.pdf",
        "submissionDate": "15-Feb-2025"
    }
]

MOCK_CREDIT_RATING_RESPONSE = [
    {
        "NameOfCRAgency": "CRISIL",
        "CreditRating": "AAA",
        "RatingAction": "Affirmed",
        "Outlook": "Stable",
        "DateOfVer": "10-Jan-2025",
        "CreditRatingEarlier": "AAA",
        "Savithri Parekh": "Company Secretary",
        "Designation": "Company Secretary and Compliance Officer"
    }
]

MOCK_CONFIG = {
    "Announcements": {
        "endpoint": "corporate-announcements",
        "params": {
            "index": "equities",
            "from_date": "",
            "to_date": "",
            "symbol": "",
            "issuer": ""
        },
        "referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
        "description": "Returns corporate announcements by companies listed on NSE.",
        "active": True
    },
    "BoardMeetings": {
        "endpoint": "corporate-board-meetings",
        "params": {
            "index": "equities",
            "from_date": "",
            "to_date": "",
            "symbol": "",
            "issuer": ""
        },
        "referer": "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings",
        "description": "Returns information about board meetings including date and purpose.",
        "active": True
    },
    "CorporateActions": {
        "endpoint": "corporates-corporateActions",
        "params": {
            "index": "equities",
            "from_date": "",
            "to_date": "",
            "symbol": "",
            "issuer": ""
        },
        "referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
        "description": "Returns information about corporate actions (dividends, splits, etc.).",
        "active": True
    },
    "AnnualReports": {
        "endpoint": "annual-reports",
        "params": {
            "index": "equities",
            "symbol": ""
        },
        "referer": "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports",
        "description": "Returns links to annual reports (.pdf or .zip files) of NSE-listed companies.",
        "active": True
    },
    "BussinessSustainabilitiyReport": {
        "endpoint": "corporate-bussiness-sustainabilitiy",
        "params": {
            "index": "equities",
            "symbol": "",
            "issuer": ""
        },
        "referer": "https://www.nseindia.com/companies-listing/corporate-filings-bussiness-sustainabilitiy-reports",
        "description": "Returns links to Business Sustainability reports (.pdf or .zip files).",
        "active": True
    },
    "CreditRating": {
        "endpoint": "corporate-credit-rating",
        "params": {
            "index": "",
            "issuer": ""
        },
        "referer": "https://www.nseindia.com/companies-listing/debt-centralised-database/crd",
        "description": "Returns the credit rating information of the company.",
        "active": True
    }
}

MOCK_SCHEMA = {
    "Announcements": {
        "event_type": "desc",
        "attachment": "attchmntFile",
        "description": "attchmntText",
        "time": "exchdisstime"
    },
    "BoardMeetings": {
        "date": "bm_date",
        "symbol": "bm_purpose",
        "description": "bm_desc",
        "attachment": "attachment"
    },
    "CorporateActions": {
        "date": "exDate",
        "subject": "subject"
    },
    "AnnualReports": {
        "from_year": "fromYr",
        "to_year": "toYr",
        "attachment": "fileName"
    },
    "BussinessSustainabilitiyReport": {
        "from_year": "fyFrom",
        "to_year": "fyTo",
        "attachment": "attachmentFile",
        "submission_date": "submissionDate"
    },
    "CreditRating": {
        "agency": "NameOfCRAgency",
        "rating": "CreditRating",
        "rating_action": "RatingAction",
        "outlook": "Outlook",
        "date": "DateOfVer",
        "older_rating": "CreditRatingEarlier",
        "signatory_name": "Savithri Parekh",
        "signatory_designation": "Designation"
    }
}

MOCK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"
}

MOCK_COOKIES = {
    "_ga": "GA1.1.1579713621.1719746123",
    "nsit": "boR2K8ZHg2PSjjj8ZEo5Dzzu"
}


@pytest.fixture
def config():
    """Fixture providing configuration for the NSE tool."""
    return {
        "company": MOCK_COMPANY,
        "symbol": MOCK_SYMBOL
    }


@pytest.fixture
def mock_logger():
    """Fixture for mocking the logger"""
    mock_log = MagicMock(spec=logging.Logger)
    
    with patch("utils.logging.get_logger") as mock_get_logger:
        mock_get_logger.return_value = mock_log
        yield mock_log


@pytest_asyncio.fixture
async def mock_aiohttp_session():
    """Fixture to create a mock aiohttp ClientSession."""
    mock_session = MagicMock()
    
    # Mock the cookie jar
    mock_cookie_jar = MagicMock()
    mock_cookie = MagicMock()
    mock_cookie.key = "nsit"
    mock_cookie.value = "mock_cookie_value"
    mock_cookie_jar.__iter__.return_value = [mock_cookie]
    mock_session.cookie_jar = mock_cookie_jar
    
    # Mock the get method
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    # Create read and text methods
    mock_response.read = AsyncMock(return_value=b'{"test": "data"}')
    mock_response.text = AsyncMock(return_value='{"test": "data"}')
    mock_response.headers = {"Content-Encoding": ""}
    
    # Mock the get method to return our mock response
    mock_session.get = AsyncMock(return_value=mock_response)
    
    # Return the mock session
    return mock_session


@pytest_asyncio.fixture
async def nse_tool(config, mock_logger, mock_aiohttp_session):
    """Fixture to create an NSETool with mocked dependencies."""
    
    # Create mock YAML loading function
    def mock_load_yaml(path):
        if "config" in path:
            return MOCK_CONFIG
        elif "headers" in path:
            return MOCK_HEADERS
        elif "cookies" in path:
            return MOCK_COOKIES
        elif "schema" in path:
            return MOCK_SCHEMA
        return {}
    
    # Patch the _create_new_session method to return our mock session
    with patch("tools.nse_tool.NSETool._load_yaml", side_effect=mock_load_yaml), \
         patch("tools.nse_tool.NSETool._create_new_session", return_value=mock_aiohttp_session), \
         patch("tools.nse_tool.NSETool._refresh_session", AsyncMock()):
        
        # Create the NSETool instance
        tool = NSETool(config)
        tool.logger = mock_logger
        tool.session = mock_aiohttp_session  # Directly set the session
        
        # Mock make_request to return different responses based on the URL
        async def mock_make_request(url, referer):
            if "corporate-announcements" in url:
                return MOCK_ANNOUNCEMENTS_RESPONSE
            elif "corporate-board-meetings" in url:
                return MOCK_BOARD_MEETINGS_RESPONSE
            elif "corporates-corporateActions" in url:
                return MOCK_CORPORATE_ACTIONS_RESPONSE
            elif "annual-reports" in url:
                return MOCK_ANNUAL_REPORTS_RESPONSE
            elif "corporate-bussiness-sustainabilitiy" in url:
                return MOCK_BUSINESS_SUSTAINABILITY_RESPONSE
            elif "corporate-credit-rating" in url:
                return MOCK_CREDIT_RATING_RESPONSE
            return []
        
        tool.make_request = mock_make_request
        
        yield tool
        
        # Cleanup
        if hasattr(tool, 'close'):
            await tool.close()


@pytest.mark.asyncio
async def test_fetch_data_from_nse(nse_tool):
    """Test fetching data from different NSE streams."""
    # Test announcements
    announcements = await nse_tool.fetch_data_from_nse("Announcements", {})
    assert len(announcements) == 2
    assert announcements[0]["desc"] == "Financial Results"
    
    # Test board meetings
    board_meetings = await nse_tool.fetch_data_from_nse("BoardMeetings", {})
    assert len(board_meetings) == 2
    assert board_meetings[0]["bm_desc"] == "To consider and approve quarterly results"
    
    # Test corporate actions
    corporate_actions = await nse_tool.fetch_data_from_nse("CorporateActions", {})
    assert len(corporate_actions) == 2
    assert corporate_actions[0]["subject"] == "Dividend Payment"


@pytest.mark.asyncio
async def test_filter_on_schema(nse_tool):
    """Test filtering data using schema."""
    # Test announcements
    announcements = await nse_tool.fetch_data_from_nse("Announcements", {})
    schema = nse_tool._get_schema("Announcements")
    filtered = nse_tool._filter_on_schema(announcements, schema)
    
    assert len(filtered) == 2
    assert "event_type" in filtered[0]
    assert "attachment" in filtered[0]
    assert "description" in filtered[0]
    assert "time" in filtered[0]
    assert filtered[0]["event_type"] == "Financial Results"
    assert filtered[0]["attachment"] == "announcement1.pdf"


@pytest.mark.asyncio
async def test_process_announcements(nse_tool):
    """Test the announcements processor."""
    schema = nse_tool._get_schema("Announcements")
    processed = await nse_tool._process_announcements({}, schema)
    
    assert len(processed) == 2
    assert "event_type" in processed[0]
    assert "attachment" in processed[0]
    assert processed[0]["event_type"] == "Financial Results"


@pytest.mark.asyncio
async def test_process_board_meetings(nse_tool):
    """Test the board meetings processor."""
    schema = nse_tool._get_schema("BoardMeetings")
    processed = await nse_tool._process_board_meetings({}, schema)
    
    assert len(processed) == 2
    assert "date" in processed[0]
    assert "description" in processed[0]
    assert processed[0]["date"] == "10-Mar-2025"
    assert processed[0]["description"] == "To consider and approve quarterly results"


@pytest.mark.asyncio
async def test_process_corporate_actions(nse_tool):
    """Test the corporate actions processor."""
    schema = nse_tool._get_schema("CorporateActions")
    processed = await nse_tool._process_corporate_actions({}, schema)
    
    assert len(processed) == 2
    assert "date" in processed[0]
    assert "subject" in processed[0]
    assert processed[0]["date"] == "20-Mar-2025"
    assert processed[0]["subject"] == "Dividend Payment"


@pytest.mark.asyncio
async def test_process_annual_reports(nse_tool):
    """Test the annual reports processor."""
    schema = nse_tool._get_schema("AnnualReports")
    processed = await nse_tool._process_annual_reports({}, schema)
    
    assert len(processed) == 1
    assert "from_year" in processed[0]
    assert "to_year" in processed[0]
    assert "attachment" in processed[0]
    assert processed[0]["from_year"] == "2023"
    assert processed[0]["to_year"] == "2024"
    assert processed[0]["attachment"] == "annual_report_2024.pdf"


@pytest.mark.asyncio
async def test_process_esg_reports(nse_tool):
    """Test the ESG reports processor."""
    schema = nse_tool._get_schema("BussinessSustainabilitiyReport")
    processed = await nse_tool._process_esg_reports({}, schema)
    
    assert len(processed) == 1
    assert "from_year" in processed[0]
    assert "to_year" in processed[0]
    assert "attachment" in processed[0]
    assert "submission_date" in processed[0]
    assert processed[0]["from_year"] == "2023"
    assert processed[0]["to_year"] == "2024"
    assert processed[0]["attachment"] == "sustainability_report_2024.pdf"


@pytest.mark.asyncio
async def test_fetch_data_from_multiple_streams(nse_tool):
    """Test fetching data from multiple streams."""
    stream_dict = {
        "Announcements": {
            "active": True,
            "input_params": {"from_date": "01-03-2025", "to_date": "31-03-2025"}
        },
        "BoardMeetings": {
            "active": True,
            "input_params": {"from_date": "01-03-2025", "to_date": "31-03-2025"}
        },
        "CorporateActions": {
            "active": True,
            "input_params": {}
        }
    }
    
    results = await nse_tool.fetch_data_from_multiple_streams(stream_dict)
    
    assert "Announcements" in results
    assert "BoardMeetings" in results
    assert "CorporateActions" in results
    assert len(results["Announcements"]) == 2
    assert len(results["BoardMeetings"]) == 2
    assert len(results["CorporateActions"]) == 2


@pytest.mark.asyncio
async def test_run_fetch_data(nse_tool):
    """Test the run method with fetch_data command."""
    result = await nse_tool.run(
        command="fetch_data",
        stream="Announcements",
        input_params={"from_date": "01-03-2025", "to_date": "31-03-2025"}
    )
    
    assert result.success is True
    assert len(result.data) == 2
    assert result.data[0]["desc"] == "Financial Results"


@pytest.mark.asyncio
async def test_run_fetch_multiple(nse_tool):
    """Test the run method with fetch_multiple command."""
    stream_dict = {
        "Announcements": {
            "active": True,
            "input_params": {"from_date": "01-03-2025", "to_date": "31-03-2025"}
        },
        "BoardMeetings": {
            "active": True,
            "input_params": {"from_date": "01-03-2025", "to_date": "31-03-2025"}
        }
    }
    
    result = await nse_tool.run(
        command="fetch_multiple",
        stream_dict=stream_dict
    )
    
    assert result.success is True
    assert "Announcements" in result.data
    assert "BoardMeetings" in result.data
    assert len(result.data["Announcements"]) == 2
    assert len(result.data["BoardMeetings"]) == 2


@pytest.mark.asyncio
async def test_run_get_streams(nse_tool):
    """Test the run method with get_streams command."""
    result = await nse_tool.run(command="get_streams")
    
    assert result.success is True
    assert isinstance(result.data, dict)
    assert "Announcements" in result.data
    assert "params" in result.data["Announcements"]
    assert "description" in result.data["Announcements"]


@pytest.mark.asyncio
async def test_run_get_announcements(nse_tool):
    """Test the run method with get_announcements command."""
    result = await nse_tool.run(
        command="get_announcements",
        params={"from_date": "01-03-2025", "to_date": "31-03-2025"}
    )
    
    assert result.success is True
    assert len(result.data) == 2
    assert "event_type" in result.data[0]
    assert "attachment" in result.data[0]
    assert "description" in result.data[0]
    assert "time" in result.data[0]


@pytest.mark.asyncio
async def test_run_unknown_command(nse_tool):
    """Test the run method with an unknown command."""
    result = await nse_tool.run(command="unknown_command")
    
    assert result.success is False
    assert "error" in result.__dict__
    assert "Unknown command" in result.error


@pytest.mark.asyncio
async def test_run_missing_required_param(nse_tool):
    """Test the run method with missing required parameter."""
    result = await nse_tool.run(command="fetch_data")
    
    assert result.success is False
    assert "error" in result.__dict__
    assert "Stream parameter is required" in result.error


@pytest.mark.asyncio
async def test_error_handling(nse_tool):
    """Test error handling."""
    # Force an error by making make_request return None
    nse_tool.make_request = AsyncMock(return_value=None)
    
    result = await nse_tool.run(
        command="fetch_data",
        stream="Announcements",
        input_params={}
    )
    
    assert result.success is True  # The method itself should not fail
    assert len(result.data) == 0  # But it should return an empty list


@pytest.mark.asyncio
async def test_construct_url(nse_tool):
    """Test URL construction."""
    url = nse_tool._construct_url("test-endpoint", {"param1": "value1", "param2": "value2"})
    
    assert "https://www.nseindia.com/api/test-endpoint" in url
    assert "param1=value1" in url
    assert "param2=value2" in url


@pytest.mark.asyncio
async def test_get_schema(nse_tool):
    """Test schema retrieval."""
    schema = nse_tool._get_schema("Announcements")
    
    assert "event_type" in schema
    assert "attachment" in schema
    assert "description" in schema
    assert "time" in schema
    assert schema["event_type"] == "desc"
    assert schema["attachment"] == "attchmntFile"