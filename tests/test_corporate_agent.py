# tests/test_corporate_agent.py
import pytest
import asyncio
import json
import os
import yaml
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from base.base_agents import AgentState

from agents.corporate_agent import CorporateAgent, CorporateGovernanceError


@pytest.fixture
def config():
    return {
        "postgres": {},
        "nse": {},
        "models": {
            "lookup": "gemini-2.0-flash",
            "analysis": "gemini-2.0-pro",
            "report": "gemini-2.0-pro"
        },
        "retry": {
            "max_attempts": 2,
            "multiplier": 0,
            "min_wait": 0,
            "max_wait": 0
        }
    }


@pytest.fixture
def state():
    return {
        "company": "Test Company",
        "industry": "Technology"
    }


@pytest.fixture
def governance_data():
    return {
        "TEST": {
            "board_of_directors": [
                {
                    "director_name": "John Doe",
                    "din": "12345678",
                    "designation": "Chairperson",
                    "tenure": "5",
                    "membership": ["Audit Committee"]
                },
                {
                    "director_name": "Jane Smith",
                    "din": "87654321",
                    "designation": "Independent Director",
                    "tenure": "3",
                    "membership": ["Nomination & Remuneration Committee"]
                }
            ],
            "communities": {
                "Audit Committee": [
                    {
                        "name": "John Doe",
                        "designation": "Independent Director",
                        "community_designation": "Chairperson"
                    }
                ],
                "Nomination & Remuneration Committee": [
                    {
                        "name": "Jane Smith",
                        "designation": "Independent Director",
                        "community_designation": "Member"
                    }
                ]
            }
        }
    }


@pytest.fixture
def stream_config():
    return {
        "Announcements": {
            "active": True,
            "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
        },
        "BoardMeetings": {
            "active": True,
            "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
        }
    }


@pytest.fixture
def default_stream_config():
    return {
        "Announcements": {
            "active": True,
            "input_params": {"from_date": "", "to_date": ""}
        },
        "BoardMeetings": {
            "active": True,
            "input_params": {"from_date": "", "to_date": ""}
        },
        "CorporateActions": {
            "active": True,
            "input_params": {}
        }
    }


@pytest.fixture
def nse_result_data():
    return {
        "Announcements": [
            {"desc": "Financial Results", "attchmntFile": "results.pdf", "attchmntText": "Q2 Results", "exchdisstime": "01-01-2023 10:00:00"}
        ],
        "BoardMeetings": [
            {"bm_date": "15-01-2023", "bm_purpose": "TEST", "bm_desc": "Quarterly Meeting", "attachment": "agenda.pdf"}
        ]
    }


@pytest.fixture
def agent(config):
    # Create proper tool mocks
    postgres_mock = MagicMock()
    postgres_mock.run = AsyncMock()
    
    nse_mock = MagicMock()
    nse_mock.run = AsyncMock()
    
    # Create prompt manager mock
    prompt_manager_mock = MagicMock()
    prompt_manager_mock.get_prompt = MagicMock()
    
    # Create logger mock
    logger_mock = MagicMock()
    
    # Use the constructor patches to return our configured mocks
    with patch("utils.prompt_manager.get_prompt_manager", return_value=prompt_manager_mock), \
         patch("utils.logging.get_logger", return_value=logger_mock), \
         patch("tools.postgres_tool.PostgresTool", return_value=postgres_mock), \
         patch("tools.nse_tool.NSETool", return_value=nse_mock):
        
        agent = CorporateAgent(config)
        
        # Add metrics attribute if missing
        metrics_mock = MagicMock()
        metrics_mock.execution_time_ms = 100.0
        agent.metrics = metrics_mock
        
        return agent


@pytest.mark.asyncio
async def test_get_company_symbol_from_config(agent):
    """Test getting company symbol from NSE tool config"""
    # Mock the NSE tool config
    agent.nse_tool.config = MagicMock()
    agent.nse_tool.config.symbol = "TEST"
    
    symbol = await agent.get_company_symbol("Test Company")
    
    assert symbol == "TEST"
    # Verify postgres_tool.run was not called
    assert not agent.postgres_tool.run.called


@pytest.mark.asyncio
async def test_get_company_symbol_from_db(agent):
    """Test getting company symbol from database"""
    # Replace config with one that doesn't have symbol
    agent.nse_tool.config = MagicMock()
    agent.nse_tool.config.symbol = None
    
    # Mock database response
    db_result = MagicMock()
    db_result.success = True
    db_result.data = [{"symbol": "TEST"}]
    agent.postgres_tool.run.return_value = db_result
    
    symbol = await agent.get_company_symbol("Test Company")
    
    assert symbol == "TEST"
    agent.postgres_tool.run.assert_called_once()


@pytest.mark.asyncio
async def test_get_company_symbol_fallback(agent):
    """Test fallback when symbol not found"""
    # Replace config with one that doesn't have symbol
    agent.nse_tool.config = MagicMock()
    agent.nse_tool.config.symbol = None
    
    # Mock database with no results
    db_result = MagicMock()
    db_result.success = True
    db_result.data = []
    agent.postgres_tool.run.return_value = db_result
    
    symbol = await agent.get_company_symbol("Test Company")
    
    assert symbol == "Test Company"  # Fallback to company name
    agent.postgres_tool.run.assert_called_once()


@pytest.mark.asyncio
async def test_get_corporate_governance_data(agent, governance_data):
    """Test getting corporate governance data from file"""
    # Mock open and json.load
    with patch("builtins.open", mock_open()), \
         patch("os.path.exists", return_value=True), \
         patch("json.load", return_value=governance_data):
        
        result = await agent.get_corporate_governance_data("TEST")
        
        assert result == governance_data["TEST"]


@pytest.mark.asyncio
async def test_get_corporate_governance_data_not_found(agent):
    """Test governance data not found for symbol"""
    # Mock open and json.load
    with patch("builtins.open", mock_open()), \
         patch("os.path.exists", return_value=True), \
         patch("json.load", return_value={"OTHER": {}}):
        
        result = await agent.get_corporate_governance_data("TEST")
        
        assert result == {}


@pytest.mark.asyncio
async def test_get_corporate_governance_data_file_not_found(agent):
    """Test governance data file not found"""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(CorporateGovernanceError):
            result = await agent.get_corporate_governance_data("TEST")


@pytest.mark.asyncio
async def test_collect_corporate_data_success(agent, nse_result_data, governance_data, stream_config):
    """Test successful corporate data collection"""
    # Mock get_company_symbol
    agent.get_company_symbol = AsyncMock(return_value="TEST")
    
    # Mock get_corporate_governance_data
    agent.get_corporate_governance_data = AsyncMock(return_value=governance_data["TEST"])
    
    # Mock NSE tool response
    nse_result = MagicMock()
    nse_result.success = True
    nse_result.data = nse_result_data
    agent.nse_tool.run.return_value = nse_result
    
    result = await agent.collect_corporate_data("Test Company", stream_config)
    
    assert result["success"] is True
    assert result["company"] == "Test Company"
    assert result["symbol"] == "TEST"
    assert result["governance"] == governance_data["TEST"]
    assert result["data"] == nse_result_data
    assert "timestamp" in result
    assert "summary" in result
    assert result["summary"]["total_streams"] == 2
    assert result["summary"]["stream_counts"]["Announcements"] == 1
    assert result["summary"]["stream_counts"]["BoardMeetings"] == 1


@pytest.mark.asyncio
async def test_collect_corporate_data_nse_failure(agent, stream_config):
    """Test handling NSE tool failure"""
    # Mock get_company_symbol
    agent.get_company_symbol = AsyncMock(return_value="TEST")
    
    # Mock get_corporate_governance_data
    agent.get_corporate_governance_data = AsyncMock(return_value={})
    
    # Mock NSE tool failure
    nse_result = MagicMock()
    nse_result.success = False
    nse_result.error = "NSE API Error"
    agent.nse_tool.run.return_value = nse_result
    
    result = await agent.collect_corporate_data("Test Company", stream_config)
    
    assert result["success"] is False
    assert "error" in result
    assert "NSE API Error" in result["error"]
    assert result["company"] == "Test Company"
    assert result["symbol"] == "TEST"
    assert "governance" in result
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_get_default_stream_config(agent, default_stream_config):
    """Test loading default stream config"""
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open()), \
         patch("yaml.safe_load", return_value=default_stream_config):
        
        result = agent._get_default_stream_config()
        
        assert result == default_stream_config
        # Check that date parameters were updated for streams that need them
        for stream, config in result.items():
            if 'input_params' in config and config['input_params'] and any(param in config['input_params'] for param in ['from_date', 'to_date']):
                assert config['input_params'] == agent.default_date_params


@pytest.mark.asyncio
async def test_get_default_stream_config_file_not_found(agent):
    """Test handling config file not found"""
    with patch("os.path.exists", return_value=False):
        result = agent._get_default_stream_config()
        assert result == {}


@pytest.mark.asyncio
async def test_run_success(agent, state, nse_result_data, governance_data, stream_config):
    """Test successful run method"""
    # Mock agent methods
    agent._get_default_stream_config = MagicMock(return_value=stream_config)
    agent.collect_corporate_data = AsyncMock(return_value={
        "success": True,
        "company": "Test Company",
        "symbol": "TEST",
        "timestamp": "2023-01-01T00:00:00",
        "governance": governance_data["TEST"],
        "data": nse_result_data,
        "summary": {
            "total_streams": 2,
            "stream_counts": {"Announcements": 1, "BoardMeetings": 1}
        }
    })
    agent._log_start = MagicMock()
    agent._log_completion = MagicMock()
    
    # Create NSE tool config
    agent.nse_tool.config = MagicMock()
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["corporate_status"] == "DONE"
    assert "corporate_results" in result
    assert result["corporate_results"]["success"] is True
    assert result["corporate_results"]["company"] == "Test Company"
    
    # Verify config was updated
    assert agent.nse_tool.config.company == "Test Company"


@pytest.mark.asyncio
async def test_run_with_symbol_in_state(agent, state):
    """Test run with symbol provided in state"""
    # Add symbol to state
    state["symbol"] = "TEST"
    
    # Mock agent methods
    agent._get_default_stream_config = MagicMock(return_value={})
    agent.collect_corporate_data = AsyncMock(return_value={"success": True})
    agent._log_start = MagicMock()
    agent._log_completion = MagicMock()
    
    # Create NSE tool config
    agent.nse_tool.config = MagicMock()
    
    result = await agent.run(state)
    
    # Verify symbol was set in config
    assert agent.nse_tool.config.company == "Test Company"
    assert agent.nse_tool.config.symbol == "TEST"


@pytest.mark.asyncio
async def test_run_missing_company(agent):
    """Test run with missing company name"""
    state = {"industry": "Technology"}  # No company name
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["corporate_status"] == "ERROR"
    assert "error" in result
    assert "Company name is missing" in result["error"]


@pytest.mark.asyncio
async def test_run_collect_data_failure(agent, state):
    """Test run with failure in collect_corporate_data"""
    # Mock agent methods
    agent._get_default_stream_config = MagicMock(return_value={})
    agent.collect_corporate_data = AsyncMock(return_value={
        "success": False,
        "error": "Failed to collect data",
        "company": "Test Company",
        "timestamp": "2023-01-01T00:00:00"
    })
    agent._log_start = MagicMock()
    agent._log_completion = MagicMock()
    
    # Create NSE tool config
    agent.nse_tool.config = MagicMock()
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["corporate_status"] == "ERROR"
    assert "error" in result
    assert "Failed to collect data" in result["error"]


@pytest.mark.asyncio
async def test_run_synchronous_pipeline(agent, state):
    """Test run in synchronous pipeline mode"""
    # Set synchronous pipeline flag and next agent
    state["synchronous_pipeline"] = True
    state["next_agent"] = "next_test_agent"
    
    # Mock agent methods
    agent._get_default_stream_config = MagicMock(return_value={})
    agent.collect_corporate_data = AsyncMock(return_value={"success": True})
    agent._log_start = MagicMock()
    agent._log_completion = MagicMock()
    
    # Create NSE tool config
    agent.nse_tool.config = MagicMock()
    
    result = await agent.run(state)
    
    assert result["goto"] == "next_test_agent"
    assert result["corporate_status"] == "DONE"


@pytest.mark.asyncio
async def test_run_unhandled_error(agent, state):
    """Test handling of unhandled exceptions"""
    # Force an error in the run method
    agent._log_start = MagicMock(side_effect=Exception("Unexpected error"))
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["corporate_status"] == "ERROR"
    assert "error" in result
    assert "Unexpected error" in result["error"]


@pytest.mark.asyncio
async def test_execute_method(agent, state):
    """Test the _execute method"""
    # Create an AgentState object
    from base.base_agents import AgentState
    agent_state = AgentState(**state)
    
    # Mock the run method
    agent.run = AsyncMock(return_value={"result": "success"})
    
    # Call _execute
    result = await agent._execute(agent_state)
    
    # Verify result
    assert result["result"] == "success"
    agent.run.assert_called_once()
    
    # Verify run was called with the correct state
    args, kwargs = agent.run.call_args
    assert args[0]["company"] == state["company"]
    assert args[0]["industry"] == state["industry"]