import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from base.base_agents import AgentState

from agents.corporate_agent import CorporateAgent
from utils.llm_provider import get_llm_provider


@pytest.fixture
def config():
    return {
        "postgres": {},
        "nse": {},
        "models": {
            "lookup": "gemini-2.0-flash",
            "analysis": "gemini-2.0-pro",
            "report": "gemini-2.0-pro"
        }
    }


@pytest.fixture
def state():
    return {
        "company": "Test Company",
        "industry": "Technology"
    }


@pytest.fixture
def company_info():
    return {
        "name": "Test Company",
        "industry": "Technology",
        "founded": 2010,
        "headquarters": "San Francisco, CA",
        "ceo": "John Doe",
        "market_cap": "10B"
    }


@pytest.fixture
def financial_analysis():
    return {
        "financial_health": "Strong",
        "growth_trend": "Positive",
        "red_flags": ["Q2 profit margin declined slightly"],
        "key_metrics": {"profit_margin": "10%"},
        "risk_level": "Low",
        "recommendation": "Monitor Q3 performance"
    }


@pytest.fixture
def regulatory_filings():
    return [
        {
            "date": "2023-12-01",
            "type": "Annual Report",
            "summary": "Filed annual report with SEC",
            "url": "https://example.com/filing1"
        },
        {
            "date": "2023-09-15",
            "type": "Quarterly Report",
            "summary": "Filed Q3 results with SEC",
            "url": "https://example.com/filing2"
        }
    ]


@pytest.fixture
def regulatory_analysis():
    return {
        "compliance": "Good",
        "pending_issues": "None",
        "historical_violations": 0,
        "recent_changes": "N/A",
        "red_flags": ["Minor delay in Q2 filing"],
        "risk_assessment": "Low"
    }


@pytest.fixture
def management_info():
    return [
        {
            "name": "John Doe",
            "position": "CEO",
            "tenure": "5 years",
            "background": "Previously CEO at Tech Corp"
        },
        {
            "name": "Jane Smith",
            "position": "CFO",
            "tenure": "3 years",
            "background": "Previously CFO at Finance Inc"
        }
    ]


@pytest.fixture
def management_analysis():
    return {
        "leadership_stability": "High",
        "experience_level": "Very Experienced",
        "succession_planning": "Formal plan in place",
        "red_flags": ["CEO sold shares in Q2"],
        "risk_assessment": "Low"
    }


@pytest.fixture
def market_data():
    return {
        "price_data": [
            {"date": "2023-12-01", "price": 100.0},
            {"date": "2023-12-02", "price": 102.5},
            {"date": "2023-12-03", "price": 101.0}
        ],
        "unusual_patterns": [
            {"date": "2023-12-02", "pattern": "Sudden spike", "significance": "Medium"}
        ]
    }


@pytest.fixture
def market_analysis():
    return {
        "stock_performance": "Above average",
        "volatility": "Low",
        "correlation_to_sector": "High",
        "momentum": "Positive",
        "red_flags": ["Unusual volume on Dec 2nd"],
        "risk_assessment": "Low to Medium"
    }


@pytest.fixture
def corporate_report():
    return {
        "executive_summary": "Overall strong company with minor concerns",
        "financial_health": "Strong",
        "regulatory_status": "Compliant",
        "management_assessment": "Experienced team",
        "market_performance": "Above average",
        "risk_factors": ["Minor delays in filings", "CEO stock sales"],
        "conclusion": "Low risk profile with positive outlook",
        "recommendation": "Continue monitoring"
    }


@pytest.fixture
def agent(config):
    # Create proper tool mocks
    postgres_mock = MagicMock()
    nse_mock = MagicMock()
    
    # Create LLM provider mock
    llm_provider_instance = MagicMock()
    llm_provider_instance.generate_text = AsyncMock()
    llm_provider_mock = AsyncMock(return_value=llm_provider_instance)
    
    # Create prompt manager mock
    prompt_manager_mock = MagicMock()
    prompt_manager_mock.get_prompt = MagicMock(return_value=("System prompt", "Human prompt"))
    
    # Create logger mock
    logger_mock = MagicMock()
    
    # Use the constructor patches to return our configured mocks
    with patch("utils.prompt_manager.get_prompt_manager", return_value=prompt_manager_mock), \
         patch("utils.logging.get_logger", return_value=logger_mock), \
         patch("tools.postgres_tool.PostgresTool", return_value=postgres_mock), \
         patch("tools.nse_tool.NSETool", return_value=nse_mock), \
         patch("utils.llm_provider.get_llm_provider", llm_provider_mock):
        
        agent = CorporateAgent(config)
        
        # Add metrics attribute if missing
        metrics_mock = MagicMock()
        metrics_mock.execution_time_ms = 100.0
        agent.metrics = metrics_mock
        
        # Add method to agent to get the mocked LLM provider
        agent.get_llm_provider = llm_provider_mock
        
        return agent


@pytest.mark.asyncio
async def test_fetch_company_info(agent, company_info):
    # Create a new mock for the postgres tool
    postgres_mock = MagicMock()
    postgres_mock.run = AsyncMock()
    
    # Replace agent's postgres_tool with our mock
    agent.postgres_tool = postgres_mock
    
    # Test database hit
    db_result = MagicMock()
    db_result.success = True
    db_result.data = [company_info]
    postgres_mock.run.return_value = db_result
    
    result = await agent.fetch_company_info("Test Company")
    
    assert result == company_info
    assert agent.company_data == result
    
    postgres_mock.run.assert_called_once()
    
    # Reset for next test
    postgres_mock.run.reset_mock()
    
    # Test database miss, LLM generation
    db_result = MagicMock()
    db_result.success = False
    db_result.data = None
    postgres_mock.run.return_value = db_result
    
    # Set up LLM response
    llm_provider = await agent.get_llm_provider()
    llm_provider.generate_text.return_value = json.dumps(company_info)
    
    result = await agent.fetch_company_info("Test Company")
    
    assert result == company_info
    assert agent.company_data == result
    
    assert postgres_mock.run.call_count == 2  # First to check, second to save
    llm_provider.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_financial_statements(agent, financial_analysis):
    # Create a new mock for the nse tool
    nse_mock = MagicMock()
    nse_mock.run = AsyncMock()
    
    # Replace agent's nse_tool with our mock
    agent.nse_tool = nse_mock
    
    # Set up mock response
    nse_result = MagicMock()
    nse_result.success = True
    nse_result.data = {
        "quarters": [
            {"quarter": "Q1", "revenue": 1000, "profit": 100},
            {"quarter": "Q2", "revenue": 1100, "profit": 110}
        ]
    }
    nse_mock.run.return_value = nse_result
    
    # Get the LLM provider mock
    llm_provider = await agent.get_llm_provider()
    llm_provider.generate_text.return_value = json.dumps(financial_analysis)
    
    result = await agent.analyze_financial_statements("Test Company")
    
    assert result == financial_analysis
    assert agent.financial_data["raw_data"] == nse_result.data
    assert agent.financial_data["analysis"] == result
    
    nse_mock.run.assert_called_once()
    llm_provider.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_check_regulatory_filings(agent, regulatory_filings, regulatory_analysis):
    # Create new mocks for both tools
    postgres_mock = MagicMock()
    postgres_mock.run = AsyncMock()
    
    nse_mock = MagicMock()
    nse_mock.run = AsyncMock()
    
    # Replace agent's tools with our mocks
    agent.postgres_tool = postgres_mock
    agent.nse_tool = nse_mock
    
    # Test with existing filings in database
    db_result = MagicMock()
    db_result.success = True
    db_result.data = regulatory_filings
    postgres_mock.run.return_value = db_result
    
    # Get the LLM provider mock
    llm_provider = await agent.get_llm_provider()
    llm_provider.generate_text.return_value = json.dumps(regulatory_analysis)
    
    result = await agent.check_regulatory_filings("Test Company")
    
    assert result == regulatory_analysis
    assert agent.regulatory_data["filings"] == regulatory_filings
    assert agent.regulatory_data["analysis"] == result
    
    postgres_mock.run.assert_called_once()
    llm_provider.generate_text.assert_called_once()
    
    # Reset mocks
    postgres_mock.run.reset_mock()
    llm_provider.generate_text.reset_mock()
    
    # Test with fetching from NSE (fewer than 5 filings in DB)
    db_result.data = regulatory_filings[:1]  # Only one filing in DB
    postgres_mock.run.return_value = db_result
    
    nse_result = MagicMock()
    nse_result.success = True
    nse_result.data = regulatory_filings
    nse_mock.run.return_value = nse_result
    
    result = await agent.check_regulatory_filings("Test Company")
    
    assert result == regulatory_analysis
    assert agent.regulatory_data["filings"] == regulatory_filings
    assert agent.regulatory_data["analysis"] == result
    
    postgres_mock.run.assert_called()
    nse_mock.run.assert_called_once()
    llm_provider.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_management_team(agent, management_info, management_analysis):
    # Create a new mock for the postgres tool
    postgres_mock = MagicMock()
    postgres_mock.run = AsyncMock()
    
    # Replace agent's postgres_tool with our mock
    agent.postgres_tool = postgres_mock
    
    # Test with existing management info in database
    db_result = MagicMock()
    db_result.success = True
    db_result.data = management_info
    postgres_mock.run.return_value = db_result
    
    # Get the LLM provider mock
    llm_provider = await agent.get_llm_provider()
    llm_provider.generate_text.return_value = json.dumps(management_analysis)
    
    result = await agent.analyze_management_team("Test Company")
    
    assert result == management_analysis
    assert agent.management_data["management"] == management_info
    assert agent.management_data["analysis"] == result
    
    postgres_mock.run.assert_called_once()
    llm_provider.generate_text.assert_called_once()
    
    # Reset mocks
    postgres_mock.run.reset_mock()
    llm_provider.generate_text.reset_mock()
    
    # Test with no management info in database
    db_result.success = True
    db_result.data = None
    postgres_mock.run.return_value = db_result
    
    llm_provider.generate_text.side_effect = [
        json.dumps(management_info),  # First call returns management info
        json.dumps(management_analysis)  # Second call returns analysis
    ]
    
    result = await agent.analyze_management_team("Test Company")
    
    assert result == management_analysis
    assert agent.management_data["management"] == management_info
    assert agent.management_data["analysis"] == result
    
    assert postgres_mock.run.call_count >= 2  # First to check, then to save
    assert llm_provider.generate_text.call_count == 2


@pytest.mark.asyncio
async def test_analyze_market_data(agent, market_data, market_analysis):
    # Create a new mock for the nse tool
    nse_mock = MagicMock()
    nse_mock.run = AsyncMock()
    
    # Replace agent's nse_tool with our mock
    agent.nse_tool = nse_mock
    
    # Mock price result
    price_result = MagicMock()
    price_result.success = True
    price_result.data = market_data["price_data"]
    
    # Mock pattern result
    pattern_result = MagicMock()
    pattern_result.success = True
    pattern_result.data = market_data["unusual_patterns"]
    
    # Configure NSE tool to return different results based on command
    def mock_nse_run(command, **kwargs):
        if command == "get_stock_price_history":
            return price_result
        elif command == "detect_unusual_patterns":
            return pattern_result
        return MagicMock(success=False)
        
    nse_mock.run.side_effect = mock_nse_run
    
    # Get the LLM provider mock
    llm_provider = await agent.get_llm_provider()
    llm_provider.generate_text.return_value = json.dumps(market_analysis)
    
    result = await agent.analyze_market_data("Test Company")
    
    assert result == market_analysis
    assert agent.market_data["price_data"] == market_data["price_data"]
    assert agent.market_data["unusual_patterns"] == market_data["unusual_patterns"]
    assert agent.market_data["analysis"] == result
    
    assert nse_mock.run.call_count == 2  # Called for price data and patterns
    llm_provider.generate_text.assert_called_once()
    
    # Test error handling
    nse_mock.run.reset_mock()
    llm_provider.generate_text.reset_mock()
    
    price_result.success = False
    price_result.error = "Connection error"
    
    result = await agent.analyze_market_data("Test Company")
    
    assert "error" in result
    assert "Failed to fetch market data" in result["error"]
    
    nse_mock.run.assert_called_once()
    llm_provider.generate_text.assert_not_called()


@pytest.mark.asyncio
async def test_generate_corporate_report(agent, corporate_report):
    # Create a new mock for the postgres tool
    postgres_mock = MagicMock()
    postgres_mock.run = AsyncMock()
    
    # Replace agent's postgres_tool with our mock
    agent.postgres_tool = postgres_mock
    
    # Set up agent data
    agent.company_data = {"name": "Test Company"}
    agent.financial_data = {"analysis": {"financial_health": "Strong"}}
    agent.regulatory_data = {"analysis": {"compliance": "Good"}}
    agent.management_data = {"analysis": {"leadership_stability": "High"}}
    agent.market_data = {"analysis": {"stock_performance": "Above average"}}
    
    # Get the LLM provider mock
    llm_provider = await agent.get_llm_provider()
    llm_provider.generate_text.return_value = json.dumps(corporate_report)
    
    postgres_mock.run.return_value = MagicMock(success=True)
    
    result = await agent.generate_corporate_report("Test Company")
    
    assert result == corporate_report
    
    llm_provider.generate_text.assert_called_once()
    postgres_mock.run.assert_called_once()  # To save the report
    
    # Input to LLM should include all analysis data
    args, kwargs = llm_provider.generate_text.call_args
    assert "financial_health" in str(args)
    assert "compliance" in str(args)
    assert "leadership_stability" in str(args)
    assert "stock_performance" in str(args)


@pytest.mark.asyncio
async def test_run(agent, state, company_info, financial_analysis, regulatory_analysis, 
                  management_analysis, market_analysis, corporate_report):
    # Create method mocks
    agent.fetch_company_info = AsyncMock(return_value=company_info)
    agent.analyze_financial_statements = AsyncMock(return_value=financial_analysis)
    agent.check_regulatory_filings = AsyncMock(return_value=regulatory_analysis)
    agent.analyze_management_team = AsyncMock(return_value=management_analysis)
    agent.analyze_market_data = AsyncMock(return_value=market_analysis)
    agent.generate_corporate_report = AsyncMock(return_value=corporate_report)
    agent._log_start = MagicMock()
    agent._log_completion = MagicMock()
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["corporate_status"] == "DONE"
    assert "corporate_results" in result
    assert result["corporate_results"]["company_info"] == company_info
    assert result["corporate_results"]["financial_analysis"] == financial_analysis
    assert result["corporate_results"]["regulatory_analysis"] == regulatory_analysis
    assert result["corporate_results"]["management_analysis"] == management_analysis
    assert result["corporate_results"]["market_analysis"] == market_analysis
    assert result["corporate_results"]["corporate_report"] == corporate_report
    assert len(result["corporate_results"]["red_flags"]) > 0
    
    # Verify all methods were called
    agent.fetch_company_info.assert_called_once()
    agent.analyze_financial_statements.assert_called_once()
    agent.check_regulatory_filings.assert_called_once()
    agent.analyze_management_team.assert_called_once()
    agent.analyze_market_data.assert_called_once()
    agent.generate_corporate_report.assert_called_once()
    
    # Test synchronous pipeline behavior
    state["synchronous_pipeline"] = True
    state["next_agent"] = "next_test_agent"
    
    result = await agent.run(state)
    
    assert result["goto"] == "next_test_agent"


@pytest.mark.asyncio
async def test_execute_method(agent, state):
    # Create an AgentState object from the state dict
    agent_state = AgentState(**state)
    
    # Mock the run method
    agent.run = AsyncMock(return_value={"result": "success"})
    
    # Call _execute
    result = await agent._execute(agent_state)
    
    # Verify result
    assert result["result"] == "success"
    assert agent.run.called
    
    # Verify run was called with the correct state
    args, kwargs = agent.run.call_args
    assert args[0]["company"] == state["company"]
    assert args[0]["industry"] == state["industry"]


@pytest.mark.asyncio
async def test_error_handling(agent, state):
    # Test missing company name
    state_without_company = state.copy()
    state_without_company.pop("company")
    
    result = await agent.run(state_without_company)
    
    assert result["goto"] == "meta_agent"
    assert result["corporate_status"] == "ERROR"
    assert "error" in result
    
    # Test exception in a method
    agent.fetch_company_info = AsyncMock(side_effect=Exception("Test error"))
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["corporate_status"] == "ERROR"
    assert "error" in result
    assert "Test error" in result["error"]