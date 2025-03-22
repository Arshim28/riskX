# tests/test_corporate_agent.py
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

from agents.corporate_agent import CorporateAgent


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
def agent(config):
    with patch("utils.prompt_manager.get_prompt_manager"), \
         patch("utils.logging.get_logger"), \
         patch("tools.postgres_tool.PostgresTool"), \
         patch("tools.nse_tool.NSETool"):
        return CorporateAgent(config)


@pytest.mark.asyncio
async def test_fetch_company_info(agent):
    # Test database hit
    with patch.object(agent, "postgres_tool") as mock_postgres:
        db_result = MagicMock()
        db_result.success = True
        db_result.data = {"name": "Test Company", "industry": "Technology"}
        mock_postgres.run = AsyncMock(return_value=db_result)
        
        result = await agent.fetch_company_info("Test Company")
        
        assert result["name"] == "Test Company"
        assert result["industry"] == "Technology"
        assert agent.company_data == result
        
        mock_postgres.run.assert_called_once()
    
    # Test database miss, LLM generation
    with patch.object(agent, "postgres_tool") as mock_postgres, \
         patch("utils.llm_provider.get_llm_provider") as mock_provider:
        
        db_result = MagicMock()
        db_result.success = False
        db_result.data = None
        mock_postgres.run = AsyncMock(return_value=db_result)
        
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = '{"name": "Test Company", "industry": "Technology", "founded": 2010}'
        mock_provider.return_value = mock_llm
        
        result = await agent.fetch_company_info("Test Company")
        
        assert result["name"] == "Test Company"
        assert result["industry"] == "Technology"
        assert result["founded"] == 2010
        assert agent.company_data == result
        
        assert mock_postgres.run.call_count == 2  # First to check, second to save
        mock_llm.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_financial_statements(agent):
    with patch.object(agent, "nse_tool") as mock_nse, \
         patch("utils.llm_provider.get_llm_provider") as mock_provider:
        
        nse_result = MagicMock()
        nse_result.success = True
        nse_result.data = {
            "quarters": [
                {"quarter": "Q1", "revenue": 1000, "profit": 100},
                {"quarter": "Q2", "revenue": 1100, "profit": 110}
            ]
        }
        mock_nse.run = AsyncMock(return_value=nse_result)
        
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = '''
        {
            "financial_health": "Strong",
            "growth_trend": "Positive",
            "red_flags": ["Q2 profit margin declined slightly"],
            "key_metrics": {"profit_margin": "10%"}
        }
        '''
        mock_provider.return_value = mock_llm
        
        result = await agent.analyze_financial_statements("Test Company")
        
        assert result["financial_health"] == "Strong"
        assert result["growth_trend"] == "Positive"
        assert len(result["red_flags"]) == 1
        assert "profit_margin" in result["key_metrics"]
        assert agent.financial_data["raw_data"] == nse_result.data
        assert agent.financial_data["analysis"] == result
        
        mock_nse.run.assert_called_once()
        mock_llm.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_run(agent, state):
    with patch.object(agent, "fetch_company_info") as mock_company, \
         patch.object(agent, "analyze_financial_statements") as mock_financial, \
         patch.object(agent, "check_regulatory_filings") as mock_regulatory, \
         patch.object(agent, "analyze_management_team") as mock_management, \
         patch.object(agent, "analyze_market_data") as mock_market, \
         patch.object(agent, "generate_corporate_report") as mock_report:
        
        # Set up mock returns
        company_info = {"name": "Test Company", "industry": "Technology"}
        mock_company.return_value = company_info
        
        financial_analysis = {
            "financial_health": "Strong",
            "red_flags": ["Flag 1"]
        }
        mock_financial.return_value = financial_analysis
        
        regulatory_analysis = {
            "compliance": "Good",
            "red_flags": ["Flag 2"]
        }
        mock_regulatory.return_value = regulatory_analysis
        
        management_analysis = {
            "leadership": "Experienced",
            "red_flags": ["Flag 3"]
        }
        mock_management.return_value = management_analysis
        
        market_analysis = {
            "stock_performance": "Above average",
            "red_flags": ["Flag 4"]
        }
        mock_market.return_value = market_analysis
        
        corporate_report = {
            "executive_summary": "Overall strong company with some concerns",
            "detailed_analysis": "...",
            "conclusion": "Recommend further investigation"
        }
        mock_report.return_value = corporate_report
        
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
        assert len(result["corporate_results"]["red_flags"]) == 4
        
        mock_company.assert_called_once()
        mock_financial.assert_called_once()
        mock_regulatory.assert_called_once()
        mock_management.assert_called_once()
        mock_market.assert_called_once()
        mock_report.assert_called_once()