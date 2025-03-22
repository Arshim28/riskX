# tests/test_research_agent.py
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

from agents.research_agent import ResearchAgent
from tools.search_tool import SearchResult


@pytest.fixture
def config():
    return {
        "research": {
            "search_engine": "google",
            "max_results": 15
        },
        "models": {
            "planning": "gemini-2.0-flash"
        }
    }


@pytest.fixture
def state():
    return {
        "company": "Test Company",
        "industry": "Technology",
        "research_plan": [{"objective": "Test", "query_categories": {"category1": "query1"}}],
        "search_history": [],
        "search_type": "google_news",
        "return_type": "clustered"
    }


@pytest.fixture
def search_results():
    return [
        SearchResult(
            title="Test Article 1",
            link="https://example.com/1",
            snippet="Test snippet 1",
            source="Source 1",
            date="2023-01-01",
            is_quarterly_report=False
        ),
        SearchResult(
            title="Test Article 2",
            link="https://example.com/2",
            snippet="Test snippet 2",
            source="Source 2",
            date="2023-01-02",
            is_quarterly_report=True
        )
    ]


@pytest.fixture
def agent(config):
    with patch("utils.prompt_manager.get_prompt_manager"), \
         patch("utils.logging.get_logger"):
        return ResearchAgent(config)


@pytest.mark.asyncio
async def test_generate_queries(agent):
    # Directly use the mock response from test_utils.py
    from tests.test_utils import MOCK_LLM_RESPONSES
    
    with patch("utils.llm_provider.get_llm_provider") as mock_provider:
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = MOCK_LLM_RESPONSES["query_generation"]
        mock_provider.return_value = mock_llm
        
        result = await agent.generate_queries(
            "Test Company", 
            "Technology", 
            {"objective": "Test"}, 
            []
        )
        
        assert isinstance(result, dict)
        assert "financial" in result
        assert "regulatory" in result
        assert len(result["financial"]) > 0
        assert len(result["regulatory"]) > 0
        
        mock_llm.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_group_results(agent, search_results):
    # Directly use the mock response from test_utils.py
    from tests.test_utils import MOCK_LLM_RESPONSES
    
    with patch("utils.llm_provider.get_llm_provider") as mock_provider:
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = MOCK_LLM_RESPONSES["article_clustering"]
        mock_provider.return_value = mock_llm
        
        result = await agent.group_results("Test Company", search_results, "Technology")
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "Financial Irregularities Investigation (2023) - High" in result
        assert "Quarterly Financial Results (Q2 2023) - Low" in result
        
        mock_llm.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_run(agent, state):
    with patch.object(agent, "generate_queries", return_value={"category1": ["query1"]}), \
         patch.object(agent, "search_tool") as mock_search_tool, \
         patch.object(agent, "group_results", return_value={"Event 1": {"articles": [], "importance_score": 70, "article_count": 1}}):
        
        mock_search_result = MagicMock()
        mock_search_result.success = True
        mock_search_result.data = [
            MagicMock(model_dump=lambda: {"title": "Test", "link": "https://example.com", "snippet": "Test"})
        ]
        mock_search_tool.run.return_value = mock_search_result
        
        result = await agent.run(state)
        
        assert result["goto"] == "meta_agent"
        assert result["research_agent_status"] == "DONE"
        assert "research_results" in result
        assert "event_metadata" in result