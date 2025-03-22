import pytest
import pytest_asyncio
import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Tuple, Optional

from agents.analyst_agent import AnalystAgent
from tools.content_parser_tool import ContentParserTool
from base.base_tools import ToolResult
from utils.prompt_manager import PromptManager, get_prompt_manager
from utils.llm_provider import LLMProvider, get_llm_provider


# Mock data for tests
COMPANY = "Test Company"
EVENT_NAME = "Test Event"
ARTICLE_INFO = {
    "title": "Test Article",
    "link": "https://example.com/article",
    "snippet": "This is a test snippet",
    "source": "Test Source",
    "date": "2025-03-20"
}
MOCK_CONTENT = "This is mock article content for testing forensic analysis."
MOCK_METADATA = {
    "source_domain": "example.com",
    "fetch_timestamp": "2025-03-20T12:00:00",
    "fetch_method": "test",
    "content_size": len(MOCK_CONTENT),
    "extraction_success": True
}
MOCK_INSIGHTS = {
    "ALLEGATIONS": "Test allegations",
    "ENTITIES": "Test entities",
    "TIMELINE": "2025-03-20",
    "MAGNITUDE": "Unknown",
    "EVIDENCE": "Test evidence",
    "RESPONSE": "Test response",
    "STATUS": "Ongoing",
    "CREDIBILITY": "Medium"
}
MOCK_SYNTHESIS = {
    "cross_validation": "Test validation",
    "timeline": [{"date": "2025-03-20", "description": "Test event"}],
    "key_entities": [{"name": "Test Entity", "role": "Test Role"}],
    "evidence_assessment": "Test assessment",
    "severity_assessment": "Medium",
    "credibility_score": 7,
    "red_flags": ["Test red flag"],
    "narrative": "Test narrative"
}
MOCK_COMPANY_ANALYSIS = {
    "executive_summary": "Test summary",
    "risk_assessment": {
        "financial_integrity_risk": "Low",
        "legal_regulatory_risk": "Medium",
        "reputational_risk": "Low",
        "operational_risk": "Low"
    },
    "key_patterns": ["Test pattern"],
    "critical_entities": [{"name": "Test Entity", "role": "Test Role"}],
    "red_flags": ["Test red flag"],
    "timeline": [{"date": "2025-03-20", "description": "Test event"}],
    "forensic_assessment": "No significant issues identified",
    "report_markdown": "# Test Report\n\nTest content."
}


@pytest.fixture(scope="module", autouse=True)
def setup_llm_provider():
    """Initialize the LLM Provider once for all tests"""
    # Store the original provider instance
    import utils.llm_provider
    original_instance = utils.llm_provider._provider_instance
    
    # Create a mock provider
    mock_provider = AsyncMock(spec=LLMProvider)
    
    # Initialize the global provider with our mock
    utils.llm_provider._provider_instance = mock_provider
    
    # This is yielded to tests that need it
    yield mock_provider
    
    # Restore the original provider instance after all tests
    utils.llm_provider._provider_instance = original_instance


@pytest_asyncio.fixture
async def mock_logger():
    """Fixture for mocking the logger"""
    mock_log = MagicMock(spec=logging.Logger)
    
    with patch("utils.logging.get_logger") as mock_get_logger:
        mock_get_logger.return_value = mock_log
        yield mock_log


@pytest_asyncio.fixture
async def mock_content_parser():
    """Fixture for mocking the ContentParserTool"""
    mock_parser = MagicMock(spec=ContentParserTool)
    
    # Set up the mock run method
    parser_result = ToolResult(
        success=True, 
        data={
            "url": ARTICLE_INFO["link"],
            "content": MOCK_CONTENT,
            "metadata": MOCK_METADATA,
            "success": True
        }
    )
    mock_parser.run = AsyncMock(return_value=parser_result)
    
    yield mock_parser


@pytest_asyncio.fixture
async def mock_prompt_manager():
    """Fixture for mocking the PromptManager"""
    mock_manager = MagicMock(spec=PromptManager)
    mock_manager.get_prompt = MagicMock(return_value=("mocked system prompt", "mocked human prompt"))
    
    # Store original function to restore later
    import utils.prompt_manager
    original_get_pm = utils.prompt_manager.get_prompt_manager
    
    # Replace with our mock
    utils.prompt_manager.get_prompt_manager = MagicMock(return_value=mock_manager)
    
    yield mock_manager
    
    # Restore original function
    utils.prompt_manager.get_prompt_manager = original_get_pm


@pytest_asyncio.fixture
async def analyst_agent(mock_content_parser, mock_logger, setup_llm_provider, mock_prompt_manager):
    """Fixture to create an AnalystAgent with mocked dependencies"""
    with patch.object(ContentParserTool, "__init__", return_value=None):
        with patch.object(AnalystAgent, "__init__", return_value=None):
            agent = AnalystAgent()
            
            # Set the mocked properties
            agent.logger = mock_logger
            agent.content_parser_tool = mock_content_parser
            agent.prompt_manager = mock_prompt_manager
            agent.config = {
                "forensic_analysis": {
                    "model": "test-model"
                }
            }
            agent.name = "analyst_agent"
            
            # Initialize knowledge_base and processing_stats
            agent.knowledge_base = {
                "events": {},          
                "entities": {},        
                "relationships": {},   
                "patterns": {},        
                "red_flags": [],       
                "evidence": {},        
                "timeline": [],        
                "sources": {},         
                "metadata": {}         
            }
            agent.processing_stats = {
                "total_events": 0,
                "total_articles": 0,
                "processed_articles": 0,
                "articles_with_insights": 0,
                "events_with_insights": 0,
                "failed_articles": 0
            }
            
            # Add _log_start and _log_completion methods for safety
            agent._log_start = MagicMock(return_value=None)
            agent._log_completion = MagicMock(return_value=None)
            
            yield agent


@pytest.mark.asyncio
async def test_process_article(analyst_agent):
    """Test that an article is correctly processed and insights are extracted"""
    # Create a direct mock for the extract_forensic_insights method
    with patch.object(analyst_agent, "extract_forensic_insights") as mock_extract:
        # Set up the mock to return our test insights
        mock_extract.return_value = {**MOCK_INSIGHTS, "article_title": ARTICLE_INFO["title"], "event_category": EVENT_NAME}
        
        # Call the method
        result = await analyst_agent.process_article(
            COMPANY, EVENT_NAME, ARTICLE_INFO, 0, 1
        )
        
        # Verify interactions and results
        analyst_agent.content_parser_tool.run.assert_called_once_with(url=ARTICLE_INFO["link"])
        mock_extract.assert_called_once()
        
        assert result is not None
        assert result["article_title"] == ARTICLE_INFO["title"]
        assert result["event_category"] == EVENT_NAME
        assert result["url"] == ARTICLE_INFO["link"]
        assert result["metadata"] == MOCK_METADATA
        
        # Check that stats were updated
        assert analyst_agent.processing_stats["processed_articles"] == 1
        assert analyst_agent.processing_stats["articles_with_insights"] == 1


@pytest.mark.asyncio
async def test_process_article_no_insights(analyst_agent, setup_llm_provider):
    """Test handling of articles without forensic insights"""
    # Setup LLM response to indicate no forensic content
    setup_llm_provider.generate_text.return_value = "NO_FORENSIC_CONTENT"
    
    # Call the method
    result = await analyst_agent.process_article(
        COMPANY, EVENT_NAME, ARTICLE_INFO, 0, 1
    )
    
    # Verify results
    assert result is None
    
    # Check that stats were updated correctly
    assert analyst_agent.processing_stats["processed_articles"] == 1
    assert analyst_agent.processing_stats["articles_with_insights"] == 0


@pytest.mark.asyncio
async def test_process_article_fetch_failure(analyst_agent, setup_llm_provider):
    """Test handling of articles that can't be fetched"""
    # Make content parser fail
    analyst_agent.content_parser_tool.run.return_value = ToolResult(
        success=False,
        error="Failed to fetch content"
    )
    
    # Call the method
    result = await analyst_agent.process_article(
        COMPANY, EVENT_NAME, ARTICLE_INFO, 0, 1
    )
    
    # Verify results
    assert result is None
    
    # Check that stats were updated correctly
    assert analyst_agent.processing_stats["processed_articles"] == 1
    assert analyst_agent.processing_stats["failed_articles"] == 1


@pytest.mark.asyncio
async def test_synthesize_event_insights(analyst_agent):
    """Test synthesis of multiple insights into a coherent event analysis"""
    # Create direct mock for the LLM provider
    mock_llm = AsyncMock()
    mock_llm.generate_text = AsyncMock(return_value=json.dumps(MOCK_SYNTHESIS))
    
    # Direct patch to get_llm_provider
    with patch('agents.analyst_agent.get_llm_provider', return_value=mock_llm):
        # Create mock insights
        insights = [
            {**MOCK_INSIGHTS, "article_title": "Article 1", "url": "https://example.com/1"},
            {**MOCK_INSIGHTS, "article_title": "Article 2", "url": "https://example.com/2"}
        ]
        
        # Call the method
        result = await analyst_agent.synthesize_event_insights(
            COMPANY, EVENT_NAME, insights
        )
        
        # Verify calls
        mock_llm.generate_text.assert_called_once()
        
        # Compare individual items with expected MOCK_SYNTHESIS
        for key, value in MOCK_SYNTHESIS.items():
            assert key in result, f"Key '{key}' missing from result"
            assert result[key] == value, f"Values for key '{key}' don't match"


@pytest.mark.asyncio
async def test_generate_company_analysis(analyst_agent):
    """Test generation of comprehensive company analysis"""
    # Create direct mock for the LLM provider
    mock_llm = AsyncMock()
    mock_llm.generate_text = AsyncMock(return_value=json.dumps(MOCK_COMPANY_ANALYSIS))
    
    # Direct patch to get_llm_provider
    with patch('agents.analyst_agent.get_llm_provider', return_value=mock_llm):
        # Create mock event synthesis
        event_synthesis = {
            "Event 1": MOCK_SYNTHESIS,
            "Event 2": MOCK_SYNTHESIS
        }
        
        # Call the method
        result = await analyst_agent.generate_company_analysis(
            COMPANY, event_synthesis
        )
        
        # Verify calls
        mock_llm.generate_text.assert_called_once()
        
        # Compare with expected result
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Verify all keys from the mock exist in the result
        for key in MOCK_COMPANY_ANALYSIS.keys():
            assert key in result, f"Expected key '{key}' missing from result"
            
        # Check specific key/value pairs
        assert result["executive_summary"] == MOCK_COMPANY_ANALYSIS["executive_summary"]
        assert result["risk_assessment"]["financial_integrity_risk"] == MOCK_COMPANY_ANALYSIS["risk_assessment"]["financial_integrity_risk"]
        assert result["report_markdown"] == MOCK_COMPANY_ANALYSIS["report_markdown"]


@pytest.mark.asyncio
async def test_run_basic_flow(analyst_agent, setup_llm_provider):
    """Test the basic flow of the run method"""
    # Setup mocks
    with patch.object(analyst_agent, "process_article") as mock_process_article, \
         patch.object(analyst_agent, "synthesize_event_insights") as mock_synthesize, \
         patch.object(analyst_agent, "generate_company_analysis") as mock_analyze:
        
        mock_process_article.return_value = {**MOCK_INSIGHTS, "url": ARTICLE_INFO["link"]}
        mock_synthesize.return_value = MOCK_SYNTHESIS
        mock_analyze.return_value = MOCK_COMPANY_ANALYSIS
        
        # Create state
        state = {
            "company": COMPANY,
            "research_results": {
                EVENT_NAME: [ARTICLE_INFO]
            }
        }
        
        # Run the agent
        result = await analyst_agent.run(state)
        
        # Verify results
        assert result["goto"] == "writer_agent"
        assert result["analyst_status"] == "DONE"
        assert "analysis_results" in result
        assert "final_report" in result
        assert "analysis_stats" in result
        
        # Verify method calls
        mock_process_article.assert_called_once()
        mock_synthesize.assert_called_once()
        mock_analyze.assert_called_once()


@pytest.mark.asyncio
async def test_run_no_company(analyst_agent):
    """Test error handling when no company is provided"""
    # Create state with missing company
    state = {
        "research_results": {
            EVENT_NAME: [ARTICLE_INFO]
        }
    }
    
    # Run the agent
    result = await analyst_agent.run(state)
    
    # Verify error handling
    assert result["goto"] == "writer_agent"
    assert result["analyst_status"] == "ERROR"
    assert "error" in result
    assert "Company name missing" in result["error"]


@pytest.mark.asyncio
async def test_run_no_research_results(analyst_agent):
    """Test error handling when no research results are provided"""
    # Create state with missing research results
    state = {
        "company": COMPANY
    }
    
    # Run the agent
    result = await analyst_agent.run(state)
    
    # Verify error handling
    assert result["goto"] == "writer_agent"
    assert result["analyst_status"] == "ERROR"
    assert "error" in result
    assert "No research results" in result["error"]


@pytest.mark.asyncio
async def test_run_no_insights(analyst_agent, setup_llm_provider):
    """Test handling when no insights are found in any articles"""
    # Setup mocks
    with patch.object(analyst_agent, "process_article") as mock_process_article:
        
        # No insights found
        mock_process_article.return_value = None
        
        # Create state
        state = {
            "company": COMPANY,
            "research_results": {
                EVENT_NAME: [ARTICLE_INFO]
            }
        }
        
        # Run the agent
        result = await analyst_agent.run(state)
        
        # Check result components
        assert result["goto"] == "writer_agent"
        assert result["analyst_status"] == "DONE"
        assert "analysis_results" in result
        assert "final_report" in result
        
        # Check for text that should be in the final report
        assert "Forensic Analysis of Test Company" in result["final_report"]
        assert "Executive Summary" in result["final_report"]
        assert "Analysis Process" in result["final_report"]
        assert "no significant forensic concerns" in result["final_report"].lower()