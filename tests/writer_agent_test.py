import pytest
import pytest_asyncio
import asyncio
import json
import logging
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from typing import Dict, List, Any, Tuple, Optional

from agents.writer_agent import WriterAgent
from utils.prompt_manager import PromptManager
from utils.llm_provider import LLMProvider, get_llm_provider


# Mock data for tests
COMPANY = "Test Company"

MOCK_RESEARCH_RESULTS = {
    "Event 1: Test Event - High": [
        {"title": "Test article 1", "link": "https://example.com/1", "source": "Source 1", "snippet": "Test snippet 1", "date": "2025-03-01"},
        {"title": "Test article 2", "link": "https://example.com/2", "source": "Source 2", "snippet": "Test snippet 2", "date": "2025-03-02"}
    ],
    "Event 2: Another Test - Medium": [
        {"title": "Test article 3", "link": "https://example.com/3", "source": "Source 3", "snippet": "Test snippet 3", "date": "2025-03-03"}
    ],
    "Event 3: Low Priority": [
        {"title": "Test article 4", "link": "https://example.com/4", "source": "Source 4", "snippet": "Test snippet 4", "date": "2025-03-04"}
    ],
    "Event 4: Very Low Priority": [
        {"title": "Test article 5", "link": "https://example.com/5", "source": "Source 5", "snippet": "Test snippet 5", "date": "2025-03-05"}
    ],
    "Event 5: Another Low Priority": [
        {"title": "Test article 6", "link": "https://example.com/6", "source": "Source 6", "snippet": "Test snippet 6", "date": "2025-03-06"}
    ],
    "Event 6: Yet Another Low": [
        {"title": "Test article 7", "link": "https://example.com/7", "source": "Source 7", "snippet": "Test snippet 7", "date": "2025-03-07"}
    ],
    "Event 7: Last Priority": [
        {"title": "Test article 8", "link": "https://example.com/8", "source": "Source 8", "snippet": "Test snippet 8", "date": "2025-03-08"}
    ]
}

MOCK_EVENT_METADATA = {
    "Event 1: Test Event - High": {
        "importance_score": 75,
        "article_count": 2,
        "is_quarterly_report": False
    },
    "Event 2: Another Test - Medium": {
        "importance_score": 60,
        "article_count": 1,
        "is_quarterly_report": False
    },
    "Event 3: Low Priority": {
        "importance_score": 45,
        "article_count": 1,
        "is_quarterly_report": False
    },
    "Event 4: Very Low Priority": {
        "importance_score": 30,
        "article_count": 1,
        "is_quarterly_report": False
    },
    "Event 5: Another Low Priority": {
        "importance_score": 35,
        "article_count": 1,
        "is_quarterly_report": False
    },
    "Event 6: Yet Another Low": {
        "importance_score": 40,
        "article_count": 1,
        "is_quarterly_report": False
    },
    "Event 7: Last Priority": {
        "importance_score": 25,
        "article_count": 1,
        "is_quarterly_report": False
    }
}

MOCK_ANALYSIS_RESULTS = {
    "forensic_insights": {
        "Event 1: Test Event - High": [
            {
                "ALLEGATIONS": "Test allegations",
                "ENTITIES": "Test entities",
                "TIMELINE": "Test timeline",
                "MAGNITUDE": "Test magnitude",
                "EVIDENCE": "Test evidence",
                "RESPONSE": "Test response",
                "STATUS": "Test status",
                "CREDIBILITY": "Medium"
            }
        ],
        "Event 2: Another Test - Medium": [
            {
                "ALLEGATIONS": "More allegations",
                "ENTITIES": "More entities",
                "TIMELINE": "More timeline",
                "MAGNITUDE": "More magnitude",
                "EVIDENCE": "More evidence",
                "RESPONSE": "More response",
                "STATUS": "More status",
                "CREDIBILITY": "Low"
            }
        ]
    }
}

MOCK_DETAILED_EVENT_SECTION = """
## Test Event - High

### Background
This is a test event with high importance.

### Key Facts
- Fact 1
- Fact 2

### Timeline
1. Event happened
2. Consequences occurred
"""

MOCK_OTHER_EVENTS_SECTION = """
# Other Notable Events

## Event 3: Low Priority
This event had lower importance but still notable.

## Event 4: Very Low Priority
This event had minimal impact.
"""

MOCK_EXECUTIVE_SUMMARY = """
# Executive Summary

This report presents findings from an analysis of Test Company.
Key events include serious allegations and regulatory concerns.
"""

MOCK_PATTERN_SECTION = """
# Pattern Recognition

The analysis identified several recurring patterns across events:
1. Pattern A
2. Pattern B
"""

MOCK_RECOMMENDATIONS = """
# Recommendations

1. Recommendation 1
2. Recommendation 2
3. Recommendation 3
"""


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
async def writer_agent(mock_logger, setup_llm_provider, mock_prompt_manager):
    """Fixture to create a WriterAgent with mocked dependencies"""
    config = {
        "models": {
            "report": "test-report-model"
        }
    }
    
    with patch.object(WriterAgent, "__init__", return_value=None):
        agent = WriterAgent(None)
        
        # Set the mocked properties
        agent.config = config
        agent.logger = mock_logger
        agent.prompt_manager = mock_prompt_manager
        agent.name = "writer_agent"
        
        # Add _log_start and _log_completion methods for safety
        agent._log_start = MagicMock(return_value=None)
        agent._log_completion = MagicMock(return_value=None)
        
        yield agent


@pytest.mark.asyncio
async def test_select_top_events(writer_agent):
    """Test selecting top events based on importance scores"""
    # Call the method
    top_events, other_events = writer_agent._select_top_events(
        MOCK_RESEARCH_RESULTS, MOCK_EVENT_METADATA, max_detailed_events=3
    )
    
    # Verify results
    assert len(top_events) == 3
    assert len(other_events) == 4
    
    # Check order - should be sorted by importance
    assert top_events[0] == "Event 1: Test Event - High"
    assert top_events[1] == "Event 2: Another Test - Medium"
    assert top_events[2] == "Event 3: Low Priority"  # Third highest score


@pytest.mark.asyncio
async def test_select_top_events_with_ties(writer_agent):
    """Test selecting top events when there are tied scores"""
    # Create event metadata with tied scores
    tied_metadata = {
        "Event 1": {"importance_score": 75},
        "Event 2": {"importance_score": 60},
        "Event 3": {"importance_score": 60},  # Tied with Event 2
        "Event 4": {"importance_score": 45}
    }
    
    events = {
        "Event 1": [],
        "Event 2": [],
        "Event 3": [],
        "Event 4": []
    }
    
    # Call the method
    top_events, other_events = writer_agent._select_top_events(
        events, tied_metadata, max_detailed_events=3
    )
    
    # Verify results
    assert len(top_events) == 3
    assert len(other_events) == 1
    
    # Check order - should be sorted by importance, then by name for ties
    assert top_events[0] == "Event 1"
    # Event 2 and Event 3 should both be included (tied score)
    assert "Event 2" in top_events
    assert "Event 3" in top_events
    assert "Event 4" in other_events


@pytest.mark.asyncio
async def test_select_top_events_with_missing_metadata(writer_agent):
    """Test selecting top events when metadata is missing"""
    # Create events without corresponding metadata
    events = {
        "Event 1": [],
        "Event 2": [],
        "Event With No Metadata": []
    }
    
    metadata = {
        "Event 1": {"importance_score": 75},
        "Event 2": {"importance_score": 60}
        # Missing metadata for "Event With No Metadata"
    }
    
    # Call the method
    top_events, other_events = writer_agent._select_top_events(
        events, metadata, max_detailed_events=2
    )
    
    # Verify results
    assert len(top_events) == 2
    assert len(other_events) == 1
    
    # Check order - events with metadata should be prioritized
    assert top_events[0] == "Event 1"
    assert top_events[1] == "Event 2"
    assert other_events[0] == "Event With No Metadata"  # Should be last due to default score of 0


@pytest.mark.asyncio
async def test_generate_detailed_event_section(writer_agent, setup_llm_provider):
    """Test generating a detailed event section"""
    # Setup LLM response
    setup_llm_provider.generate_text.return_value = MOCK_DETAILED_EVENT_SECTION
    
    # Call the method
    result = await writer_agent.generate_detailed_event_section(
        COMPANY, 
        "Event 1: Test Event - High", 
        MOCK_RESEARCH_RESULTS["Event 1: Test Event - High"]
    )
    
    # Verify interactions and results
    setup_llm_provider.generate_text.assert_called_once()
    assert "## Event 1: Test Event - High" in result
    assert "Background" in result
    assert "Key Facts" in result
    assert "Timeline" in result


@pytest.mark.asyncio
async def test_generate_detailed_event_section_with_error(writer_agent, setup_llm_provider):
    """Test error handling in generate_detailed_event_section"""
    # Setup LLM to raise an exception
    setup_llm_provider.generate_text.side_effect = Exception("Test error")
    
    # Call the method
    result = await writer_agent.generate_detailed_event_section(
        COMPANY, 
        "Event 1: Test Event - High", 
        MOCK_RESEARCH_RESULTS["Event 1: Test Event - High"]
    )
    
    # Verify error handling
    assert "Unable to generate detailed analysis due to technical error" in result
    assert "Test error" in result


@pytest.mark.asyncio
async def test_generate_other_events_section(writer_agent, setup_llm_provider):
    """Test generating the other events section"""
    # Setup LLM response
    setup_llm_provider.generate_text.return_value = MOCK_OTHER_EVENTS_SECTION
    
    # Call the method
    result = await writer_agent.generate_other_events_section(
        COMPANY, 
        MOCK_RESEARCH_RESULTS, 
        MOCK_EVENT_METADATA, 
        ["Event 3: Low Priority", "Event 4: Very Low Priority"]
    )
    
    # Verify interactions and results
    setup_llm_provider.generate_text.assert_called_once()
    assert "# Other Notable Events" in result
    assert "Event 3: Low Priority" in result
    assert "Event 4: Very Low Priority" in result


@pytest.mark.asyncio
async def test_generate_other_events_section_empty(writer_agent, setup_llm_provider):
    """Test generating the other events section with no events"""
    # Call the method with empty list
    result = await writer_agent.generate_other_events_section(
        COMPANY, 
        MOCK_RESEARCH_RESULTS, 
        MOCK_EVENT_METADATA, 
        []
    )
    
    # Verify handling of empty list
    assert result == ""
    setup_llm_provider.generate_text.assert_not_called()


@pytest.mark.asyncio
async def test_generate_other_events_section_with_error(writer_agent, setup_llm_provider):
    """Test error handling in generate_other_events_section"""
    # Setup LLM to raise an exception
    setup_llm_provider.generate_text.side_effect = Exception("Test error")
    
    # Call the method
    result = await writer_agent.generate_other_events_section(
        COMPANY, 
        MOCK_RESEARCH_RESULTS, 
        MOCK_EVENT_METADATA, 
        ["Event 3: Low Priority", "Event 4: Very Low Priority"]
    )
    
    # Verify error handling - should generate basic summary
    assert "# Other Notable Events" in result
    assert "following events were also identified" in result
    assert "Event 3: Low Priority" in result


@pytest.mark.asyncio
async def test_generate_executive_summary(writer_agent, setup_llm_provider):
    """Test generating the executive summary"""
    # Setup LLM response
    setup_llm_provider.generate_text.return_value = MOCK_EXECUTIVE_SUMMARY
    
    # Call the method
    result = await writer_agent.generate_executive_summary(
        COMPANY,
        ["Event 1: Test Event - High", "Event 2: Another Test - Medium"],
        MOCK_RESEARCH_RESULTS,
        MOCK_EVENT_METADATA
    )
    
    # Verify interactions and results
    setup_llm_provider.generate_text.assert_called_once()
    assert "# Executive Summary" in result
    assert "Test Company" in result


@pytest.mark.asyncio
async def test_generate_executive_summary_with_error(writer_agent, setup_llm_provider):
    """Test error handling in generate_executive_summary"""
    # Setup LLM to raise an exception
    setup_llm_provider.generate_text.side_effect = Exception("Test error")
    
    # Call the method
    result = await writer_agent.generate_executive_summary(
        COMPANY,
        ["Event 1: Test Event - High", "Event 2: Another Test - Medium"],
        MOCK_RESEARCH_RESULTS,
        MOCK_EVENT_METADATA
    )
    
    # Verify error handling - should generate basic summary
    assert "# Executive Summary" in result
    assert "Test Company" in result
    assert "events analyzed in detail include" in result.lower()


@pytest.mark.asyncio
async def test_generate_pattern_section(writer_agent, setup_llm_provider):
    """Test generating the pattern recognition section"""
    # Setup LLM response
    setup_llm_provider.generate_text.return_value = MOCK_PATTERN_SECTION
    
    # Call the method
    result = await writer_agent.generate_pattern_section(
        COMPANY,
        ["Event 1: Test Event - High", "Event 2: Another Test - Medium"],
        MOCK_EVENT_METADATA
    )
    
    # Verify interactions and results
    setup_llm_provider.generate_text.assert_called_once()
    assert "# Pattern Recognition" in result
    assert "recurring patterns" in result


@pytest.mark.asyncio
async def test_generate_pattern_section_single_event(writer_agent, setup_llm_provider):
    """Test generating pattern section with only one event"""
    # Call the method with only one event
    result = await writer_agent.generate_pattern_section(
        COMPANY,
        ["Event 1: Test Event - High"],
        MOCK_EVENT_METADATA
    )
    
    # Verify handling of single event - should return empty string
    assert result == ""
    setup_llm_provider.generate_text.assert_not_called()


@pytest.mark.asyncio
async def test_generate_recommendations(writer_agent, setup_llm_provider):
    """Test generating recommendations"""
    # Setup LLM response
    setup_llm_provider.generate_text.return_value = MOCK_RECOMMENDATIONS
    
    # Call the method
    result = await writer_agent.generate_recommendations(
        COMPANY,
        ["Event 1: Test Event - High", "Event 2: Another Test - Medium"]
    )
    
    # Verify interactions and results
    setup_llm_provider.generate_text.assert_called_once()
    assert "# Recommendations" in result
    assert "Recommendation 1" in result
    assert "Recommendation 2" in result


@pytest.mark.asyncio
async def test_generate_recommendations_with_error(writer_agent, setup_llm_provider):
    """Test error handling in generate_recommendations"""
    # Setup LLM to raise an exception
    setup_llm_provider.generate_text.side_effect = Exception("Test error")
    
    # Call the method
    result = await writer_agent.generate_recommendations(
        COMPANY,
        ["Event 1: Test Event - High", "Event 2: Another Test - Medium"]
    )
    
    # Verify error handling - should generate basic recommendations
    assert "# Recommendations" in result
    assert "Consider performing a more detailed investigation" in result


@pytest.mark.asyncio
async def test_save_debug_report(writer_agent):
    """Test saving a debug copy of the report"""
    # Mock os.path.exists and os.makedirs
    with patch("os.path.exists", return_value=False), \
         patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open()) as mock_file:
        
        # Call the method
        await writer_agent.save_debug_report(COMPANY, "Test report content")
        
        # Verify interactions
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once()
        mock_file().write.assert_called_once_with("Test report content")


@pytest.mark.asyncio
async def test_save_debug_report_with_error(writer_agent, mock_logger):
    """Test error handling when saving debug report"""
    # Mock os.path.exists to raise an exception
    with patch("os.path.exists", side_effect=Exception("Test error")):
        
        # Call the method
        await writer_agent.save_debug_report(COMPANY, "Test report content")
        
        # Verify error handling
        mock_logger.error.assert_called_once()
        assert "Test error" in mock_logger.error.call_args[0][0]


@pytest.mark.asyncio
async def test_run_basic_flow(writer_agent):
    """Test the basic flow of the run method"""
    # Mock component generation methods
    with patch.object(writer_agent, "_select_top_events", return_value=(["Event 1", "Event 2"], ["Event 3"])), \
         patch.object(writer_agent, "generate_executive_summary", return_value="# Executive Summary\n\nTest summary"), \
         patch.object(writer_agent, "generate_detailed_event_section", return_value="## Event\n\nTest details"), \
         patch.object(writer_agent, "generate_other_events_section", return_value="# Other Events\n\nTest others"), \
         patch.object(writer_agent, "generate_pattern_section", return_value="# Patterns\n\nTest patterns"), \
         patch.object(writer_agent, "generate_recommendations", return_value="# Recommendations\n\nTest recommendations"), \
         patch.object(writer_agent, "save_debug_report") as mock_save:
        
        # Create state with analyst complete
        state = {
            "company": COMPANY,
            "research_results": MOCK_RESEARCH_RESULTS,
            "event_metadata": MOCK_EVENT_METADATA,
            "analysis_results": MOCK_ANALYSIS_RESULTS,
            "analyst_status": "DONE"
        }
        
        # Run the agent
        result = await writer_agent.run(state)
        
        # Verify results
        assert result["goto"] == "END"
        assert "final_report" in result
        assert "report_sections" in result
        assert "top_events" in result
        assert "other_events" in result
        
        # Check that save_debug_report was called
        mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_run_analyst_not_done(writer_agent):
    """Test handling when analyst is not done"""
    # Create state with analyst not complete
    state = {
        "company": COMPANY,
        "research_results": MOCK_RESEARCH_RESULTS,
        "event_metadata": MOCK_EVENT_METADATA,
        "analysis_results": MOCK_ANALYSIS_RESULTS,
        "analyst_status": "IN_PROGRESS"  # Not DONE
    }
    
    # Run the agent
    result = await writer_agent.run(state)
    
    # Verify results - should wait for analyst
    assert result["goto"] == "writer_agent"
    assert "final_report" not in result


@pytest.mark.asyncio
async def test_run_with_error(writer_agent, mock_logger):
    """Test error handling in the run method"""
    # Mock _select_top_events to raise an exception
    with patch.object(writer_agent, "_select_top_events", side_effect=Exception("Test error")):
        
        # Create state with analyst complete
        state = {
            "company": COMPANY,
            "research_results": MOCK_RESEARCH_RESULTS,
            "event_metadata": MOCK_EVENT_METADATA,
            "analysis_results": MOCK_ANALYSIS_RESULTS,
            "analyst_status": "DONE"
        }
        
        # Run the agent
        result = await writer_agent.run(state)
        
        # Verify error handling
        assert result["goto"] == "END"
        assert "final_report" in result
        assert "Test error" in result.get("error", "")
        
        # Check fallback report content
        fallback_report = result["final_report"]
        assert "Forensic News Analysis Report" in fallback_report
        assert COMPANY in fallback_report
        assert "Technical Issue" in fallback_report


@pytest.mark.asyncio
async def test_run_with_invalid_data(writer_agent):
    """Test handling of invalid data types in state"""
    # Create state with invalid data types
    state = {
        "company": COMPANY,
        "research_results": "invalid string instead of dict",  # Wrong type
        "event_metadata": None,  # Wrong type
        "analysis_results": MOCK_ANALYSIS_RESULTS,
        "analyst_status": "DONE"
    }
    
    # Run the agent
    result = await writer_agent.run(state)
    
    # Verify results - should handle invalid data gracefully
    assert result["goto"] == "END"
    assert "final_report" in result
    assert "type" in mock_logger.error.call_args_list[0][0][0]  # Error log about wrong type