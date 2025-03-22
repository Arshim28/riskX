import pytest
import pytest_asyncio
import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Tuple, Optional

from agents.meta_agent import MetaAgent
from utils.prompt_manager import PromptManager
from utils.llm_provider import LLMProvider, get_llm_provider


# Mock data for tests
COMPANY = "Test Company"
INDUSTRY = "Technology"

MOCK_RESEARCH_RESULTS = {
    "Event 1: Test Event - High": [
        {"title": "Test article 1", "link": "https://example.com/1", "source": "Source 1"},
        {"title": "Test article 2", "link": "https://example.com/2", "source": "Source 2"}
    ],
    "Event 2: Another Test - Medium": [
        {"title": "Test article 3", "link": "https://example.com/3", "source": "Source 3"}
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
    }
}

MOCK_QUALITY_ASSESSMENT = {
    "overall_score": 7,
    "coverage_score": 7,
    "balance_score": 8,
    "credibility_score": 6,
    "assessment": "Good quality research with comprehensive coverage.",
    "recommendations": {
        "regulatory": "Investigate regulatory issues more thoroughly",
        "legal": "Expand on legal implications"
    }
}

MOCK_LOW_QUALITY_ASSESSMENT = {
    "overall_score": 4,
    "coverage_score": 4,
    "balance_score": 5,
    "credibility_score": 3,
    "assessment": "Insufficient research with limited coverage.",
    "recommendations": {
        "regulatory": "Investigate regulatory issues more thoroughly",
        "legal": "Expand on legal implications"
    }
}

MOCK_RESEARCH_GAPS = {
    "regulatory_details": "Missing specific details about regulatory investigations",
    "legal_precedents": "Need information on similar legal cases"
}

MOCK_RESEARCH_PLAN = {
    "objective": "Expand research on regulatory and legal aspects",
    "key_areas_of_focus": ["Regulatory investigations", "Legal precedents"],
    "query_categories": {
        "regulatory": "Detailed investigation process",
        "legal": "Legal precedents and outcomes"
    },
    "query_generation_guidelines": "Focus on specific details and outcomes"
}

MOCK_ANALYSIS_GUIDANCE = {
    "focus_areas": ["Regulatory compliance", "Financial impact"],
    "priorities": ["Regulatory issues", "Legal exposure"],
    "analysis_strategies": ["Compare with industry standards", "Timeline analysis"],
    "red_flags": ["Unusual regulatory patterns", "Inconsistent reporting"]
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

MOCK_PRELIMINARY_GUIDELINES = {
    "objective": "Initial investigation into Test Company",
    "key_areas_of_focus": ["Company structure", "Recent news", "Regulatory compliance"],
    "query_categories": {
        "structure": "Test Company structure leadership",
        "news": "Test Company recent news controversy"
    },
    "query_generation_guidelines": "Focus on potential issues"
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
async def meta_agent(mock_logger, setup_llm_provider, mock_prompt_manager):
    """Fixture to create a MetaAgent with mocked dependencies"""
    config = {
        "models": {
            "evaluation": "test-evaluation-model",
            "planning": "test-planning-model"
        },
        "quality_thresholds": {
            "min_quality_score": 6
        },
        "max_iterations": 3,
        "max_event_iterations": 2
    }
    
    with patch.object(MetaAgent, "__init__", return_value=None):
        agent = MetaAgent(None)
        
        # Set the mocked properties
        agent.config = config
        agent.logger = mock_logger
        agent.prompt_manager = mock_prompt_manager
        agent.name = "meta_agent"
        
        # Add _log_start and _log_completion methods for safety
        agent._log_start = MagicMock(return_value=None)
        agent._log_completion = MagicMock(return_value=None)
        
        yield agent


@pytest.mark.asyncio
async def test_evaluate_research_quality(meta_agent, setup_llm_provider):
    """Test that research quality is correctly evaluated"""
    # Setup LLM response
    setup_llm_provider.generate_text.return_value = json.dumps(MOCK_QUALITY_ASSESSMENT)
    
    # Call the method
    result = await meta_agent.evaluate_research_quality(
        COMPANY, INDUSTRY, MOCK_RESEARCH_RESULTS
    )
    
    # Verify interactions and results
    setup_llm_provider.generate_text.assert_called_once()
    assert result == MOCK_QUALITY_ASSESSMENT
    assert result["overall_score"] == 7


@pytest.mark.asyncio
async def test_evaluate_research_quality_empty_results(meta_agent, setup_llm_provider):
    """Test research quality evaluation with empty results"""
    # Call the method with empty research results
    result = await meta_agent.evaluate_research_quality(
        COMPANY, INDUSTRY, {}
    )
    
    # Verify handling of empty results
    assert result["overall_score"] == 0
    assert "No research results available" in result["assessment"]
    setup_llm_provider.generate_text.assert_not_called()


@pytest.mark.asyncio
async def test_identify_research_gaps(meta_agent, setup_llm_provider):
    """Test identification of research gaps"""
    # Setup LLM response
    setup_llm_provider.generate_text.return_value = json.dumps(MOCK_RESEARCH_GAPS)
    
    # Call the method
    result = await meta_agent.identify_research_gaps(
        COMPANY, 
        INDUSTRY,
        "Event 1: Test Event - High", 
        MOCK_ANALYSIS_RESULTS["forensic_insights"]["Event 1: Test Event - High"],
        []  # No previous research plans
    )
    
    # Verify interactions and results
    setup_llm_provider.generate_text.assert_called_once()
    assert result == MOCK_RESEARCH_GAPS
    assert "regulatory_details" in result


@pytest.mark.asyncio
async def test_create_research_plan(meta_agent, setup_llm_provider):
    """Test creation of research plan based on gaps"""
    # Setup LLM response
    setup_llm_provider.generate_text.return_value = json.dumps(MOCK_RESEARCH_PLAN)
    
    # Call the method
    result = await meta_agent.create_research_plan(
        COMPANY, MOCK_RESEARCH_GAPS, []
    )
    
    # Verify interactions and results
    setup_llm_provider.generate_text.assert_called_once()
    assert result == MOCK_RESEARCH_PLAN
    assert result["objective"] == "Expand research on regulatory and legal aspects"


@pytest.mark.asyncio
async def test_create_research_plan_empty_gaps(meta_agent, setup_llm_provider):
    """Test creating research plan with empty gaps"""
    # Call the method with empty gaps
    result = await meta_agent.create_research_plan(
        COMPANY, {}
    )
    
    # Verify handling of empty gaps
    assert result == {}
    setup_llm_provider.generate_text.assert_not_called()


@pytest.mark.asyncio
async def test_generate_analysis_guidance(meta_agent, setup_llm_provider):
    """Test generation of analysis guidance"""
    # Setup LLM response
    setup_llm_provider.generate_text.return_value = json.dumps(MOCK_ANALYSIS_GUIDANCE)
    
    # Call the method
    result = await meta_agent.generate_analysis_guidance(
        COMPANY, MOCK_RESEARCH_RESULTS
    )
    
    # Verify interactions and results
    setup_llm_provider.generate_text.assert_called_once()
    assert result == MOCK_ANALYSIS_GUIDANCE
    assert "focus_areas" in result


@pytest.mark.asyncio
async def test_load_preliminary_guidelines_existing_file(meta_agent):
    """Test loading preliminary guidelines from existing file"""
    # Mock the os.path.exists and open functions
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value=MOCK_PRELIMINARY_GUIDELINES):
        
        # Call the method
        result = await meta_agent._load_preliminary_guidelines(COMPANY, INDUSTRY)
        
        # Verify results
        assert result == MOCK_PRELIMINARY_GUIDELINES
        assert result["objective"] == "Initial investigation into Test Company"


@pytest.mark.asyncio
async def test_load_preliminary_guidelines_file_not_found(meta_agent):
    """Test loading preliminary guidelines when file doesn't exist"""
    # Mock the os.path.exists function to return False
    with patch("os.path.exists", return_value=False):
        
        # Call the method
        result = await meta_agent._load_preliminary_guidelines(COMPANY, INDUSTRY)
        
        # Verify results - should generate a default plan
        assert "objective" in result
        assert COMPANY in result["objective"]
        assert "key_areas_of_focus" in result
        assert "query_categories" in result


@pytest.mark.asyncio
async def test_run_initial_phase(meta_agent):
    """Test the first phase of the run method - initial research"""
    # Mock the _load_preliminary_guidelines method
    with patch.object(meta_agent, "_load_preliminary_guidelines", return_value=MOCK_PRELIMINARY_GUIDELINES):
        
        # Create initial state
        state = {
            "company": COMPANY,
            "industry": INDUSTRY
        }
        
        # Run the agent
        result = await meta_agent.run(state)
        
        # Verify results
        assert result["goto"] == "research_agent"
        assert "research_plan" in result
        assert result["research_plan"] == [MOCK_PRELIMINARY_GUIDELINES]
        assert result["search_type"] == "google_news"
        assert result["return_type"] == "clustered"
        assert result["meta_iteration"] == 1


@pytest.mark.asyncio
async def test_run_quality_assessment_phase(meta_agent):
    """Test the quality assessment phase of the run method"""
    # Mock the evaluate_research_quality method
    with patch.object(meta_agent, "evaluate_research_quality", return_value=MOCK_QUALITY_ASSESSMENT):
        
        # Create state with research results
        state = {
            "company": COMPANY,
            "industry": INDUSTRY,
            "meta_iteration": 0,
            "search_history": [],
            "event_research_iterations": {},
            "research_results": MOCK_RESEARCH_RESULTS,
            "research_plan": []
        }
        
        # Run the agent
        result = await meta_agent.run(state)
        
        # Verify results
        assert result["goto"] == "analyst_agent"
        assert "quality_assessment" in result
        assert result["quality_assessment"] == MOCK_QUALITY_ASSESSMENT
        assert "analysis_guidance" in result
        assert result["meta_iteration"] == 1


@pytest.mark.asyncio
async def test_run_low_quality_assessment_phase(meta_agent):
    """Test handling of low quality assessment"""
    # Mock the evaluate_research_quality and create_research_plan methods
    with patch.object(meta_agent, "evaluate_research_quality", return_value=MOCK_LOW_QUALITY_ASSESSMENT), \
         patch.object(meta_agent, "create_research_plan", return_value=MOCK_RESEARCH_PLAN):
        
        # Create state with research results
        state = {
            "company": COMPANY,
            "industry": INDUSTRY,
            "meta_iteration": 0,
            "search_history": [],
            "event_research_iterations": {},
            "research_results": MOCK_RESEARCH_RESULTS,
            "research_plan": []
        }
        
        # Run the agent
        result = await meta_agent.run(state)
        
        # Verify results - should go back to research_agent
        assert result["goto"] == "research_agent"
        assert "quality_assessment" in result
        assert result["quality_assessment"] == MOCK_LOW_QUALITY_ASSESSMENT
        assert "research_plan" in result
        assert result["research_plan"] == [MOCK_RESEARCH_PLAN]
        assert result["meta_iteration"] == 1


@pytest.mark.asyncio
async def test_run_gap_analysis_phase(meta_agent):
    """Test the gap analysis phase after initial analysis"""
    # Mock the identify_research_gaps and create_research_plan methods
    with patch.object(meta_agent, "identify_research_gaps", return_value=MOCK_RESEARCH_GAPS), \
         patch.object(meta_agent, "create_research_plan", return_value=MOCK_RESEARCH_PLAN):
        
        # Create state with research and analysis results
        state = {
            "company": COMPANY,
            "industry": INDUSTRY,
            "meta_iteration": 0,
            "search_history": [],
            "event_research_iterations": {},
            "research_results": MOCK_RESEARCH_RESULTS,
            "research_plan": [],
            "quality_assessment": MOCK_QUALITY_ASSESSMENT,
            "analysis_results": MOCK_ANALYSIS_RESULTS
        }
        
        # Run the agent
        result = await meta_agent.run(state)
        
        # Verify results - should go back to research_agent with new plans
        assert result["goto"] == "research_agent"
        assert "research_plan" in result
        assert result["meta_iteration"] == 1
        
        # Check event research iterations were incremented
        assert "event_research_iterations" in result
        for event in MOCK_ANALYSIS_RESULTS["forensic_insights"]:
            assert event in result["event_research_iterations"]
            assert result["event_research_iterations"][event] == 1


@pytest.mark.asyncio
async def test_run_final_analysis_phase(meta_agent):
    """Test the final analysis phase"""
    # Create state with all previous phases completed and additional research done
    state = {
        "company": COMPANY,
        "industry": INDUSTRY,
        "meta_iteration": 2,
        "search_history": [["query1"], ["query2"]],
        "event_research_iterations": {
            "Event 1: Test Event - High": 1,
            "Event 2: Another Test - Medium": 1
        },
        "research_results": MOCK_RESEARCH_RESULTS,
        "research_plan": [MOCK_RESEARCH_PLAN],
        "quality_assessment": MOCK_QUALITY_ASSESSMENT,
        "analysis_results": MOCK_ANALYSIS_RESULTS,
        "additional_research_completed": True
    }
    
    # Run the agent
    result = await meta_agent.run(state)
    
    # Verify results - should go to analyst_agent for final analysis
    assert result["goto"] == "analyst_agent"
    assert result["final_analysis_requested"] is True
    assert "final_analysis_completed" not in result
    assert result["meta_iteration"] == 3


@pytest.mark.asyncio
async def test_run_completion_phase(meta_agent):
    """Test the completion phase after final analysis"""
    # Create state with all phases completed including final analysis
    state = {
        "company": COMPANY,
        "industry": INDUSTRY,
        "meta_iteration": 3,
        "search_history": [["query1"], ["query2"], ["query3"]],
        "event_research_iterations": {
            "Event 1: Test Event - High": 1,
            "Event 2: Another Test - Medium": 1
        },
        "research_results": MOCK_RESEARCH_RESULTS,
        "research_plan": [MOCK_RESEARCH_PLAN],
        "quality_assessment": MOCK_QUALITY_ASSESSMENT,
        "analysis_results": MOCK_ANALYSIS_RESULTS,
        "additional_research_completed": True,
        "final_analysis_requested": True
    }
    
    # Run the agent
    result = await meta_agent.run(state)
    
    # Verify results - should go to writer_agent with completion status
    assert result["goto"] == "writer_agent"
    assert result["status"] == "complete"
    assert result["final_analysis_completed"] is True
    assert result["meta_iteration"] == 4


@pytest.mark.asyncio
async def test_run_missing_company(meta_agent):
    """Test error handling when company name is missing"""
    # Create state without company name
    state = {
        "industry": INDUSTRY
    }
    
    # Run the agent
    result = await meta_agent.run(state)
    
    # Verify error handling
    assert result["goto"] == "END"
    assert "error" in result
    assert "Company name is missing" in result["error"]


@pytest.mark.asyncio
async def test_error_handling_in_evaluate_research_quality(meta_agent, setup_llm_provider):
    """Test error handling in evaluate_research_quality method"""
    # Setup LLM to raise an exception
    setup_llm_provider.generate_text.side_effect = Exception("Test error")
    
    # Call the method
    result = await meta_agent.evaluate_research_quality(
        COMPANY, INDUSTRY, MOCK_RESEARCH_RESULTS
    )
    
    # Verify error handling
    assert result["overall_score"] == 0
    assert "Error during evaluation" in result["assessment"]
    assert "recommendations" in result


@pytest.mark.asyncio
async def test_error_handling_in_create_research_plan(meta_agent, setup_llm_provider):
    """Test error handling in create_research_plan method"""
    # Setup LLM to raise an exception
    setup_llm_provider.generate_text.side_effect = Exception("Test error")
    
    # Call the method
    result = await meta_agent.create_research_plan(
        COMPANY, MOCK_RESEARCH_GAPS
    )
    
    # Verify error handling - should return a minimal valid plan
    assert "objective" in result
    assert COMPANY in result["objective"]
    assert "key_areas_of_focus" in result
    assert "query_categories" in result
    assert "query_generation_guidelines" in result