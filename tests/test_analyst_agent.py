import pytest
import asyncio
import json
import datetime
from unittest.mock import patch, MagicMock, AsyncMock, call
from copy import deepcopy

@pytest.fixture
def config():
    return {
        "forensic_analysis": {
            "max_workers": 3,
            "batch_size": 5,
            "concurrent_events": 2,
            "task_timeout": 300,
            "evidence_strength": 3,
            "model": "gemini-2.0-pro",
        },
        "postgres": {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password"
        }
    }

@pytest.fixture
def state():
    return {
        "company": "Test Company",
        "industry": "Technology",
        "research_results": {
            "Event 1 - High": [
                {"title": "Article 1", "link": "https://example.com/1", "snippet": "Test snippet 1"},
                {"title": "Article 2", "link": "https://example.com/2", "snippet": "Test snippet 2"},
            ],
            "Event 2 - Medium": [
                {"title": "Article 3", "link": "https://example.com/3", "snippet": "Test snippet 3"},
            ]
        },
        "corporate_results": {
            "company_info": {
                "name": "Test Company",
                "industry": "Technology",
                "description": "A test company"
            },
            "executives": [
                {"name": "John Doe", "position": "CEO"},
                {"name": "Jane Smith", "position": "CFO"}
            ]
        },
        "goto": "analyst_agent"
    }

@pytest.fixture
def parsed_content():
    return {
        "success": True,
        "data": {
            "content": "This is some article content for testing.",
            "metadata": {
                "source_domain": "example.com",
                "fetch_timestamp": "2025-01-01T12:00:00"
            }
        }
    }

@pytest.fixture
def forensic_insights():
    return {
        "key_findings": ["Finding 1", "Finding 2"],
        "entities_involved": [
            {"name": "Test Person", "role": "Executive", "involvement": "High"},
            {"name": "Test Entity", "role": "Corporation", "involvement": "Medium"}
        ],
        "timeline": [
            {"date": "2024-01-01", "event": "First event", "significance": "High"},
            {"date": "2024-02-01", "event": "Second event", "significance": "Medium"}
        ],
        "red_flags": ["Red flag 1", "Red flag 2"],
        "source_credibility": "High",
        "evidence_strength": 4,
        "article_title": "Article 1",
        "event_category": "Event 1 - High"
    }

@pytest.fixture
def event_synthesis():
    return {
        "key_findings": ["Synthesized finding 1", "Synthesized finding 2"],
        "key_entities": [
            {"name": "John Doe", "role": "CEO", "involvement": "High"},
            {"name": "Test Corporation", "role": "Parent Company", "involvement": "Medium"}
        ],
        "narrative": "This is a test narrative of events.",
        "timeline": [
            {"date": "2024-01-01", "event": "Timeline event 1", "significance": "High"},
            {"date": "2024-02-01", "event": "Timeline event 2", "significance": "Medium"}
        ],
        "red_flags": ["Synthesized red flag 1", "Synthesized red flag 2"],
        "evidence_summary": "This is a summary of the evidence.",
        "importance_level": "High"
    }

@pytest.fixture
def company_analysis():
    return {
        "summary": "This is a company analysis summary.",
        "key_risks": ["Risk 1", "Risk 2"],
        "risk_areas": {"financial": "Medium", "regulatory": "High", "governance": "Low"},
        "entity_analysis": {"CEO": "High risk", "Board": "Low risk"},
        "patterns_identified": ["Pattern 1", "Pattern 2"],
        "conclusion": "This is the conclusion.",
        "report_markdown": "# Test Company Analysis\n\nTest content."
    }

@pytest.fixture
def agent(config):
    # Create mock for prompt manager
    prompt_manager_mock = MagicMock()
    prompt_manager_mock.get_prompt.return_value = ("System prompt", "Human prompt")
    
    # Create mock for logger
    logger_mock = MagicMock()
    
    # Create mock for content parser tool
    content_parser_mock = MagicMock()
    content_parser_mock.run = AsyncMock()
    
    # Create mock for postgres tool
    postgres_mock = MagicMock()
    postgres_mock.run = AsyncMock()
    
    # Create mock for LLM provider
    llm_mock = MagicMock()
    llm_mock.generate_text = AsyncMock()
    # We'll patch get_llm_provider to return llm_mock
    llm_provider_mock = AsyncMock(return_value=llm_mock)
    
    # Create mock for AgentMetrics
    metrics_mock = MagicMock()
    metrics_mock.execution_time_ms = 100.0
    
    # Set up base agent classes and metrics
    from base.base_agents import AgentMetrics
    
    with patch("utils.prompt_manager.get_prompt_manager", return_value=prompt_manager_mock), \
         patch("utils.logging.get_logger", return_value=logger_mock), \
         patch("tools.content_parser_tool.ContentParserTool", return_value=content_parser_mock), \
         patch("tools.postgres_tool.PostgresTool", return_value=postgres_mock), \
         patch("agents.analyst_agent.get_llm_provider", return_value=llm_provider_mock), \
         patch("base.base_agents.BaseAgent._log_start"), \
         patch("base.base_agents.BaseAgent._log_completion"):
        
        from agents.analyst_agent import AnalystAgent
        agent = AnalystAgent(config)
        
        # Add metrics attribute if missing
        agent.metrics = metrics_mock
        
        return agent

@pytest.mark.asyncio
async def test_extract_forensic_insights(agent, forensic_insights):
    # We need to set up a mock that returns actual strings, not coroutines
    llm_mock = MagicMock()
    # Important: Return actual string values
    llm_mock.generate_text = AsyncMock(side_effect=[
        "VALID_FORENSIC_CONTENT_HERE",  # First call for extract
        json.dumps(forensic_insights)    # Second call for analyze
    ])
    
    # Patch the get_llm_provider in the analyst_agent namespace so the agent uses our mock
    get_llm_mock = AsyncMock(return_value=llm_mock)
    
    with patch("agents.analyst_agent.get_llm_provider", get_llm_mock):
        # Use a longer content string to pass the minimum length check in the method
        long_content = (
            "This is article content with enough characters to pass the minimum length check. "
            "Let's add some more text to ensure it's definitely over 100 characters long. "
            "This should be sufficient to pass the initial validation check in the method."
        )
        
        result = await agent.extract_forensic_insights(
            company="Test Company",
            title="Test Article",
            content=long_content,
            event_name="Test Event"
        )
        
        # Verify results
        assert result is not None
        assert "key_findings" in result
        assert result["key_findings"] == forensic_insights["key_findings"]
        assert result["article_title"] == "Test Article"
        assert result["event_category"] == "Test Event"
        
        # Verify LLM was called twice
        assert llm_mock.generate_text.call_count == 2
        
        # Verify database save attempt
        assert agent.postgres_tool.run.called

@pytest.mark.asyncio
async def test_get_optimal_concurrency(agent):
    concurrency = await agent._get_optimal_concurrency()
    assert concurrency > 0
    assert concurrency <= agent.max_workers

@pytest.mark.asyncio
async def test_build_entity_network(agent):
    # Setup test data
    agent.knowledge_base["entities"] = {
        "John Doe": {
            "name": "John Doe",
            "type": "Person",
            "role": "CEO",
            "events": ["Event 1", "Event 2"],
            "relationships": [
                {"target": "Jane Smith", "type": "reports_to", "strength": 3}
            ]
        },
        "Jane Smith": {
            "name": "Jane Smith",
            "type": "Person",
            "role": "CFO",
            "events": ["Event 1"],
            "relationships": []
        },
        "No Connections": {
            "name": "No Connections",
            "type": "Person",
            "role": "Unknown",
            "events": [],
            "relationships": []
        }
    }
    
    event_synthesis = {
        "Event 1": {"importance_level": "High"},
        "Event 2": {"importance_level": "Medium"}
    }
    
    # Build network
    network = agent._build_entity_network(event_synthesis)
    
    # Verify network structure
    assert "John Doe" in network
    assert "Jane Smith" in network
    assert "No Connections" not in network  # Should exclude entities without connections
    
    # Verify connection details
    assert len(network["John Doe"]["connections"]) > 0
    event_connections = [c for c in network["John Doe"]["connections"] if "event" in c]
    assert len(event_connections) > 0

@pytest.mark.asyncio
async def test_integrate_corporate_data(agent, state):
    # Clear existing data
    agent.knowledge_base["entities"] = {}
    agent.result_tracker["entities_identified"] = set()
    
    # Integrate data
    await agent.integrate_corporate_data("Test Company", state)
    
    # Verify company was tracked as entity
    assert "Test Company" in agent.result_tracker["entities_identified"]
    assert "Test Company" in agent.knowledge_base["entities"]
    
    # Verify executives were tracked
    assert "John Doe" in agent.result_tracker["entities_identified"]
    assert "Jane Smith" in agent.result_tracker["entities_identified"]
    
    # Verify entity metadata
    assert agent.knowledge_base["entities"]["John Doe"]["role"] == "CEO"
    assert agent.knowledge_base["entities"]["Jane Smith"]["role"] == "CFO"

@pytest.mark.asyncio
async def test_process_events_concurrently(agent, event_synthesis):
    # Mock event processing
    with patch.object(agent, "process_event", AsyncMock(return_value=(["insight"], event_synthesis))), \
         patch.object(agent, "_get_optimal_concurrency", AsyncMock(return_value=2)):
        
        research_results = {
            "Event 1": [{"title": "Article 1"}, {"title": "Article 2"}],
            "Event 2": [{"title": "Article 3"}],
            "Event 3": [{"title": "Article 4"}]
        }
        
        results = await agent.process_events_concurrently("Test Company", research_results)
        
        # Verify results
        assert len(results) == 3
        assert "Event 1" in results
        assert "Event 2" in results
        assert "Event 3" in results
        assert results["Event 1"] == event_synthesis

@pytest.mark.asyncio
async def test_run_method(agent, state, event_synthesis, company_analysis):
    # Mock all the intensive methods
    with patch.object(agent, "integrate_corporate_data", AsyncMock()), \
         patch.object(agent, "integrate_youtube_data", AsyncMock()), \
         patch.object(agent, "process_events_concurrently", AsyncMock(return_value={"Event 1": event_synthesis})), \
         patch.object(agent, "generate_company_analysis", AsyncMock(return_value=company_analysis)), \
         patch.object(agent, "_build_entity_network", AsyncMock(return_value={"John Doe": {"name": "John Doe"}})), \
         patch.object(agent, "_log_start"), \
         patch.object(agent, "_log_completion"):
        
        # Run the agent
        result = await agent.run(state)
        
        # Verify the result structure
        assert result["goto"] == "meta_agent"
        assert result["analyst_status"] == "DONE"
        assert "analysis_results" in result
        assert "final_report" in result
        assert "analysis_stats" in result
        
        # Verify analysis results
        assert "event_synthesis" in result["analysis_results"]
        assert "company_analysis" in result["analysis_results"]
        assert "entity_network" in result["analysis_results"]
        assert "timeline" in result["analysis_results"]
        
        # Verify methods were called
        assert agent.integrate_corporate_data.called
        assert agent.integrate_youtube_data.called
        assert agent.process_events_concurrently.called
        assert agent.generate_company_analysis.called

@pytest.mark.asyncio
async def test_execute_method(agent, state):
    # Mock the run method
    with patch.object(agent, "run", AsyncMock(return_value={"result": "success"})):
        # Create an AgentState object
        from base.base_agents import AgentState
        agent_state = AgentState(**state)
        
        # Call _execute
        result = await agent._execute(agent_state)
        
        # Verify result
        assert result["result"] == "success"
        assert agent.run.called