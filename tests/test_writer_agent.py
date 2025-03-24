import pytest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock, call
from datetime import datetime
from copy import deepcopy

@pytest.fixture
def config():
    return {
        "writer": {
            "max_concurrent_sections": 3,
            "enable_iterative_improvement": True,
            "quality_threshold": 7
        },
        "postgres": {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password"
        },
        "models": {
            "report": "gemini-2.0-pro",
            "evaluation": "gemini-2.0-pro"
        }
    }

@pytest.fixture
def state():
    return {
        "company": "Test Company",
        "industry": "Technology",
        "research_results": {
            "Event 1": [
                {"title": "Article 1", "link": "https://example.com/1", "snippet": "Test snippet 1"},
                {"title": "Article 2", "link": "https://example.com/2", "snippet": "Test snippet 2"},
            ],
            "Event 2": [
                {"title": "Article 3", "link": "https://example.com/3", "snippet": "Test snippet 3"},
            ]
        },
        "event_metadata": {
            "Event 1": {"importance_score": 8, "is_quarterly_report": True},
            "Event 2": {"importance_score": 6, "is_quarterly_report": False},
        },
        "analysis_results": {
            "key_findings": ["Finding 1", "Finding 2"],
            "sentiment": "neutral"
        },
        "analyst_agent_status": "DONE",
        "goto": "writer_agent"
    }

@pytest.fixture
def template():
    return {
        "name": "standard_forensic_report",
        "sections": [
            {
                "name": "executive_summary",
                "title": "Executive Summary",
                "type": "markdown",
                "required": True,
                "variables": ["company", "top_events", "total_events"]
            },
            {
                "name": "key_events",
                "title": "Key Events Analysis",
                "type": "markdown",
                "required": True,
                "variables": ["company", "event"]
            },
            {
                "name": "other_events",
                "title": "Other Notable Events",
                "type": "markdown",
                "required": False,
                "variables": ["company", "event_summaries"]
            },
            {
                "name": "pattern_recognition",
                "title": "Pattern Recognition",
                "type": "markdown",
                "required": False,
                "variables": ["company", "events"]
            },
            {
                "name": "recommendations",
                "title": "Recommendations",
                "type": "markdown",
                "required": True,
                "variables": ["company", "top_events"]
            }
        ],
        "variables": {
            "company": "",
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "top_events": [],
            "total_events": 0,
            "event": {},
            "event_summaries": []
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "author": "System",
            "description": "Standard forensic analysis report template",
            "version": "1.0"
        }
    }

@pytest.fixture
def event_synthesis():
    return {
        "key_findings": ["Synthesized finding 1", "Synthesized finding 2"],
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

@pytest_asyncio.fixture
async def agent(config, template):
    # Create mock for prompt manager
    prompt_manager_mock = MagicMock()
    prompt_manager_mock.get_prompt.return_value = ("System prompt", "Human prompt")
    
    # Create mock for logger
    logger_mock = MagicMock()
    
    # Create mock for postgres tool
    postgres_mock = MagicMock()
    postgres_mock.run = AsyncMock()
    
    # Create mock for LLM provider
    llm_mock = MagicMock()
    llm_mock.generate_text = AsyncMock(return_value="Test generated content")
    # We'll patch get_llm_provider to return llm_mock
    llm_provider_mock = AsyncMock(return_value=llm_mock)
    
    # Create mock for AgentMetrics
    metrics_mock = MagicMock()
    metrics_mock.execution_time_ms = 100.0
    
    with patch("utils.prompt_manager.get_prompt_manager", return_value=prompt_manager_mock), \
         patch("utils.logging.get_logger", return_value=logger_mock), \
         patch("utils.llm_provider.get_llm_provider", return_value=llm_provider_mock), \
         patch("tools.postgres_tool.PostgresTool", return_value=postgres_mock), \
         patch("base.base_agents.BaseAgent._log_start"), \
         patch("base.base_agents.BaseAgent._log_completion"):
        
        from agents.writer_agent import WriterAgent
        agent = WriterAgent(config)
        
        # Add metrics attribute if missing
        agent.metrics = metrics_mock
        
        # Mock the select_template method to return our template fixture
        template_obj = MagicMock()
        for key, value in template.items():
            setattr(template_obj, key, value)
        agent.select_template = AsyncMock(return_value=template_obj)
        
        return agent

@pytest.mark.asyncio
async def test_initialize(agent):
    # Mock the necessary methods
    with patch.object(agent, "create_default_template", AsyncMock()):
        # Test with a mocked result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"template": "test"}
        agent.postgres_tool.run.return_value = mock_result
        
        await agent.load_templates()
        
        assert agent.loaded_templates is True
        agent.postgres_tool.run.assert_called_once()

@pytest.mark.asyncio
async def test_select_top_events(agent, state):
    events = state["research_results"]
    event_metadata = state["event_metadata"]
    
    top_events, other_events = agent._select_top_events(events, event_metadata)
    
    assert len(top_events) > 0
    assert "Event 1" in top_events  # Event 1 has higher importance score
    assert all(event in events.keys() for event in top_events)
    assert all(event in events.keys() for event in other_events)

@pytest.mark.asyncio
async def test_update_section_status(agent):
    section_name = "executive_summary"
    
    await agent.update_section_status(section_name, "PENDING")
    assert section_name in agent.section_statuses
    assert agent.section_statuses[section_name]["status"] == "PENDING"
    
    await agent.update_section_status(section_name, "GENERATING")
    assert agent.section_statuses[section_name]["status"] == "GENERATING"
    assert agent.section_statuses[section_name]["started_at"] is not None
    
    await agent.update_section_status(section_name, "DONE")
    assert agent.section_statuses[section_name]["status"] == "DONE"
    assert agent.section_statuses[section_name]["completed_at"] is not None
    
    await agent.update_section_status(section_name, "ERROR", "Test error")
    assert agent.section_statuses[section_name]["status"] == "ERROR"
    assert agent.section_statuses[section_name]["error"] == "Test error"

@pytest.mark.asyncio
async def test_generate_executive_summary(agent, template):
    company = "Test Company"
    top_events = ["Event 1", "Event 2"]
    all_events = {"Event 1": [], "Event 2": [], "Event 3": []}
    event_metadata = {
        "Event 1": {"importance_score": 8, "is_quarterly_report": True},
        "Event 2": {"importance_score": 6, "is_quarterly_report": False},
    }
    template_section = template["sections"][0]
    
    llm_mock = MagicMock()
    llm_mock.generate_text = AsyncMock(return_value="This is an executive summary.")
    
    with patch("utils.llm_provider.get_llm_provider", AsyncMock(return_value=llm_mock)):
        result = await agent.generate_executive_summary(company, top_events, all_events, event_metadata, template_section)
        
        assert "Executive Summary" in result
        assert "This is an executive summary" in result
        assert agent.postgres_tool.run.called
        assert agent.section_statuses["executive_summary"]["status"] == "DONE"

@pytest.mark.asyncio
async def test_generate_detailed_event_section(agent, template):
    company = "Test Company"
    event_name = "Event 1"
    event_data = [
        {"title": "Article 1", "source": "Source 1", "date": "2023-01-01", "snippet": "Snippet 1"},
        {"title": "Article 2", "source": "Source 2", "date": "2023-01-02", "snippet": "Snippet 2"}
    ]
    template_section = template["sections"][1]
    
    llm_mock = MagicMock()
    llm_mock.generate_text = AsyncMock(return_value="This is a detailed event analysis.")
    
    with patch("utils.llm_provider.get_llm_provider", AsyncMock(return_value=llm_mock)):
        result = await agent.generate_detailed_event_section(company, event_name, event_data, template_section)
        
        assert event_name in result
        assert "This is a detailed event analysis" in result
        assert agent.postgres_tool.run.called
        assert agent.section_statuses[f"key_events_{event_name}"]["status"] == "DONE"

@pytest.mark.asyncio
async def test_generate_meta_feedback(agent):
    company = "Test Company"
    full_report = "# Test Report\n\nThis is a test report with multiple sections."
    
    llm_mock = MagicMock()
    llm_mock.generate_text = AsyncMock(return_value=json.dumps({
        "quality_score": 8,
        "strengths": ["Well organized", "Clear explanations"],
        "weaknesses": ["Could use more examples"],
        "improvements": ["Add more specific examples"]
    }))
    
    with patch("utils.llm_provider.get_llm_provider", AsyncMock(return_value=llm_mock)):
        result = await agent.generate_meta_feedback(company, full_report)
        
        assert isinstance(result, dict)
        assert "quality_score" in result
        assert "strengths" in result
        assert "weaknesses" in result
        assert "improvements" in result
        assert agent.postgres_tool.run.called
        assert len(agent.feedback_history) > 0

@pytest.mark.asyncio
async def test_apply_iterative_improvements(agent):
    company = "Test Company"
    report_sections = {
        "executive_summary": "# Executive Summary\n\nThis is a summary.",
        "key_events": "# Key Events\n\nThese are the key events.",
        "recommendations": "# Recommendations\n\nThese are the recommendations."
    }
    feedback = {
        "quality_score": 6,
        "improvements": [
            "Make the executive summary more concise",
            "Add more details to recommendations"
        ]
    }
    
    # Mock revise_section
    with patch.object(agent, "revise_section", AsyncMock(return_value="Revised content")):
        result = await agent.apply_iterative_improvements(company, report_sections, feedback)
        
        assert isinstance(result, dict)
        assert "executive_summary" in result
        assert result["executive_summary"] == "Revised content"
        assert "recommendations" in result
        assert result["recommendations"] == "Revised content"

@pytest.mark.asyncio
async def test_run_method(agent, state, template):
    # Mock essential methods
    with patch.object(agent, "generate_sections_concurrently", AsyncMock(return_value={
            "executive_summary": "# Executive Summary\n\nContent",
            "key_events": "# Key Events\n\nContent",
            "recommendations": "# Recommendations\n\nContent",
            "header": "# Forensic News Analysis Report: Test Company\n\nReport Date: 2025-03-25\n\n"
         })), \
         patch.object(agent, "generate_meta_feedback", AsyncMock(return_value={
            "quality_score": 8,
            "strengths": ["Good"],
            "weaknesses": ["Could improve"],
            "improvements": []
         })), \
         patch.object(agent, "generate_executive_briefing", AsyncMock(return_value="Executive briefing content")), \
         patch.object(agent, "save_debug_report", AsyncMock(return_value="report.md")):
        
        # Run the agent
        result = await agent.run(state)
        
        # Verify the result structure
        assert result["goto"] == "meta_agent"
        assert result["writer_status"] == "DONE"
        assert "final_report" in result
        assert "report_sections" in result
        assert "report_feedback" in result
        assert "executive_briefing" in result
        assert "report_filename" in result
        
        # Verify methods were called
        assert agent.select_template.called
        assert agent.generate_sections_concurrently.called
        assert agent.generate_meta_feedback.called
        assert agent.generate_executive_briefing.called
        assert agent.save_debug_report.called

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

@pytest.mark.asyncio
async def test_error_handling(agent, state):
    # Test error handling during report generation
    with patch.object(agent, "select_template", AsyncMock(side_effect=Exception("Test error"))), \
         patch.object(agent, "attempt_recovery", AsyncMock(return_value=(True, state))), \
         patch.object(agent, "save_debug_report", AsyncMock(return_value="error_report.md")):
        
        result = await agent.run(state)
        
        # Verify recovery was attempted
        assert agent.attempt_recovery.called
        assert agent.save_debug_report.called

@pytest.mark.asyncio
async def test_generate_workflow_status(agent):
    state = {"company": "Test Company"}
    
    # Set up some section statuses
    agent.section_statuses = {
        "executive_summary": {"status": "DONE", "started_at": "2023-01-01T12:00:00", "completed_at": "2023-01-01T12:05:00", "retries": 0, "error": None},
        "key_events": {"status": "DONE", "started_at": "2023-01-01T12:05:00", "completed_at": "2023-01-01T12:10:00", "retries": 0, "error": None},
        "recommendations": {"status": "PENDING", "started_at": None, "completed_at": None, "retries": 0, "error": None},
        "other_events": {"status": "ERROR", "started_at": "2023-01-01T12:10:00", "completed_at": "2023-01-01T12:15:00", "retries": 1, "error": "Test error"}
    }
    
    result = await agent.generate_workflow_status(state)
    
    assert result["company"] == "Test Company"
    assert result["overall_status"] == "IN_PROGRESS"
    assert len(result["sections"]) == 4
    assert "progress_percentage" in result
    assert len(result["errors"]) == 1
    assert len(result["next_steps"]) > 0