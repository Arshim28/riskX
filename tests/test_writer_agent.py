# tests/test_writer_agent.py
import pytest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

from agents.writer_agent import WriterAgent, ReportTemplate


@pytest.fixture
def config():
    return {
        "writer": {
            "max_concurrent_sections": 3,
            "enable_iterative_improvement": True
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
        "analyst_status": "DONE",
        "research_results": {
            "Event 1": [
                {"title": "Article 1", "link": "https://example.com/1", "snippet": "Test snippet 1"},
                {"title": "Article 2", "link": "https://example.com/2", "snippet": "Test snippet 2"}
            ],
            "Event 2": [
                {"title": "Article 3", "link": "https://example.com/3", "snippet": "Test snippet 3"}
            ]
        },
        "event_metadata": {
            "Event 1": {"importance_score": 80, "article_count": 2, "is_quarterly_report": False},
            "Event 2": {"importance_score": 60, "article_count": 1, "is_quarterly_report": False}
        },
        "analysis_results": {
            "event_synthesis": {
                "Event 1": {"key_entities": [{"name": "Entity1", "role": "Subject"}]},
                "Event 2": {"key_entities": [{"name": "Entity2", "role": "Subject"}]}
            },
            "red_flags": ["Flag1", "Flag2"]
        }
    }


@pytest.fixture
def template():
    return ReportTemplate(
        template_name="standard_forensic_report",
        sections=[
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
            }
        ],
        variables={
            "company": "",
            "report_date": "2023-01-01",
            "top_events": [],
            "total_events": 0
        },
        metadata={
            "created_at": "2023-01-01T00:00:00",
            "author": "System",
            "description": "Standard forensic analysis report template",
            "version": "1.0"
        }
    )


@pytest.fixture
def agent(config):
    with patch("utils.prompt_manager.get_prompt_manager"), \
         patch("utils.logging.get_logger"), \
         patch("tools.postgres_tool.PostgresTool"):
        agent = WriterAgent(config)
        agent.templates["standard_forensic_report"] = None  # To be replaced in tests
        agent.loaded_templates = True
        return agent


@pytest.mark.asyncio
async def test_select_top_events(agent, state):
    top_events, other_events = agent._select_top_events(
        state["research_results"],
        state["event_metadata"],
        max_detailed_events=1
    )
    
    assert len(top_events) == 1
    assert len(other_events) == 1
    assert "Event 1" in top_events  # Higher importance score
    assert "Event 2" in other_events


@pytest.mark.asyncio
async def test_generate_executive_summary(agent, state):
    with patch("utils.llm_provider.get_llm_provider") as mock_provider, \
         patch.object(agent, "postgres_tool"):
        
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "This is an executive summary."
        mock_provider.return_value = mock_llm
        
        template_section = {
            "name": "executive_summary",
            "title": "Executive Summary"
        }
        
        result = await agent.generate_executive_summary(
            "Test Company",
            ["Event 1"],
            state["research_results"],
            state["event_metadata"],
            template_section
        )
        
        assert result.startswith("# Executive Summary")
        assert "This is an executive summary." in result
        mock_llm.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_run(agent, state, template):
    agent.templates["standard_forensic_report"] = template
    
    with patch.object(agent, "select_template", return_value=template), \
         patch.object(agent, "generate_sections_concurrently") as mock_generate, \
         patch.object(agent, "generate_meta_feedback") as mock_feedback, \
         patch.object(agent, "apply_iterative_improvements") as mock_improve, \
         patch.object(agent, "generate_executive_briefing") as mock_briefing, \
         patch.object(agent, "save_debug_report") as mock_save:
        
        mock_generate.return_value = {
            "executive_summary": "# Executive Summary\n\nThis is a summary.",
            "key_events": "# Key Events\n\nEvent details."
        }
        
        mock_feedback.return_value = {
            "quality_score": 7,
            "strengths": ["Good overview"],
            "weaknesses": ["Some missing details"],
            "improvements": ["Add more context to executive summary"]
        }
        
        mock_improve.return_value = {
            "executive_summary": "# Executive Summary\n\nThis is an improved summary.",
            "key_events": "# Key Events\n\nImproved event details."
        }
        
        mock_briefing.return_value = "This is an executive briefing."
        
        mock_save.return_value = "report.md"
        
        result = await agent.run(state)
        
        assert result["goto"] == "meta_agent"
        assert result["writer_status"] == "DONE"
        assert "final_report" in result
        assert "report_sections" in result
        assert "report_feedback" in result
        assert "executive_briefing" in result
        assert "report_filename" in result
        
        mock_generate.assert_called_once()
        mock_feedback.assert_called()
        mock_improve.assert_called_once()
        mock_briefing.assert_called_once()
        mock_save.assert_called_once()