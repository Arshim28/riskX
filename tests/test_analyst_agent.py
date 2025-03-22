# tests/test_analyst_agent.py
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

from agents.analyst_agent import AnalystAgent, AnalysisTask


@pytest.fixture
def config():
    return {
        "forensic_analysis": {
            "max_workers": 5,
            "batch_size": 10,
            "concurrent_events": 2,
            "task_timeout": 300,
            "model": "gemini-2.0-pro"
        }
    }


@pytest.fixture
def state():
    return {
        "company": "Test Company",
        "research_results": {
            "Event 1": [
                {"title": "Article 1", "link": "https://example.com/1", "snippet": "Test snippet 1"},
                {"title": "Article 2", "link": "https://example.com/2", "snippet": "Test snippet 2"}
            ]
        },
        "event_metadata": {
            "Event 1": {"importance_score": 70, "article_count": 2, "is_quarterly_report": False}
        }
    }


@pytest.fixture
def agent(config):
    with patch("utils.prompt_manager.get_prompt_manager"), \
         patch("utils.logging.get_logger"), \
         patch("tools.content_parser_tool.ContentParserTool"), \
         patch("tools.postgres_tool.PostgresTool"):
        return AnalystAgent(config)


@pytest.mark.asyncio
async def test_extract_forensic_insights(agent):
    with patch("utils.llm_provider.get_llm_provider") as mock_provider:
        mock_llm = AsyncMock()
        # First call for extraction
        mock_llm.generate_text.side_effect = [
            "Test extracted content",
            '{"allegations": "Test", "entities": ["Entity1"], "timeline": "2023", "magnitude": "Large", "evidence": "Document", "response": "Denied", "status": "Ongoing", "credibility": 7}'
        ]
        mock_provider.return_value = mock_llm
        
        result = await agent.extract_forensic_insights(
            "Test Company", 
            "Test Article", 
            "This is a test article content with substantial text for analysis.",
            "Event 1"
        )
        
        assert result is not None
        assert "allegations" in result
        assert "entities" in result
        assert "raw_extract" in result
        assert "article_title" in result
        assert result["article_title"] == "Test Article"
        
        assert mock_llm.generate_text.call_count == 2


@pytest.mark.asyncio
async def test_process_article(agent):
    task = AnalysisTask(
        company="Test Company",
        event_name="Event 1",
        article_info={"title": "Test Article", "link": "https://example.com/1"},
        article_index=0,
        total_articles=1
    )
    
    with patch.object(agent, "content_parser_tool") as mock_parser, \
         patch.object(agent, "extract_forensic_insights") as mock_extract:
        
        mock_parser_result = MagicMock()
        mock_parser_result.success = True
        mock_parser_result.data = {"content": "Test content", "metadata": {}}
        mock_parser.run = AsyncMock(return_value=mock_parser_result)
        
        mock_extract.return_value = {"allegations": "Test"}
        
        result = await agent.process_article(task)
        
        assert result is not None
        assert "allegations" in result
        assert task.completed is True
        assert task.processing_time > 0
        assert agent.processing_stats["processed_articles"] == 1
        assert agent.processing_stats["articles_with_insights"] == 1


@pytest.mark.asyncio
async def test_process_event(agent):
    with patch.object(agent, "process_articles_batch") as mock_batch, \
         patch.object(agent, "synthesize_event_insights") as mock_synthesize:
        
        mock_batch.return_value = [{"allegations": "Test"}]
        mock_synthesize.return_value = {
            "cross_validation": "Test",
            "timeline": [{"date": "2023-01-01", "description": "Event"}],
            "key_entities": [{"name": "Entity1", "role": "Subject"}],
            "red_flags": ["Flag1"]
        }
        
        articles = [
            {"title": "Article 1", "link": "https://example.com/1"},
            {"title": "Article 2", "link": "https://example.com/2"}
        ]
        
        insights, synthesis = await agent.process_event("Test Company", "Event 1", articles)
        
        assert len(insights) == 1
        assert synthesis is not None
        assert "cross_validation" in synthesis
        assert "timeline" in synthesis
        assert "key_entities" in synthesis
        assert "red_flags" in synthesis
        assert "Event 1" in agent.result_tracker["events_analyzed"]
        assert "Entity1" in agent.result_tracker["entities_identified"]
        assert "Flag1" in agent.result_tracker["red_flags_found"]


@pytest.mark.asyncio
async def test_run(agent, state):
    with patch.object(agent, "process_events_concurrently") as mock_process, \
         patch.object(agent, "integrate_corporate_data"), \
         patch.object(agent, "integrate_youtube_data"), \
         patch.object(agent, "generate_company_analysis") as mock_analysis:
        
        event_synthesis = {
            "Event 1": {
                "cross_validation": "Test",
                "timeline": [{"date": "2023-01-01", "description": "Event"}],
                "key_entities": [{"name": "Entity1", "role": "Subject"}],
                "red_flags": ["Flag1"]
            }
        }
        mock_process.return_value = event_synthesis
        
        mock_analysis.return_value = {
            "executive_summary": "Test summary",
            "report_markdown": "# Test Report\n\nThis is a test report."
        }
        
        result = await agent.run(state)
        
        assert result["goto"] == "meta_agent"
        assert result["analyst_status"] == "DONE"
        assert "analysis_results" in result
        assert "final_report" in result
        assert result["analysis_results"]["event_synthesis"] == event_synthesis
        assert len(result["analysis_results"]["red_flags"]) > 0