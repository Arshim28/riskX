# tests/test_youtube_agent.py
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

from agents.youtube_agent import YouTubeAgent, VideoData


@pytest.fixture
def config():
    return {
        "youtube": {
            "youtube_api_key": "YOUTUBE_API_KEY_PLACEHOLDER"
        },
        "models": {
            "analysis": "gemini-2.0-pro",
            "summary": "gemini-2.0-pro"
        }
    }


@pytest.fixture
def state():
    return {
        "company": "Test Company",
        "industry": "Technology",
        "research_plan": [
            {
                "objective": "Test",
                "query_categories": {
                    "category1": ["query1", "query2"]
                }
            }
        ]
    }


@pytest.fixture
def video_data():
    return VideoData(
        video_id="test_id",
        title="Test Video",
        channel="Test Channel",
        description="Test Description",
        transcript="This is a test transcript with relevant information about Test Company."
    )


@pytest.fixture
def agent(config):
    with patch("utils.prompt_manager.get_prompt_manager"), \
         patch("utils.logging.get_logger"), \
         patch("tools.youtube_tool.YoutubeTool"):
        return YouTubeAgent(config)


@pytest.mark.asyncio
async def test_search_videos(agent):
    with patch.object(agent, "youtube_tool") as mock_tool:
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"videos": [{"id": "test_id", "title": "Test Video"}]}
        mock_tool.run = AsyncMock(return_value=mock_result)
        
        result = await agent.search_videos("Test Company fraud", max_results=5)
        
        assert result == [{"id": "test_id", "title": "Test Video"}]
        mock_tool.run.assert_called_with(
            action="search_videos",
            query="Test Company fraud",
            max_results=5
        )


@pytest.mark.asyncio
async def test_get_transcript(agent):
    with patch.object(agent, "youtube_tool") as mock_tool:
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"transcript": "This is a test transcript."}
        mock_tool.run = AsyncMock(return_value=mock_result)
        
        result = await agent.get_transcript("test_id")
        
        assert result == "This is a test transcript."
        mock_tool.run.assert_called_with(
            action="get_transcript",
            video_id="test_id"
        )


@pytest.mark.asyncio
async def test_analyze_transcript(agent, video_data):
    with patch("utils.llm_provider.get_llm_provider", AsyncMock()) as mock_provider:
        mock_llm = AsyncMock()
        mock_llm.generate_text = AsyncMock(return_value="""
        {
            "forensic_relevance": "medium",
            "red_flags": ["Potential accounting irregularity"],
            "summary": "The video discusses financial reporting issues."
        }
        """)
        mock_provider.return_value = mock_llm
        
        result = await agent.analyze_transcript(video_data, "Test Company")
        
        assert result["forensic_relevance"] == "medium"
        assert len(result["red_flags"]) == 1
        assert "financial reporting" in result["summary"].lower()
        assert video_data.forensic_summary is not None
        assert video_data.relevance_score == 0.6  # medium = 0.6
        
        mock_llm.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_generate_video_summary(agent):
    videos_data = [
        VideoData(
            video_id="test_id_1",
            title="Test Video 1",
            channel="Test Channel",
            description="Test Description 1",
            transcript="Transcript 1"
        ),
        VideoData(
            video_id="test_id_2",
            title="Test Video 2",
            channel="Test Channel",
            description="Test Description 2",
            transcript="Transcript 2"
        )
    ]
    
    videos_data[0].forensic_summary = {
        "forensic_relevance": "high",
        "red_flags": ["Flag 1"],
        "summary": "Summary 1"
    }
    videos_data[0].relevance_score = 0.9
    
    videos_data[1].forensic_summary = {
        "forensic_relevance": "low",
        "red_flags": ["Flag 2"],
        "summary": "Summary 2"
    }
    videos_data[1].relevance_score = 0.3
    
    with patch("utils.llm_provider.get_llm_provider", AsyncMock()) as mock_provider:
        mock_llm = AsyncMock()
        mock_llm.generate_text = AsyncMock(return_value="""
        {
            "overall_assessment": "Some concerns found",
            "key_insights": ["The company has been mentioned in negative context"],
            "red_flags": ["Financial reporting issues", "Management concerns"],
            "notable_videos": ["test_id_1"],
            "summary": "Analysis reveals potential issues that warrant further investigation."
        }
        """)
        mock_provider.return_value = mock_llm
        
        result = await agent.generate_video_summary(videos_data, "Test Company")
        
        assert result["overall_assessment"] == "Some concerns found"
        assert len(result["key_insights"]) == 1
        assert len(result["red_flags"]) == 2
        assert len(result["notable_videos"]) == 1
        assert "investigation" in result["summary"].lower()
        
        mock_llm.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_run(agent, state):
    with patch.object(agent, "search_videos", AsyncMock()) as mock_search, \
         patch.object(agent, "get_channel_videos", AsyncMock()) as mock_channel, \
         patch.object(agent, "get_transcript", AsyncMock()) as mock_transcript, \
         patch.object(agent, "analyze_transcript", AsyncMock()) as mock_analyze, \
         patch.object(agent, "generate_video_summary", AsyncMock()) as mock_summary:
        
        # Set up mock returns
        mock_search.return_value = [
            {"id": "test_id_1", "title": "Test Video 1", "channel_title": "Channel 1", "description": "Desc 1"}
        ]
        
        mock_channel.return_value = [
            {"id": "test_id_2", "title": "Test Video 2", "channel_title": "Channel 1", "description": "Desc 2"}
        ]
        
        mock_transcript.return_value = "This is a test transcript."
        
        mock_analyze.return_value = {
            "forensic_relevance": "medium",
            "red_flags": ["Potential issue"],
            "summary": "Test summary."
        }
        
        mock_summary.return_value = {
            "overall_assessment": "Some concerns",
            "key_insights": ["Insight 1"],
            "red_flags": ["Flag 1", "Flag 2"],
            "notable_videos": ["test_id_1"],
            "summary": "Overall summary."
        }
        
        result = await agent.run(state)
        
        assert result["goto"] == "meta_agent"
        assert result["youtube_status"] == "DONE"
        assert "youtube_results" in result
        assert "videos" in result["youtube_results"]
        assert "channels" in result["youtube_results"]
        assert "summary" in result["youtube_results"]
        assert "red_flags" in result["youtube_results"]
        assert len(result["youtube_results"]["red_flags"]) == 2
        
        mock_search.assert_called()
        mock_channel.assert_called()
        mock_transcript.assert_called()
        mock_analyze.assert_called()
        mock_summary.assert_called_once()