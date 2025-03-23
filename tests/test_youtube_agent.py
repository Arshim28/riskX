# tests/test_youtube_agent.py
import pytest
import asyncio
import json
import datetime
from unittest.mock import patch, MagicMock, AsyncMock, call
from copy import deepcopy

# Import error classes directly
from agents.youtube_agent import (
    YouTubeError, YouTubeConnectionError, YouTubeRateLimitError, 
    YouTubeDataError, YouTubeValidationError, VideoData, YouTubeAgent
)

@pytest.fixture
def config():
    return {
        "youtube": {"api_key": "test_key"},
        "models": {
            "analysis": "gemini-2.0-pro",
            "summary": "gemini-2.0-pro"
        },
        "retry": {
            "max_attempts": 3,
            "multiplier": 1,
            "min_wait": 2,
            "max_wait": 10
        }
    }

@pytest.fixture
def state():
    return {
        "company": "Test Company",
        "industry": "Technology",
        "goto": "youtube_agent",
        "research_plan": [{
            "query_categories": {
                "general": "test query",
                "controversy": "test controversy"
            }
        }]
    }

@pytest.fixture
def agent(config):
    # Create mock for LLM provider
    llm_mock = MagicMock()
    llm_mock.generate_text = AsyncMock()
    llm_provider_mock = AsyncMock(return_value=llm_mock)
    
    with patch("utils.prompt_manager.get_prompt_manager"), \
         patch("utils.logging.get_logger"), \
         patch("tools.youtube_tool.YoutubeTool"), \
         patch("agents.youtube_agent.get_llm_provider", return_value=llm_provider_mock):
        
        from agents.youtube_agent import YouTubeAgent
        return YouTubeAgent(config)

@pytest.fixture
def video_data():
    return VideoData(
        video_id="test_id",
        title="Test Video",
        channel="Test Channel",
        description="Test Description",
        transcript="This is a test transcript with content about the company."
    )

@pytest.mark.asyncio
async def test_search_videos_success(agent, state):
    # Mock YouTube tool response
    videos = [
        {"id": "video1", "title": "Test Video 1", "channel_title": "Channel 1"},
        {"id": "video2", "title": "Test Video 2", "channel_title": "Channel 2"}
    ]
    
    # Create a proper tool result
    result_mock = MagicMock()
    result_mock.success = True
    result_mock.data = {"videos": videos}
    agent.youtube_tool.run.return_value = result_mock
    
    result = await agent.search_videos("Test Company", max_results=2)
    
    assert len(result) == 2
    assert result[0]["id"] == "video1"
    assert result[1]["id"] == "video2"
    assert agent.youtube_tool.run.called
    agent.youtube_tool.run.assert_called_with(
        action="search_videos",
        query="Test Company",
        max_results=2
    )

@pytest.mark.asyncio
async def test_search_videos_api_error(agent, state):
    # Test error handling for API failure
    result_mock = MagicMock()
    result_mock.success = False
    result_mock.error = "API error"
    agent.youtube_tool.run.return_value = result_mock
    
    with pytest.raises(YouTubeDataError):
        await agent.search_videos("Test Company")

@pytest.mark.asyncio
async def test_get_transcript_success(agent, state):
    # Mock successful transcript retrieval
    result_mock = MagicMock()
    result_mock.success = True
    result_mock.data = {"transcript": "Test transcript"}
    agent.youtube_tool.run.return_value = result_mock
    
    result = await agent.get_transcript("video1")
    
    assert result == "Test transcript"
    assert agent.youtube_tool.run.called
    agent.youtube_tool.run.assert_called_with(
        action="get_transcript",
        video_id="video1"
    )

@pytest.mark.asyncio
async def test_get_transcript_not_available(agent, state):
    # Test handling of missing transcript
    result_mock = MagicMock()
    result_mock.success = False
    result_mock.error = "No transcript"
    agent.youtube_tool.run.return_value = result_mock
    
    result = await agent.get_transcript("video1")
    
    assert result is None

@pytest.mark.asyncio
async def test_get_transcript_api_error(agent, state):
    # Test error handling for API failure
    result_mock = MagicMock()
    result_mock.success = False
    result_mock.error = "Rate limit exceeded"
    agent.youtube_tool.run.return_value = result_mock
    
    # This should try to handle the error and return None instead of raising
    result = await agent.get_transcript("video1")
    assert result is None

@pytest.mark.asyncio
async def test_analyze_transcript_success(agent, video_data):
    # Mock LLM response
    llm_provider_mock = AsyncMock()
    llm_provider_mock.generate_text.return_value = '{"forensic_relevance": "high", "red_flags": ["Issue found"], "summary": "Test summary"}'
    
    with patch("agents.youtube_agent.get_llm_provider", return_value=AsyncMock(return_value=llm_provider_mock)):
        result = await agent.analyze_transcript(video_data, "Test Company")
        
        assert result["forensic_relevance"] == "high"
        assert len(result["red_flags"]) == 1
        assert result["summary"] == "Test summary"
        assert result["video_id"] == video_data.video_id
        assert result["title"] == video_data.title
        assert video_data.forensic_summary == result
        assert video_data.relevance_score == 0.9  # high relevance

@pytest.mark.asyncio
async def test_analyze_transcript_no_transcript(agent, video_data):
    # Test handling when no transcript is available
    video_data.transcript = None
    
    result = await agent.analyze_transcript(video_data, "Test Company")
    
    assert result["forensic_relevance"] == "low"
    assert len(result["red_flags"]) == 0
    assert "No transcript available" in result["summary"]
    assert video_data.relevance_score == 0.3  # low relevance

@pytest.mark.asyncio
async def test_analyze_transcript_json_error(agent, video_data):
    # Test error handling for LLM returning invalid JSON
    llm_provider_mock = AsyncMock()
    llm_provider_mock.generate_text.return_value = 'Invalid JSON'
    
    with patch("agents.youtube_agent.get_llm_provider", return_value=AsyncMock(return_value=llm_provider_mock)):
        # Should handle error and return a default result instead of raising
        result = await agent.analyze_transcript(video_data, "Test Company")
        
        assert "forensic_relevance" in result
        assert "red_flags" in result
        assert "summary" in result
        assert "Error" in result["summary"] or result["forensic_relevance"] == "unknown"

@pytest.mark.asyncio
async def test_get_channel_videos_success(agent, state):
    # Mock responses for channel ID and playlists
    channel_id = "channel1"
    playlists = [{"id": "playlist1", "title": "Playlist 1"}]
    playlist_videos = [
        {"id": "video1", "title": "Video 1"},
        {"id": "video2", "title": "Video 2"}
    ]
    
    # Create proper tool results
    channel_result = MagicMock()
    channel_result.success = True
    channel_result.data = {"channel_id": channel_id}
    
    playlists_result = MagicMock()
    playlists_result.success = True
    playlists_result.data = {"playlists": playlists}
    
    videos_result = MagicMock()
    videos_result.success = True
    videos_result.data = {"videos": playlist_videos}
    
    # Configure side_effect to return different results for each call
    agent.youtube_tool.run.side_effect = [
        channel_result,
        playlists_result,
        videos_result
    ]
    
    result = await agent.get_channel_videos("Test Channel", max_results=5)
    
    assert len(result) == 2
    assert result[0]["id"] == "video1"
    assert result[1]["id"] == "video2"
    assert agent.youtube_tool.run.call_count == 3

@pytest.mark.asyncio
async def test_get_channel_videos_fallback_to_search(agent, state):
    # Test fallback to search when playlist data is insufficient
    channel_id = "channel1"
    search_videos = [
        {"id": "video3", "title": "Video 3"},
        {"id": "video4", "title": "Video 4"}
    ]
    
    # Create proper tool results
    channel_result = MagicMock()
    channel_result.success = True
    channel_result.data = {"channel_id": channel_id}
    
    playlists_result = MagicMock()
    playlists_result.success = True
    playlists_result.data = {"playlists": []}  # Empty playlists
    
    search_result = MagicMock()
    search_result.success = True
    search_result.data = {"videos": search_videos}  # Search results
    
    # Configure side_effect to return different results for each call
    agent.youtube_tool.run.side_effect = [
        channel_result,
        playlists_result,
        search_result
    ]
    
    result = await agent.get_channel_videos("Test Channel", max_results=5)
    
    assert len(result) == 2
    assert result[0]["id"] == "video3"
    assert result[1]["id"] == "video4"
    assert agent.youtube_tool.run.call_count == 3

@pytest.mark.asyncio
async def test_generate_video_summary_success(agent, state):
    # Setup test video data
    videos = []
    for i in range(3):
        video = VideoData(
            video_id=f"video{i}",
            title=f"Video {i}",
            channel="Test Channel",
            description="Description",
            transcript="Transcript"
        )
        video.forensic_summary = {
            "forensic_relevance": "medium",
            "red_flags": [f"Issue {i}"],
            "summary": f"Summary {i}"
        }
        video.relevance_score = 0.6
        videos.append(video)
    
    # Mock LLM response
    llm_provider_mock = AsyncMock()
    llm_provider_mock.generate_text.return_value = '{"overall_assessment": "Concerning", "key_insights": ["Insight 1"], "red_flags": ["Major issue"], "notable_videos": ["video0"], "summary": "Overall summary"}'
    
    with patch("agents.youtube_agent.get_llm_provider", return_value=AsyncMock(return_value=llm_provider_mock)):
        result = await agent.generate_video_summary(videos, "Test Company")
        
        assert result["overall_assessment"] == "Concerning"
        assert len(result["key_insights"]) == 1
        assert len(result["red_flags"]) == 1
        assert len(result["notable_videos"]) == 1
        assert result["summary"] == "Overall summary"
        assert "timestamp" in result
        assert result["company"] == "Test Company"
        assert result["total_videos_analyzed"] == 3

@pytest.mark.asyncio
async def test_generate_video_summary_error(agent, state):
    # Test error handling in summary generation
    videos = [VideoData(
        video_id="video1",
        title="Video 1",
        channel="Test Channel",
        description="Description",
        transcript=None  # No transcript
    )]
    
    # Mock LLM error
    llm_provider_mock = AsyncMock()
    llm_provider_mock.generate_text.side_effect = Exception("LLM error")
    
    with patch("agents.youtube_agent.get_llm_provider", return_value=AsyncMock(return_value=llm_provider_mock)):
        # Should handle error and return a default result
        result = await agent.generate_video_summary(videos, "Test Company")
        
        assert "overall_assessment" in result
        assert "key_insights" in result
        assert "red_flags" in result
        assert "Error" in result["overall_assessment"] or "Failed" in " ".join(result["key_insights"])

@pytest.mark.asyncio
async def test_run_success_path(agent, state):
    # Mock all component methods
    search_videos = [{"id": "video1", "title": "Video 1", "channel_title": "Channel 1"}]
    channel_videos = [{"id": "video2", "title": "Video 2", "channel_title": "Channel 1"}]
    
    # Configure search_videos responses
    agent.search_videos = AsyncMock(return_value=search_videos)
    agent.get_channel_videos = AsyncMock(return_value=channel_videos)
    agent.get_transcript = AsyncMock(return_value="Test transcript")
    agent.analyze_transcript = AsyncMock(return_value={
        "forensic_relevance": "high",
        "red_flags": ["Issue"],
        "summary": "Summary",
        "video_id": "video1",
        "title": "Video 1"
    })
    agent.generate_video_summary = AsyncMock(return_value={
        "overall_assessment": "Concerning",
        "key_insights": ["Insight"],
        "red_flags": ["Major issue"],
        "notable_videos": ["video1"],
        "summary": "Overall summary",
        "timestamp": datetime.datetime.now().isoformat(),
        "company": "Test Company",
        "total_videos_analyzed": 2
    })
    
    # Mock logging methods
    agent._log_start = MagicMock()
    agent._log_completion = MagicMock()
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["youtube_status"] == "DONE"
    assert "youtube_results" in result
    assert len(result["youtube_results"]["red_flags"]) == 1
    assert "videos" in result["youtube_results"]
    assert agent._log_completion.called

@pytest.mark.asyncio
async def test_run_with_search_errors(agent, state):
    # Test run continuing despite search errors
    agent.search_videos = AsyncMock(side_effect=agent.YouTubeDataError("API error"))
    agent.get_channel_videos = AsyncMock(return_value=[])
    agent.generate_video_summary = AsyncMock(return_value={
        "overall_assessment": "Error",
        "key_insights": ["No data"],
        "red_flags": [],
        "notable_videos": [],
        "summary": "No data available",
        "timestamp": datetime.datetime.now().isoformat(),
        "company": "Test Company",
        "total_videos_analyzed": 0
    })
    
    # Mock logging methods
    agent._log_start = MagicMock()
    agent._log_completion = MagicMock()
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["youtube_status"] == "DONE"
    assert "youtube_results" in result
    assert "errors" in result["youtube_results"]
    assert "search_errors" in result["youtube_results"]["errors"]

@pytest.mark.asyncio
async def test_run_no_research_plan(agent, state):
    # Test handling when no research plan is provided
    state.pop("research_plan")
    agent.search_videos = AsyncMock(return_value=[])
    agent.get_channel_videos = AsyncMock(return_value=[])
    agent.generate_video_summary = AsyncMock(return_value={
        "overall_assessment": "Unknown",
        "key_insights": [],
        "red_flags": [],
        "notable_videos": [],
        "summary": "No data",
        "timestamp": datetime.datetime.now().isoformat(),
        "company": "Test Company",
        "total_videos_analyzed": 0
    })
    
    # Mock logging methods
    agent._log_start = MagicMock()
    agent._log_completion = MagicMock()
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["youtube_status"] == "DONE"
    assert "youtube_results" in result
    # Verify default queries were used
    assert len(result["youtube_results"]["queries"]) >= 3

@pytest.mark.asyncio
async def test_run_critical_error(agent, state):
    # Test handling of critical errors that affect the entire run
    agent._log_start = AsyncMock(side_effect=Exception("Critical error"))
    
    result = await agent.run(state)
    
    assert result["goto"] == "meta_agent"
    assert result["youtube_status"] == "ERROR"
    assert "error" in result

@pytest.mark.asyncio
async def test_parse_json_response_success(agent, state):
    # Test normal JSON parsing
    response = '{"key": "value"}'
    result = agent._parse_json_response(response)
    assert result == {"key": "value"}
    
    # Test with code block
    response = '```json\n{"key": "value"}\n```'
    result = agent._parse_json_response(response)
    assert result == {"key": "value"}

@pytest.mark.asyncio
async def test_parse_json_response_error(agent, state):
    # Test error handling for invalid JSON
    response = 'Invalid JSON'
    with pytest.raises(YouTubeDataError):
        agent._parse_json_response(response)

@pytest.mark.asyncio
async def test_validate_result_success(agent, state):
    # Test validation passes with all required fields
    result = {"field1": "value1", "field2": "value2"}
    agent._validate_result(result, ["field1", "field2"])
    # No exception means validation passed

@pytest.mark.asyncio
async def test_validate_result_error(agent, state):
    # Test validation failure with missing fields
    result = {"field1": "value1"}
    with pytest.raises(YouTubeValidationError):
        agent._validate_result(result, ["field1", "field2"])

@pytest.mark.asyncio
async def test_sanitize_transcript(agent, state):
    # Test transcript sanitization
    # Normal case
    transcript = "This is a normal transcript."
    result = agent._sanitize_transcript(transcript)
    assert result == transcript
    
    # Long transcript
    long_transcript = "Word " * 20000  # Will exceed max length
    result = agent._sanitize_transcript(long_transcript)
    assert len(result) <= 32003  # Max length + "..."
    assert result.endswith("...")
    
    # None case
    result = agent._sanitize_transcript(None)
    assert result is None
    
    # Extra whitespace
    transcript = "  This  has\nextra \n\n whitespace  "
    result = agent._sanitize_transcript(transcript)
    assert result == "This has extra whitespace"

@pytest.mark.asyncio
async def test_get_retry_decorator(agent, state):
    # Test retry decorator creation
    decorator = agent._get_retry_decorator("test_operation")
    assert callable(decorator)
    
    # Apply decorator to a function and test
    @decorator
    async def test_func():
        return "success"
    
    result = await test_func()
    assert result == "success"

@pytest.mark.asyncio
async def test_retry_mechanism(agent, state):
    # Test the retry mechanism in action
    call_count = 0
    
    # Create a function that fails twice then succeeds
    @agent._get_retry_decorator("test_retry")
    async def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise YouTubeConnectionError("Temporary connection error")
        return "success"
    
    # Function should succeed after retries
    result = await flaky_function()
    assert result == "success"
    assert call_count == 3