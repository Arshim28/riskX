import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import logging
from typing import Dict, List, Any, Optional

from tools.youtube_tool import YoutubeTool, YoutubeToolConfig
from base.base_tools import ToolResult


# Mock data for tests
YOUTUBE_API_KEY = "YOUTUBE_API_KEY_PLACEHOLDER"
TEST_VIDEO_ID = "abc12345"
TEST_CHANNEL_ID = "channel123"
TEST_CHANNEL_NAME = "Test Channel"
TEST_PLAYLIST_ID = "playlist123"
TEST_QUERY = "test search query"

# Mock response data
MOCK_TRANSCRIPT = "This is a test transcript for video content."

MOCK_VIDEO_DETAILS = {
    "title": "Test Video Title",
    "description": "Test video description",
    "published_at": "2023-01-01T12:00:00Z",
    "channel_id": TEST_CHANNEL_ID,
    "channel_title": "Test Channel",
    "duration": "PT10M30S",
    "view_count": 1000,
    "like_count": 100,
    "comment_count": 50
}

MOCK_TRANSCRIPT_RESPONSE = [
    {"text": "This is a test", "start": 0.0, "duration": 2.0},
    {"text": "transcript for video content.", "start": 2.0, "duration": 3.0}
]

MOCK_SEARCH_RESULTS = [
    {
        "id": {"videoId": "video1"},
        "snippet": {
            "title": "Test Video 1",
            "description": "Description for test video 1",
            "channelTitle": "Test Channel",
            "publishedAt": "2023-01-01T12:00:00Z"
        }
    },
    {
        "id": {"videoId": "video2"},
        "snippet": {
            "title": "Test Video 2",
            "description": "Description for test video 2",
            "channelTitle": "Test Channel",
            "publishedAt": "2023-01-02T12:00:00Z"
        }
    }
]

MOCK_SEARCH_RESULTS = [
    {
        "id": {"videoId": "video1"},
        "snippet": {
            "title": "Test Video 1",
            "description": "Description for test video 1",
            "channelTitle": "Test Channel",
            "publishedAt": "2023-01-01T12:00:00Z"
        }
    },
    {
        "id": {"videoId": "video2"},
        "snippet": {
            "title": "Test Video 2",
            "description": "Description for test video 2",
            "channelTitle": "Test Channel",
            "publishedAt": "2023-01-02T12:00:00Z"
        }
    }
]

MOCK_CHANNEL_RESULTS = [
    {
        "id": {"channelId": TEST_CHANNEL_ID},
        "snippet": {
            "title": TEST_CHANNEL_NAME,
            "description": "Test channel description"
        }
    }
]

MOCK_PLAYLIST_RESULTS = [
    {
        "id": "playlist1",
        "snippet": {
            "title": "Test Playlist 1",
            "description": "Description for test playlist 1"
        },
        "contentDetails": {"itemCount": 10}
    },
    {
        "id": "playlist2",
        "snippet": {
            "title": "Test Playlist 2",
            "description": "Description for test playlist 2"
        },
        "contentDetails": {"itemCount": 5}
    }
]

MOCK_PLAYLIST_ITEMS = [
    {
        "snippet": {
            "title": "Playlist Video 1",
            "description": "Description for playlist video 1",
            "publishedAt": "2023-01-01T12:00:00Z"
        },
        "contentDetails": {"videoId": "playlistVideo1"}
    },
    {
        "snippet": {
            "title": "Playlist Video 2",
            "description": "Description for playlist video 2",
            "publishedAt": "2023-01-02T12:00:00Z"
        },
        "contentDetails": {"videoId": "playlistVideo2"}
    }
]

MOCK_VIDEO_RESPONSE = {
    "items": [
        {
            "snippet": {
                "title": MOCK_VIDEO_DETAILS["title"],
                "description": MOCK_VIDEO_DETAILS["description"],
                "publishedAt": MOCK_VIDEO_DETAILS["published_at"],
                "channelId": MOCK_VIDEO_DETAILS["channel_id"],
                "channelTitle": MOCK_VIDEO_DETAILS["channel_title"]
            },
            "contentDetails": {
                "duration": MOCK_VIDEO_DETAILS["duration"]
            },
            "statistics": {
                "viewCount": str(MOCK_VIDEO_DETAILS["view_count"]),
                "likeCount": str(MOCK_VIDEO_DETAILS["like_count"]),
                "commentCount": str(MOCK_VIDEO_DETAILS["comment_count"])
            }
        }
    ]
}


class MockYouTubeClient:
    """Mock for the Google YouTube API client"""
    
    def __init__(self):
        # Create mock services
        self.videos = MagicMock()
        self.search = MagicMock()
        self.playlists = MagicMock()
        self.playlistItems = MagicMock()
        
        # Set up mock list methods with execute methods
        self.videos().list = MagicMock()
        self.videos().list().execute = MagicMock(return_value=MOCK_VIDEO_RESPONSE)
        
        self.search().list = MagicMock()
        self.search().list().execute = MagicMock(return_value={"items": MOCK_SEARCH_RESULTS})
        
        self.playlists().list = MagicMock()
        self.playlists().list().execute = MagicMock(return_value={"items": MOCK_PLAYLIST_RESULTS})
        
        self.playlistItems().list = MagicMock()
        self.playlistItems().list().execute = MagicMock(return_value={"items": MOCK_PLAYLIST_ITEMS})


class MockTranscriptAPI:
    """Not using this anymore, but leaving it for reference"""
    pass


@pytest.fixture
def mock_logger():
    """Fixture for mocking the logger"""
    mock_log = MagicMock(spec=logging.Logger)
    
    with patch("utils.logging.get_logger") as mock_get_logger:
        mock_get_logger.return_value = mock_log
        yield mock_log


@pytest.fixture
def mock_youtube_client():
    """Fixture for mocking the YouTube API client"""
    return MockYouTubeClient()


@pytest.fixture
def mock_transcript_api():
    """No longer needed with our approach, but kept for compatibility"""
    return None


@pytest_asyncio.fixture
async def youtube_tool(mock_logger, mock_youtube_client):
    """Fixture to create a YouTubeTool with mocked dependencies"""
    
    # Use a placeholder API key for testing
    config = {
        "youtube": {
            "youtube_api_key": "YOUTUBE_API_KEY_PLACEHOLDER",
            "retry_limit": 1,  # Low values for testing
            "multiplier": 0,
            "min_wait": 0,
            "max_wait": 0
        }
    }
    
    # Create a direct mock for the transcriptor get_transcript method
    transcript_mock = Mock()
    transcript_mock.get_transcript.return_value = MOCK_TRANSCRIPT_RESPONSE
    
    # Patch the build function and the YouTubeTranscriptApi class
    with patch("googleapiclient.discovery.build", return_value=mock_youtube_client), \
         patch.object(YoutubeTool, "_update_retry_params"):  # Skip retry params setup
        
        # Instantiate the tool
        tool = YoutubeTool(config)
        tool.logger = mock_logger
        
        # Directly replace the transcriptor with our mock
        tool.transcriptor = transcript_mock
        
        yield tool


@pytest.mark.asyncio
async def test_get_transcript(youtube_tool):
    """Test fetching video transcript"""
    # Make the transcript get_transcript method awaitable
    youtube_tool.transcriptor.get_transcript = AsyncMock(return_value=MOCK_TRANSCRIPT_RESPONSE)
    
    # Call the method directly
    transcript = await youtube_tool.get_transcript(TEST_VIDEO_ID)
    
    # Verify the result - should match our expected transcript string
    assert transcript == MOCK_TRANSCRIPT
    
    # Verify that the transcript mock was called
    youtube_tool.transcriptor.get_transcript.assert_called_once_with(TEST_VIDEO_ID)


@pytest.mark.asyncio
async def test_get_video_details(youtube_tool, mock_youtube_client):
    """Test fetching video details"""
    # Mock the run_in_executor function
    with patch("asyncio.get_event_loop") as mock_loop:
        # Set up the mock executor with AsyncMock
        mock_executor = AsyncMock()
        mock_executor.return_value = MOCK_VIDEO_RESPONSE
        mock_loop.return_value.run_in_executor = mock_executor
        
        # Call the method
        details = await youtube_tool.get_video_details(TEST_VIDEO_ID)
        
        # Verify the result
        assert details.title == MOCK_VIDEO_DETAILS["title"]
        assert details.description == MOCK_VIDEO_DETAILS["description"]
        assert details.published_at == MOCK_VIDEO_DETAILS["published_at"]
        assert details.channel_id == MOCK_VIDEO_DETAILS["channel_id"]
        assert details.view_count == MOCK_VIDEO_DETAILS["view_count"]


@pytest.mark.asyncio
async def test_search_videos(youtube_tool, mock_youtube_client):
    """Test searching for videos"""
    # Mock the run_in_executor function
    with patch("asyncio.get_event_loop") as mock_loop:
        # Set up the mock executor with AsyncMock
        mock_executor = AsyncMock()
        mock_executor.return_value = {"items": MOCK_SEARCH_RESULTS}
        mock_loop.return_value.run_in_executor = mock_executor
        
        # Call the method
        videos = await youtube_tool.search_videos(TEST_QUERY, max_results=2)
        
        # Verify the result
        assert len(videos) == 2
        assert videos[0].id == "video1"
        assert videos[0].title == "Test Video 1"
        assert videos[1].id == "video2"


@pytest.mark.asyncio
async def test_get_channel_id_by_name(youtube_tool, mock_youtube_client):
    """Test getting channel ID by name"""
    # Mock the run_in_executor function
    with patch("asyncio.get_event_loop") as mock_loop:
        # Set up the mock executor with AsyncMock
        mock_executor = AsyncMock()
        mock_executor.return_value = {"items": MOCK_CHANNEL_RESULTS}
        mock_loop.return_value.run_in_executor = mock_executor
        
        # Call the method
        channel_id = await youtube_tool.get_channel_id_by_name(TEST_CHANNEL_NAME)
        
        # Verify the result
        assert channel_id == TEST_CHANNEL_ID


@pytest.mark.asyncio
async def test_get_playlists_from_channel(youtube_tool, mock_youtube_client):
    """Test getting playlists from a channel"""
    # Mock the run_in_executor function
    with patch("asyncio.get_event_loop") as mock_loop:
        # Set up the mock executor with AsyncMock
        mock_executor = AsyncMock()
        mock_executor.return_value = {"items": MOCK_PLAYLIST_RESULTS}
        mock_loop.return_value.run_in_executor = mock_executor
        
        # Call the method
        playlists = await youtube_tool.get_playlists_from_channel(TEST_CHANNEL_ID)
        
        # Verify the result
        assert len(playlists) == 2
        assert playlists[0].id == "playlist1"
        assert playlists[0].title == "Test Playlist 1"
        assert playlists[0].video_count == 10


@pytest.mark.asyncio
async def test_get_videos_from_playlist(youtube_tool, mock_youtube_client):
    """Test getting videos from a playlist"""
    # Mock the run_in_executor function
    with patch("asyncio.get_event_loop") as mock_loop:
        # Set up the mock executor with AsyncMock
        mock_executor = AsyncMock()
        mock_executor.return_value = {"items": MOCK_PLAYLIST_ITEMS}
        mock_loop.return_value.run_in_executor = mock_executor
        
        # Call the method
        videos = await youtube_tool.get_videos_from_playlist(TEST_PLAYLIST_ID)
        
        # Verify the result
        assert len(videos) == 2
        assert videos[0].id == "playlistVideo1"
        assert videos[0].title == "Playlist Video 1"


@pytest.mark.asyncio
async def test_run_search_videos(youtube_tool):
    with patch.object(youtube_tool, "search_videos") as mock_search:
        # Return VideoData objects as the actual method would
        from tools.youtube_tool import VideoData
        mock_search.return_value = [
            VideoData(id="video1", title="Test Video 1", description="Description 1", published_at="2023-01-01"),
            VideoData(id="video2", title="Test Video 2", description="Description 2", published_at="2023-01-02")
        ]
        
        # Call the run method
        result = await youtube_tool.run(
            action="search_videos",
            query=TEST_QUERY,
            max_results=2
        )
        
        # Verify the result
        assert result.success is True
        assert "videos" in result.data
        assert len(result.data["videos"]) == 2
        mock_search.assert_called_once_with(TEST_QUERY, 2, "relevance")


@pytest.mark.asyncio
async def test_run_get_transcript(youtube_tool):
    """Test the run method with get_transcript action"""
    # Mock the get_transcript method
    with patch.object(youtube_tool, "get_transcript") as mock_transcript:
        mock_transcript.return_value = MOCK_TRANSCRIPT
        
        # Call the run method
        result = await youtube_tool.run(
            action="get_transcript",
            video_id=TEST_VIDEO_ID
        )
        
        # Verify the result
        assert result.success is True
        assert "transcript" in result.data
        assert result.data["transcript"] == MOCK_TRANSCRIPT
        mock_transcript.assert_called_once_with(TEST_VIDEO_ID)


@pytest.mark.asyncio
async def test_run_get_video_details(youtube_tool):
    """Test the run method with get_video_details action"""
    # Mock the get_video_details method
    with patch.object(youtube_tool, "get_video_details") as mock_details:
        mock_details.return_value = MagicMock(model_dump=MagicMock(return_value=MOCK_VIDEO_DETAILS))
        
        # Call the run method
        result = await youtube_tool.run(
            action="get_video_details",
            video_id=TEST_VIDEO_ID
        )
        
        # Verify the result
        assert result.success is True
        assert "details" in result.data
        mock_details.assert_called_once_with(TEST_VIDEO_ID)


@pytest.mark.asyncio
async def test_run_get_channel_id(youtube_tool):
    """Test the run method with get_channel_id action"""
    # Mock the get_channel_id_by_name method
    with patch.object(youtube_tool, "get_channel_id_by_name") as mock_channel:
        mock_channel.return_value = TEST_CHANNEL_ID
        
        # Call the run method
        result = await youtube_tool.run(
            action="get_channel_id",
            channel_name=TEST_CHANNEL_NAME
        )
        
        # Verify the result
        assert result.success is True
        assert "channel_id" in result.data
        assert result.data["channel_id"] == TEST_CHANNEL_ID
        mock_channel.assert_called_once_with(TEST_CHANNEL_NAME)


@pytest.mark.asyncio
async def test_run_invalid_action(youtube_tool):
    """Test the run method with an invalid action"""
    result = await youtube_tool.run(
        action="invalid_action",
        video_id=TEST_VIDEO_ID
    )
    
    # Verify the result
    assert result.success is False
    assert result.error is not None
    assert "Unknown action" in result.error


@pytest.mark.asyncio
async def test_run_missing_required_param(youtube_tool):
    """Test the run method with missing required parameter"""
    result = await youtube_tool.run(
        action="get_transcript",
        # Missing video_id parameter
    )
    
    # Verify the result
    assert result.success is False
    assert result.error is not None
    assert "required" in result.error.lower()


@pytest.mark.asyncio
async def test_api_error_handling(youtube_tool):
    """Test handling of API errors"""
    # Mock the get_transcript method to raise an exception
    with patch.object(youtube_tool, "get_transcript") as mock_transcript:
        mock_transcript.side_effect = Exception("API Error")
        
        # Call the run method
        result = await youtube_tool.run(
            action="get_transcript",
            video_id=TEST_VIDEO_ID
        )
        
        # Verify the result
        assert result.success is False
        assert result.error is not None
        assert "API Error" in result.error