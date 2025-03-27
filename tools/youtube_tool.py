import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

import googleapiclient.discovery
import googleapiclient.errors
from youtube_transcript_api import YouTubeTranscriptApi
from utils.logging import get_logger
from base.base_tools import BaseTool, ToolResult

RETRY_LIMIT = 3
MULTIPLIER = 1
MIN_WAIT = 2
MAX_WAIT = 10

class VideoData(BaseModel):
    id: str
    title: str
    description: str
    published_at: str
    channel_title: Optional[str] = None


class PlaylistData(BaseModel):
    id: str
    title: str
    description: str
    video_count: int


class VideoDetails(BaseModel):
    title: str
    description: str
    published_at: str
    channel_id: str
    channel_title: str
    duration: str
    view_count: int
    like_count: int
    comment_count: int

class YoutubeToolConfig(BaseModel):
    youtube_api_key: str
    retry_limit: int = RETRY_LIMIT
    multiplier: int = MULTIPLIER
    min_wait: int = MIN_WAIT
    max_wait: int = MAX_WAIT


class YoutubeTool(BaseTool):
    name = "youtube_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.config = YoutubeToolConfig(**config)
        

        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=self.config.youtube_api_key
        )
        self.transcriptor = YouTubeTranscriptApi()
        self._update_retry_params()
    
    def _update_retry_params(self):
        global RETRY_LIMIT, MULTIPLIER, MIN_WAIT, MAX_WAIT
        RETRY_LIMIT = self.config.retry_limit
        MULTIPLIER = self.config.multiplier
        MIN_WAIT = self.config.min_wait
        MAX_WAIT = self.config.max_wait
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def get_transcript(self, video_id: str) -> Optional[str]:
        try:
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None,
                lambda: self.transcriptor.get_transcript(video_id)
            )

            self.logger.info(f"Successfully fetched transcript for video: {video_id}")

            # Process transcript depending on its format
            # Log the transcript format for debugging
            self.logger.info(f"Raw transcript type: {type(transcript)}")
            if transcript:
                if isinstance(transcript, list):
                    self.logger.info(f"Transcript is a list of length {len(transcript)}")
                    if transcript and isinstance(transcript[0], dict):
                        self.logger.info("Transcript is a list of dictionaries")
                        full_text = " ".join(snippet.get('text', '') for snippet in transcript)
                        self.logger.debug(f"Processed transcript preview: {full_text[:200]}...")
                        return full_text
                    else:
                        self.logger.info("Transcript is a list of non-dictionary objects")
                        full_text = " ".join(getattr(snippet, 'text', '') for snippet in transcript)
                        self.logger.debug(f"Processed transcript preview: {full_text[:200]}...")
                        return full_text
                # If it's already a string, return it directly
                elif isinstance(transcript, str):
                    self.logger.info("Transcript is already a string")
                    self.logger.debug(f"Transcript preview: {transcript[:200]}...")
                    return transcript
                else:
                    self.logger.warning(f"Unexpected transcript format: {type(transcript)}")
                    # Try to convert to string as a fallback
                    try:
                        transcript_str = str(transcript)
                        self.logger.info(f"Converted transcript to string, length: {len(transcript_str)}")
                        self.logger.debug(f"Converted transcript preview: {transcript_str[:200]}...")
                        return transcript_str
                    except Exception as e:
                        self.logger.error(f"Failed to convert transcript to string: {e}")
                        return None

            return None
        except Exception as e:
            self.logger.error(f"Error fetching transcript: {e}")
            return None
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def get_playlists_from_channel(self, channel_id: str, max_results: int = 50) -> List[PlaylistData]:
        playlists = []
        next_page_token = None
        
        try:
            loop = asyncio.get_event_loop()
            
            while True:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.youtube.playlists().list(
                        part="snippet,contentDetails",
                        channelId=channel_id,
                        maxResults=min(50, max_results - len(playlists)),
                        pageToken=next_page_token
                    ).execute()
                )
                
                for item in response.get("items", []):
                    playlist_data = PlaylistData(
                        id=item["id"],
                        title=item["snippet"]["title"],
                        description=item["snippet"]["description"],
                        video_count=item["contentDetails"]["itemCount"]
                    )
                    playlists.append(playlist_data)
                    
                    if len(playlists) >= max_results:
                        self.logger.info(f"Retrieved {len(playlists)} playlists from channel: {channel_id}")
                        return playlists
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            
            self.logger.info(f"Retrieved {len(playlists)} playlists from channel: {channel_id}")
            return playlists
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error getting playlists: {e}")
            return []
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def get_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        try:
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.youtube.search().list(
                    part="snippet",
                    q=channel_name,
                    type="channel",
                    maxResults=1
                ).execute()
            )
            
            if response.get("items"):
                channel_id = response["items"][0]["id"]["channelId"]
                self.logger.info(f"Found channel ID {channel_id} for channel name: {channel_name}")
                return channel_id
                
            self.logger.warning(f"No channel found for name: {channel_name}")
            return None
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error getting channel ID: {e}")
            return None
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def get_videos_from_playlist(self, playlist_id: str, max_results: int = 50) -> List[VideoData]:
        videos = []
        next_page_token = None
        
        try:
            loop = asyncio.get_event_loop()
            
            while True:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.youtube.playlistItems().list(
                        part="snippet,contentDetails",
                        playlistId=playlist_id,
                        maxResults=min(50, max_results - len(videos)),
                        pageToken=next_page_token
                    ).execute()
                )
                
                for item in response.get("items", []):
                    video_data = VideoData(
                        id=item["contentDetails"]["videoId"],
                        title=item["snippet"]["title"],
                        description=item["snippet"]["description"],
                        published_at=item["snippet"]["publishedAt"]
                    )
                    videos.append(video_data)
                    
                    if len(videos) >= max_results:
                        self.logger.info(f"Retrieved {len(videos)} videos from playlist: {playlist_id}")
                        return videos
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            
            self.logger.info(f"Retrieved {len(videos)} videos from playlist: {playlist_id}")
            return videos
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error getting videos from playlist: {e}")
            return []
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def get_video_details(self, video_id: str) -> Optional[VideoDetails]:
        try:
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=video_id
                ).execute()
            )
            
            if response.get("items"):
                video = response["items"][0]
                details = VideoDetails(
                    title=video["snippet"]["title"],
                    description=video["snippet"]["description"],
                    published_at=video["snippet"]["publishedAt"],
                    channel_id=video["snippet"]["channelId"],
                    channel_title=video["snippet"]["channelTitle"],
                    duration=video["contentDetails"]["duration"],
                    view_count=int(video["statistics"].get("viewCount", 0)),
                    like_count=int(video["statistics"].get("likeCount", 0)),
                    comment_count=int(video["statistics"].get("commentCount", 0))
                )
                
                self.logger.info(f"Retrieved details for video: {video_id}")
                return details
                
            self.logger.warning(f"No details found for video: {video_id}")
            return None
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error getting video details: {e}")
            return None
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    async def search_videos(self, query: str, max_results: int = 50, order: str = "relevance") -> List[VideoData]:
        videos = []
        next_page_token = None
        
        try:
            loop = asyncio.get_event_loop()
            
            while True:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.youtube.search().list(
                        part="snippet",
                        q=query,
                        type="video",
                        order=order,
                        maxResults=min(50, max_results - len(videos)),
                        pageToken=next_page_token
                    ).execute()
                )
                
                for item in response.get("items", []):
                    video_data = VideoData(
                        id=item["id"]["videoId"],
                        title=item["snippet"]["title"],
                        description=item["snippet"]["description"],
                        channel_title=item["snippet"]["channelTitle"],
                        published_at=item["snippet"]["publishedAt"]
                    )
                    videos.append(video_data)
                    
                    if len(videos) >= max_results:
                        self.logger.info(f"Found {len(videos)} videos for query: {query}")
                        return videos
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            
            self.logger.info(f"Found {len(videos)} videos for query: {query}")
            return videos
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error searching videos: {e}")
            return []

    async def _handle_error(self, error: Exception) -> ToolResult:
        error_message = str(error)
        self.logger.error(f"YouTube tool error: {error_message}")
        return ToolResult(success=False, error=error_message)
    
    async def _execute(self, action: str, **kwargs) -> ToolResult:
        return await self.run(action, **kwargs)

    @retry(
        stop=stop_after_attempt(RETRY_LIMIT), 
        wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT),
        reraise=True
    )
    async def run(self, action: str, **kwargs) -> ToolResult[Dict[str, Any]]:
        try:
            self.logger.info(f"Running action: {action} with parameters: {kwargs}")
            
            if action == "search_videos":
                query = kwargs.get("query")
                if not query:
                    raise ValueError("Query parameter is required for search_videos action")
                    
                max_results = kwargs.get("max_results", 50)
                order = kwargs.get("order", "relevance")
                
                self.logger.info(f"Searching for videos with query: {query}, max_results: {max_results}")
                videos = await self.search_videos(query, max_results, order)
                self.logger.info(f"Found {len(videos)} videos for query: {query}")
                return ToolResult(success=True, data={
                    "videos": [video.model_dump() for video in videos]
                })
                
            elif action == "get_transcript":
                video_id = kwargs.get("video_id")
                if not video_id:
                    raise ValueError("Video ID parameter is required for get_transcript action")
                
                self.logger.info(f"Getting transcript for video: {video_id}")
                try:
                    transcript = await self.get_transcript(video_id)
                    # Add debug logging for transcript format
                    self.logger.info(f"Transcript type: {type(transcript)}, length: {len(transcript) if transcript else 0}")
                    if transcript:
                        self.logger.debug(f"Transcript preview: {transcript[:200]}...")
                    else:
                        self.logger.warning(f"No transcript available for video: {video_id}")
                    
                    # Return plain text transcript directly instead of wrapping in a data object
                    return ToolResult(success=True, data=transcript)
                except Exception as e:
                    import traceback
                    self.logger.error(f"Error getting transcript for {video_id}: {str(e)}\nTraceback: {traceback.format_exc()}")
                    return ToolResult(success=False, error=f"Error getting transcript: {str(e)}")
                
            elif action == "get_video_details":
                video_id = kwargs.get("video_id")
                if not video_id:
                    raise ValueError("Video ID parameter is required for get_video_details action")
                    
                self.logger.info(f"Getting details for video: {video_id}")
                details = await self.get_video_details(video_id)
                if details:
                    self.logger.info(f"Successfully got details for video: {video_id}")
                else:
                    self.logger.warning(f"Failed to get details for video: {video_id}")
                    
                return ToolResult(success=True, data={
                    "details": details.model_dump() if details else None
                })
                
            elif action == "get_channel_id":
                channel_name = kwargs.get("channel_name")
                if not channel_name:
                    raise ValueError("Channel name parameter is required for get_channel_id action")
                    
                self.logger.info(f"Getting channel ID for: {channel_name}")
                channel_id = await self.get_channel_id_by_name(channel_name)
                if channel_id:
                    self.logger.info(f"Found channel ID for {channel_name}: {channel_id}")
                else:
                    self.logger.warning(f"No channel ID found for: {channel_name}")
                    
                return ToolResult(success=True, data={
                    "channel_id": channel_id
                })
                
            elif action == "get_playlists":
                channel_id = kwargs.get("channel_id")
                if not channel_id:
                    raise ValueError("Channel ID parameter is required for get_playlists action")
                    
                max_results = kwargs.get("max_results", 50)
                
                self.logger.info(f"Getting playlists for channel: {channel_id}, max_results: {max_results}")
                playlists = await self.get_playlists_from_channel(channel_id, max_results)
                self.logger.info(f"Found {len(playlists)} playlists for channel: {channel_id}")
                
                return ToolResult(success=True, data={
                    "playlists": [playlist.model_dump() for playlist in playlists]
                })
                
            elif action == "get_playlist_videos":
                playlist_id = kwargs.get("playlist_id")
                if not playlist_id:
                    raise ValueError("Playlist ID parameter is required for get_playlist_videos action")
                    
                max_results = kwargs.get("max_results", 50)
                
                self.logger.info(f"Getting videos from playlist: {playlist_id}, max_results: {max_results}")
                videos = await self.get_videos_from_playlist(playlist_id, max_results)
                self.logger.info(f"Found {len(videos)} videos in playlist: {playlist_id}")
                
                return ToolResult(success=True, data={
                    "videos": [video.model_dump() for video in videos]
                })
                
            else:
                error_msg = f"Unknown action: {action}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            import traceback
            self.logger.error(f"Error in YouTube tool run method: {str(e)}\nTraceback: {traceback.format_exc()}")
            return await self._handle_error(e)