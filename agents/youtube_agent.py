from typing import Dict, List, Any, Optional, Tuple, Union
import json
import asyncio
import traceback
import logging
from datetime import datetime
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
    before_sleep_log
)
import aiohttp

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.youtube_tool import YoutubeTool


class VideoData:
    def __init__(self, video_id: str, title: str, channel: str, description: str, transcript: str = None):
        self.video_id = video_id
        self.title = title
        self.channel = channel
        self.description = description
        self.transcript = transcript
        self.forensic_summary = None
        self.relevance_score = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization and logging."""
        return {
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "description": self.description[:100] + "..." if len(self.description) > 100 else self.description,
            "has_transcript": self.transcript is not None,
            "transcript_length": len(self.transcript) if self.transcript else 0,
            "forensic_summary": self.forensic_summary,
            "relevance_score": self.relevance_score
        }


class YouTubeError(Exception):
    """Base exception for YouTube-related errors."""
    pass


class YouTubeConnectionError(YouTubeError):
    """Exception raised when there's a connection issue with YouTube API."""
    pass


class YouTubeRateLimitError(YouTubeError):
    """Exception raised when YouTube API rate limit is exceeded."""
    pass


class YouTubeDataError(YouTubeError):
    """Exception raised when there's an issue processing YouTube data."""
    pass


class YouTubeValidationError(YouTubeError):
    """Exception raised when YouTube data validation fails."""
    pass


class YouTubeAgent(BaseAgent):
    name = "youtube_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager()
        self.youtube_tool = YoutubeTool(config.get("youtube", {}))
        
        # Set default retry parameters
        self.retry_attempts = config.get("retry", {}).get("max_attempts", 3)
        self.retry_multiplier = config.get("retry", {}).get("multiplier", 1)
        self.retry_min_wait = config.get("retry", {}).get("min_wait", 2)
        self.retry_max_wait = config.get("retry", {}).get("max_wait", 10)
    
    def _get_retry_decorator(self, operation_name: str):
        """Create a standardized retry decorator with logging."""
        return retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=self.retry_multiplier, min=self.retry_min_wait, max=self.retry_max_wait),
            retry=(
                retry_if_exception_type(YouTubeConnectionError) | 
                retry_if_exception_type(YouTubeRateLimitError) |
                retry_if_exception_type(aiohttp.ClientError)
            ),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
            reraise=True
        )
    
    def _parse_json_response(self, response: str, context: str = "response") -> Dict[str, Any]:
        """Parse JSON response with better error handling."""
        if not response:
            error_msg = f"Empty {context} received"
            self.logger.error(error_msg)
            raise YouTubeDataError(error_msg)
            
        try:
            # Clean response if it contains markdown code blocks
            if "```json" in response:
                response = response.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in response:
                response = response.split("```", 1)[1].split("```", 1)[0]
                    
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse {context} JSON: {str(e)}"
            self.logger.error(error_msg)
            raise YouTubeDataError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing {context}: {str(e)}"
            self.logger.error(error_msg)
            raise YouTubeDataError(error_msg)
    
    def _validate_result(self, result: Any, required_fields: List[str], context: str = "result") -> None:
        """Validate that the result contains all required fields."""
        if not result:
            raise YouTubeValidationError(f"Empty {context} received")
            
        if not isinstance(result, dict):
            raise YouTubeValidationError(f"{context} must be a dictionary")
            
        for field in required_fields:
            if field not in result:
                raise YouTubeValidationError(f"Missing required field '{field}' in {context}")
    
    def _sanitize_transcript(self, transcript: Optional[str]) -> Optional[str]:
        """Clean and sanitize a transcript for processing."""
        if not transcript:
            return None
            
        # Remove any excessive whitespace
        cleaned = " ".join(transcript.split())
        
        # Truncate if too long (8000 tokens is approximately 32000 chars)
        max_length = 32000
        if len(cleaned) > max_length:
            self.logger.warning(f"Transcript too long ({len(cleaned)} chars), truncating to {max_length} chars")
            cleaned = cleaned[:max_length] + "..."
            
        return cleaned
    
    async def search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for videos with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("search_videos")
        
        @retry_decorator
        async def _search_with_retry():
            try:
                self.logger.info(f"Searching YouTube for: {query}")
                
                search_result = await self.youtube_tool.run(
                    action="search_videos",
                    query=query,
                    max_results=max_results
                )
                
                if not search_result.success:
                    error_msg = f"YouTube search failed: {search_result.error}"
                    self.logger.error(error_msg)
                    
                    # Map error types
                    if "rate limit" in str(search_result.error).lower():
                        raise YouTubeRateLimitError(error_msg)
                    elif "connection" in str(search_result.error).lower() or "network" in str(search_result.error).lower():
                        raise YouTubeConnectionError(error_msg)
                    else:
                        raise YouTubeDataError(error_msg)
                
                videos = search_result.data.get("videos", [])
                
                # Validate the response
                if not isinstance(videos, list):
                    raise YouTubeValidationError(f"Invalid search result format for query: {query}")
                
                self.logger.info(f"Found {len(videos)} videos for query: {query}")
                return videos
                
            except aiohttp.ClientError as e:
                raise YouTubeConnectionError(f"Connection error searching videos: {str(e)}")
            except YouTubeError:
                # Re-raise YouTube-specific errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error searching videos: {str(e)}")
                raise YouTubeDataError(f"Error searching videos: {str(e)}")
        
        try:
            return await _search_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise YouTubeConnectionError(f"Failed to search videos after {self.retry_attempts} attempts")
    
    async def get_transcript(self, video_id: str) -> Optional[str]:
        """Get video transcript with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("get_transcript")
        
        @retry_decorator
        async def _get_transcript_with_retry():
            try:
                self.logger.info(f"Getting transcript for video: {video_id}")
                
                transcript_result = await self.youtube_tool.run(
                    action="get_transcript",
                    video_id=video_id
                )
                
                if not transcript_result.success:
                    error_msg = f"Failed to get transcript for video {video_id}: {transcript_result.error}"
                    
                    # Not all videos have transcripts, so this is not necessarily an error
                    if "No transcript" in str(transcript_result.error):
                        self.logger.info(f"No transcript available for video {video_id}")
                        return None
                    
                    self.logger.warning(error_msg)
                    
                    # Map error types for retryable failures
                    if "rate limit" in str(transcript_result.error).lower():
                        raise YouTubeRateLimitError(error_msg)
                    elif "connection" in str(transcript_result.error).lower() or "network" in str(transcript_result.error).lower():
                        raise YouTubeConnectionError(error_msg)
                    
                    # For other errors, just return None
                    return None
                
                transcript = transcript_result.data.get("transcript")
                
                # Defensive check for transcript type
                if transcript is None:
                    self.logger.warning(f"No transcript data received for video {video_id}")
                    return None
                    
                # Sanitize the transcript
                sanitized_transcript = self._sanitize_transcript(transcript)
                
                if sanitized_transcript:
                    self.logger.info(f"Successfully got transcript for video {video_id} ({len(sanitized_transcript)} chars)")
                else:
                    self.logger.warning(f"Empty transcript received for video {video_id}")
                
                return sanitized_transcript
                
            except aiohttp.ClientError as e:
                raise YouTubeConnectionError(f"Connection error getting transcript: {str(e)}")
            except YouTubeError:
                # Re-raise YouTube-specific errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error getting transcript: {str(e)}")
                # For get_transcript, we'll return None instead of raising an exception
                # because we want to continue processing videos even if some transcripts fail
                return None
        
        try:
            return await _get_transcript_with_retry()
        except RetryError as e:
            self.logger.warning(f"Failed to get transcript after {self.retry_attempts} attempts: {str(e.last_attempt.exception())}")
            return None
    
    async def analyze_transcript(self, video_data: VideoData, company: str) -> Dict[str, Any]:
        """Analyze transcript with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("analyze_transcript")
        
        @retry_decorator
        async def _analyze_with_retry():
            try:
                self.logger.info(f"Analyzing transcript for video: {video_data.video_id}")
                
                if not video_data.transcript:
                    self.logger.warning(f"No transcript available for video {video_data.video_id}")
                    return {
                        "forensic_relevance": "low",
                        "red_flags": [],
                        "summary": "No transcript available for analysis",
                        "video_id": video_data.video_id,
                        "title": video_data.title
                    }
                
                llm_provider = await get_llm_provider()
                
                # Truncate description if too long
                description = video_data.description
                if len(description) > 1000:
                    description = description[:997] + "..."
                
                # Truncate transcript if too long to avoid token limits
                transcript = video_data.transcript
                max_transcript_length = 8000
                if len(transcript) > max_transcript_length:
                    self.logger.warning(f"Transcript too long ({len(transcript)} chars), truncating to {max_transcript_length}")
                    transcript = transcript[:max_transcript_length] + "..."
                
                variables = {
                    "company": company,
                    "video_title": video_data.title,
                    "video_description": description,
                    "transcript": transcript
                }
                
                system_prompt, human_prompt = self.prompt_manager.get_prompt(
                    agent_name=self.name,
                    operation="analyze_transcript",
                    variables=variables
                )
                
                input_message = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                response = await llm_provider.generate_text(
                    input_message,
                    model_name=self.config.get("models", {}).get("analysis")
                )
                
                # Add defensive check for empty response
                if not response:
                    self.logger.warning(f"Empty analysis response received for video {video_data.video_id}")
                    return {
                        "forensic_relevance": "unknown",
                        "red_flags": [],
                        "summary": "Analysis failed due to empty response",
                        "video_id": video_data.video_id,
                        "title": video_data.title
                    }
                
                try:
                    analysis_result = self._parse_json_response(response, f"transcript analysis for {video_data.video_id}")
                except YouTubeDataError as e:
                    self.logger.warning(f"Error parsing analysis response: {str(e)}")
                    return {
                        "forensic_relevance": "unknown",
                        "red_flags": [],
                        "summary": f"Error analyzing transcript: {str(e)}",
                        "video_id": video_data.video_id,
                        "title": video_data.title
                    }
                
                # Validate required fields with default values if missing
                if not isinstance(analysis_result, dict):
                    analysis_result = {}
                    
                analysis_result["forensic_relevance"] = analysis_result.get("forensic_relevance", "unknown")
                analysis_result["red_flags"] = analysis_result.get("red_flags", [])
                analysis_result["summary"] = analysis_result.get("summary", "No summary available")
                
                # Add video metadata to the result
                analysis_result["video_id"] = video_data.video_id
                analysis_result["title"] = video_data.title
                
                video_data.forensic_summary = analysis_result
                
                # Map relevance to score
                relevance_mapping = {
                    "high": 0.9,
                    "medium": 0.6,
                    "low": 0.3,
                    "unknown": 0.1
                }
                
                relevance = analysis_result.get("forensic_relevance", "unknown").lower()
                video_data.relevance_score = relevance_mapping.get(relevance, 0.1)
                
                self.logger.info(f"Analyzed transcript for video {video_data.video_id} (relevance: {relevance})")
                
                return analysis_result
                
            except YouTubeValidationError as e:
                # For validation errors, return a default result rather than failing
                self.logger.warning(f"Validation error analyzing transcript: {str(e)}")
                return {
                    "forensic_relevance": "unknown",
                    "red_flags": [],
                    "summary": f"Error analyzing transcript: {str(e)}",
                    "video_id": video_data.video_id,
                    "title": video_data.title
                }
            except YouTubeError:
                # Re-raise YouTube-specific errors for retry
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error analyzing transcript: {str(e)}")
                raise YouTubeDataError(f"Error analyzing transcript: {str(e)}")
        
        try:
            return await _analyze_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                self.logger.error(f"Failed to analyze transcript after {self.retry_attempts} attempts: {str(e.last_attempt.exception())}")
            
            # Return a default result rather than failing the entire process
            return {
                "forensic_relevance": "unknown",
                "red_flags": ["Analysis failed after multiple attempts"],
                "summary": f"Failed to analyze transcript after {self.retry_attempts} attempts",
                "video_id": video_data.video_id,
                "title": video_data.title
            }
    
    async def get_channel_videos(self, channel_name: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Get channel videos with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("get_channel_videos")
        
        @retry_decorator
        async def _get_channel_videos_with_retry():
            try:
                self.logger.info(f"Getting videos from channel: {channel_name}")
                
                # First, get the channel ID
                channel_id_result = await self.youtube_tool.run(
                    action="get_channel_id",
                    channel_name=channel_name
                )
                
                if not channel_id_result.success or not channel_id_result.data.get("channel_id"):
                    error_msg = f"Failed to get channel ID for {channel_name}: {channel_id_result.error}"
                    self.logger.error(error_msg)
                    
                    # Map error types
                    if "rate limit" in str(channel_id_result.error).lower():
                        raise YouTubeRateLimitError(error_msg)
                    elif "connection" in str(channel_id_result.error).lower() or "network" in str(channel_id_result.error).lower():
                        raise YouTubeConnectionError(error_msg)
                    else:
                        raise YouTubeDataError(error_msg)
                
                channel_id = channel_id_result.data.get("channel_id")
                
                # Get the channel's playlists
                playlists_result = await self.youtube_tool.run(
                    action="get_playlists",
                    channel_id=channel_id,
                    max_results=5
                )
                
                all_videos = []
                
                # Process playlists if available
                if playlists_result.success and playlists_result.data.get("playlists"):
                    playlists = playlists_result.data.get("playlists")
                    
                    # Validate playlists data
                    if not isinstance(playlists, list):
                        self.logger.warning(f"Invalid playlists format for channel {channel_name}")
                    else:
                        for playlist in playlists:
                            if not playlist.get("id"):
                                continue
                                
                            playlist_id = playlist.get("id")
                            
                            # Get videos from playlist
                            playlist_videos_result = await self.youtube_tool.run(
                                action="get_playlist_videos",
                                playlist_id=playlist_id,
                                max_results=10
                            )
                            
                            if playlist_videos_result.success and playlist_videos_result.data.get("videos"):
                                videos = playlist_videos_result.data.get("videos")
                                
                                # Validate videos data
                                if isinstance(videos, list):
                                    all_videos.extend(videos)
                                    self.logger.info(f"Added {len(videos)} videos from playlist {playlist_id}")
                                else:
                                    self.logger.warning(f"Invalid videos format for playlist {playlist_id}")
                            
                            # Stop if we have enough videos
                            if len(all_videos) >= max_results:
                                break
                
                # If we don't have enough videos from playlists, search for more
                if len(all_videos) < max_results:
                    search_query = f"channel:{channel_name}"
                    search_result = await self.youtube_tool.run(
                        action="search_videos",
                        query=search_query,
                        max_results=max_results - len(all_videos)
                    )
                    
                    if search_result.success and search_result.data.get("videos"):
                        videos = search_result.data.get("videos")
                        
                        # Validate videos data
                        if isinstance(videos, list):
                            all_videos.extend(videos)
                            self.logger.info(f"Added {len(videos)} videos from search for channel {channel_name}")
                        else:
                            self.logger.warning(f"Invalid videos format from channel search for {channel_name}")
                
                # Deduplicate videos by ID
                unique_videos = {}
                for video in all_videos:
                    video_id = video.get("id")
                    if video_id and video_id not in unique_videos:
                        unique_videos[video_id] = video
                
                self.logger.info(f"Got {len(unique_videos)} unique videos from channel {channel_name}")
                return list(unique_videos.values())
                
            except aiohttp.ClientError as e:
                raise YouTubeConnectionError(f"Connection error getting channel videos: {str(e)}")
            except YouTubeError:
                # Re-raise YouTube-specific errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error getting channel videos: {str(e)}")
                raise YouTubeDataError(f"Error getting channel videos: {str(e)}")
        
        try:
            return await _get_channel_videos_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise YouTubeConnectionError(f"Failed to get channel videos after {self.retry_attempts} attempts")
    
    async def generate_video_summary(self, videos_data: List[VideoData], company: str) -> Dict[str, Any]:
        """Generate video summary with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("generate_video_summary")
        
        @retry_decorator
        async def _generate_with_retry():
            try:
                self.logger.info(f"Generating video summary for {len(videos_data)} videos about {company}")
                
                # Sort videos by relevance score (highest first)
                sorted_videos = sorted(videos_data, key=lambda x: x.relevance_score, reverse=True)
                
                # Take the top 10 videos for the summary
                top_videos = sorted_videos[:10]
                
                video_summaries = []
                for video in top_videos:
                    summary = {
                        "video_id": video.video_id,
                        "title": video.title,
                        "channel": video.channel,
                        "forensic_relevance": video.forensic_summary.get("forensic_relevance", "unknown") if video.forensic_summary else "unknown",
                        "red_flags": video.forensic_summary.get("red_flags", []) if video.forensic_summary else [],
                        "summary": video.forensic_summary.get("summary", "") if video.forensic_summary else "No summary available"
                    }
                    video_summaries.append(summary)
                
                llm_provider = await get_llm_provider()
                
                variables = {
                    "company": company,
                    "video_count": len(videos_data),
                    "analyzed_count": len([v for v in videos_data if v.forensic_summary is not None]),
                    "video_summaries": json.dumps(video_summaries)
                }
                
                system_prompt, human_prompt = self.prompt_manager.get_prompt(
                    agent_name=self.name,
                    operation="generate_summary",
                    variables=variables
                )
                
                input_message = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                response = await llm_provider.generate_text(
                    input_message,
                    model_name=self.config.get("models", {}).get("summary")
                )
                
                summary_result = self._parse_json_response(response, "video summary")
                
                # Validate required fields
                self._validate_result(
                    summary_result,
                    ["overall_assessment", "key_insights", "red_flags", "notable_videos"],
                    "video summary"
                )
                
                # Add timestamp and metadata
                summary_result["timestamp"] = datetime.now().isoformat()
                summary_result["company"] = company
                summary_result["total_videos_analyzed"] = len(videos_data)
                
                self.logger.info(f"Generated video summary for {company} with {len(summary_result.get('red_flags', []))} red flags")
                
                return summary_result
                
            except YouTubeValidationError as e:
                # For validation errors, return a default result
                self.logger.warning(f"Validation error generating video summary: {str(e)}")
                return {
                    "overall_assessment": "Unknown",
                    "key_insights": [f"Error generating summary: {str(e)}"],
                    "red_flags": ["Summary generation failed"],
                    "notable_videos": [],
                    "summary": f"Analysis of YouTube content for {company} encountered validation errors.",
                    "company": company,
                    "timestamp": datetime.now().isoformat(),
                    "total_videos_analyzed": len(videos_data)
                }
            except YouTubeError:
                # Re-raise YouTube-specific errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error generating video summary: {str(e)}")
                raise YouTubeDataError(f"Error generating video summary: {str(e)}")
        
        try:
            return await _generate_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                self.logger.error(f"Failed to generate video summary after {self.retry_attempts} attempts: {str(e.last_attempt.exception())}")
            
            # Return a default result
            return {
                "overall_assessment": "Error",
                "key_insights": ["Failed to generate summary after multiple attempts"],
                "red_flags": ["Summary generation failed"],
                "notable_videos": [],
                "summary": f"Analysis of YouTube content for {company} encountered errors.",
                "company": company,
                "timestamp": datetime.now().isoformat(),
                "total_videos_analyzed": len(videos_data)
            }
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self.run(state)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the YouTube agent with enhanced error handling and logging."""
        try:
            self._log_start(state)
            
            company = state.get("company", "")
            industry = state.get("industry", "")
            research_plan = state.get("research_plan", [])
            
            if not company:
                self.logger.error("Company name is missing!")
                return {**state, "goto": "meta_agent", "youtube_status": "ERROR", "error": "Company name is missing"}
            
            if not research_plan:
                self.logger.warning("No research plan provided, using default queries")
                research_plan = [{
                    "query_categories": {
                        "general": f"{company}",
                        "industry": f"{company} {industry}",
                        "controversy": f"{company} controversy scandal",
                        "financial": f"{company} financial reporting earnings",
                        "legal": f"{company} lawsuit legal issues"
                    }
                }]
            
            youtube_queries = []
            current_plan = research_plan[-1]
            
            for category, queries in current_plan.get("query_categories", {}).items():
                if isinstance(queries, list):
                    youtube_queries.extend([f"{company} {q}" for q in queries])
                else:
                    youtube_queries.append(f"{company} {queries}")
            
            if not youtube_queries:
                youtube_queries.append(company)
                if industry:
                    youtube_queries.append(f"{company} {industry}")
                youtube_queries.append(f"{company} fraud")
                youtube_queries.append(f"{company} scandal")
                youtube_queries.append(f"{company} controversy")
            
            self.logger.info(f"YouTube research plan for {company} with {len(youtube_queries)} queries")
            
            youtube_results = {
                "videos": [],
                "channels": {},
                "summary": {},
                "red_flags": [],
                "queries": youtube_queries
            }
            
            # Keep track of errors for reporting
            search_errors = []
            transcript_errors = []
            analysis_errors = []
            
            all_videos = []
            
            # Search for videos based on queries
            for query in youtube_queries:
                try:
                    videos = await self.search_videos(query, max_results=5)
                    all_videos.extend(videos)
                    self.logger.info(f"Found {len(videos)} videos for query: {query}")
                except Exception as e:
                    error_msg = f"Error searching for '{query}': {str(e)}"
                    self.logger.error(error_msg)
                    search_errors.append(error_msg)
                    # Continue with other queries
            
            # Deduplicate videos
            unique_videos = {}
            for video in all_videos:
                video_id = video.get("id")
                if video_id and video_id not in unique_videos:
                    unique_videos[video_id] = video
            
            # Count videos per channel to find popular channels
            channels = {}
            for video in unique_videos.values():
                channel = video.get("channel_title")
                if channel and channel not in channels:
                    channels[channel] = 1
                elif channel:
                    channels[channel] += 1
            
            # Get videos from top channels
            top_channels = sorted(channels.items(), key=lambda x: x[1], reverse=True)[:3]
            for channel_name, _ in top_channels:
                try:
                    channel_videos = await self.get_channel_videos(channel_name, max_results=10)
                    all_videos.extend(channel_videos)
                    youtube_results["channels"][channel_name] = len(channel_videos)
                    self.logger.info(f"Found {len(channel_videos)} videos from channel: {channel_name}")
                except Exception as e:
                    error_msg = f"Error getting videos from channel '{channel_name}': {str(e)}"
                    self.logger.error(error_msg)
                    search_errors.append(error_msg)
                    # Continue with other channels
            
            # Final deduplication of videos
            unique_videos = {}
            for video in all_videos:
                video_id = video.get("id")
                if video_id and video_id not in unique_videos:
                    unique_videos[video_id] = video
            
            self.logger.info(f"Collected {len(unique_videos)} unique videos about {company}")
            
            # If no videos found, return early
            if not unique_videos:
                youtube_results["error"] = f"No videos found for {company}"
                state["youtube_results"] = youtube_results
                state["youtube_status"] = "DONE"
                
                goto = "meta_agent"
                if state.get("synchronous_pipeline", False):
                    goto = state.get("next_agent", "meta_agent")
                    
                self._log_completion({**state, "goto": goto})
                return {**state, "goto": goto}
            
            # Convert to VideoData objects
            videos_data = []
            for video in unique_videos.values():
                video_data = VideoData(
                    video_id=video.get("id"),
                    title=video.get("title", "Unknown title"),
                    channel=video.get("channel_title", "Unknown channel"),
                    description=video.get("description", "")
                )
                videos_data.append(video_data)
            
            # Sort videos by potential relevance (basic heuristic - look for keywords in title)
            def relevance_score(video):
                title_lower = video.title.lower()
                keywords = ["fraud", "scandal", "controversy", "investigation", "sec", "lawsuit"]
                score = sum(3 for kw in keywords if kw in title_lower)
                score += 2 if company.lower() in title_lower else 0
                return score
                
            videos_data.sort(key=relevance_score, reverse=True)
            
            # Analyze a subset of videos
            max_to_analyze = min(20, len(videos_data))
            videos_to_analyze = videos_data[:max_to_analyze]
            
            self.logger.info(f"Analyzing {len(videos_to_analyze)} videos out of {len(videos_data)} total")
            
            # Process in batches to avoid overwhelming the API
            batch_size = 5
            for i in range(0, len(videos_to_analyze), batch_size):
                batch = videos_to_analyze[i:i+batch_size]
                
                # Get transcripts for the batch
                transcript_tasks = []
                for video_data in batch:
                    task = asyncio.create_task(self.get_transcript(video_data.video_id))
                    transcript_tasks.append((video_data, task))
                    
                for video_data, task in transcript_tasks:
                    try:
                        transcript = await task
                        if transcript:
                            video_data.transcript = transcript
                            self.logger.info(f"Got transcript for video: {video_data.video_id}")
                        else:
                            self.logger.warning(f"No transcript available for video: {video_data.video_id}")
                    except Exception as e:
                        error_msg = f"Error getting transcript for video {video_data.video_id}: {str(e)}"
                        self.logger.error(error_msg)
                        transcript_errors.append(error_msg)
                        # Continue with other videos
                
                # Analyze transcripts - only for videos that have transcripts
                analysis_tasks = []
                for video_data in batch:
                    if video_data.transcript:
                        try:
                            # Use create_task to handle exceptions properly
                            task = asyncio.create_task(self.analyze_transcript(video_data, company))
                            analysis_tasks.append((video_data, task))
                        except Exception as e:
                            error_msg = f"Error creating analysis task for video {video_data.video_id}: {str(e)}"
                            self.logger.error(error_msg)
                            analysis_errors.append(error_msg)
                
                for video_data, task in analysis_tasks:
                    try:
                        analysis = await task
                        if analysis:
                            self.logger.info(f"Analyzed transcript for video: {video_data.video_id} - Relevance: {analysis.get('forensic_relevance', 'unknown')}")
                        else:
                            self.logger.warning(f"No analysis result for video: {video_data.video_id}")
                    except Exception as e:
                        error_msg = f"Error analyzing transcript for video {video_data.video_id}: {str(e)}"
                        self.logger.error(error_msg)
                        analysis_errors.append(error_msg)
                        # Continue with other videos
                
                # Pause between batches to avoid rate limits
                await asyncio.sleep(1)
            
            # Generate summary with defensive checks
            try:
                # Only include videos with valid analysis data in the summary
                videos_with_analysis = [v for v in videos_data if v.forensic_summary is not None]
                
                if not videos_with_analysis:
                    self.logger.warning(f"No videos with valid analysis available for summary generation")
                    summary = {
                        "overall_assessment": "Unknown",
                        "key_insights": [f"No video analysis available for {company}"],
                        "red_flags": [],
                        "notable_videos": [],
                        "summary": f"No analyzable YouTube content found for {company}."
                    }
                else:
                    summary = await self.generate_video_summary(videos_with_analysis, company)
                
                youtube_results["summary"] = summary
                
                # Collect red flags
                if "red_flags" in summary and isinstance(summary["red_flags"], list):
                    youtube_results["red_flags"] = summary["red_flags"]
                    
                self.logger.info(f"Generated YouTube summary for {company} with {len(youtube_results['red_flags'])} red flags")
                
            except Exception as e:
                error_msg = f"Error generating summary for {company}: {str(e)}"
                self.logger.error(error_msg)
                youtube_results["summary"] = {
                    "error": error_msg,
                    "overall_assessment": "Error",
                    "key_insights": ["Failed to generate summary"],
                    "red_flags": [],
                    "notable_videos": [],
                    "summary": f"Failed to generate summary for YouTube content about {company}."
                }
            
            # Add all videos to the result
            for video_data in videos_data:
                video_result = {
                    "video_id": video_data.video_id,
                    "title": video_data.title,
                    "channel": video_data.channel,
                    "description": video_data.description[:200] + "..." if len(video_data.description) > 200 else video_data.description,
                    "has_transcript": video_data.transcript is not None,
                    "transcript_length": len(video_data.transcript) if video_data.transcript else 0,
                    "forensic_relevance": video_data.forensic_summary.get("forensic_relevance", "unknown") if video_data.forensic_summary else "unknown",
                    "red_flags": video_data.forensic_summary.get("red_flags", []) if video_data.forensic_summary else [],
                    "summary": video_data.forensic_summary.get("summary", "") if video_data.forensic_summary else "No summary available"
                }
                youtube_results["videos"].append(video_result)
            
            # Add error information
            if search_errors or transcript_errors or analysis_errors:
                youtube_results["errors"] = {
                    "search_errors": search_errors,
                    "transcript_errors": transcript_errors,
                    "analysis_errors": analysis_errors,
                    "total_errors": len(search_errors) + len(transcript_errors) + len(analysis_errors)
                }
            
            state["youtube_results"] = youtube_results
            state["youtube_status"] = "DONE"
            
            goto = "meta_agent"
            if state.get("synchronous_pipeline", False):
                goto = state.get("next_agent", "meta_agent")
                
            self._log_completion({**state, "goto": goto})
            return {**state, "goto": goto}
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Unhandled error in youtube_agent run method: {str(e)}\n{tb}"
            self.logger.error(error_msg)
            
            # Try to return partial results if available
            if 'youtube_results' not in state:
                state['youtube_results'] = {
                    "videos": [],
                    "channels": {},
                    "summary": {
                        "error": error_msg,
                        "overall_assessment": "Error",
                        "key_insights": ["Agent encountered an error"],
                        "red_flags": ["YouTube agent failed with error"],
                        "notable_videos": []
                    },
                    "red_flags": ["YouTube agent failed with error"],
                    "error": error_msg
                }
                
            return {
                **state, 
                "goto": "meta_agent", 
                "error": error_msg,
                "youtube_status": "ERROR"
            }