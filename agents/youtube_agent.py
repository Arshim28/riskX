from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os
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
    pass


class YouTubeConnectionError(YouTubeError):
    pass


class YouTubeRateLimitError(YouTubeError):
    pass


class YouTubeDataError(YouTubeError):
    pass


class YouTubeValidationError(YouTubeError):
    pass


class YouTubeAgent(BaseAgent):
    name = "youtube_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager()
        self.youtube_tool = YoutubeTool(config.get("youtube", {}))
        
        self.retry_attempts = config.get("retry", {}).get("max_attempts", 3)
        self.retry_multiplier = config.get("retry", {}).get("multiplier", 1)
        self.retry_min_wait = config.get("retry", {}).get("min_wait", 2)
        self.retry_max_wait = config.get("retry", {}).get("max_wait", 10)
    
    def _get_retry_decorator(self, operation_name: str):
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
        if not response:
            error_msg = f"Empty {context} received"
            self.logger.error(error_msg)
            raise YouTubeDataError(error_msg)
            
        try:
            # More robust JSON extraction
            self.logger.info(f"Attempting to parse JSON from response of length {len(response)}")
            
            # First look for JSON blocks
            if "```json" in response:
                json_content = response.split("```json", 1)[1].split("```", 1)[0].strip()
                self.logger.info(f"Extracted JSON from ```json block, length: {len(json_content)}")
            elif "```" in response:
                json_content = response.split("```", 1)[1].split("```", 1)[0].strip()
                self.logger.info(f"Extracted JSON from generic ``` block, length: {len(json_content)}")
            else:
                # If no markdown blocks, try to find JSON structure directly
                # First, try to find opening and closing braces for a complete JSON object
                if response.strip().startswith("{") and response.strip().endswith("}"):
                    json_content = response.strip()
                    self.logger.info(f"Using raw response as JSON object, length: {len(json_content)}")
                else:
                    # Last resort - use the whole response and hope for the best
                    json_content = response.strip()
                    self.logger.info(f"Using entire response content as JSON, length: {len(json_content)}")
            
            # Try to parse, with fallback to a default structure if parsing fails
            try:
                parsed_json = json.loads(json_content)
                self.logger.info(f"Successfully parsed JSON with {len(parsed_json)} top-level keys")
                return parsed_json
            except json.JSONDecodeError as e:
                self.logger.error(f"JSONDecodeError: {str(e)}, falling back to default structure")
                
                # Create a default response structure as fallback
                default_response = {
                    "forensic_relevance": "unknown",
                    "red_flags": [],
                    "summary": f"Failed to parse response for {context}: {str(e)}. Original content: {response[:200]}..."
                }
                return default_response
                
        except Exception as e:
            error_msg = f"Unexpected error parsing {context}: {str(e)}"
            self.logger.error(error_msg)
            
            # Log the error and the raw text for debugging
            self.logger.debug(f"Raw response that failed to parse: {response[:300]}...")
            
            # Return default structure instead of raising error
            return {
                "forensic_relevance": "unknown",
                "red_flags": ["Parsing error"],
                "summary": f"Error processing content: {str(e)}"
            }
    
    def _validate_result(self, result: Any, required_fields: List[str], context: str = "result") -> None:
        if not result:
            raise YouTubeValidationError(f"Empty {context} received")
            
        if not isinstance(result, dict):
            raise YouTubeValidationError(f"{context} must be a dictionary")
            
        for field in required_fields:
            if field not in result:
                raise YouTubeValidationError(f"Missing required field '{field}' in {context}")
    
    def _sanitize_transcript(self, transcript: Optional[str]) -> Optional[str]:
        if not transcript:
            return None
            
        cleaned = " ".join(transcript.split())
        
        max_length = 32000
        if len(cleaned) > max_length:
            self.logger.warning(f"Transcript too long ({len(cleaned)} chars), truncating to {max_length} chars")
            cleaned = cleaned[:max_length] + "..."
            
        return cleaned
    
    async def search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
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
                    
                    if "rate limit" in str(search_result.error).lower():
                        raise YouTubeRateLimitError(error_msg)
                    elif "connection" in str(search_result.error).lower() or "network" in str(search_result.error).lower():
                        raise YouTubeConnectionError(error_msg)
                    else:
                        raise YouTubeDataError(error_msg)
                
                videos = search_result.data.get("videos", [])
                
                if not isinstance(videos, list):
                    raise YouTubeValidationError(f"Invalid search result format for query: {query}")
                
                self.logger.info(f"Found {len(videos)} videos for query: {query}")
                return videos
                
            except aiohttp.ClientError as e:
                raise YouTubeConnectionError(f"Connection error searching videos: {str(e)}")
            except YouTubeError:
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
        retry_decorator = self._get_retry_decorator("get_transcript")
        
        @retry_decorator
        async def _get_transcript_with_retry():
            try:
                self.logger.info(f"Getting transcript for video: {video_id}")
                
                # Add traceback context for better debugging
                self.logger.debug(f"Transcript call stack for {video_id}:\n{traceback.format_stack()}")
                
                transcript_result = await self.youtube_tool.run(
                    action="get_transcript",
                    video_id=video_id
                )
                
                if not transcript_result.success:
                    error_msg = f"Failed to get transcript for video {video_id}: {transcript_result.error}"
                    
                    if "No transcript" in str(transcript_result.error):
                        self.logger.info(f"No transcript available for video {video_id}")
                        return None
                    
                    self.logger.warning(error_msg)
                    
                    if "rate limit" in str(transcript_result.error).lower():
                        raise YouTubeRateLimitError(error_msg)
                    elif "connection" in str(transcript_result.error).lower() or "network" in str(transcript_result.error).lower():
                        raise YouTubeConnectionError(error_msg)
                    elif "quota" in str(transcript_result.error).lower():
                        raise YouTubeRateLimitError(f"YouTube API quota exceeded: {transcript_result.error}")
                    
                    return None
                
                # Handle plain text transcript from the API (no JSON wrapper)
                transcript = transcript_result.data
                
                if transcript is None:
                    self.logger.warning(f"No transcript data received for video {video_id}")
                    return None
                    
                sanitized_transcript = self._sanitize_transcript(transcript)
                
                if sanitized_transcript:
                    self.logger.info(f"Successfully got transcript for video {video_id} ({len(sanitized_transcript)} chars)")
                else:
                    self.logger.warning(f"Empty transcript received for video {video_id}")
                
                return sanitized_transcript
                
            except aiohttp.ClientError as e:
                self.logger.error(f"Connection error getting transcript: {str(e)}\nTraceback: {traceback.format_exc()}")
                raise YouTubeConnectionError(f"Connection error getting transcript: {str(e)}")
            except YouTubeError:
                self.logger.error(f"YouTube error getting transcript: {traceback.format_exc()}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error getting transcript: {str(e)}\nTraceback: {traceback.format_exc()}")
                return None
        
        try:
            self.logger.info(f"Starting transcript retrieval for video {video_id} with retries")
            result = await _get_transcript_with_retry()
            if result:
                self.logger.info(f"Successfully retrieved transcript for {video_id} after retries")
            else:
                self.logger.warning(f"Failed to get transcript for {video_id} after retries")
            return result
        except RetryError as e:
            self.logger.error(f"Failed to get transcript after {self.retry_attempts} attempts for video {video_id}: {str(e.last_attempt.exception())}\nTraceback: {traceback.format_exc()}")
            return None
    
    async def analyze_transcript(self, video_data: VideoData, company: str) -> Dict[str, Any]:
        retry_decorator = self._get_retry_decorator("analyze_transcript")
        
        @retry_decorator
        async def _analyze_with_retry():
            try:
                self.logger.info(f"Analyzing transcript for video: {video_data.video_id}")
                
                # Add debugging context
                self.logger.debug(f"Analysis call stack for {video_data.video_id}:\n{traceback.format_stack()}")
                
                if not video_data.transcript:
                    self.logger.warning(f"No transcript available for video {video_data.video_id}")
                    return {
                        "forensic_relevance": "low",
                        "red_flags": [],
                        "summary": "No transcript available for analysis",
                        "video_id": video_data.video_id,
                        "title": video_data.title
                    }
                
                try:
                    llm_provider = await get_llm_provider()
                    self.logger.info(f"LLM provider obtained for analysis of video {video_data.video_id}")
                except Exception as e:
                    self.logger.error(f"Failed to get LLM provider for video {video_data.video_id}: {str(e)}\nTraceback: {traceback.format_exc()}")
                    raise YouTubeDataError(f"LLM provider error: {str(e)}")
                
                description = video_data.description
                if len(description) > 1000:
                    description = description[:997] + "..."
                
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
                
                try:
                    self.logger.info(f"Getting prompts for video {video_data.video_id}")
                    prompt_result = self.prompt_manager.get_prompt(
                        agent_name=self.name,
                        operation="analyze_transcript",
                        variables=variables
                    )
                    
                    if not prompt_result or not isinstance(prompt_result, tuple) or len(prompt_result) != 2:
                        self.logger.error(f"Invalid prompt format for video {video_data.video_id}. Using default prompts.")
                        system_prompt = f"""Analyze this transcript for {company}. Identify any forensically relevant information.
Focus on financial irregularities, disclosures, governance issues, and potential red flags.
You will receive an unstructured video transcript to analyze."""
                        human_prompt = f"""Please analyze this transcript for the video titled '{video_data.title}'. 
Extract any important information related to financial forensics, corporate governance, or potential irregularities.

The transcript is:
{transcript}

Provide a thorough analysis focusing on:
1. Whether this content is forensically relevant (high/medium/low)
2. Any potential red flags or concerning statements
3. A summary of key points relevant to a financial investigation"""
                    else:
                        system_prompt, human_prompt = prompt_result
                        self.logger.info(f"Successfully retrieved prompts for video {video_data.video_id}")
                except Exception as e:
                    self.logger.error(f"Error getting prompts for video {video_data.video_id}: {str(e)}\nTraceback: {traceback.format_exc()}. Using default prompts.")
                    system_prompt = f"""Analyze this transcript for {company}. Identify any forensically relevant information.
Focus on financial irregularities, disclosures, governance issues, and potential red flags.
You will receive an unstructured video transcript to analyze."""
                    human_prompt = f"""Please analyze this transcript for the video titled '{video_data.title}'. 
Extract any important information related to financial forensics, corporate governance, or potential irregularities.

The transcript is:
{transcript}

Provide a thorough analysis focusing on:
1. Whether this content is forensically relevant (high/medium/low)
2. Any potential red flags or concerning statements
3. A summary of key points relevant to a financial investigation"""
                
                input_message = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                self.logger.info(f"Calling LLM for analysis of video {video_data.video_id}")
                try:
                    # Fixed: Changed model_name to model
                    model = self.config.get("models", {}).get("analysis")
                    self.logger.info(f"Using model {model} for analysis of video {video_data.video_id}")
                    
                    response = await llm_provider.generate_text(
                        input_message,
                        model=model
                    )
                    
                    # Log the raw response for debugging
                    self.logger.info(f"Received LLM response for video {video_data.video_id}, length: {len(response)}")
                    self.logger.debug(f"Raw analysis response preview for {video_data.video_id}: {response[:300]}...")
                except Exception as e:
                    self.logger.error(f"Error generating text for video {video_data.video_id}: {str(e)}\nTraceback: {traceback.format_exc()}")
                    raise YouTubeDataError(f"LLM generation error: {str(e)}")
                
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
                    # First try to parse as JSON for backward compatibility
                    self.logger.info(f"Parsing analysis response for video {video_data.video_id}")
                    try:
                        analysis_result = self._parse_json_response(response, f"transcript analysis for {video_data.video_id}")
                        self.logger.info(f"Successfully parsed JSON response for video {video_data.video_id}")
                    except (YouTubeDataError, json.JSONDecodeError):
                        # If JSON parsing fails, treat response as unstructured text and convert to our format
                        self.logger.info(f"Response is not JSON, processing as unstructured text for {video_data.video_id}")
                        self.logger.debug(f"Raw transcript response: {response[:200]}...")
                        # Create a JSON-like structure from the unstructured text
                        summary = response.strip()
                        
                        # Try to determine relevance from the text
                        relevance = "medium"  # Default value
                        if "high" in summary.lower() and "relevance" in summary.lower():
                            relevance = "high"
                        elif "low" in summary.lower() and "relevance" in summary.lower():
                            relevance = "low"
                        
                        # Try to extract red flags if mentioned in the text
                        red_flags = []
                        if "red flag" in summary.lower() or "concern" in summary.lower():
                            # Simple extraction attempt - not perfect but better than empty
                            lines = summary.split('\n')
                            for line in lines:
                                if "red flag" in line.lower() or "concern" in line.lower() or "issue" in line.lower():
                                    red_flags.append(line.strip())
                        
                        analysis_result = {
                            "forensic_relevance": relevance,
                            "red_flags": red_flags[:5],  # Limit to 5 items
                            "summary": summary
                        }
                        
                        self.logger.info(f"Created structured data from unstructured response: relevance={relevance}, red_flags={len(red_flags)}")
                except Exception as e:
                    self.logger.error(f"Error parsing analysis response for video {video_data.video_id}: {str(e)}\nTraceback: {traceback.format_exc()}")
                    # Use the raw response as the summary rather than discarding it
                    return {
                        "forensic_relevance": "unknown",
                        "red_flags": [],
                        "summary": response if response else f"Error analyzing transcript: {str(e)}",
                        "video_id": video_data.video_id,
                        "title": video_data.title
                    }
                
                if not isinstance(analysis_result, dict):
                    self.logger.warning(f"Analysis result is not a dictionary for video {video_data.video_id}, converting")
                    analysis_result = {}
                    
                analysis_result["forensic_relevance"] = analysis_result.get("forensic_relevance", "unknown")
                analysis_result["red_flags"] = analysis_result.get("red_flags", [])
                analysis_result["summary"] = analysis_result.get("summary", "No summary available")
                
                analysis_result["video_id"] = video_data.video_id
                analysis_result["title"] = video_data.title
                
                video_data.forensic_summary = analysis_result
                
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
            except YouTubeError as e:
                self.logger.error(f"YouTube error analyzing transcript for video {video_data.video_id}: {str(e)}\nTraceback: {traceback.format_exc()}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error analyzing transcript for video {video_data.video_id}: {str(e)}\nTraceback: {traceback.format_exc()}")
                return {
                    "forensic_relevance": "unknown",
                    "red_flags": ["Analysis failed with error"],
                    "summary": f"Error analyzing transcript: {str(e)}",
                    "video_id": video_data.video_id,
                    "title": video_data.title
                }
        
        try:
            self.logger.info(f"Starting transcript analysis for video {video_data.video_id} with retries")
            result = await _analyze_with_retry()
            self.logger.info(f"Completed transcript analysis for video {video_data.video_id}")
            return result
        except RetryError as e:
            self.logger.error(f"Failed to analyze transcript after {self.retry_attempts} attempts for video {video_data.video_id}: {str(e.last_attempt.exception())}\nTraceback: {traceback.format_exc()}")
            
            return {
                "forensic_relevance": "unknown",
                "red_flags": ["Analysis failed after multiple attempts"],
                "summary": f"Failed to analyze transcript after {self.retry_attempts} attempts",
                "video_id": video_data.video_id,
                "title": video_data.title
            }
        
    async def get_channel_videos(self, channel_name: str, max_results: int = 20) -> List[Dict[str, Any]]:
        retry_decorator = self._get_retry_decorator("get_channel_videos")
        
        @retry_decorator
        async def _get_channel_videos_with_retry():
            try:
                self.logger.info(f"Getting videos from channel: {channel_name}")
                
                channel_id_result = await self.youtube_tool.run(
                    action="get_channel_id",
                    channel_name=channel_name
                )
                
                if not channel_id_result.success or not channel_id_result.data.get("channel_id"):
                    error_msg = f"Failed to get channel ID for {channel_name}: {channel_id_result.error}"
                    self.logger.error(error_msg)
                    
                    if "rate limit" in str(channel_id_result.error).lower():
                        raise YouTubeRateLimitError(error_msg)
                    elif "connection" in str(channel_id_result.error).lower() or "network" in str(channel_id_result.error).lower():
                        raise YouTubeConnectionError(error_msg)
                    else:
                        raise YouTubeDataError(error_msg)
                
                channel_id = channel_id_result.data.get("channel_id")
                
                playlists_result = await self.youtube_tool.run(
                    action="get_playlists",
                    channel_id=channel_id,
                    max_results=5
                )
                
                all_videos = []
                
                if playlists_result.success and playlists_result.data.get("playlists"):
                    playlists = playlists_result.data.get("playlists")
                    
                    if not isinstance(playlists, list):
                        self.logger.warning(f"Invalid playlists format for channel {channel_name}")
                    else:
                        for playlist in playlists:
                            if not playlist.get("id"):
                                continue
                                
                            playlist_id = playlist.get("id")
                            
                            playlist_videos_result = await self.youtube_tool.run(
                                action="get_playlist_videos",
                                playlist_id=playlist_id,
                                max_results=10
                            )
                            
                            if playlist_videos_result.success and playlist_videos_result.data.get("videos"):
                                videos = playlist_videos_result.data.get("videos")
                                
                                if isinstance(videos, list):
                                    all_videos.extend(videos)
                                    self.logger.info(f"Added {len(videos)} videos from playlist {playlist_id}")
                                else:
                                    self.logger.warning(f"Invalid videos format for playlist {playlist_id}")
                            
                            if len(all_videos) >= max_results:
                                break
                
                if len(all_videos) < max_results:
                    search_query = f"channel:{channel_name}"
                    search_result = await self.youtube_tool.run(
                        action="search_videos",
                        query=search_query,
                        max_results=max_results - len(all_videos)
                    )
                    
                    if search_result.success and search_result.data.get("videos"):
                        videos = search_result.data.get("videos")
                        
                        if isinstance(videos, list):
                            all_videos.extend(videos)
                            self.logger.info(f"Added {len(videos)} videos from search for channel {channel_name}")
                        else:
                            self.logger.warning(f"Invalid videos format from channel search for {channel_name}")
                
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
        retry_decorator = self._get_retry_decorator("generate_video_summary")
        
        @retry_decorator
        async def _generate_with_retry():
            try:
                self.logger.info(f"Generating video summary for {len(videos_data)} videos about {company}")
                self.logger.debug(f"Summary generation call stack:\n{traceback.format_stack()}")
                
                sorted_videos = sorted(videos_data, key=lambda x: x.relevance_score, reverse=True)
                
                top_videos = sorted_videos[:10]
                self.logger.info(f"Selected top {len(top_videos)} videos for detailed summary")
                
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
                
                try:
                    llm_provider = await get_llm_provider()
                    self.logger.info(f"LLM provider obtained for video summary generation for {company}")
                except Exception as e:
                    self.logger.error(f"Failed to get LLM provider for summary generation: {str(e)}\nTraceback: {traceback.format_exc()}")
                    raise YouTubeDataError(f"LLM provider error in summary generation: {str(e)}")
                
                variables = {
                    "company": company,
                    "video_count": len(videos_data),
                    "analyzed_count": len([v for v in videos_data if v.forensic_summary is not None]),
                    "video_summaries": json.dumps(video_summaries)
                }
                
                try:
                    self.logger.info(f"Getting prompts for summary generation for {company}")
                    system_prompt, human_prompt = self.prompt_manager.get_prompt(
                        agent_name=self.name,
                        operation="generate_summary",
                        variables=variables
                    )
                    self.logger.info(f"Successfully retrieved prompts for summary generation")
                except Exception as e:
                    error_details = f"Error getting prompts for summary generation: {str(e)}\nTraceback: {traceback.format_exc()}"
                    self.logger.error(error_details)
                    
                    # Use default prompts as fallback
                    self.logger.info("Using default prompts for summary generation")
                    system_prompt = f"""You are a forensic financial analyst. Analyze these YouTube video summaries about {company} and identify key insights and red flags."""
                    
                    human_prompt = f"""I have analyzed {len(videos_data)} YouTube videos about {company}, and I need you to create a summary of the findings.
                    
Here are the summaries of the most relevant videos:

{json.dumps(video_summaries, indent=2)}

Please provide:
1. An overall assessment of the YouTube content
2. Key insights from the videos
3. Any potential red flags or concerning issues
4. A list of the most notable videos and why they matter

Format your response as a JSON with these keys: "overall_assessment", "key_insights", "red_flags", "notable_videos", and "summary"."""
                
                input_message = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                try:
                    # Fixed: Changed model_name to model
                    model = self.config.get("models", {}).get("summary")
                    self.logger.info(f"Using model {model} for summary generation")
                    
                    response = await llm_provider.generate_text(
                        input_message,
                        model=model
                    )
                    
                    # Log the raw response for debugging
                    self.logger.info(f"Received LLM response for summary generation, length: {len(response)}")
                    self.logger.debug(f"Raw summary response preview: {response[:300]}...")
                except Exception as e:
                    self.logger.error(f"Error in LLM text generation for summary: {str(e)}\nTraceback: {traceback.format_exc()}")
                    raise YouTubeDataError(f"LLM generation error in summary: {str(e)}")
                
                if not response:
                    self.logger.warning("Empty response received from LLM for summary generation")
                    raise YouTubeDataError("Empty response from LLM for summary generation")
                
                try:
                    self.logger.info("Parsing JSON response for summary")
                    summary_result = self._parse_json_response(response, "video summary")
                    self.logger.info(f"Successfully parsed JSON for summary with {len(summary_result)} keys")
                except Exception as e:
                    self.logger.error(f"Error parsing JSON for summary: {str(e)}\nTraceback: {traceback.format_exc()}")
                    raise YouTubeDataError(f"JSON parsing error in summary: {str(e)}")
                
                try:
                    self.logger.info("Validating summary result")
                    self._validate_result(
                        summary_result,
                        ["overall_assessment", "key_insights", "red_flags", "notable_videos"],
                        "video summary"
                    )
                    self.logger.info("Summary validation successful")
                except YouTubeValidationError as e:
                    self.logger.warning(f"Validation failed for summary: {str(e)}\nAttempting to fix...")
                    # Try to fix the summary by adding missing fields
                    if not isinstance(summary_result, dict):
                        summary_result = {}
                    
                    if "overall_assessment" not in summary_result:
                        summary_result["overall_assessment"] = "Unknown"
                    if "key_insights" not in summary_result:
                        summary_result["key_insights"] = []
                    if "red_flags" not in summary_result:
                        summary_result["red_flags"] = []
                    if "notable_videos" not in summary_result:
                        summary_result["notable_videos"] = []
                    
                    self.logger.info("Fixed summary structure with missing fields")
                
                summary_result["timestamp"] = datetime.now().isoformat()
                summary_result["company"] = company
                summary_result["total_videos_analyzed"] = len(videos_data)
                
                num_flags = len(summary_result.get('red_flags', []))
                self.logger.info(f"Generated video summary for {company} with {num_flags} red flags")
                
                return summary_result
                
            except YouTubeValidationError as e:
                self.logger.error(f"Validation error generating video summary: {str(e)}\nTraceback: {traceback.format_exc()}")
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
            except YouTubeError as e:
                self.logger.error(f"YouTube error generating video summary: {str(e)}\nTraceback: {traceback.format_exc()}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error generating video summary: {str(e)}\nTraceback: {traceback.format_exc()}")
                raise YouTubeDataError(f"Error generating video summary: {str(e)}")
        
        try:
            self.logger.info(f"Starting video summary generation for {company} with retries")
            result = await _generate_with_retry()
            self.logger.info(f"Successfully completed video summary generation for {company}")
            return result
        except RetryError as e:
            error_details = f"Failed to generate video summary after {self.retry_attempts} attempts"
            if e.last_attempt.exception():
                error_details += f": {str(e.last_attempt.exception())}"
            
            self.logger.error(f"{error_details}\nTraceback: {traceback.format_exc()}")
            
            return {
                "overall_assessment": "Error",
                "key_insights": ["Failed to generate summary after multiple attempts"],
                "red_flags": ["Summary generation failed"],
                "notable_videos": [],
                "summary": f"Analysis of YouTube content for {company} encountered errors after {self.retry_attempts} attempts.",
                "company": company,
                "timestamp": datetime.now().isoformat(),
                "total_videos_analyzed": len(videos_data)
            }
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self.run(state)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.logger.info("Starting YouTube agent run method")
            self.logger.debug(f"YouTube agent run call stack:\n{traceback.format_stack()}")
            
            self._log_start(state)
            
            company = state.get("company", "")
            industry = state.get("industry", "")
            research_plan = state.get("research_plan", [])
            
            # Set status early to properly track that we started processing
            state["youtube_agent_status"] = "RUNNING"
            self.logger.info(f"YouTube agent status set to RUNNING for company: {company}")
            
            if not company:
                self.logger.error("Company name is missing!")
                return {**state, "goto": "meta_agent", "youtube_agent_status": "ERROR", "error": "Company name is missing"}
            
            # Load templates for youtube analyze_transcript if not found
            try:
                self.logger.info("Preloading YouTube templates to ensure they exist")
                if not os.path.exists("prompts/youtube_agent"):
                    self.logger.warning("Creating youtube_agent prompt directory")
                    os.makedirs("prompts/youtube_agent", exist_ok=True)
                    
                # Create default templates if they don't exist
                if not os.path.exists("prompts/youtube_agent/analyze_transcript_system.j2"):
                    with open("prompts/youtube_agent/analyze_transcript_system.j2", "w") as f:
                        f.write("""You are a forensic financial analyst examining YouTube videos.
                        
Your task is to analyze this transcript for any information that could be relevant to a financial forensic investigation of {{ company }}.

Focus on:
1. Mentions of financial irregularities, accounting issues, or suspicious practices
2. References to regulatory investigations, lawsuits, or legal actions
3. Discussions of management issues, conflicts of interest, or governance problems
4. Information about company structure, subsidiaries, or relationships that could enable financial manipulation
5. Comments on the company's financial health, debt, or financial reporting quality

Return your analysis as a JSON with:
- forensic_relevance: "high", "medium", or "low" 
- red_flags: a list of concerning items found (empty list if none)
- summary: a concise analysis focusing only on financially relevant information""")
                
                if not os.path.exists("prompts/youtube_agent/analyze_transcript_human.j2"):
                    with open("prompts/youtube_agent/analyze_transcript_human.j2", "w") as f:
                        f.write("""Please analyze this transcript for the video titled "{{ video_title }}" about {{ company }}.

Description: {{ video_description }}

Transcript:
{{ transcript }}

Analyze the transcript for any information that would be relevant in a financial forensic investigation. Focus specifically on any potential red flags, financial irregularities, accounting issues, or regulatory/legal concerns.

Return your analysis in JSON format with these fields:
- forensic_relevance: "high", "medium", or "low" based on how relevant this content is for investigation
- red_flags: list of specific issues or concerns found (empty list if none)
- summary: concise overview of key points relevant to financial forensic analysis""")
            except Exception as e:
                self.logger.warning(f"Error setting up templates: {str(e)}. Will use default prompts.")
            
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
            
            search_errors = []
            transcript_errors = []
            analysis_errors = []
            
            all_videos = []
            
            # We'll retry each query up to 3 times before giving up
            for query in youtube_queries:
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        self.logger.info(f"Searching for videos with query: {query} (attempt {retry_count+1})")
                        videos = await self.search_videos(query, max_results=5)
                        all_videos.extend(videos)
                        self.logger.info(f"Found {len(videos)} videos for query: {query}")
                        break  # Success, exit retry loop
                    except Exception as e:
                        retry_count += 1
                        error_msg = f"Error searching for '{query}': {str(e)}"
                        self.logger.error(error_msg)
                        if retry_count >= max_retries:
                            search_errors.append(error_msg)
                        else:
                            self.logger.info(f"Retrying query {query} ({retry_count}/{max_retries})")
                            await asyncio.sleep(2)  # Brief pause before retry
            
            unique_videos = {}
            for video in all_videos:
                video_id = video.get("id")
                if video_id and video_id not in unique_videos:
                    unique_videos[video_id] = video
            
            channels = {}
            for video in unique_videos.values():
                channel = video.get("channel_title")
                if channel and channel not in channels:
                    channels[channel] = 1
                elif channel:
                    channels[channel] += 1
            
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
            
            unique_videos = {}
            for video in all_videos:
                video_id = video.get("id")
                if video_id and video_id not in unique_videos:
                    unique_videos[video_id] = video
            
            self.logger.info(f"Collected {len(unique_videos)} unique videos about {company}")
            
            if not unique_videos:
                youtube_results["error"] = f"No videos found for {company}"
                state["youtube_results"] = youtube_results
                state["youtube_status"] = "DONE"
                
                goto = "meta_agent"
                if state.get("synchronous_pipeline", False):
                    goto = state.get("next_agent", "meta_agent")
                    
                self._log_completion({**state, "goto": goto})
                return {**state, "goto": goto}
            
            videos_data = []
            for video in unique_videos.values():
                video_data = VideoData(
                    video_id=video.get("id"),
                    title=video.get("title", "Unknown title"),
                    channel=video.get("channel_title", "Unknown channel"),
                    description=video.get("description", "")
                )
                videos_data.append(video_data)
            
            def relevance_score(video):
                title_lower = video.title.lower()
                keywords = ["fraud", "scandal", "controversy", "investigation", "sec", "lawsuit"]
                score = sum(3 for kw in keywords if kw in title_lower)
                score += 2 if company.lower() in title_lower else 0
                return score
                
            videos_data.sort(key=relevance_score, reverse=True)
            
            max_to_analyze = min(20, len(videos_data))
            videos_to_analyze = videos_data[:max_to_analyze]
            
            self.logger.info(f"Analyzing {len(videos_to_analyze)} videos out of {len(videos_data)} total")
            
            batch_size = 5
            for i in range(0, len(videos_to_analyze), batch_size):
                batch = videos_to_analyze[i:i+batch_size]
                
                transcript_tasks = []
                for video_data in batch:
                    task = asyncio.create_task(self.get_transcript(video_data.video_id))
                    transcript_tasks.append((video_data, task))
                    
                for video_data, task in transcript_tasks:
                    try:
                        self.logger.info(f"Awaiting transcript task for video: {video_data.video_id}")
                        transcript = await task
                        if transcript:
                            video_data.transcript = transcript
                            self.logger.info(f"Got transcript for video: {video_data.video_id}, length: {len(transcript)}")
                        else:
                            self.logger.warning(f"No transcript available for video: {video_data.video_id}")
                    except Exception as e:
                        error_msg = f"Error getting transcript for video {video_data.video_id}: {str(e)}"
                        self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
                        transcript_errors.append(error_msg)
                
                self.logger.info(f"Completed transcript tasks for batch of {len(batch)} videos, creating analysis tasks")
                analysis_tasks = []
                for video_data in batch:
                    if video_data.transcript:
                        try:
                            self.logger.info(f"Creating analysis task for video: {video_data.video_id}")
                            task = asyncio.create_task(self.analyze_transcript(video_data, company))
                            analysis_tasks.append((video_data, task))
                        except Exception as e:
                            error_msg = f"Error creating analysis task for video {video_data.video_id}: {str(e)}"
                            self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
                            analysis_errors.append(error_msg)
                
                self.logger.info(f"Created {len(analysis_tasks)} analysis tasks, awaiting results")
                for video_data, task in analysis_tasks:
                    try:
                        self.logger.info(f"Awaiting analysis task for video: {video_data.video_id}")
                        analysis = await task
                        if analysis:
                            relevance = analysis.get('forensic_relevance', 'unknown')
                            self.logger.info(f"Successfully analyzed transcript for video: {video_data.video_id} - Relevance: {relevance}")
                        else:
                            self.logger.warning(f"No analysis result for video: {video_data.video_id}")
                    except Exception as e:
                        error_msg = f"Error analyzing transcript for video {video_data.video_id}: {str(e)}"
                        self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
                        analysis_errors.append(error_msg)
                
                self.logger.info(f"Completed analysis tasks for batch, sleeping before next batch")
                await asyncio.sleep(1)
            
            try:
                self.logger.info(f"Starting summary generation with {len(videos_data)} videos")
                videos_with_analysis = [v for v in videos_data if v.forensic_summary is not None]
                self.logger.info(f"Found {len(videos_with_analysis)}/{len(videos_data)} videos with valid analysis")
                
                if not videos_with_analysis:
                    self.logger.warning(f"No videos with valid analysis available for summary generation for {company}")
                    summary = {
                        "overall_assessment": "Unknown",
                        "key_insights": [f"No video analysis available for {company}"],
                        "red_flags": [],
                        "notable_videos": [],
                        "summary": f"No analyzable YouTube content found for {company}."
                    }
                    self.logger.info(f"Created default empty summary for {company}")
                else:
                    self.logger.info(f"Generating video summary for {company} with {len(videos_with_analysis)} analyzed videos")
                    try:
                        summary = await self.generate_video_summary(videos_with_analysis, company)
                        self.logger.info(f"Successfully generated summary for {company}")
                    except Exception as e:
                        error_details = f"Error in generate_video_summary: {str(e)}\nTraceback: {traceback.format_exc()}"
                        self.logger.error(error_details)
                        summary = {
                            "overall_assessment": "Error",
                            "key_insights": [f"Error generating summary: {str(e)}"],
                            "red_flags": ["Summary generation failed with error"],
                            "notable_videos": [],
                            "summary": f"Error generating summary for {company}."
                        }
                        self.logger.info(f"Created fallback summary after error for {company}")
                
                youtube_results["summary"] = summary
                
                if "red_flags" in summary and isinstance(summary["red_flags"], list):
                    youtube_results["red_flags"] = summary["red_flags"]
                    count = len(youtube_results["red_flags"])
                    self.logger.info(f"Extracted {count} red flags from summary for {company}")
                else:
                    self.logger.warning(f"No valid red_flags in summary for {company}")
                    youtube_results["red_flags"] = []
                    
                self.logger.info(f"Completed YouTube summary for {company} with {len(youtube_results['red_flags'])} red flags")
                
            except Exception as e:
                error_msg = f"Error generating summary for {company}: {str(e)}"
                error_details = f"Error details: {traceback.format_exc()}"
                self.logger.error(f"{error_msg}\n{error_details}")
                youtube_results["summary"] = {
                    "error": error_msg,
                    "overall_assessment": "Error",
                    "key_insights": ["Failed to generate summary"],
                    "red_flags": [],
                    "notable_videos": [],
                    "summary": f"Failed to generate summary for YouTube content about {company}."
                }
                self.logger.info(f"Created error summary for {company} after exception")
            
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
                
            self.logger.info(f"YouTube agent completing successfully, returning control to {goto}")
            state["youtube_agent_status"] = "DONE"
            self._log_completion({**state, "goto": goto})
            self.logger.info("YouTube agent run method completed")
            return {**state, "goto": goto}
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Unhandled error in youtube_agent run method: {str(e)}\n{tb}"
            self.logger.error(error_msg)
            self.logger.error(f"Error occurred at:\n{traceback.format_exc()}")
            
            # Create youtube_results if it doesn't exist yet
            if 'youtube_results' not in state:
                self.logger.info("Creating empty youtube_results structure due to error")
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
            else:
                # Update existing results with error information
                self.logger.info("Updating existing youtube_results with error information")
                if "summary" not in state['youtube_results']:
                    state['youtube_results']["summary"] = {}
                
                state['youtube_results']["summary"]["error"] = error_msg
                state['youtube_results']["summary"]["overall_assessment"] = "Error"
                state['youtube_results']["error"] = error_msg
            
            # Make sure we have videos list even if empty
            if "videos" not in state['youtube_results']:
                state['youtube_results']["videos"] = []
                
            self.logger.info("Setting youtube_agent_status to ERROR and returning to meta_agent")
            
            error_result = {
                **state, 
                "goto": "meta_agent", 
                "error": error_msg,
                "youtube_agent_status": "ERROR"
            }
            
            try:
                self._log_completion(error_result)
            except Exception as log_error:
                self.logger.error(f"Error during error logging: {str(log_error)}")
            
            self.logger.info("YouTube agent completed with errors")
            return error_result