from typing import Dict, List, Any, Optional, Union, Tuple
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


class YouTubeAgent(BaseAgent):
    name = "youtube_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager(self.name)
        self.youtube_tool = YoutubeTool(config.get("youtube", {}))
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for videos related to the query"""
        self.logger.info(f"Searching YouTube for: {query}")
        
        search_result = await self.youtube_tool.run(
            action="search_videos",
            query=query,
            max_results=max_results
        )
        
        if not search_result.success:
            self.logger.error(f"YouTube search failed: {search_result.error}")
            return []
            
        return search_result.data.get("videos", [])
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript for a video"""
        self.logger.info(f"Getting transcript for video: {video_id}")
        
        transcript_result = await self.youtube_tool.run(
            action="get_transcript",
            video_id=video_id
        )
        
        if not transcript_result.success:
            self.logger.warning(f"Failed to get transcript for video {video_id}: {transcript_result.error}")
            return None
            
        return transcript_result.data.get("transcript")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_transcript(self, video_data: VideoData, company: str) -> Dict[str, Any]:
        """Analyze video transcript for forensic insights"""
        self.logger.info(f"Analyzing transcript for video: {video_data.video_id}")
        
        if not video_data.transcript:
            self.logger.warning(f"No transcript available for video {video_data.video_id}")
            return {
                "forensic_relevance": "low",
                "red_flags": [],
                "summary": "No transcript available for analysis"
            }
        
        # Use LLM to analyze the transcript
        llm_provider = await get_llm_provider()
        
        variables = {
            "company": company,
            "video_title": video_data.title,
            "video_description": video_data.description,
            "transcript": video_data.transcript[:8000]  # Limit length to avoid token limits
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
            prompt=input_message,
            model_name=self.config.get("models", {}).get("analysis")
        )
        
        try:
            analysis_result = json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse transcript analysis JSON for video {video_data.video_id}")
            analysis_result = {
                "forensic_relevance": "unknown",
                "red_flags": [],
                "summary": "Error parsing analysis results"
            }
        
        video_data.forensic_summary = analysis_result
        
        # Set relevance score based on forensic_relevance
        relevance_mapping = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
            "unknown": 0.1
        }
        video_data.relevance_score = relevance_mapping.get(
            analysis_result.get("forensic_relevance", "unknown").lower(), 
            0.1
        )
        
        return analysis_result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_channel_videos(self, channel_name: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Get videos from a specific channel"""
        self.logger.info(f"Getting videos from channel: {channel_name}")
        
        # First get the channel ID
        channel_id_result = await self.youtube_tool.run(
            action="get_channel_id",
            channel_name=channel_name
        )
        
        if not channel_id_result.success or not channel_id_result.data.get("channel_id"):
            self.logger.error(f"Failed to get channel ID for {channel_name}: {channel_id_result.error}")
            return []
            
        channel_id = channel_id_result.data.get("channel_id")
        
        # Get playlists from the channel
        playlists_result = await self.youtube_tool.run(
            action="get_playlists",
            channel_id=channel_id,
            max_results=5  # Limit to top playlists
        )
        
        all_videos = []
        
        # Get videos from each playlist
        if playlists_result.success and playlists_result.data.get("playlists"):
            for playlist in playlists_result.data.get("playlists"):
                playlist_id = playlist.get("id")
                
                playlist_videos_result = await self.youtube_tool.run(
                    action="get_playlist_videos",
                    playlist_id=playlist_id,
                    max_results=10  # Limit videos per playlist
                )
                
                if playlist_videos_result.success and playlist_videos_result.data.get("videos"):
                    all_videos.extend(playlist_videos_result.data.get("videos"))
                    
                if len(all_videos) >= max_results:
                    break
        
        # If we didn't get enough videos from playlists, search for videos from the channel
        if len(all_videos) < max_results:
            search_query = f"channel:{channel_name}"
            search_result = await self.youtube_tool.run(
                action="search_videos",
                query=search_query,
                max_results=max_results - len(all_videos)
            )
            
            if search_result.success and search_result.data.get("videos"):
                all_videos.extend(search_result.data.get("videos"))
        
        # Deduplicate based on video ID
        unique_videos = {}
        for video in all_videos:
            video_id = video.get("id")
            if video_id and video_id not in unique_videos:
                unique_videos[video_id] = video
        
        return list(unique_videos.values())
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_video_summary(self, videos_data: List[VideoData], company: str) -> Dict[str, Any]:
        """Generate a summary of all analyzed videos"""
        self.logger.info(f"Generating video summary for {len(videos_data)} videos about {company}")
        
        # Sort videos by relevance score
        sorted_videos = sorted(videos_data, key=lambda x: x.relevance_score, reverse=True)
        
        # Prepare video summaries for LLM
        video_summaries = []
        for i, video in enumerate(sorted_videos[:10]):  # Limit to top 10 videos
            summary = {
                "video_id": video.video_id,
                "title": video.title,
                "channel": video.channel,
                "forensic_relevance": video.forensic_summary.get("forensic_relevance", "unknown") if video.forensic_summary else "unknown",
                "red_flags": video.forensic_summary.get("red_flags", []) if video.forensic_summary else [],
                "summary": video.forensic_summary.get("summary", "") if video.forensic_summary else "No summary available"
            }
            video_summaries.append(summary)
        
        # Use LLM to generate a comprehensive summary
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
            prompt=input_message,
            model_name=self.config.get("models", {}).get("summary")
        )
        
        try:
            summary_result = json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse video summary JSON for {company}")
            summary_result = {
                "overall_assessment": "Unknown",
                "key_insights": ["Error parsing summary results"],
                "red_flags": [],
                "notable_videos": [],
                "summary": f"Analysis of YouTube content for {company} encountered parsing errors."
            }
        
        return summary_result
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the YouTube agent workflow"""
        self._log_start(state)
        
        company = state.get("company", "")
        industry = state.get("industry", "")
        research_plan = state.get("research_plan", [])
        
        if not company:
            self.logger.error("Company name is missing!")
            return {**state, "goto": "meta_agent", "error": "Company name is missing", "youtube_status": "ERROR"}
        
        if not research_plan:
            self.logger.error("No research plan provided!")
            return {**state, "goto": "meta_agent", "error": "No research plan provided", "youtube_status": "ERROR"}
        
        # Extract YouTube search queries from the research plan
        youtube_queries = []
        current_plan = research_plan[-1]  # Get most recent research plan
        
        for category, queries in current_plan.get("query_categories", {}).items():
            if isinstance(queries, list):
                youtube_queries.extend([f"{company} {q}" for q in queries])
            else:
                youtube_queries.append(f"{company} {queries}")
        
        # Add company name as a basic query if no others are available
        if not youtube_queries:
            youtube_queries.append(company)
            if industry:
                youtube_queries.append(f"{company} {industry}")
            youtube_queries.append(f"{company} fraud")
            youtube_queries.append(f"{company} scandal")
            youtube_queries.append(f"{company} controversy")
        
        self.logger.info(f"YouTube research plan for {company} with {len(youtube_queries)} queries")
        
        # Initialize results structure
        youtube_results = {
            "videos": [],
            "channels": {},
            "summary": {},
            "red_flags": []
        }
        
        # Collect videos from all queries
        all_videos = []
        
        for query in youtube_queries:
            try:
                videos = await self.search_videos(query, max_results=5)
                all_videos.extend(videos)
                self.logger.info(f"Found {len(videos)} videos for query: {query}")
            except Exception as e:
                self.logger.error(f"Error searching videos for query '{query}': {str(e)}")
        
        # Deduplicate videos based on ID
        unique_videos = {}
        for video in all_videos:
            video_id = video.get("id")
            if video_id and video_id not in unique_videos:
                unique_videos[video_id] = video
        
        # Collect unique channel names
        channels = {}
        for video in unique_videos.values():
            channel = video.get("channel_title")
            if channel and channel not in channels:
                channels[channel] = 1
            elif channel:
                channels[channel] += 1
        
        # Get more videos from top channels
        top_channels = sorted(channels.items(), key=lambda x: x[1], reverse=True)[:3]
        for channel_name, _ in top_channels:
            try:
                channel_videos = await self.get_channel_videos(channel_name, max_results=10)
                all_videos.extend(channel_videos)
                youtube_results["channels"][channel_name] = len(channel_videos)
                self.logger.info(f"Found {len(channel_videos)} videos from channel: {channel_name}")
            except Exception as e:
                self.logger.error(f"Error getting videos from channel '{channel_name}': {str(e)}")
        
        # Re-deduplicate after adding channel videos
        unique_videos = {}
        for video in all_videos:
            video_id = video.get("id")
            if video_id and video_id not in unique_videos:
                unique_videos[video_id] = video
        
        self.logger.info(f"Collected {len(unique_videos)} unique videos about {company}")
        
        # Process videos - get transcripts and analyze
        videos_data = []
        
        # Create a list of VideoData objects
        for video in unique_videos.values():
            video_data = VideoData(
                video_id=video.get("id"),
                title=video.get("title"),
                channel=video.get("channel_title", "Unknown"),
                description=video.get("description", "")
            )
            videos_data.append(video_data)
        
        # Prioritize video processing - analyze only top videos based on view count or relevance
        max_to_analyze = min(20, len(videos_data))  # Limit the number of videos to analyze
        videos_to_analyze = videos_data[:max_to_analyze]
        
        self.logger.info(f"Analyzing {len(videos_to_analyze)} videos out of {len(videos_data)} total")
        
        # Process videos in batches to avoid overloading
        batch_size = 5
        for i in range(0, len(videos_to_analyze), batch_size):
            batch = videos_to_analyze[i:i+batch_size]
            
            # Get transcripts
            transcript_tasks = []
            for video_data in batch:
                task = asyncio.create_task(self.get_transcript(video_data.video_id))
                transcript_tasks.append((video_data, task))
                
            # Wait for transcripts and process
            for video_data, task in transcript_tasks:
                try:
                    transcript = await task
                    if transcript:
                        video_data.transcript = transcript
                        self.logger.info(f"Got transcript for video: {video_data.video_id}")
                    else:
                        self.logger.warning(f"No transcript available for video: {video_data.video_id}")
                except Exception as e:
                    self.logger.error(f"Error getting transcript for video {video_data.video_id}: {str(e)}")
            
            # Analyze transcripts
            analysis_tasks = []
            for video_data in batch:
                if video_data.transcript:
                    task = asyncio.create_task(self.analyze_transcript(video_data, company))
                    analysis_tasks.append((video_data, task))
            
            # Wait for analysis
            for video_data, task in analysis_tasks:
                try:
                    analysis = await task
                    self.logger.info(f"Analyzed transcript for video: {video_data.video_id} - Relevance: {analysis.get('forensic_relevance', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"Error analyzing transcript for video {video_data.video_id}: {str(e)}")
        
        # Generate summary
        try:
            summary = await self.generate_video_summary(videos_data, company)
            youtube_results["summary"] = summary
            
            # Extract red flags
            if "red_flags" in summary and isinstance(summary["red_flags"], list):
                youtube_results["red_flags"] = summary["red_flags"]
                
            self.logger.info(f"Generated YouTube summary for {company} with {len(youtube_results['red_flags'])} red flags")
        except Exception as e:
            self.logger.error(f"Error generating video summary for {company}: {str(e)}")
            youtube_results["summary"] = {
                "error": f"Failed to generate summary: {str(e)}",
                "overall_assessment": "Unknown"
            }
        
        # Prepare video results for state
        for video_data in videos_data:
            video_result = {
                "video_id": video_data.video_id,
                "title": video_data.title,
                "channel": video_data.channel,
                "description": video_data.description,
                "has_transcript": video_data.transcript is not None,
                "transcript_length": len(video_data.transcript) if video_data.transcript else 0,
                "forensic_relevance": video_data.forensic_summary.get("forensic_relevance", "unknown") if video_data.forensic_summary else "unknown",
                "red_flags": video_data.forensic_summary.get("red_flags", []) if video_data.forensic_summary else [],
                "summary": video_data.forensic_summary.get("summary", "") if video_data.forensic_summary else "No summary available"
            }
            youtube_results["videos"].append(video_result)
        
        # Update state
        state["youtube_results"] = youtube_results
        state["youtube_status"] = "DONE"
        
        # Determine next step
        goto = "meta_agent"  # Default to meta_agent for orchestration
        if state.get("synchronous_pipeline", False):  # If running in synchronous mode
            goto = state.get("next_agent", "meta_agent")
            
        self._log_completion({**state, "goto": goto})
        return {**state, "goto": goto}