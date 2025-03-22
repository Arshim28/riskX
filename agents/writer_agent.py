from typing import Dict, List, Any, Tuple
import json
import os
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger


class WriterAgent(BaseAgent):
    name = "writer_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager(self.name)
    
    def _select_top_events(self, events: Dict, event_metadata: Dict, max_detailed_events: int = 6) -> Tuple[List[str], List[str]]:
        """
        Select top events for detailed analysis based on importance scores.
        
        Args:
            events: Dictionary of event data
            event_metadata: Dictionary containing metadata about events, including importance scores
            max_detailed_events: Maximum number of events to select for detailed analysis
            
        Returns:
            Tuple containing:
                1. List of event names selected for detailed analysis
                2. List of remaining event names for summary
        """
        self.logger.info(f"Selecting top events from {len(events)} total events")
        
        # Create list of (event_name, score) tuples with proper default handling
        event_scores = []
        for name in events.keys():
            # Get metadata with fallback to empty dict
            meta = event_metadata.get(name, {})
            # Get importance_score with fallback to 0
            score = meta.get("importance_score", 0)
            # Use a secondary sorting key (event name) for consistent ordering of tied scores
            event_scores.append((name, score, name))
        
        # Sort by score (descending) and then by name (for consistent tie-breaking)
        sorted_events = sorted(event_scores, key=lambda x: (x[1], x[2]), reverse=True)
        
        # Extract just the event names
        top_events = [name for name, _, _ in sorted_events[:max_detailed_events]]
        other_events = [name for name, _, _ in sorted_events[max_detailed_events:]]
        
        self.logger.info(f"Selected {len(top_events)} top events and {len(other_events)} other events")
        return top_events, other_events
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_detailed_event_section(self, company: str, event_name: str, event_data: List[Dict]) -> str:
        """Generate a detailed analysis section for a single event."""
        self.logger.info(f"Generating detailed analysis for event: {event_name}")
        
        article_summaries = [{
            "title": a.get("title", ""),
            "source": a.get("source", "Unknown"),
            "date": a.get("date", "Unknown"),
            "snippet": a.get("snippet", "")
        } for a in event_data]
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "event_name": event_name,
                "articles": json.dumps(article_summaries, indent=2)
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="detailed_event_analysis",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("report"))
            detailed_section = response.strip()
            
            return f"## {event_name}\n\n{detailed_section}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error generating detailed event section for '{event_name}': {str(e)}")
            return f"## {event_name}\n\nUnable to generate detailed analysis due to technical error: {str(e)[:100]}...\n\n"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_other_events_section(self, company: str, events: Dict, event_metadata: Dict, other_event_names: List[str]) -> str:
        """Generate a summary section for events not covered in detail."""
        if not other_event_names:
            return ""
            
        self.logger.info(f"Generating summary for {len(other_event_names)} other events")
        
        event_summaries = []
        for event_name in other_event_names:
            articles = events.get(event_name, [])
            importance = event_metadata.get(event_name, {}).get("importance_score", 0)
            
            article_summaries = [{
                "title": a.get("title", ""),
                "source": a.get("source", "Unknown"),
                "date": a.get("date", "Unknown")
            } for a in articles[:3]]  # Limit to first 3 articles
            
            event_summaries.append({
                "name": event_name,
                "importance": importance,
                "article_count": len(articles),
                "articles": article_summaries
            })
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "event_summaries": json.dumps(event_summaries, indent=2)
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="other_events_summary",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("report"))
            other_events_section = response.strip()
            
            return f"# Other Notable Events\n\n{other_events_section}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error generating other events section: {str(e)}")
            
            # Generate a basic summary of events without LLM
            basic_summary = "The following events were also identified but not analyzed in detail:\n\n"
            for event in event_summaries:
                basic_summary += f"- **{event['name']}** ({event['article_count']} articles)\n"
                
            return f"# Other Notable Events\n\n{basic_summary}\n\n"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_executive_summary(self, company: str, top_events: List[str], all_events: Dict, event_metadata: Dict) -> str:
        """Generate an executive summary based on the top events."""
        self.logger.info(f"Generating executive summary with focus on top {len(top_events)} events")
        
        top_event_info = []
        for event_name in top_events:
            metadata = event_metadata.get(event_name, {})
            top_event_info.append({
                "name": event_name,
                "importance_score": metadata.get("importance_score", 0),
                "is_quarterly_report": metadata.get("is_quarterly_report", False)
            })
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "top_event_info": json.dumps(top_event_info, indent=2),
                "total_events": len(all_events)
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="executive_summary",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("report"))
            summary = response.strip()
            
            return f"# Executive Summary\n\n{summary}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            
            # Generate a basic executive summary without LLM
            basic_summary = f"""
This report presents findings from an analysis of {len(all_events)} events related to {company}. 
The events span various categories including financial reporting, regulatory actions, and news coverage.

Key events analyzed in detail include:
"""
            for event in top_event_info:
                basic_summary += f"- {event['name']}\n"
                
            return f"# Executive Summary\n\n{basic_summary}\n\n"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_pattern_section(self, company: str, top_events: List[str], event_metadata: Dict) -> str:
        """Generate a pattern recognition section identifying common themes across events."""
        if len(top_events) <= 1:
            return ""
            
        self.logger.info(f"Generating pattern recognition section for {len(top_events)} events")
        
        event_info = [{
            "name": event,
            "importance": event_metadata.get(event, {}).get("importance_score", 0)
        } for event in top_events]
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "events": json.dumps(event_info, indent=2)
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="pattern_recognition",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("report"))
            pattern_section = response.strip()
            
            return f"# Pattern Recognition\n\n{pattern_section}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error generating pattern section: {str(e)}")
            return f"# Pattern Recognition\n\nThe system attempted to identify patterns across {len(top_events)} events but encountered an error: {str(e)[:100]}...\n\n"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_recommendations(self, company: str, top_events: List[str]) -> str:
        """Generate recommendations based on the analysis."""
        self.logger.info(f"Generating recommendations based on {len(top_events)} top events")
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "top_events": json.dumps(top_events, indent=2)
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="recommendations",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("report"))
            recommendations = response.strip()
            
            return f"# Recommendations\n\n{recommendations}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return f"# Recommendations\n\n1. Consider performing a more detailed investigation into {company}\n2. Gather additional information about each major event\n3. Review the available data for completeness\n\n"
    
    async def save_debug_report(self, company: str, full_report: str) -> None:
        """Save a debug copy of the report to disk."""
        try:
            debug_dir = "debug/reports"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
            
            # Add timestamp to provide more uniqueness
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sanitized_company = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in company)
            debug_filename = f"{debug_dir}/{sanitized_company}_{timestamp}.md"
            
            with open(debug_filename, "w") as f:
                f.write(full_report)
                
            self.logger.info(f"Debug copy saved to {debug_filename}")
            
        except Exception as e:
            self.logger.error(f"Could not save debug copy: {str(e)}")
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the writer agent workflow."""
        self._log_start(state)
        
        if state.get("analyst_status") != "DONE":
            self.logger.info("Analyst not done yet. Waiting...")
            return {**state, "goto": "writer_agent"}
        
        self.logger.info("Analyst work complete. Starting report generation.")
        
        company = state.get("company", "Unknown Company")
        research_results = state.get("research_results", {})
        event_metadata = state.get("event_metadata", {})
        analysis_results = state.get("analysis_results", {})
        
        # Input validation
        if not isinstance(research_results, dict):
            self.logger.error(f"Expected research_results to be a dictionary, got {type(research_results)}")
            research_results = {}
            
        if not isinstance(event_metadata, dict):
            self.logger.error(f"Expected event_metadata to be a dictionary, got {type(event_metadata)}")
            event_metadata = {}
        
        report_sections = []
        
        try:
            top_events, other_events = self._select_top_events(
                research_results, event_metadata, max_detailed_events=6
            )
            
            self.logger.info(f"Selected {len(top_events)} events for detailed analysis and {len(other_events)} for summary")
            
            timestamp = datetime.now().strftime("%Y-%m-%d")
            report_sections.append(f"# Forensic News Analysis Report: {company}\n\nReport Date: {timestamp}\n\n")
            
            # Generate executive summary
            executive_summary = await self.generate_executive_summary(
                company, top_events, research_results, event_metadata
            )
            report_sections.append(executive_summary)
            
            # Generate detailed sections for top events
            detailed_events_section = "# Key Events Analysis\n\n"
            for event_name in top_events:
                event_data = research_results.get(event_name, [])
                if event_data:
                    event_section = await self.generate_detailed_event_section(
                        company, event_name, event_data
                    )
                    detailed_events_section += event_section
            
            report_sections.append(detailed_events_section)
            
            # Generate summary for other events
            if other_events:
                other_events_section = await self.generate_other_events_section(
                    company, research_results, event_metadata, other_events
                )
                report_sections.append(other_events_section)
            
            # Generate pattern recognition section
            pattern_section = await self.generate_pattern_section(
                company, top_events, event_metadata
            )
            if pattern_section:
                report_sections.append(pattern_section)
            
            # Generate recommendations
            recommendations = await self.generate_recommendations(company, top_events)
            report_sections.append(recommendations)
            
            full_report = "\n".join(report_sections)
            
            state["final_report"] = full_report
            state["report_sections"] = report_sections
            state["top_events"] = top_events
            state["other_events"] = other_events
            
            self.logger.info("Report generation successfully completed.")
            
            # Save debug copy
            await self.save_debug_report(company, full_report)
            
        except Exception as e:
            self.logger.error(f"Error in report generation: {str(e)}")
            
            # Create a more informative fallback report
            research_count = len(research_results) if isinstance(research_results, dict) else 0
            
            fallback_report = f"""
# Forensic News Analysis Report: {company}

Report Date: {datetime.now().strftime("%Y-%m-%d")}

## Executive Summary

This report presents the findings of a forensic news analysis conducted on {company}. Due to technical issues during report generation, this is a simplified version of the analysis.

## Key Findings

The analysis identified {research_count} significant events related to {company}.

"""
            
            # Include event names in the fallback report if available
            if isinstance(research_results, dict) and research_results:
                fallback_report += "## Events Identified\n\n"
                for event_name in research_results.keys():
                    article_count = len(research_results[event_name]) if isinstance(research_results[event_name], list) else 0
                    fallback_report += f"- {event_name} ({article_count} articles)\n"
            
            fallback_report += """
## Technical Issue

The full report could not be generated due to a technical error. Please refer to the logs for more information.
"""
            
            state["final_report"] = fallback_report
            state["error"] = str(e)
            self.logger.info("Generated fallback report due to errors.")
        
        self._log_completion({**state, "goto": "END"})
        return {**state, "goto": "END"}