import os
import asyncio
import json
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.postgres_tool import PostgresTool


class ReportTemplate:
    def __init__(self, template_name: str, sections: List[Dict[str, Any]], 
                 variables: Dict[str, Any], metadata: Dict[str, Any]):
        self.name = template_name
        self.sections = sections  # [{"name": "section_name", "content": "content_template"}]
        self.variables = variables  # {"variable_name": "default_value"}
        self.metadata = metadata  # {"created_at": "timestamp", "author": "author_name", ...}


class EnhancedWriterAgent(BaseAgent):
    name = "writer_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager(self.name)
        self.postgres_tool = PostgresTool(config.get("postgres", {}))
        
        # Enhanced reporting capabilities
        self.templates = {}
        self.loaded_templates = False
        self.revision_history = {}
        self.report_sections = {}
        self.current_template = None
        self.feedback_history = []
        
        # Max number of sections to process concurrently
        self.max_concurrent_sections = config.get("writer", {}).get("max_concurrent_sections", 3)
        
        # Load templates as part of initialization
        asyncio.create_task(self.load_templates())
    
    async def load_templates(self) -> None:
        """Load report templates from database"""
        try:
            result = await self.postgres_tool.run(
                command="execute_query",
                query="SELECT * FROM report_templates"
            )
            
            if result.success and result.data:
                templates = {}
                for template_data in result.data:
                    template_name = template_data["template_name"]
                    sections = json.loads(template_data["sections"])
                    variables = json.loads(template_data["variables"])
                    metadata = json.loads(template_data["metadata"])
                    
                    templates[template_name] = ReportTemplate(
                        template_name=template_name,
                        sections=sections,
                        variables=variables,
                        metadata=metadata
                    )
                    
                self.templates = templates
                self.loaded_templates = True
                self.logger.info(f"Loaded {len(templates)} report templates from database")
            else:
                # Create and save default template if none exists
                await self.create_default_template()
                
        except Exception as e:
            self.logger.error(f"Error loading templates: {str(e)}")
            # Create and save default template if error
            await self.create_default_template()
    
    async def create_default_template(self) -> None:
        """Create and save a default report template"""
        template_name = "standard_forensic_report"
        sections = [
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
            },
            {
                "name": "other_events",
                "title": "Other Notable Events",
                "type": "markdown",
                "required": False,
                "variables": ["company", "event_summaries"]
            },
            {
                "name": "pattern_recognition",
                "title": "Pattern Recognition",
                "type": "markdown",
                "required": False,
                "variables": ["company", "events"]
            },
            {
                "name": "recommendations",
                "title": "Recommendations",
                "type": "markdown",
                "required": True,
                "variables": ["company", "top_events"]
            }
        ]
        
        variables = {
            "company": "",
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "top_events": [],
            "total_events": 0,
            "event": {},
            "event_summaries": []
        }
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "author": "System",
            "description": "Standard forensic analysis report template",
            "version": "1.0"
        }
        
        default_template = ReportTemplate(
            template_name=template_name,
            sections=sections,
            variables=variables,
            metadata=metadata
        )
        
        self.templates[template_name] = default_template
        self.loaded_templates = True
        
        # Save to database
        try:
            await self.postgres_tool.run(
                command="execute_query",
                query="INSERT INTO report_templates (template_name, sections, variables, metadata) VALUES ($1, $2, $3, $4) ON CONFLICT (template_name) DO UPDATE SET sections = $2, variables = $3, metadata = $4",
                params=[
                    template_name,
                    json.dumps(sections),
                    json.dumps(variables),
                    json.dumps(metadata)
                ]
            )
            self.logger.info(f"Created and saved default template: {template_name}")
        except Exception as e:
            self.logger.error(f"Error saving default template: {str(e)}")
    
    async def select_template(self, company: str, analysis_results: Dict) -> Optional[ReportTemplate]:
        """Select the most appropriate template based on analysis results"""
        if not self.loaded_templates:
            await self.load_templates()
            
        # For now, use the standard template if available
        if "standard_forensic_report" in self.templates:
            template = self.templates["standard_forensic_report"]
            self.logger.info(f"Selected template: {template.name}")
            self.current_template = template
            return template
            
        # Fallback if no templates are available
        self.logger.warning("No templates available, creating default")
        await self.create_default_template()
        template = self.templates["standard_forensic_report"]
        self.current_template = template
        return template
    
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
        
        event_scores = []
        for name in events.keys():
            meta = event_metadata.get(name, {})
            score = meta.get("importance_score", 0)
            event_scores.append((name, score, name))
        
        # Sort by score (descending) and then by name (for consistent tie-breaking)
        sorted_events = sorted(event_scores, key=lambda x: (x[1], x[2]), reverse=True)
        
        # Extract just the event names
        top_events = [name for name, _, _ in sorted_events[:max_detailed_events]]
        other_events = [name for name, _, _ in sorted_events[max_detailed_events:]]
        
        self.logger.info(f"Selected {len(top_events)} top events and {len(other_events)} other events")
        return top_events, other_events
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_detailed_event_section(self, company: str, event_name: str, event_data: List[Dict], template_section: Dict) -> str:
        """Generate a detailed analysis section for a single event using the template."""
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
                "articles": json.dumps(article_summaries, indent=2),
                "section_title": template_section.get("title", "Event Analysis"),
                "section_requirements": json.dumps(template_section.get("requirements", {}))
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
            
            # Save to database for future reference
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content, event_name) VALUES ($1, $2, $3, $4) ON CONFLICT (company, section_name, event_name) DO UPDATE SET section_content = $3",
                    params=[company, "key_events", detailed_section, event_name]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save event section to database: {str(db_error)[:100]}...")
            
            return f"## {event_name}\n\n{detailed_section}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error generating detailed event section for '{event_name}': {str(e)}")
            return f"## {event_name}\n\nUnable to generate detailed analysis due to technical error: {str(e)[:100]}...\n\n"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_other_events_section(self, company: str, events: Dict, event_metadata: Dict, other_event_names: List[str], template_section: Dict) -> str:
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
                "event_summaries": json.dumps(event_summaries, indent=2),
                "section_title": template_section.get("title", "Other Notable Events"),
                "section_requirements": json.dumps(template_section.get("requirements", {}))
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
            
            # Save to database for future reference
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content) VALUES ($1, $2, $3) ON CONFLICT (company, section_name) DO UPDATE SET section_content = $3",
                    params=[company, "other_events", other_events_section]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save other events section to database: {str(db_error)[:100]}...")
            
            return f"# Other Notable Events\n\n{other_events_section}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error generating other events section: {str(e)}")
            
            # Generate a basic summary of events without LLM
            basic_summary = "The following events were also identified but not analyzed in detail:\n\n"
            for event in event_summaries:
                basic_summary += f"- **{event['name']}** ({event['article_count']} articles)\n"
                
            return f"# Other Notable Events\n\n{basic_summary}\n\n"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_executive_summary(self, company: str, top_events: List[str], all_events: Dict, event_metadata: Dict, template_section: Dict) -> str:
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
        
        # Get corporate and YouTube data if available
        corporate_data = {}
        youtube_data = {}
        
        # Check if these are in the state held by the current template
        if hasattr(self, 'current_template') and self.current_template:
            template_vars = self.current_template.variables
            if "corporate_results" in template_vars:
                corporate_data = template_vars["corporate_results"]
            if "youtube_results" in template_vars:
                youtube_data = template_vars["youtube_results"]
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "top_event_info": json.dumps(top_event_info, indent=2),
                "total_events": len(all_events),
                "section_title": template_section.get("title", "Executive Summary"),
                "section_requirements": json.dumps(template_section.get("requirements", {})),
                "corporate_data": json.dumps(corporate_data),
                "youtube_data": json.dumps(youtube_data)
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
            
            # Save to database for future reference
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content) VALUES ($1, $2, $3) ON CONFLICT (company, section_name) DO UPDATE SET section_content = $3",
                    params=[company, "executive_summary", summary]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save executive summary to database: {str(db_error)[:100]}...")
            
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
    async def generate_pattern_section(self, company: str, top_events: List[str], event_metadata: Dict, template_section: Dict) -> str:
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
                "events": json.dumps(event_info, indent=2),
                "section_title": template_section.get("title", "Pattern Recognition"),
                "section_requirements": json.dumps(template_section.get("requirements", {}))
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
            
            # Save to database for future reference
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content) VALUES ($1, $2, $3) ON CONFLICT (company, section_name) DO UPDATE SET section_content = $3",
                    params=[company, "pattern_recognition", pattern_section]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save pattern section to database: {str(db_error)[:100]}...")
            
            return f"# Pattern Recognition\n\n{pattern_section}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error generating pattern section: {str(e)}")
            return f"# Pattern Recognition\n\nThe system attempted to identify patterns across {len(top_events)} events but encountered an error: {str(e)[:100]}...\n\n"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_recommendations(self, company: str, top_events: List[str], template_section: Dict) -> str:
        """Generate recommendations based on the analysis."""
        self.logger.info(f"Generating recommendations based on {len(top_events)} top events")
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "top_events": json.dumps(top_events, indent=2),
                "section_title": template_section.get("title", "Recommendations"),
                "section_requirements": json.dumps(template_section.get("requirements", {}))
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
            
            # Save to database for future reference
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content) VALUES ($1, $2, $3) ON CONFLICT (company, section_name) DO UPDATE SET section_content = $3",
                    params=[company, "recommendations", recommendations]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save recommendations to database: {str(db_error)[:100]}...")
            
            return f"# Recommendations\n\n{recommendations}\n\n"
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return f"# Recommendations\n\n1. Consider performing a more detailed investigation into {company}\n2. Gather additional information about each major event\n3. Review the available data for completeness\n\n"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def revise_section(self, company: str, section_name: str, current_content: str, feedback: str) -> str:
        """Revise a section based on feedback"""
        self.logger.info(f"Revising section '{section_name}' based on feedback")
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "section_name": section_name,
                "current_content": current_content,
                "feedback": feedback
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="revise_section",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("report"))
            revised_content = response.strip()
            
            # Save revision to database for future reference
            revision_id = f"{section_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_revisions (company, section_name, revision_id, original_content, revised_content, feedback) VALUES ($1, $2, $3, $4, $5, $6)",
                    params=[company, section_name, revision_id, current_content, revised_content, feedback]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save revision to database: {str(db_error)[:100]}...")
            
            # Update the current section content
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="UPDATE report_sections SET section_content = $1 WHERE company = $2 AND section_name = $3",
                    params=[revised_content, company, section_name]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to update section content in database: {str(db_error)[:100]}...")
            
            # Track revision in memory
            if section_name not in self.revision_history:
                self.revision_history[section_name] = []
            
            self.revision_history[section_name].append({
                "revision_id": revision_id,
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback
            })
            
            return revised_content
            
        except Exception as e:
            self.logger.error(f"Error revising section '{section_name}': {str(e)}")
            return current_content
    
    async def generate_section_concurrently(self, company: str, section_name: str, section_template: Dict, 
                                          state: Dict[str, Any]) -> Optional[str]:
        """Generate a single report section based on template and state"""
        self.logger.info(f"Generating report section: {section_name}")
        
        try:
            # Get data needed for section generation
            research_results = state.get("research_results", {})
            event_metadata = state.get("event_metadata", {})
            analysis_results = state.get("analysis_results", {})
            top_events = state.get("top_events", [])
            other_events = state.get("other_events", [])
            
            # Generate section based on type
            if section_name == "executive_summary":
                return await self.generate_executive_summary(
                    company, top_events, research_results, event_metadata, section_template
                )
            elif section_name == "key_events":
                # Generate each event section and combine
                detailed_events_section = "# Key Events Analysis\n\n"
                
                # Process events concurrently with limit
                event_sections = []
                for event_name in top_events:
                    event_data = research_results.get(event_name, [])
                    if event_data:
                        event_section = await self.generate_detailed_event_section(
                            company, event_name, event_data, section_template
                        )
                        event_sections.append(event_section)
                        
                detailed_events_section += "\n".join(event_sections)
                return detailed_events_section
                
            elif section_name == "other_events" and other_events:
                return await self.generate_other_events_section(
                    company, research_results, event_metadata, other_events, section_template
                )
                
            elif section_name == "pattern_recognition" and len(top_events) > 1:
                return await self.generate_pattern_section(
                    company, top_events, event_metadata, section_template
                )
                
            elif section_name == "recommendations":
                return await self.generate_recommendations(
                    company, top_events, section_template
                )
                
            else:
                self.logger.warning(f"Unknown section type: {section_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating section '{section_name}': {str(e)}")
            return None
    
    async def generate_sections_concurrently(self, company: str, template: ReportTemplate, 
                                           state: Dict[str, Any]) -> Dict[str, str]:
        """Generate all report sections concurrently with semaphore to limit concurrency"""
        self.logger.info(f"Generating all report sections using template: {template.name}")
        
        # First, determine top events for detailed analysis
        research_results = state.get("research_results", {})
        event_metadata = state.get("event_metadata", {})
        
        # Select top events if not already selected
        if "top_events" not in state or "other_events" not in state:
            top_events, other_events = self._select_top_events(
                research_results, event_metadata, max_detailed_events=6
            )
            state["top_events"] = top_events
            state["other_events"] = other_events
            self.logger.info(f"Selected {len(top_events)} top events and {len(other_events)} other events")
            
        # Make template variables available to all sections
        template.variables.update({
            "company": company,
            "top_events": state.get("top_events", []),
            "other_events": state.get("other_events", []),
            "total_events": len(research_results),
            "corporate_results": state.get("corporate_results", {}),
            "youtube_results": state.get("youtube_results", {})
        })
        
        sections = {}
        semaphore = asyncio.Semaphore(self.max_concurrent_sections)
        
        async def generate_with_semaphore(section_name, section_template):
            async with semaphore:
                section_content = await self.generate_section_concurrently(
                    company, section_name, section_template, state
                )
                return section_name, section_content
        
        # Create tasks for each section
        tasks = []
        for section in template.sections:
            section_name = section.get("name")
            if not section_name:
                continue
                
            task = asyncio.create_task(
                generate_with_semaphore(section_name, section)
            )
            tasks.append(task)
        
        # Wait for all sections to complete
        for task in asyncio.as_completed(tasks):
            try:
                section_name, section_content = await task
                if section_content:
                    sections[section_name] = section_content
                    self.report_sections[section_name] = section_content
            except Exception as e:
                self.logger.error(f"Error in section generation task: {str(e)}")
        
        return sections
    
    async def save_debug_report(self, company: str, full_report: str) -> str:
        """Save a debug copy of the report to disk and database."""
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
        
        # Also save to database
        try:
            await self.postgres_tool.run(
                command="execute_query",
                query="INSERT INTO final_reports (company, report_date, report_content, filename) VALUES ($1, $2, $3, $4) ON CONFLICT (company) DO UPDATE SET report_date = $2, report_content = $3, filename = $4",
                params=[company, datetime.now().isoformat(), full_report, debug_filename]
            )
            self.logger.info(f"Report saved to database for {company}")
        except Exception as e:
            self.logger.error(f"Could not save report to database: {str(e)}")
            
        return debug_filename
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_meta_feedback(self, company: str, full_report: str) -> Dict[str, Any]:
        """Generate feedback on report quality using LLM"""
        self.logger.info(f"Generating meta feedback for {company} report")
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "report": full_report
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="meta_feedback",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = await llm_provider.generate_text(
                input_message, 
                model_name=self.config.get("models", {}).get("evaluation")
            )
            
            try:
                feedback = json.loads(response)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse meta feedback JSON")
                feedback = {
                    "quality_score": 7,
                    "strengths": ["Unable to parse specific strengths"],
                    "weaknesses": ["Unable to parse specific weaknesses"],
                    "improvements": ["Unable to parse specific improvements"]
                }
            
            # Track feedback for historical purposes
            self.feedback_history.append({
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback
            })
            
            # Save feedback to database
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_feedback (company, feedback_date, feedback_data) VALUES ($1, $2, $3)",
                    params=[company, datetime.now().isoformat(), json.dumps(feedback)]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save feedback to database: {str(db_error)[:100]}...")
            
            return feedback
            
        except Exception as e:
            self.logger.error(f"Error generating meta feedback: {str(e)}")
            return {
                "quality_score": 5,
                "strengths": ["Error generating specific strengths"],
                "weaknesses": ["Error occurred during feedback generation"],
                "improvements": ["Unable to suggest specific improvements due to error"]
            }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_executive_briefing(self, company: str, full_report: str) -> str:
        """Generate a concise executive briefing based on the full report"""
        self.logger.info(f"Generating executive briefing for {company}")
        
        try:
            llm_provider = await get_llm_provider()
            
            variables = {
                "company": company,
                "report": full_report
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="executive_briefing",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = await llm_provider.generate_text(
                input_message, 
                model_name=self.config.get("models", {}).get("report")
            )
            
            briefing = response.strip()
            
            # Save briefing to database
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO executive_briefings (company, briefing_date, briefing_content) VALUES ($1, $2, $3) ON CONFLICT (company) DO UPDATE SET briefing_date = $2, briefing_content = $3",
                    params=[company, datetime.now().isoformat(), briefing]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save executive briefing to database: {str(db_error)[:100]}...")
            
            return briefing
            
        except Exception as e:
            self.logger.error(f"Error generating executive briefing: {str(e)}")
            return f"# Executive Briefing for {company}\n\nUnable to generate executive briefing due to technical error."
    
    async def apply_iterative_improvements(self, company: str, report_sections: Dict[str, str], 
                                        feedback: Dict[str, Any]) -> Dict[str, str]:
        """Apply iterative improvements to report sections based on meta feedback"""
        self.logger.info(f"Applying iterative improvements based on feedback (quality score: {feedback.get('quality_score', 0)})")
        
        # If quality score is already high, skip improvements
        quality_score = feedback.get("quality_score", 0)
        if quality_score >= 8:
            self.logger.info(f"Report quality already high ({quality_score}/10), skipping improvements")
            return report_sections
        
        # Get specific improvement suggestions
        improvements = feedback.get("improvements", [])
        if not improvements:
            self.logger.info("No specific improvements suggested")
            return report_sections
            
        improved_sections = report_sections.copy()
        
        # Process each improvement suggestion
        for improvement in improvements:
            # Try to determine which section the improvement applies to
            target_section = None
            improvement_text = improvement.lower()
            
            if "executive summary" in improvement_text:
                target_section = "executive_summary"
            elif "key events" in improvement_text:
                target_section = "key_events"
            elif "pattern" in improvement_text:
                target_section = "pattern_recognition"
            elif "recommendation" in improvement_text:
                target_section = "recommendations"
            elif "other events" in improvement_text:
                target_section = "other_events"
            
            # Apply improvement to section if found
            if target_section and target_section in improved_sections:
                self.logger.info(f"Applying improvement to section: {target_section}")
                revised_content = await self.revise_section(
                    company, 
                    target_section, 
                    improved_sections[target_section],
                    improvement
                )
                improved_sections[target_section] = revised_content
                
        return improved_sections
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the writer agent workflow."""
        self._log_start(state)
        
        if state.get("analyst_status") != "DONE":
            self.logger.info("Analyst not done yet. Waiting...")
            return {**state, "goto": "meta_agent", "writer_status": "WAITING"}
        
        self.logger.info("Analyst work complete. Starting report generation.")
        
        company = state.get("company", "Unknown Company")
        research_results = state.get("research_results", {})
        event_metadata = state.get("event_metadata", {})
        analysis_results = state.get("analysis_results", {})
        
        # Optional corporate and YouTube results
        corporate_results = state.get("corporate_results", {})
        youtube_results = state.get("youtube_results", {})
        
        # Input validation
        if not isinstance(research_results, dict):
            self.logger.error(f"Expected research_results to be a dictionary, got {type(research_results)}")
            research_results = {}
            
        if not isinstance(event_metadata, dict):
            self.logger.error(f"Expected event_metadata to be a dictionary, got {type(event_metadata)}")
            event_metadata = {}
        
        # Initialize section results
        report_sections = {}
        
        try:
            # Select template based on analysis
            template = await self.select_template(company, analysis_results)
            
            # If top events not already selected, select them now
            if "top_events" not in state or "other_events" not in state:
                top_events, other_events = self._select_top_events(
                    research_results, event_metadata, max_detailed_events=6
                )
                state["top_events"] = top_events
                state["other_events"] = other_events
            else:
                top_events = state["top_events"]
                other_events = state["other_events"]
                
            self.logger.info(f"Selected {len(top_events)} events for detailed analysis and {len(other_events)} for summary")
            
            # Generate all sections concurrently
            report_sections = await self.generate_sections_concurrently(company, template, state)
            
            # Combine sections to form full report
            timestamp = datetime.now().strftime("%Y-%m-%d")
            report_sections["header"] = f"# Forensic News Analysis Report: {company}\n\nReport Date: {timestamp}\n\n"
            
            # Determine section order from template
            section_order = [section["name"] for section in template.sections]
            section_order.insert(0, "header")  # Add header at beginning
            
            # Assemble full report in correct order
            full_report_parts = []
            for section_name in section_order:
                if section_name in report_sections:
                    full_report_parts.append(report_sections[section_name])
                    
            full_report = "\n".join(full_report_parts)
            
            # Generate meta feedback
            feedback = await self.generate_meta_feedback(company, full_report)
            
            # Apply iterative improvements if needed
            if self.config.get("writer", {}).get("enable_iterative_improvement", True):
                improved_sections = await self.apply_iterative_improvements(company, report_sections, feedback)
                
                # Regenerate full report with improvements
                improved_report_parts = []
                for section_name in section_order:
                    if section_name in improved_sections:
                        improved_report_parts.append(improved_sections[section_name])
                
                full_report = "\n".join(improved_report_parts)
                
                # Re-evaluate after improvements
                new_feedback = await self.generate_meta_feedback(company, full_report)
                self.logger.info(f"Report quality improved from {feedback.get('quality_score', 0)} to {new_feedback.get('quality_score', 0)}")
                feedback = new_feedback
            
            # Generate executive briefing
            executive_briefing = await self.generate_executive_briefing(company, full_report)
            
            # Save final report
            report_filename = await self.save_debug_report(company, full_report)
            
            # Update state
            state["final_report"] = full_report
            state["report_sections"] = report_sections
            state["report_feedback"] = feedback
            state["executive_briefing"] = executive_briefing
            state["report_filename"] = report_filename
            state["writer_status"] = "DONE"
            state["top_events"] = top_events
            state["other_events"] = other_events
            
            self.logger.info("Report generation successfully completed.")
            
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
            
            # Save fallback report
            await self.save_debug_report(company, fallback_report)
            
            state["final_report"] = fallback_report
            state["error"] = str(e)
            state["writer_status"] = "ERROR"
            self.logger.info("Generated fallback report due to errors.")
        
        self._log_completion({**state, "goto": "meta_agent"})
        return {**state, "goto": "meta_agent"}