import os
import asyncio
import json
import traceback
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.postgres_tool import PostgresTool


class ReportSection(BaseAgent):
    """Represents a section in a report with its status and content."""
    
    name = "report_section"
    
    def __init__(self, section_name: str, section_title: str, required: bool = True):
        self.section_name = section_name
        self.section_title = section_title
        self.required = required
        self.content = ""
        self.status = "PENDING"  # PENDING, GENERATING, DONE, ERROR
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.retries = 0
        self.max_retries = 3
    
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Base implementation, not actually used
        return state


class ReportTemplate:
    def __init__(self, template_name: str, sections: List[Dict[str, Any]], 
                 variables: Dict[str, Any], metadata: Dict[str, Any]):
        self.name = template_name
        self.sections = sections
        self.variables = variables
        self.metadata = metadata


class WriterAgent(BaseAgent):
    name = "writer_agent"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager()
        self.postgres_tool = PostgresTool(config.get("postgres", {}))
        
        self.templates = {}
        self.loaded_templates = False
        self.revision_history = {}
        self.report_sections = {}
        self.section_statuses = {}
        self.current_template = None
        self.feedback_history = []
        
        # Lock for concurrent state updates
        self.state_lock = asyncio.Lock()
        
        self.max_concurrent_sections = config.get("writer", {}).get("max_concurrent_sections", 3)
        self.enable_iterative_improvement = config.get("writer", {}).get("enable_iterative_improvement", True)
        self.quality_threshold = config.get("writer", {}).get("quality_threshold", 7)
        
        # Error tracking
        self.last_error = None
        self.error_count = 0
        self.critical_error_threshold = 3
        
        # Initialize templates
        asyncio.create_task(self.load_templates())
    
    async def load_templates(self) -> None:
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
                await self.create_default_template()
                
        except Exception as e:
            self.logger.error(f"Error loading templates: {str(e)}")
            await self.create_default_template()
    
    async def create_default_template(self) -> None:
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
        if not self.loaded_templates:
            await self.load_templates()
            
        if "standard_forensic_report" in self.templates:
            template = self.templates["standard_forensic_report"]
            self.logger.info(f"Selected template: {template.name}")
            self.current_template = template
            return template
            
        self.logger.warning("No templates available, creating default")
        await self.create_default_template()
        template = self.templates["standard_forensic_report"]
        self.current_template = template
        return template
    
    def _select_top_events(self, events: Dict, event_metadata: Dict, max_detailed_events: int = 6) -> Tuple[List[str], List[str]]:
        self.logger.info(f"Selecting top events from {len(events)} total events")
        
        event_scores = []
        for name in events.keys():
            meta = event_metadata.get(name, {})
            score = meta.get("importance_score", 0)
            event_scores.append((name, score, name))
        
        sorted_events = sorted(event_scores, key=lambda x: (x[1], x[2]), reverse=True)
        
        top_events = [name for name, _, _ in sorted_events[:max_detailed_events]]
        other_events = [name for name, _, _ in sorted_events[max_detailed_events:]]
        
        self.logger.info(f"Selected {len(top_events)} top events and {len(other_events)} other events")
        return top_events, other_events
    
    async def update_section_status(self, section_name: str, status: str, error: Optional[str] = None) -> None:
        """Update the status of a report section with proper state tracking."""
        async with self.state_lock:
            if section_name not in self.section_statuses:
                self.section_statuses[section_name] = {
                    "status": "PENDING",
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                    "retries": 0
                }
                
            prev_status = self.section_statuses[section_name]["status"]
            self.section_statuses[section_name]["status"] = status
            
            # Status transitions with timestamp updates
            if status == "GENERATING" and prev_status != "GENERATING":
                self.section_statuses[section_name]["started_at"] = datetime.now().isoformat()
                    
            elif status in ["DONE", "ERROR"] and prev_status != status:
                self.section_statuses[section_name]["completed_at"] = datetime.now().isoformat()
                
                if status == "ERROR":
                    self.section_statuses[section_name]["error"] = error
                    self.last_error = error
                    self.error_count += 1
            
            self.logger.info(f"Updated section {section_name} status: {prev_status} -> {status}")
    
    async def should_retry_section(self, section_name: str) -> bool:
        """Determine if a failed section should be retried."""
        if section_name not in self.section_statuses:
            return False
            
        section = self.section_statuses[section_name]
        
        # Check if we've exceeded retry limit
        if section["retries"] >= 3:  # Max retries
            self.logger.info(f"Section {section_name} has reached maximum retries (3)")
            return False
            
        # Increment retry count
        section["retries"] += 1
        
        # Reset status to PENDING
        section["status"] = "PENDING"
        section["error"] = None
        
        self.logger.info(f"Retrying section {section_name} (attempt {section['retries']})")
        return True
    
    async def get_section_status(self, section_name: str) -> str:
        """Get current status of a section."""
        if section_name not in self.section_statuses:
            return "PENDING"
        return self.section_statuses[section_name]["status"]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_detailed_event_section(self, company: str, event_name: str, event_data: List[Dict], template_section: Dict) -> str:
        await self.update_section_status(f"key_events_{event_name}", "GENERATING")
        
        try:
            self.logger.info(f"Generating detailed analysis for event: {event_name}")
            
            article_summaries = [{
                "title": a.get("title", ""),
                "source": a.get("source", "Unknown"),
                "date": a.get("date", "Unknown"),
                "snippet": a.get("snippet", "")
            } for a in event_data]
            
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
            
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content, event_name) VALUES ($1, $2, $3, $4) ON CONFLICT (company, section_name, event_name) DO UPDATE SET section_content = $3",
                    params=[company, "key_events", detailed_section, event_name]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save event section to database: {str(db_error)[:100]}...")
            
            await self.update_section_status(f"key_events_{event_name}", "DONE")
            return f"## {event_name}\n\n{detailed_section}\n\n"
        except Exception as e:
            error_msg = f"Error generating detailed event section for {event_name}: {str(e)}"
            self.logger.error(error_msg)
            await self.update_section_status(f"key_events_{event_name}", "ERROR", error_msg)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_other_events_section(self, company: str, events: Dict, event_metadata: Dict, other_event_names: List[str], template_section: Dict) -> str:
        if not other_event_names:
            return ""
            
        await self.update_section_status("other_events", "GENERATING")
        
        try:
            self.logger.info(f"Generating summary for {len(other_event_names)} other events")
            
            event_summaries = []
            for event_name in other_event_names:
                articles = events.get(event_name, [])
                importance = event_metadata.get(event_name, {}).get("importance_score", 0)
                
                article_summaries = [{
                    "title": a.get("title", ""),
                    "source": a.get("source", "Unknown"),
                    "date": a.get("date", "Unknown")
                } for a in articles[:3]]
                
                event_summaries.append({
                    "name": event_name,
                    "importance": importance,
                    "article_count": len(articles),
                    "articles": article_summaries
                })
            
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
            
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content) VALUES ($1, $2, $3) ON CONFLICT (company, section_name) DO UPDATE SET section_content = $3",
                    params=[company, "other_events", other_events_section]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save other events section to database: {str(db_error)[:100]}...")
            
            await self.update_section_status("other_events", "DONE")
            return f"# Other Notable Events\n\n{other_events_section}\n\n"
        except Exception as e:
            error_msg = f"Error generating other events section: {str(e)}"
            self.logger.error(error_msg)
            await self.update_section_status("other_events", "ERROR", error_msg)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_executive_summary(self, company: str, top_events: List[str], all_events: Dict, event_metadata: Dict, template_section: Dict) -> str:
        await self.update_section_status("executive_summary", "GENERATING")
        
        try:
            self.logger.info(f"Generating executive summary with focus on top {len(top_events)} events")
            
            top_event_info = []
            for event_name in top_events:
                metadata = event_metadata.get(event_name, {})
                top_event_info.append({
                    "name": event_name,
                    "importance_score": metadata.get("importance_score", 0),
                    "is_quarterly_report": metadata.get("is_quarterly_report", False)
                })
            
            corporate_data = {}
            youtube_data = {}
            
            if hasattr(self, 'current_template') and self.current_template:
                template_vars = self.current_template.variables
                if "corporate_results" in template_vars:
                    corporate_data = template_vars["corporate_results"]
                if "youtube_results" in template_vars:
                    youtube_data = template_vars["youtube_results"]
            
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
            
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content) VALUES ($1, $2, $3) ON CONFLICT (company, section_name) DO UPDATE SET section_content = $3",
                    params=[company, "executive_summary", summary]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save executive summary to database: {str(db_error)[:100]}...")
            
            await self.update_section_status("executive_summary", "DONE")
            return f"# Executive Summary\n\n{summary}\n\n"
        except Exception as e:
            error_msg = f"Error generating executive summary: {str(e)}"
            self.logger.error(error_msg)
            await self.update_section_status("executive_summary", "ERROR", error_msg)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_pattern_section(self, company: str, top_events: List[str], event_metadata: Dict, template_section: Dict) -> str:
        if len(top_events) <= 1:
            return ""
            
        await self.update_section_status("pattern_recognition", "GENERATING")
        
        try:
            self.logger.info(f"Generating pattern recognition section for {len(top_events)} events")
            
            event_info = [{
                "name": event,
                "importance": event_metadata.get(event, {}).get("importance_score", 0)
            } for event in top_events]
            
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
            
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content) VALUES ($1, $2, $3) ON CONFLICT (company, section_name) DO UPDATE SET section_content = $3",
                    params=[company, "pattern_recognition", pattern_section]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save pattern section to database: {str(db_error)[:100]}...")
            
            await self.update_section_status("pattern_recognition", "DONE")
            return f"# Pattern Recognition\n\n{pattern_section}\n\n"
        except Exception as e:
            error_msg = f"Error generating pattern section: {str(e)}"
            self.logger.error(error_msg)
            await self.update_section_status("pattern_recognition", "ERROR", error_msg)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_recommendations(self, company: str, top_events: List[str], template_section: Dict) -> str:
        await self.update_section_status("recommendations", "GENERATING")
        
        try:
            self.logger.info(f"Generating recommendations based on {len(top_events)} top events")
            
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
            
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_sections (company, section_name, section_content) VALUES ($1, $2, $3) ON CONFLICT (company, section_name) DO UPDATE SET section_content = $3",
                    params=[company, "recommendations", recommendations]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save recommendations to database: {str(db_error)[:100]}...")
            
            await self.update_section_status("recommendations", "DONE")
            return f"# Recommendations\n\n{recommendations}\n\n"
        except Exception as e:
            error_msg = f"Error generating recommendations: {str(e)}"
            self.logger.error(error_msg)
            await self.update_section_status("recommendations", "ERROR", error_msg)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def revise_section(self, company: str, section_name: str, current_content: str, feedback: str) -> str:
        self.logger.info(f"Revising section '{section_name}' based on feedback")
        
        await self.update_section_status(f"{section_name}_revision", "GENERATING")
        
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
            
            revision_id = f"{section_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO report_revisions (company, section_name, revision_id, original_content, revised_content, feedback) VALUES ($1, $2, $3, $4, $5, $6)",
                    params=[company, section_name, revision_id, current_content, revised_content, feedback]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to save revision to database: {str(db_error)[:100]}...")
            
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="UPDATE report_sections SET section_content = $1 WHERE company = $2 AND section_name = $3",
                    params=[revised_content, company, section_name]
                )
            except Exception as db_error:
                self.logger.warning(f"Failed to update section content in database: {str(db_error)[:100]}...")
            
            if section_name not in self.revision_history:
                self.revision_history[section_name] = []
            
            self.revision_history[section_name].append({
                "revision_id": revision_id,
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback
            })
            
            await self.update_section_status(f"{section_name}_revision", "DONE")
            return revised_content
        except Exception as e:
            error_msg = f"Error revising section {section_name}: {str(e)}"
            self.logger.error(error_msg)
            await self.update_section_status(f"{section_name}_revision", "ERROR", error_msg)
            raise
    
    async def generate_section_concurrently(self, company: str, section_name: str, section_template: Dict, 
                                          state: Dict[str, Any]) -> Optional[str]:
        self.logger.info(f"Generating report section: {section_name}")
        
        research_results = state.get("research_results", {})
        event_metadata = state.get("event_metadata", {})
        analysis_results = state.get("analysis_results", {})
        top_events = state.get("top_events", [])
        other_events = state.get("other_events", [])
        
        if section_name == "executive_summary":
            return await self.generate_executive_summary(
                company, top_events, research_results, event_metadata, section_template
            )
        elif section_name == "key_events":
            detailed_events_section = "# Key Events Analysis\n\n"
            
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
    
    async def generate_sections_concurrently(self, company: str, template: ReportTemplate, 
                                           state: Dict[str, Any]) -> Dict[str, str]:
        self.logger.info(f"Generating all report sections using template: {template.name}")
        
        research_results = state.get("research_results", {})
        event_metadata = state.get("event_metadata", {})
        
        if "top_events" not in state or "other_events" not in state:
            top_events, other_events = self._select_top_events(
                research_results, event_metadata, max_detailed_events=6
            )
            state["top_events"] = top_events
            state["other_events"] = other_events
            self.logger.info(f"Selected {len(top_events)} top events and {len(other_events)} other events")
            
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
        
        tasks = []
        for section in template.sections:
            section_name = section.get("name")
            if not section_name:
                continue
                
            task = asyncio.create_task(
                generate_with_semaphore(section_name, section)
            )
            tasks.append(task)
        
        for task in asyncio.as_completed(tasks):
            try:
                section_name, section_content = await task
                if section_content:
                    sections[section_name] = section_content
                    self.report_sections[section_name] = section_content
            except Exception as e:
                self.logger.error(f"Error in section generation task: {str(e)}")
                # Continue with other sections even if one fails
                continue
        
        return sections
    
    async def save_debug_report(self, company: str, full_report: str) -> str:
        debug_dir = "debug/reports"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sanitized_company = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in company)
        debug_filename = f"{debug_dir}/{sanitized_company}_{timestamp}.md"
        
        with open(debug_filename, "w") as f:
            f.write(full_report)
            
        self.logger.info(f"Debug copy saved to {debug_filename}")
        
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
            
            self.feedback_history.append({
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback
            })
            
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
                "quality_score": 6,
                "strengths": ["Unable to generate detailed feedback due to error"],
                "weaknesses": ["Error in feedback generation process"],
                "improvements": [f"Retry feedback generation: {str(e)}"]
            }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_executive_briefing(self, company: str, full_report: str) -> str:
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
            return f"Executive briefing generation failed: {str(e)}"
    
    async def apply_iterative_improvements(self, company: str, report_sections: Dict[str, str], 
                                        feedback: Dict[str, Any]) -> Dict[str, str]:
        self.logger.info(f"Applying iterative improvements based on feedback (quality score: {feedback.get('quality_score', 0)})")
        
        quality_score = feedback.get("quality_score", 0)
        if quality_score >= self.quality_threshold:
            self.logger.info(f"Report quality already high ({quality_score}/10), skipping improvements")
            return report_sections
        
        improvements = feedback.get("improvements", [])
        if not improvements:
            self.logger.info("No specific improvements suggested")
            return report_sections
            
        improved_sections = report_sections.copy()
        
        for improvement in improvements:
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
    
    async def attempt_recovery(self, state: Dict[str, Any], error: str) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to recover from errors during report generation"""
        self.logger.warning(f"Attempting recovery from error: {error}")
        
        # Check if we've accumulated too many errors
        if self.error_count > self.critical_error_threshold:
            self.logger.error(f"Error count ({self.error_count}) exceeds threshold, generating fallback report")
            return False, state
        
        # Strategy 1: Check for failed sections and retry if possible
        recovery_attempted = False
        for section_name, status_info in self.section_statuses.items():
            if status_info["status"] == "ERROR":
                if await self.should_retry_section(section_name):
                    recovery_attempted = True
                    self.logger.info(f"Retrying failed section: {section_name}")
        
        # Strategy 2: For critical sections (executive_summary, key_events), provide fallbacks
        if not recovery_attempted:
            company = state.get("company", "Unknown Company")
            research_results = state.get("research_results", {})
            top_events = state.get("top_events", [])
            
            if "executive_summary" in self.section_statuses and self.section_statuses["executive_summary"]["status"] == "ERROR":
                # Create minimal executive summary
                fallback_summary = f"""# Executive Summary

This report presents findings related to {company}. Our analysis identified {len(research_results)} significant events, with {len(top_events)} highlighted as particularly important.

*Note: This is a simplified summary generated due to technical issues with the full summary generation.*
"""
                self.report_sections["executive_summary"] = fallback_summary
                await self.update_section_status("executive_summary", "DONE")
                recovery_attempted = True
                self.logger.info("Generated fallback executive summary")
            
            if "recommendations" in self.section_statuses and self.section_statuses["recommendations"]["status"] == "ERROR":
                # Create minimal recommendations
                fallback_recommendations = f"""# Recommendations

Based on our analysis of {company}, we recommend:
- Continue monitoring key events
- Consider deeper investigation into highlighted issues
- Follow standard due diligence procedures

*Note: These are generalized recommendations generated due to technical issues with the detailed recommendations.*
"""
                self.report_sections["recommendations"] = fallback_recommendations
                await self.update_section_status("recommendations", "DONE")
                recovery_attempted = True
                self.logger.info("Generated fallback recommendations")
        
        return recovery_attempted, state
    
    async def generate_workflow_status(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive status of the report generation process"""
        company = state.get("company", "Unknown")
        
        status = {
            "company": company,
            "overall_status": "IN_PROGRESS",
            "sections": {},
            "progress_percentage": 0,
            "errors": [],
            "next_steps": [],
            "state_timestamp": datetime.now().isoformat()
        }
        
        # Calculate progress
        total_sections = len(self.section_statuses)
        completed = sum(1 for s in self.section_statuses.values() if s["status"] == "DONE")
        
        if total_sections > 0:
            status["progress_percentage"] = int((completed / total_sections) * 100)
        
        # Overall status
        if self.error_count > self.critical_error_threshold:
            status["overall_status"] = "ERROR"
        elif completed == total_sections and total_sections > 0:
            status["overall_status"] = "DONE"
        
        # Section statuses
        for section_name, status_info in self.section_statuses.items():
            status["sections"][section_name] = {
                "status": status_info["status"],
                "started_at": status_info["started_at"],
                "completed_at": status_info["completed_at"],
                "retries": status_info["retries"]
            }
            
            if status_info["status"] == "ERROR" and status_info["error"]:
                status["errors"].append(f"{section_name}: {status_info['error']}")
        
        # Next steps
        pending_sections = [name for name, info in self.section_statuses.items() if info["status"] == "PENDING"]
        if pending_sections:
            status["next_steps"].append(f"Generate {len(pending_sections)} pending sections")
        
        error_sections = [name for name, info in self.section_statuses.items() if info["status"] == "ERROR"]
        if error_sections:
            status["next_steps"].append(f"Address {len(error_sections)} failed sections")
        
        if status["overall_status"] == "DONE":
            status["next_steps"].append("Generate meta feedback and executive briefing")
        
        return status
    
    async def save_workflow_status(self, state: Dict[str, Any], status: Dict[str, Any]) -> None:
        """Save workflow status to database"""
        company = state.get("company", "Unknown")
        
        try:
            await self.postgres_tool.run(
                command="execute_query",
                query="INSERT INTO workflow_status (company, status_data) VALUES ($1, $2) ON CONFLICT (company) DO UPDATE SET status_data = $2",
                params=[company, json.dumps(status)]
            )
            self.logger.debug(f"Saved workflow status to database for {company}")
        except Exception as e:
            self.logger.error(f"Failed to save workflow status: {str(e)}")
    
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self.run(state)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._log_start(state)
        
        if not state.get("analyst_agent_status") == "DONE" and not state.get("analyst_status") == "DONE":
            self.logger.info("Analyst not done yet. Waiting...")
            return {**state, "goto": "meta_agent", "writer_status": "WAITING"}
        
        self.logger.info("Analyst work complete. Starting report generation.")
        
        company = state.get("company", "Unknown Company")
        research_results = state.get("research_results", {})
        event_metadata = state.get("event_metadata", {})
        analysis_results = state.get("analysis_results", {})
        
        corporate_results = state.get("corporate_results", {})
        youtube_results = state.get("youtube_results", {})
        
        if not isinstance(research_results, dict):
            self.logger.error(f"Expected research_results to be a dictionary, got {type(research_results)}")
            research_results = {}
            
        if not isinstance(event_metadata, dict):
            self.logger.error(f"Expected event_metadata to be a dictionary, got {type(event_metadata)}")
            event_metadata = {}
        
        report_sections = {}
        
        try:
            template = await self.select_template(company, analysis_results)
            
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
            
            # Generate workflow status
            status = await self.generate_workflow_status(state)
            await self.save_workflow_status(state, status)
            
            # Generate report sections
            report_sections = await self.generate_sections_concurrently(company, template, state)
            
            timestamp = datetime.now().strftime("%Y-%m-%d")
            report_sections["header"] = f"# Forensic News Analysis Report: {company}\n\nReport Date: {timestamp}\n\n"
            
            section_order = [section["name"] for section in template.sections]
            section_order.insert(0, "header")
            
            full_report_parts = []
            for section_name in section_order:
                if section_name in report_sections:
                    full_report_parts.append(report_sections[section_name])
                    
            full_report = "\n".join(full_report_parts)
            
            feedback = await self.generate_meta_feedback(company, full_report)
            
            if self.enable_iterative_improvement:
                improved_sections = await self.apply_iterative_improvements(company, report_sections, feedback)
                
                improved_report_parts = []
                for section_name in section_order:
                    if section_name in improved_sections:
                        improved_report_parts.append(improved_sections[section_name])
                
                full_report = "\n".join(improved_report_parts)
                
                new_feedback = await self.generate_meta_feedback(company, full_report)
                self.logger.info(f"Report quality improved from {feedback.get('quality_score', 0)} to {new_feedback.get('quality_score', 0)}")
                feedback = new_feedback
            
            executive_briefing = await self.generate_executive_briefing(company, full_report)
            
            report_filename = await self.save_debug_report(company, full_report)
            
            state["final_report"] = full_report
            state["report_sections"] = report_sections
            state["report_feedback"] = feedback
            state["executive_briefing"] = executive_briefing
            state["report_filename"] = report_filename
            state["writer_status"] = "DONE"
            state["top_events"] = top_events
            state["other_events"] = other_events
            
            # Updated workflow status
            status = await self.generate_workflow_status(state)
            await self.save_workflow_status(state, status)
            
            self.logger.info("Report generation successfully completed.")
            
        except Exception as e:
            self.logger.error(f"Error in report generation: {str(e)}\n{traceback.format_exc()}")
            
            # Attempt recovery
            recovery_success, updated_state = await self.attempt_recovery(state, str(e))
            
            if recovery_success:
                self.logger.info("Recovery successful, continuing with report generation")
                state = updated_state
                
                # Generate a simple report from recovered sections
                section_parts = []
                timestamp = datetime.now().strftime("%Y-%m-%d")
                section_parts.append(f"# Forensic News Analysis Report: {company}\n\nReport Date: {timestamp}\n\n")
                
                for section_name, content in self.report_sections.items():
                    section_parts.append(content)
                
                full_report = "\n".join(section_parts)
                report_filename = await self.save_debug_report(company, full_report)
                
                state["final_report"] = full_report
                state["report_sections"] = self.report_sections
                state["report_filename"] = report_filename
                state["writer_status"] = "DONE"
                
                self.logger.info("Generated report from recovered sections")
            else:
                # Generate minimal fallback report
                research_count = len(research_results) if isinstance(research_results, dict) else 0
                
                fallback_report = f"""
# Forensic News Analysis Report: {company}

Report Date: {datetime.now().strftime("%Y-%m-%d")}

## Executive Summary

This report presents the findings of a forensic news analysis conducted on {company}. Due to technical issues during report generation, this is a simplified version of the analysis.

## Key Findings

The analysis identified {research_count} significant events related to {company}.

"""
                
                if isinstance(research_results, dict) and research_results:
                    fallback_report += "## Events Identified\n\n"
                    for event_name in research_results.keys():
                        article_count = len(research_results[event_name]) if isinstance(research_results[event_name], list) else 0
                        fallback_report += f"- {event_name} ({article_count} articles)\n"
                
                fallback_report += """
## Technical Issue

The full report could not be generated due to a technical error. Please refer to the logs for more information.
"""
                
                report_filename = await self.save_debug_report(company, fallback_report)
                
                state["final_report"] = fallback_report
                state["error"] = str(e)
                state["report_filename"] = report_filename
                state["writer_status"] = "ERROR"
                self.logger.info("Generated fallback report due to errors.")
        
        self._log_completion({**state, "goto": "meta_agent"})
        return {**state, "goto": "meta_agent"}