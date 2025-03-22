from typing import Dict, List, Optional, Tuple, Any, Set
import json
import asyncio
from datetime import datetime
import re
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.content_parser_tool import ContentParserTool
from tools.postgres_tool import PostgresTool


class AnalysisTask:
    def __init__(self, company: str, event_name: str, article_info: Dict, article_index: int, total_articles: int):
        self.company = company
        self.event_name = event_name
        self.article_info = article_info
        self.article_index = article_index
        self.total_articles = total_articles
        self.result = None
        self.error = None
        self.completed = False
        self.processing_time = 0


class AnalystAgent(BaseAgent):
    name = "analyst_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager()
        self.content_parser_tool = ContentParserTool(config)
        self.postgres_tool = PostgresTool(config.get("postgres", {}))
        
        self.knowledge_base = {
            "events": {},
            "entities": {},
            "relationships": {},
            "patterns": {},
            "red_flags": [],
            "evidence": {},
            "timeline": [],
            "sources": {},
            "metadata": {}
        }
        
        self.processing_stats = {
            "total_events": 0,
            "total_articles": 0,
            "processed_articles": 0,
            "articles_with_insights": 0,
            "events_with_insights": 0,
            "failed_articles": 0
        }
        
        self.max_workers = config.get("forensic_analysis", {}).get("max_workers", 5)
        self.batch_size = config.get("forensic_analysis", {}).get("batch_size", 10)
        self.concurrent_events = config.get("forensic_analysis", {}).get("concurrent_events", 2)
        self.task_timeout = config.get("forensic_analysis", {}).get("task_timeout", 300)
        
        self.processed_tasks = []
        self.currently_processing = set()
        self.task_queue = asyncio.Queue()
        self.processing_semaphore = asyncio.Semaphore(self.max_workers)
        self.evidence_strength_threshold = config.get("forensic_analysis", {}).get("evidence_strength", 3)
        
        self.result_tracker = {
            "events_analyzed": set(),
            "entities_identified": set(),
            "red_flags_found": set(),
            "timelines_created": {}
        }
    
    @retry(stop_after_attempt=3, wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_forensic_insights(self, company: str, title: str, content: str, event_name: str) -> Optional[Dict]:
        if not content or len(content.strip()) < 100:
            return None
            
        llm_provider = await get_llm_provider()
        variables = {"company": company, "title": title, "event_name": event_name, "content": content}
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="extract_forensic_insight",
            variables=variables
        )
        
        extract_prompt = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = await llm_provider.generate_text(extract_prompt, model_name=self.config.get("forensic_analysis", {}).get("model"))
        
        extracted_content = response.strip()
        
        if extracted_content == "NO_FORENSIC_CONTENT":
            self.logger.info(f"No forensic content found in article: {title}")
            return None
            
        variables = {"company": company, "extracted_content": extracted_content}
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="analyze_forensic_content",
            variables=variables
        )
        
        analysis_prompt = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        analysis_response = await llm_provider.generate_text(analysis_prompt, model_name=self.config.get("forensic_analysis", {}).get("model"))
        
        response_content = analysis_response.strip()
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content
            
        forensic_insights = json.loads(json_content)
        
        forensic_insights["raw_extract"] = extracted_content
        forensic_insights["article_title"] = title
        forensic_insights["event_category"] = event_name
        
        try:
            await self.postgres_tool.run(
                command="execute_query",
                query="INSERT INTO forensic_insights (company, event_name, article_title, insights_data) VALUES ($1, $2, $3, $4) ON CONFLICT (company, article_title) DO UPDATE SET insights_data = $4",
                params=[company, event_name, title, json.dumps(forensic_insights)]
            )
        except Exception as db_error:
            self.logger.warning(f"Failed to store forensic insights in database: {str(db_error)[:100]}...")
        
        self.logger.info(f"Successfully extracted forensic insights from: {title}")
        return forensic_insights
    
    async def process_article(self, task: AnalysisTask) -> Optional[Dict]:
        company = task.company
        event_name = task.event_name
        article_info = task.article_info
        article_index = task.article_index
        total_articles = task.total_articles
        
        article_title = article_info["title"]
        article_url = article_info["link"]
        
        start_time = datetime.now()
        
        self.logger.info(f"[{article_index+1}/{total_articles}] Processing: {article_title}")
        
        parser_result = await self.content_parser_tool.run(url=article_url)
        
        self.processing_stats["processed_articles"] += 1
        
        if not parser_result.success:
            self.logger.warning(f"Failed to fetch content for: {article_url}")
            self.processing_stats["failed_articles"] += 1
            task.error = f"Failed to fetch content: {parser_result.error}"
            task.completed = True
            task.processing_time = (datetime.now() - start_time).total_seconds()
            return None
            
        content = parser_result.data.get("content")
        metadata = parser_result.data.get("metadata")
        
        insights = await self.extract_forensic_insights(company, article_title, content, event_name)
        
        if not insights:
            self.logger.info(f"No relevant forensic insights found in: {article_title}")
            task.completed = True
            task.processing_time = (datetime.now() - start_time).total_seconds()
            return None
        
        self.processing_stats["articles_with_insights"] += 1
        
        insights["url"] = article_url
        insights["metadata"] = metadata
        
        task.result = insights
        task.completed = True
        task.processing_time = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"Successfully processed article: {article_title} in {task.processing_time:.2f}s")
        return insights
    
    async def process_articles_batch(self, tasks: List[AnalysisTask]) -> List[Dict]:
        self.logger.info(f"Processing batch of {len(tasks)} articles")
        
        results = []
        
        async def process_with_semaphore(task):
            async with self.processing_semaphore:
                self.currently_processing.add(task.article_info["title"])
                result = await self.process_article(task)
                self.currently_processing.remove(task.article_info["title"])
                return result
        
        tasks_coroutines = [process_with_semaphore(task) for task in tasks]
        batch_results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                self.logger.error(f"Task error: {str(result)[:100]}...")
                tasks[i].error = str(result)
                tasks[i].completed = True
            elif result is not None:
                results.append(result)
                
        self.processed_tasks.extend(tasks)
        return results
    
    @retry(stop_after_attempt=3, wait=wait_exponential(multiplier=1, min=2, max=10))
    async def synthesize_event_insights(self, company: str, event_name: str, insights_list: List[Dict]) -> Dict:
        if not insights_list or len(insights_list) == 0:
            return None
            
        self.logger.info(f"Synthesizing {len(insights_list)} insights for event: {event_name}")
            
        try:
            db_result = await self.postgres_tool.run(
                command="execute_query",
                query="SELECT synthesis_data FROM event_synthesis WHERE company = $1 AND event_name = $2",
                params=[company, event_name]
            )
            
            if db_result.success and db_result.data:
                self.logger.info(f"Found existing synthesis in database for event: {event_name}")
                return json.loads(db_result.data[0]["synthesis_data"])
        except Exception as db_error:
            self.logger.warning(f"Failed to check database for event synthesis: {str(db_error)[:100]}...")
        
        simplified_insights = []
        for insight in insights_list:
            simplified = {k: v for k, v in insight.items() if k not in ["raw_extract", "metadata"]}
            for key, value in simplified.items():
                if isinstance(value, str) and len(value) > 1000:
                    simplified[key] = value[:1000] + "... [truncated]"
            simplified_insights.append(simplified)
        
        llm_provider = await get_llm_provider()
        
        variables = {
            "company": company,
            "event_name": event_name,
            "num_sources": len(simplified_insights),
            "insights": json.dumps(simplified_insights, indent=2)
        }
        
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="synthesize_event_insights",
            variables=variables
        )
        
        input_message = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = await llm_provider.generate_text(input_message, model_name=self.config.get("forensic_analysis", {}).get("model"))
        
        response_content = response.strip()
        
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content
            
        synthesis = json.loads(json_content)
        
        try:
            await self.postgres_tool.run(
                command="execute_query",
                query="INSERT INTO event_synthesis (company, event_name, synthesis_data) VALUES ($1, $2, $3) ON CONFLICT (company, event_name) DO UPDATE SET synthesis_data = $3",
                params=[company, event_name, json.dumps(synthesis)]
            )
        except Exception as db_error:
            self.logger.warning(f"Failed to store event synthesis in database: {str(db_error)[:100]}...")
        
        self.logger.info(f"Successfully synthesized insights for event: {event_name}")
        
        if "key_entities" in synthesis:
            for entity in synthesis.get("key_entities", []):
                if "name" in entity and entity["name"] != "Unknown":
                    self.result_tracker["entities_identified"].add(entity["name"])
        
        if "red_flags" in synthesis:
            for flag in synthesis.get("red_flags", []):
                self.result_tracker["red_flags_found"].add(flag)
        
        return synthesis
    
    @retry(stop_after_attempt=3, wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_company_analysis(self, company: str, events_synthesis: Dict) -> Dict:
        self.logger.info(f"Generating comprehensive analysis for {company} based on {len(events_synthesis)} events")
        
        try:
            db_result = await self.postgres_tool.run(
                command="execute_query",
                query="SELECT analysis_data FROM company_analysis WHERE company = $1",
                params=[company]
            )
            
            if db_result.success and db_result.data:
                self.logger.info(f"Found existing analysis in database for company: {company}")
                return json.loads(db_result.data[0]["analysis_data"])
        except Exception as db_error:
            self.logger.warning(f"Failed to check database for company analysis: {str(db_error)[:100]}...")
        
        simplified_events = {}
        for event_name, event_data in events_synthesis.items():
            event_copy = event_data.copy()
            if "narrative" in event_copy and len(event_copy["narrative"]) > 500:
                event_copy["narrative"] = event_copy["narrative"][:500] + "... [truncated]"
            simplified_events[event_name] = event_copy
        
        corporate_data = {}
        if "corporate_results" in self.knowledge_base:
            corporate_data = self.knowledge_base["corporate_results"]
        
        llm_provider = await get_llm_provider()
        
        variables = {
            "company": company,
            "num_events": len(simplified_events),
            "events_synthesis": json.dumps(simplified_events, indent=2),
            "corporate_data": json.dumps(corporate_data, indent=2)
        }
        
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="company_analysis",
            variables=variables
        )
        
        input_message = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = await llm_provider.generate_text(input_message, model_name=self.config.get("forensic_analysis", {}).get("model"))
        
        response_content = response.strip()
        
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content
            
        analysis = json.loads(json_content)
        
        try:
            await self.postgres_tool.run(
                command="execute_query",
                query="INSERT INTO company_analysis (company, analysis_data) VALUES ($1, $2) ON CONFLICT (company) DO UPDATE SET analysis_data = $2",
                params=[company, json.dumps(analysis)]
            )
        except Exception as db_error:
            self.logger.warning(f"Failed to store company analysis in database: {str(db_error)[:100]}...")
        
        self.logger.info(f"Successfully generated comprehensive analysis for {company}")
        return analysis
    
    async def process_event(self, company: str, event_name: str, articles: List) -> Tuple[List[Dict], Dict]:
        self.logger.info(f"Processing event: {event_name} with {len(articles)} articles")
        self.result_tracker["events_analyzed"].add(event_name)
        
        tasks = []
        for i, article in enumerate(articles):
            task = AnalysisTask(
                company=company,
                event_name=event_name,
                article_info=article,
                article_index=i,
                total_articles=len(articles)
            )
            tasks.append(task)
            
        all_insights = []
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i+self.batch_size]
            batch_insights = await self.process_articles_batch(batch)
            all_insights.extend(batch_insights)
            
        if all_insights:
            self.logger.info(f"Collected {len(all_insights)} insights for event: {event_name}")
            self.knowledge_base["events"][event_name] = all_insights
            self.processing_stats["events_with_insights"] += 1
            
            event_synthesis = await self.synthesize_event_insights(company, event_name, all_insights)
            
            if event_synthesis and "timeline" in event_synthesis:
                timeline_items = []
                for timeline_item in event_synthesis.get("timeline", []):
                    if "date" in timeline_item and timeline_item["date"] != "Unknown":
                        timeline_item["event"] = event_name
                        timeline_items.append(timeline_item)
                
                if timeline_items:
                    self.knowledge_base["timeline"].extend(timeline_items)
                    self.result_tracker["timelines_created"][event_name] = len(timeline_items)
            
            return all_insights, event_synthesis
        else:
            self.logger.info(f"No insights collected for event: {event_name}")
            return [], None
    
    async def process_events_concurrently(self, company: str, research_results: Dict) -> Dict[str, Dict]:
        event_synthesis = {}
        pending_events = list(research_results.keys())
        
        while pending_events:
            batch_size = min(self.concurrent_events, len(pending_events))
            current_batch = pending_events[:batch_size]
            pending_events = pending_events[batch_size:]
            
            self.logger.info(f"Processing batch of {len(current_batch)} events concurrently")
            
            tasks = []
            for event_name in current_batch:
                event_task = asyncio.create_task(
                    self.process_event(company, event_name, research_results[event_name])
                )
                tasks.append((event_name, event_task))
            
            for event_name, task in tasks:
                try:
                    insights, synthesis = await task
                    
                    if synthesis:
                        event_synthesis[event_name] = synthesis
                        self.logger.info(f"Added synthesis for event: {event_name}")
                except Exception as e:
                    self.logger.error(f"Error processing event {event_name}: {str(e)}")
        
        return event_synthesis
    
    async def integrate_corporate_data(self, company: str, state: Dict[str, Any]) -> None:
        if "corporate_results" in state and state["corporate_results"]:
            self.logger.info(f"Integrating corporate data for {company}")
            
            corporate_results = state["corporate_results"]
            self.knowledge_base["corporate_results"] = corporate_results
            
            if "company_info" in corporate_results:
                company_info = corporate_results["company_info"]
                if "name" in company_info:
                    self.result_tracker["entities_identified"].add(company_info["name"])
                
            if "red_flags" in corporate_results and isinstance(corporate_results["red_flags"], list):
                for flag in corporate_results["red_flags"]:
                    self.knowledge_base["red_flags"].append(flag)
                    self.result_tracker["red_flags_found"].add(flag)
    
    async def integrate_youtube_data(self, company: str, state: Dict[str, Any]) -> None:
        if "youtube_results" in state and state["youtube_results"]:
            self.logger.info(f"Integrating YouTube data for {company}")
            
            youtube_results = state["youtube_results"]
            self.knowledge_base["youtube_results"] = youtube_results
            
            if "red_flags" in youtube_results and isinstance(youtube_results["red_flags"], list):
                for flag in youtube_results["red_flags"]:
                    self.knowledge_base["red_flags"].append(flag)
                    self.result_tracker["red_flags_found"].add(flag)
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._log_start(state)
        
        company = state.get("company", "")
        research_results = state.get("research_results", {})
        analysis_guidance = state.get("analysis_guidance", {})
        
        if not company:
            self.logger.error("Company name missing!")
            return {**state, "goto": "meta_agent", "analyst_status": "ERROR", "error": "Company name missing"}
        
        if not research_results:
            self.logger.error("No research results to analyze!")
            return {**state, "goto": "meta_agent", "analyst_status": "ERROR", "error": "No research results"}
        
        self.logger.info(f"Analyzing {len(research_results)} events for company: {company}")
        
        self.processing_stats["total_events"] = len(research_results)
        self.processing_stats["total_articles"] = sum(len(articles) for articles in research_results.values())
        
        analysis_results = {
            "forensic_insights": {},    
            "event_synthesis": {},      
            "company_analysis": {},     
            "red_flags": [],            
            "evidence_map": {},         
            "entity_network": {},       
            "timeline": [],             
        }
        
        await self.integrate_corporate_data(company, state)
        await self.integrate_youtube_data(company, state)
        
        event_synthesis = await self.process_events_concurrently(company, research_results)
        
        analysis_results["event_synthesis"] = event_synthesis
        
        timeline_items = self.knowledge_base["timeline"]
        sorted_timeline = sorted(
            timeline_items, 
            key=lambda x: datetime.fromisoformat(x["date"]) if re.match(r'\d{4}-\d{2}-\d{2}', x.get("date", "")) else datetime.now(),
            reverse=True
        )
        analysis_results["timeline"] = sorted_timeline
        
        analysis_results["red_flags"] = list(self.result_tracker["red_flags_found"])
        
        if event_synthesis:
            company_analysis = await self.generate_company_analysis(
                company, 
                event_synthesis
            )
            analysis_results["company_analysis"] = company_analysis
            
            final_report = company_analysis.get("report_markdown", f"# Forensic Analysis of {company}\n\nNo significant findings.")
        else:
            self.logger.info(f"No significant forensic insights found for {company}")
            
            stats = self.processing_stats
            completion_percentage = (
                round((stats['processed_articles'] / stats['total_articles']) * 100, 1)
                if stats['total_articles'] > 0 else 0
            )
            
            final_report = f"""
            # Forensic Analysis of {company}
            
            ## Executive Summary
            
            After analyzing {stats['total_articles']} articles across {stats['total_events']} potential events, 
            no significant forensic concerns were identified for {company}. The available information does not 
            indicate material issues related to financial integrity, regulatory compliance, or corporate governance.
            
            ## Analysis Process
            
            - Total events examined: {stats['total_events']}
            - Total articles processed: {stats['processed_articles']}
            - Articles with potential forensic content: {stats['articles_with_insights']}
            - Events with synthesized insights: {stats['events_with_insights']}
            
            ## Conclusion
            
            Based on the available information, there are no significant red flags or forensic concerns 
            that would warrant further investigation at this time.
            """
        
        entity_network = {}
        for entity_name in self.result_tracker["entities_identified"]:
            entity_info = {"name": entity_name, "connections": []}
            
            for event_name, synthesis in event_synthesis.items():
                if "key_entities" in synthesis:
                    for entity in synthesis["key_entities"]:
                        if entity.get("name") == entity_name:
                            if "role" in entity:
                                entity_info["role"] = entity.get("role")
                            
                            connection = {"event": event_name}
                            entity_info["connections"].append(connection)
            
            if entity_info["connections"]:
                entity_network[entity_name] = entity_info
                
        analysis_results["entity_network"] = entity_network
        
        state["analysis_results"] = analysis_results
        state["final_report"] = final_report
        state["analyst_status"] = "DONE"
        state["analysis_stats"] = self.processing_stats
        
        try:
            await self.postgres_tool.run(
                command="execute_query",
                query="INSERT INTO analysis_results (company, report_date, analysis_data, red_flags) VALUES ($1, $2, $3, $4)",
                params=[company, datetime.now().isoformat(), json.dumps(analysis_results), json.dumps(analysis_results["red_flags"])]
            )
        except Exception as db_error:
            self.logger.warning(f"Failed to save analysis results to database: {str(db_error)[:100]}...")
        
        self.logger.info(f"Analysis complete. Processed {self.processing_stats['processed_articles']}/{self.processing_stats['total_articles']} articles.")
        self.logger.info(f"Found forensic insights in {self.processing_stats['articles_with_insights']} articles.")
        self.logger.info(f"Failed to process {self.processing_stats['failed_articles']} articles.")
        
        self._log_completion({**state, "goto": "meta_agent"})
        return {**state, "goto": "meta_agent"}