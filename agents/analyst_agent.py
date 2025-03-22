from typing import Dict, List, Optional, Tuple, Any
import json
import asyncio
from datetime import datetime
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager, PromptManager
from utils.logging import get_logger
from tools.content_parser_tool import ContentParserTool


class AnalystAgent(BaseAgent):
    name = "analyst_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager(self.name)
        self.content_parser_tool = ContentParserTool(config)
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_forensic_insights(self, company: str, title: str, content: str, event_name: str) -> Optional[Dict]:
        if not content or len(content.strip()) < 100:
            return None
            
        try:
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
            
            self.logger.info(f"Successfully extracted forensic insights from: {title}")
            return forensic_insights
            
        except Exception as e:
            self.logger.error(f"Error during content analysis: {str(e)[:100]}...")
            return None
    
    async def process_article(self, company: str, event_name: str, article_info: Dict, article_index: int, total_articles: int) -> Optional[Dict]:
        article_title = article_info["title"]
        article_url = article_info["link"]
        
        try:
            self.logger.info(f"[{article_index+1}/{total_articles}] Processing: {article_title}")
            
            # Use the ContentParserTool instead of direct fetch_article_content
            parser_result = await self.content_parser_tool.run(url=article_url)
            
            self.processing_stats["processed_articles"] += 1
            
            if not parser_result.success:
                self.logger.warning(f"Failed to fetch content for: {article_url}")
                self.processing_stats["failed_articles"] += 1
                return None
                
            content = parser_result.data.get("content")
            metadata = parser_result.data.get("metadata")
            
            insights = await self.extract_forensic_insights(company, article_title, content, event_name)
            
            if not insights:
                self.logger.info(f"No relevant forensic insights found in: {article_title}")
                return None
            
            self.processing_stats["articles_with_insights"] += 1
            
            insights["url"] = article_url
            insights["metadata"] = metadata
            
            self.logger.info(f"Successfully processed article: {article_title}")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error processing article {article_title}: {str(e)[:100]}...")
            self.processing_stats["failed_articles"] += 1
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def synthesize_event_insights(self, company: str, event_name: str, insights_list: List[Dict]) -> Dict:
        if not insights_list or len(insights_list) == 0:
            return None
            
        self.logger.info(f"Synthesizing {len(insights_list)} insights for event: {event_name}")
            
        try:
            llm_provider = await get_llm_provider()
            
            simplified_insights = []
            for insight in insights_list:
                simplified = {k: v for k, v in insight.items() if k not in ["raw_extract", "metadata"]}
                for key, value in simplified.items():
                    if isinstance(value, str) and len(value) > 1000:
                        simplified[key] = value[:1000] + "... [truncated]"
                simplified_insights.append(simplified)
            
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
            
            self.logger.info(f"Successfully synthesized insights for event: {event_name}")
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Error during synthesis for event {event_name}: {str(e)[:100]}...")
            
            return {
                "cross_validation": "Could not synthesize due to technical error",
                "timeline": [{"date": "Unknown", "description": "Event occurred"}],
                "key_entities": [{"name": company, "role": "Subject company"}],
                "evidence_assessment": "Error during synthesis",
                "severity_assessment": "Unknown",
                "credibility_score": 5,
                "red_flags": ["Technical error prevented complete analysis"],
                "narrative": f"Analysis of {event_name} involving {company} could not be completed due to technical error."
            }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_company_analysis(self, company: str, events_synthesis: Dict) -> Dict:
        self.logger.info(f"Generating comprehensive analysis for {company} based on {len(events_synthesis)} events")
        
        try:
            llm_provider = await get_llm_provider()
            
            simplified_events = {}
            for event_name, event_data in events_synthesis.items():
                event_copy = event_data.copy()
                if "narrative" in event_copy and len(event_copy["narrative"]) > 500:
                    event_copy["narrative"] = event_copy["narrative"][:500] + "... [truncated]"
                simplified_events[event_name] = event_copy
            
            variables = {
                "company": company,
                "num_events": len(simplified_events),
                "events_synthesis": json.dumps(simplified_events, indent=2)
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
            
            self.logger.info(f"Successfully generated comprehensive analysis for {company}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive analysis: {str(e)[:100]}...")
            
            return {
                "executive_summary": f"Analysis of {company} could not be completed due to technical error.",
                "risk_assessment": {
                    "financial_integrity_risk": "Unknown",
                    "legal_regulatory_risk": "Unknown",
                    "reputational_risk": "Unknown",
                    "operational_risk": "Unknown"
                },
                "key_patterns": ["Technical error prevented pattern analysis"],
                "critical_entities": [{"name": company, "role": "Subject company"}],
                "red_flags": ["Analysis incomplete due to technical error"],
                "timeline": [{"date": "Unknown", "description": "Analysis attempted"}],
                "forensic_assessment": "Analysis could not be completed",
                "report_markdown": f"# Forensic Analysis of {company}\n\nAnalysis could not be completed due to technical error."
            }
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._log_start(state)
        
        company = state.get("company", "")
        research_results = state.get("research_results", {})
        analysis_guidance = state.get("analysis_guidance", {})
        
        if not company:
            self.logger.error("Company name missing!")
            return {**state, "goto": "writer_agent", "analyst_status": "ERROR", "error": "Company name missing"}
        
        if not research_results:
            self.logger.error("No research results to analyze!")
            return {**state, "goto": "writer_agent", "analyst_status": "ERROR", "error": "No research results"}
        
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
        
        for event_name, articles in research_results.items():
            self.logger.info(f"Processing {len(articles)} articles for event: {event_name}")
            
            event_insights = []
            article_tasks = []
            
            for i, article in enumerate(articles):
                task = self.process_article(company, event_name, article, i, len(articles))
                article_tasks.append(task)
            
            results = await asyncio.gather(*article_tasks)
            
            for result in results:
                if result:
                    event_insights.append(result)
            
            if event_insights:
                self.logger.info(f"Collected {len(event_insights)} insights for event: {event_name}")
                analysis_results["forensic_insights"][event_name] = event_insights
                self.knowledge_base["events"][event_name] = event_insights
                self.processing_stats["events_with_insights"] += 1
                
                event_synthesis = await self.synthesize_event_insights(company, event_name, event_insights)
                if event_synthesis:
                    analysis_results["event_synthesis"][event_name] = event_synthesis
                    
                    if "timeline" in event_synthesis:
                        timeline_items = []
                        for timeline_item in event_synthesis.get("timeline", []):
                            if "date" in timeline_item and timeline_item["date"] != "Unknown":
                                timeline_item["event"] = event_name
                                timeline_items.append(timeline_item)
                        if timeline_items:
                            self.knowledge_base["timeline"].extend(timeline_items)
                            analysis_results["timeline"].extend(timeline_items)
                    
                    if "red_flags" in event_synthesis:
                        for flag in event_synthesis.get("red_flags", []):
                            if flag not in self.knowledge_base["red_flags"]:
                                self.knowledge_base["red_flags"].append(flag)
                            if flag not in analysis_results["red_flags"]:
                                analysis_results["red_flags"].append(flag)
                    
                    if "key_entities" in event_synthesis:
                        entities = {}
                        for entity_info in event_synthesis.get("key_entities", []):
                            if "name" in entity_info and entity_info["name"] != "Unknown":
                                entity_name = entity_info["name"]
                                entities[entity_name] = entity_info
                        if entities:
                            for entity, info in entities.items():
                                if entity not in self.knowledge_base["entities"]:
                                    self.knowledge_base["entities"][entity] = info
                                else:
                                    self.knowledge_base["entities"][entity].update(info)
            else:
                self.logger.info(f"No insights collected for event: {event_name}")
        
        analysis_results["timeline"] = sorted(
            analysis_results["timeline"], 
            key=lambda x: datetime.fromisoformat(x["date"]) if re.match(r'\d{4}-\d{2}-\d{2}', x.get("date", "")) else datetime.now(),
            reverse=True
        )
        
        if analysis_results["event_synthesis"]:
            company_analysis = await self.generate_company_analysis(
                company, 
                analysis_results["event_synthesis"]
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
        
        state["analysis_results"] = analysis_results
        state["final_report"] = final_report
        state["analyst_status"] = "DONE"
        state["analysis_stats"] = self.processing_stats
        
        self.logger.info(f"Analysis complete. Processed {self.processing_stats['processed_articles']}/{self.processing_stats['total_articles']} articles.")
        self.logger.info(f"Found forensic insights in {self.processing_stats['articles_with_insights']} articles.")
        self.logger.info(f"Failed to process {self.processing_stats['failed_articles']} articles.")
        
        self._log_completion({**state, "goto": "writer_agent"})
        return {**state, "goto": "writer_agent"}