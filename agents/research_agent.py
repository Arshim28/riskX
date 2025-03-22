from typing import Dict, List, Any, Tuple, Set, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import asyncio
import re
from datetime import datetime

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider, LLMProvider
from utils.prompt_manager import PromptManager
from utils.logging import get_logger
from tools.search_tool import SearchTool, SearchResult


class ResearchAgent(BaseAgent):
    name = "research_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = PromptManager()  
        self.search_tool = SearchTool(config.get("research", {}))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_queries(self, company: str, industry: str, research_plan: Dict, query_history: List[Dict[str, List]]) -> Dict[str, List[str]]:
        self.logger.info(f"Generating queries based on research plan: {research_plan.get('objective', 'No objective')}")
        
        llm_provider = await get_llm_provider()
        
        variables = {
            "company": company,
            "industry": industry,
            "research_plan": json.dumps(research_plan, indent=4),
            "query_history": json.dumps(query_history, indent=4)
        }
        
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="query_generation",
            variables=variables
        )
        
        input_message = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = await llm_provider.generate_text(
            prompt=input_message,
            model_name=self.config.get("models", {}).get("planning")
        )
        
        response_content = response.strip()
        
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content

        query_categories = json.loads(json_content)
        
        total_queries = sum(len(v) for v in query_categories.values())
        self.logger.info(f"Generated {total_queries} queries across {len(query_categories)} categories")
        
        return query_categories
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def group_results(self, company: str, articles: List[SearchResult], industry: str = None) -> Dict[str, List[Dict]]:
        if not articles:
            self.logger.info("No articles to cluster")
            return {}
        
        quarterly_report_articles = [a for a in articles if a.is_quarterly_report]
        other_articles = [a for a in articles if not a.is_quarterly_report]
        
        self.logger.info(f"Identified {len(quarterly_report_articles)} quarterly report articles")
        self.logger.info(f"Processing {len(other_articles)} non-quarterly report articles")
        
        regular_events = {}
        if other_articles:
            try:
                llm_provider = await get_llm_provider()
                
                simplified_articles = []
                for i, article in enumerate(other_articles):
                    simplified_articles.append({
                        "index": i,
                        "title": article.title,
                        "snippet": article.snippet or "",
                        "date": article.date or "Unknown date",
                        "source": article.source or "Unknown source",
                        "category": article.category or "general"
                    })
                
                variables = {
                    "company": company,
                    "industry": industry if industry else "Unknown",
                    "simplified_articles": json.dumps(simplified_articles)
                }
                
                system_prompt, human_prompt = self.prompt_manager.get_prompt(
                    agent_name=self.name,
                    operation="article_clustering",
                    variables=variables
                )
                
                input_message = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                response = await llm_provider.generate_text(
                    prompt=input_message,
                    model_name=self.config.get("models", {}).get("planning")
                )
                
                response_content = response.strip()
                
                if "```json" in response_content:
                    json_content = response_content.split("```json")[1].split("```")[0].strip()
                elif "```" in response_content:
                    json_content = response_content.split("```")[1].strip()
                else:
                    json_content = response_content
                    
                clustered_indices = json.loads(json_content)
                
                for event_name, indices in clustered_indices.items():
                    valid_indices = []
                    for idx in indices:
                        if isinstance(idx, str) and idx.isdigit():
                            idx = int(idx)
                        if isinstance(idx, int) and 0 <= idx < len(other_articles):
                            valid_indices.append(idx)
                    
                    if valid_indices:
                        regular_events[event_name] = [other_articles[i].model_dump() for i in valid_indices]
                
                self.logger.info(f"Grouped non-quarterly articles into {len(regular_events)} events")
                
            except Exception as e:
                self.logger.error(f"Error clustering non-quarterly articles: {str(e)}")
                
                for i, article in enumerate(other_articles):
                    event_name = f"News: {article.title[:50]}..."
                    regular_events[event_name] = [article.model_dump()]
        
        if quarterly_report_articles:
            dates = [article.date or "" for article in quarterly_report_articles]
            valid_dates = [d for d in dates if d and d.lower() != "unknown date"]
            
            date_str = ""
            if valid_dates:
                try:
                    parsed_dates = []
                    for date_text in valid_dates:
                        try:
                            for fmt in ["%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y"]:
                                try:
                                    parsed_date = datetime.strptime(date_text, fmt)
                                    parsed_dates.append(parsed_date)
                                    break
                                except:
                                    continue
                        except:
                            pass
                    
                    if parsed_dates:
                        most_recent = max(parsed_dates)
                        date_str = f" ({most_recent.strftime('%b %Y')})"
                except:
                    date_str = f" ({valid_dates[0]})"
            
            quarterly_event_name = f"Financial Reporting: Quarterly/Annual Results{date_str} - Low"
            regular_events[quarterly_event_name] = [article.model_dump() for article in quarterly_report_articles]
            self.logger.info(f"Created a consolidated event for {len(quarterly_report_articles)} quarterly report articles")
        
        final_events = {}
        importance_scores = {}
        
        for event_name, event_articles in regular_events.items():
            importance = await self._calculate_event_importance(event_name, event_articles)
            importance_scores[event_name] = importance
            
            event_data = {
                "articles": event_articles,
                "importance_score": importance,
                "article_count": len(event_articles)
            }
            final_events[event_name] = event_data
        
        self.logger.info(f"Assigned importance scores to {len(importance_scores)} events")
        
        return final_events
    
    async def _calculate_event_importance(self, event_name: str, articles: List[Dict]) -> int:
        score = 50
        
        event_name_lower = event_name.lower()
        
        # Fix for issue #2: Make sure quarterly reports get a lower score
        if any(term in event_name_lower for term in ['quarterly report', 'financial results', 'earnings report', 'agm', 'board meeting']):
            score -= 60  # Reduce by more to ensure it's below 50
        
        if any(term in event_name_lower for term in ['fraud', 'scam', 'lawsuit', 'investigation', 'scandal', 'fine', 'penalty', 'cbI raid', 'ed probe', 'bribery', 'allegation']):
            score += 30
        
        if 'criminal' in event_name_lower or 'money laundering' in event_name_lower:
            score += 40
        
        if 'class action' in event_name_lower or 'public interest litigation' in event_name_lower:
            score += 25
        
        if any(term in event_name_lower for term in ['sebi', 'rbi', 'cbi', 'ed', 'income tax', 'competition commission']):
            score += 20
        
        article_count = len(articles)
        score += min(article_count * 2.5, 25)  
        
        if '- High' in event_name:
            score += 25
        elif '- Medium' in event_name:
            score += 10
        
        reputable_sources = ['economic times', 'business standard', 'mint', 'hindu business line', 'moneycontrol', 'ndtv', 'the hindu', 'times of india']
        for article in articles:
            source = article.get('source', '').lower()
            if any(rep_source in source for rep_source in reputable_sources):
                score += 2
        
        return score
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research agent workflow"""
        try:
            # Initialize research_results if not present (fix for issue #3)
            if "research_results" not in state:
                state["research_results"] = {}
                
            # Log the start of processing
            self.logger.info(f"Starting research agent with state: {state.get('company')}")
            
            company = state.get("company", "")
            industry = state.get("industry", "Unknown")
            research_plans = state.get("research_plan", [])
            search_history = state.get("search_history", [])
            current_results = state.get("research_results", {})
            search_type = state.get("search_type", "google_search")
            return_type = state.get("return_type", "clustered")
            
            if not company:
                self.logger.error("Missing company name")
                return {**state, "goto": "meta_agent", "error": "Missing company name"}
                
            if not research_plans:
                self.logger.error("No research plan provided")
                return {**state, "goto": "meta_agent", "error": "No research plan provided"}
            
            current_plan = research_plans[-1]
            self.logger.info(f"Processing research plan: {current_plan.get('objective', 'No objective specified')}")
            
            target_event = current_plan.get("event_name", None)
            if target_event:
                self.logger.info(f"This is a targeted research plan for event: {target_event}")
            
            all_articles = []
            executed_queries = []
            
            query_categories = await self.generate_queries(company, industry, current_plan, search_history)
            
            for category, queries in query_categories.items():
                self.logger.info(f"Processing category: {category}")
                for query in queries:
                    if query in [q for sublist in search_history for q in sublist]:
                        self.logger.info(f"Skipping duplicate query: {query}")
                        continue
                    
                    self.logger.info(f"Executing search query: {query}")
                    executed_queries.append(query)
                    
                    search_params = {
                        "tbm": "nws" if search_type == "google_news" else None
                    }
                    
                    try:
                        result = await self.search_tool.run(query=query, **search_params)
                        
                        if result.success and result.data:
                            for article in result.data:
                                article_dict = article.model_dump()
                                article_dict["category"] = category
                                all_articles.append(article_dict)
                        
                        await asyncio.sleep(1)  
                    except Exception as e:
                        self.logger.error(f"Error executing query '{query}': {str(e)}")
            
            search_history.append(executed_queries)
            state["search_history"] = search_history
            
            self.logger.info(f"Collected {len(all_articles)} total articles across all categories")
            
            if not all_articles:
                self.logger.info("No articles found with targeted queries. Trying fallback query.")
                try:
                    fallback_query = f'"{company}" negative news'
                    if fallback_query not in [q for sublist in search_history for q in sublist]:
                        search_params = {
                            "tbm": "nws" if search_type == "google_news" else None
                        }
                        
                        search_history[-1].append(fallback_query)
                        result = await self.search_tool.run(query=fallback_query, **search_params)
                        
                        if result.success and result.data:
                            for article in result.data:
                                article_dict = article.model_dump()
                                article_dict["category"] = "general"
                                all_articles.append(article_dict)
                            
                        self.logger.info(f"Fallback query returned {len(all_articles)} articles")
                except Exception as e:
                    self.logger.error(f"Error with fallback query: {str(e)}")
            
            unique_articles = []
            seen_urls = set()
            for article in all_articles:
                if article["link"] not in seen_urls:
                    seen_urls.add(article["link"])
                    unique_articles.append(article)
            
            self.logger.info(f"Deduplicated to {len(unique_articles)} unique articles")
            
            if target_event and return_type == "clustered":
                if target_event in current_results:
                    existing_articles = current_results[target_event]
                    existing_urls = {article["link"] for article in existing_articles}
                    
                    new_articles = [a for a in unique_articles if a["link"] not in existing_urls]
                    current_results[target_event].extend(new_articles)
                    
                    self.logger.info(f"Added {len(new_articles)} new articles to event: {target_event}")
                else:
                    current_results[target_event] = unique_articles
                    self.logger.info(f"Created new event '{target_event}' with {len(unique_articles)} articles")
                
                state["additional_research_completed"] = True
            else:
                if return_type == "clustered" and unique_articles:
                    search_results = [SearchResult(**article) for article in unique_articles]
                    grouped_results = await self.group_results(company, search_results, industry)
                    
                    final_results = {}
                    event_metadata = {}
                    
                    for event_name, event_data in grouped_results.items():
                        final_results[event_name] = event_data["articles"]
                        event_metadata[event_name] = {
                            "importance_score": event_data["importance_score"],
                            "article_count": event_data["article_count"],
                            "is_quarterly_report": any(a.get("is_quarterly_report", False) for a in event_data["articles"])
                        }
                    
                    state["research_results"] = final_results
                    state["event_metadata"] = event_metadata
                    self.logger.info(f"Grouped articles into {len(final_results)} distinct events")
                elif return_type != "clustered":
                    state["research_results"] = unique_articles
                    self.logger.info(f"Returning {len(unique_articles)} unclustered articles")
            
            self.logger.info(f"Research completed for {company}")
            return {**state, "goto": "meta_agent"}
            
        except Exception as e:
            self.logger.error(f"Unexpected error in research agent: {str(e)}")
            # Make sure we always return research_results, even on error (fix for issue #3)
            if "research_results" not in state:
                state["research_results"] = {}
            return {
                **state, 
                "goto": "meta_agent",
                "error": f"Research agent error: {str(e)}"
            }