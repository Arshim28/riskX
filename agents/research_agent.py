from typing import Dict, List, Any, Tuple, Set, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import asyncio
import re
from datetime import datetime

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.search_tool import SearchTool, SearchResult


class ResearchAgent(BaseAgent):
    name = "research_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager()  
        self.search_tool = SearchTool(config.get("research", {}))
        
        # Track executed queries to prevent duplicates
        self.executed_queries = set()
        
        # Importance score thresholds
        self.high_importance_threshold = 75
        self.medium_importance_threshold = 50
        self.low_importance_threshold = 25
        
        # Result validation settings
        self.min_snippet_length = 20
        self.min_title_length = 5
        self.max_duplicate_similarity = 0.85
        self.min_relevance_score = 0.3
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_queries(self, company: str, industry: str, research_plan: Dict, query_history: List[Dict[str, List]]) -> Dict[str, List[str]]:
        """Generate diverse and effective search queries based on research plan"""
        self.logger.info(f"Generating queries based on research plan: {research_plan.get('objective', 'No objective')}")
        
        # Flatten previous queries for deduplication
        previous_queries = set()
        for query_set in query_history:
            for queries in query_set.values():
                previous_queries.update(queries)
        
        # Clean company name for search
        company_variants = self._generate_company_variants(company)
        
        llm_provider = await get_llm_provider()
        
        variables = {
            "company": company,
            "company_variants": json.dumps(company_variants),
            "industry": industry,
            "research_plan": json.dumps(research_plan, indent=4),
            "query_history": json.dumps(query_history, indent=4),
            "previous_queries": json.dumps(list(previous_queries))
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
        
        # Validate and deduplicate queries
        validated_categories = {}
        for category, queries in query_categories.items():
            valid_queries = []
            for query in queries:
                # Skip already executed queries
                if query in self.executed_queries or query in previous_queries:
                    continue
                    
                # Validate query
                if self._validate_query(query, company):
                    valid_queries.append(query)
                    
            if valid_queries:
                validated_categories[category] = valid_queries
        
        total_queries = sum(len(v) for v in validated_categories.values())
        self.logger.info(f"Generated {total_queries} queries across {len(validated_categories)} categories")
        
        return validated_categories
    
    def _generate_company_variants(self, company: str) -> List[str]:
        """Generate variations of company name for more effective searching"""
        variants = [company]
        
        # Add without common suffixes if present
        suffixes = [" Inc", " Corp", " Corporation", " Ltd", " LLC", " Limited", " Group"]
        for suffix in suffixes:
            if company.endswith(suffix):
                variants.append(company[:-len(suffix)].strip())
                break
                
        # Add ticker symbol if present in name (e.g., "Apple (AAPL)")
        ticker_match = re.search(r'\(([A-Z]{1,5})\)', company)
        if ticker_match:
            variants.append(ticker_match.group(1))
            
        # Add base name without parenthetical elements
        base_name = re.sub(r'\([^)]*\)', '', company).strip()
        if base_name != company:
            variants.append(base_name)
            
        return variants
    
    def _validate_query(self, query: str, company: str) -> bool:
        """Validate that a query is well-formed and likely to be effective"""
        if len(query.strip()) < 10:
            return False
            
        # Ensure query contains the company name or variants
        variants = self._generate_company_variants(company)
        has_company_ref = any(variant.lower() in query.lower() for variant in variants)
        
        if not has_company_ref:
            return False
            
        # Check for overuse of special characters
        special_char_count = sum(1 for c in query if not c.isalnum() and not c.isspace())
        if special_char_count > len(query) * 0.3:
            return False
            
        return True
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def group_results(self, company: str, articles: List[SearchResult], industry: str = None) -> Dict[str, List[Dict]]:
        """Group search results into meaningful clusters with improved accuracy"""
        if not articles:
            self.logger.info("No articles to cluster")
            return {}
        
        # Filter out invalid or low-quality articles first
        valid_articles = self._validate_articles(articles)
        if len(valid_articles) < len(articles):
            self.logger.info(f"Filtered out {len(articles) - len(valid_articles)} low-quality articles")
        
        # Further deduplicate articles
        deduplicated_articles = self._deduplicate_articles(valid_articles)
        if len(deduplicated_articles) < len(valid_articles):
            self.logger.info(f"Deduplicated {len(valid_articles) - len(deduplicated_articles)} similar articles")
        
        quarterly_report_articles = [a for a in deduplicated_articles if a.is_quarterly_report]
        other_articles = [a for a in deduplicated_articles if not a.is_quarterly_report]
        
        self.logger.info(f"Identified {len(quarterly_report_articles)} quarterly report articles")
        self.logger.info(f"Processing {len(other_articles)} non-quarterly report articles")
        
        regular_events = {}
        if other_articles:
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
                    event_articles = [other_articles[i].model_dump() for i in valid_indices]
                    # Add importance level to event name if not present
                    if not any(level in event_name for level in ["High", "Medium", "Low"]):
                        importance = await self._calculate_event_importance(event_name, event_articles)
                        level = self._get_importance_level(importance)
                        if level and " - " not in event_name:
                            event_name = f"{event_name} - {level}"
                    
                    regular_events[event_name] = event_articles
            
            self.logger.info(f"Grouped non-quarterly articles into {len(regular_events)} events")
        
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
    
    def _validate_articles(self, articles: List[SearchResult]) -> List[SearchResult]:
        """Validate articles for quality and relevance"""
        valid_articles = []
        
        for article in articles:
            # Skip articles with very short titles or snippets
            if len(article.title) < self.min_title_length:
                continue
                
            if article.snippet and len(article.snippet) < self.min_snippet_length:
                continue
                
            # Calculate rough relevance score (more sophisticated in production)
            relevance = self._calculate_article_relevance(article)
            if relevance < self.min_relevance_score:
                continue
                
            valid_articles.append(article)
            
        return valid_articles
    
    def _calculate_article_relevance(self, article: SearchResult) -> float:
        """Calculate a relevance score for an article (0.0 to 1.0)"""
        # This is a simplified relevance calculation
        # In production, this would use more sophisticated NLP
        
        title_score = 0.6
        snippet_score = 0.4
        
        # Sources we consider more reliable
        reliable_sources = [
            'economic times', 'business standard', 'mint', 'hindu business line',
            'moneycontrol', 'bloomberg', 'reuters', 'wsj', 'financial times',
            'cnbc', 'nyt', 'forbes', 'harvard business', 'mckinsey'
        ]
        
        # Source quality bonus
        source_bonus = 0.0
        if article.source:
            source_lower = article.source.lower()
            if any(rs in source_lower for rs in reliable_sources):
                source_bonus = 0.2
                
        return min(1.0, title_score + snippet_score + source_bonus)
    
    def _deduplicate_articles(self, articles: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate or near-duplicate articles"""
        if not articles:
            return []
            
        # Keep track of unique articles by content similarity
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Create a normalized version of the title for comparison
            normalized_title = ''.join(c.lower() for c in article.title if c.isalnum())
            
            # Skip if we've seen a very similar title
            title_already_seen = False
            for seen_title in seen_titles:
                similarity = self._text_similarity(normalized_title, seen_title)
                if similarity > self.max_duplicate_similarity:
                    title_already_seen = True
                    break
            
            if not title_already_seen:
                unique_articles.append(article)
                seen_titles.add(normalized_title)
                
        return unique_articles
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings (0.0 to 1.0)"""
        # This is a simplified implementation using character-level comparison
        # Production would use more sophisticated methods (NLP, embeddings)
        
        if not text1 or not text2:
            return 0.0
            
        # Character-level Jaccard similarity
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _calculate_event_importance(self, event_name: str, articles: List[Dict]) -> int:
        """Calculate importance score for an event based on various signals"""
        score = 50
        
        event_name_lower = event_name.lower()
        
        # Lower score for routine events
        if any(term in event_name_lower for term in ['quarterly report', 'financial results', 'earnings report', 'agm', 'board meeting']):
            score -= 60
        
        # Increase score for potential red flags
        if any(term in event_name_lower for term in ['fraud', 'scam', 'lawsuit', 'investigation', 'scandal', 'fine', 'penalty', 'cbi raid', 'ed probe', 'bribery', 'allegation']):
            score += 30
        
        # Higher score for serious criminal matters
        if 'criminal' in event_name_lower or 'money laundering' in event_name_lower:
            score += 40
        
        # Increase score for collective actions
        if 'class action' in event_name_lower or 'public interest litigation' in event_name_lower:
            score += 25
        
        # Regulatory involvement signals importance
        if any(term in event_name_lower for term in ['sebi', 'rbi', 'cbi', 'ed', 'income tax', 'competition commission']):
            score += 20
        
        # More articles suggest more significant news
        article_count = len(articles)
        score += min(article_count * 2.5, 25)  
        
        # Consider existing importance level in event name
        if '- High' in event_name:
            score += 25
        elif '- Medium' in event_name:
            score += 10
        
        # Bonus for reputable sources
        reputable_sources = ['economic times', 'business standard', 'mint', 'hindu business line', 'moneycontrol', 'ndtv', 'the hindu', 'times of india', 'bloomberg', 'reuters', 'wsj']
        for article in articles:
            source = article.get('source', '').lower()
            if any(rep_source in source for rep_source in reputable_sources):
                score += 2
        
        # Article recency bonus
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        for article in articles:
            date_str = article.get('date', '')
            if date_str:
                try:
                    # Try different date formats
                    for fmt in ["%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y"]:
                        try:
                            date = datetime.strptime(date_str, fmt)
                            # More recent articles increase importance
                            if date.year == current_year:
                                score += 5
                                if date.month == current_month:
                                    score += 5
                            break
                        except:
                            continue
                except:
                    pass
        
        return score
    
    def _get_importance_level(self, score: int) -> str:
        """Convert numeric importance score to categorical level"""
        if score >= self.high_importance_threshold:
            return "High"
        elif score >= self.medium_importance_threshold:
            return "Medium"
        else:
            return "Low"
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self.run(state)
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if "research_results" not in state:
                state["research_results"] = {}
                
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
            
            # Get previous execution state, if any
            if "execution_state" in state:
                self.executed_queries = set(state["execution_state"].get("executed_queries", []))
            
            query_categories = await self.generate_queries(company, industry, current_plan, search_history)
            
            for category, queries in query_categories.items():
                self.logger.info(f"Processing category: {category}")
                for query in queries:
                    if query in [q for sublist in search_history for q in sublist]:
                        self.logger.info(f"Skipping duplicate query: {query}")
                        continue
                    
                    self.logger.info(f"Executing search query: {query}")
                    executed_queries.append(query)
                    self.executed_queries.add(query)
                    
                    search_params = {
                        "tbm": "nws" if search_type == "google_news" else None
                    }
                    
                    result = await self.search_tool.run(query=query, **search_params)
                    
                    if result.success and result.data:
                        for article in result.data:
                            article_dict = article.model_dump()
                            article_dict["category"] = category
                            all_articles.append(article_dict)
                    
                    await asyncio.sleep(1)  
            
            search_history.append(executed_queries)
            state["search_history"] = search_history
            
            self.logger.info(f"Collected {len(all_articles)} total articles across all categories")
            
            if not all_articles:
                self.logger.info("No articles found with targeted queries. Trying fallback query.")
                fallback_query = f'"{company}" negative news'
                if fallback_query not in [q for sublist in search_history for q in sublist]:
                    search_params = {
                        "tbm": "nws" if search_type == "google_news" else None
                    }
                    
                    search_history[-1].append(fallback_query)
                    self.executed_queries.add(fallback_query)
                    result = await self.search_tool.run(query=fallback_query, **search_params)
                    
                    if result.success and result.data:
                        for article in result.data:
                            article_dict = article.model_dump()
                            article_dict["category"] = "general"
                            all_articles.append(article_dict)
                        
                    self.logger.info(f"Fallback query returned {len(all_articles)} articles")
            
            unique_articles = []
            seen_urls = set()
            for article in all_articles:
                if article["link"] not in seen_urls:
                    seen_urls.add(article["link"])
                    unique_articles.append(article)
            
            self.logger.info(f"Deduplicated to {len(unique_articles)} unique articles")
            
            # Save execution state for future incremental research
            state["execution_state"] = {
                "executed_queries": list(self.executed_queries),
                "seen_urls": list(seen_urls)
            }
            
            # Handle targeted research (incremental capabilities)
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
                state["research_results"] = current_results
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
                    
                    # Validate overall results
                    validation_results = self._validate_research_results(final_results)
                    
                    state["research_results"] = final_results
                    state["event_metadata"] = event_metadata
                    state["validation_results"] = validation_results
                    
                    self.logger.info(f"Grouped articles into {len(final_results)} distinct events")
                elif return_type != "clustered":
                    state["research_results"] = unique_articles
                    self.logger.info(f"Returning {len(unique_articles)} unclustered articles")
            
            self.logger.info(f"Research completed for {company}")
            return {**state, "goto": "meta_agent", "research_agent_status": "DONE"}
            
        except Exception as e:
            self.logger.error(f"Unexpected error in research agent: {str(e)}")
            if "research_results" not in state:
                state["research_results"] = {}
            return {
                **state, 
                "goto": "meta_agent",
                "error": f"Research agent error: {str(e)}",
                "research_agent_status": "ERROR"
            }
    
    def _validate_research_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Validate the overall research results for quality and coverage"""
        validation = {
            "total_events": len(results),
            "total_articles": sum(len(articles) for articles in results.values()),
            "high_importance_events": 0,
            "medium_importance_events": 0,
            "low_importance_events": 0,
            "articles_per_event": {},
            "source_diversity": 0.0,
            "red_flags": []
        }
        
        all_sources = set()
        
        for event_name, articles in results.items():
            # Count importance levels
            if "- High" in event_name:
                validation["high_importance_events"] += 1
            elif "- Medium" in event_name:
                validation["medium_importance_events"] += 1
            else:
                validation["low_importance_events"] += 1
                
            # Track articles per event
            validation["articles_per_event"][event_name] = len(articles)
            
            # Collect sources for diversity calculation
            event_sources = set()
            for article in articles:
                if article.get("source"):
                    all_sources.add(article.get("source"))
                    event_sources.add(article.get("source"))
            
            # Check for potential red flags in research quality
            if len(articles) == 1:
                validation["red_flags"].append(f"Event '{event_name}' has only one article")
                
            if len(event_sources) < min(3, len(articles)):
                validation["red_flags"].append(f"Event '{event_name}' has low source diversity")
        
        # Calculate source diversity ratio (0.0-1.0)
        total_articles = validation["total_articles"]
        if total_articles > 0 and all_sources:
            validation["source_diversity"] = min(1.0, len(all_sources) / total_articles)
        
        return validation