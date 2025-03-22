import json
import os
from typing import Dict, List, Union, Any, Optional
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger


class MetaAgent(BaseAgent):
    name = "meta_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager(self.name)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def evaluate_research_quality(self, company: str, industry: str, research_results: Dict) -> Dict:
        if not research_results:
            return {
                "overall_score": 0,
                "coverage_score": 0,
                "balance_score": 0,
                "credibility_score": 0,
                "assessment": "No research results available.",
                "recommendations": {"default recommendation": "Continue with available research while addressing technical issues."}
            }
        
        self.logger.info(f"Evaluating research quality for {company}")
        
        llm_provider = await get_llm_provider()
        
        variables = {
            "company": company,
            "industry": industry,
            "research_results": json.dumps(research_results)
        }
        
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="research_quality_eval",
            variables=variables
        )
        
        input_message = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("evaluation"))
        
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        assessment = json.loads(cleaned_response)
        
        self.logger.info(f"Research quality score: {assessment.get('overall_score', 0)}/10")
        
        return assessment
    
    async def identify_research_gaps(self, company: str, industry: str, event_name: str, 
                               event_data: List[Dict], previous_research_plans: List[Dict]) -> Dict[str, str]:
        self.logger.info(f"Identifying research gaps for event: {event_name}")
        
        llm_provider = await get_llm_provider()
        
        previous_queries = []
        for plan in previous_research_plans:
            query_cats = plan.get("query_categories", {})
            for cat, desc in query_cats.items():
                previous_queries.append(f"{cat}: {desc}")
        
        variables = {
            "company": company,
            "industry": industry,
            "event_name": event_name,
            "event_data": json.dumps(event_data),
            "previous_queries": json.dumps(previous_queries)
        }
        
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="research_gaps_identification",
            variables=variables
        )
        
        input_message = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("planning"))
        
        response_content = response.strip()
        
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content
            
        gaps = json.loads(json_content)
        
        if gaps:
            self.logger.info(f"Found {len(gaps)} research gaps for event '{event_name}'")
        else:
            self.logger.info(f"No research gaps found for event '{event_name}'")
            
        return gaps
    
    async def create_research_plan(self, company: str, research_gaps: Union[List[Dict], Dict], previous_plans: List[Dict] = None) -> Dict:
        if not research_gaps:
            return {}
            
        self.logger.info(f"Creating research plan for {company} based on identified gaps")
        
        llm_provider = await get_llm_provider()
        
        variables = {
            "company": company,
            "research_gaps": json.dumps(research_gaps),
            "previous_plans": json.dumps(previous_plans or [])
        }
        
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="research_plan_creation",
            variables=variables
        )
        
        input_message = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("planning"))
        
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        plan = json.loads(cleaned_response)
        
        self.logger.info(f"Created research plan: {plan.get('objective', 'No objective specified')}")
        
        return plan
    
    async def generate_analysis_guidance(self, company: str, research_results: Dict) -> Dict:
        if not research_results:
            return {
                "focus_areas": ["General company assessment"],
                "priorities": ["Establish baseline understanding of company"],
                "analysis_strategies": ["Conduct general background research"],
                "red_flags": ["Insufficient information available"]
            }
        
        self.logger.info(f"Generating analysis guidance for {company}")
        
        llm_provider = await get_llm_provider()
        
        variables = {
            "company": company,
            "research_results": json.dumps(research_results)
        }
        
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="analysis_guidance",
            variables=variables
        )
        
        input_message = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = await llm_provider.generate_text(input_message, model_name=self.config.get("models", {}).get("planning"))
        
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        guidance = json.loads(cleaned_response)
        
        self.logger.info(f"Generated guidance with {len(guidance.get('focus_areas', []))} focus areas")
        
        return guidance
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._log_start(state)
        
        company = state.get("company", "")
        industry = state.get("industry", "Unknown")
        
        if not company:
            self.logger.error("Company name is missing!")
            return {**state, "goto": "END", "error": "Company name is missing"}
        
        if "meta_iteration" not in state:
            state["meta_iteration"] = 0
        if "search_history" not in state:
            state["search_history"] = []
        if "event_research_iterations" not in state:
            state["event_research_iterations"] = {}
        
        state["meta_iteration"] += 1
        current_iteration = state["meta_iteration"]
        
        self.logger.info(f"Starting iteration {current_iteration}")
        
        research_results = state.get("research_results", {})
        analysis_results = state.get("analysis_results", {})
        previous_research_plans = state.get("research_plan", [])
        
        research_threshold = self.config.get("quality_thresholds", {}).get("min_quality_score", 6)
        max_iterations = self.config.get("max_iterations", 3)
        max_event_iterations = self.config.get("max_event_iterations", 2)
        
        if not research_results and current_iteration == 1:
            self.logger.info("Starting preliminary research")
            try:
                preliminary_guidelines = await self._load_preliminary_guidelines(company, industry)
                state["research_plan"] = [preliminary_guidelines]
                state["search_type"] = "google_news"
                state["return_type"] = "clustered"
                self.logger.info("Loaded preliminary research guidelines")
                return {**state, "goto": "research_agent"}
            except Exception as e:
                self.logger.error(f"Error loading preliminary guidelines: {e}")
                basic_plan = {
                    "objective": f"Investigate potential issues related to {company}",
                    "key_areas_of_focus": ["Legal issues", "Financial concerns", "Regulatory actions"],
                    "query_categories": {"general": "Investigate potential issues"},
                    "query_generation_guidelines": "Focus on negative news and regulatory concerns"
                }
                state["research_plan"] = [basic_plan]
                state["search_type"] = "google_news"
                state["return_type"] = "clustered"
                return {**state, "goto": "research_agent"}
        
        if research_results and (not state.get("quality_assessment") or 
                               state.get("quality_assessment", {}).get("overall_score", 0) < research_threshold):
            self.logger.info("Evaluating research quality")
            quality_assessment = await self.evaluate_research_quality(company, industry, research_results)
            state["quality_assessment"] = quality_assessment
            
            if quality_assessment.get('overall_score', 0) < research_threshold and current_iteration < max_iterations:
                self.logger.info("Research quality below threshold. Generating targeted research plan")
                research_gaps = quality_assessment.get('recommendations', {})
                if research_gaps:
                    research_plan = await self.create_research_plan(company, research_gaps, previous_research_plans)
                    state["research_plan"].append(research_plan)
                    return {**state, "goto": "research_agent"}
        
        if (state.get("quality_assessment", {}).get("overall_score", 0) >= research_threshold or 
            current_iteration >= max_iterations) and not analysis_results:
            self.logger.info("Moving to analysis phase")
            analysis_guidance = await self.generate_analysis_guidance(company, research_results)
            state["analysis_guidance"] = analysis_guidance
            return {**state, "goto": "analyst_agent"}
        
        if analysis_results and current_iteration < max_iterations:
            self.logger.info("Analysis completed. Identifying research gaps in events")
            
            all_event_gaps = {}
            for event_name, event_data in analysis_results.get("forensic_insights", {}).items():
                current_event_iterations = state["event_research_iterations"].get(event_name, 0)
                if current_event_iterations >= max_event_iterations:
                    self.logger.info(f"Skipping event '{event_name}' - reached max iterations ({max_event_iterations})")
                    continue
                    
                event_gaps = await self.identify_research_gaps(
                    company, 
                    industry,
                    event_name, 
                    event_data, 
                    previous_research_plans
                )
                
                if event_gaps:
                    all_event_gaps[event_name] = event_gaps
            
            if all_event_gaps:
                targeted_plans = []
                for event_name, gaps in all_event_gaps.items():
                    event_plan = await self.create_research_plan(
                        company, 
                        gaps, 
                        previous_research_plans
                    )
                    
                    if event_name not in state["event_research_iterations"]:
                        state["event_research_iterations"][event_name] = 0
                    state["event_research_iterations"][event_name] += 1
                    
                    event_plan["event_name"] = event_name
                    targeted_plans.append(event_plan)
                
                if targeted_plans:
                    state["research_plan"].extend(targeted_plans)
                    self.logger.info(f"Created {len(targeted_plans)} targeted research plans")
                    return {**state, "goto": "research_agent"}
        
        if analysis_results and state.get("additional_research_completed") and not state.get("final_analysis_completed"):
            self.logger.info("Running final analysis")
            state["final_analysis_requested"] = True
            state["final_analysis_completed"] = True
            return {**state, "goto": "analyst_agent"}
        
        self.logger.info(f"Process complete after {current_iteration} iterations")
        
        self._log_completion({**state, "goto": "writer_agent"})
        return {**state, "goto": "writer_agent", "status": "complete"}
    
    async def _load_preliminary_guidelines(self, company: str, industry: str) -> Dict:
        prompt_path = "prompts/meta_agent/preliminary_guidelines.json"
        
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as file:
                guidelines = json.load(file)
                return guidelines
        else:
            raise Exception("No preliminary research guidelines found")