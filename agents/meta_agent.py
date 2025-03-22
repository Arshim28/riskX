import json
import os
from typing import Dict, List, Union, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.postgres_tool import PostgresTool


class AgentTask(BaseModel):
    agent_name: str
    priority: int = 0
    dependencies: List[str] = Field(default_factory=list)
    is_parallel: bool = True
    timeout_seconds: int = 300
    status: str = "PENDING"
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class EnhancedMetaAgent(BaseAgent):
    name = "meta_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager(self.name)
        self.postgres_tool = PostgresTool(config.get("postgres", {}))
        
        # Configure agent execution parameters
        self.max_parallel_agents = config.get("meta_agent", {}).get("max_parallel_agents", 3)
        self.research_quality_threshold = config.get("quality_thresholds", {}).get("min_quality_score", 6)
        self.max_iterations = config.get("max_iterations", 3)
        self.max_event_iterations = config.get("max_event_iterations", 2)
        
        # Configure agent dependencies and parallel execution
        self.parallel_execution = config.get("meta_agent", {}).get("parallel_execution", True)
        self.agent_dependencies = {
            "research_agent": [],
            "youtube_agent": [],
            "corporate_agent": [],
            "analyst_agent": ["research_agent"],
            "rag_agent": [],
            "writer_agent": ["analyst_agent", "corporate_agent"]
        }
        
        # Agent priority (higher = more important)
        self.agent_priorities = {
            "research_agent": 80,
            "youtube_agent": 60,
            "corporate_agent": 70,
            "analyst_agent": 50,
            "rag_agent": 40,
            "writer_agent": 30
        }
        
        # Agent status tracking
        self.agent_tasks = {}
        self.completed_agents = set()
        self.running_agents = set()
        self.pending_agents = set()
        self.failed_agents = set()
        
        self.last_error = None
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def evaluate_research_quality(self, company: str, industry: str, research_results: Dict) -> Dict:
        if not research_results:
            self.logger.warning(f"No research results available for {company}")
            return {
                "overall_score": 0,
                "coverage_score": 0,
                "balance_score": 0,
                "credibility_score": 0,
                "assessment": "No research results available.",
                "recommendations": {"default recommendation": "Continue with available research while addressing technical issues."}
            }
        
        self.logger.info(f"Evaluating research quality for {company}")
        
        try:
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
        except Exception as e:
            self.logger.error(f"Error during research quality evaluation: {str(e)}")
            return {
                "overall_score": 0,
                "coverage_score": 0,
                "balance_score": 0,
                "credibility_score": 0,
                "assessment": f"Error during evaluation: {str(e)[:100]}...",
                "recommendations": {"default recommendation": "Continue with available research while addressing technical issues."}
            }
    
    async def identify_research_gaps(self, company: str, industry: str, event_name: str, 
                               event_data: List[Dict], previous_research_plans: List[Dict]) -> Dict[str, str]:
        self.logger.info(f"Identifying research gaps for event: {event_name}")
        
        try:
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
        except Exception as e:
            self.logger.error(f"Error identifying research gaps: {str(e)}")
            return {}
    
    async def create_research_plan(self, company: str, research_gaps: Union[List[Dict], Dict], previous_plans: List[Dict] = None) -> Dict:
        if not research_gaps:
            self.logger.warning(f"No research gaps provided for {company}, returning empty plan")
            return {}
            
        self.logger.info(f"Creating research plan for {company} based on identified gaps")
        
        try:
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
        except Exception as e:
            self.logger.error(f"Error creating research plan: {str(e)}")
            # Return a minimal valid plan rather than empty dict
            return {
                "objective": f"Investigate {company} with focus on basic information",
                "key_areas_of_focus": ["General company information"],
                "query_categories": {"general": f"Basic information about {company}"},
                "query_generation_guidelines": "Focus on company structure, industry, and recent news"
            }
    
    async def generate_analysis_guidance(self, company: str, research_results: Dict) -> Dict:
        if not research_results:
            self.logger.warning(f"No research results available for {company} to generate guidance")
            return {
                "focus_areas": ["General company assessment"],
                "priorities": ["Establish baseline understanding of company"],
                "analysis_strategies": ["Conduct general background research"],
                "red_flags": ["Insufficient information available"]
            }
        
        self.logger.info(f"Generating analysis guidance for {company}")
        
        try:
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
        except Exception as e:
            self.logger.error(f"Error generating analysis guidance: {str(e)}")
            return {
                "focus_areas": ["General company assessment"],
                "priorities": ["Review available information thoroughly"],
                "analysis_strategies": ["Focus on factual information"],
                "red_flags": ["Technical issues in guidance generation"]
            }
    
    async def _load_preliminary_guidelines(self, company: str, industry: str) -> Dict:
        """Load or generate preliminary research guidelines"""
        prompt_path = "prompts/meta_agent/preliminary_guidelines.json"
        
        # Try to load from file
        if os.path.exists(prompt_path):
            try:
                with open(prompt_path, "r") as file:
                    guidelines = json.load(file)
                    self.logger.info(f"Loaded preliminary guidelines from {prompt_path}")
                    return guidelines
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing preliminary guidelines JSON: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error loading preliminary guidelines: {str(e)}")
        
        # If file doesn't exist or loading failed, generate a basic plan
        self.logger.warning("No preliminary guidelines file found, generating basic plan")
        return {
            "objective": f"Initial investigation into {company}",
            "key_areas_of_focus": [
                "Company structure and leadership",
                f"Recent news related to {company}",
                "Regulatory compliance",
                "Financial reporting",
                "Legal issues"
            ],
            "query_categories": {
                "structure": f"{company} company structure leadership executives",
                "news": f"{company} recent news controversy",
                "regulatory": f"{company} regulatory investigation compliance",
                "financial": f"{company} financial reporting earnings",
                "legal": f"{company} lawsuit legal issues"
            },
            "query_generation_guidelines": f"Focus on factual information about {company} with emphasis on potential issues"
        }
    
    async def generate_workflow_status(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate workflow status report for tracking progress"""
        company = state.get("company", "Unknown")
        iteration = state.get("meta_iteration", 0)
        
        status = {
            "company": company,
            "iteration": iteration,
            "agents": {},
            "overall_status": "IN_PROGRESS",
            "current_phase": "Unknown",
            "next_steps": [],
            "errors": [],
            "progress_percentage": 0
        }
        
        # Calculate the progress of each agent
        total_agents = len(self.agent_tasks)
        completed_count = len(self.completed_agents)
        running_count = len(self.running_agents)
        pending_count = len(self.pending_agents)
        failed_count = len(self.failed_agents)
        
        if total_agents > 0:
            status["progress_percentage"] = int((completed_count / total_agents) * 100)
        
        # Determine current phase
        if "research_results" not in state or not state["research_results"]:
            status["current_phase"] = "Initial Research"
        elif "analysis_results" not in state or not state["analysis_results"]:
            status["current_phase"] = "Analysis"
        elif "final_report" not in state or not state["final_report"]:
            status["current_phase"] = "Report Generation"
        else:
            status["current_phase"] = "Complete"
            status["overall_status"] = "DONE"
        
        # Add agent statuses
        for agent_name, task in self.agent_tasks.items():
            agent_status = {
                "status": task.status,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error": task.error
            }
            status["agents"][agent_name] = agent_status
            
            if task.error:
                status["errors"].append(f"{agent_name}: {task.error}")
        
        # Determine next steps
        if pending_count > 0:
            status["next_steps"].append(f"Wait for {pending_count} pending agents to complete")
        
        if failed_count > 0:
            status["next_steps"].append(f"Address {failed_count} failed agents")
        
        if status["overall_status"] == "DONE":
            status["next_steps"].append("Workflow complete")
        
        return status
    
    async def save_workflow_status(self, state: Dict[str, Any], status: Dict[str, Any]) -> None:
        """Save workflow status to database"""
        company = state.get("company", "Unknown")
        
        # Try to save status to database
        try:
            await self.postgres_tool.run(
                command="execute_query",
                query="INSERT INTO workflow_status (company, status_data) VALUES ($1, $2) ON CONFLICT (company) DO UPDATE SET status_data = $2",
                params=[company, json.dumps(status)]
            )
        except Exception as e:
            self.logger.error(f"Failed to save workflow status: {str(e)}")
    
    def initialize_agent_tasks(self, state: Dict[str, Any]) -> None:
        """Initialize agent tasks based on configuration and dependencies"""
        company = state.get("company", "Unknown")
        self.logger.info(f"Initializing agent tasks for {company}")
        
        self.agent_tasks = {}
        self.completed_agents = set()
        self.running_agents = set()
        self.pending_agents = set()
        self.failed_agents = set()
        
        # Create agent tasks
        for agent_name, dependencies in self.agent_dependencies.items():
            priority = self.agent_priorities.get(agent_name, 0)
            
            # Determine if agent can run in parallel
            is_parallel = True
            if not self.parallel_execution:
                is_parallel = False
            
            # Create task
            task = AgentTask(
                agent_name=agent_name,
                priority=priority,
                dependencies=dependencies,
                is_parallel=is_parallel
            )
            
            self.agent_tasks[agent_name] = task
            self.pending_agents.add(agent_name)
    
    def get_agent_status(self, agent_name: str) -> str:
        """Get status of agent task"""
        if agent_name in self.agent_tasks:
            return self.agent_tasks[agent_name].status
        return "UNKNOWN"
    
    def update_agent_status(self, agent_name: str, status: str, error: Optional[str] = None) -> None:
        """Update status of agent task"""
        if agent_name in self.agent_tasks:
            prev_status = self.agent_tasks[agent_name].status
            self.agent_tasks[agent_name].status = status
            
            # Update tracking sets
            if status == "RUNNING" and prev_status != "RUNNING":
                self.running_agents.add(agent_name)
                if agent_name in self.pending_agents:
                    self.pending_agents.remove(agent_name)
                    
            elif status == "DONE" and prev_status != "DONE":
                self.completed_agents.add(agent_name)
                if agent_name in self.running_agents:
                    self.running_agents.remove(agent_name)
                    
            elif status == "ERROR" and prev_status != "ERROR":
                self.failed_agents.add(agent_name)
                if agent_name in self.running_agents:
                    self.running_agents.remove(agent_name)
                self.agent_tasks[agent_name].error = error
                self.last_error = error
    
    def are_dependencies_satisfied(self, agent_name: str) -> bool:
        """Check if all dependencies for an agent are satisfied"""
        if agent_name not in self.agent_tasks:
            return False
            
        task = self.agent_tasks[agent_name]
        
        for dependency in task.dependencies:
            if dependency not in self.completed_agents:
                return False
                
        return True
    
    def get_next_agents(self) -> List[str]:
        """Get next agents that can be run based on dependencies and priorities"""
        available_agents = []
        
        # Check which pending agents have all dependencies satisfied
        for agent_name in self.pending_agents:
            if self.are_dependencies_satisfied(agent_name):
                available_agents.append(agent_name)
        
        # Sort by priority (highest first)
        available_agents.sort(key=lambda name: self.agent_tasks[name].priority, reverse=True)
        
        # Limit based on parallel execution setting
        if self.parallel_execution:
            max_to_run = self.max_parallel_agents - len(self.running_agents)
            if max_to_run <= 0:
                return []
            return available_agents[:max_to_run]
        else:
            # In sequential mode, only return one agent if none are currently running
            if not self.running_agents and available_agents:
                return [available_agents[0]]
            return []
    
    def is_workflow_complete(self) -> bool:
        """Check if the workflow is complete"""
        # All agents must be either completed or failed
        return len(self.pending_agents) == 0 and len(self.running_agents) == 0
    
    def is_workflow_stalled(self) -> bool:
        """Check if the workflow is stalled (no progress possible)"""
        # If there are pending agents but none can be started due to dependencies on failed agents
        if self.pending_agents and not self.get_next_agents():
            # Check if all dependencies trace back to failed agents
            for agent_name in self.pending_agents:
                task = self.agent_tasks[agent_name]
                
                all_deps_failed = True
                for dependency in task.dependencies:
                    if dependency not in self.failed_agents:
                        all_deps_failed = False
                        break
                
                if all_deps_failed:
                    return True
                    
        return False
    
    async def manage_workflow(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Manage the workflow and determine next steps"""
        # Initialize agent tasks if not already done
        if not self.agent_tasks:
            self.initialize_agent_tasks(state)
        
        # Generate and save workflow status
        status = await self.generate_workflow_status(state)
        await self.save_workflow_status(state, status)
        
        # Check if workflow is complete or stalled
        if self.is_workflow_complete():
            self.logger.info("Workflow complete")
            return state, "END"
            
        if self.is_workflow_stalled():
            self.logger.warning(f"Workflow stalled due to failed dependencies: {self.last_error}")
            return {**state, "error": f"Workflow stalled: {self.last_error}"}, "END"
        
        # Get next agents to run
        next_agents = self.get_next_agents()
        
        if not next_agents:
            # No agents can run right now, wait for running agents to complete
            self.logger.info(f"Waiting for {len(self.running_agents)} running agents to complete")
            return state, "WAIT"
        
        # Select the highest priority agent
        next_agent = next_agents[0]
        self.logger.info(f"Next agent to run: {next_agent}")
        
        # Update state and return
        self.update_agent_status(next_agent, "RUNNING")
        return state, next_agent
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the meta agent workflow"""
        self._log_start(state)
        
        company = state.get("company", "")
        industry = state.get("industry", "Unknown")
        
        if not company:
            self.logger.error("Company name is missing!")
            return {**state, "goto": "END", "error": "Company name is missing"}
        
        # Initialize state variables if not present
        if "meta_iteration" not in state:
            state["meta_iteration"] = 0
        if "search_history" not in state:
            state["search_history"] = []
        if "event_research_iterations" not in state:
            state["event_research_iterations"] = {}
        
        state["meta_iteration"] += 1
        current_iteration = state["meta_iteration"]
        
        self.logger.info(f"Starting iteration {current_iteration} for {company}")
        
        # Check for agent completion or errors
        for agent_name in ["research_agent", "youtube_agent", "corporate_agent", "analyst_agent", "rag_agent", "writer_agent"]:
            status_key = f"{agent_name}_status"
            if status_key in state:
                if state[status_key] == "DONE":
                    self.update_agent_status(agent_name, "DONE")
                elif state[status_key] == "ERROR":
                    error = state.get("error", f"Error in {agent_name}")
                    self.update_agent_status(agent_name, "ERROR", error)
        
        # Check for initial setup
        if current_iteration == 1 and not state.get("research_plan"):
            self.logger.info("Starting preliminary research phase")
            
            # Initialize agent tasks
            self.initialize_agent_tasks(state)
            
            # Create the initial research plan
            preliminary_guidelines = await self._load_preliminary_guidelines(company, industry)
            state["research_plan"] = [preliminary_guidelines]
            state["search_type"] = "google_news"
            state["return_type"] = "clustered"
            
            # Update and save workflow status
            status = await self.generate_workflow_status(state)
            await self.save_workflow_status(state, status)
            
            # Manual override for first operation to start with research
            self.update_agent_status("research_agent", "RUNNING")
            return {**state, "goto": "research_agent"}
        
        # Handle returning from an agent
        if "goto" in state and state["goto"] == "meta_agent":
            # Get the agent that just completed
            completed_agent = None
            for agent_name in self.running_agents:
                status_key = f"{agent_name}_status"
                if status_key in state and (state[status_key] == "DONE" or state[status_key] == "ERROR"):
                    completed_agent = agent_name
                    break
            
            if completed_agent:
                status = state.get(f"{completed_agent}_status", "UNKNOWN")
                error = state.get("error")
                
                self.update_agent_status(completed_agent, status, error)
                self.logger.info(f"Agent {completed_agent} completed with status: {status}")
        
        # Determine next steps in workflow management
        updated_state, next_step = await self.manage_workflow(state)
        
        if next_step == "END":
            self.logger.info("Workflow complete, finalizing")
            return {**updated_state, "goto": "END"}
            
        elif next_step == "WAIT":
            self.logger.info("Waiting for running agents to complete")
            return {**updated_state, "goto": "WAIT"}
            
        else:
            # Next step is an agent name
            self.logger.info(f"Directing workflow to {next_step}")
            return {**updated_state, "goto": next_step}