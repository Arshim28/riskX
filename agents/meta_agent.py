import json
import os
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime
import traceback
import asyncio
import copy
import time
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

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
    retries: int = 0
    max_retries: int = 3


class WorkflowStateSnapshot(BaseModel):
    """Snapshot of workflow state for tracking/rollback"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    pending_agents: Set[str] = Field(default_factory=set)
    running_agents: Set[str] = Field(default_factory=set)
    completed_agents: Set[str] = Field(default_factory=set)
    failed_agents: Set[str] = Field(default_factory=set)
    agent_statuses: Dict[str, str] = Field(default_factory=dict)


class MetaAgent(BaseAgent):
    name = "meta_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager(self.name)
        self.postgres_tool = PostgresTool(config.get("postgres", {}))
        
        # Configuration parameters
        self.max_parallel_agents = config.get("meta_agent", {}).get("max_parallel_agents", 3)
        self.research_quality_threshold = config.get("quality_thresholds", {}).get("min_quality_score", 6)
        self.max_iterations = config.get("max_iterations", 3)
        self.max_event_iterations = config.get("max_event_iterations", 2)
        self.enable_recovery = config.get("meta_agent", {}).get("enable_recovery", True)
        self.parallel_execution = config.get("meta_agent", {}).get("parallel_execution", True)
        
        # Agent dependency configuration - updated for direct agent structure
        self.agent_dependencies = {
            "research_agent": [],
            "youtube_agent": [],
            "corporate_agent": [],
            "rag_agent": [],
            "analyst_agent": ["research_agent", "corporate_agent", "youtube_agent"],
            "writer_agent": ["analyst_agent"]
        }
        
        # Agent priority configuration (higher number = higher priority)
        self.agent_priorities = {
            "research_agent": 60,
            "youtube_agent": 90,
            "corporate_agent": 80,
            "rag_agent": 70,
            "analyst_agent": 40,
            "writer_agent": 30
        }
        
        # State tracking for individual agents
        self.agent_tasks = {}
        self.completed_agents = set()
        self.running_agents = set()
        self.pending_agents = set()
        self.failed_agents = set()
        
        # State history for rollback/recovery
        self.state_history = []
        self.max_history = 10
        
        # Error tracking
        self.last_error = None
        self.error_count = 0
        self.critical_error_threshold = 3
        
        # Lock for concurrent state updates
        self.state_lock = asyncio.Lock()
    
    async def save_state_snapshot(self, state: Dict[str, Any]) -> None:
        """Save current workflow state for possible rollback"""
        # Filter out any None status values to avoid validation errors
        agent_statuses = {name: task.status for name, task in self.agent_tasks.items() 
                          if task.status is not None}
        
        snapshot = WorkflowStateSnapshot(
            pending_agents=self.pending_agents.copy(),
            running_agents=self.running_agents.copy(),
            completed_agents=self.completed_agents.copy(),
            failed_agents=self.failed_agents.copy(),
            agent_statuses=agent_statuses
        )
        
        self.state_history.append(snapshot)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
            
        try:
            company = state.get("company", "unknown")
            workflow_id = state.get("workflow_id", f"wf_{int(time.time())}")
            serializable_snapshot = {
                "timestamp": snapshot.timestamp,
                "pending_agents": list(snapshot.pending_agents),
                "running_agents": list(snapshot.running_agents),
                "completed_agents": list(snapshot.completed_agents),
                "failed_agents": list(snapshot.failed_agents),
                "agent_statuses": snapshot.agent_statuses
            }
        
            # Modified query to include all required columns and ensure it works properly
            await self.postgres_tool.run(
                command="execute_query",
                query="""
                    INSERT INTO workflow_snapshots 
                        (workflow_id, company, timestamp, snapshot_data) 
                    VALUES 
                        ($1, $2, CURRENT_TIMESTAMP, $3) 
                    ON CONFLICT (company) DO UPDATE SET 
                        snapshot_data = $3,
                        timestamp = CURRENT_TIMESTAMP
                """,
                params=[workflow_id, company, json.dumps(serializable_snapshot)]
            )
        except Exception as e:
            self.logger.warning(f"Failed to persist workflow snapshot: {str(e)}")    

    async def rollback_to_last_snapshot(self) -> bool:
        """Rollback workflow state to last stable snapshot"""
        if not self.state_history:
            self.logger.warning("No state history available for rollback")
            return False
            
        snapshot = self.state_history.pop()
        
        async with self.state_lock:
            self.pending_agents = snapshot.pending_agents
            self.running_agents = snapshot.running_agents
            self.completed_agents = snapshot.completed_agents
            self.failed_agents = snapshot.failed_agents
            
            for agent_name, status in snapshot.agent_statuses.items():
                if agent_name in self.agent_tasks:
                    self.agent_tasks[agent_name].status = status
        
        self.logger.info(f"Rolled back workflow state to snapshot from {snapshot.timestamp}")
        return True
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def evaluate_research_quality(self, company: str, industry: str, research_results: Dict) -> Dict:
        """Evaluate the quality of research results"""
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def identify_research_gaps(self, company: str, industry: str, event_name: str, 
                               event_data: List[Dict], previous_research_plans: List[Dict]) -> Dict[str, str]:
        """Identify gaps in current research data"""
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_research_plan(self, company: str, research_gaps: Dict, previous_plans: List[Dict] = None) -> Dict:
        """Create a research plan based on identified gaps"""
        if not research_gaps:
            self.logger.warning(f"No research gaps provided for {company}, returning empty plan")
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_analysis_guidance(self, company: str, research_results: Dict) -> Dict:
        """Generate guidance for analysis phase based on research results"""
        if not research_results:
            self.logger.warning(f"No research results available for {company} to generate guidance")
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
    
    async def load_preliminary_guidelines(self, company: str, industry: str) -> Dict:
        """Load preliminary research guidelines"""
        prompt_path = "prompts/meta_agent/preliminary_guidelines.json"
        
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
        """Generate a comprehensive status of the workflow"""
        company = state.get("company", "Unknown")
        iteration = state.get("meta_iteration", 0)
        
        status = {
            "company": company,
            "iteration": iteration,
            "agents": {},
            "overall_status": "IN_PROGRESS",
            "current_phase": state.get("current_phase", "RESEARCH"),
            "next_steps": [],
            "errors": [],
            "progress_percentage": 0,
            "state_timestamp": datetime.now().isoformat()
        }
        
        # Track agent statuses
        for agent_name in ["research_agent", "corporate_agent", "youtube_agent", "rag_agent", 
                         "analyst_agent", "writer_agent"]:
            status_field = f"{agent_name}_status"
            agent_status = state.get(status_field)
            
            status["agents"][agent_name] = {
                "status": agent_status,
                "started_at": self.agent_tasks[agent_name].started_at if agent_name in self.agent_tasks else None,
                "completed_at": self.agent_tasks[agent_name].completed_at if agent_name in self.agent_tasks else None,
                "error": state.get("error") if agent_status == "ERROR" else None
            }
            
            if agent_status == "ERROR" and state.get("error"):
                status["errors"].append(f"{agent_name}: {state.get('error')}")
        
        # Calculate progress percentage based on phase and agent completion
        current_phase = state.get("current_phase", "RESEARCH")
        
        # Define phase weights and agent weights within phases
        phase_weights = {
            "RESEARCH": 0.4,
            "ANALYSIS": 0.3,
            "REPORT_GENERATION": 0.2,
            "REPORT_REVIEW": 0.05,
            "COMPLETE": 0.05
        }
        
        phase_agents = {
            "RESEARCH": ["research_agent", "corporate_agent", "youtube_agent", "rag_agent"],
            "ANALYSIS": ["analyst_agent"],
            "REPORT_GENERATION": ["writer_agent"],
            "REPORT_REVIEW": [],
            "COMPLETE": []
        }
        
        # Calculate phase completion
        phase_completion = {}
        overall_progress = 0
        
        for phase, weight in phase_weights.items():
            if phase == current_phase:
                # Current phase - calculate partial completion
                phase_agents_count = len(phase_agents[phase])
                if phase_agents_count == 0:
                    phase_completion[phase] = 1.0  # No agents means phase is complete
                else:
                    completed_count = sum(1 for agent in phase_agents[phase] 
                                       if state.get(f"{agent}_status") == "DONE")
                    phase_completion[phase] = completed_count / phase_agents_count
                
                # Add to overall progress
                overall_progress += weight * phase_completion[phase]
                
                # All previous phases are complete
                break
            else:
                # Previous phases are complete
                phase_completion[phase] = 1.0
                overall_progress += weight
        
        status["progress_percentage"] = int(overall_progress * 100)
        
        # Determine next steps
        if current_phase == "RESEARCH":
            # Check which research agents are not done
            for agent in phase_agents["RESEARCH"]:
                if state.get(f"{agent}_status") != "DONE":
                    status["next_steps"].append(f"Complete {agent} execution")
            
            if not status["next_steps"]:
                status["next_steps"].append("Transition to ANALYSIS phase")
        
        elif current_phase == "ANALYSIS":
            if state.get("analyst_agent_status") != "DONE":
                status["next_steps"].append("Complete analyst_agent execution")
            else:
                status["next_steps"].append("Transition to REPORT_GENERATION phase")
        
        elif current_phase == "REPORT_GENERATION":
            if state.get("writer_agent_status") != "DONE":
                status["next_steps"].append("Complete writer_agent execution")
            else:
                status["next_steps"].append("Transition to REPORT_REVIEW phase")
        
        elif current_phase == "REPORT_REVIEW":
            status["next_steps"].append("Review final report")
        
        elif current_phase == "COMPLETE":
            status["next_steps"].append("Workflow complete")
            status["overall_status"] = "DONE"
        
        return status
    
    async def save_workflow_status(self, state: Dict[str, Any], status: Dict[str, Any]) -> None:
        """Save workflow status to database"""
        company = state.get("company", "Unknown")
        workflow_id = state.get("workflow_id", f"wf_{int(time.time())}")
        current_phase = state.get("current_phase", "RESEARCH")
        status_text = status.get("status", "RUNNING")
        
        try:
            # First ensure the column exists
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="ALTER TABLE workflow_status ADD COLUMN IF NOT EXISTS status_data JSONB",
                    params=[]
                )
            except Exception as e:
                self.logger.warning(f"Failed to ensure status_data column exists: {str(e)}")
            
            # Now do the insert/update with all required columns
            await self.postgres_tool.run(
                command="execute_query",
                query="""
                    INSERT INTO workflow_status 
                        (workflow_id, company, status, current_phase, status_data, last_updated) 
                    VALUES 
                        ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP) 
                    ON CONFLICT (company) DO UPDATE SET 
                        status_data = $5,
                        status = $3,
                        current_phase = $4,
                        last_updated = CURRENT_TIMESTAMP
                """,
                params=[workflow_id, company, status_text, current_phase, json.dumps(status)]
            )
            self.logger.debug(f"Saved workflow status to database for {company}")
        except Exception as e:
            self.logger.error(f"Failed to save workflow status: {str(e)}")
    
    async def initialize_agent_tasks(self, state: Dict[str, Any]) -> None:
        """Initialize tasks for all individual agents with proper dependencies and priorities"""
        async with self.state_lock:
            company = state.get("company", "Unknown")
            self.logger.info(f"Initializing agent tasks for {company}")
            
            self.agent_tasks = {}
            self.completed_agents = set()
            self.running_agents = set()
            self.pending_agents = set()
            self.failed_agents = set()
            
            # Add all individual agents
            for agent_name, dependencies in self.agent_dependencies.items():
                priority = self.agent_priorities.get(agent_name, 0)
                self.logger.info(f"Initializing task for agent {agent_name} with priority {priority}")
                
                # Create agent task
                task = AgentTask(
                    agent_name=agent_name,
                    priority=priority,
                    dependencies=dependencies,
                    is_parallel=self.parallel_execution,
                    timeout_seconds=self.config.get(agent_name, {}).get("timeout", 300)
                )
                
                self.agent_tasks[agent_name] = task
                self.pending_agents.add(agent_name)
            
            await self.save_state_snapshot(state)
    
    async def get_agent_status(self, agent_name: str) -> str:
        """Get current status of an agent"""
        if agent_name in self.agent_tasks:
            return self.agent_tasks[agent_name].status
        return "UNKNOWN"
    
    async def update_agent_status(self, agent_name: str, status: str, error: Optional[str] = None) -> None:
        """Update the status of an agent with proper state transitions"""
        async with self.state_lock:
            if agent_name not in self.agent_tasks:
                self.logger.warning(f"Attempted to update status for unknown agent: {agent_name}")
                return
                
            prev_status = self.agent_tasks[agent_name].status
            
            # Special handling for RAG agent errors that are non-critical
            if agent_name == "rag_agent" and status == "ERROR" and self.is_recoverable_rag_error(error):
                self.logger.info(f"Converting non-critical RAG error to success: {error}")
                status = "DONE"  # Convert to success
                error = None     # Clear the error
            
            self.agent_tasks[agent_name].status = status
            
            # Status transitions with proper set updates
            if status == "RUNNING" and prev_status != "RUNNING":
                self.running_agents.add(agent_name)
                if agent_name in self.pending_agents:
                    self.pending_agents.remove(agent_name)
                
                # Record start time
                self.agent_tasks[agent_name].started_at = datetime.now().isoformat()
                    
            elif status == "DONE" and prev_status != "DONE":
                self.completed_agents.add(agent_name)
                if agent_name in self.running_agents:
                    self.running_agents.remove(agent_name)
                
                # Record completion time
                self.agent_tasks[agent_name].completed_at = datetime.now().isoformat()
                    
            elif status == "ERROR" and prev_status != "ERROR":
                self.failed_agents.add(agent_name)
                if agent_name in self.running_agents:
                    self.running_agents.remove(agent_name)
                
                # Record error details and completion time
                self.agent_tasks[agent_name].error = error
                self.agent_tasks[agent_name].completed_at = datetime.now().isoformat()
                self.last_error = error
                
                # Increment error count for recovery decisions
                self.error_count += 1
            
            self.logger.info(f"Updated agent {agent_name} status: {prev_status} -> {status}")
    
    def is_recoverable_rag_error(self, error_message: str) -> bool:
        """Check if RAG error is recoverable (non-critical)"""
        if not error_message:
            return False

        recoverable_messages = [
        "No documents loaded",
        "No documents to categorize",
        "No vector store",
        "Vector store not initialized"
        ]

        for message in recoverable_messages:
            if message in error_message:
                return True

        return False
    
    async def are_dependencies_satisfied(self, agent_name: str) -> bool:
        """Check if all dependencies for an agent are satisfied"""
        if agent_name not in self.agent_tasks:
            return False
            
        task = self.agent_tasks[agent_name]
        
        for dependency in task.dependencies:
            if dependency not in self.completed_agents:
                return False
                
        return True
    
    async def get_next_agents(self) -> List[str]:
        """Get list of agents that are ready to run based on dependencies"""
        async with self.state_lock:
            available_agents = []
            
            for agent_name in self.pending_agents:
                if await self.are_dependencies_satisfied(agent_name):
                    available_agents.append(agent_name)
            
            # Sort by priority (higher number = higher priority)
            available_agents.sort(key=lambda name: self.agent_tasks[name].priority, reverse=True)
            
            # Limit by parallelism configuration
            if self.parallel_execution:
                max_to_run = self.max_parallel_agents - len(self.running_agents)
                return available_agents[:max(0, max_to_run)]
            else:
                # In non-parallel mode, return at most one agent
                return available_agents[:1] if not self.running_agents else []
    
    async def should_retry_agent(self, agent_name: str) -> bool:
        """Determine if a failed agent should be retried"""
        if agent_name not in self.agent_tasks:
            return False
            
        task = self.agent_tasks[agent_name]
        
        # Check if we've exceeded retry limit
        if task.retries >= task.max_retries:
            self.logger.info(f"Agent {agent_name} has reached maximum retries ({task.max_retries})")
            return False
            
        # Increment retry count
        task.retries += 1
        
        # Reset status to PENDING
        self.failed_agents.remove(agent_name)
        self.pending_agents.add(agent_name)
        task.status = "PENDING"
        task.error = None
        
        self.logger.info(f"Retrying agent {agent_name} (attempt {task.retries})")
        return True
    
    async def is_workflow_complete(self) -> bool:
        """Check if the workflow is complete (all agents done or failed)"""
        # First check if any agent is still pending or running
        if self.pending_agents or self.running_agents:
            return False
        
        # Next check if all required agents completed successfully
        required_agents = ["research_agent", "analyst_agent", "writer_agent"]
        for agent_name in required_agents:
            if agent_name not in self.completed_agents and agent_name not in self.failed_agents:
                return False
        
        # If we get here, workflow is complete
        return True
    
    async def is_workflow_stalled(self) -> bool:
        """Check if the workflow is stalled and can't proceed"""
        # Check if there are pending agents but none can run
        if self.pending_agents and not await self.get_next_agents():
            # For each pending agent, check if all its dependencies have failed
            for agent_name in self.pending_agents:
                task = self.agent_tasks[agent_name]
                
                if not task.dependencies:
                    # Agent with no dependencies should be runnable
                    continue
                    
                all_deps_failed = True
                for dependency in task.dependencies:
                    if dependency not in self.failed_agents:
                        all_deps_failed = False
                        break
                
                if all_deps_failed:
                    self.logger.warning(f"Workflow stalled: agent {agent_name} has all dependencies failed")
                    return True
                    
        # Check if error count exceeds critical threshold
        if self.error_count >= self.critical_error_threshold:
            self.logger.warning(f"Workflow potentially stalled: error count ({self.error_count}) exceeds threshold")
            # Don't return True here to allow for recovery attempts
        
        return False
    
    async def attempt_recovery(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to recover from stalled or error state"""
        if not self.enable_recovery:
            return False, state
            
        self.logger.info("Attempting workflow recovery")
        
        if "rag_agent" in self.failed_agents:
            error = self.agent_tasks["rag_agent"].error
            if self.is_recoverable_rag_error(error):
                self.logger.info(f"Converting non-critical rag error to success: {error}")

                self.failed_agents.remove("rag_agent")
                self.completed_agents.add("rag_agent")
                self.agent_tasks['rag_agent'].status = "DONE"
                self.agent_tasks['rag_agent'].error = None

                state["rag_agent_status"] = "DONE"
                state["rag_results"] = {
                    "response": "No document-based analysis available",
                    "retrieval_results": {
                        "success": False,
                        "message": f"RAG agent status: No docs available"
                    },
                    "query": state.get("query", "No query processed")
                }
                return True, state

        # Strategy 1: Retry failed agents with critical dependencies
        recovery_attempted = False
        critical_agents = ["research_agent", "analyst_agent", "writer_agent"]
        
        for agent_name in critical_agents:
            if agent_name in self.failed_agents:
                if await self.should_retry_agent(agent_name):
                    recovery_attempted = True
        
        # Strategy 2: If previous retries don't help, try rolling back state
        if not recovery_attempted and len(self.state_history) > 1:
            # Don't use the very last snapshot as it might be corrupted
            rollback_success = await self.rollback_to_last_snapshot()
            if rollback_success:
                recovery_attempted = True
                self.error_count = 0  # Reset error count after rollback
        
        # Strategy 3: Last resort - if specific agents failed, try to continue with fallbacks
        if not recovery_attempted:
            if "research_agent" in self.failed_agents and "research_results" not in state:
                # Critical failure in research phase, provide empty fallback
                state["research_results"] = {}
                state["research_agent_status"] = "DONE"  # Pretend it succeeded
                await self.update_agent_status("research_agent", "DONE")
                recovery_attempted = True
                self.logger.warning("Recovery: Using empty research results as fallback")
            
            if "analyst_agent" in self.failed_agents and "analysis_results" not in state:
                # Critical failure in analysis phase, provide empty fallback
                state["analysis_results"] = {
                    "forensic_insights": {},
                    "event_synthesis": {},
                    "company_analysis": {},
                    "red_flags": [],
                    "timeline": []
                }
                state["analyst_agent_status"] = "DONE"
                await self.update_agent_status("analyst_agent", "DONE")
                recovery_attempted = True
                self.logger.warning("Recovery: Using empty analysis results as fallback")
        
        if recovery_attempted:
            self.logger.info("Recovery attempt completed")
            # Save new workflow status after recovery
            status = await self.generate_workflow_status(state)
            await self.save_workflow_status(state, status)
            
        return recovery_attempted, state
    
    async def check_workflow_status(self, state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Check workflow status and handle completion or stalled state"""
        # Check if workflow is complete
        if await self.is_workflow_complete():
            self.logger.info("Workflow complete")
            return True, "END", state
        
        # Check if workflow is stalled and attempt recovery if needed
        if await self.is_workflow_stalled():
            self.logger.warning(f"Workflow stalled due to failed dependencies: {self.last_error}")
            
            # Attempt recovery
            recovery_success, updated_state = await self.attempt_recovery(state)
            
            if recovery_success:
                self.logger.info("Recovery successful, continuing workflow")
                return False, "", updated_state
            else:
                # If recovery failed, end the workflow
                self.logger.error("Recovery failed, ending workflow")
                error_state = {**state, "error": f"Workflow stalled and recovery failed: {self.last_error}"}
                return True, "END", error_state
        
        return False, "", state
        
    async def determine_next_agent(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Determine the next agent to run based on current phase and dependencies"""
        # Get current phase
        current_phase = state.get("current_phase", "RESEARCH")
        
        # Check if explicit goto is set
        goto = state.get("goto")
        if goto and goto != "meta_agent":
            self.logger.info(f"Explicit goto set to {goto}")
            
            # Clear goto to prevent loops
            state["goto"] = None
            
            # Update agent status if needed
            if goto in self.agent_tasks:
                await self.update_agent_status(goto, "RUNNING")
            
            return goto, state
        
        # Phase-based routing
        if current_phase == "RESEARCH":
            # Check which research agents need to run
            # Sort research agents by priority (higher priority first)
            research_agents = ["research_agent", "corporate_agent", "youtube_agent", "rag_agent"]
            research_agents_with_priority = [(agent, self.agent_priorities.get(agent, 0)) for agent in research_agents]
            sorted_research_agents = [agent for agent, _ in sorted(research_agents_with_priority, key=lambda x: x[1], reverse=True)]
            
            self.logger.info(f"Prioritized research agents order: {sorted_research_agents}")
            
            for agent in sorted_research_agents:
                if state.get(f"{agent}_status") != "DONE":
                    await self.update_agent_status(agent, "RUNNING")
                    self.logger.info(f"Selected {agent} for execution in RESEARCH phase (priority: {self.agent_priorities.get(agent, 0)})")
                    return agent, state
            
            # If all research agents are done, transition to ANALYSIS phase
            self.logger.info("All research agents complete, transitioning to ANALYSIS phase")
            state["current_phase"] = "ANALYSIS"
            await self.update_agent_status("analyst_agent", "RUNNING")
            return "analyst_agent", state
            
        elif current_phase == "ANALYSIS":
            # Check if analysis is done
            if state.get("analyst_agent_status") != "DONE":
                await self.update_agent_status("analyst_agent", "RUNNING")
                self.logger.info("Selected analyst_agent for execution in ANALYSIS phase")
                return "analyst_agent", state
                
            # If analysis is done, transition to REPORT_GENERATION phase
            self.logger.info("Analysis complete, transitioning to REPORT_GENERATION phase")
            state["current_phase"] = "REPORT_GENERATION"
            await self.update_agent_status("writer_agent", "RUNNING")
            return "writer_agent", state
            
        elif current_phase == "REPORT_GENERATION":
            # Check if report generation is done
            if state.get("writer_agent_status") != "DONE":
                await self.update_agent_status("writer_agent", "RUNNING")
                self.logger.info("Selected writer_agent for execution in REPORT_GENERATION phase")
                return "writer_agent", state
                
            # If report generation is done, transition to COMPLETE phase
            self.logger.info("Report generation complete, transitioning to COMPLETE phase")
            state["current_phase"] = "COMPLETE"
            return "END", state
            
        elif current_phase == "COMPLETE":
            self.logger.info("Workflow is complete")
            return "END", state
            
        # If we get here, workflow is in an unknown state
        self.logger.warning(f"Unknown phase: {current_phase}, waiting for next instruction")
        return "WAIT", state
    
    async def manage_workflow(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Manage workflow execution with improved error handling"""
        # Initialize agent tasks if not already done
        if not self.agent_tasks:
            await self.initialize_agent_tasks(state)
        
        # Generate and save workflow status
        status = await self.generate_workflow_status(state)
        await self.save_workflow_status(state, status)
        
        # Check workflow completion or stalled state
        is_terminal, terminal_state, updated_state = await self.check_workflow_status(state)
        if is_terminal:
            return updated_state, terminal_state
        
        # State may have been updated during recovery
        state = updated_state
        
        # Determine next agent to run
        next_step, state = await self.determine_next_agent(state)
        
        return state, next_step
    
    async def merge_agent_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from agent executions into the main state"""
        # This is called when an agent returns to ensure all results are properly merged
        # Check for agent-specific results in state and update accordingly
        updated_state = state.copy()
        
        # Check status fields for each agent
        for agent_name in self.agent_tasks:
            status_key = f"{agent_name}_status"
            
            if status_key in state and state[status_key] is not None:
                # Update our internal tracking
                error = state.get("error")
                if agent_name == "rag_agent" and state[status_key] == "ERROR" and self.is_recoverable_rag_error(error):
                    self.logger.info(f"Converting non-critical RAG Agent error to success: {error}")
                    updated_state[status_key] = "DONE"

                    if "goto" in updated_state and updated_state["goto"] == "meta_agent":
                        updated_state["error"] = None

                    updated_state["rag_results"] = {
                        "response": "No document-based analysis available",
                        "retrieval_results": {
                            "success": False,
                            "message": f"RAG agent status: No docs available"
                        },
                        "query": updated_state.get("query", "No query processed")
                    }
                await self.update_agent_status(
                    agent_name, 
                    updated_state[status_key],
                    state.get("error") if updated_state[status_key] == "ERROR" else None
                )
        
        # Check if all research agents are complete to transition phases
        research_agents = ["research_agent", "corporate_agent", "youtube_agent", "rag_agent"]
        all_research_complete = True
        
        for agent in research_agents:
            # RAG agent is optional if disabled
            if agent == "rag_agent" and not updated_state.get("enable_rag", True):
                continue
                
            if updated_state.get(f"{agent}_status") != "DONE":
                all_research_complete = False
                break
                
        if all_research_complete and updated_state.get("current_phase") == "RESEARCH":
            self.logger.info("All research agents complete, transitioning to ANALYSIS phase")
            updated_state["current_phase"] = "ANALYSIS"
                
        # Check if analysis is complete to transition to report generation
        if updated_state.get("analyst_agent_status") == "DONE" and updated_state.get("current_phase") == "ANALYSIS":
            self.logger.info("Analysis complete, transitioning to REPORT_GENERATION phase")
            updated_state["current_phase"] = "REPORT_GENERATION"
                
        # Check if report generation is complete
        if updated_state.get("writer_agent_status") == "DONE" and updated_state.get("current_phase") == "REPORT_GENERATION":
            self.logger.info("Report generation complete, transitioning to COMPLETE phase")
            updated_state["current_phase"] = "COMPLETE"
        
        # Take a snapshot of the current state for recovery
        await self.save_state_snapshot(updated_state)
        
        return updated_state
    
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implements the abstract _execute method required by BaseAgent.
        This is a wrapper around the run method.
        """
        return await self.run(state)
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method with improved error handling and state management"""
        self._log_start(state)
        
        # Deep copy state to avoid mutation issues
        current_state = copy.deepcopy(state)
        
        company = current_state.get("company", "")
        industry = current_state.get("industry", "Unknown")
        
        if not company:
            self.logger.error("Company name is missing!")
            return {**current_state, "goto": "END", "error": "Company name is missing"}
        
        # Initialize iteration counter if not present
        if "meta_iteration" not in current_state:
            current_state["meta_iteration"] = 0
        if "search_history" not in current_state:
            current_state["search_history"] = []
        if "event_research_iterations" not in current_state:
            current_state["event_research_iterations"] = {}
        
        # Increment iteration counter
        current_state["meta_iteration"] += 1
        iteration = current_state["meta_iteration"]
        
        self.logger.info(f"Starting iteration {iteration} for {company}")
        
        try:
            # Update agent statuses from state
            if self.agent_tasks:
                current_state = await self.merge_agent_results(current_state)
            
            # Handle first iteration - initialize research plan
            if iteration == 1 and not current_state.get("research_plan"):
                self.logger.info("Starting preliminary research phase")
                
                await self.initialize_agent_tasks(current_state)
                
                # Load preliminary guidelines
                preliminary_guidelines = await self.load_preliminary_guidelines(company, industry)
                current_state["research_plan"] = [preliminary_guidelines]
                current_state["search_type"] = "google_news"
                current_state["return_type"] = "clustered"
                
                # Generate and save initial workflow status
                status = await self.generate_workflow_status(current_state)
                await self.save_workflow_status(current_state, status)
                
                # Get the highest priority agent (YouTube) instead of hardcoding research_agent
                research_agents = ["research_agent", "corporate_agent", "youtube_agent", "rag_agent"]
                research_agents_with_priority = [(agent, self.agent_priorities.get(agent, 0)) for agent in research_agents]
                sorted_research_agents = [agent for agent, _ in sorted(research_agents_with_priority, key=lambda x: x[1], reverse=True)]
                
                self.logger.info(f"First iteration: Prioritized agents: {sorted_research_agents}")
                first_agent = sorted_research_agents[0]
                await self.update_agent_status(first_agent, "RUNNING")
                self.logger.info(f"Starting workflow with highest priority agent: {first_agent}")
                return {**current_state, "goto": first_agent}
            
            # If we're returning from another agent, handle the transition
            if "goto" in current_state and current_state["goto"] == "meta_agent":
                # Manage workflow to determine next steps
                updated_state, next_step = await self.manage_workflow(current_state)
                
                if next_step == "END":
                    self.logger.info("Workflow complete, finalizing")
                    return {**updated_state, "goto": "END"}
                    
                elif next_step == "WAIT":
                    self.logger.info("Waiting for running agents to complete")
                    return {**updated_state, "goto": "WAIT"}
                    
                else:
                    self.logger.info(f"Directing workflow to {next_step}")
                    return {**updated_state, "goto": next_step}
            else:
                # Default to continuing with workflow management
                updated_state, next_step = await self.manage_workflow(current_state)
                
                if next_step == "END":
                    self.logger.info("Workflow complete, finalizing")
                    return {**updated_state, "goto": "END"}
                    
                elif next_step == "WAIT":
                    self.logger.info("Waiting for running agents to complete")
                    return {**updated_state, "goto": "WAIT"}
                    
                else:
                    self.logger.info(f"Directing workflow to {next_step}")
                    return {**updated_state, "goto": next_step}
                
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Error in meta_agent: {str(e)}\n{tb}"
            self.logger.error(error_msg)
            
            # Save error state to database for debugging
            try:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO workflow_errors (company, error_data) VALUES ($1, $2)",
                    params=[company, json.dumps({
                        "error": str(e),
                        "traceback": tb,
                        "timestamp": datetime.now().isoformat(),
                        "iteration": iteration
                    })]
                )
            except Exception as db_error:
                self.logger.error(f"Failed to save error to database: {str(db_error)}")
            
            # Attempt recovery if enabled
            if self.enable_recovery:
                recovery_success, recovered_state = await self.attempt_recovery(current_state)
                if recovery_success:
                    self.logger.info("Recovered from error, continuing workflow")
                    updated_state, next_step = await self.manage_workflow(recovered_state)
                    
                    if next_step != "END":
                        return {**updated_state, "goto": next_step}
            
            # If we can't recover, return error state
            return {**current_state, "goto": "END", "error": error_msg}