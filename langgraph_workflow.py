import os
import json
import asyncio
from typing import Dict, List, Any, Tuple, Optional, Union, Annotated, TypedDict, Literal, Set
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor

from langchain.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemoryCheckpoint
from langgraph.graph.graph import CompiledGraph

from base.base_graph import BaseGraph, GraphConfig
from utils.logging import get_logger, setup_logging
from utils.llm_provider import init_llm_provider, get_llm_provider
from utils.prompt_manager import init_prompt_manager

# Import agents
from agents.meta_agent import EnhancedMetaAgent
from agents.research_agent import ResearchAgent
from agents.youtube_agent import YouTubeAgent
from agents.corporate_agent import CorporateAgent
from agents.analyst_agent import EnhancedAnalystAgent
from agents.rag_agent import RAGAgent
from agents.writer_agent import EnhancedWriterAgent


class WorkflowState(TypedDict):
    # Core workflow information
    company: str
    industry: Optional[str]
    meta_iteration: int
    
    # User interaction state
    user_approved: bool
    user_feedback: Optional[Dict[str, str]]
    requires_user_approval: bool
    user_approval_type: Optional[str]
    
    # Agent control
    goto: Optional[str]
    error: Optional[str]
    
    # Agent status tracking
    meta_agent_status: Optional[str]
    research_agent_status: Optional[str]
    corporate_agent_status: Optional[str]
    youtube_agent_status: Optional[str]
    analyst_agent_status: Optional[str]
    rag_agent_status: Optional[str]
    writer_agent_status: Optional[str]
    
    # Parallel execution tracking
    parallel_agents: List[str]
    running_agents: Set[str]
    completed_agents: Set[str]
    failed_agents: Set[str]
    agent_results: Dict[str, Any]
    
    # Research tracking
    research_plan: List[Dict[str, Any]]
    search_history: List[List[str]]
    search_type: Optional[str]
    return_type: Optional[str]
    event_research_iterations: Dict[str, int]
    
    # Agent results
    research_results: Dict[str, List[Dict[str, Any]]]
    event_metadata: Dict[str, Dict[str, Any]]
    corporate_results: Dict[str, Any]
    youtube_results: Dict[str, Any]
    analysis_results: Dict[str, Any]
    
    # Analyst pool tracking
    analyst_tasks: List[Dict[str, Any]]
    analyst_task_results: Dict[str, Any]
    analyst_pool_size: int
    
    # Quality tracking
    quality_assessment: Dict[str, Any]
    analysis_guidance: Dict[str, Any]
    
    # Final outputs
    final_report: Optional[str]
    report_sections: Dict[str, str]
    top_events: List[str]
    other_events: List[str]
    executive_briefing: Optional[str]
    
    # Workflow control
    additional_research_completed: bool
    final_analysis_completed: bool
    final_analysis_requested: bool
    synchronous_pipeline: bool
    workflow_status: Dict[str, Any]
    execution_mode: str


class EnhancedForensicWorkflow(BaseGraph):
    """Enhanced graph-based workflow for financial forensic analysis with parallel execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.config = config or {}
        self.logger = get_logger("forensic_workflow")
        
        # Initialize LLM provider and prompt manager
        init_llm_provider(self.config)
        init_prompt_manager()
        
        # Initialize agents
        self.meta_agent = EnhancedMetaAgent(self.config)
        self.research_agent = ResearchAgent(self.config)
        self.youtube_agent = YouTubeAgent(self.config)
        self.corporate_agent = CorporateAgent(self.config)
        self.analyst_agent = EnhancedAnalystAgent(self.config)
        self.rag_agent = RAGAgent(self.config)
        self.writer_agent = EnhancedWriterAgent(self.config)
        
        # Initialize agent mapping
        self.agents = {
            "meta_agent": self.meta_agent,
            "research_agent": self.research_agent,
            "youtube_agent": self.youtube_agent,
            "corporate_agent": self.corporate_agent,
            "analyst_agent": self.analyst_agent, 
            "rag_agent": self.rag_agent,
            "writer_agent": self.writer_agent,
            "meta_agent_final": self.meta_agent  # Use same instance with different node name
        }
        
        # Configure execution parameters
        self.max_parallel_agents = config.get("workflow", {}).get("max_parallel_agents", 3)
        self.analyst_pool_size = config.get("workflow", {}).get("analyst_pool_size", 5)
        self.require_plan_approval = config.get("workflow", {}).get("require_plan_approval", True)
        
        # Create the workflow graph
        self.graph = self.build_graph()
        
    def build_graph(self) -> CompiledGraph:
        """Build the enhanced workflow graph with parallel execution."""
        # State and workflow settings
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self.create_agent_node(agent, agent_name))
        
        # Add special nodes for parallel execution, user interaction and coordination
        workflow.add_node("parallel_executor", self.parallel_executor_node)
        workflow.add_node("plan_approval", self.plan_approval_node)
        workflow.add_node("research_complete", self.research_complete_node)
        workflow.add_node("analyst_pool", self.analyst_pool_node)
        
        # Add entry point to meta_agent
        workflow.set_entry_point("meta_agent")
        
        # Connect meta_agent to plan_approval
        workflow.add_edge("meta_agent", "plan_approval")
        
        # Connect plan_approval to parallel_executor or back to meta_agent
        workflow.add_conditional_edges(
            "plan_approval",
            self.route_from_plan_approval,
            {
                "parallel_executor": "parallel_executor",
                "meta_agent": "meta_agent"
            }
        )
        
        # Connect parallel_executor to research_complete
        workflow.add_conditional_edges(
            "parallel_executor",
            self.route_from_parallel_executor,
            {
                "research_complete": "research_complete",
                "parallel_executor": "parallel_executor"
            }
        )
        
        # Connect research_complete to meta_agent
        workflow.add_edge("research_complete", "meta_agent")
        
        # Connect meta_agent to analyst_pool or writer_agent
        workflow.add_conditional_edges(
            "meta_agent",
            self.route_from_meta_agent,
            {
                "analyst_pool": "analyst_pool",
                "writer_agent": "writer_agent",
                "parallel_executor": "parallel_executor",
                "END": END
            }
        )
        
        # Connect analyst_pool to meta_agent
        workflow.add_edge("analyst_pool", "meta_agent")
        
        # Connect writer_agent to meta_agent_final
        workflow.add_edge("writer_agent", "meta_agent_final")
        
        # Connect meta_agent_final to END
        workflow.add_edge("meta_agent_final", END)
        
        # Add error handler to capture exceptions
        if self.config.get("enable_error_handling", True):
            workflow.add_node("error_handler", self.handle_error)
            workflow.set_error_handler("error_handler")
        
        # Compile graph
        memory = MemoryCheckpoint()
        return workflow.compile(checkpointer=memory)
    
    def create_agent_node(self, agent, agent_name: str):
        """Create a function that runs an agent and handles async execution."""
        async def run_agent(state: WorkflowState) -> WorkflowState:
            # Update state if agent has specific status field
            agent_status_field = f"{agent.name}_status"
            if agent_status_field not in state:
                state[agent_status_field] = "RUNNING"
                
            try:
                # Execute agent
                updated_state = await agent.run(dict(state))
                
                # Ensure goto field is preserved 
                if "goto" not in updated_state:
                    updated_state["goto"] = "meta_agent"
                    
                # Ensure status field is updated
                if agent_status_field not in updated_state:
                    updated_state[agent_status_field] = "DONE"
                
                # If this is part of parallel execution, track in completed_agents
                if agent_name in state.get("parallel_agents", []):
                    if "completed_agents" not in updated_state:
                        updated_state["completed_agents"] = set()
                    updated_state["completed_agents"].add(agent_name)
                    
                    # Store agent results in the agent_results dict
                    result_key = f"{agent_name}_results"
                    agent_results = updated_state.get("agent_results", {})
                    
                    # Extract agent-specific results based on agent type
                    if agent_name == "research_agent":
                        agent_results["research_agent"] = {
                            "research_results": updated_state.get("research_results", {}),
                            "event_metadata": updated_state.get("event_metadata", {})
                        }
                    elif agent_name == "corporate_agent":
                        agent_results["corporate_agent"] = {
                            "corporate_results": updated_state.get("corporate_results", {})
                        }
                    elif agent_name == "youtube_agent":
                        agent_results["youtube_agent"] = {
                            "youtube_results": updated_state.get("youtube_results", {})
                        }
                        
                    updated_state["agent_results"] = agent_results
                
                return updated_state
                
            except Exception as e:
                self.logger.error(f"Error in {agent.name}: {str(e)}")
                
                # If this is part of parallel execution, track in failed_agents
                if agent_name in state.get("parallel_agents", []):
                    if "failed_agents" not in state:
                        state["failed_agents"] = set()
                    state["failed_agents"].add(agent_name)
                
                return {
                    **state,
                    "goto": "meta_agent",
                    "error": f"Error in {agent.name}: {str(e)}",
                    agent_status_field: "ERROR"
                }
        
        def run_agent_sync(state: WorkflowState) -> WorkflowState:
            """Synchronous wrapper for the async agent execution."""
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop if current one is already running
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(run_agent(state))
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(loop)
            else:
                return loop.run_until_complete(run_agent(state))
                
        return run_agent_sync
    
    def parallel_executor_node(self, state: WorkflowState) -> WorkflowState:
        """Node for executing multiple agents in parallel."""
        # Initialize tracking sets if not present
        if "running_agents" not in state:
            state["running_agents"] = set()
        if "completed_agents" not in state:
            state["completed_agents"] = set()
        if "failed_agents" not in state:
            state["failed_agents"] = set()
        if "agent_results" not in state:
            state["agent_results"] = {}
            
        # Get the list of parallel agents to run
        parallel_agents = state.get("parallel_agents", [])
        if not parallel_agents:
            # Default to research, corporate, and youtube agents
            parallel_agents = ["research_agent", "corporate_agent", "youtube_agent"]
            state["parallel_agents"] = parallel_agents
        
        # Track which agents are still running
        running = state["running_agents"]
        completed = state["completed_agents"]
        failed = state["failed_agents"]
        
        # Determine next step
        if running.union(completed).union(failed) == set(parallel_agents):
            # All agents have been started, check if all are done
            if running:
                # Some agents are still running, return to same node to wait
                self.logger.info(f"Waiting for {len(running)} agents to complete")
                return {**state, "goto": "parallel_executor"}
            else:
                # All agents are done (completed or failed), go to research_complete
                self.logger.info(f"All parallel agents completed: {completed} | failed: {failed}")
                
                # Merge results from all agents
                self.merge_parallel_results(state)
                
                return {**state, "goto": "research_complete"}
        else:
            # There are agents that haven't been started yet
            available_agents = set(parallel_agents) - running - completed - failed
            
            # Limit the number of concurrent agents
            max_concurrent = min(self.max_parallel_agents - len(running), len(available_agents))
            
            if max_concurrent <= 0:
                # At max concurrency, wait for some to complete
                return {**state, "goto": "parallel_executor"}
                
            # Start new agents up to max_concurrent
            next_agents = list(available_agents)[:max_concurrent]
            self.logger.info(f"Starting parallel agents: {next_agents}")
            
            # Update running agents
            state["running_agents"] = running.union(set(next_agents))
            
            # Set which agent to run next
            return {**state, "goto": next_agents[0]}
    
    def merge_parallel_results(self, state: WorkflowState) -> None:
        """Merge results from parallel agents into the main state."""
        agent_results = state.get("agent_results", {})
        
        # Merge research agent results
        if "research_agent" in agent_results:
            research_data = agent_results["research_agent"]
            if "research_results" in research_data:
                state["research_results"] = research_data["research_results"]
            if "event_metadata" in research_data:
                state["event_metadata"] = research_data["event_metadata"]
        
        # Merge corporate agent results
        if "corporate_agent" in agent_results:
            corporate_data = agent_results["corporate_agent"]
            if "corporate_results" in corporate_data:
                state["corporate_results"] = corporate_data["corporate_results"]
        
        # Merge youtube agent results
        if "youtube_agent" in agent_results:
            youtube_data = agent_results["youtube_agent"]
            if "youtube_results" in youtube_data:
                state["youtube_results"] = youtube_data["youtube_results"]
    
    def plan_approval_node(self, state: WorkflowState) -> WorkflowState:
        """Node for user approval of research plan."""
        # Check if plan requires approval
        if self.require_plan_approval and not state.get("user_approved", False):
            # Set state to require user approval
            state["requires_user_approval"] = True
            state["user_approval_type"] = "research_plan"
            
            # If user has provided feedback or approval, process it
            if state.get("user_feedback"):
                feedback = state.get("user_feedback", {})
                
                if feedback.get("approved", False):
                    # User approved the plan
                    state["user_approved"] = True
                    state["requires_user_approval"] = False
                    state["user_approval_type"] = None
                    
                    # Apply any modifications to the research plan
                    if "modified_plan" in feedback:
                        state["research_plan"][-1] = feedback["modified_plan"]
                    
                    # Continue to parallel execution
                    return {**state, "goto": "parallel_executor"}
                else:
                    # User rejected the plan, go back to meta_agent for revision
                    state["user_approved"] = False
                    state["requires_user_approval"] = False
                    state["user_approval_type"] = None
                    
                    if "feedback_text" in feedback:
                        # Add feedback to state for meta_agent to use
                        state["plan_feedback"] = feedback["feedback_text"]
                    
                    return {**state, "goto": "meta_agent"}
            
            # Waiting for user approval, stay in the same state
            return state
        else:
            # No approval required or already approved, proceed to parallel execution
            state["user_approved"] = True
            return {**state, "goto": "parallel_executor"}
    
    def research_complete_node(self, state: WorkflowState) -> WorkflowState:
        """Node for coordinating end of research phase and starting analysis."""
        # Check if research was successful
        research_results = state.get("research_results", {})
        
        if not research_results:
            # No research results, log error and return to meta_agent for handling
            self.logger.warning("No research results found. Returning to meta_agent.")
            return {**state, "goto": "meta_agent", "error": "No research results found"}
        
        # Clear parallel execution tracking for next phase
        state["parallel_agents"] = []
        state["running_agents"] = set()
        state["completed_agents"] = set()
        state["failed_agents"] = set()
        
        # Set research completion flag
        state["research_completed"] = True
        
        # Proceed to meta_agent for planning the analysis phase
        return {**state, "goto": "meta_agent"}
    
    def analyst_pool_node(self, state: WorkflowState) -> WorkflowState:
        """Node for managing pool of analyst agents working on different tasks."""
        # Initialize analyst pool tracking if not present
        if "analyst_tasks" not in state:
            state["analyst_tasks"] = []
        if "analyst_task_results" not in state:
            state["analyst_task_results"] = {}
        if "analyst_pool_size" not in state:
            state["analyst_pool_size"] = self.analyst_pool_size
        
        # Get current tasks and results
        tasks = state["analyst_tasks"]
        results = state["analyst_task_results"]
        
        # Check if all tasks are completed
        if not tasks:
            # All tasks completed, merge results and return to meta_agent
            
            # If we have analysis results from tasks, merge them
            if results:
                # Basic merging logic - in a real implementation this would be more sophisticated
                all_analysis = {}
                for event_name, event_analysis in results.items():
                    all_analysis[event_name] = event_analysis
                
                state["analysis_results"] = {"forensic_insights": all_analysis}
                
            # Set analysis completion flag
            state["analysis_completed"] = True
            
            return {**state, "goto": "meta_agent"}
        
        # Create a thread pool to process tasks in parallel
        with ThreadPoolExecutor(max_workers=state["analyst_pool_size"]) as executor:
            # Function to process a task
            def process_task(task):
                # Extract task data
                event_name = task["event_name"]
                event_data = task["event_data"]
                
                # Clone the analyst agent for this task
                task_agent = EnhancedAnalystAgent(self.config)
                
                # Create task state
                task_state = {
                    "company": state["company"],
                    "event_name": event_name,
                    "event_data": event_data,
                    "analysis_type": task.get("analysis_type", "standard")
                }
                
                # Run the agent synchronously in this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    task_result = loop.run_until_complete(task_agent.process_event(
                        state["company"], event_name, event_data
                    ))
                    loop.close()
                    return event_name, task_result
                except Exception as e:
                    loop.close()
                    self.logger.error(f"Error processing analyst task for {event_name}: {str(e)}")
                    return event_name, {"error": str(e)}
            
            # Submit all tasks to the thread pool
            future_results = {executor.submit(process_task, task): task for task in tasks}
            
            # Process completed tasks as they finish
            for future in future_results:
                try:
                    event_name, task_result = future.result()
                    results[event_name] = task_result
                except Exception as e:
                    self.logger.error(f"Task execution failed: {str(e)}")
        
        # All tasks have been processed
        state["analyst_tasks"] = []
        state["analyst_task_results"] = results
        
        # Continue to meta_agent
        return {**state, "goto": "meta_agent"}
    
    def route_from_meta_agent(self, state: WorkflowState) -> str:
        """Routing logic from meta_agent to next node."""
        # Check for explicit routing
        goto = state.get("goto")
        
        if goto == "END":
            return "END"
            
        # Check workflow phase
        if not state.get("research_completed", False):
            # Still in research phase, route based on context
            if not state.get("research_plan"):
                # No research plan yet, first iteration
                return "plan_approval"
            else:
                # Has research plan, either do more research or move to analysis
                if state.get("requires_additional_research", False):
                    # Setup parallel agents for more research
                    state["parallel_agents"] = ["research_agent"]
                    return "parallel_executor"
                else:
                    # Start analysis phase
                    return "analyst_pool"
        elif not state.get("analysis_completed", False):
            # In analysis phase
            return "analyst_pool"
        elif not state.get("report_completed", False):
            # In report generation phase
            return "writer_agent"
        else:
            # All phases completed
            return "meta_agent_final"
    
    def route_from_parallel_executor(self, state: WorkflowState) -> str:
        """Determine if parallel execution is complete or should continue."""
        # Get tracking sets
        running = state.get("running_agents", set())
        completed = state.get("completed_agents", set())
        failed = state.get("failed_agents", set())
        parallel_agents = state.get("parallel_agents", [])
        
        # Check if all parallel agents have completed or failed
        if len(completed) + len(failed) == len(parallel_agents) and not running:
            return "research_complete"
        else:
            # Continue executing parallel agents
            return "parallel_executor"
    
    def route_from_plan_approval(self, state: WorkflowState) -> str:
        """Determine route based on plan approval status."""
        if state.get("user_approved", False):
            return "parallel_executor"
        else:
            return "meta_agent"
    
    def handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle errors in the workflow."""
        error_info = traceback.format_exc()
        
        self.logger.error(f"Workflow error: {error_info}")
        
        # Update state with error information
        state["error"] = f"Workflow error: {error_info[:500]}..." if len(error_info) > 500 else error_info
        state["goto"] = "meta_agent"  # Route back to meta_agent to handle error
        
        # Update status fields to indicate error
        for agent_name in self.agents:
            status_field = f"{agent_name}_status" 
            if status_field in state and state[status_field] == "RUNNING":
                state[status_field] = "ERROR"
        
        return state
    
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow with the given initial state."""
        # Validate and prepare initial state
        prepared_state = self.prepare_initial_state(initial_state)
        
        # Log workflow start
        self.logger.info(f"Starting workflow for company: {prepared_state['company']}")
        
        # Execute graph
        result = None
        for event in self.graph.stream(prepared_state):
            current_state = event['state']
            current_node = event.get('current_node', 'unknown')
            
            # Log progress
            self.logger.info(f"Executed node: {current_node}")
            
            # Check for errors
            if current_state.get("error"):
                self.logger.error(f"Error in node {current_node}: {current_state['error']}")
                
            # Check for user approval needed
            if current_state.get("requires_user_approval", False):
                # This would normally wait for user input via API
                # For demonstration, we simulate approval after 1 second
                self.logger.info(f"Requires user approval of type: {current_state.get('user_approval_type')}")
                
                # In a real implementation, this would wait for user input via API
                # For now, we just continue with auto-approval for demonstration
                if "user_feedback" not in current_state:
                    current_state["user_feedback"] = {"approved": True}
                
            # Save checkpoint if configured
            if self.config.get("checkpoint_path"):
                self._save_checkpoint(current_state, f"{current_node}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                
            # Store final state
            result = current_state
        
        # Log workflow completion
        self.logger.info(f"Workflow completed for company: {prepared_state['company']}")
        
        return result
    
    def prepare_initial_state(self, initial_state: Dict[str, Any]) -> WorkflowState:
        """Prepare and validate the initial state."""
        # Ensure required fields are present
        if "company" not in initial_state:
            raise ValueError("Company name is required in initial state")
            
        # Initialize default values
        default_state = {
            "company": "",
            "industry": None,
            "meta_iteration": 0,
            "goto": "meta_agent",
            "error": None,
            "meta_agent_status": None,
            "research_agent_status": None,
            "corporate_agent_status": None,
            "youtube_agent_status": None,
            "analyst_agent_status": None,
            "rag_agent_status": None,
            "writer_agent_status": None,
            "research_plan": [],
            "search_history": [],
            "search_type": "google_news",
            "return_type": "clustered",
            "event_research_iterations": {},
            "research_results": {},
            "event_metadata": {},
            "corporate_results": {},
            "youtube_results": {},
            "analysis_results": {},
            "quality_assessment": {},
            "analysis_guidance": {},
            "final_report": None,
            "report_sections": {},
            "top_events": [],
            "other_events": [],
            "executive_briefing": None,
            "additional_research_completed": False,
            "final_analysis_completed": False,
            "final_analysis_requested": False,
            "synchronous_pipeline": False,
            "workflow_status": {},
            "user_approved": False,
            "requires_user_approval": False,
            "user_approval_type": None,
            "user_feedback": None,
            "parallel_agents": [],
            "running_agents": set(),
            "completed_agents": set(),
            "failed_agents": set(),
            "agent_results": {},
            "analyst_tasks": [],
            "analyst_task_results": {},
            "analyst_pool_size": 5,
            "execution_mode": "parallel"
        }
        
        # Merge with provided state, preferring provided values
        complete_state = {**default_state, **initial_state}
        
        return complete_state
    
    def run_sync(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for running the workflow."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new loop if current one is already running
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.run(initial_state))
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        else:
            return loop.run_until_complete(self.run(initial_state))


# Function to create and run the workflow from arguments
def create_and_run_workflow(
    company: str,
    industry: Optional[str] = None,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create and run a forensic workflow for the given company."""
    # Load configuration
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    # Setup logging
    setup_logging("forensic_workflow", level=config.get("log_level", "INFO"))
    logger = get_logger("forensic_workflow")
    
    logger.info(f"Creating workflow for company: {company}")
    
    # Create workflow
    workflow = EnhancedForensicWorkflow(config)
    
    # Prepare initial state
    initial_state = {
        "company": company,
        "industry": industry
    }
    
    # Run workflow
    logger.info(f"Running workflow for company: {company}")
    result = workflow.run_sync(initial_state)
    
    logger.info(f"Workflow completed for company: {company}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Financial Forensic Analysis')
    parser.add_argument('--company', type=str, required=True, help='Company name to analyze')
    parser.add_argument('--industry', type=str, help='Industry of the company')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    result = create_and_run_workflow(
        company=args.company,
        industry=args.industry,
        config_path=args.config
    )
    
    # Print summary of results
    print("\n" + "="*80)
    print(f"Analysis completed for: {args.company}")
    print("="*80)
    
    if result.get("final_report"):
        print(f"Report generated with {len(result.get('final_report'))} characters")
        
        # Save report to file
        output_file = f"{args.company.replace(' ', '_')}_report.md"
        with open(output_file, 'w') as f:
            f.write(result["final_report"])
        print(f"Report saved to: {output_file}")
        
    else:
        print("No report generated. Check logs for errors.")
        
    if result.get("error"):
        print(f"Error: {result['error']}")