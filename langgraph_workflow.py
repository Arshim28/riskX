import os
import json
import yaml
import asyncio
import concurrent.futures
import sys
import time
from typing import Dict, List, Any, Tuple, Optional, Union, Annotated, TypedDict, Literal, Set, Callable, Type
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from dotenv import load_dotenv

from base.base_agents import BaseAgent
from base.base_graph import BaseGraph, GraphConfig
from utils.logging import get_logger, setup_logging
from utils.llm_provider import init_llm_provider, get_llm_provider
from utils.prompt_manager import init_prompt_manager
from utils.config_utils import load_config_with_env_vars, get_nested_config

# Import agents
from agents.meta_agent import MetaAgent
from agents.research_agent import ResearchAgent
from agents.youtube_agent import YouTubeAgent
from agents.corporate_agent import CorporateAgent
from agents.analyst_agent import AnalystAgent
from agents.rag_agent import RAGAgent
from agents.writer_agent import WriterAgent

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# Enhanced debugging function
def debug_print(msg, level="INFO"):
    """Print debug messages with timestamps and log levels"""
    timestamp = datetime.now().isoformat()
    print(f"DEBUG [{level}] {timestamp}: {msg}", flush=True)


class WorkflowError(Exception):
    """Custom exception for workflow errors with detailed information."""
    def __init__(self, message, agent_name=None, node=None, details=None):
        self.agent_name = agent_name
        self.node = node
        self.details = details
        full_message = f"[{agent_name or 'UNKNOWN'}] {message}"
        if details:
            full_message += f"\nDetails: {details}"
        super().__init__(full_message)


class WorkflowState(TypedDict):
    company: str
    industry: Optional[str]
    meta_iteration: int
    
    user_approved: bool
    user_feedback: Optional[Dict[str, str]]
    requires_user_approval: bool
    user_approval_type: Optional[str]
    
    goto: Optional[str]
    error: Optional[str]
    
    meta_agent_status: Optional[str]
    research_agent_status: Optional[str]
    corporate_agent_status: Optional[str]
    youtube_agent_status: Optional[str]
    analyst_agent_status: Optional[str]
    rag_agent_status: Optional[str]
    writer_agent_status: Optional[str]

    agent_results: Dict[str, Any]
    
    research_plan: List[Dict[str, Any]]
    search_history: List[List[str]]
    search_type: Optional[str]
    return_type: Optional[str]
    event_research_iterations: Dict[str, int]
    
    research_results: Dict[str, List[Dict[str, Any]]]
    event_metadata: Dict[str, Dict[str, Any]]
    corporate_results: Dict[str, Any]
    youtube_results: Dict[str, Any]
    rag_results: Dict[str, Any]
    analysis_results: Dict[str, Any]
    
    analyst_tasks: List[Dict[str, Any]]
    analyst_task_results: Dict[str, Any]
    
    quality_assessment: Dict[str, Any]
    analysis_guidance: Dict[str, Any]
    
    final_report: Optional[str]
    report_sections: Dict[str, str]
    top_events: List[str]
    other_events: List[str]
    executive_briefing: Optional[str]
    
    rag_initialized: bool
    enable_rag: bool
    vector_store_dir: Optional[str]
    
    additional_research_completed: bool
    final_analysis_completed: bool
    final_analysis_requested: bool
    
    workflow_status: Dict[str, Any]
    execution_mode: str
    current_phase: str


class EnhancedForensicWorkflow(BaseGraph):
    """
    Enhanced workflow implementing a direct agent-based architecture with no pool structure.
    Each agent is a standalone node in the workflow graph with centralized orchestration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.config = config or {}
        self.logger = get_logger("forensic_workflow")
        
        debug_print("Initializing EnhancedForensicWorkflow")
        debug_print("Python version: " + sys.version)
        
        try:
            debug_print("Initializing LLM provider")
            init_llm_provider(self.config)
            debug_print("LLM provider initialized")
            
            debug_print("Initializing prompt manager")
            init_prompt_manager()
            debug_print("Prompt manager initialized")
        except Exception as e:
            debug_print(f"Error during initialization: {str(e)}", "ERROR")
            debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
            raise
        
        try:
            # Initialize all agents directly
            debug_print("Initializing MetaAgent")
            self.meta_agent = MetaAgent(self.config)
            
            debug_print("Initializing ResearchAgent")
            self.research_agent = ResearchAgent(self.config)
            
            debug_print("Initializing CorporateAgent")
            self.corporate_agent = CorporateAgent(self.config)
            
            debug_print("Initializing YouTubeAgent")
            self.youtube_agent = YouTubeAgent(self.config)
            
            debug_print("Initializing RAGAgent")
            self.rag_agent = RAGAgent(self.config)
            
            debug_print("Initializing AnalystAgent")
            self.analyst_agent = AnalystAgent(self.config)
            
            debug_print("Initializing WriterAgent")
            self.writer_agent = WriterAgent(self.config)
            
            # Initialize agent mapping with all individual agents
            debug_print("Setting up agent mapping")
            self.agents = {
                "meta_agent": self.meta_agent,
                "research_agent": self.research_agent,
                "corporate_agent": self.corporate_agent,
                "youtube_agent": self.youtube_agent,
                "rag_agent": self.rag_agent,
                "analyst_agent": self.analyst_agent,
                "writer_agent": self.writer_agent,
                "meta_agent_final": self.meta_agent  # Use same instance with different node name
            }
            
            # Configure execution parameters
            self.require_plan_approval = config.get("workflow", {}).get("require_plan_approval", True)
            
            # Create the workflow graph
            debug_print("Building workflow graph")
            self.graph = self.build_graph()
            debug_print("Workflow graph built successfully")
            
        except Exception as e:
            debug_print(f"Error during workflow initialization: {str(e)}", "ERROR")
            debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
            raise
    
    # Implementing abstract methods required by BaseGraph
    def add_node(self, name: str, agent: Type[BaseAgent]) -> None:
        """Implementation of abstract method from BaseGraph."""
        self.logger.warning(
            f"add_node method called directly on EnhancedForensicWorkflow for node {name}. "
            "This is not the intended usage pattern as the graph is pre-built."
        )
        # No-op implementation to satisfy abstract class requirement
        self.nodes[name] = agent
    
    def add_edge(self, source: str, target: str) -> None:
        """Implementation of abstract method from BaseGraph."""
        self.logger.warning(
            f"add_edge method called directly on EnhancedForensicWorkflow from {source} to {target}. "
            "This is not the intended usage pattern as the graph is pre-built."
        )
        # No-op implementation to satisfy abstract class requirement
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append(target)
    
    def add_conditional_edges(self, source: str, router: Callable) -> None:
        """Implementation of abstract method from BaseGraph."""
        self.logger.warning(
            f"add_conditional_edges method called directly on EnhancedForensicWorkflow for source {source}. "
            "This is not the intended usage pattern as the graph is pre-built."
        )
        # No-op implementation to satisfy abstract class requirement
        self.conditional_edges[source] = router
    
    def build_graph(self) -> CompiledGraph:
        """
        Build the enhanced workflow graph with a direct agent-based architecture.
        Each agent is a separate node with the meta_agent as the central orchestrator.
        """
        debug_print("Starting build_graph")
        try:
            # Create state graph
            debug_print("Creating StateGraph")
            workflow = StateGraph(WorkflowState)
            debug_print("StateGraph created successfully")
            
            # Add nodes for each individual agent
            debug_print(f"Adding {len(self.agents)} nodes to graph")
            for agent_name, agent in self.agents.items():
                debug_print(f"Adding node: {agent_name}")
                workflow.add_node(agent_name, self.create_agent_node(agent, agent_name))
            
            # Set entry point to meta_agent
            debug_print("Setting entry point to meta_agent")
            workflow.set_entry_point("meta_agent")
            
            # Central routing from meta_agent to all other agents
            debug_print("Adding conditional edges from meta_agent")
            workflow.add_conditional_edges(
                "meta_agent",
                self.route_from_meta_agent,
                {
                    "research_agent": "research_agent",
                    "corporate_agent": "corporate_agent",
                    "youtube_agent": "youtube_agent",
                    "rag_agent": "rag_agent",
                    "analyst_agent": "analyst_agent",
                    "writer_agent": "writer_agent",
                    "meta_agent_final": "meta_agent_final",
                    "END": END
                }
            )
            
            # All agents return to meta_agent
            debug_print("Adding edges from all agents back to meta_agent")
            workflow.add_edge("research_agent", "meta_agent")
            workflow.add_edge("corporate_agent", "meta_agent")
            workflow.add_edge("youtube_agent", "meta_agent")
            workflow.add_edge("rag_agent", "meta_agent")
            workflow.add_edge("analyst_agent", "meta_agent")
            workflow.add_edge("writer_agent", "meta_agent")
            
            # Final meta_agent node connects to END
            debug_print("Adding edge from meta_agent_final to END")
            workflow.add_edge("meta_agent_final", END)
            
            # Compile graph without a checkpointer to avoid issues with async execution
            debug_print("Compiling graph")
            compiled_graph = workflow.compile()
            debug_print("Graph compiled successfully")
            return compiled_graph
            
        except Exception as e:
            debug_print(f"Error in build_graph: {str(e)}", "ERROR")
            debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
            raise
    
    def create_agent_node(self, agent, agent_name: str):
        """Create a function that runs an agent and handles async execution."""
        debug_print(f"Creating node function for agent: {agent_name}")
        
        async def run_agent(state: WorkflowState) -> WorkflowState:
            debug_print(f"run_agent async function started for: {agent_name}")
            # Update state if agent has specific status field
            agent_status_field = f"{agent.name}_status" 
            if agent_status_field not in state:
                state[agent_status_field] = "RUNNING"
                debug_print(f"Set {agent_status_field}=RUNNING")
                
            try:
                # Check for errors in the state
                if "error" in state and state["error"]:
                    error_msg = f"Stopping due to previous error: {state['error']}"
                    debug_print(error_msg, "ERROR")
                    raise WorkflowError(error_msg, agent_name=agent_name, node=agent_name)
                
                # Execute agent
                debug_print(f"Executing agent.run for: {agent_name}")
                updated_state = await agent.run(dict(state))
                debug_print(f"agent.run completed for: {agent_name}")
                
                # Ensure status field is updated
                if agent_status_field not in updated_state:
                    updated_state[agent_status_field] = "DONE"
                    debug_print(f"Set {agent_status_field}=DONE")
                
                # Check for errors in the result
                if "error" in updated_state and updated_state["error"]:
                    error_msg = f"Error in {agent_name}: {updated_state['error']}"
                    debug_print(error_msg, "ERROR")
                    raise WorkflowError(error_msg, agent_name=agent_name, node=agent_name)
                
                return updated_state
                
            except Exception as e:
                error_msg = f"Error in {agent_name}: {str(e)}"
                self.logger.error(error_msg)
                debug_print(error_msg, "ERROR")
                debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                
                # Propagate the original error if it's already a WorkflowError
                if isinstance(e, WorkflowError):
                    raise e
                # Otherwise wrap it in a WorkflowError
                raise WorkflowError(
                    error_msg,
                    agent_name=agent_name,
                    node=agent_name,
                    details=traceback.format_exc()
                )
        
        def run_agent_sync(state: WorkflowState) -> WorkflowState:
            """Synchronous wrapper for the async agent execution."""
            debug_print(f"run_agent_sync called for: {agent_name}")
            
            # Check if the state already has an error, stop execution if it does
            if "error" in state and state["error"]:
                error_msg = f"Stopping execution due to previous error: {state['error']}"
                debug_print(error_msg, "ERROR")
                raise WorkflowError(error_msg, agent_name=agent_name, node=agent_name)
            
            # Check if we're already in an async context - if so, this needs special handling
            try:
                current_task = asyncio.current_task()
                debug_print(f"Current asyncio task: {current_task}")
                
                if current_task is not None:
                    debug_print(f"Already in async context for: {agent_name}")
                    debug_print(f"Running {agent_name} in a separate thread to avoid deadlock")
                    
                    # Run in a new thread with its own event loop
                    def run_in_thread():
                        debug_print(f"Starting separate thread for {agent_name}")
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            debug_print(f"Running {agent_name} in separate event loop")
                            result = new_loop.run_until_complete(run_agent(state))
                            debug_print(f"Agent {agent_name} completed in separate thread")
                            return result
                        except Exception as e:
                            debug_print(f"Error in thread execution for {agent_name}: {str(e)}", "ERROR")
                            debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                            
                            # Propagate the original error if it's already a WorkflowError
                            if isinstance(e, WorkflowError):
                                raise e
                            # Otherwise wrap it in a WorkflowError
                            raise WorkflowError(
                                f"Error in thread for {agent_name}: {str(e)}",
                                agent_name=agent_name,
                                node=agent_name,
                                details=traceback.format_exc()
                            )
                        finally:
                            debug_print(f"Closing event loop for {agent_name}")
                            new_loop.close()
                    
                    # Use a ThreadPoolExecutor to run the agent in a separate thread
                    debug_print(f"Creating ThreadPoolExecutor for {agent_name}")
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        try:
                            debug_print(f"Submitting {agent_name} to executor")
                            result = executor.submit(run_in_thread).result(timeout=600)  # 10 minute timeout
                            debug_print(f"Got result from executor for {agent_name}")
                            return result
                        except concurrent.futures.TimeoutError:
                            error_msg = f"Timeout running {agent_name} in separate thread"
                            debug_print(error_msg, "ERROR")
                            raise WorkflowError(
                                error_msg,
                                agent_name=agent_name,
                                node=agent_name
                            )
                        except Exception as e:
                            error_msg = f"Error executing {agent_name} in thread: {str(e)}"
                            debug_print(error_msg, "ERROR")
                            debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                            
                            # Propagate the original error if it's already a WorkflowError
                            if isinstance(e, WorkflowError):
                                raise e
                            # Otherwise wrap it in a WorkflowError
                            raise WorkflowError(
                                error_msg,
                                agent_name=agent_name,
                                node=agent_name,
                                details=traceback.format_exc()
                            )
                        
            except Exception as e:
                debug_print(f"Error checking async context: {str(e)}", "WARNING")
                debug_print(f"Traceback: {traceback.format_exc()}", "WARNING")
            
            # Standard synchronous execution path
            debug_print(f"Using standard synchronous execution path for: {agent_name}")
            try:
                try:
                    debug_print(f"Getting event loop")
                    loop = asyncio.get_event_loop()
                    debug_print(f"Got existing event loop: {loop}")
                except RuntimeError:
                    # No event loop in this thread, create a new one
                    debug_print(f"No event loop, creating new one")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    debug_print(f"Created new event loop: {loop}")
                
                # Run the coroutine on the event loop
                debug_print(f"Running coroutine on event loop for: {agent_name}")
                start_time = time.time()
                result = loop.run_until_complete(run_agent(state))
                elapsed = time.time() - start_time
                debug_print(f"Coroutine completed in {elapsed:.2f}s for: {agent_name}")
                return result
                
            except Exception as e:
                error_msg = f"Error in run_agent_sync for {agent_name}: {str(e)}"
                self.logger.error(error_msg)
                debug_print(error_msg, "ERROR")
                debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                
                # Propagate the original error if it's already a WorkflowError
                if isinstance(e, WorkflowError):
                    raise e
                # Otherwise wrap it in a WorkflowError
                raise WorkflowError(
                    error_msg,
                    agent_name=agent_name,
                    node=agent_name,
                    details=traceback.format_exc()
                )
                
        return run_agent_sync    
    
    def route_from_meta_agent(self, state: WorkflowState) -> str:
        """
        Centralized routing logic from meta_agent to next agent.
        All routing decisions are made here based on the current workflow phase and agent statuses.
        """
        debug_print(f"route_from_meta_agent called, current phase: {state.get('current_phase', 'UNKNOWN')}")
        
        # Check for errors - stop execution if any error is found
        if "error" in state and state["error"]:
            # Check if this is a RAG "No documents" error - not a critical error
            if "rag_agent" in state.get("goto", "") and "No documents loaded" in state["error"]:
                debug_print("Ignoring non-critical RAG error: No documents loaded", "WARNING")
                # Clear the error and continue the workflow
                state["error"] = None
            else:
                # For other errors, stop the workflow
                error_msg = f"Stopping workflow due to error: {state['error']}"
                debug_print(error_msg, "ERROR")
                # Stop the workflow by going to END
                return "END"
        
        # Check for explicit routing in the goto field
        goto = state.get("goto")
        debug_print(f"State goto field: {goto}")
        
        if goto in ["research_agent", "corporate_agent", "youtube_agent", "rag_agent", 
                   "analyst_agent", "writer_agent", "meta_agent_final", "END"]:
            debug_print(f"Routing to explicit goto: {goto}")
            # We'll now store the goto value for debugging but not clear it yet
            # This allows us to handle the logic for YouTube first and then clear
            current_goto = goto
            return current_goto
        
        # Handle phase transitions and routing based on current phase
        current_phase = state.get("current_phase", "RESEARCH")
        
        if current_phase == "RESEARCH":
            # In research phase, route to research agents
            # First check if phase is complete
            research_complete = (
                state.get("research_agent_status") == "DONE" and
                state.get("corporate_agent_status") == "DONE" and
                state.get("youtube_agent_status") == "DONE" and
                (state.get("rag_agent_status") == "DONE" or not state.get("enable_rag", True))
            )
            
            if research_complete:
                # All research agents done, transition to ANALYSIS phase
                debug_print("All research agents completed, transitioning to ANALYSIS phase")
                state["current_phase"] = "ANALYSIS"
                return "analyst_agent"
            
            # Get priorities from MetaAgent
            agent_priorities = {
                "research_agent": 60,
                "youtube_agent": 90,
                "corporate_agent": 80,
                "rag_agent": 70
            }
            
            # Create a list of incomplete agents
            incomplete_agents = []
            if state.get("research_agent_status") != "DONE":
                incomplete_agents.append("research_agent")
            if state.get("corporate_agent_status") != "DONE":
                incomplete_agents.append("corporate_agent")
            if state.get("youtube_agent_status") != "DONE":
                incomplete_agents.append("youtube_agent")
            if state.get("rag_agent_status") != "DONE" and state.get("enable_rag", True):
                incomplete_agents.append("rag_agent")
                
            # Sort agents by priority (higher first)
            sorted_agents = sorted(incomplete_agents, key=lambda x: agent_priorities.get(x, 0), reverse=True)
            
            if sorted_agents:
                next_agent = sorted_agents[0]
                debug_print(f"Routing to {next_agent} (priority: {agent_priorities.get(next_agent, 0)})")
                debug_print(f"Prioritized agent order: {sorted_agents}")
                return next_agent
                
            # If we get here, all agents are done or disabled
        
        elif current_phase == "ANALYSIS":
            # In analysis phase, route to analyst agent
            if state.get("analyst_agent_status") != "DONE":
                debug_print("Routing to analyst_agent")
                return "analyst_agent"
            else:
                # Analysis complete, transition to REPORT_GENERATION phase
                debug_print("Analysis complete, transitioning to REPORT_GENERATION phase")
                state["current_phase"] = "REPORT_GENERATION"
                return "writer_agent"
        
        elif current_phase == "REPORT_GENERATION":
            # In report generation phase, route to writer agent
            if state.get("writer_agent_status") != "DONE":
                debug_print("Routing to writer_agent")
                return "writer_agent"
            else:
                # Report generation complete, transition to COMPLETE phase
                debug_print("Report generation complete, transitioning to COMPLETE phase")
                state["current_phase"] = "COMPLETE"
                return "meta_agent_final"
        
        elif current_phase == "REPORT_REVIEW":
            # In report review phase, route to final meta agent
            debug_print("Routing to meta_agent_final based on REPORT_REVIEW phase")
            return "meta_agent_final"
        
        elif current_phase == "COMPLETE":
            # Workflow is complete
            debug_print("Routing to END based on COMPLETE phase")
            return "END"
        
        # Default to ending the workflow if phase is unknown
        debug_print(f"Unknown phase: {current_phase}, routing to END", "WARNING")
        self.logger.warning(f"Unknown phase: {current_phase}, ending workflow")
        return "END"
    
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow with the given initial state."""
        debug_print("Starting workflow.run")
        # Validate and prepare initial state
        prepared_state = self.prepare_initial_state(initial_state)
        
        # Log workflow start
        debug_print(f"Starting workflow for company: {prepared_state['company']}")
        self.logger.info(f"Starting workflow for company: {prepared_state['company']}")
        
        # Execute graph
        result = None
        current_node = "start"
        debug_print("Starting graph.stream")
        try:
            debug_print("Starting graph execution loop")
            for event in self.graph.stream(prepared_state):
                # Get state from the event - handle both formats from different LangGraph versions 
                if 'state' in event:
                    current_state = event['state']
                else:
                    # For newer LangGraph versions that return the state directly
                    current_state = event
                
                current_node = event.get('current_node', 'unknown')
                
                # Log progress
                debug_print(f"Executed node: {current_node}")
                self.logger.info(f"Executed node: {current_node}")
                
                # Check for errors
                if current_state.get("error"):
                    error_msg = f"Error in node {current_node}: {current_state['error']}"
                    self.logger.error(error_msg)
                    debug_print(error_msg, "ERROR")
                    # Stop execution by raising an exception
                    raise WorkflowError(error_msg, node=current_node)
                    
                # Check for user approval needed
                if current_state.get("requires_user_approval", False):
                    # This would normally wait for user input via API
                    debug_print(f"Requires user approval of type: {current_state.get('user_approval_type')}")
                    self.logger.info(f"Requires user approval of type: {current_state.get('user_approval_type')}")
                    
                    # In a real implementation, this would wait for user input via API
                    # For now, we just continue with auto-approval for demonstration
                    if "user_feedback" not in current_state:
                        current_state["user_feedback"] = {"approved": True}
                        debug_print("Auto-approved user approval request")
                
                # Save checkpoint if configured
                if self.config.get("checkpoint_path"):
                    debug_print(f"Saving checkpoint for node: {current_node}")
                    self._save_checkpoint(current_state, f"{current_node}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                    
                # Store final state
                result = current_state
                
            debug_print("Graph execution loop completed")
            
        except WorkflowError as e:
            error_msg = f"Workflow error: {str(e)}"
            self.logger.error(error_msg)
            debug_print(error_msg, "ERROR")
            
            # Create detailed error report
            error_report = {
                "error": str(e),
                "agent": getattr(e, "agent_name", "unknown"),
                "node": getattr(e, "node", current_node),
                "details": getattr(e, "details", traceback.format_exc()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to return a partial result if available
            if result is None:
                result = prepared_state
            
            # Add error information to the state
            result["error"] = str(e)
            result["error_report"] = error_report
            result["workflow_status"] = "ERROR"
            
            # Log detailed error information
            debug_print(f"Workflow error details: {json.dumps(error_report, indent=2)}", "ERROR")
            
        except Exception as e:
            error_msg = f"Unhandled error during graph execution: {str(e)}"
            self.logger.error(error_msg)
            debug_print(error_msg, "ERROR")
            debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
            
            # Try to return a partial result if available
            if result is None:
                result = prepared_state
            
            # Add error information to the state
            result["error"] = error_msg
            result["workflow_status"] = "ERROR"
            result["error_details"] = traceback.format_exc()
        
        # Log workflow completion
        self.logger.info(f"Workflow completed for company: {prepared_state['company']}")
        debug_print(f"Workflow completed for company: {prepared_state['company']}")
        
        return result
    
    def prepare_initial_state(self, initial_state: Dict[str, Any]) -> WorkflowState:
        """Prepare and validate the initial state."""
        debug_print("Preparing initial state")
        # Ensure required fields are present
        if "company" not in initial_state:
            error_msg = "Company name is required in initial state"
            debug_print(error_msg, "ERROR")
            raise ValueError(error_msg)
            
        # Initialize default values
        default_state = {
            "company": "",
            "industry": None,
            "meta_iteration": 0,
            "goto": None,
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
            "rag_results": {},
            "analysis_results": {},
            "quality_assessment": {},
            "analysis_guidance": {},
            "final_report": None,
            "report_sections": {},
            "top_events": [],
            "other_events": [],
            "executive_briefing": None,
            "rag_initialized": False,
            "enable_rag": True,
            "vector_store_dir": "vector_store",
            "additional_research_completed": False,
            "final_analysis_completed": False,
            "final_analysis_requested": False,
            "workflow_status": {},
            "user_approved": False,
            "requires_user_approval": False,
            "user_approval_type": None,
            "user_feedback": None,
            "agent_results": {},
            "analyst_tasks": [],
            "analyst_task_results": {},
            "execution_mode": "sequential",  # Changed from parallel to sequential
            "current_phase": "RESEARCH"  # Initial phase
        }
        
        # Merge with provided state, preferring provided values
        complete_state = {**default_state, **initial_state}
        debug_print(f"Initial state prepared for company: {complete_state['company']}")
        
        return complete_state
    
    def run_sync(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for running the workflow."""
        debug_print("run_sync called")
        
        # In an async context, just return the coroutine to be awaited by the caller
        try:
            current_task = asyncio.current_task()
            debug_print(f"Current asyncio task: {current_task}")
            
            if current_task is not None:
                debug_print("Already in async context, returning coroutine")
                return self.run(initial_state)
        except Exception as e:
            debug_print(f"Error checking async context: {str(e)}", "WARNING")
            debug_print(f"Traceback: {traceback.format_exc()}", "WARNING")
            
        # In a synchronous context, run with event loop
        debug_print("Using synchronous execution path")
        try:
            debug_print("Getting event loop")
            loop = asyncio.get_event_loop()
            debug_print(f"Got existing event loop: {loop}")
        except RuntimeError:
            # No event loop in this thread, create a new one
            debug_print("No event loop, creating new one")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            debug_print(f"Created new event loop: {loop}")
            
        # Check if loop is running
        if loop.is_running():
            debug_print("Event loop is already running, cannot use run_until_complete")
            # We're in a nested event loop scenario, just return the coroutine
            return self.run(initial_state)
        else:
            # We can safely run the event loop
            debug_print("Running coroutine on event loop")
            try:
                result = loop.run_until_complete(self.run(initial_state))
                debug_print("Coroutine completed successfully")
                return result
            except Exception as e:
                error_msg = f"Error in workflow execution: {str(e)}"
                debug_print(error_msg, "ERROR")
                debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                # Return a basic error state
                return {
                    **initial_state,
                    "error": error_msg,
                    "workflow_status": "ERROR",
                    "error_details": traceback.format_exc()
                }


# Function to create and run the workflow from arguments
async def create_and_run_workflow(
    company: str,
    industry: Optional[str] = None,
    config_path: Optional[str] = None,
    initial_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create and run a forensic workflow for the given company."""
    debug_print(f"create_and_run_workflow called for company: {company}, industry: {industry}")
    # Load configuration
    config = {}
    if config_path and os.path.exists(config_path):
        debug_print(f"Loading config from: {config_path}")
        try:
            config = load_config_with_env_vars(config_path)
            debug_print(f"Config loaded successfully with {len(config)} keys")
                        
            # Debug statements for SerpAPI key
            if 'research' in config:
                print(f"DEBUG WORKFLOW: Research config keys: {list(config.get('research', {}).keys())}")
                print(f"DEBUG WORKFLOW: SerpAPI key present: {'api_key' in config.get('research', {})}")
                if 'api_key' in config.get('research', {}):
                    api_key = config['research']['api_key']
                    print(f"DEBUG WORKFLOW: SerpAPI key first few chars: {api_key[:5]}..." if api_key else "EMPTY")
        except Exception as e:
            debug_print(f"Error loading config: {str(e)}", "ERROR")
            debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
            raise WorkflowError(f"Failed to load config: {str(e)}", details=traceback.format_exc())
            
    # Setup logging
    debug_print(f"Setting up logging with level: {config.get('log_level', 'INFO')}")
    setup_logging("forensic_workflow", level=config.get("log_level", "INFO"))
    logger = get_logger("forensic_workflow")
    
    logger.info(f"Creating workflow for company: {company}")
    
    # Simplified analysis when RAG is disabled
    if initial_state and initial_state.get("enable_rag") is False:
        debug_print("RAG disabled - returning simplified analysis")
        logger.info("RAG disabled - returning simplified analysis")
        return {
            "final_report": f"# Financial Analysis for {company}\n\n"
                           f"This is a simplified analysis report for {company} in the {industry or 'unknown'} industry.\n\n"
                           f"## Note\n\nThis is a basic report generated without using the full workflow.",
            "top_events": [],
            "analysis_results": {
                "red_flags": [],
                "forensic_insights": {}
            }
        }
    
    # Create workflow
    debug_print(f"Creating EnhancedForensicWorkflow instance")
    try:
        workflow = EnhancedForensicWorkflow(config)
        debug_print(f"EnhancedForensicWorkflow instance created successfully")
    except Exception as e:
        error_msg = f"Error creating workflow: {str(e)}"
        logger.error(error_msg)
        debug_print(error_msg, "ERROR")
        debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
        raise WorkflowError(error_msg, details=traceback.format_exc())
    
    # Prepare initial state if not provided
    if initial_state is None:
        debug_print("Creating default initial state")
        initial_state = {
            "company": company,
            "industry": industry,
            "enable_rag": True,
            "vector_store_dir": "vector_store",
            "rag_initialized": False
        }
    else:
        debug_print("Using provided initial state")
        # Ensure company and industry are set in the initial state
        if "company" not in initial_state:
            initial_state["company"] = company
            debug_print(f"Added company to initial state: {company}")
        if "industry" not in initial_state and industry is not None:
            initial_state["industry"] = industry
            debug_print(f"Added industry to initial state: {industry}")
    
    # Run workflow - use a coroutine-friendly approach
    logger.info(f"Running workflow for company: {company}")
    debug_print(f"Running workflow for company: {company}")
    
    # We're already in an async context, just await the run method
    debug_print("Awaiting workflow.run")
    result = await workflow.run(initial_state)
    debug_print("workflow.run completed")
    
    logger.info(f"Workflow completed for company: {company}")
    debug_print(f"Workflow completed for company: {company}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')
    debug_print(f"Loading environment variables from: {DOTENV_PATH}")
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    debug_print("Script started as main")
    debug_print(f"Serpapi key present: {os.getenv('SERPAPI_API_KEY')}")
    
    parser = argparse.ArgumentParser(description='Run Financial Forensic Analysis')
    parser.add_argument('--company', type=str, required=True, help='Company name to analyze')
    parser.add_argument('--industry', type=str, help='Industry of the company')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--enable_rag', type=str, default='True', help='Enable RAG (True/False)')
    parser.add_argument('--debug', action='store_true', help='Enable additional debug output')
    
    args = parser.parse_args()
    debug_print(f"Command line args: {args}")
    
    # Create initial state with RAG setting
    initial_state = {
        "company": args.company,
        "industry": args.industry,
        "enable_rag": args.enable_rag.lower() == 'true',
        "vector_store_dir": "vector_store" 
    }
    
    debug_print(f"Creating asyncio event loop")
    try:
        # Create event loop and run the workflow
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        debug_print(f"Running create_and_run_workflow in event loop")
        result = loop.run_until_complete(create_and_run_workflow(
            company=args.company,
            industry=args.industry,
            config_path=args.config,
            initial_state=initial_state
        ))
        debug_print(f"create_and_run_workflow completed")
    except WorkflowError as e:
        debug_print(f"Workflow error: {str(e)}", "ERROR")
        debug_print(f"Agent: {getattr(e, 'agent_name', 'unknown')}", "ERROR")
        debug_print(f"Node: {getattr(e, 'node', 'unknown')}", "ERROR")
        if hasattr(e, 'details') and e.details:
            debug_print(f"Details: {e.details}", "ERROR")
        result = {
            "error": f"Workflow error: {str(e)}",
            "error_agent": getattr(e, 'agent_name', 'unknown'),
            "error_node": getattr(e, 'node', 'unknown'),
            "error_details": getattr(e, 'details', None),
            "company": args.company,
            "industry": args.industry
        }
    except Exception as e:
        debug_print(f"Error in main execution: {str(e)}", "ERROR")
        debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
        result = {
            "error": f"Workflow execution failed: {str(e)}",
            "error_details": traceback.format_exc(),
            "company": args.company,
            "industry": args.industry
        }
    
    # Print summary of results
    print("\n" + "="*80)
    print(f"Analysis completed for: {args.company}")
    print("="*80)
    
    # Check for errors
    if "error" in result and result["error"]:
        print(f"\nERROR: {result['error']}")
        if "error_agent" in result:
            print(f"Agent: {result['error_agent']}")
        if "error_node" in result:
            print(f"Node: {result['error_node']}")
        if "error_details" in result and result["error_details"]:
            print("\nError Details:")
            print("-"*40)
            print(result["error_details"])
            print("-"*40)
    
    elif result.get("final_report"):
        print(f"Report generated with {len(result.get('final_report'))} characters")
        
        # Save report to file
        output_file = f"{args.company.replace(' ', '_')}_report.md"
        with open(output_file, 'w') as f:
            f.write(result["final_report"])
        print(f"Report saved to: {output_file}")
        
    else:
        print("No report generated. Check logs for errors.")