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


class ResearchPool(BaseAgent):
    name = "research_pool"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = get_logger(self.name)
        
        # Initialize component agents
        debug_print("Initializing Research Agent")
        self.research_agent = ResearchAgent(config)
        debug_print("Initializing YouTube Agent")
        self.youtube_agent = YouTubeAgent(config)
        debug_print("Initializing Corporate Agent")
        self.corporate_agent = CorporateAgent(config)
        debug_print("Initializing RAG Agent")
        self.rag_agent = RAGAgent(config)
        
        # Configuration parameters
        self.max_parallel_agents = config.get("workflow", {}).get("max_parallel_agents", 3)
        debug_print(f"ResearchPool initialized with max_parallel_agents={self.max_parallel_agents}")
    
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Implements the abstract _execute method required by BaseAgent."""
        debug_print(f"ResearchPool._execute called for company: {state.get('company', 'unknown')}")
        return await self.run(state)
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the research pool with parallel execution of multiple agents.
        This pool manages ResearchAgent, YouTubeAgent, and CorporateAgent.
        """
        debug_print(f"ResearchPool.run started for company: {state.get('company', 'unknown')}")
        self._log_start(state)
        
        company = state.get("company", "")
        if not company:
            error_msg = "Company name is missing!"
            self.logger.error(error_msg)
            debug_print("ERROR: Company name is missing in ResearchPool.run", "ERROR")
            raise WorkflowError(error_msg, agent_name=self.name)
        
        self.logger.info(f"Starting research pool for {company}")
        
        # Prepare a results container
        pool_results = {
            "research_results": {},
            "corporate_results": {},
            "youtube_results": {},
            "rag_results": {},
            "event_metadata": {}
        }
        
        # Determine which agents to run in the new specified order
        agents_to_run = []
        
        # 1. Run corporate agent if company information is needed and not already present
        if not state.get("corporate_results"):
            agents_to_run.append(self.corporate_agent)
            debug_print(f"Added corporate_agent to agents_to_run")
        
        # 2. Run YouTube agent if video research is needed and not already present
        if not state.get("youtube_results"):
            agents_to_run.append(self.youtube_agent)
            debug_print(f"Added youtube_agent to agents_to_run")
            
        # 3. Run RAG agent if enabled and not already processed
        if not state.get("rag_results") and state.get("enable_rag", True):
            debug_print(f"RAG is enabled: {state.get('enable_rag', True)}")
            # Initialize RAG agent if needed
            if not state.get("rag_initialized"):
                debug_print("Initializing RAG agent")
                # Prepare state for RAG agent
                rag_state = {
                    "command": "initialize",
                    "vector_store_dir": state.get("vector_store_dir", "vector_store"),
                    "company": state.get('company', "Unknown")
                }
                
                # Initialize RAG agent
                try:
                    debug_print(f"Calling rag_agent.run with state: {rag_state}")
                    init_result = await self.rag_agent.run(rag_state)
                    debug_print(f"RAG initialization result: {init_result.get('initialized', False)}")
                    if init_result.get("initialized", False):
                        state["rag_initialized"] = True
                        self.logger.info("RAG agent initialized successfully")
                    else:
                        error_msg = f"RAG agent initialization failed: {init_result.get('error')}"
                        self.logger.error(error_msg)
                        debug_print(error_msg, "ERROR")
                        raise WorkflowError(error_msg, agent_name="rag_agent")
                except Exception as e:
                    error_msg = f"Error initializing RAG agent: {str(e)}"
                    self.logger.error(error_msg)
                    debug_print(error_msg, "ERROR")
                    debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                    raise WorkflowError(error_msg, agent_name="rag_agent", details=traceback.format_exc())
            
            # Add RAG agent to the pool if initialized
            if state.get("rag_initialized", False):
                agents_to_run.append(self.rag_agent)
                debug_print(f"Added rag_agent to agents_to_run")
        
        # 4. Always add the research agent last
        agents_to_run.append(self.research_agent)
        debug_print(f"Added research_agent to agents_to_run")
        
        # Run agents concurrently
        debug_print(f"Running {len(agents_to_run)} research agents concurrently")
        self.logger.info(f"Running {len(agents_to_run)} research agents concurrently")
        
        # Create a ThreadPoolExecutor for parallel execution
        debug_print(f"Creating ThreadPoolExecutor with max_workers={self.max_parallel_agents}")
        with ThreadPoolExecutor(max_workers=self.max_parallel_agents) as executor:
            # Function to process an agent
            def process_agent(agent):
                debug_print(f"process_agent started for agent: {agent.name}")
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Special handling for RAG agent
                    if agent.name == "rag_agent":
                        debug_print(f"Preparing special handling for RAG agent")
                        # Prepare RAG query based on company and research context
                        rag_query = f"Provide key information and risk factors about {company}"
                        if state.get("industry"):
                            rag_query += f" in the {state['industry']} industry"
                            
                        # Create RAG state
                        rag_state = {
                            **state,
                            "command": "query",
                            "query": rag_query
                        }
                        
                        # Add special error handling for RAG agent
                        try:
                            debug_print(f"Calling rag_agent.run with query: {rag_query}")
                            agent_result = loop.run_until_complete(agent.run(rag_state))
                            debug_print(f"RAG agent run completed")
                            
                            # Check for "no documents" error and convert to a non-error state
                            if agent_result.get("rag_status") == "ERROR" and "No documents loaded" in agent_result.get("error", ""):
                                debug_print("RAG agent 'No documents loaded' is non-critical - converting to success state")
                                # Convert to a non-error result
                                agent_result = {
                                    **agent_result,
                                    "rag_status": "NO_DOCUMENTS",
                                    "error": None,  # Clear error
                                    "retrieval_results": {
                                        "success": False,
                                        "message": "No documents available in vector store"
                                    },
                                    "response": f"No document-based information is available for {company}. Analysis will be based on other sources."
                                }
                        except Exception as e:
                            debug_print(f"Caught error in RAG agent execution: {str(e)}", "WARNING")
                            
                            # Check if this is a "no documents" error that we can handle
                            if "No documents loaded" in str(e):
                                debug_print("RAG agent 'No documents loaded' is non-critical - creating fallback result")
                                agent_result = {
                                    "rag_status": "NO_DOCUMENTS",
                                    "error": None,  # Important: don't set an error
                                    "retrieval_results": {
                                        "success": False,
                                        "message": f"No documents available: {str(e)}"
                                    },
                                    "response": f"No document-based information is available for {company}. Analysis will be based on other sources."
                                }
                            else:
                                # For other types of errors, re-raise
                                raise
                    else:
                        # Run other agents normally
                        debug_print(f"Calling {agent.name}.run")
                        agent_result = loop.run_until_complete(agent.run(state))
                        debug_print(f"{agent.name} run completed")
                    
                    loop.close()
                    debug_print(f"process_agent completed for agent: {agent.name}")
                    
                    # Check for errors in the result and raise an exception
                    # Modified to ignore non-critical RAG errors
                    if "error" in agent_result and agent_result["error"] and not (
                        agent.name == "rag_agent" and 
                        agent_result.get("rag_status") in ["NO_DOCUMENTS", "NO_VECTOR_STORE"]
                    ):
                        error_msg = f"Error in {agent.name}: {agent_result['error']}"
                        debug_print(error_msg, "ERROR")
                        raise WorkflowError(error_msg, agent_name=agent.name)
                        
                    return agent.name, agent_result
                except Exception as e:
                    loop.close()
                    error_msg = f"Error running {agent.name}: {str(e)}"
                    self.logger.error(error_msg)
                    debug_print(error_msg, "ERROR")
                    debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                    
                    # Special case for RAG agent "No documents" errors - don't propagate these
                    if agent.name == "rag_agent" and "No documents loaded" in str(e):
                        debug_print("RAG agent 'No documents loaded' is non-critical - creating fallback result", "WARNING")
                        fallback_result = {
                            "rag_status": "NO_DOCUMENTS",
                            "error": None,  # Important: don't set an error
                            "retrieval_results": {
                                "success": False,
                                "message": f"No documents available: {str(e)}"
                            },
                            "response": f"No document-based information is available for {company}. Analysis will be based on other sources."
                        }
                        return agent.name, fallback_result
                    
                    # Propagate the original error if it's already a WorkflowError
                    if isinstance(e, WorkflowError):
                        raise e
                    # Otherwise wrap it in a WorkflowError
                    raise WorkflowError(
                        f"Error in {agent.name}: {str(e)}", 
                        agent_name=agent.name,
                        details=traceback.format_exc()
                    )            
            # Submit all agents to the thread pool
            debug_print(f"Submitting {len(agents_to_run)} agents to thread pool")
            future_results = {executor.submit(process_agent, agent): agent for agent in agents_to_run}
            
            # Process completed agents as they finish
            debug_print(f"Processing results as agents complete")
            for future in future_results:
                try:
                    debug_print(f"Getting result for a completed agent")
                    agent_name, agent_result = future.result()
                    debug_print(f"Got result for agent: {agent_name}")
                    
                    # Extract and merge results based on agent type
                    if agent_name == "research_agent":
                        if "research_results" in agent_result:
                            pool_results["research_results"] = agent_result["research_results"]
                            debug_print(f"Extracted research_results with {len(agent_result['research_results'])} items")
                        if "event_metadata" in agent_result:
                            pool_results["event_metadata"] = agent_result["event_metadata"]
                            debug_print(f"Extracted event_metadata")
                        
                    elif agent_name == "corporate_agent":
                        if "corporate_results" in agent_result:
                            pool_results["corporate_results"] = agent_result["corporate_results"]
                            debug_print(f"Extracted corporate_results")
                            
                    elif agent_name == "youtube_agent":
                        if "youtube_results" in agent_result:
                            pool_results["youtube_results"] = agent_result["youtube_results"]
                            debug_print(f"Extracted youtube_results")
                    
                    elif agent_name == "rag_agent":
                        # Process RAG agent results
                        if agent_result.get("rag_status") == "RESPONSE_READY":
                            # Store RAG results
                            pool_results["rag_results"] = {
                                "response": agent_result.get("response", ""),
                                "retrieval_results": agent_result.get("retrieval_results", {}),
                                "query": agent_result.get("query", "")
                            }
                            debug_print(f"Extracted rag_results")
                    
                except Exception as e:
                    error_msg = f"Error processing agent result: {str(e)}"
                    self.logger.error(error_msg)
                    debug_print(error_msg, "ERROR")
                    debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                    
                    # Propagate the original error if it's already a WorkflowError
                    if isinstance(e, WorkflowError):
                        raise e
                    # Otherwise wrap it in a WorkflowError
                    raise WorkflowError(
                        error_msg, 
                        agent_name=future_results[future].name,
                        details=traceback.format_exc()
                    )
        
        # Combine all results
        debug_print(f"Combining results from all agents")
        updated_state = {**state}
        
        # Update state with pool results
        for key, value in pool_results.items():
            if value:  # Only update if we have results
                updated_state[key] = value
                debug_print(f"Updated state with {key}")
        
        # Set status for all agents
        updated_state["research_agent_status"] = "DONE" if "research_results" in pool_results else state.get("research_agent_status", "UNKNOWN")
        updated_state["corporate_agent_status"] = "DONE" if "corporate_results" in pool_results else state.get("corporate_agent_status", "UNKNOWN")
        updated_state["youtube_agent_status"] = "DONE" if "youtube_results" in pool_results else state.get("youtube_agent_status", "UNKNOWN")
        updated_state["rag_agent_status"] = "DONE" if "rag_results" in pool_results else state.get("rag_agent_status", "UNKNOWN")
        
        # Return to meta_agent for next steps
        updated_state["goto"] = "meta_agent"
        debug_print(f"ResearchPool.run completed, returning to meta_agent")
        
        return updated_state

class AnalystPool(BaseAgent):
    name = "analyst_pool"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = get_logger(self.name)
        
        # Initialize the analyst agent
        debug_print("Initializing Analyst Agent")
        self.analyst_agent = AnalystAgent(config)
        
        # Configuration parameters
        self.max_workers = config.get("workflow", {}).get("analyst_pool_size", 5)
        debug_print(f"AnalystPool initialized with max_workers={self.max_workers}")
        
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Implements the abstract _execute method required by BaseAgent."""
        debug_print(f"AnalystPool._execute called for company: {state.get('company', 'unknown')}")
        return await self.run(state)
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the analyst pool to process analytical tasks in parallel.
        This pool manages analytical tasks using multiple analyst agent instances.
        """
        debug_print(f"AnalystPool.run started for company: {state.get('company', 'unknown')}")
        self._log_start(state)
        
        company = state.get("company", "")
        if not company:
            error_msg = "Company name is missing!"
            self.logger.error(error_msg)
            debug_print("ERROR: Company name is missing in AnalystPool.run", "ERROR")
            raise WorkflowError(error_msg, agent_name=self.name)
        
        # Get analyst tasks
        tasks = state.get("analyst_tasks", [])
        
        # If no tasks are specified but we have research results, create tasks from events
        if not tasks and "research_results" in state:
            debug_print("No analyst tasks provided, creating from research results")
            research_results = state.get("research_results", {})
            tasks = self._create_tasks_from_research(research_results)
            self.logger.info(f"Created {len(tasks)} analyst tasks from research results")
            debug_print(f"Created {len(tasks)} analyst tasks from research results")
        
        if not tasks:
            error_msg = "No analyst tasks to process"
            self.logger.error(error_msg)
            debug_print("ERROR: No analyst tasks to process", "ERROR")
            raise WorkflowError(error_msg, agent_name=self.name)
        
        # Initialize results
        results = {}
        
        # Run tasks in parallel with thread pool
        self.logger.info(f"Processing {len(tasks)} analyst tasks with up to {self.max_workers} workers")
        debug_print(f"Processing {len(tasks)} analyst tasks with up to {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Function to process a task
            def process_task(task):
                debug_print(f"process_task started for event: {task.get('event_name', 'unknown')}")
                event_name = task.get("event_name")
                event_data = task.get("event_data")
                
                # Skip invalid tasks
                if not event_name or not event_data:
                    error_msg = f"Invalid task: missing event_name or event_data"
                    debug_print(error_msg, "ERROR")
                    raise WorkflowError(error_msg, agent_name=self.name)
                
                # Create a new analyst agent for this task (to avoid state conflicts)
                debug_print(f"Creating new AnalystAgent for task: {event_name}")
                task_agent = AnalystAgent(self.config)
                
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Process the event
                    debug_print(f"Calling process_event for: {event_name}")
                    task_result = loop.run_until_complete(task_agent.process_event(
                        company, event_name, event_data
                    ))
                    debug_print(f"process_event completed for: {event_name}")
                    loop.close()
                    return event_name, task_result
                except Exception as e:
                    loop.close()
                    error_msg = f"Error processing event {event_name}: {str(e)}"
                    debug_print(error_msg, "ERROR")
                    debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                    
                    # Propagate the original error if it's already a WorkflowError
                    if isinstance(e, WorkflowError):
                        raise e
                    # Otherwise wrap it in a WorkflowError
                    raise WorkflowError(
                        error_msg,
                        agent_name="analyst_agent",
                        details=traceback.format_exc()
                    )
            
            # Submit all tasks to the thread pool
            debug_print(f"Submitting {len(tasks)} tasks to thread pool")
            future_results = {executor.submit(process_task, task): task for task in tasks}
            
            # Process completed tasks as they finish
            debug_print(f"Processing results as tasks complete")
            for future in future_results:
                try:
                    debug_print(f"Getting result for a completed task")
                    event_name, task_result = future.result()
                    if event_name:  # Skip results with no event name
                        debug_print(f"Got result for event: {event_name}")
                        
                        # Check for errors in the task result
                        if isinstance(task_result, dict) and "error" in task_result and task_result["error"]:
                            error_msg = f"Error in task for event {event_name}: {task_result['error']}"
                            debug_print(error_msg, "ERROR")
                            raise WorkflowError(error_msg, agent_name=self.name)
                            
                        results[event_name] = task_result
                except Exception as e:
                    error_msg = f"Error in task execution: {str(e)}"
                    self.logger.error(error_msg)
                    debug_print(error_msg, "ERROR")
                    debug_print(f"Traceback: {traceback.format_exc()}", "ERROR")
                    
                    # Propagate the original error if it's already a WorkflowError
                    if isinstance(e, WorkflowError):
                        raise e
                    # Otherwise wrap it in a WorkflowError
                    raise WorkflowError(
                        error_msg,
                        agent_name=self.name,
                        details=traceback.format_exc()
                    )
        
        # Combine the results into a structured analysis result
        debug_print(f"Combining results from {len(results)} tasks")
        analysis_results = self._combine_analysis_results(results, state)
        
        # Update state with analysis results
        updated_state = {
            **state,
            "goto": "meta_agent",
            "analyst_agent_status": "DONE",
            "analysis_results": analysis_results,
            "analyst_task_results": results
        }
        
        self.logger.info(f"Analyst pool completed processing {len(tasks)} tasks")
        debug_print(f"AnalystPool.run completed, processed {len(tasks)} tasks")
        return updated_state
    
    def _create_tasks_from_research(self, research_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Create analysis tasks from research results."""
        debug_print(f"Creating tasks from research results with {len(research_results)} events")
        tasks = []
        
        for event_name, event_articles in research_results.items():
            # Skip events with no articles
            if not event_articles:
                debug_print(f"Skipping event with no articles: {event_name}")
                continue
                
            # Create a task for this event
            debug_print(f"Creating task for event: {event_name} with {len(event_articles)} articles")
            tasks.append({
                "event_name": event_name,
                "event_data": event_articles,
                "analysis_type": "standard"
            })
        
        return tasks
    
    def _combine_analysis_results(self, task_results: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Combine individual task results into a comprehensive analysis result."""
        debug_print(f"Combining analysis results from {len(task_results)} tasks")
        combined_results = {
            "event_synthesis": {},
            "forensic_insights": {},
            "timeline": [],
            "red_flags": [],
            "entity_network": {},
            "rag_insights": {}
        }
        
        # Merge existing analysis results if available
        if "analysis_results" in state and state["analysis_results"]:
            debug_print(f"Merging with existing analysis results")
            for key in combined_results.keys():
                if key in state["analysis_results"]:
                    combined_results[key] = state["analysis_results"][key]
        
        # Add task results
        for event_name, result in task_results.items():
            # Skip failed tasks
            if "error" in result and result["error"]:
                error_msg = f"Error in task result for {event_name}: {result['error']}"
                debug_print(error_msg, "ERROR")
                raise WorkflowError(error_msg, agent_name=self.name)
                
            # Add to event synthesis
            if "event_synthesis" in result:
                combined_results["event_synthesis"][event_name] = result["event_synthesis"]
                debug_print(f"Added event synthesis for: {event_name}")
                
            # Add to forensic insights
            if "forensic_insights" in result:
                combined_results["forensic_insights"][event_name] = result["forensic_insights"]
                debug_print(f"Added forensic insights for: {event_name}")
                
            # Add to timeline events
            if "timeline" in result:
                timeline_items = 0
                for event in result.get("timeline", []):
                    if isinstance(event, dict) and "date" in event and "event" in event:
                        combined_results["timeline"].append(event)
                        timeline_items += 1
                debug_print(f"Added {timeline_items} timeline items for: {event_name}")
                        
            # Add to red flags
            if "red_flags" in result:
                flags_added = 0
                for flag in result.get("red_flags", []):
                    if flag and flag not in combined_results["red_flags"]:
                        combined_results["red_flags"].append(flag)
                        flags_added += 1
                debug_print(f"Added {flags_added} red flags for: {event_name}")
            
            # Add entity information if available
            if "entity_network" in result:
                entities = len(result.get("entity_network", {}))
                combined_results["entity_network"].update(result.get("entity_network", {}))
                debug_print(f"Added {entities} entities for: {event_name}")
        
        # Sort timeline by date if possible
        try:
            debug_print(f"Sorting timeline with {len(combined_results['timeline'])} items")
            combined_results["timeline"] = sorted(
                combined_results["timeline"],
                key=lambda x: datetime.strptime(x.get("date", "2000-01-01"), "%Y-%m-%d")
            )
            debug_print(f"Timeline sorted successfully")
        except Exception as e:
            # If sorting fails (e.g., due to date format), don't sort
            error_msg = f"Error sorting timeline: {str(e)}"
            debug_print(error_msg, "WARNING")
            debug_print(f"Traceback: {traceback.format_exc()}", "WARNING")
            # We don't need to raise an exception here since sorting failure
            # isn't critical to the overall process
        
        # Integrate RAG results if available
        if "rag_results" in state and state["rag_results"]:
            debug_print(f"Integrating RAG results")
            rag_response = state["rag_results"].get("response", "")
            if rag_response:
                # Add RAG insights to the combined results
                combined_results["rag_insights"] = {
                    "response": rag_response,
                    "query": state["rag_results"].get("query", ""),
                    "sources": []
                }
                debug_print(f"Added RAG response with {len(rag_response)} characters")
                
                # Extract source information
                retrieval_results = state["rag_results"].get("retrieval_results", {})
                sources_added = 0
                for result in retrieval_results.get("results", []):
                    if result.get("metadata"):
                        source_info = {
                            "source": result.get("metadata", {}).get("source", "Unknown"),
                            "page": result.get("metadata", {}).get("page", "Unknown"),
                            "relevance": result.get("score", 0)
                        }
                        combined_results["rag_insights"]["sources"].append(source_info)
                        sources_added += 1
                debug_print(f"Added {sources_added} RAG sources")
                
                # Extract any additional red flags from RAG response
                if rag_response and "red flag" in rag_response.lower():
                    flags_added = 0
                    # Simple extraction of lines containing "red flag"
                    for line in rag_response.split("\n"):
                        if "red flag" in line.lower() and len(line) > 15:  # Ensure it's a meaningful line
                            flag = line.strip()
                            if flag not in combined_results["red_flags"]:
                                combined_results["red_flags"].append(flag)
                                flags_added += 1
                    debug_print(f"Extracted {flags_added} red flags from RAG response")
        
        return combined_results


class EnhancedForensicWorkflow(BaseGraph):
    """
    Enhanced workflow implementing a true agent-based architecture with centralized orchestration.
    This refactored version follows the simplified design with agent pools and MetaAgent orchestration.
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
            # Initialize agents and pools
            debug_print("Initializing MetaAgent")
            self.meta_agent = MetaAgent(self.config)
            debug_print("Initializing ResearchPool")
            self.research_pool = ResearchPool(self.config)
            debug_print("Initializing AnalystPool")
            self.analyst_pool = AnalystPool(self.config)
            debug_print("Initializing WriterAgent")
            self.writer_agent = WriterAgent(self.config)
            
            # Initialize agent mapping
            debug_print("Setting up agent mapping")
            self.agents = {
                "meta_agent": self.meta_agent,
                "research_pool": self.research_pool,
                "analyst_pool": self.analyst_pool,
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
        Build the enhanced workflow graph with a true agent-based architecture.
        This simplified design has MetaAgent as the central orchestrator with agent pools.
        """
        debug_print("Starting build_graph")
        try:
            # State and workflow settings
            debug_print("Creating StateGraph")
            workflow = StateGraph(WorkflowState)
            debug_print("StateGraph created successfully")
            
            # Add nodes for each agent/pool
            debug_print(f"Adding {len(self.agents)} nodes to graph")
            for agent_name, agent in self.agents.items():
                debug_print(f"Adding node: {agent_name}")
                workflow.add_node(agent_name, self.create_agent_node(agent, agent_name))
            
            # Set entry point to meta_agent
            debug_print("Setting entry point to meta_agent")
            workflow.set_entry_point("meta_agent")
            
            # MetaAgent is the central orchestration point
            # It connects to all agent pools and receives control back after each pool completes
            
            # Connect meta_agent to other agents/pools based on routing logic
            debug_print("Adding conditional edges from meta_agent")
            workflow.add_conditional_edges(
                "meta_agent",
                self.route_from_meta_agent,
                {
                    "research_pool": "research_pool",
                    "analyst_pool": "analyst_pool",
                    "writer_agent": "writer_agent",
                    "meta_agent_final": "meta_agent_final",
                    "END": END
                }
            )
            
            # All other agents/pools return to meta_agent
            debug_print("Adding edges from pools back to meta_agent")
            workflow.add_edge("research_pool", "meta_agent")
            workflow.add_edge("analyst_pool", "meta_agent")
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
        Centralized routing logic from meta_agent to next node.
        All routing decisions are now made in the MetaAgent based on the current phase.
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
        
        if goto in ["research_pool", "analyst_pool", "writer_agent", "meta_agent_final", "END"]:
            debug_print(f"Routing to explicit goto: {goto}")
            return goto
        
        # Check current phase to determine routing
        current_phase = state.get("current_phase", "RESEARCH")
        
        if current_phase == "RESEARCH":
            # In research phase, route to research pool
            debug_print("Routing to research_pool based on RESEARCH phase")
            return "research_pool"
        
        elif current_phase == "ANALYSIS":
            # In analysis phase, route to analyst pool
            debug_print("Routing to analyst_pool based on ANALYSIS phase")
            return "analyst_pool"
        
        elif current_phase == "REPORT_GENERATION":
            # In report generation phase, route to writer agent
            debug_print("Routing to writer_agent based on REPORT_GENERATION phase")
            return "writer_agent"
        
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
            "execution_mode": "parallel",
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