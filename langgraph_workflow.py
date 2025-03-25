import os
import json
import asyncio
from typing import Dict, List, Any, Tuple, Optional, Union, Annotated, TypedDict, Literal, Set, Callable, Type
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph

from base.base_agents import BaseAgent
from base.base_graph import BaseGraph, GraphConfig
from utils.logging import get_logger, setup_logging
from utils.llm_provider import init_llm_provider, get_llm_provider
from utils.prompt_manager import init_prompt_manager

# Import agents
from agents.meta_agent import MetaAgent
from agents.research_agent import ResearchAgent
from agents.youtube_agent import YouTubeAgent
from agents.corporate_agent import CorporateAgent
from agents.analyst_agent import AnalystAgent
from agents.rag_agent import RAGAgent
from agents.writer_agent import WriterAgent


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
        self.research_agent = ResearchAgent(config)
        self.youtube_agent = YouTubeAgent(config)
        self.corporate_agent = CorporateAgent(config)
        self.rag_agent = RAGAgent(config)
        
        # Configuration parameters
        self.max_parallel_agents = config.get("workflow", {}).get("max_parallel_agents", 3)
        
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Implements the abstract _execute method required by BaseAgent."""
        return await self.run(state)
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the research pool with parallel execution of multiple agents.
        This pool manages ResearchAgent, YouTubeAgent, and CorporateAgent.
        """
        self._log_start(state)
        
        company = state.get("company", "")
        if not company:
            self.logger.error("Company name is missing!")
            return {**state, "goto": "meta_agent", "error": "Company name is missing"}
        
        self.logger.info(f"Starting research pool for {company}")
        
        # Prepare a results container
        pool_results = {
            "research_results": {},
            "corporate_results": {},
            "youtube_results": {},
            "rag_results": {},
            "event_metadata": {}
        }
        
        # Determine which agents to run
        agents_to_run = []
        
        # Always run the research agent
        agents_to_run.append(self.research_agent)
        
        # Run corporate agent if company information is needed and not already present
        if not state.get("corporate_results"):
            agents_to_run.append(self.corporate_agent)
        
        # Run YouTube agent if video research is needed and not already present
        if not state.get("youtube_results"):
            agents_to_run.append(self.youtube_agent)
            
        # Run RAG agent if enabled and not already processed
        if not state.get("rag_results") and state.get("enable_rag", True):
            # Initialize RAG agent if needed
            if not state.get("rag_initialized"):
                # Prepare state for RAG agent
                rag_state = {
                    "command": "initialize",
                    "vector_store_dir": state.get("vector_store_dir", "vector_store")
                }
                
                # Initialize RAG agent
                try:
                    init_result = await self.rag_agent.run(rag_state)
                    if init_result.get("initialized", False):
                        state["rag_initialized"] = True
                        self.logger.info("RAG agent initialized successfully")
                    else:
                        self.logger.warning(f"RAG agent initialization failed: {init_result.get('error')}")
                except Exception as e:
                    self.logger.error(f"Error initializing RAG agent: {str(e)}")
            
            # Add RAG agent to the pool if initialized
            if state.get("rag_initialized", False):
                agents_to_run.append(self.rag_agent)
        
        # Run agents concurrently
        self.logger.info(f"Running {len(agents_to_run)} research agents concurrently")
        
        # Create a ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_parallel_agents) as executor:
            # Function to process an agent
            def process_agent(agent):
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Special handling for RAG agent
                    if agent.name == "rag_agent":
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
                        agent_result = loop.run_until_complete(agent.run(rag_state))
                    else:
                        # Run other agents normally
                        agent_result = loop.run_until_complete(agent.run(state))
                    
                    loop.close()
                    return agent.name, agent_result
                except Exception as e:
                    loop.close()
                    self.logger.error(f"Error running {agent.name}: {str(e)}")
                    # Return error state
                    return agent.name, {
                        **state,
                        "goto": "meta_agent",
                        "error": f"Error in {agent.name}: {str(e)}",
                        f"{agent.name}_status": "ERROR"
                    }
            
            # Submit all agents to the thread pool
            future_results = {executor.submit(process_agent, agent): agent for agent in agents_to_run}
            
            # Process completed agents as they finish
            for future in future_results:
                try:
                    agent_name, agent_result = future.result()
                    
                    # Extract and merge results based on agent type
                    if agent_name == "research_agent":
                        if "research_results" in agent_result:
                            pool_results["research_results"] = agent_result["research_results"]
                        if "event_metadata" in agent_result:
                            pool_results["event_metadata"] = agent_result["event_metadata"]
                        
                    elif agent_name == "corporate_agent":
                        if "corporate_results" in agent_result:
                            pool_results["corporate_results"] = agent_result["corporate_results"]
                            
                    elif agent_name == "youtube_agent":
                        if "youtube_results" in agent_result:
                            pool_results["youtube_results"] = agent_result["youtube_results"]
                    
                    elif agent_name == "rag_agent":
                        # Process RAG agent results
                        if agent_result.get("rag_status") == "RESPONSE_READY":
                            # Store RAG results
                            pool_results["rag_results"] = {
                                "response": agent_result.get("response", ""),
                                "retrieval_results": agent_result.get("retrieval_results", {}),
                                "query": agent_result.get("query", "")
                            }
                    
                    # Merge any error message
                    if "error" in agent_result and agent_result["error"]:
                        if "errors" not in pool_results:
                            pool_results["errors"] = {}
                        pool_results["errors"][agent_name] = agent_result["error"]
                    
                except Exception as e:
                    self.logger.error(f"Error processing agent result: {str(e)}")
        
        # Combine all results
        updated_state = {**state}
        
        # Update state with pool results
        for key, value in pool_results.items():
            if value:  # Only update if we have results
                updated_state[key] = value
        
        # Set status for all agents
        updated_state["research_agent_status"] = "DONE" if "research_results" in pool_results else state.get("research_agent_status", "UNKNOWN")
        updated_state["corporate_agent_status"] = "DONE" if "corporate_results" in pool_results else state.get("corporate_agent_status", "UNKNOWN")
        updated_state["youtube_agent_status"] = "DONE" if "youtube_results" in pool_results else state.get("youtube_agent_status", "UNKNOWN")
        updated_state["rag_agent_status"] = "DONE" if "rag_results" in pool_results else state.get("rag_agent_status", "UNKNOWN")
        
        # Check for any errors
        if "errors" in pool_results and pool_results["errors"]:
            error_msgs = [f"{agent}: {error}" for agent, error in pool_results["errors"].items()]
            error_summary = "; ".join(error_msgs)
            updated_state["error"] = f"Research pool errors: {error_summary}"
            self.logger.warning(f"Research pool completed with errors: {error_summary}")
        else:
            self.logger.info("Research pool completed successfully")
            
        # Return to meta_agent for next steps
        updated_state["goto"] = "meta_agent"
        
        return updated_state


class AnalystPool(BaseAgent):
    name = "analyst_pool"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = get_logger(self.name)
        
        # Initialize the analyst agent
        self.analyst_agent = AnalystAgent(config)
        
        # Configuration parameters
        self.max_workers = config.get("workflow", {}).get("analyst_pool_size", 5)
        
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Implements the abstract _execute method required by BaseAgent."""
        return await self.run(state)
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the analyst pool to process analytical tasks in parallel.
        This pool manages analytical tasks using multiple analyst agent instances.
        """
        self._log_start(state)
        
        company = state.get("company", "")
        if not company:
            self.logger.error("Company name is missing!")
            return {**state, "goto": "meta_agent", "error": "Company name is missing"}
        
        # Get analyst tasks
        tasks = state.get("analyst_tasks", [])
        
        # If no tasks are specified but we have research results, create tasks from events
        if not tasks and "research_results" in state:
            research_results = state.get("research_results", {})
            tasks = self._create_tasks_from_research(research_results)
            self.logger.info(f"Created {len(tasks)} analyst tasks from research results")
        
        if not tasks:
            self.logger.warning("No analyst tasks to process")
            return {
                **state,
                "goto": "meta_agent",
                "analyst_agent_status": "DONE",
                "analysis_results": state.get("analysis_results", {}),
                "error": "No analyst tasks to process"
            }
        
        # Initialize results
        results = {}
        
        # Run tasks in parallel with thread pool
        self.logger.info(f"Processing {len(tasks)} analyst tasks with up to {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Function to process a task
            def process_task(task):
                event_name = task.get("event_name")
                event_data = task.get("event_data")
                
                # Skip invalid tasks
                if not event_name or not event_data:
                    return event_name, {"error": "Invalid task: missing event_name or event_data"}
                
                # Create a new analyst agent for this task (to avoid state conflicts)
                task_agent = AnalystAgent(self.config)
                
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Process the event
                    task_result = loop.run_until_complete(task_agent.process_event(
                        company, event_name, event_data
                    ))
                    loop.close()
                    return event_name, task_result
                except Exception as e:
                    loop.close()
                    error_msg = f"Error processing event {event_name}: {str(e)}"
                    return event_name, {"error": error_msg}
            
            # Submit all tasks to the thread pool
            future_results = {executor.submit(process_task, task): task for task in tasks}
            
            # Process completed tasks as they finish
            for future in future_results:
                try:
                    event_name, task_result = future.result()
                    if event_name:  # Skip results with no event name
                        results[event_name] = task_result
                except Exception as e:
                    self.logger.error(f"Error in task execution: {str(e)}")
        
        # Combine the results into a structured analysis result
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
        return updated_state
    
    def _create_tasks_from_research(self, research_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Create analysis tasks from research results."""
        tasks = []
        
        for event_name, event_articles in research_results.items():
            # Skip events with no articles
            if not event_articles:
                continue
                
            # Create a task for this event
            tasks.append({
                "event_name": event_name,
                "event_data": event_articles,
                "analysis_type": "standard"
            })
        
        return tasks
    
    def _combine_analysis_results(self, task_results: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Combine individual task results into a comprehensive analysis result."""
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
            for key in combined_results.keys():
                if key in state["analysis_results"]:
                    combined_results[key] = state["analysis_results"][key]
        
        # Add task results
        for event_name, result in task_results.items():
            # Skip failed tasks
            if "error" in result and result["error"]:
                continue
                
            # Add to event synthesis
            if "event_synthesis" in result:
                combined_results["event_synthesis"][event_name] = result["event_synthesis"]
                
            # Add to forensic insights
            if "forensic_insights" in result:
                combined_results["forensic_insights"][event_name] = result["forensic_insights"]
                
            # Add to timeline events
            if "timeline" in result:
                for event in result.get("timeline", []):
                    if isinstance(event, dict) and "date" in event and "event" in event:
                        combined_results["timeline"].append(event)
                        
            # Add to red flags
            if "red_flags" in result:
                for flag in result.get("red_flags", []):
                    if flag and flag not in combined_results["red_flags"]:
                        combined_results["red_flags"].append(flag)
            
            # Add entity information if available
            if "entity_network" in result:
                combined_results["entity_network"].update(result.get("entity_network", {}))
        
        # Sort timeline by date if possible
        try:
            combined_results["timeline"] = sorted(
                combined_results["timeline"],
                key=lambda x: datetime.strptime(x.get("date", "2000-01-01"), "%Y-%m-%d")
            )
        except:
            # If sorting fails (e.g., due to date format), don't sort
            pass
        
        # Integrate RAG results if available
        if "rag_results" in state and state["rag_results"]:
            rag_response = state["rag_results"].get("response", "")
            if rag_response:
                # Add RAG insights to the combined results
                combined_results["rag_insights"] = {
                    "response": rag_response,
                    "query": state["rag_results"].get("query", ""),
                    "sources": []
                }
                
                # Extract source information
                retrieval_results = state["rag_results"].get("retrieval_results", {})
                for result in retrieval_results.get("results", []):
                    if result.get("metadata"):
                        source_info = {
                            "source": result.get("metadata", {}).get("source", "Unknown"),
                            "page": result.get("metadata", {}).get("page", "Unknown"),
                            "relevance": result.get("score", 0)
                        }
                        combined_results["rag_insights"]["sources"].append(source_info)
                
                # Extract any additional red flags from RAG response
                if rag_response and "red flag" in rag_response.lower():
                    # Simple extraction of lines containing "red flag"
                    for line in rag_response.split("\n"):
                        if "red flag" in line.lower() and len(line) > 15:  # Ensure it's a meaningful line
                            flag = line.strip()
                            if flag not in combined_results["red_flags"]:
                                combined_results["red_flags"].append(flag)
        
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
        
        init_llm_provider(self.config)
        init_prompt_manager()
        
        # Initialize agents and pools
        self.meta_agent = MetaAgent(self.config)
        self.research_pool = ResearchPool(self.config)
        self.analyst_pool = AnalystPool(self.config)
        self.writer_agent = WriterAgent(self.config)
        
        # Initialize agent mapping
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
        self.graph = self.build_graph()
    
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
        # State and workflow settings
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent/pool
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self.create_agent_node(agent, agent_name))
        
        # Set entry point to meta_agent
        workflow.set_entry_point("meta_agent")
        
        # MetaAgent is the central orchestration point
        # It connects to all agent pools and receives control back after each pool completes
        
        # Connect meta_agent to other agents/pools based on routing logic
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
        workflow.add_edge("research_pool", "meta_agent")
        workflow.add_edge("analyst_pool", "meta_agent")
        workflow.add_edge("writer_agent", "meta_agent")
        
        # Final meta_agent node connects to END
        workflow.add_edge("meta_agent_final", END)
        
        # Compile graph
        memory_saver = MemorySaver()
        return workflow.compile(checkpointer=memory_saver)
    
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
                
                # Ensure status field is updated
                if agent_status_field not in updated_state:
                    updated_state[agent_status_field] = "DONE"
                
                return updated_state
                
            except Exception as e:
                self.logger.error(f"Error in {agent.name}: {str(e)}")
                
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
    
    def route_from_meta_agent(self, state: WorkflowState) -> str:
        """
        Centralized routing logic from meta_agent to next node.
        All routing decisions are now made in the MetaAgent based on the current phase.
        """
        # Check for explicit routing in the goto field
        goto = state.get("goto")
        if goto in ["research_pool", "analyst_pool", "writer_agent", "meta_agent_final", "END"]:
            return goto
        
        # Check current phase to determine routing
        current_phase = state.get("current_phase", "RESEARCH")
        
        if current_phase == "RESEARCH":
            # In research phase, route to research pool
            return "research_pool"
        
        elif current_phase == "ANALYSIS":
            # In analysis phase, route to analyst pool
            return "analyst_pool"
        
        elif current_phase == "REPORT_GENERATION":
            # In report generation phase, route to writer agent
            return "writer_agent"
        
        elif current_phase == "REPORT_REVIEW":
            # In report review phase, route to final meta agent
            return "meta_agent_final"
        
        elif current_phase == "COMPLETE":
            # Workflow is complete
            return "END"
        
        # Default to ending the workflow if phase is unknown
        self.logger.warning(f"Unknown phase: {current_phase}, ending workflow")
        return "END"
    
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
    config_path: Optional[str] = None,
    initial_state: Optional[Dict[str, Any]] = None
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
    
    # Prepare initial state if not provided
    if initial_state is None:
        initial_state = {
            "company": company,
            "industry": industry,
            "enable_rag": True,
            "vector_store_dir": "vector_store",
            "rag_initialized": False
        }
    else:
        # Ensure company and industry are set in the initial state
        if "company" not in initial_state:
            initial_state["company"] = company
        if "industry" not in initial_state and industry is not None:
            initial_state["industry"] = industry
    
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