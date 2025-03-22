import os
import json
import asyncio
from typing import Dict, List, Any, Tuple, Optional, Union, Annotated, TypedDict, Literal
from datetime import datetime
from pydantic import BaseModel, Field
import traceback

from langchain.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemoryCheckpoint
from langgraph.graph.graph import CompiledGraph

from base.base_graph import BaseGraph, GraphConfig
from utils.logging import get_logger, setup_logging
from utils.llm_provider import init_llm_provider
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


class ForensicWorkflow(BaseGraph):
    """Graph-based workflow for financial forensic analysis."""
    
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
            "writer_agent": self.writer_agent
        }
        
        # Create the workflow graph
        self.graph = self.build_graph()
        
    def build_graph(self) -> CompiledGraph:
        """Build the workflow graph."""
        # State and workflow settings
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self.create_agent_node(agent))
        
        # Add edge from START to meta_agent as the entry point
        workflow.set_entry_point("meta_agent")
        
        # Add conditional edges from meta_agent to other agents based on state routing
        workflow.add_conditional_edges(
            "meta_agent",
            self.route_from_meta_agent,
            {
                "research_agent": "research_agent",
                "youtube_agent": "youtube_agent",
                "corporate_agent": "corporate_agent",
                "analyst_agent": "analyst_agent",
                "rag_agent": "rag_agent",
                "writer_agent": "writer_agent",
                "END": END,
                "WAIT": "meta_agent"
            }
        )
        
        # All agents return to meta_agent for orchestration
        for agent_name in ["research_agent", "youtube_agent", "corporate_agent", 
                          "analyst_agent", "rag_agent", "writer_agent"]:
            workflow.add_edge(agent_name, "meta_agent")
            
        # Add error handler to capture exceptions
        if self.config.get("enable_error_handling", True):
            workflow.add_node("error_handler", self.handle_error)
            workflow.set_error_handler("error_handler")
        
        # Compile graph
        memory = MemoryCheckpoint()
        return workflow.compile(checkpointer=memory)
    
    def create_agent_node(self, agent):
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
        """Determine next node from meta_agent based on the state goto field."""
        goto = state.get("goto")
        
        if not goto or goto == "meta_agent":
            return "meta_agent"  # Loop back if no clear direction
            
        if goto == "END":
            return "END"
            
        if goto == "WAIT":
            # Small wait to avoid rapid polling
            import time
            time.sleep(0.5)
            return "WAIT"
            
        # Check if target is a valid agent
        if goto in self.agents:
            return goto
            
        # Default to meta_agent if unknown target
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
            "workflow_status": {}
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
    workflow = ForensicWorkflow(config)
    
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