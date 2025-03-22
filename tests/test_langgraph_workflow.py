import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import logging
from typing import Dict, List, Any, Optional

from langgraph_workflow import EnhancedForensicWorkflow, WorkflowState, create_and_run_workflow


# Test fixtures
@pytest.fixture
def config():
    return {
        "workflow": {
            "max_parallel_agents": 2,
            "analyst_pool_size": 3,
            "require_plan_approval": False
        }
    }


@pytest.fixture
def initial_state():
    return {
        "company": "Test Company",
        "industry": "Technology"
    }


@pytest_asyncio.fixture
async def workflow(config):
    # We'll need to patch various dependencies
    with          patch("utils.logging.setup_logging"), \
         patch("utils.llm_provider.init_llm_provider"), \
         patch("utils.prompt_manager.init_prompt_manager"), \
         patch("agents.meta_agent.MetaAgent"), \
         patch("agents.research_agent.ResearchAgent"), \
         patch("agents.youtube_agent.YouTubeAgent"), \
         patch("agents.corporate_agent.CorporateAgent"), \
         patch("agents.analyst_agent.AnalystAgent"), \
         patch("agents.rag_agent.RAGAgent"), \
         patch("agents.writer_agent.WriterAgent"):
        
        workflow = EnhancedForensicWorkflow(config)
        yield workflow


@pytest.mark.asyncio
async def test_prepare_initial_state(workflow, initial_state):
    """Test preparing the initial state with required values."""
    prepared_state = workflow.prepare_initial_state(initial_state)
    
    # Check that required fields are present
    assert prepared_state["company"] == "Test Company"
    assert prepared_state["industry"] == "Technology"
    assert prepared_state["meta_iteration"] == 0
    assert prepared_state["goto"] == "meta_agent"
    
    # Check that default values for all fields are set
    assert "research_plan" in prepared_state
    assert "search_history" in prepared_state
    assert "research_results" in prepared_state
    assert "event_metadata" in prepared_state
    assert "corporate_results" in prepared_state
    assert "youtube_results" in prepared_state
    assert "analysis_results" in prepared_state


@pytest.mark.asyncio
async def test_build_graph(workflow):
    """Test that the graph is properly built with expected nodes."""
    # This will have been called in __init__, so we can inspect the graph property
    assert workflow.graph is not None
    
    # We'd need to mock methods inside the LangGraph object to test more,
    # but this is tricky since it's a complex object from an external library


@pytest.mark.asyncio
async def test_create_agent_node(workflow):
    """Test the agent node creation function."""
    agent_mock = MagicMock()
    agent_mock.name = "test_agent"
    agent_mock.run = AsyncMock(return_value={"goto": "next_agent", "test_agent_status": "DONE"})
    
    # Create the node function
    node_func = workflow.create_agent_node(agent_mock, "test_agent")
    
    # Test with simple state
    input_state = {
        "company": "Test Company",
        "industry": "Technology",
        "goto": "test_agent"
    }
    
    # Execute the node function
    result_state = node_func(input_state)
    
    # Verify the state is updated correctly
    assert result_state["goto"] == "next_agent"
    assert result_state["test_agent_status"] == "DONE"
    
    # Verify the agent's run method was called
    agent_mock.run.assert_called_once()


@pytest.mark.asyncio
async def test_parallel_executor_node(workflow):
    """Test the parallel execution of agents."""
    # Setup initial state with some agents already completed
    state = {
        "company": "Test Company",
        "industry": "Technology",
        "parallel_agents": ["research_agent", "corporate_agent", "youtube_agent"],
        "running_agents": set(),
        "completed_agents": set(["research_agent"]),
        "failed_agents": set()
    }
    
    # Patch the max_parallel_agents value
    workflow.max_parallel_agents = 2
    
    # Run the parallel executor node
    result = workflow.parallel_executor_node(state)
    
    # Should start up to max_parallel_agents (2) agents that aren't running/completed/failed
    assert result["goto"] in ["corporate_agent", "youtube_agent"]
    assert len(result["running_agents"]) == 1
    
    # Run again with all agents running
    state["running_agents"] = set(["corporate_agent", "youtube_agent"])
    result = workflow.parallel_executor_node(state)
    
    # Should wait as we're at max concurrency
    assert result["goto"] == "parallel_executor"
    
    # Test with all agents completed
    state["running_agents"] = set()
    state["completed_agents"] = set(["research_agent", "corporate_agent", "youtube_agent"])
    result = workflow.parallel_executor_node(state)
    
    # Should move to research_complete when all agents are done
    assert result["goto"] == "research_complete"


@pytest.mark.asyncio
async def test_plan_approval_node(workflow):
    """Test the plan approval node's behavior."""
    # Set workflow to require approval
    workflow.require_plan_approval = True
    
    # Test state waiting for approval
    state = {
        "company": "Test Company",
        "industry": "Technology",
        "user_approved": False,
        "user_feedback": None
    }
    
    result = workflow.plan_approval_node(state)
    
    # Should be waiting for approval
    assert result["requires_user_approval"] is True
    assert result["user_approval_type"] == "research_plan"
    
    # Test with approval provided
    state["user_feedback"] = {"approved": True}
    result = workflow.plan_approval_node(state)
    
    # Should proceed to parallel execution
    assert result["goto"] == "parallel_executor"
    assert result["user_approved"] is True
    assert result["requires_user_approval"] is False
    
    # Test with rejection feedback
    state["user_approved"] = False
    state["user_feedback"] = {"approved": False, "feedback_text": "Please revise"}
    result = workflow.plan_approval_node(state)
    
    # Should go back to meta agent for revision
    assert result["goto"] == "meta_agent"
    assert result["user_approved"] is False
    assert "plan_feedback" in result
    assert result["plan_feedback"] == "Please revise"


@pytest.mark.asyncio
async def test_research_complete_node(workflow):
    """Test the research complete node's behavior."""
    # Test with successful research results
    state = {
        "company": "Test Company",
        "industry": "Technology",
        "research_results": {"Event 1": ["article1", "article2"]},
        "parallel_agents": ["research_agent", "corporate_agent"],
        "running_agents": set(),
        "completed_agents": set(["research_agent", "corporate_agent"]),
        "failed_agents": set()
    }
    
    result = workflow.research_complete_node(state)
    
    # Should clear parallel execution tracking and set research_completed flag
    assert result["parallel_agents"] == []
    assert result["running_agents"] == set()
    assert result["completed_agents"] == set()
    assert result["failed_agents"] == set()
    assert result["research_completed"] is True
    assert result["goto"] == "meta_agent"
    
    # Test with no research results
    state["research_results"] = {}
    result = workflow.research_complete_node(state)
    
    # Should return to meta_agent with error
    assert result["goto"] == "meta_agent"
    assert "error" in result
    assert "No research results" in result["error"]


@pytest.mark.asyncio
async def test_route_from_meta_agent(workflow):
    """Test routing logic from the meta agent."""
    # Test initial state (research phase)
    state = {
        "company": "Test Company",
        "industry": "Technology",
        "research_completed": False,
        "research_plan": None,
        "goto": None
    }
    
    route = workflow.route_from_meta_agent(state)
    assert route == "plan_approval"  # First iteration goes to plan approval
    
    # Test with research plan but still in research phase
    state["research_plan"] = [{"objective": "Test"}]
    state["requires_additional_research"] = True
    route = workflow.route_from_meta_agent(state)
    assert route == "parallel_executor"  # Should do more parallel research
    
    # Test moving to analysis phase
    state["requires_additional_research"] = False
    route = workflow.route_from_meta_agent(state)
    assert route == "analyst_pool"  # Should go to analysis
    
    # Test in analysis phase
    state["research_completed"] = True
    state["analysis_completed"] = False
    route = workflow.route_from_meta_agent(state)
    assert route == "analyst_pool"  # Should stay in analysis
    
    # Test in report generation phase
    state["analysis_completed"] = True
    state["report_completed"] = False
    route = workflow.route_from_meta_agent(state)
    assert route == "writer_agent"  # Should go to report generation
    
    # Test workflow completed
    state["report_completed"] = True
    route = workflow.route_from_meta_agent(state)
    assert route == "meta_agent_final"  # Should finish workflow


@pytest.mark.asyncio
async def test_handle_error(workflow):
    """Test the error handling logic."""
    state = {
        "company": "Test Company",
        "industry": "Technology",
        "research_agent_status": "RUNNING"
    }
    
    # Mock the logger
    with patch.object(workflow, "logger") as mock_logger:
        result = workflow.handle_error(state)
        
        # Verify logger was called
        assert mock_logger.error.called
        
        # Verify state updates
        assert "error" in result
        assert result["goto"] == "meta_agent"
        assert result["research_agent_status"] == "ERROR"


@pytest.mark.asyncio
async def test_run(workflow, initial_state):
    """Test the main run method of the workflow."""
    # This is complex to test as it involves the full graph execution
    # We'll use a simplified approach by mocking the graph
    
    mock_graph = MagicMock()
    mock_event = {
        'state': {
            'company': 'Test Company',
            'final_report': 'This is a test report',
            'meta_iteration': 1
        },
        'current_node': 'meta_agent_final'
    }
    mock_graph.stream.return_value = [mock_event]
    workflow.graph = mock_graph
    
    # Run the workflow
    result = await workflow.run(initial_state)
    
    # Verify the result
    assert result['company'] == 'Test Company'
    assert result['final_report'] == 'This is a test report'
    assert result['meta_iteration'] == 1
    
    # Verify the graph was called
    mock_graph.stream.assert_called_once()


@pytest.mark.asyncio
async def test_run_sync(workflow, initial_state):
    """Test the synchronous wrapper for running the workflow."""
    # Mock the async run method
    workflow.run = AsyncMock(return_value={"company": "Test Company", "final_report": "Test report"})
    
    # Run the sync version
    result = workflow.run_sync(initial_state)
    
    # Verify result and that run was called
    assert result["company"] == "Test Company"
    assert result["final_report"] == "Test report"
    workflow.run.assert_called_once_with(initial_state)


@pytest.mark.asyncio
async def test_create_and_run_workflow():
    """Test the creation and running of a workflow from the helper function."""
    with patch("langgraph_workflow.setup_logging"), \
         patch("langgraph_workflow.get_logger"), \
         patch("langgraph_workflow.EnhancedForensicWorkflow") as mock_workflow_class:
        
        # Setup mock workflow instance
        mock_workflow = MagicMock()
        mock_workflow.run_sync.return_value = {"company": "Test Company", "final_report": "Test report"}
        mock_workflow_class.return_value = mock_workflow
        
        # Call the function
        result = create_and_run_workflow("Test Company", "Technology", None)
        
        # Verify workflow creation and running
        mock_workflow_class.assert_called_once()
        mock_workflow.run_sync.assert_called_once()
        
        # Verify result
        assert result["company"] == "Test Company"
        assert result["final_report"] == "Test report"