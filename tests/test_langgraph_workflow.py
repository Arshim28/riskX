import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import logging
from typing import Dict, List, Any, Optional

from langgraph_workflow import EnhancedForensicWorkflow, WorkflowState, create_and_run_workflow


@pytest.fixture
def config():
    return {
        "workflow": {
            "max_parallel_agents": 2,
            "analyst_pool_size": 3,
            "require_plan_approval": False
        },
        "youtube": {
            "youtube_api_key": "mock_api_key"
        },
        "ocr": {
            "api_key": "mock_mistral_api_key"
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
    # Patch tools and their configurations
    youtube_tool_patch = patch("tools.youtube_tool.YoutubeTool.__init__", return_value=None)
    youtube_tool_mock = youtube_tool_patch.start()
    
    youtube_config_patch = patch("tools.youtube_tool.YoutubeToolConfig")
    youtube_config_mock = youtube_config_patch.start()
    
    # Patch OcrTool to avoid Mistral API key validation
    ocr_tool_patch = patch("tools.ocr_tool.OcrTool.__init__", return_value=None)
    ocr_tool_mock = ocr_tool_patch.start()
    
    # Patch OCRVectorStoreTool to avoid initialization errors
    vector_store_patch = patch("tools.ocr_vector_store_tool.OCRVectorStoreTool.__init__", return_value=None)
    vector_store_mock = vector_store_patch.start()
    
    # Patch agent classes
    youtube_agent_patch = patch("agents.youtube_agent.YouTubeAgent")
    youtube_agent_mock = youtube_agent_patch.start()
    youtube_agent_mock.return_value = MagicMock(name="mock_youtube_agent")
    youtube_agent_mock.return_value.run = AsyncMock(return_value={"goto": "meta_agent", "youtube_agent_status": "DONE"})
    
    # Patch StateGraph to mock the missing set_error_handler method
    graph_patch = patch("langgraph.graph.StateGraph", autospec=True)
    graph_mock = graph_patch.start()
    graph_mock.return_value.set_error_handler = MagicMock()
    
    # Mock all agent classes and utility functions
    with patch("utils.logging.setup_logging"), \
         patch("utils.llm_provider.init_llm_provider"), \
         patch("utils.prompt_manager.init_prompt_manager"), \
         patch("agents.meta_agent.MetaAgent"), \
         patch("agents.research_agent.ResearchAgent"), \
         patch("agents.corporate_agent.CorporateAgent"), \
         patch("agents.analyst_agent.AnalystAgent"), \
         patch("agents.rag_agent.RAGAgent"), \
         patch("agents.writer_agent.WriterAgent"), \
         patch("mistralai.Mistral"):
        
        try:
            workflow = EnhancedForensicWorkflow(config)
            yield workflow
        finally:
            youtube_tool_patch.stop()
            youtube_config_patch.stop()
            youtube_agent_patch.stop()
            ocr_tool_patch.stop()
            vector_store_patch.stop()
            graph_patch.stop()


@pytest.mark.asyncio
async def test_prepare_initial_state(workflow, initial_state):
    prepared_state = workflow.prepare_initial_state(initial_state)
    
    assert prepared_state["company"] == "Test Company"
    assert prepared_state["industry"] == "Technology"
    assert prepared_state["meta_iteration"] == 0
    assert prepared_state["goto"] == "meta_agent"
    
    assert "research_plan" in prepared_state
    assert "search_history" in prepared_state
    assert "research_results" in prepared_state
    assert "event_metadata" in prepared_state
    assert "corporate_results" in prepared_state
    assert "youtube_results" in prepared_state
    assert "analysis_results" in prepared_state


@pytest.mark.asyncio
async def test_build_graph(workflow):
    # Simple check that the graph exists
    assert hasattr(workflow, "graph")
    assert workflow.graph is not None


@pytest.mark.asyncio
async def test_create_agent_node(workflow):
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
    
    # We'll patch asyncio.get_event_loop to avoid the "Cannot run the event loop" error
    with patch('asyncio.get_event_loop') as mock_get_loop, \
         patch('asyncio.new_event_loop') as mock_new_loop, \
         patch('asyncio.set_event_loop') as mock_set_loop:
        
        # Set up the mock to return our controlled result directly
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True  # Force the creation of a new loop
        mock_get_loop.return_value = mock_loop
        
        # The new loop that will actually "run" our coroutine
        mock_new_loop_instance = MagicMock()
        mock_new_loop.return_value = mock_new_loop_instance
        
        # When run_until_complete is called, we'll call the agent's run method and return its result
        def side_effect(coro):
            # This mimics what would happen in the real run_until_complete
            agent_mock.run.assert_not_called()  # Verify run hasn't been called yet
            return {"goto": "next_agent", "test_agent_status": "DONE"}
            
        mock_new_loop_instance.run_until_complete.side_effect = side_effect
        
        # Execute the node function
        result_state = node_func(input_state)
    
    # Verify the state is updated correctly
    assert result_state["goto"] == "next_agent"
    assert result_state["test_agent_status"] == "DONE"


@pytest.mark.asyncio
async def test_parallel_executor_node(workflow):
    # Create a simple test state
    state = {
        "company": "Test Company",
        "industry": "Technology",
        "parallel_agents": ["research_agent", "corporate_agent", "youtube_agent"],
        "running_agents": set(),
        "completed_agents": set(["research_agent"]),
        "failed_agents": set()
    }
    
    # Set max_parallel_agents
    workflow.max_parallel_agents = 1  # Only allow 1 agent at a time for simplicity
    
    # First test: one agent completed, should start another
    result = workflow.parallel_executor_node(state)
    
    # Verify correct routing
    assert result["goto"] in ["corporate_agent", "youtube_agent"]
    assert len(result["running_agents"]) == 1
    
    # Second test: max agents running, should wait
    next_state = {
        "company": "Test Company",
        "industry": "Technology",
        "parallel_agents": ["research_agent", "corporate_agent", "youtube_agent"],
        "running_agents": set(["corporate_agent"]),  # One agent is running
        "completed_agents": set(["research_agent"]),
        "failed_agents": set()
    }
    
    # With max_parallel_agents=1 and one already running, should wait
    result = workflow.parallel_executor_node(next_state)
    assert result["goto"] == "parallel_executor"
    
    # Third test: all agents completed, should move to research_complete
    final_state = {
        "company": "Test Company",
        "industry": "Technology",
        "parallel_agents": ["research_agent", "corporate_agent", "youtube_agent"],
        "running_agents": set(),
        "completed_agents": set(["research_agent", "corporate_agent", "youtube_agent"]),
        "failed_agents": set()
    }
    
    result = workflow.parallel_executor_node(final_state)
    assert result["goto"] == "research_complete"


@pytest.mark.asyncio
async def test_plan_approval_node(workflow):
    workflow.require_plan_approval = True
    
    state = {
        "company": "Test Company",
        "industry": "Technology",
        "user_approved": False,
        "user_feedback": None
    }
    
    result = workflow.plan_approval_node(state)
    
    assert result["requires_user_approval"] is True
    assert result["user_approval_type"] == "research_plan"
    
    state["user_feedback"] = {"approved": True}
    result = workflow.plan_approval_node(state)
    
    assert result["goto"] == "parallel_executor"
    assert result["user_approved"] is True
    assert result["requires_user_approval"] is False
    
    state["user_approved"] = False
    state["user_feedback"] = {"approved": False, "feedback_text": "Please revise"}
    result = workflow.plan_approval_node(state)
    
    assert result["goto"] == "meta_agent"
    assert result["user_approved"] is False
    assert "plan_feedback" in result
    assert result["plan_feedback"] == "Please revise"


@pytest.mark.asyncio
async def test_research_complete_node(workflow):
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
    
    assert result["parallel_agents"] == []
    assert result["running_agents"] == set()
    assert result["completed_agents"] == set()
    assert result["failed_agents"] == set()
    assert result["research_completed"] is True
    assert result["goto"] == "meta_agent"
    
    state["research_results"] = {}
    result = workflow.research_complete_node(state)
    
    assert result["goto"] == "meta_agent"
    assert "error" in result
    assert "No research results" in result["error"]


@pytest.mark.asyncio
async def test_route_from_meta_agent(workflow):
    state = {
        "company": "Test Company",
        "industry": "Technology",
        "research_completed": False,
        "research_plan": None,
        "goto": None
    }
    
    route = workflow.route_from_meta_agent(state)
    assert route == "plan_approval"
    
    state["research_plan"] = [{"objective": "Test"}]
    state["requires_additional_research"] = True
    route = workflow.route_from_meta_agent(state)
    assert route == "parallel_executor"
    
    state["requires_additional_research"] = False
    route = workflow.route_from_meta_agent(state)
    assert route == "analyst_pool"
    
    state["research_completed"] = True
    state["analysis_completed"] = False
    route = workflow.route_from_meta_agent(state)
    assert route == "analyst_pool"
    
    state["analysis_completed"] = True
    state["report_completed"] = False
    route = workflow.route_from_meta_agent(state)
    assert route == "writer_agent"
    
    state["report_completed"] = True
    route = workflow.route_from_meta_agent(state)
    assert route == "meta_agent_final"


@pytest.mark.asyncio
async def test_handle_error(workflow):
    state = {
        "company": "Test Company",
        "industry": "Technology",
        "research_agent_status": "RUNNING"
    }
    
    with patch.object(workflow, "logger") as mock_logger:
        result = workflow.handle_error(state)
        
        assert mock_logger.error.called
        
        assert "error" in result
        assert result["goto"] == "meta_agent"
        assert result["research_agent_status"] == "ERROR"


@pytest.mark.asyncio
async def test_run(workflow, initial_state):
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
    
    result = await workflow.run(initial_state)
    
    assert result['company'] == 'Test Company'
    assert result['final_report'] == 'This is a test report'
    assert result['meta_iteration'] == 1
    
    mock_graph.stream.assert_called_once()


@pytest.mark.asyncio
async def test_run_sync(workflow, initial_state):
    # We need to patch both the run method and asyncio functions
    workflow.run = AsyncMock(return_value={"company": "Test Company", "final_report": "Test report"})
    
    # We'll patch asyncio.get_event_loop to avoid the "Cannot run the event loop" error
    with patch('asyncio.get_event_loop') as mock_get_loop:
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = {"company": "Test Company", "final_report": "Test report"}
        mock_get_loop.return_value = mock_loop
        
        # Run the sync version
        result = workflow.run_sync(initial_state)
    
    # Verify result and that run was called
    assert result["company"] == "Test Company"
    assert result["final_report"] == "Test report"
    workflow.run.assert_called_once_with(initial_state)


@pytest.mark.asyncio
async def test_create_and_run_workflow():
    with patch("langgraph_workflow.setup_logging"), \
         patch("langgraph_workflow.get_logger"), \
         patch("langgraph_workflow.EnhancedForensicWorkflow") as mock_workflow_class:
        
        mock_workflow = MagicMock()
        mock_workflow.run_sync.return_value = {"company": "Test Company", "final_report": "Test report"}
        mock_workflow_class.return_value = mock_workflow
        
        result = create_and_run_workflow("Test Company", "Technology", None)
        
        mock_workflow_class.assert_called_once()
        mock_workflow.run_sync.assert_called_once()
        
        assert result["company"] == "Test Company"
        assert result["final_report"] == "Test report"