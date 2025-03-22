# tests/test_meta_agent.py
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

from agents.meta_agent import MetaAgent


@pytest.fixture
def config():
    return {
        "meta_agent": {
            "max_parallel_agents": 3,
            "parallel_execution": True
        },
        "quality_thresholds": {
            "min_quality_score": 6
        },
        "max_iterations": 3,
        "models": {
            "planning": "gemini-2.0-flash",
            "evaluation": "gemini-2.0-pro"
        }
    }


@pytest.fixture
def state():
    return {
        "company": "Test Company",
        "industry": "Technology",
        "meta_iteration": 0,
        "research_plan": [],
        "search_history": []
    }


@pytest.fixture
def agent(config):
    with patch("utils.prompt_manager.get_prompt_manager"), \
         patch("utils.logging.get_logger"), \
         patch("tools.postgres_tool.PostgresTool"):
        return MetaAgent(config)


@pytest.mark.asyncio
async def test_initialize_agent_tasks(agent, state):
    agent.initialize_agent_tasks(state)
    
    assert len(agent.agent_tasks) > 0
    assert "research_agent" in agent.agent_tasks
    assert "analyst_agent" in agent.agent_tasks
    assert "writer_agent" in agent.agent_tasks
    assert len(agent.pending_agents) > 0


@pytest.mark.asyncio
async def test_update_agent_status(agent, state):
    agent.initialize_agent_tasks(state)
    
    agent.update_agent_status("research_agent", "RUNNING")
    assert "research_agent" in agent.running_agents
    assert "research_agent" not in agent.pending_agents
    
    agent.update_agent_status("research_agent", "DONE")
    assert "research_agent" in agent.completed_agents
    assert "research_agent" not in agent.running_agents
    
    agent.update_agent_status("analyst_agent", "ERROR", "Test error")
    assert "analyst_agent" in agent.failed_agents
    assert agent.agent_tasks["analyst_agent"].error == "Test error"


@pytest.mark.asyncio
async def test_get_next_agents(agent, state):
    agent.initialize_agent_tasks(state)
    
    # First call should return highest priority agent with no dependencies
    next_agents = agent.get_next_agents()
    assert "research_agent" in next_agents
    
    # Mark research_agent as running
    agent.update_agent_status("research_agent", "RUNNING")
    
    # No more agents until dependencies are met
    next_agents = agent.get_next_agents()
    assert len(next_agents) == 0
    
    # Complete research_agent
    agent.update_agent_status("research_agent", "DONE")
    
    # Now analyst_agent should be available
    next_agents = agent.get_next_agents()
    assert "analyst_agent" in next_agents


@pytest.mark.asyncio
async def test_run_first_iteration(agent, state):
    with patch.object(agent, "_load_preliminary_guidelines", AsyncMock(return_value={"objective": "Test"})), \
         patch.object(agent, "generate_workflow_status", return_value={}), \
         patch.object(agent, "save_workflow_status"):
        
        result = await agent.run(state)
        
        assert result["goto"] == "research_agent"
        assert result["research_plan"] == [{"objective": "Test"}]
        assert result["meta_iteration"] == 1


@pytest.mark.asyncio
async def test_run_later_iteration(agent, state):
    state["meta_iteration"] = 1
    state["research_plan"] = [{"objective": "Test"}]
    state["goto"] = "meta_agent"
    state["research_agent_status"] = "DONE"
    
    with patch.object(agent, "manage_workflow", AsyncMock(return_value=(state, "analyst_agent"))):
        result = await agent.run(state)
        
        assert result["goto"] == "analyst_agent"
        assert result["meta_iteration"] == 2