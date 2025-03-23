import pytest
import asyncio
from typing import Dict, Any
import json

from base.base_agents import BaseAgent, AgentState, ValidationError, ExecutionError

class TestAgent(BaseAgent):
    name = "test_agent"
    
    async def _execute(self, state: AgentState) -> Dict[str, Any]:
        if state.company == "error_company":
            raise ExecutionError("Test execution error", agent_name=self.name)
        
        if state.company == "sleep_company":
            await asyncio.sleep(0.1)
        
        # Test different state update patterns
        results = {
            "result": "success",
            "goto": "next_agent",
            f"{self.name}_status": "DONE"
        }
        
        return results

@pytest.mark.asyncio
async def test_agent_basic_execution():
    agent = TestAgent()
    
    result = await agent.run({
        "company": "Test Corp",
        "industry": "Technology"
    })
    
    assert result["result"] == "success"
    assert result["goto"] == "next_agent"
    assert result["test_agent_status"] == "DONE"
    # State should maintain original fields
    assert result["company"] == "Test Corp"
    assert result["industry"] == "Technology"

@pytest.mark.asyncio
async def test_agent_error_handling():
    agent = TestAgent()
    
    result = await agent.run({
        "company": "error_company",
        "industry": "Technology"
    })
    
    assert "error" in result
    assert result["goto"] == "meta_agent"
    assert result["test_agent_status"] == "ERROR"
    assert "test_agent_error_details" in result

@pytest.mark.asyncio
async def test_agent_validation_error():
    agent = TestAgent()
    
    # Missing required company field
    result = await agent.run({
        "industry": "Technology"
    })
    
    assert "error" in result
    assert "validation failed" in result["error"].lower()
    assert result["goto"] == "meta_agent"
    assert result["test_agent_status"] == "ERROR"

@pytest.mark.asyncio
async def test_agent_metrics():
    agent = TestAgent()
    
    await agent.run({
        "company": "sleep_company",
        "industry": "Technology"
    })
    
    assert agent.metrics.execution_time_ms >= 100  # Should be at least 100ms
    assert "execute" in agent.metrics.sub_operations
    assert agent.metrics.sub_operations["execute"] >= 100

@pytest.mark.asyncio
async def test_agent_state_operations():
    class StateTrackingAgent(TestAgent):
        async def _execute(self, state: AgentState) -> Dict[str, Any]:
            # Read operations
            company = state["company"]
            industry = state["industry"]
            
            # Write operations
            state["new_field"] = "new value"
            state["result"] = f"Processed {company}"
            
            # Return standard result plus state tracking info
            result = {
                "result": f"Processed {company}",
                "goto": "next_agent",
                f"{self.name}_status": "DONE",
                "state_operations": state._state_ops.get_operations_summary()
            }
            
            return result
    
    agent = StateTrackingAgent()
    
    result = await agent.run({
        "company": "Test Corp",
        "industry": "Technology"
    })
    
    # Verify state operations were tracked
    assert "state_operations" in result
    assert "company" in result["state_operations"]["read"]
    assert "industry" in result["state_operations"]["read"]
    assert "new_field" in result["state_operations"]["written"]
    assert "result" in result["state_operations"]["written"]