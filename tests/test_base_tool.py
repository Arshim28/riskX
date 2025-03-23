import pytest
import asyncio
from typing import Dict, Any
import time

from base.base_tools import BaseTool, ToolResult, ErrorInfo, ToolMetrics

class TestTool(BaseTool):
    name = "test_tool"
    
    async def _execute(self, command: str, **kwargs) -> ToolResult:
        if command == "success":
            return ToolResult.success_result({"result": "success"})
        elif command == "error":
            raise ValueError("Test error")
        elif command == "sleep":
            await asyncio.sleep(0.1)
            return ToolResult.success_result({"result": "after sleep"})
        else:
            return ToolResult.error_result("Unknown command")

@pytest.mark.asyncio
async def test_tool_success():
    tool = TestTool()
    result = await tool.run(command="success")
    
    assert result.success is True
    assert result.data["result"] == "success"
    assert result.metrics.execution_time_ms > 0

@pytest.mark.asyncio
async def test_tool_error():
    tool = TestTool()
    result = await tool.run(command="error")
    
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "ValueError"
    assert "Test error" in result.error.error_message
    assert result.metrics.execution_time_ms > 0

@pytest.mark.asyncio
async def test_tool_metrics():
    tool = TestTool()
    result = await tool.run(command="sleep")
    
    assert result.success is True
    assert result.metrics.execution_time_ms >= 100  # Should be at least 100ms

@pytest.mark.asyncio
async def test_resource_management():
    class ResourceTool(TestTool):
        def __init__(self):
            super().__init__()
            self.resource_initialized = False
            self.resource_cleaned = False
            
        async def _resource_context(self, **kwargs):
            self.resource_initialized = True
            try:
                yield
            finally:
                self.resource_cleaned = True
    
    tool = ResourceTool()
    result = await tool.run(command="success")
    
    assert tool.resource_initialized is True
    assert tool.resource_cleaned is True