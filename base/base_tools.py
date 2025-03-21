from abc import ABC, abstractmethod
from typing import List, Any, Optional, TypeVar, Generic, Dict
from pydantic import BaseModel

T = TypeVar('T')

class ToolResult(BaseModel, Generic[T]):
	success: bool
	data: Optional[T] = None
	error: Optional[T] = None

class BaseTool(ABC):
	name: str = "base_tool"

	@abstractmethod
	async def run(self, **kwargs) -> ToolResult:
		pass

	async def _handle_error(self, error: Exception) -> ToolResult:
		error_msg = f"{type(error).__name__}: {str(error)}"
		return ToolResult(success=False, error=error_msg)