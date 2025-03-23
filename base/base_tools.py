from abc import ABC, abstractmethod
from typing import List, Any, Optional, TypeVar, Generic, Dict, Union, Type, Callable
from pydantic import BaseModel, Field, create_model
import asyncio
import time
import traceback
import logging
from contextlib import asynccontextmanager
from functools import wraps

T = TypeVar('T')
R = TypeVar('R')

class ToolMetrics(BaseModel):
    execution_time_ms: float = 0
    start_time: float = 0
    end_time: float = 0
    retry_count: int = 0
    memory_usage_mb: Optional[float] = None
    
    def record_start(self) -> None:
        self.start_time = time.time()
    
    def record_end(self) -> None:
        self.end_time = time.time()
        self.execution_time_ms = (self.end_time - self.start_time) * 1000
    
    def record_retry(self) -> None:
        self.retry_count += 1

class ErrorInfo(BaseModel):
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_exception(cls, exc: Exception, include_traceback: bool = True, context: Dict[str, Any] = None) -> 'ErrorInfo':
        return cls(
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            traceback=traceback.format_exc() if include_traceback else None,
            context=context or {}
        )

class ToolResult(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[Union[str, ErrorInfo]] = None
    metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    
    @classmethod
    def success_result(cls, data: T) -> 'ToolResult[T]':
        return cls(success=True, data=data)
    
    @classmethod
    def error_result(cls, error: Union[str, Exception, ErrorInfo]) -> 'ToolResult[T]':
        if isinstance(error, str):
            return cls(success=False, error=error)
        elif isinstance(error, Exception):
            return cls(success=False, error=ErrorInfo.from_exception(error))
        elif isinstance(error, ErrorInfo):
            return cls(success=False, error=error)
        else:
            return cls(success=False, error=str(error))

class BaseTool(ABC):
    name: str = "base_tool"
    
    def __init__(self):
        self.logger = logging.getLogger(f"tool.{self.name}")
        
    @asynccontextmanager
    async def _resource_context(self, **kwargs):
        try:
            yield
        finally:
            pass
    
    async def _pre_execute(self, **kwargs) -> None:
        pass
        
    async def _post_execute(self, result: ToolResult, **kwargs) -> None:
        pass
    
    async def _handle_error(self, error: Exception, **kwargs) -> ToolResult:
        self.logger.error(f"Error in {self.name}: {str(error)}", exc_info=True)
        
        context = {
            "tool": self.name,
            "params": {k: v for k, v in kwargs.items() if not k.startswith("_")}
        }
        
        error_info = ErrorInfo.from_exception(error, context=context)
        return ToolResult.error_result(error_info)
    
    @abstractmethod
    async def _execute(self, **kwargs) -> ToolResult:
        pass
    
    async def run(self, **kwargs) -> ToolResult:
        metrics = ToolMetrics()
        metrics.record_start()
        
        try:
            await self._pre_execute(**kwargs)
            
            async with self._resource_context(**kwargs):
                result = await self._execute(**kwargs)
            
            # Attach metrics to result
            metrics.record_end()
            result.metrics = metrics
            
            await self._post_execute(result, **kwargs)
            return result
            
        except Exception as e:
            metrics.record_end()
            error_result = await self._handle_error(e, **kwargs) 
            error_result.metrics = metrics
            return error_result