from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic, Type, Set
from pydantic import BaseModel, Field, validator, root_validator
import time
import asyncio
import traceback
import logging
import json
import copy
from contextlib import asynccontextmanager

T = TypeVar('T')

class AgentMetrics(BaseModel):
    """Performance metrics for agent execution."""
    start_time: float = 0
    end_time: float = 0
    execution_time_ms: float = 0
    memory_usage_mb: Optional[float] = None
    sub_operations: Dict[str, float] = Field(default_factory=dict)
    
    def record_start(self) -> None:
        """Record the start time."""
        self.start_time = time.time()
    
    def record_end(self) -> None:
        """Record the end time and calculate duration."""
        self.end_time = time.time()
        self.execution_time_ms = (self.end_time - self.start_time) * 1000
    
    def record_operation(self, operation: str, duration_ms: float) -> None:
        """Record the duration of a sub-operation."""
        self.sub_operations[operation] = duration_ms

class StateOperation:
    """Tracking class for state operations and modifications."""
    def __init__(self):
        self.read_keys: Set[str] = set()
        self.written_keys: Set[str] = set()
        self.deleted_keys: Set[str] = set()
        
    def record_read(self, key: str) -> None:
        """Record a state read operation."""
        self.read_keys.add(key)
        
    def record_write(self, key: str) -> None:
        """Record a state write operation."""
        self.written_keys.add(key)
        
    def record_delete(self, key: str) -> None:
        """Record a state delete operation."""
        self.deleted_keys.add(key)
        
    def get_operations_summary(self) -> Dict[str, List[str]]:
        """Get a summary of all operations."""
        return {
            "read": sorted(list(self.read_keys)),
            "written": sorted(list(self.written_keys)),
            "deleted": sorted(list(self.deleted_keys))
        }

class AgentState(BaseModel):
    """Base model for agent state with enhanced validation and tracking."""
    goto: Optional[str] = None
    error: Optional[str] = None
    company: str
    industry: Optional[str] = None
    iteration: int = 0
    
    # Add tracking for state operations
    state_ops: StateOperation = Field(default_factory=StateOperation, exclude=True)
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow"  # Allow extra fields for agent-specific state
    }
    
    @root_validator(pre=True)
    def validate_required_fields(cls, values):
        """Validate that required fields are present."""
        if "company" not in values:
            raise ValueError("company is required in agent state")
        return values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary with appropriate serialization."""
        state_dict = self.model_dump(exclude={"state_ops"})
        # Handle non-serializable types if needed
        return state_dict
    
    def copy_with_updates(self, updates: Dict[str, Any]) -> "AgentState":
        """Create a new state with updates applied."""
        # Create a deep copy of current state
        state_dict = self.model_dump(exclude={"state_ops"})
        # Apply updates
        state_dict.update(updates)
        # Create new state object
        return self.__class__(**state_dict)
    
    def get_updates_from(self, original_state: "AgentState") -> Dict[str, Any]:
        """Get the fields that have changed from the original state."""
        original_dict = original_state.model_dump(exclude={"state_ops"})
        current_dict = self.model_dump(exclude={"state_ops"})
        
        updates = {}
        for key, value in current_dict.items():
            if key not in original_dict or original_dict[key] != value:
                updates[key] = value
                
        return updates
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to state."""
        self.state_ops.record_read(key)
        if hasattr(self, key):
            return getattr(self, key)
        # For dynamic attributes added via extra fields
        return self.__dict__[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like assignment to state."""
        self.state_ops.record_write(key)
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            # For dynamic attributes
            self.__dict__[key] = value

class AgentError(Exception):
    """Base exception for agent errors with structured information."""
    def __init__(self, 
                 message: str, 
                 agent_name: str = None, 
                 details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.agent_name = agent_name
        self.details = details or {}
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to a dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "agent_name": self.agent_name,
            "details": self.details,
            "traceback": self.traceback
        }

class ValidationError(AgentError):
    """Error raised when state validation fails."""
    pass

class ExecutionError(AgentError):
    """Error raised during agent execution."""
    pass

class BaseAgent(ABC):
    """Base class for all agents with enhanced state management and metrics."""
    name: str = "base_agent"
    
    def __init__(self):
        """Initialize the agent."""
        self.logger = logging.getLogger(f"agent.{self.name}")
        self.metrics: AgentMetrics = AgentMetrics()
    
    def _create_state(self, state_dict: Dict[str, Any]) -> AgentState:
        """Create a validated state object from dictionary."""
        try:
            state = AgentState(**state_dict)
            return state
        except Exception as e:
            raise ValidationError(
                f"State validation failed: {str(e)}",
                agent_name=self.name,
                details={"state": state_dict}
            )
    
    def _validate_input(self, state: Dict[str, Any]) -> AgentState:
        """Validate input state and convert to AgentState object."""
        return self._create_state(state)
    
    def _log_start(self, state: Dict[str, Any]) -> None:
        """Log the start of agent execution with state info."""
        state_copy = {k: v for k, v in state.items() if k not in ["goto"]}
        state_str = json.dumps(state_copy, default=str)[:500] + "..." if len(json.dumps(state_copy, default=str)) > 500 else json.dumps(state_copy, default=str)
        
        # Handle missing company key gracefully
        company = state.get('company', 'Unknown')
        self.logger.info(f"Starting {self.name} for {company} with state: {state_str}")  
          
    def _log_completion(self, state: Dict[str, Any]) -> None:
        """Log completion of this agent's execution."""
        company = state.get('company', 'Unknown')
        updates = state.get('goto', 'Unknown')
        
        # Check if metrics attribute exists before trying to access it
        execution_time_msg = ""
        if hasattr(self, 'metrics') and hasattr(self.metrics, 'execution_time_ms'):
            execution_time_msg = f"{self.metrics.execution_time_ms:.2f}ms"
        else:
            execution_time_msg = "unknown time"
        
        self.logger.info(
            f"Completed {self.name} for {company} in "
            f"{execution_time_msg} with updates: {json.dumps(updates)}"
        )    
    def _log_error(self, error: Exception, state: AgentState) -> None:
        """Log an error that occurred during agent execution."""
        self.logger.error(
            f"Error in {self.name} for {state['company']}: {str(error)}",
            exc_info=True
        )
    
    @asynccontextmanager
    async def _agent_context(self, state: AgentState):
        """Context manager for agent execution with resource management."""
        try:
            # Setup resources
            yield
        finally:
            # Cleanup resources
            pass
    
    async def _pre_execute(self, state: AgentState) -> None:
        """Hook executed before running the agent."""
        pass
    
    async def _post_execute(self, original_state: AgentState, updated_state: AgentState) -> None:
        """Hook executed after running the agent."""
        pass
    
    async def _handle_error(self, error: Exception, state: AgentState) -> Dict[str, Any]:
        """Convert an exception to a state update with error information."""
        self._log_error(error, state)
        
        # Create error dictionary with agent name and details
        error_msg = f"Error in {self.name}: {str(error)}"
        if isinstance(error, AgentError):
            error_details = error.to_dict()
        else:
            error_details = {
                "error_type": error.__class__.__name__,
                "traceback": traceback.format_exc()
            }
            
        # Return state updates with error information
        return {
            "goto": "meta_agent",  # Default error handler
            "error": error_msg,
            f"{self.name}_status": "ERROR",
            f"{self.name}_error_details": error_details
        }
    
    @abstractmethod
    async def _execute(self, state: AgentState) -> Dict[str, Any]:
        """Core execution logic to be implemented by subclasses."""
        pass
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with metrics tracking and error handling."""
        self.metrics = AgentMetrics()
        self.metrics.record_start()
        
        try:
            # Validate and create state object
            validated_state = self._validate_input(state)
            
            # Log start and perform pre-execution tasks
            self._log_start(validated_state)
            await self._pre_execute(validated_state)
            
            # Execute the agent within resource context
            async with self._agent_context(validated_state):
                op_start = time.time()
                updates = await self._execute(validated_state)
                op_duration = (time.time() - op_start) * 1000
                self.metrics.record_operation("execute", op_duration)
            
            # Create updated state
            updated_state = validated_state.copy_with_updates(updates)
            
            # Perform post-execution tasks
            await self._post_execute(validated_state, updated_state)
            
            # Record metrics and log completion
            self.metrics.record_end()
            self._log_completion(updated_state)
            
            # Return the updated state as dictionary
            return updated_state.to_dict()
            
        except Exception as e:
            # Handle error and return updated state with error information
            self.metrics.record_end()
            error_updates = await self._handle_error(e, self._create_state(state))
            
            # Merge original state with error updates
            result = {**state, **error_updates}
            return result