from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from utils.logging import ForensicLogger, get_logger

class AgentState(BaseModel):
	goto: Optional[str] = None
	error: Optional[str] = None
	company: str
	industry: Optional[str] = None
	iteration: int = 0

class BaseAgent(ABC):
	name: str = "base_agent"

	@abstractmethod
	async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
		pass

	def _validate_input(self, state: Dict[str, Any]) -> AgentState:
		return AgentState(**state)

	def _log_start(self, state: Dict[str, Any]) -> AgentState:
		self.logger.info(f"Starting {self.name} with state: {state}")
		return AgentState(**state)

	def _log_completion(self, state: Dict[str, Any]) -> AgentState:
		self.logger.info(f"Completed {self.name} with state: {state}")
		return AgentState(**state)
