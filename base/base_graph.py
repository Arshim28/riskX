from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Type
from pydantic import BaseModel, Field

from base.base_agents import BaseAgent

class GraphConfig(BaseModel):
	max_iterations: int = 5
	enable_error_handling: bool = True
	checkpoint_path: Optional[str] = None

class BaseGraph(ABC):
	def __init__(self, config: Optional[Dict[str, Any]] = None):
		self.config = GraphConfig(**(config or {}))
		self.nodes = {}
		self.edges = {}
		self.conditional_edges = {}

	@abstractmethod
	def add_node(self, name: str, agent: Type[BaseAgent]) -> None:
		pass

	@abstractmethod
	def add_edge(self, source: str, target: str) -> None:
		pass

	@abstractmethod
	def add_conditional_edges(self, source: str, router: Callable) -> None:
		pass

	@abstractmethod
	def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
		pass

	def _save_checkpoint(self, state: Dict[str, Any], marker: str = "") -> None:
		pass

	def _load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
		pass