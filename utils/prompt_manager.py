from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PromptParams(BaseModel):
    agent_name: str
    operation: str = "system"
    variables: Dict[str, Any] = Field(default_factory=dict)

class PromptManager:
    def __init__(self, prompt_dir: Optional[str] = None):
        self.prompt_dir = self._resolve_prompt_directory(prompt_dir)
        self.env = self._setup_jinja_environment()
        self.templates: Dict[str, Tuple[Template, Template]] = {}
        logger.info(f"PromptManager initialized with template directory: {self.prompt_dir}")
    
    def _resolve_prompt_directory(self, prompt_dir: Optional[str] = None) -> Path:
        if prompt_dir:
            path = Path(prompt_dir)
        else:
            path = Path(__file__).resolve().parent.parent / "prompts"
        
        if not path.exists():
            raise FileNotFoundError(f"Prompt directory not found: {path}")
            
        return path
    
    def _setup_jinja_environment(self) -> Environment:
        return Environment(
            loader=FileSystemLoader([str(self.prompt_dir)]),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def load_template(self, agent_name: str, operation: str = "system") -> Optional[Tuple[Template, Template]]:
        template_key = f"{agent_name}/{operation}"
        
        if template_key in self.templates:
            return self.templates[template_key]
        
        agent_dir = self.prompt_dir / agent_name
        if agent_dir.exists() and agent_dir.is_dir():
            human_template_filename = f"{operation}_human.j2"
            system_template_filename = f"{operation}_system.j2"
            
            try:
                human_template = self.env.get_template(f"{agent_name}/{human_template_filename}")
                system_template = self.env.get_template(f"{agent_name}/{system_template_filename}")
                
                self.templates[template_key] = (human_template, system_template)
                logger.debug(f"Loaded templates: {agent_name}/{human_template_filename}, {agent_name}/{system_template_filename}")
                
                return (human_template, system_template)
            except Exception as e:
                logger.debug(f"Error loading template for {agent_name}/{operation}: {e}")
                return None
        
        return None
    
    def render_template(self, params: PromptParams) -> Optional[Tuple[str, str]]:
        templates = self.load_template(params.agent_name, params.operation)
        
        if not templates:
            return None
        
        human_template, system_template = templates
        
        try:
            human_rendered = human_template.render(**params.variables)
            system_rendered = system_template.render(**params.variables)
            return (system_rendered, human_rendered)
        except Exception as e:
            logger.error(f"Error rendering template {params.agent_name}/{params.operation}: {e}")
            return None
    
    def get_prompt(self, agent_name: str, operation: str, 
                  variables: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, str]]:
        
        params = PromptParams(
            agent_name=agent_name,
            operation=operation,
            variables=variables or {}
        )
        
        return self.render_template(params)


# Global prompt manager instance
_prompt_manager_instance = None


def init_prompt_manager(prompt_dir: Optional[str] = None) -> PromptManager:
    """Initialize the global prompt manager instance."""
    global _prompt_manager_instance
    _prompt_manager_instance = PromptManager(prompt_dir)
    return _prompt_manager_instance


def get_prompt_manager(agent_name: Optional[str] = None) -> PromptManager:
    """Get the global prompt manager instance. If not initialized, create a default one."""
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance