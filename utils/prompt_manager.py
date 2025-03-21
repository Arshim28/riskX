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
            return (human_rendered, system_rendered)
        except Exception as e:
            logger.error(f"Error rendering template {params.agent_name}/{params.operation}: {e}")
            return None
    
    def get_prompt(self, agent_name: Optional[str] = None, operation: Optional[str] = None, 
                  variables: Optional[Dict[str, Any]] = None, params: Optional[PromptParams] = None) -> Optional[Tuple[str, str]]:
        if params is None:
            if agent_name is None:
                logger.error("No agent name or params provided to get_prompt")
                return None
            params = PromptParams(
                agent_name=agent_name,
                operation=operation or "system",
                variables=variables or {}
            )
        
        return self.render_template(params)
    
    def create_template(self, agent_name: str, operation: str, content: str) -> bool:
        agent_dir = self.prompt_dir / agent_name
        
        try:
            agent_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Created agent directory: {agent_dir}")
        except Exception as e:
            logger.error(f"Error creating agent directory {agent_dir}: {e}")
            
            try:
                flat_template_path = self.prompt_dir / f"{agent_name}_{operation}.j2"
                flat_template_path.write_text(content)
                logger.info(f"Created template file (flat structure): {flat_template_path}")
                return True
            except Exception as e2:
                logger.error(f"Error creating template file {flat_template_path}: {e2}")
                return False
        
        template_path = agent_dir / f"{operation}.j2"
        
        try:
            template_path.write_text(content)
            logger.info(f"Created template file: {template_path}")
            
            template_key = f"{agent_name}/{operation}"
            if template_key in self.templates:
                del self.templates[template_key]
                
            return True
        except Exception as e:
            logger.error(f"Error creating template file {template_path}: {e}")
            return False
    
    def create_template_if_not_exists(self, agent_name: str, operation: str, content: str) -> bool:
        agent_dir = self.prompt_dir / agent_name
        template_path = agent_dir / f"{operation}.j2"
        
        if template_path.exists():
            return True
        
        flat_template_path = self.prompt_dir / f"{agent_name}_{operation}.j2"
        if flat_template_path.exists():
            return True
        
        return self.create_template(agent_name, operation, content)