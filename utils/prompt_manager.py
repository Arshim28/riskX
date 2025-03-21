import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from string import Template
import yaml


class PromptManager:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> Dict[str, str]:
        prompts = {}
        prompt_dir = Path(f"prompts/{self.agent_name}")
        
        if not prompt_dir.exists():
            return prompts
            
        for file_path in prompt_dir.glob("*.txt"):
            prompt_name = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                prompts[prompt_name] = f.read()
                
        for file_path in prompt_dir.glob("*.yaml"):
            prompt_name = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                prompts[prompt_name] = yaml.safe_load(f)
                
        for file_path in prompt_dir.glob("*.json"):
            prompt_name = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                prompts[prompt_name] = json.load(f)
                
        return prompts
        
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found for agent '{self.agent_name}'")
            
        prompt = self.prompts[prompt_name]
        
        if isinstance(prompt, str):
            return Template(prompt).safe_substitute(**kwargs)
        elif isinstance(prompt, dict):
            if "template" in prompt:
                return Template(prompt["template"]).safe_substitute(**kwargs)
            else:
                return json.dumps(prompt)
        else:
            return str(prompt)


_prompt_managers: Dict[str, PromptManager] = {}


def get_prompt_manager(agent_name: str) -> PromptManager:
    if agent_name not in _prompt_managers:
        _prompt_managers[agent_name] = PromptManager(agent_name)
    return _prompt_managers[agent_name]