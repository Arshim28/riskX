import os
from typing import Dict, List, Any, Optional, Union, Tuple, Literal

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


ProviderType = Literal["google", "anthropic"]
ModelType = Dict[str, Any]


class LLMProvider:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, ModelType] = {}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_llm(self, model_name: Optional[str] = None) -> ModelType:
        selected_model = model_name or self.config.get("models", {}).get("default", "gemini-2.0-flash")
        
        if selected_model in self.models:
            return self.models[selected_model]
        
        provider = self.config.get("provider", "google")
        temperature = self.config.get("temperature", 0.1)
        
        if provider == "google":
            llm = ChatGoogleGenerativeAI(model=selected_model, temperature=temperature)
        elif provider == "anthropic":
            llm = ChatAnthropic(model=selected_model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.models[selected_model] = llm
        return llm
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_text(self, 
                           prompt: Union[str, List[Tuple[str, str]]], 
                           model_name: Optional[str] = None,
                           temperature: Optional[float] = None) -> str:
        llm = await self.get_llm(model_name)
        
        if temperature is not None:
            llm.temperature = temperature
            
        if isinstance(prompt, str):
            response = llm.invoke(prompt)
        else:
            response = llm.invoke(prompt)
            
        return response.content


_provider_instance = None


def init_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    global _provider_instance
    _provider_instance = LLMProvider(config)
    return _provider_instance


async def get_llm_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is None:
        raise RuntimeError("LLM Provider not initialized. Call init_llm_provider first.")
    return _provider_instance