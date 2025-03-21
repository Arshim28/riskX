import os
from typing import Dict, List, Any, Optional, Union, Tuple, Literal

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


ProviderType = Literal[
    "google", "anthropic", "openai", "azure_openai", "cohere", 
    "google_vertexai", "fireworks", "ollama", "together", 
    "mistralai", "huggingface", "groq", "bedrock", 
    "dashscope", "xai", "deepseek", "litellm", "gigachat"
]
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
        
        llm = self._create_llm(provider, selected_model, temperature)
        self.models[selected_model] = llm
        return llm
    
    def _create_llm(self, provider: str, model: str, temperature: float) -> ModelType:
        kwargs = {"model": model, "temperature": temperature}
        
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(**kwargs)
        
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(**kwargs)
        
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(**kwargs)
        
        elif provider == "azure_openai":
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(azure_deployment=model, **kwargs)
        
        elif provider == "cohere":
            from langchain_cohere import ChatCohere
            return ChatCohere(**kwargs)
        
        elif provider == "google_vertexai":
            from langchain_google_vertexai import ChatVertexAI
            return ChatVertexAI(**kwargs)
        
        elif provider == "fireworks":
            from langchain_fireworks import ChatFireworks
            return ChatFireworks(**kwargs)
        
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), **kwargs)
        
        elif provider == "together":
            from langchain_together import ChatTogether
            return ChatTogether(**kwargs)
        
        elif provider == "mistralai":
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(**kwargs)
        
        elif provider == "huggingface":
            from langchain_huggingface import ChatHuggingFace
            return ChatHuggingFace(model_id=model, **{k: v for k, v in kwargs.items() if k != "model"})
        
        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(**kwargs)
        
        elif provider == "bedrock":
            from langchain_aws import ChatBedrock
            return ChatBedrock(model_id=model, model_kwargs={k: v for k, v in kwargs.items() if k != "model"})
        
        elif provider == "dashscope":
            from langchain_dashscope import ChatDashScope
            return ChatDashScope(**kwargs)
        
        elif provider == "xai":
            from langchain_xai import ChatXAI
            return ChatXAI(**kwargs)
        
        elif provider == "deepseek":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                openai_api_base='https://api.deepseek.com',
                openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
                **kwargs
            )
        
        elif provider == "litellm":
            from langchain_community.chat_models.litellm import ChatLiteLLM
            return ChatLiteLLM(**kwargs)
        
        elif provider == "gigachat":
            from langchain_gigachat.chat_models import GigaChat
            return GigaChat(**{k: v for k, v in kwargs.items() if k != "model"})
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
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