import os
import json
import time
import enum
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Literal, TypeVar, Generic, Callable, Type
from pydantic import BaseModel, Field, ValidationError, validator

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from tenacity.wait import wait_base
import logging

# Import LLM providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

T = TypeVar('T')

# Define standardized types
ProviderType = Literal[
    "google", "anthropic", "openai", "azure_openai", "cohere", 
    "google_vertexai", "fireworks", "ollama", "together", 
    "mistralai", "huggingface", "groq", "bedrock", 
    "dashscope", "xai", "deepseek", "litellm", "gigachat"
]

MessageRole = Literal["system", "human", "assistant", "tool", "function"]
ModelType = Dict[str, Any]

class LLMException(Exception):
    """Base exception class for LLM-related errors."""
    pass

class RateLimitError(LLMException):
    """Exception raised when hitting rate limits."""
    pass

class AuthenticationError(LLMException):
    """Exception raised for authentication issues."""
    pass

class ContentFilterError(LLMException):
    """Exception raised when content is filtered by the provider."""
    pass

class ServiceUnavailableError(LLMException):
    """Exception raised when the service is unavailable."""
    pass

class InvalidRequestError(LLMException):
    """Exception raised for invalid requests."""
    pass

class ResponseValidationError(LLMException):
    """Exception raised when response validation fails."""
    pass

class ResponseFormat(BaseModel):
    """Base class for response formats."""
    pass

class JsonResponseFormat(ResponseFormat):
    """Format specification for JSON responses."""
    schema: Optional[Dict[str, Any]] = None
    
    @validator('schema')
    def validate_schema(cls, v):
        if v is not None:
            try:
                # Basic schema validation
                if not isinstance(v, dict):
                    raise ValueError("Schema must be a dictionary")
            except Exception as e:
                raise ValueError(f"Invalid schema: {str(e)}")
        return v

class TextResponseFormat(ResponseFormat):
    """Format specification for text responses."""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    @validator('min_length', 'max_length')
    def validate_length(cls, v):
        if v is not None and v < 0:
            raise ValueError("Length cannot be negative")
        return v

class MessageContent(BaseModel):
    """Model for message content."""
    role: MessageRole
    content: str

class LLMResponse(BaseModel, Generic[T]):
    """Standardized response from LLM providers."""
    content: T
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    response_ms: Optional[int] = None
    raw_response: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LLMRequest(BaseModel):
    """Standardized request to LLM providers."""
    messages: List[Union[Tuple[str, str], MessageContent]]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
    stop_sequences: Optional[List[str]] = None
    timeout: Optional[float] = None
    provider_specific: Dict[str, Any] = Field(default_factory=dict)

class ProviderConfig(BaseModel):
    """Configuration for a provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: str
    timeout: float = 30.0
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    additional_config: Dict[str, Any] = Field(default_factory=dict)

class RetryStrategy(enum.Enum):
    CONSTANT = "constant"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"

class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_retries: int = 5
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    initial_delay: float = 1.0
    max_delay: float = 60.0
    retry_rate_limits: bool = True
    retry_auth_errors: bool = False
    retry_connection_errors: bool = True
    retry_service_unavailable: bool = True
    jitter: bool = True

class ValidationConfig(BaseModel):
    """Configuration for response validation."""
    validate_json: bool = True
    check_empty_responses: bool = True
    check_content_length: bool = False
    min_content_length: Optional[int] = None
    max_content_length: Optional[int] = None
    custom_validators: List[Callable[[str], bool]] = Field(default_factory=list)

class LLMProviderConfig(BaseModel):
    """Main configuration for LLM provider."""
    default_provider: ProviderType = "google"
    default_model: Optional[str] = None
    providers: Dict[ProviderType, ProviderConfig] = Field(default_factory=dict)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging_level: int = logging.INFO

class AdaptiveWaitStrategy(wait_base):
    """Custom wait strategy that adapts based on error types."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.exp_wait = wait_exponential(
            multiplier=config.initial_delay,
            min=config.initial_delay,
            max=config.max_delay
        )
        
    def __call__(self, retry_state):
        exception = retry_state.outcome.exception()
        
        # Adjust wait time based on exception type
        if isinstance(exception, RateLimitError):
            # Use longer wait for rate limit errors
            return min(self.exp_wait(retry_state) * 2, self.config.max_delay)
        elif isinstance(exception, ServiceUnavailableError):
            # Use moderate wait for service issues
            return min(self.exp_wait(retry_state) * 1.5, self.config.max_delay)
        else:
            # Use standard wait for other errors
            return self.exp_wait(retry_state)

class LLMProvider:
    def __init__(self, config: Union[Dict[str, Any], LLMProviderConfig]):
        if isinstance(config, dict):
            try:
                # Handle nested dictionaries within the config
                if "llm_provider" in config:
                    self.config = LLMProviderConfig(**config.get("llm_provider", {}))
                else:
                    self.config = LLMProviderConfig(**config)
            except Exception as e:
                print(f"Error parsing LLM provider config: {str(e)}")
                # Create default configuration
                self.config = LLMProviderConfig()
        else:
            self.config = config
            
        # Set up logging
        self.logger = logging.getLogger("llm_provider")
        self.logger.setLevel(self.config.logging_level)
        
        # Add a console handler if none exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.config.logging_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.models = {}
        self._init_providers()
        
    def _init_providers(self) -> None:
        """Initialize configured providers."""
        # Set default API keys from environment if not provided
        for provider_name, provider_config in self.config.providers.items():
            if provider_config.api_key is None:
                env_var = f"{provider_name.upper()}_API_KEY"
                provider_config.api_key = os.environ.get(env_var)
            elif provider_config.api_key.startswith("${") and provider_config.api_key.endswith("}"):
                # Handle environment variable placeholders like ${VAR_NAME}
                env_var = provider_config.api_key[2:-1]  # Remove ${ and }
                provider_config.api_key = os.environ.get(env_var)
                
            # Validate API keys
            if not provider_config.api_key:
                self.logger.warning(f"No API key found for provider '{provider_name}'. "
                                   f"Set the {provider_name.upper()}_API_KEY environment variable "
                                   f"or configure it in the config file.")
                
        # Log initialization status
        configured_providers = [p for p, c in self.config.providers.items() if c.api_key]
        self.logger.info(f"Initialized LLM provider with {len(configured_providers)} configured providers: {configured_providers}")
        print(f"Initialized LLM provider with {len(configured_providers)} configured providers: {configured_providers}")
    
    def _get_retry_decorator(self):
        """Create a retry decorator based on config."""
        config = self.config.retry
        
        # Define which exceptions to retry
        retry_exceptions = []
        if config.retry_rate_limits:
            retry_exceptions.append(RateLimitError)
        if config.retry_connection_errors:
            retry_exceptions.append(ConnectionError)
        if config.retry_service_unavailable:
            retry_exceptions.append(ServiceUnavailableError)
        if config.retry_auth_errors:
            retry_exceptions.append(AuthenticationError)
            
        # Fall back to standard exceptions if none specified
        if not retry_exceptions:
            retry_exceptions = [Exception]
            
        retry_condition = retry_if_exception_type(tuple(retry_exceptions))
        
        # Create wait strategy
        wait_strategy = AdaptiveWaitStrategy(config)
        
        # Create and return the retry decorator
        return retry(
            stop=stop_after_attempt(config.max_retries),
            wait=wait_strategy,
            retry=retry_condition,
            before_sleep=before_sleep_log(self.logger, logging.WARNING)
        )
        
    def _map_provider_error(self, provider: str, error: Exception) -> Exception:
        """Map provider-specific errors to standardized exceptions."""
        error_str = str(error).lower()
        
        # Rate limit errors
        if any(term in error_str for term in ["rate limit", "ratelimit", "too many requests", "429"]):
            return RateLimitError(f"{provider} rate limit exceeded: {str(error)}")
            
        # Authentication errors
        elif any(term in error_str for term in ["auth", "key", "credentials", "unauthorized", "permission", "401"]):
            return AuthenticationError(f"{provider} authentication error: {str(error)}")
            
        # Content filter errors
        elif any(term in error_str for term in ["content filter", "filtered", "content policy", "moderation"]):
            return ContentFilterError(f"{provider} content filtered: {str(error)}")
            
        # Service unavailable
        elif any(term in error_str for term in ["unavailable", "overloaded", "maintenance", "503", "502"]):
            return ServiceUnavailableError(f"{provider} service unavailable: {str(error)}")
            
        # Invalid request
        elif any(term in error_str for term in ["invalid", "malformed", "bad request", "400"]):
            return InvalidRequestError(f"{provider} invalid request: {str(error)}")
            
        # Default case
        return LLMException(f"{provider} error: {str(error)}")
    
    def _create_llm(self, provider: ProviderType, model: str, config: ProviderConfig) -> ModelType:
        """Create an LLM instance for the specified provider and model."""
        base_kwargs = {
            "model": model,
            "temperature": 0.1,
            "timeout": config.timeout
        }
        
        # Add API key if available
        if config.api_key:
            base_kwargs["api_key"] = config.api_key
            
        # Add base URL if available
        if config.base_url:
            base_kwargs["base_url"] = config.base_url
            
        # Add any additional configuration
        # Need special handling for nested dictionaries
        additional_config = config.additional_config.copy() if config.additional_config else {}
        
        # For each key in additional_config, if it's not a nested structure, add it to base_kwargs
        for key, value in additional_config.items():
            if isinstance(value, dict):
                # For nested dictionaries, we'll handle them specially for each provider
                continue
            base_kwargs[key] = value
        
        self.logger.info(f"Creating {provider} LLM instance for model {model}")
        print(f"Creating {provider} LLM instance for model {model}")
        
        try:
            if provider == "google":
                # Handle safety settings format for Google provider
                if "safety_settings" in base_kwargs.get("additional_config", {}):
                    # Extract safety settings and ensure they're formatted correctly
                    safety_settings = base_kwargs["additional_config"].pop("safety_settings", {})
                    
                    # Convert any string keys/values to integers
                    formatted_safety_settings = {}
                    for key, value in safety_settings.items():
                        # Convert key to int if it's a string that can be parsed as int
                        try:
                            key_int = int(key) if isinstance(key, str) and key.isdigit() else key
                        except (ValueError, TypeError):
                            # Map string harm categories to their integer values
                            harm_category_map = {
                                "HARASSMENT": 1,
                                "HATE_SPEECH": 2,
                                "SEXUALLY_EXPLICIT": 3,
                                "DANGEROUS_CONTENT": 4
                            }
                            key_int = harm_category_map.get(key, key)
                        
                        # Convert value to int if it's a string that can be parsed as int
                        try:
                            value_int = int(value) if isinstance(value, str) and value.isdigit() else value
                        except (ValueError, TypeError):
                            # Map string threshold values to their integer values
                            threshold_map = {
                                "block_none": 0,
                                "BLOCK_NONE": 0,
                                "block_low": 1,
                                "BLOCK_LOW": 1,
                                "block_medium": 2,
                                "BLOCK_MEDIUM": 2,
                                "block_high": 3,
                                "BLOCK_HIGH": 3,
                                "block_very_high": 4, 
                                "BLOCK_VERY_HIGH": 4
                            }
                            value_int = threshold_map.get(value, value)
                            
                        # Add to formatted safety settings if both key and value are integers
                        if isinstance(key_int, int) and isinstance(value_int, int):
                            formatted_safety_settings[key_int] = value_int
                    
                    # Set the formatted safety settings
                    if formatted_safety_settings:
                        base_kwargs["safety_settings"] = formatted_safety_settings
                        print(f"Configured safety settings for Google provider: {formatted_safety_settings}")
                    
                return ChatGoogleGenerativeAI(**base_kwargs)
                
            elif provider == "anthropic":
                return ChatAnthropic(**base_kwargs)
                
            elif provider == "openai":
                return ChatOpenAI(**base_kwargs)
                
            elif provider == "azure_openai":
                from langchain_openai import AzureChatOpenAI
                azure_kwargs = {k: v for k, v in base_kwargs.items() if k != "model"}
                return AzureChatOpenAI(azure_deployment=model, **azure_kwargs)
                
            elif provider == "cohere":
                from langchain_cohere import ChatCohere
                return ChatCohere(**base_kwargs)
                
            elif provider == "google_vertexai":
                from langchain_google_vertexai import ChatVertexAI
                return ChatVertexAI(**base_kwargs)
                
            elif provider == "fireworks":
                from langchain_fireworks import ChatFireworks
                return ChatFireworks(**base_kwargs)
                
            elif provider == "ollama":
                from langchain_ollama import ChatOllama
                ollama_base_url = config.base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                return ChatOllama(base_url=ollama_base_url, **{k: v for k, v in base_kwargs.items() if k != "base_url"})
                
            elif provider == "together":
                from langchain_together import ChatTogether
                return ChatTogether(**base_kwargs)
                
            elif provider == "mistralai":
                from langchain_mistralai import ChatMistralAI
                return ChatMistralAI(**base_kwargs)
                
            elif provider == "huggingface":
                from langchain_huggingface import ChatHuggingFace
                return ChatHuggingFace(model_id=model, **{k: v for k, v in base_kwargs.items() if k != "model"})
                
            elif provider == "groq":
                from langchain_groq import ChatGroq
                return ChatGroq(**base_kwargs)
                
            elif provider == "bedrock":
                from langchain_aws import ChatBedrock
                return ChatBedrock(model_id=model, model_kwargs={k: v for k, v in base_kwargs.items() if k != "model"})
                
            elif provider == "dashscope":
                from langchain_dashscope import ChatDashScope
                return ChatDashScope(**base_kwargs)
                
            elif provider == "xai":
                from langchain_xai import ChatXAI
                return ChatXAI(**base_kwargs)
                
            elif provider == "deepseek":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    openai_api_base='https://api.deepseek.com',
                    openai_api_key=config.api_key or os.environ.get("DEEPSEEK_API_KEY"),
                    **{k: v for k, v in base_kwargs.items() if k != "api_key"}
                )
                
            elif provider == "litellm":
                from langchain_community.chat_models.litellm import ChatLiteLLM
                return ChatLiteLLM(**base_kwargs)
                
            elif provider == "gigachat":
                from langchain_gigachat.chat_models import GigaChat
                return GigaChat(**{k: v for k, v in base_kwargs.items() if k != "model"})
                
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Error creating LLM instance for {provider}/{model}: {e}")
            raise self._map_provider_error(provider, e)
    
    async def get_llm(self, provider: Optional[ProviderType] = None, model: Optional[str] = None) -> ModelType:
        """Get an LLM instance for the specified provider and model."""
        selected_provider = provider or self.config.default_provider
        
        # Get provider config
        if selected_provider not in self.config.providers:
            error_msg = f"Provider '{selected_provider}' not configured"
            self.logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            print(f"Available providers: {list(self.config.providers.keys())}")
            raise ValueError(error_msg)
            
        provider_config = self.config.providers[selected_provider]
        
        # Check if the provider has a valid API key
        if not provider_config.api_key:
            error_msg = f"No API key available for provider '{selected_provider}'"
            self.logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            print(f"Please set the {selected_provider.upper()}_API_KEY environment variable")
            raise AuthenticationError(error_msg)
        
        # Determine model to use
        selected_model = model or self.config.default_model or provider_config.default_model
        
        # Check if the model is specified
        if not selected_model:
            error_msg = f"No model specified for provider '{selected_provider}'"
            self.logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Check cache
        model_key = f"{selected_provider}:{selected_model}"
        if model_key in self.models:
            self.logger.debug(f"Using cached LLM instance for {model_key}")
            return self.models[model_key]
        
        # Create new LLM instance
        self.logger.info(f"Creating new LLM instance for {model_key}")
        print(f"Creating new LLM instance for {model_key}")
        llm = self._create_llm(selected_provider, selected_model, provider_config)
        self.models[model_key] = llm
        
        return llm
    
    def _standardize_messages(self, messages: List[Union[Tuple[str, str], MessageContent]]) -> List[Dict[str, str]]:
        """Convert various message formats to a standardized format."""
        standardized = []
        
        for msg in messages:
            if isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                standardized.append({"role": role, "content": content})
            elif isinstance(msg, MessageContent):
                standardized.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                standardized.append(msg)
            else:
                raise ValueError(f"Unsupported message format: {msg}")
                
        return standardized
    
    def _validate_response(self, response: str, request: LLMRequest) -> None:
        """Validate the response according to validation config and request format."""
        validation_config = self.config.validation
        
        # Check for empty responses
        if validation_config.check_empty_responses and not response.strip():
            raise ResponseValidationError("Empty response received")
            
        # Check content length if configured
        if validation_config.check_content_length:
            length = len(response)
            if validation_config.min_content_length and length < validation_config.min_content_length:
                raise ResponseValidationError(f"Response too short: {length} chars (min: {validation_config.min_content_length})")
            if validation_config.max_content_length and length > validation_config.max_content_length:
                raise ResponseValidationError(f"Response too long: {length} chars (max: {validation_config.max_content_length})")
        
        # Validate response format if specified
        if request.response_format:
            if isinstance(request.response_format, JsonResponseFormat) and validation_config.validate_json:
                try:
                    parsed = json.loads(response)
                    
                    # Validate against schema if provided
                    if request.response_format.schema:
                        # This is a simplified schema validation
                        # In a production environment, you might want to use a library like jsonschema
                        self._validate_against_schema(parsed, request.response_format.schema)
                        
                except json.JSONDecodeError:
                    raise ResponseValidationError("Response is not valid JSON")
                    
            elif isinstance(request.response_format, TextResponseFormat):
                if request.response_format.min_length and len(response) < request.response_format.min_length:
                    raise ResponseValidationError(f"Response text too short: {len(response)} chars")
                if request.response_format.max_length and len(response) > request.response_format.max_length:
                    raise ResponseValidationError(f"Response text too long: {len(response)} chars")
        
        # Run custom validators
        for validator in validation_config.custom_validators:
            if not validator(response):
                raise ResponseValidationError(f"Custom validation failed for response")
    
    def _validate_against_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """Validate data against a simple schema definition."""
        if not isinstance(schema, dict):
            raise ResponseValidationError("Schema must be a dictionary")
            
        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                raise ResponseValidationError(f"Expected object, got {type(data).__name__}")
            elif expected_type == "array" and not isinstance(data, list):
                raise ResponseValidationError(f"Expected array, got {type(data).__name__}")
            elif expected_type == "string" and not isinstance(data, str):
                raise ResponseValidationError(f"Expected string, got {type(data).__name__}")
            elif expected_type == "number" and not isinstance(data, (int, float)):
                raise ResponseValidationError(f"Expected number, got {type(data).__name__}")
            elif expected_type == "boolean" and not isinstance(data, bool):
                raise ResponseValidationError(f"Expected boolean, got {type(data).__name__}")
                
        if "properties" in schema and isinstance(data, dict):
            for key, prop_schema in schema["properties"].items():
                if "required" in schema and key in schema["required"] and key not in data:
                    raise ResponseValidationError(f"Missing required property: {key}")
                if key in data:
                    self._validate_against_schema(data[key], prop_schema)
    
    async def generate_text(self, 
                           messages: Union[str, List[Union[Tuple[str, str], MessageContent]], LLMRequest], 
                           model: Optional[str] = None,
                           provider: Optional[ProviderType] = None,
                           temperature: Optional[float] = None,
                           response_format: Optional[ResponseFormat] = None) -> Union[str, LLMResponse[str]]:
        """Generate text using the configured LLM."""
        # Convert simple string to a request with a single human message
        if isinstance(messages, str):
            request = LLMRequest(
                messages=[("human", messages)],
                model=model,
                temperature=temperature,
                response_format=response_format
            )
        # Convert message list to a request
        elif isinstance(messages, list):
            request = LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                response_format=response_format
            )
        # Use the request object directly
        elif isinstance(messages, LLMRequest):
            request = messages
        else:
            raise ValueError(f"Unsupported message type: {type(messages)}")
            
        # Apply retry decorator
        retry_decorator = self._get_retry_decorator()
        
        @retry_decorator
        async def _generate_with_retry() -> LLMResponse[str]:
            start_time = time.time()
            
            try:
                # Get the LLM instance
                llm = await self.get_llm(provider, request.model)
                
                # Standardize messages
                std_messages = self._standardize_messages(request.messages)
                
                # Set temperature if provided
                if request.temperature is not None:
                    llm.temperature = request.temperature
                elif temperature is not None:
                    llm.temperature = temperature
                
                # Get provider and model information
                provider_info = getattr(llm, "_llm_type", "unknown")
                model_info = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
                
                # Invoke the model
                self.logger.info(f"Calling {provider_info} with model {model_info}")
                print(f"Calling {provider_info} with model {model_info}")
                response = llm.invoke(std_messages)
                
                # Get the content
                content = response.content
                
                # Validate response
                self._validate_response(content, request)
                
                # Calculate response time
                response_ms = int((time.time() - start_time) * 1000)
                
                # Create and return standardized response
                return LLMResponse(
                    content=content,
                    model=model_info,
                    provider=provider_info,
                    response_ms=response_ms,
                    raw_response=response,
                    metadata={}
                )
                
            except Exception as e:
                # Map to a standardized exception
                mapped_error = self._map_provider_error(provider or self.config.default_provider, e)
                self.logger.error(f"Error generating text: {mapped_error}")
                raise mapped_error
        
        # Generate the response
        response = await _generate_with_retry()
        
        # Return only the content string for backward compatibility
        return response.content
    
    async def generate_structured(self, 
                               messages: Union[List[Union[Tuple[str, str], MessageContent]], LLMRequest],
                               output_schema: Type[T],
                               model: Optional[str] = None,
                               provider: Optional[ProviderType] = None,
                               temperature: Optional[float] = None,
                               max_attempts: int = 3) -> LLMResponse[T]:
        """Generate a structured response validated against a Pydantic model."""
        # Create a response format for JSON
        json_format = JsonResponseFormat()
        
        if isinstance(messages, list):
            request = LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                response_format=json_format
            )
        elif isinstance(messages, LLMRequest):
            request = messages.copy()
            request.response_format = json_format
        else:
            raise ValueError(f"Unsupported message type for structured generation: {type(messages)}")
        
        # Try multiple times to get a valid response
        last_error = None
        for attempt in range(max_attempts):
            try:
                # Generate text
                response_text = await self.generate_text(request, model, provider, temperature)
                
                # Try to parse the JSON
                try:
                    # Strip any markdown code block markers
                    cleaned_text = response_text
                    if "```json" in cleaned_text:
                        cleaned_text = cleaned_text.split("```json", 1)[1].split("```", 1)[0]
                    elif "```" in cleaned_text:
                        cleaned_text = cleaned_text.split("```", 1)[1].split("```", 1)[0]
                    
                    # Parse the JSON
                    parsed_data = json.loads(cleaned_text.strip())
                    
                    # Validate against the schema
                    validated_data = output_schema.parse_obj(parsed_data)
                    
                    # Calculate response time (approximate since we don't have the original)
                    response_ms = 0  # We don't have the actual time here
                    
                    # Return the validated response
                    return LLMResponse(
                        content=validated_data,
                        model=request.model or "unknown",
                        provider=provider or self.config.default_provider,
                        response_ms=response_ms,
                        metadata={"attempts": attempt + 1}
                    )
                    
                except (json.JSONDecodeError, ValidationError) as e:
                    last_error = e
                    self.logger.warning(f"Attempt {attempt+1}/{max_attempts}: Invalid response format: {str(e)}")
                    
                    # Add a more explicit instruction for the next attempt
                    if attempt < max_attempts - 1:
                        # Add a clarification message
                        if isinstance(messages, list):
                            fixed_messages = list(messages)
                            fixed_messages.append(("human", f"The previous response was not in the correct format. Please provide a valid JSON response that matches this schema: {output_schema.schema_json()}"))
                            request.messages = fixed_messages
                        
            except Exception as e:
                last_error = e
                self.logger.error(f"Attempt {attempt+1}/{max_attempts}: Error in generate_structured: {e}")
                
                # Don't retry on certain errors
                if isinstance(e, (AuthenticationError, InvalidRequestError)):
                    break
        
        # If we've exhausted all attempts, raise the last error
        if last_error:
            if isinstance(last_error, ValidationError):
                raise ResponseValidationError(f"Failed to validate response against schema after {max_attempts} attempts: {str(last_error)}")
            elif isinstance(last_error, json.JSONDecodeError):
                raise ResponseValidationError(f"Failed to parse JSON after {max_attempts} attempts: {str(last_error)}")
            else:
                raise last_error
        else:
            raise ResponseValidationError(f"Failed to generate a valid structured response after {max_attempts} attempts")


# Global provider instance
_provider_instance = None

def init_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """Initialize the global LLM provider instance."""
    global _provider_instance
    
    # Log the configuration
    config_str = str(config)
    if len(config_str) > 500:
        config_str = config_str[:500] + "..."
    print(f"Initializing LLM provider with config: {config_str}")
    
    # Process environment variables in the config
    def process_env_vars(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]  # Remove ${ and }
            return os.environ.get(env_var)
        return value

    def process_config(config_dict):
        if not isinstance(config_dict, dict):
            return config_dict
            
        processed = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                processed[key] = process_config(value)
            elif isinstance(value, list):
                processed[key] = [process_config(item) if isinstance(item, dict) else process_env_vars(item) for item in value]
            else:
                processed[key] = process_env_vars(value)
        return processed
        
    # Process any environment variables in the config
    if isinstance(config, dict):
        config = process_config(config)
    
    try:
        _provider_instance = LLMProvider(config)
        print(f"LLM provider initialized with {len(_provider_instance.config.providers)} providers")
        return _provider_instance
    except Exception as e:
        print(f"Error initializing LLM provider: {str(e)}")
        # Create a minimal working provider as fallback
        try:
            # Try to create a minimal configuration with available environment variables
            fallback_config = {
                "default_provider": "google", 
                "providers": {}
            }
            
            # Try to get API keys from environment
            for provider in ["google", "anthropic", "openai"]:
                api_key = os.environ.get(f"{provider.upper()}_API_KEY")
                if api_key:
                    fallback_config["providers"][provider] = {
                        "api_key": api_key,
                        "default_model": "gemini-2.0-flash" if provider == "google" else 
                                        "claude-3-sonnet-20240229" if provider == "anthropic" else
                                        "gpt-4-0125-preview"
                    }
            
            if not fallback_config["providers"]:
                print("WARNING: No API keys found in environment variables. LLM provider initialization will fail.")
                
            print(f"Using fallback configuration with providers: {list(fallback_config['providers'].keys())}")
            _provider_instance = LLMProvider(fallback_config)
            return _provider_instance
        except Exception as fallback_error:
            print(f"Failed to create fallback LLM provider: {str(fallback_error)}")
            # Create an empty provider that will raise appropriate errors when used
            _provider_instance = LLMProvider({"providers": {}})
            return _provider_instance

async def get_llm_provider() -> LLMProvider:
    """Get the global LLM provider instance."""
    global _provider_instance
    if _provider_instance is None:
        print("WARNING: LLM Provider not initialized. Creating a default instance.")
        # Try to create a default instance
        _provider_instance = LLMProvider({
            "default_provider": "google",
            "providers": {}
        })
        
        # Check if any API keys are available in environment variables
        for provider in ["google", "anthropic", "openai"]:
            api_key = os.environ.get(f"{provider.upper()}_API_KEY")
            if api_key:
                print(f"Found {provider.upper()}_API_KEY in environment variables")
                _provider_instance.config.providers[provider] = ProviderConfig(
                    api_key=api_key,
                    default_model="gemini-2.0-flash" if provider == "google" else 
                                "claude-3-sonnet-20240229" if provider == "anthropic" else
                                "gpt-4-0125-preview"
                )
        
    return _provider_instance