import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field

import numpy as np
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


class WatchdogTimer:
    def __init__(self, timeout: int, operation_name: str = "Operation"):
        self.timeout = timeout
        self.operation_name = operation_name
        self.timer_task = None
        self.logger = get_logger("watchdog_timer")
        
    async def _timer_callback(self):
        await asyncio.sleep(self.timeout)
        self.logger.warning(f"WATCHDOG ALERT: {self.operation_name} exceeded timeout of {self.timeout} seconds")
        
    def start(self):
        if asyncio.get_event_loop().is_running():
            self.timer_task = asyncio.create_task(self._timer_callback())
        else:
            self.logger.warning(f"Event loop not running, watchdog for {self.operation_name} not started")
        
    def stop(self):
        if self.timer_task and not self.timer_task.done():
            self.timer_task.cancel()


class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None
    dimension: Optional[int] = None


class EmbeddingResult(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    timing_ms: float


class EmbeddingTool(BaseTool):
    name = "embedding_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.api_key = config.get("api_key", os.environ.get("GOOGLE_API_KEY"))
        if not self.api_key:
            raise ValueError("Google API key not provided")
        
        self.model_name = config.get("model", "gemini-embedding-exp-03-07")
        self.embedding_dimension = config.get("dimension")
        self.max_tokens = config.get("max_tokens", 8192)
        self.retry_max_attempts = config.get("retry_max_attempts", 5)
        self.retry_base_delay = config.get("retry_base_delay", 1)
        self.request_delay = config.get("request_delay", 0.2)
        
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        try:

            self.client = genai.Client(api_key=self.api_key)
            self.logger.info(f"Initialized Google Generative AI client with model {self.model_name}")
        except ImportError:
            self.logger.error("Failed to import google-generativeai. Please install it with: pip install google-generativeai")
            raise ImportError("google-generativeai package is required for EmbeddingTool")
    
    async def log_memory_usage(self, marker: str) -> None:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        self.logger.info(f"MEMORY [{marker}]: "
                        f"RSS={memory_info.rss / (1024 * 1024):.1f}MB, "
                        f"VMS={memory_info.vms / (1024 * 1024):.1f}MB")
    
    async def log_array_info(self, name: str, arr: np.ndarray) -> None:
        self.logger.info(f"ARRAY [{name}]: "
                        f"shape={arr.shape}, "
                        f"dtype={arr.dtype}, "
                        f"size={arr.size}, "
                        f"memory={arr.nbytes / (1024 * 1024):.1f}MB")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=(
            retry_if_exception_type(TimeoutError) | 
            retry_if_exception_type(ConnectionError) |
            retry_if_exception_type(Exception)
        )
    )
    async def _get_single_embedding_with_retry(self, text: str) -> np.ndarray:
        return await self._get_single_embedding(text)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, min=5, max=120),
        retry=(
            retry_if_exception_type(TimeoutError) | 
            retry_if_exception_type(ConnectionError) |
            retry_if_exception_type(Exception)
        )
    )
    async def _get_single_embedding_with_batch_retry(self, text: str) -> np.ndarray:
        return await self._get_single_embedding(text)
    
    async def _get_single_embedding(self, text: str) -> np.ndarray:
        config = None
        if self.embedding_dimension is not None:

            config = types.EmbedContentConfig(
                output_dimensionality=self.embedding_dimension
            )

        self.logger.info(f"Generating embedding for text: {text[:100]}...")
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=config
                )
            )

            embedding = np.array(result.embeddings[0].values, dtype=np.float32)
            
            self.embedding_dimension = embedding.shape[0]
            self.logger.info(f"Set embedding dimension to {self.embedding_dimension}")
            

            embedding = embedding / np.linalg.norm(embedding)
            return embedding
            
        except Exception as e:
            is_rate_limit = "429" in str(e)
            is_server_error = any(code in str(e) for code in ["500", "502", "503", "504"])
            is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower()
            
            if is_rate_limit:
                self.logger.error(f"Rate limit exceeded: {e}")
                raise ConnectionError(f"Rate limit exceeded: {e}")
            elif is_server_error:
                self.logger.error(f"Server error: {e}")
                raise ConnectionError(f"Server error: {e}")
            elif is_timeout:
                self.logger.error(f"Timeout error: {e}")
                raise TimeoutError(f"Request timed out: {e}")
            else:
                self.logger.error(f"Unknown error generating embedding: {e}")
                raise Exception(f"Unknown error: {e}")
    
    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        self.logger.warning(f"START EMBEDDING: Generating embeddings for {len(texts)} texts, total text length: {sum(len(t) for t in texts)}")
        await self.log_memory_usage("before_embeddings")
        
        embeddings = []
        failed_indices = []
        
        for i, text in enumerate(texts):
            if i > 0 and i % 10 == 0:
                self.logger.warning(f"EMBEDDING PROGRESS: {i}/{len(texts)} texts processed")
                await self.log_memory_usage(f"embedding_batch_{i}")
                
            if i > 0:
                await asyncio.sleep(self.request_delay)
                
            # Set a watchdog timer for each embedding request
            watchdog = WatchdogTimer(timeout=45, operation_name=f"Embedding request {i+1}/{len(texts)}")
            watchdog.start()
            
            try:
                try:
                    embedding = await self._get_single_embedding_with_retry(text)
                    embeddings.append(embedding)
                except RetryError as e:
                    self.logger.error(f"Failed to get embedding for text index {i} after all retries: {str(e.__cause__)}")
                    failed_indices.append(i)
                    
                    try:
                        embedding = await self._get_single_embedding_with_batch_retry(text)
                        embeddings.append(embedding)
                        failed_indices.pop()
                    except RetryError as batch_e:
                        self.logger.error(f"All batch-level retries failed for text index {i}: {str(batch_e.__cause__)}")
                        if len(embeddings) > 0 and self.embedding_dimension is None:
                            self.embedding_dimension = len(embeddings[0])
                        
                        if self.embedding_dimension is not None:
                            self.logger.warning(f"Using zero vector for failed embedding with dimension {self.embedding_dimension}")
                            zero_embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                            embeddings.append(zero_embedding)
            finally:
                watchdog.stop()
        
        if failed_indices:
            self.logger.error(f"Failed to get embeddings for {len(failed_indices)} texts at indices: {failed_indices}")
        
        result = np.array(embeddings)
        
        await self.log_array_info("embeddings_result", result)
        await self.log_memory_usage("after_embeddings")
        
        return result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def run(self, texts: Union[str, List[str]], **kwargs) -> ToolResult[Dict[str, Any]]:
        try:
            start_time = time.time()
            
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]
                
            # Apply any config overrides from kwargs
            temp_model = kwargs.get('model')
            if temp_model:
                self.model_name = temp_model
                
            temp_dimension = kwargs.get('dimension')
            if temp_dimension:
                self.embedding_dimension = temp_dimension
            
            # Generate embeddings
            embeddings_np = await self.get_embeddings(texts)
            
            # Convert to Python lists for JSON serialization
            embeddings_list = [embedding.tolist() for embedding in embeddings_np]
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = EmbeddingResult(
                embeddings=embeddings_list,
                model=self.model_name,
                dimension=self.embedding_dimension,
                timing_ms=processing_time_ms
            )
            
            self.logger.info(f"Successfully generated {len(embeddings_list)} embeddings in {processing_time_ms:.2f}ms")
            
            return ToolResult(success=True, data=result.model_dump())
        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}")
            return await self._handle_error(e)