import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
import aiohttp

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


class RateLimiter:
    """Rate limiter that manages request rates to prevent API throttling"""
    
    def __init__(self, max_requests_per_min: int = 300, max_burst: int = 10):
        self.max_requests_per_min = max_requests_per_min
        self.max_burst = max_burst
        self.tokens = max_burst  # Start with max tokens
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        self.logger = get_logger("rate_limiter")
    
    async def acquire(self):
        """Acquire a token to make a request, waiting if necessary"""
        async with self.lock:
            await self._update_tokens()
            
            if self.tokens < 1:
                wait_time = (60.0 / self.max_requests_per_min) * (1 - self.tokens)
                self.logger.info(f"Rate limit reached. Waiting {wait_time:.2f}s before next request.")
                await asyncio.sleep(wait_time)
                await self._update_tokens()
            
            self.tokens -= 1
    
    async def _update_tokens(self):
        """Update available tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_update
        
        # Tokens regenerate at rate of max_requests_per_min / 60 per second
        new_tokens = elapsed * (self.max_requests_per_min / 60.0)
        self.tokens = min(self.max_burst, self.tokens + new_tokens)
        self.last_update = now


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


class BatchProcessor:
    """Helper class to batch requests efficiently"""
    
    def __init__(self, optimal_batch_size: int = 20, max_batch_size: int = 100):
        self.optimal_batch_size = optimal_batch_size
        self.max_batch_size = max_batch_size
        self.logger = get_logger("batch_processor")
    
    def create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Divide items into optimally sized batches"""
        if not items:
            return []
            
        # Use optimal batch size unless we have very few items
        if len(items) <= self.optimal_batch_size:
            return [items]
            
        # For large inputs, create optimal-sized batches
        batches = []
        for i in range(0, len(items), self.optimal_batch_size):
            end = min(i + self.optimal_batch_size, len(items))
            batches.append(items[i:end])
            
        self.logger.info(f"Created {len(batches)} batches from {len(items)} items")
        return batches


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
        
        # Create a semaphore to limit concurrent API requests
        self.semaphore = asyncio.Semaphore(config.get("max_concurrent_requests", 5))
        
        # Create rate limiter
        self.rate_limiter = RateLimiter(
            max_requests_per_min=config.get("max_requests_per_min", 300),
            max_burst=config.get("max_burst", 10)
        )
        
        # Create batch processor
        self.batch_processor = BatchProcessor(
            optimal_batch_size=config.get("optimal_batch_size", 20),
            max_batch_size=config.get("max_batch_size", 100)
        )
        
        # Create thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 5))
        
        self.client = None
        self.initialize_client()
        
        # Track API usage stats
        self.api_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retried_requests": 0,
            "total_tokens": 0,
            "total_time_ms": 0
        }
    
    def initialize_client(self):
        try:
            from google import genai
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
    async def _get_embedding_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts"""
        if not texts:
            return []
            
        # Update API stats
        self.api_stats["total_requests"] += 1
        start_time = time.time()
        
        # Acquire a token from rate limiter
        await self.rate_limiter.acquire()
        
        # Wait for semaphore to limit concurrent requests
        async with self.semaphore:
            try:
                from google.genai import types
                config = None
                if self.embedding_dimension is not None:
                    config = types.EmbedContentConfig(
                        output_dimensionality=self.embedding_dimension
                    )
                
                # Count estimated tokens
                est_tokens = sum(len(text.split()) * 1.3 for text in texts)
                self.api_stats["total_tokens"] += est_tokens
                
                # Set up watchdog timer
                watchdog = WatchdogTimer(timeout=45, operation_name=f"Batch embedding request ({len(texts)} texts)")
                watchdog.start()
                
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Run the API call in a thread to avoid blocking
                    result = await loop.run_in_executor(
                        self.executor,
                        lambda: self.client.models.embed_content_batch(
                            model=self.model_name,
                            contents_batch=texts,
                            config=config
                        )
                    )
                    
                    # Process results
                    embeddings = []
                    for embedding_result in result.embeddings:
                        embedding = np.array(embedding_result[0].values, dtype=np.float32)
                        # Normalize
                        embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(embedding)
                    
                    # Update successful stats
                    self.api_stats["successful_requests"] += 1
                    
                    # Set dimension if not set yet
                    if self.embedding_dimension is None and embeddings:
                        self.embedding_dimension = embeddings[0].shape[0]
                        self.logger.info(f"Set embedding dimension to {self.embedding_dimension}")
                    
                    return embeddings
                    
                finally:
                    watchdog.stop()
                    
            except Exception as e:
                # Update failed stats
                self.api_stats["failed_requests"] += 1
                self.api_stats["retried_requests"] += 1
                
                # Categorize error for retry logic
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
            finally:
                # Record timing
                elapsed_ms = (time.time() - start_time) * 1000
                self.api_stats["total_time_ms"] += elapsed_ms
    
    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts using efficient batching
        """
        self.logger.warning(f"START EMBEDDING: Generating embeddings for {len(texts)} texts, total text length: {sum(len(t) for t in texts)}")
        await self.log_memory_usage("before_embeddings")
        
        # Create batches for optimal processing
        batches = self.batch_processor.create_batches(texts)
        
        # Process batches concurrently with controlled concurrency
        all_embeddings = []
        failed_indices = set()
        
        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} texts")
            
            try:
                # Get embeddings for this batch
                batch_embeddings = await self._get_embedding_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # If batch succeeded but was smaller than expected, track failed indices
                if len(batch_embeddings) < len(batch):
                    for i in range(len(batch_embeddings), len(batch)):
                        batch_offset = sum(len(batches[j]) for j in range(batch_idx))
                        failed_indices.add(batch_offset + i)
                        
            except RetryError as e:
                self.logger.error(f"Batch {batch_idx+1} failed after all retries: {str(e.__cause__)}")
                
                # Mark all indices in this batch as failed
                batch_offset = sum(len(batches[j]) for j in range(batch_idx))
                for i in range(len(batch)):
                    failed_indices.add(batch_offset + i)
            
            # Yield control to other tasks
            await asyncio.sleep(0)
            
        # Handle any failed embeddings
        if failed_indices:
            self.logger.warning(f"Failed to get embeddings for {len(failed_indices)} texts")
            
            # If some embeddings succeeded, use the dimension to create zero vectors
            if all_embeddings and self.embedding_dimension is not None:
                # Fill in failed embeddings with zero vectors
                full_embeddings = []
                failed_idx_set = set(failed_indices)
                
                embed_idx = 0
                for i in range(len(texts)):
                    if i in failed_idx_set:
                        # Use zeros for failed embedding
                        zero_embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                        full_embeddings.append(zero_embedding)
                    else:
                        # Use successful embedding
                        full_embeddings.append(all_embeddings[embed_idx])
                        embed_idx += 1
                
                all_embeddings = full_embeddings
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        
        await self.log_array_info("embeddings_result", embeddings_array)
        await self.log_memory_usage("after_embeddings")
        
        return embeddings_array
    
    async def analyze_text_length(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text lengths to help with optimal batching"""
        if not texts:
            return {"count": 0, "avg_length": 0, "total_length": 0}
            
        lengths = [len(text) for text in texts]
        return {
            "count": len(texts),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": sum(lengths) / len(texts),
            "total_length": sum(lengths),
            "length_distribution": {
                "short (< 1000)": sum(1 for l in lengths if l < 1000),
                "medium (1000-5000)": sum(1 for l in lengths if 1000 <= l < 5000),
                "long (5000+)": sum(1 for l in lengths if l >= 5000)
            }
        }
    
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
            
            # Get text length stats
            length_analysis = await self.analyze_text_length(texts)
            
            # Generate embeddings with batching
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
            
            # Update stat tracking
            api_stats = self.api_stats.copy()
            api_stats["avg_request_time_ms"] = (
                api_stats["total_time_ms"] / api_stats["total_requests"] 
                if api_stats["total_requests"] > 0 else 0
            )
            
            result_data = result.model_dump()
            result_data["api_stats"] = api_stats
            result_data["text_analysis"] = length_analysis
            
            self.logger.info(f"Successfully generated {len(embeddings_list)} embeddings in {processing_time_ms:.2f}ms")
            
            return ToolResult(success=True, data=result_data)
        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}")
            error_message = str(e)
            return ToolResult(success=False, error=error_message)
    
    async def _execute(self, **kwargs) -> ToolResult:
        texts = kwargs.get("texts")
        if not texts:
            return ToolResult(success=False, error="Missing required parameter: texts")
        return await self.run(texts, **kwargs)