import os
import gc
import numpy as np
import json
import atexit
import shutil
import tempfile
import tracemalloc
import psutil
import time
import weakref
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Iterator, BinaryIO
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pydantic import BaseModel, Field, model_validator

from tenacity import retry, stop_after_attempt, wait_exponential
import faiss

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger
from utils.text_chunk import TextChunk


class VectorStoreConfig(BaseModel):
    dimension: Optional[int] = None
    index_type: str = "Flat"
    metric: str = "cosine"
    nlist: int = 100
    nprobe: int = 10
    ef_construction: int = 200
    ef_search: int = 50
    m: int = 16
    
    # Memory management settings
    max_memory_percentage: float = 80.0  # Maximum memory usage as % of system RAM
    batch_size: int = 1000  # Number of vectors to process at once
    enable_mmap: bool = True  # Use memory mapping for large indices
    
    # Performance settings
    parallel_batch_processing: bool = True
    num_workers: int = 4
    
    # Advanced settings
    temp_dir: Optional[str] = None  # Directory for temp files
    enable_auto_gc: bool = True     # Trigger GC after large operations
    
    @model_validator(mode='after')
    def validate_model(self) -> 'VectorStoreConfig':
        if self.max_memory_percentage <= 0 or self.max_memory_percentage > 95:
            raise ValueError("max_memory_percentage must be between 0 and 95")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        return self


class VectorStoreResult(BaseModel):
    texts: List[str]
    metadata_list: List[Dict[str, Any]]
    scores: List[float]
    indices: Optional[List[int]] = None


class MemoryStats(BaseModel):
    timestamp: float
    rss_mb: float
    vms_mb: float
    available_memory_mb: float
    used_memory_percentage: float
    peak_memory_mb: Optional[float] = None


class IndexStats(BaseModel):
    index_type: str
    num_vectors: int
    dimension: int
    index_size_mb: float
    loaded_in_memory: bool
    uses_mmap: bool


class VectorStoreTool(BaseTool):
    name = "vector_store_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = VectorStoreConfig(**config.get("vector_store", {}))
        self.logger = get_logger(self.name)
        self.index = None
        self.chunks = []
        self.dimension = self.config.dimension
        self.initialized = False
        self.faiss = None
        
        # Memory tracking
        self.memory_stats = []
        self.mem_alert_threshold = 0.90 * self.config.max_memory_percentage
        self._memory_critical = False
        
        # Temporary files management
        self.temp_directory = self.config.temp_dir or tempfile.mkdtemp(prefix="vectorstore_")
        self._temp_files = set()
        
        # Batch operations management
        self._is_batched_operation = False
        self._operation_batch_size = self.config.batch_size
        
        # Import FAISS
        self.import_faiss()
        
        # Register cleanup on exit
        self._finalizer = weakref.finalize(self, self._cleanup_resources)
        atexit.register(self._atexit_cleanup)
        
        # Start memory tracking if enabled
        if self.config.enable_auto_gc:
            self._start_memory_tracking()
    
    def __del__(self):
        """Explicit cleanup on deletion"""
        self._cleanup_resources()
    
    def _atexit_cleanup(self):
        """Cleanup resources at exit"""
        self._cleanup_resources()
    
    def _cleanup_resources(self):
        """Clean up all resources used by the tool"""
        try:
            # Clear the index
            if hasattr(self, 'index') and self.index is not None:
                self.index = None
            
            # Release memory for chunks
            if hasattr(self, 'chunks'):
                self.chunks = []
            
            # Clean up temporary directory
            if hasattr(self, 'temp_directory') and os.path.exists(self.temp_directory):
                try:
                    shutil.rmtree(self.temp_directory)
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
        except:
            # Ignore errors during cleanup
            pass
    
    def import_faiss(self):
        try:
            import faiss
            self.faiss = faiss
            self.logger.info("Successfully imported FAISS")
        except ImportError:
            self.logger.error("Failed to import FAISS. Please install it with: pip install faiss-cpu or faiss-gpu")
            raise ImportError("FAISS is required for VectorStoreTool")
    
    def _get_memory_usage(self) -> MemoryStats:
        """Get current memory usage statistics"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        peak_memory = None
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak / (1024 * 1024)
        
        stats = MemoryStats(
            timestamp=time.time(),
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            available_memory_mb=system_memory.available / (1024 * 1024),
            used_memory_percentage=(memory_info.rss / system_memory.total) * 100,
            peak_memory_mb=peak_memory
        )
        
        self.memory_stats.append(stats)
        # Keep only the last 100 memory stats
        if len(self.memory_stats) > 100:
            self.memory_stats = self.memory_stats[-100:]
        
        return stats
    
    def _start_memory_tracking(self):
        """Start memory tracking with tracemalloc"""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def _stop_memory_tracking(self):
        """Stop memory tracking"""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    
    def _check_memory_pressure(self, operation: str = "unknown") -> bool:
        """Check if memory usage is approaching critical levels"""
        stats = self._get_memory_usage()
        
        # Check if memory usage exceeds threshold
        if stats.used_memory_percentage > self.mem_alert_threshold:
            if not self._memory_critical:
                self.logger.warning(
                    f"High memory pressure detected during {operation}: "
                    f"{stats.used_memory_percentage:.1f}% used, "
                    f"{stats.rss_mb:.1f}MB RSS"
                )
                self._memory_critical = True
            
            # Force garbage collection
            gc.collect()
            return True
        
        self._memory_critical = False
        return False
    
    @contextmanager
    def _temp_file(self, prefix="vectorstore_", suffix=".tmp", delete=True) -> Iterator[BinaryIO]:
        """Context manager for temporary files that ensures cleanup"""
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=self.temp_directory)
        self._temp_files.add(path)
        
        try:
            with open(fd, "wb") as f:
                yield f
        finally:
            if delete and path in self._temp_files:
                try:
                    os.unlink(path)
                    self._temp_files.remove(path)
                except:
                    pass
    
    async def log_memory_usage(self, marker: str) -> None:
        """Log current memory usage with a marker for tracking"""
        stats = self._get_memory_usage()
        
        self.logger.info(
            f"MEMORY [{marker}]: "
            f"RSS={stats.rss_mb:.1f}MB, "
            f"VMS={stats.vms_mb:.1f}MB, "
            f"Available={stats.available_memory_mb:.1f}MB, "
            f"Used={stats.used_memory_percentage:.1f}%"
        )
        
        if stats.peak_memory_mb:
            self.logger.info(f"PEAK MEMORY [{marker}]: {stats.peak_memory_mb:.1f}MB")
    
    async def log_array_info(self, name: str, arr: np.ndarray) -> None:
        """Log information about a numpy array"""
        self.logger.info(
            f"ARRAY [{name}]: "
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}, "
            f"size={arr.size}, "
            f"memory={arr.nbytes / (1024 * 1024):.1f}MB"
        )
    
    async def initialize_index(self, dimension: int) -> None:
        """Initialize FAISS index with the specified dimension"""
        self.dimension = dimension
        
        self.logger.info(f"Initializing FAISS index with dimension {dimension}")
        await self.log_memory_usage("before_index_init")
        
        index_factory_str = ""
        if self.config.index_type == "Flat":
            index_factory_str = "Flat"
        elif self.config.index_type == "IVFFlat":
            index_factory_str = f"IVF{self.config.nlist},Flat"
        elif self.config.index_type == "IVFPQ":
            pq_size = min(64, dimension // 4)  # Ensure pq_size is reasonable
            index_factory_str = f"IVF{self.config.nlist},PQ{pq_size}"
        elif self.config.index_type == "HNSW":
            index_factory_str = f"HNSW{self.config.m}"
        else:
            self.logger.warning(f"Unknown index type {self.config.index_type}, falling back to Flat")
            index_factory_str = "Flat"
        
        metric_type = self.faiss.METRIC_INNER_PRODUCT if self.config.metric == "cosine" else self.faiss.METRIC_L2
        
        self.index = self.faiss.index_factory(dimension, index_factory_str, metric_type)
        
        if self.config.index_type == "IVFFlat" or self.config.index_type == "IVFPQ":
            self.index.nprobe = self.config.nprobe
        elif self.config.index_type == "HNSW":
            if hasattr(self.index, "hnsw"):
                self.index.hnsw.efConstruction = self.config.ef_construction
                self.index.hnsw.efSearch = self.config.ef_search
        
        self.initialized = True
        await self.log_memory_usage("after_index_init")
        
        self.logger.info(f"FAISS index initialized: {index_factory_str}, metric_type={metric_type}")
    
    def _get_batch_size(self, data_length: int) -> int:
        """Dynamically determine optimal batch size based on memory conditions"""
        if self._memory_critical:
            # Reduce batch size under memory pressure
            return min(self.config.batch_size // 2, 100)
        
        # Start with configured batch size
        batch_size = self.config.batch_size
        
        # For very small datasets, process all at once
        if data_length <= batch_size:
            return data_length
        
        # For larger datasets, check memory conditions
        mem_stats = self._get_memory_usage()
        if mem_stats.used_memory_percentage > 70:
            # Reduce batch size when memory usage is high
            batch_size = min(batch_size, max(100, data_length // 10))
        
        return batch_size
    
    async def _process_batch(self, batch_chunks: List[TextChunk], batch_embeddings: np.ndarray) -> None:
        """Process a single batch of chunks and embeddings"""
        if not batch_chunks or len(batch_chunks) == 0:
            return
        
        # Normalize if using cosine similarity
        if self.config.metric == "cosine":
            faiss_embeddings = batch_embeddings.copy()
            self.faiss.normalize_L2(faiss_embeddings)
        else:
            faiss_embeddings = batch_embeddings
        
        # Get current chunk count before adding
        start_idx = len(self.chunks)
        
        # Add chunks to our list
        self.chunks.extend(batch_chunks)
        
        # Add embeddings to the index
        self.index.add(faiss_embeddings)
        
        self.logger.debug(f"Added batch of {len(batch_chunks)} chunks to vector store (total: {len(self.chunks)})")
    
    async def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the vector store.
        For large datasets, processes in batches to manage memory usage.
        """
        if not chunks or len(chunks) == 0:
            self.logger.warning("No chunks provided to add_chunks")
            return
        
        if not self.initialized:
            if self.dimension is None:
                self.dimension = embeddings.shape[1]
            await self.initialize_index(self.dimension)
        
        await self.log_memory_usage("before_add_chunks")
        
        if len(chunks) != embeddings.shape[0]:
            error_msg = f"Number of chunks ({len(chunks)}) doesn't match number of embeddings ({embeddings.shape[0]})"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if embeddings.shape[1] != self.dimension:
            error_msg = f"Embedding dimension ({embeddings.shape[1]}) doesn't match index dimension ({self.dimension})"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Process in batches to manage memory
        total_chunks = len(chunks)
        batch_size = self._get_batch_size(total_chunks)
        num_batches = (total_chunks + batch_size - 1) // batch_size
        
        self.logger.info(f"Processing {total_chunks} chunks in {num_batches} batches of size {batch_size}")
        
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            self.logger.debug(f"Processing batch {i//batch_size + 1}/{num_batches}")
            
            batch_chunks = chunks[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            
            await self._process_batch(batch_chunks, batch_embeddings)
            
            # Check memory pressure after each batch
            if self._check_memory_pressure(operation="add_chunks"):
                # Reduce batch size if under memory pressure
                new_batch_size = self._get_batch_size(total_chunks - end_idx)
                if new_batch_size < batch_size:
                    self.logger.info(f"Reducing batch size from {batch_size} to {new_batch_size} due to memory pressure")
                    batch_size = new_batch_size
            
            # Give other tasks a chance to run and allow memory to be freed
            await asyncio.sleep(0)
        
        self.logger.info(f"Added {total_chunks} chunks to vector store (total: {len(self.chunks)})")
        
        # Force garbage collection to free memory after large operations
        if self.config.enable_auto_gc:
            gc.collect()
            
        await self.log_memory_usage("after_add_chunks")
    
    async def search(self, query_embedding: np.ndarray, k: int = 5, filter_func: Optional[Callable[[TextChunk], bool]] = None) -> List[Tuple[TextChunk, float]]:
        """
        Search for similar chunks in the vector store.
        Optionally apply a filter function to results.
        """
        if not self.initialized or self.index is None:
            raise ValueError("Vector store not initialized. Add chunks first.")
        
        if len(self.chunks) == 0:
            self.logger.warning("Vector store is empty. No results to return.")
            return []
        
        await self.log_memory_usage("before_search")
        
        # Reshape if necessary
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Make sure dimensions match
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension ({query_embedding.shape[1]}) doesn't match index dimension ({self.dimension})")
        
        # Normalize if using cosine similarity
        if self.config.metric == "cosine":
            query_copy = query_embedding.copy()
            self.faiss.normalize_L2(query_copy)
        else:
            query_copy = query_embedding
        
        # Ensure k is not larger than the number of chunks
        k_search = min(k * 2 if filter_func else k, len(self.chunks))
        
        # Perform search
        distances, indices = self.index.search(query_copy, k_search)
        
        # Convert distances to similarity scores
        if self.config.metric == "cosine":
            # Cosine similarity is inner product (already normalized)
            scores = distances[0]
        else:
            # Convert L2 distance to similarity score (lower distance = higher similarity)
            max_distance = np.max(distances[0]) if distances[0].size > 0 else 1.0
            scores = 1.0 - (distances[0] / max_distance)
        
        # Combine results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores)):
            if idx < len(self.chunks) and idx >= 0:
                chunk = self.chunks[idx]
                # Apply filter if provided
                if filter_func is None or filter_func(chunk):
                    results.append((chunk, float(score)))
            else:
                self.logger.warning(f"Invalid index {idx} returned from FAISS (max: {len(self.chunks)-1})")
        
        # Sort by score (highest first) and limit to requested k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:k]
        
        await self.log_memory_usage("after_search")
        return results
    
    async def save(self, directory: str) -> bool:
        """
        Save the vector store to disk.
        Implements memory-efficient saving for large indices.
        """
        if not self.initialized or self.index is None:
            self.logger.warning("Cannot save uninitialized vector store")
            return False
        
        await self.log_memory_usage("before_save")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save the index
        index_path = os.path.join(directory, "faiss.index")
        
        # For large indices, save to temporary file first then move
        temp_index_path = os.path.join(self.temp_directory, "temp_faiss.index")
        try:
            self.logger.info(f"Saving FAISS index to {temp_index_path}")
            self.faiss.write_index(self.index, temp_index_path)
            
            # Move to final location
            if os.path.exists(index_path):
                os.remove(index_path)
            shutil.move(temp_index_path, index_path)
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
            if os.path.exists(temp_index_path):
                os.remove(temp_index_path)
            raise
        
        # Save chunks in batches
        chunks_path = os.path.join(directory, "chunks.json")
        temp_chunks_path = os.path.join(self.temp_directory, "temp_chunks.json")
        
        try:
            self.logger.info(f"Saving {len(self.chunks)} chunks to {temp_chunks_path}")
            
            # Save in batches to avoid memory issues with large chunk sets
            with open(temp_chunks_path, "w", encoding="utf-8") as f:
                # Write the opening bracket
                f.write("[\n")
                
                # Process chunks in batches
                batch_size = self._get_batch_size(len(self.chunks))
                for i in range(0, len(self.chunks), batch_size):
                    end_idx = min(i + batch_size, len(self.chunks))
                    batch = self.chunks[i:end_idx]
                    
                    # Convert batch to JSON
                    batch_data = [chunk.to_dict() for chunk in batch]
                    batch_json = json.dumps(batch_data, ensure_ascii=False)[1:-1]  # Remove outer brackets
                    
                    # Write batch
                    if i > 0:
                        f.write(",\n")
                    f.write(batch_json)
                    
                    # Check memory pressure
                    self._check_memory_pressure(operation="save_chunks")
                
                # Write closing bracket
                f.write("\n]")
            
            # Move to final location
            if os.path.exists(chunks_path):
                os.remove(chunks_path)
            shutil.move(temp_chunks_path, chunks_path)
        except Exception as e:
            self.logger.error(f"Error saving chunks: {str(e)}")
            if os.path.exists(temp_chunks_path):
                os.remove(temp_chunks_path)
            raise
        
        # Save config
        config_path = os.path.join(directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.model_dump(), f)
        
        self.logger.info(f"Saved vector store to {directory} ({len(self.chunks)} chunks)")
        
        # Force garbage collection
        if self.config.enable_auto_gc:
            gc.collect()
            
        await self.log_memory_usage("after_save")
        return True
    
    async def load(self, directory: str) -> bool:
        """
        Load the vector store from disk.
        Implements memory-efficient loading for large indices.
        """
        await self.log_memory_usage("before_load")
        
        if not os.path.exists(directory):
            self.logger.error(f"Directory {directory} does not exist")
            return False
        
        # Clear current data
        await self.clear(keep_config=True)
        
        # Load config
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                # Update config but keep current settings for temp_dir, etc.
                temp_dir = self.config.temp_dir
                enable_auto_gc = self.config.enable_auto_gc
                self.config = VectorStoreConfig(**config)
                self.config.temp_dir = temp_dir
                self.config.enable_auto_gc = enable_auto_gc
        
        # Load chunks in batches
        chunks_path = os.path.join(directory, "chunks.json")
        if not os.path.exists(chunks_path):
            self.logger.error(f"Chunks file {chunks_path} does not exist")
            return False
        
        # Use streaming JSON parser for large files
        self.logger.info(f"Loading chunks from {chunks_path}")
        self.chunks = []
        try:
            chunk_file_size = os.path.getsize(chunks_path)
            
            # For large files, use batched loading to reduce memory usage
            if chunk_file_size > 100 * 1024 * 1024:  # 100 MB
                self.logger.info(f"Large chunks file detected ({chunk_file_size/1024/1024:.1f} MB), using batched loading")
                import ijson  # For streaming JSON parsing
                
                with open(chunks_path, 'rb') as f:
                    chunks_array = ijson.items(f, 'item')
                    batch = []
                    
                    for i, chunk_data in enumerate(chunks_array):
                        batch.append(TextChunk.from_dict(chunk_data))
                        
                        if len(batch) >= self.config.batch_size:
                            self.chunks.extend(batch)
                            self.logger.info(f"Loaded batch of {len(batch)} chunks (total: {len(self.chunks)})")
                            batch = []
                            
                            # Check memory pressure
                            self._check_memory_pressure(operation="load_chunks")
                            await asyncio.sleep(0)
                    
                    # Add remaining chunks
                    if batch:
                        self.chunks.extend(batch)
                        self.logger.info(f"Loaded final batch of {len(batch)} chunks (total: {len(self.chunks)})")
            else:
                # For smaller files, load all at once
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                    self.chunks = [TextChunk.from_dict(chunk) for chunk in chunks_data]
        except Exception as e:
            self.logger.error(f"Error loading chunks: {str(e)}")
            # Partial cleanup on failure
            self.chunks = []
            return False
        
        # Load index
        index_path = os.path.join(directory, "faiss.index")
        if not os.path.exists(index_path):
            self.logger.error(f"Index file {index_path} does not exist")
            return False
        
        try:
            # Use memory-mapped loading for large indices if enabled
            if self.config.enable_mmap and os.path.getsize(index_path) > 500 * 1024 * 1024:  # 500 MB
                self.logger.info(f"Using memory-mapped loading for large index ({os.path.getsize(index_path)/1024/1024:.1f} MB)")
                
                # Create a temporary IO object for memory-mapped loading
                io_obj = self.faiss.IOReader(index_path)
                self.index = self.faiss.read_index_io(io_obj, self.faiss.IO_FLAG_MMAP)
            else:
                self.index = self.faiss.read_index(index_path)
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            self.chunks = []
            return False
        
        self.dimension = self.index.d
        self.initialized = True
        
        # Set parameters
        if self.config.index_type == "IVFFlat" or self.config.index_type == "IVFPQ":
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = self.config.nprobe
        elif self.config.index_type == "HNSW":
            if hasattr(self.index, "hnsw"):
                self.index.hnsw.efSearch = self.config.ef_search
        
        self.logger.info(f"Loaded vector store from {directory} ({len(self.chunks)} chunks)")
        
        # Force garbage collection
        if self.config.enable_auto_gc:
            gc.collect()
            
        await self.log_memory_usage("after_load")
        return True
    
    async def clear(self, keep_config: bool = False) -> None:
        """
        Clear the vector store, optionally preserving configuration.
        Performs thorough resource cleanup.
        """
        await self.log_memory_usage("before_clear")
        
        saved_config = self.config if keep_config else None
        
        # Clear index
        if self.index is not None:
            # Check if there's a reset method available
            if hasattr(self.index, 'reset'):
                self.index.reset()
            self.index = None
        
        # Clear chunks with memory release
        chunk_count = len(self.chunks)
        for i in range(len(self.chunks)):
            self.chunks[i] = None
        self.chunks = []
        
        # Reset state
        self.initialized = False
        
        # Restore config if needed
        if saved_config:
            self.config = saved_config
        
        # Run garbage collection
        gc.collect()
        
        self.logger.info(f"Vector store cleared ({chunk_count} chunks removed)")
        await self.log_memory_usage("after_clear")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the vector store"""
        memory_stats = self._get_memory_usage()
        
        stats = {
            "memory": {
                "rss_mb": memory_stats.rss_mb,
                "vms_mb": memory_stats.vms_mb,
                "available_memory_mb": memory_stats.available_memory_mb,
                "used_memory_percentage": memory_stats.used_memory_percentage
            },
            "initialized": self.initialized,
            "dimension": self.dimension,
            "chunks": len(self.chunks),
            "config": self.config.model_dump()
        }
        
        # Add index-specific information if initialized
        if self.initialized and self.index is not None:
            # Get index type
            index_type = "unknown"
            if hasattr(self.index, '__class__'):
                index_type = self.index.__class__.__name__
            elif isinstance(self.index, self.faiss.IndexFlat):
                index_type = "Flat"
            elif isinstance(self.index, self.faiss.IndexIVFFlat):
                index_type = "IVFFlat"
            elif "HNSW" in str(type(self.index)):
                index_type = "HNSW"
            
            # Estimate index size
            index_size_mb = 0
            index_path = os.path.join(self.temp_directory, "temp_size_check.index")
            try:
                self.faiss.write_index(self.index, index_path)
                index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
                os.remove(index_path)
            except:
                pass
            
            stats["index"] = {
                "type": index_type,
                "num_vectors": self.index.ntotal,
                "size_mb": index_size_mb,
                "metric": self.config.metric
            }
        
        return stats
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def run(self, command: str, **kwargs) -> ToolResult[Dict[str, Any]]:
        """
        Execute a vector store command.
        Implements proper error handling and resource management.
        """
        try:
            self.logger.info(f"Running vector store command: {command}")
            
            result = {}
            
            if command == "add":
                # Add chunks and embeddings to the vector store
                chunks = kwargs.get("chunks", [])
                embeddings = kwargs.get("embeddings")
                
                if not chunks or embeddings is None:
                    return ToolResult(success=False, error="Chunks and embeddings are required for 'add' command")
                
                # Convert chunks to TextChunk objects if they're not already
                if chunks and not isinstance(chunks[0], TextChunk):
                    chunks = [TextChunk.from_dict(c) if isinstance(c, dict) else TextChunk(text=c, metadata={}) for c in chunks]
                
                # Convert embeddings to numpy array if it's not already
                if not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings, dtype=np.float32)
                
                await self.add_chunks(chunks, embeddings)
                result = {"added_chunks": len(chunks), "total_chunks": len(self.chunks)}
                
            elif command == "search":
                # Search for similar chunks
                query_embedding = kwargs.get("query_embedding")
                k = kwargs.get("k", 5)
                filter_criteria = kwargs.get("filter", None)
                
                if query_embedding is None:
                    return ToolResult(success=False, error="Query embedding is required for 'search' command")
                
                # Convert to numpy array if needed
                if not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding, dtype=np.float32)
                
                # Create filter function if criteria provided
                filter_func = None
                if filter_criteria:
                    def filter_func(chunk: TextChunk) -> bool:
                        for key, value in filter_criteria.items():
                            if key not in chunk.metadata or chunk.metadata[key] != value:
                                return False
                        return True
                
                search_results = await self.search(query_embedding, k, filter_func)
                
                # Format results for return
                result = VectorStoreResult(
                    texts=[r[0].text for r in search_results],
                    metadata_list=[r[0].metadata for r in search_results],
                    scores=[r[1] for r in search_results],
                    indices=[self.chunks.index(r[0]) for r in search_results]
                )
                
                result = result.model_dump()
                
            elif command == "save":
                # Save the vector store
                directory = kwargs.get("directory")
                
                if not directory:
                    return ToolResult(success=False, error="Directory is required for 'save' command")
                
                success = await self.save(directory)
                result = {"saved": success, "directory": directory, "chunks": len(self.chunks)}
                
            elif command == "load":
                # Load the vector store
                directory = kwargs.get("directory")
                
                if not directory:
                    return ToolResult(success=False, error="Directory is required for 'load' command")
                
                success = await self.load(directory)
                result = {"loaded": success, "directory": directory, "chunks": len(self.chunks)}
                
            elif command == "clear":
                # Clear the vector store
                keep_config = kwargs.get("keep_config", False)
                await self.clear(keep_config)
                result = {"cleared": True}
                
            elif command == "info":
                # Return information about the vector store
                detailed = kwargs.get("detailed", False)
                
                if detailed:
                    result = await self.get_stats()
                else:
                    result = {
                        "initialized": self.initialized,
                        "dimension": self.dimension,
                        "chunks": len(self.chunks),
                        "index_type": self.config.index_type,
                        "metric": self.config.metric,
                        "config": self.config.model_dump()
                    }
                
            elif command == "batch_size":
                # Dynamically adjust batch size
                new_size = kwargs.get("size")
                
                if new_size is not None:
                    if new_size <= 0:
                        return ToolResult(success=False, error="Batch size must be positive")
                    self.config.batch_size = new_size
                    result = {"old_batch_size": self._operation_batch_size, "new_batch_size": new_size}
                    self._operation_batch_size = new_size
                else:
                    result = {"batch_size": self.config.batch_size}
                    
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
            
            # Run garbage collection if enabled
            if self.config.enable_auto_gc:
                self._check_memory_pressure(operation=f"after_{command}")
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"Vector store error: {str(e)}")
            
            # Attempt to recover resources on error
            try:
                if self.config.enable_auto_gc:
                    gc.collect()
            except:
                pass
                
            error_message = str(e)
            return ToolResult(success=False, error=error_message)
    
    async def _execute(self, **kwargs) -> ToolResult:
        command = kwargs.get("command")
        if not command:
            return ToolResult(success=False, error="Missing required parameter: command")
        return await self.run(command, **kwargs)