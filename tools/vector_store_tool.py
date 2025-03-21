import os
import gc
import numpy as np
import json
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import asyncio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

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


class VectorStoreResult(BaseModel):
    texts: List[str]
    metadata_list: List[Dict[str, Any]]
    scores: List[float]
    indices: Optional[List[int]] = None


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
        self.import_faiss()
        
    def import_faiss(self):
        try:
            import faiss
            self.faiss = faiss
            self.logger.info("Successfully imported FAISS")
        except ImportError:
            self.logger.error("Failed to import FAISS. Please install it with: pip install faiss-cpu or faiss-gpu")
            raise ImportError("FAISS is required for VectorStoreTool")
    
    async def log_memory_usage(self, marker: str) -> None:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        self.logger.info(f"MEMORY [{marker}]: "
                        f"RSS={memory_info.rss / (1024 * 1024):.1f}MB, "
                        f"VMS={memory_info.vms / (1024 * 1024):.1f}MB")
    
    async def initialize_index(self, dimension: int) -> None:
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
    
    async def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
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
        
        # Normalize if using cosine similarity
        if self.config.metric == "cosine":
            faiss_embeddings = embeddings.copy()
            faiss.normalize_L2(faiss_embeddings)
        else:
            faiss_embeddings = embeddings
        
        # Get current chunk count before adding
        start_idx = len(self.chunks)
        
        # Add chunks to our list
        self.chunks.extend(chunks)
        
        # Add embeddings to the index
        self.index.add(faiss_embeddings)
        
        self.logger.info(f"Added {len(chunks)} chunks to vector store (total: {len(self.chunks)})")
        await self.log_memory_usage("after_add_chunks")
    
    async def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[TextChunk, float]]:
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
            faiss.normalize_L2(query_copy)
        else:
            query_copy = query_embedding
        
        # Ensure k is not larger than the number of chunks
        k = min(k, len(self.chunks))
        
        # Perform search
        distances, indices = self.index.search(query_copy, k)
        
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
                results.append((self.chunks[idx], float(score)))
            else:
                self.logger.warning(f"Invalid index {idx} returned from FAISS (max: {len(self.chunks)-1})")
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        await self.log_memory_usage("after_search")
        return results
    
    async def save(self, directory: str) -> bool:
        if not self.initialized or self.index is None:
            self.logger.warning("Cannot save uninitialized vector store")
            return False
        
        await self.log_memory_usage("before_save")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save the index
        index_path = os.path.join(directory, "faiss.index")
        self.faiss.write_index(self.index, index_path)
        
        # Save chunks
        chunks_path = os.path.join(directory, "chunks.json")
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # Save config
        config_path = os.path.join(directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.model_dump(), f)
        
        self.logger.info(f"Saved vector store to {directory} ({len(self.chunks)} chunks)")
        await self.log_memory_usage("after_save")
        return True
    
    async def load(self, directory: str) -> bool:
        await self.log_memory_usage("before_load")
        
        if not os.path.exists(directory):
            self.logger.error(f"Directory {directory} does not exist")
            return False
        
        # Load config
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.config = VectorStoreConfig(**config)
        
        # Load chunks
        chunks_path = os.path.join(directory, "chunks.json")
        if not os.path.exists(chunks_path):
            self.logger.error(f"Chunks file {chunks_path} does not exist")
            return False
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            self.chunks = [TextChunk.from_dict(chunk) for chunk in chunks_data]
        
        # Load index
        index_path = os.path.join(directory, "faiss.index")
        if not os.path.exists(index_path):
            self.logger.error(f"Index file {index_path} does not exist")
            return False
        
        self.index = self.faiss.read_index(index_path)
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
        await self.log_memory_usage("after_load")
        return True
    
    async def clear(self) -> None:
        await self.log_memory_usage("before_clear")
        
        self.chunks = []
        self.index = None
        self.initialized = False
        
        # Run garbage collection
        gc.collect()
        
        self.logger.info("Vector store cleared")
        await self.log_memory_usage("after_clear")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def run(self, command: str, **kwargs) -> ToolResult[Dict[str, Any]]:
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
                
                if query_embedding is None:
                    return ToolResult(success=False, error="Query embedding is required for 'search' command")
                
                # Convert to numpy array if needed
                if not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding, dtype=np.float32)
                
                search_results = await self.search(query_embedding, k)
                
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
                await self.clear()
                result = {"cleared": True}
                
            elif command == "info":
                # Return information about the vector store
                result = {
                    "initialized": self.initialized,
                    "dimension": self.dimension,
                    "chunks": len(self.chunks),
                    "index_type": self.config.index_type,
                    "metric": self.config.metric,
                    "config": self.config.model_dump()
                }
                
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"Vector store error: {str(e)}")
            return await self._handle_error(e)