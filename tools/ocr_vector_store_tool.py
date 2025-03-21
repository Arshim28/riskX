import os
import gc
import numpy as np
import tracemalloc
from typing import List, Dict, Any, Optional, Union
import asyncio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger
from utils.text_chunk import TextChunk


class OCRVectorStoreConfig(BaseModel):
    index_type: str = "Flat"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_batch: int = 50


class OCRVectorStoreTool(BaseTool):
    name = "ocr_vector_store_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = OCRVectorStoreConfig(**config.get("ocr_vector_store", {}))
        self.logger = get_logger(self.name)
        
        # Import tools here to avoid circular imports
        from tools.ocr_tool import OcrTool
        from tools.embedding_tool import EmbeddingTool
        from tools.document_processor_tool import DocumentProcessorTool
        from tools.vector_store_tool import VectorStoreTool
        
        # Initialize tools
        self.ocr_tool = OcrTool(config.get("ocr", {}))
        self.embedding_tool = EmbeddingTool(config.get("embedding", {}))
        self.document_processor = DocumentProcessorTool({
            "document_processor": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            },
            "ocr": config.get("ocr", {}),
            "embedding": config.get("embedding", {})
        })
        self.vector_store = VectorStoreTool({
            "vector_store": {
                "dimension": None,  # Will be set when first embeddings are created
                "index_type": self.config.index_type
            }
        })
        
        self.logger.info(f"Initialized OCRVectorStoreTool with chunk_size={self.config.chunk_size}, "
                         f"chunk_overlap={self.config.chunk_overlap}, index_type={self.config.index_type}")
    
    async def log_memory_usage(self, marker: str) -> None:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        self.logger.info(f"MEMORY [{marker}]: "
                        f"RSS={memory_info.rss / (1024 * 1024):.1f}MB, "
                        f"VMS={memory_info.vms / (1024 * 1024):.1f}MB")
    
    async def add_document(self, pdf_path: str) -> Dict[str, Any]:
        self.logger.warning(f"ADDING DOCUMENT: {pdf_path}")
        await self.log_memory_usage("before_add_document")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        tracemalloc.start()
        
        # Process the document to get chunks
        doc_result = await self.document_processor.run(
            pdf_path=pdf_path, 
            embed=False,  # We'll handle embedding separately for better control
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        if not doc_result.success:
            tracemalloc.stop()
            raise ValueError(f"Document processing failed: {doc_result.error}")
        
        # Get the chunks from the result
        chunks_data = doc_result.data.get("chunks", [])
        chunks = [TextChunk.from_dict(chunk) for chunk in chunks_data]
        
        if not chunks:
            self.logger.warning(f"No text extracted from {pdf_path}")
            tracemalloc.stop()
            return {"success": False, "error": "No text extracted from document"}
            
        self.logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
        
        current, peak = tracemalloc.get_traced_memory()
        self.logger.warning(f"MEMORY AFTER PDF PROCESSING: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
        
        # Process in batches if needed
        max_chunks_per_batch = self.config.max_chunks_per_batch
        if len(chunks) > max_chunks_per_batch:
            self.logger.warning(f"PROCESSING LARGE DOCUMENT IN BATCHES: {len(chunks)} chunks in batches of {max_chunks_per_batch}")
            
            for i in range(0, len(chunks), max_chunks_per_batch):
                batch_end = min(i + max_chunks_per_batch, len(chunks))
                batch_chunks = chunks[i:batch_end]
                
                self.logger.warning(f"PROCESSING BATCH {i//max_chunks_per_batch + 1}/{(len(chunks) + max_chunks_per_batch - 1)//max_chunks_per_batch}")
                
                # Extract texts from batch
                batch_texts = [chunk.text for chunk in batch_chunks]
                
                self.logger.warning(f"GENERATING EMBEDDINGS FOR BATCH ({len(batch_chunks)} chunks)")
                embedding_result = await self.embedding_tool.run(texts=batch_texts)
                
                if not embedding_result.success:
                    self.logger.error(f"Embedding generation failed: {embedding_result.error}")
                    continue
                
                batch_embeddings = np.array(embedding_result.data.get("embeddings", []), dtype=np.float32)
                
                current, peak = tracemalloc.get_traced_memory()
                self.logger.warning(f"MEMORY AFTER BATCH EMBEDDING: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
                
                self.logger.warning(f"ADDING BATCH TO VECTOR STORE")
                await self.vector_store.run(
                    command="add",
                    chunks=batch_chunks,
                    embeddings=batch_embeddings
                )
                
                current, peak = tracemalloc.get_traced_memory()
                self.logger.warning(f"MEMORY AFTER ADDING BATCH TO VECTOR STORE: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
                
                batch_chunks = None
                batch_embeddings = None
                batch_texts = None
                gc.collect()
                
                current, peak = tracemalloc.get_traced_memory()
                self.logger.warning(f"MEMORY AFTER BATCH GC: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
        else:
            # Process all chunks at once
            texts = [chunk.text for chunk in chunks]
            
            self.logger.warning("GENERATING EMBEDDINGS")
            embedding_result = await self.embedding_tool.run(texts=texts)
            
            if not embedding_result.success:
                tracemalloc.stop()
                raise ValueError(f"Embedding generation failed: {embedding_result.error}")
            
            embeddings = np.array(embedding_result.data.get("embeddings", []), dtype=np.float32)
            
            current, peak = tracemalloc.get_traced_memory()
            self.logger.warning(f"MEMORY AFTER EMBEDDING: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
            
            self.logger.warning("ADDING TO VECTOR STORE")
            add_result = await self.vector_store.run(
                command="add",
                chunks=chunks,
                embeddings=embeddings
            )
            
            if not add_result.success:
                tracemalloc.stop()
                raise ValueError(f"Failed to add to vector store: {add_result.error}")
            
            current, peak = tracemalloc.get_traced_memory()
            self.logger.warning(f"MEMORY AFTER ADDING TO VECTOR STORE: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
            
            embeddings = None
            texts = None
        
        tracemalloc.stop()
        
        chunks = None
        gc.collect()
        await self.log_memory_usage("after_add_document")
        
        # Get vector store info
        info_result = await self.vector_store.run(command="info")
        vector_store_info = info_result.data if info_result.success else {}
        
        return {
            "success": True,
            "document": os.path.basename(pdf_path),
            "chunks_processed": doc_result.data.get("chunk_count", 0),
            "vector_store": vector_store_info
        }
    
    async def answer_question(self, question: str, k: int = 5) -> Dict[str, Any]:
        self.logger.warning(f"ANSWERING QUESTION: {question}")
        await self.log_memory_usage("before_question")
        
        # Get vector store info to check if it's initialized
        info_result = await self.vector_store.run(command="info")
        
        if not info_result.success:
            return {"success": False, "error": "Failed to get vector store info"}
            
        if not info_result.data.get("initialized", False):
            return {"success": False, "error": "Vector store not initialized. Add documents first."}
        
        # Generate embedding for the question
        self.logger.warning("GENERATING QUESTION EMBEDDING")
        embedding_result = await self.embedding_tool.run(texts=[question])
        
        if not embedding_result.success:
            return {"success": False, "error": f"Failed to generate question embedding: {embedding_result.error}"}
        
        question_embedding = embedding_result.data.get("embeddings", [[]])[0]
        
        # Search for similar chunks
        self.logger.warning("SEARCHING VECTOR STORE")
        search_result = await self.vector_store.run(
            command="search",
            query_embedding=question_embedding,
            k=k
        )
        
        if not search_result.success:
            return {"success": False, "error": f"Vector store search failed: {search_result.error}"}
        
        # Format the results
        results = []
        for i in range(len(search_result.data.get("texts", []))):
            results.append({
                "text": search_result.data["texts"][i],
                "metadata": search_result.data["metadata_list"][i],
                "score": search_result.data["scores"][i]
            })
        
        await self.log_memory_usage("after_question")
        
        return {
            "success": True,
            "question": question,
            "results": results,
            "result_count": len(results)
        }
    
    async def save(self, directory: str) -> Dict[str, Any]:
        self.logger.info(f"Saving vector store to {directory}")
        await self.log_memory_usage("before_save")
        
        save_result = await self.vector_store.run(
            command="save",
            directory=directory
        )
        
        if not save_result.success:
            return {"success": False, "error": f"Failed to save vector store: {save_result.error}"}
        
        await self.log_memory_usage("after_save")
        
        return {
            "success": True,
            "saved": True,
            "directory": directory,
            "chunks": save_result.data.get("chunks", 0)
        }
    
    async def load(self, directory: str) -> Dict[str, Any]:
        self.logger.info(f"Loading vector store from {directory}")
        await self.log_memory_usage("before_load")
        
        load_result = await self.vector_store.run(
            command="load",
            directory=directory
        )
        
        if not load_result.success:
            return {"success": False, "error": f"Failed to load vector store: {load_result.error}"}
        
        await self.log_memory_usage("after_load")
        
        return {
            "success": True,
            "loaded": True,
            "directory": directory,
            "chunks": load_result.data.get("chunks", 0)
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def run(self, command: str, **kwargs) -> ToolResult[Dict[str, Any]]:
        try:
            self.logger.info(f"Running OCR vector store command: {command}")
            
            result = {}
            
            if command == "add_document":
                # Add a document to the vector store
                pdf_path = kwargs.get("pdf_path")
                
                if not pdf_path:
                    return ToolResult(success=False, error="PDF path is required for 'add_document' command")
                
                result = await self.add_document(pdf_path)
                
            elif command == "answer_question":
                # Answer a question using the vector store
                question = kwargs.get("question")
                k = kwargs.get("k", 5)
                
                if not question:
                    return ToolResult(success=False, error="Question is required for 'answer_question' command")
                
                result = await self.answer_question(question, k)
                
            elif command == "save":
                # Save the vector store
                directory = kwargs.get("directory")
                
                if not directory:
                    return ToolResult(success=False, error="Directory is required for 'save' command")
                
                result = await self.save(directory)
                
            elif command == "load":
                # Load the vector store
                directory = kwargs.get("directory")
                
                if not directory:
                    return ToolResult(success=False, error="Directory is required for 'load' command")
                
                result = await self.load(directory)
                
            elif command == "info":
                # Get information about the vector store and tools
                info_result = await self.vector_store.run(command="info")
                
                result = {
                    "ocr_tool": self.ocr_tool.name,
                    "embedding_tool": self.embedding_tool.name,
                    "document_processor": self.document_processor.name,
                    "vector_store": info_result.data if info_result.success else {},
                    "config": self.config.model_dump()
                }
                
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
            
            # Calculate memory usage
            await self.log_memory_usage(f"after_{command}")
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"OCR vector store error: {str(e)}")
            return await self._handle_error(e)