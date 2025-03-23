import os
import gc
import numpy as np
import tracemalloc
import time
import contextlib
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger
from utils.text_chunk import TextChunk


class OCRVectorStoreConfig(BaseModel):
    index_type: str = "Flat"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_batch: int = 50
    memory_threshold_mb: int = 4000  # Memory threshold in MB to trigger cleanup
    incremental_page_size: int = 5   # Number of pages to process at once
    max_retries: int = 5             # Maximum number of retries for operations


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
        
        # Document processing state
        self.current_document = None
        self.processed_pages = set()
        self.failed_pages = {}  # {page_num: retry_count}
        
        self.logger.info(f"Initialized OCRVectorStoreTool with chunk_size={self.config.chunk_size}, "
                         f"chunk_overlap={self.config.chunk_overlap}, index_type={self.config.index_type}")
    
    async def log_memory_usage(self, marker: str) -> None:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        memory_mb = memory_info.rss / (1024 * 1024)
        self.logger.info(f"MEMORY [{marker}]: "
                        f"RSS={memory_mb:.1f}MB, "
                        f"VMS={memory_info.vms / (1024 * 1024):.1f}MB")
        
        # Trigger cleanup if memory usage exceeds threshold
        if memory_mb > self.config.memory_threshold_mb:
            self.logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds threshold ({self.config.memory_threshold_mb}MB). Triggering cleanup...")
            await self._cleanup_memory()
    
    @contextlib.asynccontextmanager
    async def _memory_tracked_operation(self, operation_name: str):
        """Context manager to track memory usage before and after an operation"""
        await self.log_memory_usage(f"before_{operation_name}")
        tracemalloc.start()
        try:
            yield
        finally:
            current, peak = tracemalloc.get_traced_memory()
            self.logger.info(f"MEMORY TRACE [{operation_name}]: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
            tracemalloc.stop()
            await self.log_memory_usage(f"after_{operation_name}")
    
    async def _cleanup_memory(self):
        """Force memory cleanup"""
        self.logger.info("Running memory cleanup...")
        
        # Clear any cached data
        if hasattr(self.document_processor, 'clear_cache'):
            self.document_processor.clear_cache()
        
        # Run garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Additional platform-specific memory cleanup
        try:
            import platform
            if platform.system() == 'Linux':
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ImportError:
            pass
        
        await self.log_memory_usage("after_cleanup")
    
    async def add_document(self, pdf_path: str) -> Dict[str, Any]:
        """Add a complete document to the vector store"""
        self.logger.warning(f"ADDING DOCUMENT: {pdf_path}")
        
        async with self._memory_tracked_operation("add_document"):
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Reset document state
            self.current_document = pdf_path
            self.processed_pages = set()
            self.failed_pages = {}
            
            # Get document information first
            doc_info = await self._get_document_info(pdf_path)
            total_pages = doc_info.get("total_pages", 0)
            
            if total_pages == 0:
                self.logger.warning(f"Document {pdf_path} has no pages")
                return {"success": False, "error": "Document has no pages"}
            
            # Process document incrementally
            all_chunks = []
            page_groups = self._get_page_groups(total_pages)
            
            for group_idx, page_range in enumerate(page_groups):
                self.logger.warning(f"Processing page group {group_idx+1}/{len(page_groups)}: pages {page_range}")
                
                try:
                    group_chunks = await self._process_page_range(pdf_path, page_range)
                    all_chunks.extend(group_chunks)
                    
                    # Update processed pages
                    self.processed_pages.update(page_range)
                    
                    # Forced cleanup after each group
                    await self._cleanup_memory()
                    
                except Exception as e:
                    self.logger.error(f"Error processing page group {page_range}: {str(e)}")
                    for page in page_range:
                        self.failed_pages[page] = self.failed_pages.get(page, 0) + 1
            
            # Process any remaining failed pages with retries
            await self._retry_failed_pages(pdf_path)
            
            # Get vector store info
            info_result = await self.vector_store.run(command="info")
            vector_store_info = info_result.data if info_result.success else {}
            
            self.current_document = None
            return {
                "success": True,
                "document": os.path.basename(pdf_path),
                "total_pages": total_pages,
                "processed_pages": len(self.processed_pages),
                "failed_pages": len(self.failed_pages),
                "chunks_processed": len(all_chunks),
                "vector_store": vector_store_info
            }
    
    async def _get_document_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get information about the document"""
        try:
            # Use OCR tool to get document info
            result = await self.ocr_tool.run(pdf_path=pdf_path, pages=[0], image_limit=0)
            if result.success:
                return {
                    "total_pages": len(result.data.get("text_by_page", {})),
                    "metadata": result.data.get("metadata", {})
                }
            return {"total_pages": 0}
        except Exception as e:
            self.logger.error(f"Error getting document info: {str(e)}")
            return {"total_pages": 0, "error": str(e)}
    
    def _get_page_groups(self, total_pages: int) -> List[List[int]]:
        """Divide document into page groups for incremental processing"""
        page_size = self.config.incremental_page_size
        groups = []
        
        for i in range(0, total_pages, page_size):
            end = min(i + page_size, total_pages)
            groups.append(list(range(i, end)))
            
        return groups
    
    async def _process_page_range(self, pdf_path: str, page_range: List[int]) -> List[TextChunk]:
        """Process a specific range of pages from the document"""
        self.logger.info(f"Processing pages {page_range} from {pdf_path}")
        
        async with self._memory_tracked_operation(f"process_pages_{page_range[0]}_to_{page_range[-1]}"):
            # Extract text using OCR tool
            ocr_result = await self.ocr_tool.run(
                pdf_path=pdf_path, 
                pages=page_range,
                image_limit=5  # Limit images to reduce memory usage
            )
            
            if not ocr_result.success:
                raise ValueError(f"OCR processing failed: {ocr_result.error}")
            
            # Create chunks from the extracted text
            chunks = []
            for page_num, page_text in ocr_result.data.get("text_by_page", {}).items():
                # Skip if page number is not in our requested range
                if int(page_num) not in page_range:
                    continue
                    
                page_chunks = await self._chunk_text(page_text, {
                    "source": os.path.basename(pdf_path),
                    "page": page_num
                })
                chunks.extend(page_chunks)
            
            if not chunks:
                self.logger.warning(f"No text extracted from pages {page_range}")
                return []
            
            # Process in smaller batches to avoid memory issues
            processed_chunks = []
            batch_size = self.config.max_chunks_per_batch
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_chunks = await self._process_chunk_batch(batch)
                processed_chunks.extend(batch_chunks)
                
                # Force cleanup after each batch
                gc.collect()
            
            return processed_chunks
    
    async def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Create text chunks from a page of text"""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        # Skip if text is too short
        if len(text) < chunk_size // 2:
            chunks.append(TextChunk(text=text, metadata=metadata.copy()))
            return chunks
        
        # Create overlapping chunks
        for i in range(0, len(text), chunk_size - overlap):
            end = min(i + chunk_size, len(text))
            chunk_text = text[i:end]
            
            # Skip empty chunks
            if not chunk_text.strip():
                continue
                
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks)
            chunk_metadata["char_start"] = i
            chunk_metadata["char_end"] = end
            
            chunks.append(TextChunk(text=chunk_text, metadata=chunk_metadata))
        
        return chunks
    
    async def _process_chunk_batch(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Process a batch of chunks - generate embeddings and add to vector store"""
        if not chunks:
            return []
            
        try:
            # Generate embeddings for chunks
            texts = [chunk.text for chunk in chunks]
            embedding_result = await self.embedding_tool.run(texts=texts)
            
            if not embedding_result.success:
                raise ValueError(f"Embedding generation failed: {embedding_result.error}")
            
            embeddings = np.array(embedding_result.data.get("embeddings", []), dtype=np.float32)
            
            # Add to vector store
            await self.vector_store.run(
                command="add",
                chunks=chunks,
                embeddings=embeddings
            )
            
            return chunks
        except Exception as e:
            self.logger.error(f"Error processing chunk batch: {str(e)}")
            # Return empty list on error, but don't fail the entire operation
            return []
    
    async def _retry_failed_pages(self, pdf_path: str):
        """Retry processing failed pages with lower batch sizes"""
        if not self.failed_pages:
            return
            
        self.logger.warning(f"Retrying {len(self.failed_pages)} failed pages with adjusted parameters")
        
        # Sort pages by retry count (ascending)
        pages_to_retry = sorted(self.failed_pages.keys(), key=lambda p: self.failed_pages[p])
        
        for page in pages_to_retry:
            # Skip if max retries exceeded
            if self.failed_pages[page] >= self.config.max_retries:
                self.logger.error(f"Page {page} failed after {self.failed_pages[page]} attempts, skipping")
                continue
                
            try:
                # Process single page with reduced parameters
                self.logger.info(f"Retrying page {page} (attempt {self.failed_pages[page]+1})")
                
                await self._process_page_range(pdf_path, [page])
                
                # Update processed pages and remove from failed
                self.processed_pages.add(page)
                del self.failed_pages[page]
                
                # Forced cleanup after each page
                await self._cleanup_memory()
                
            except Exception as e:
                self.logger.error(f"Retry failed for page {page}: {str(e)}")
                self.failed_pages[page] = self.failed_pages.get(page, 0) + 1
    
    async def answer_question(self, question: str, k: int = 5, filter_docs: List[str] = None) -> Dict[str, Any]:
        """Answer a question using the vector store"""
        self.logger.warning(f"ANSWERING QUESTION: {question}")
        await self.log_memory_usage("before_question")
        
        try:
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
            
            # Prepare search parameters
            search_params = {"query_embedding": question_embedding, "k": k}
            if filter_docs:
                search_params["filter_metadata"] = {"source": {"$in": filter_docs}}
            
            # Search for similar chunks
            self.logger.warning("SEARCHING VECTOR STORE")
            search_result = await self.vector_store.run(command="search", **search_params)
            
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
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return {"success": False, "error": f"Error answering question: {str(e)}"}
    
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
                filter_docs = kwargs.get("filter_docs")
                
                if not question:
                    return ToolResult(success=False, error="Question is required for 'answer_question' command")
                
                result = await self.answer_question(question, k, filter_docs)
                
            elif command == "save":
                # Save the vector store
                directory = kwargs.get("directory")
                
                if not directory:
                    return ToolResult(success=False, error="Directory is required for 'save' command")
                
                save_result = await self.vector_store.run(
                    command="save",
                    directory=directory
                )
                
                if not save_result.success:
                    return ToolResult(success=False, error=f"Failed to save vector store: {save_result.error}")
                
                # Save document processing state
                state_path = os.path.join(directory, "processing_state.json")
                with open(state_path, 'w') as f:
                    import json
                    json.dump({
                        "processed_pages": list(self.processed_pages),
                        "failed_pages": self.failed_pages,
                        "current_document": self.current_document
                    }, f)
                
                result = {
                    "success": True,
                    "saved": True,
                    "directory": directory,
                    "chunks": save_result.data.get("chunks", 0)
                }
                
            elif command == "load":
                # Load the vector store
                directory = kwargs.get("directory")
                
                if not directory:
                    return ToolResult(success=False, error="Directory is required for 'load' command")
                
                load_result = await self.vector_store.run(
                    command="load",
                    directory=directory
                )
                
                if not load_result.success:
                    return ToolResult(success=False, error=f"Failed to load vector store: {load_result.error}")
                
                # Load document processing state if available
                state_path = os.path.join(directory, "processing_state.json")
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        import json
                        state = json.load(f)
                        self.processed_pages = set(state.get("processed_pages", []))
                        self.failed_pages = state.get("failed_pages", {})
                        self.current_document = state.get("current_document")
                
                result = {
                    "success": True,
                    "loaded": True,
                    "directory": directory,
                    "chunks": load_result.data.get("chunks", 0)
                }
                
            elif command == "info":
                # Get information about the vector store and tools
                info_result = await self.vector_store.run(command="info")
                
                result = {
                    "ocr_tool": self.ocr_tool.name,
                    "embedding_tool": self.embedding_tool.name,
                    "document_processor": self.document_processor.name,
                    "vector_store": info_result.data if info_result.success else {},
                    "config": self.config.model_dump(),
                    "processing_state": {
                        "current_document": self.current_document,
                        "processed_pages": len(self.processed_pages),
                        "failed_pages": len(self.failed_pages)
                    }
                }
                
            elif command == "resume_processing":
                # Resume processing a previously started document
                pdf_path = kwargs.get("pdf_path") or self.current_document
                
                if not pdf_path:
                    return ToolResult(success=False, error="No document to resume processing")
                
                if not os.path.exists(pdf_path):
                    return ToolResult(success=False, error=f"Document not found: {pdf_path}")
                
                # Get document info
                doc_info = await self._get_document_info(pdf_path)
                total_pages = doc_info.get("total_pages", 0)
                
                # Identify pages that still need processing
                pages_to_process = [p for p in range(total_pages) if p not in self.processed_pages]
                
                self.logger.info(f"Resuming processing of {pdf_path}: {len(pages_to_process)} pages remaining")
                
                # Process remaining pages in groups
                for i in range(0, len(pages_to_process), self.config.incremental_page_size):
                    page_group = pages_to_process[i:i+self.config.incremental_page_size]
                    await self._process_page_range(pdf_path, page_group)
                    self.processed_pages.update(page_group)
                    await self._cleanup_memory()
                
                result = {
                    "success": True,
                    "document": os.path.basename(pdf_path),
                    "pages_processed": len(pages_to_process),
                    "total_processed": len(self.processed_pages),
                    "failed_pages": len(self.failed_pages)
                }
                
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
            
            # Calculate memory usage
            await self.log_memory_usage(f"after_{command}")
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"OCR vector store error: {str(e)}")
            return await self._handle_error(e)