import os
import gc
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger
from utils.text_chunk import TextChunk


class DocumentProcessorConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200


class DocumentProcessorTool(BaseTool):
    name = "document_processor_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = DocumentProcessorConfig(**config.get("document_processor", {}))
        self.logger = get_logger(self.name)
        
        # Import tools here to avoid circular imports
        from tools.ocr_tool import OcrTool
        from tools.embedding_tool import EmbeddingTool
        
        self.ocr_tool = OcrTool(config.get("ocr", {}))
        self.embedding_tool = EmbeddingTool(config.get("embedding", {}))
    
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
    
    async def process_pdf(self, pdf_path: str) -> List[TextChunk]:
        self.logger.warning(f"PROCESSING PDF: {pdf_path}")
        await self.log_memory_usage("before_ocr")
        
        ocr_result = await self.ocr_tool.run(pdf_path=pdf_path)
        if not ocr_result.success:
            raise ValueError(f"OCR processing failed: {ocr_result.error}")
        
        await self.log_memory_usage("after_ocr")
        
        text_by_page = ocr_result.data.get("text_by_page", {})
        self.logger.warning(f"OCR COMPLETE: Extracted {len(text_by_page)} pages of text")
        
        gc.collect()
        await self.log_memory_usage("after_ocr_gc")
        
        all_chunks = []
        total_text_length = sum(len(text) for text in text_by_page.values())
        self.logger.warning(f"TOTAL TEXT LENGTH: {total_text_length} characters")
        
        for page_idx, page_text in text_by_page.items():
            self.logger.warning(f"CHUNKING PAGE {page_idx}: Text length {len(page_text)}")
            await self.log_memory_usage(f"before_chunk_page_{page_idx}")
            
            page_chunks = await self._chunk_text(page_text)
            
            self.logger.warning(f"CREATED {len(page_chunks)} CHUNKS FOR PAGE {page_idx}")
            
            for i, chunk in enumerate(page_chunks):
                metadata = {
                    "source": os.path.basename(pdf_path),
                    "page": page_idx,
                    "chunk_index": i,
                }
                
                all_chunks.append(TextChunk(text=chunk, metadata=metadata))
                
            await self.log_memory_usage(f"after_chunk_page_{page_idx}")
            gc.collect()
            await self.log_memory_usage(f"after_gc_page_{page_idx}")
        
        self.logger.warning(f"CHUNKING COMPLETE: Created {len(all_chunks)} chunks")
        
        if len(all_chunks) > 100:
            self.logger.warning(f"LARGE NUMBER OF CHUNKS: {len(all_chunks)} - may cause memory issues")
            
        await self.log_memory_usage("after_chunking")
        gc.collect()
        await self.log_memory_usage("after_chunking_gc")
        return all_chunks
    
    async def _chunk_text(self, text: str) -> List[str]:
        self.logger.info("Starting _chunk_text")
        total_length = len(text)
        self.logger.info(f"Text length to chunk: {total_length}")
        if not text:
            return []
        
        chunks = []
        i = 0
        step = self.config.chunk_size - self.config.chunk_overlap
        if step <= 0:
            self.logger.warning("chunk_overlap must be less than chunk_size. Using chunk_size as step.")
            step = self.config.chunk_size

        iteration = 0
        while i < total_length:
            if iteration % 10 == 0:
                await self.log_memory_usage(f"_chunk_text iteration {iteration}")
            
            tentative_end = min(i + self.config.chunk_size, total_length)
            
            if tentative_end < total_length:
                paragraph_break = text.rfind("\n\n", i, tentative_end)
                if paragraph_break != -1 and paragraph_break > i + self.config.chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    sentence_break = text.rfind(". ", i, tentative_end)
                    if sentence_break != -1 and sentence_break > i + self.config.chunk_size // 2:
                        end = sentence_break + 2
                    else:
                        word_break = text.rfind(" ", i, tentative_end)
                        if word_break != -1 and word_break > i + self.config.chunk_size // 2:
                            end = word_break + 1
                        else:
                            end = tentative_end
            else:
                end = tentative_end
            
            chunk = text[i:end].strip()
            if chunk:
                chunks.append(chunk)
                if len(chunks) % 10 == 0:
                    self.logger.info(f"Created chunk {len(chunks)} with {len(chunk)} characters")
            
            i += step
            iteration += 1

        if chunks:
            chunk_sizes = [len(chunk) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunks)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            self.logger.info(f"_chunk_text completed: generated {len(chunks)} chunks")
            self.logger.info(f"Chunk size statistics: min={min_size}, avg={avg_size:.1f}, max={max_size} characters")
            
        await self.log_memory_usage("_chunk_text completed")
        return chunks
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def embed_chunks(self, chunks: List[TextChunk]) -> np.ndarray:
        self.logger.warning(f"EMBEDDING {len(chunks)} CHUNKS")
        await self.log_memory_usage("before_embed_chunks")
        
        # Get chunk sizes for estimation
        sample_size = min(len(chunks), 10)
        avg_text_len = sum(len(chunk.text) for chunk in chunks[:sample_size]) / sample_size
        est_total_len = int(avg_text_len * len(chunks))
        self.logger.warning(f"ESTIMATED TOTAL TEXT TO EMBED: ~{est_total_len} characters")
        
        # Extract all texts
        texts = [chunk.text for chunk in chunks]
        
        # Use our dedicated embedding tool
        result = await self.embedding_tool.run(texts=texts)
        
        if not result.success:
            raise ValueError(f"Embedding generation failed: {result.error}")
        
        embeddings_list = result.data.get("embeddings", [])
        dimension = result.data.get("dimension")
        
        # Convert to numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # Assign embeddings back to chunks
        for i, embedding in enumerate(embeddings_list):
            if i < len(chunks):
                chunks[i].embedding = embedding
        
        await self.log_array_info("embeddings_result", embeddings)
        
        await self.log_memory_usage("after_embed_chunks")
        gc.collect()
        await self.log_memory_usage("after_embed_chunks_gc")
        return embeddings
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def run(self, pdf_path: str, embed: bool = True, **kwargs) -> ToolResult[Dict[str, Any]]:
        try:
            self.logger.info(f"Processing document: {pdf_path} with embedding={embed}")
            
            if not os.path.exists(pdf_path):
                return ToolResult(success=False, error=f"File not found: {pdf_path}")
            
            # Override config values with any provided in kwargs
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.logger.info(f"Override config: {key}={value}")
            
            chunks = await self.process_pdf(pdf_path)
            self.logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
            
            chunks_data = [chunk.to_dict() for chunk in chunks]
            
            result = {
                "pdf_path": pdf_path,
                "chunks": chunks_data,
                "chunk_count": len(chunks),
                "config": self.config.model_dump()
            }
            
            if embed:
                embeddings = await self.embed_chunks(chunks)
                result["embedding_shape"] = embeddings.shape
                result["embedding_size_mb"] = embeddings.nbytes / (1024 * 1024)
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"Document processing error: {str(e)}")
            return await self._handle_error(e)