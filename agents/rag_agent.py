from typing import Dict, List, Any, Optional, Union
import json
import asyncio
import os
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.ocr_vector_store_tool import OCRVectorStoreTool


class RAGAgent(BaseAgent):
    name = "rag_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager(self.name)
        
        # Initialize OCRVectorStoreTool
        self.vector_store_tool = OCRVectorStoreTool(config)
        
        # Load configuration settings
        self.retrieval_k = config.get("rag_agent", {}).get("retrieval_k", 5)
        self.reranking_enabled = config.get("rag_agent", {}).get("reranking_enabled", False)
        self.max_input_tokens = config.get("rag_agent", {}).get("max_input_tokens", 4000)
        
        # Initialize state tracking variables
        self.initialized = False
        self.loaded_documents = []
    
    async def initialize(self, vector_store_dir: Optional[str] = None) -> bool:
        """Initialize the RAG agent by loading or creating a vector store."""
        if self.initialized and vector_store_dir is None:
            self.logger.info("RAG agent already initialized")
            return True
            
        try:
            if vector_store_dir and os.path.exists(vector_store_dir):
                self.logger.info(f"Loading vector store from {vector_store_dir}")
                result = await self.vector_store_tool.run(
                    command="load",
                    directory=vector_store_dir
                )
                if result.success:
                    self.initialized = True
                    self.logger.info(f"Successfully loaded vector store with {result.data.get('chunks', 0)} chunks")
                    return True
                else:
                    self.logger.error(f"Failed to load vector store: {result.error}")
                    return False
            else:
                self.logger.info("Initializing empty RAG agent")
                self.initialized = True
                return True
                
        except Exception as e:
            self.logger.error(f"Error initializing RAG agent: {str(e)}")
            return False
    
    async def add_document(self, pdf_path: str) -> bool:
        """Add a document to the vector store."""
        if not os.path.exists(pdf_path):
            self.logger.error(f"Document not found: {pdf_path}")
            return False
            
        try:
            self.logger.info(f"Adding document to vector store: {pdf_path}")
            result = await self.vector_store_tool.run(
                command="add_document",
                pdf_path=pdf_path
            )
            
            if result.success:
                self.loaded_documents.append(os.path.basename(pdf_path))
                self.logger.info(f"Successfully added document: {os.path.basename(pdf_path)}")
                return True
            else:
                self.logger.error(f"Failed to add document: {result.error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            return False
    
    async def save_vector_store(self, directory: str) -> bool:
        """Save the vector store to disk."""
        if not self.initialized:
            self.logger.warning("Cannot save vector store: RAG agent not initialized")
            return False
            
        try:
            self.logger.info(f"Saving vector store to {directory}")
            result = await self.vector_store_tool.run(
                command="save",
                directory=directory
            )
            
            if result.success:
                self.logger.info(f"Successfully saved vector store to {directory}")
                return True
            else:
                self.logger.error(f"Failed to save vector store: {result.error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def answer_query(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Process a user query and retrieve relevant information."""
        if not self.initialized:
            self.logger.warning("RAG agent not initialized")
            return {
                "success": False,
                "error": "RAG agent not initialized. Please add documents first."
            }
            
        k_value = k if k is not None else self.retrieval_k
        
        try:
            self.logger.info(f"Processing query: {query}")
            result = await self.vector_store_tool.run(
                command="answer_question",
                question=query,
                k=k_value
            )
            
            if result.success:
                return result.data
            else:
                self.logger.error(f"Query processing failed: {result.error}")
                return {
                    "success": False,
                    "error": f"Query processing failed: {result.error}"
                }
                
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing query: {str(e)}"
            }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_response(self, query: str, retrieval_results: Dict[str, Any]) -> str:
        """Generate a formatted response based on retrieval results."""
        try:
            # Check for error or empty results first
            if not retrieval_results.get("success", False):
                return f"I couldn't find an answer to your question. {retrieval_results.get('error', '')}"
                
            results = retrieval_results.get("results", [])
            if not results:
                return "I couldn't find any relevant information to answer your question."
                
            # Prepare context from retrieved chunks
            context_parts = []
            for i, result in enumerate(results):
                score = result.get("score", 0)
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                page = metadata.get("page", "unknown")
                source = metadata.get("source", "unknown")
                
                # Only include chunks with reasonable similarity scores
                if score > 0.2:  # Adjust threshold as needed
                    context_parts.append(f"[Chunk {i+1}] (Source: {source}, Page: {page}, Relevance: {score:.2f})\n{text}")
            
            context = "\n\n".join(context_parts)
            
            # Get LLM provider
            llm_provider = await get_llm_provider()
            
            # Prepare prompt for response generation
            variables = {
                "query": query,
                "context": context,
                "num_sources": len(results)
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="qa_template",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            # Generate response
            response = await llm_provider.generate_text(
                input_message, 
                model_name=self.config.get("rag_agent", {}).get("model")
            )
            
            return response.strip()
                
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    async def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        try:
            result = await self.vector_store_tool.run(command="info")
            if result.success:
                info = result.data
                info.update({
                    "initialized": self.initialized,
                    "loaded_documents": self.loaded_documents
                })
                return info
            else:
                return {
                    "error": result.error,
                    "initialized": self.initialized,
                    "loaded_documents": self.loaded_documents
                }
                
        except Exception as e:
            self.logger.error(f"Error getting vector store info: {str(e)}")
            return {
                "error": f"Error getting vector store info: {str(e)}",
                "initialized": self.initialized,
                "loaded_documents": self.loaded_documents
            }
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the RAG agent workflow."""
        self._log_start(state)
        
        command = state.get("command", "query")
        
        try:
            if command == "initialize":
                # Initialize the RAG agent
                vector_store_dir = state.get("vector_store_dir")
                success = await self.initialize(vector_store_dir)
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "INITIALIZED" if success else "ERROR",
                    "initialized": success,
                    "error": None if success else "Failed to initialize RAG agent"
                }
                
            elif command == "add_document":
                # Add a document to the vector store
                pdf_path = state.get("pdf_path")
                if not pdf_path:
                    return {
                        **state,
                        "goto": state.get("goto", "END"),
                        "rag_status": "ERROR",
                        "error": "No PDF path provided"
                    }
                
                # Initialize if needed
                if not self.initialized:
                    await self.initialize()
                
                success = await self.add_document(pdf_path)
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "DOCUMENT_ADDED" if success else "ERROR",
                    "document_added": success,
                    "error": None if success else f"Failed to add document: {pdf_path}"
                }
                
            elif command == "save":
                # Save the vector store
                directory = state.get("directory")
                if not directory:
                    return {
                        **state,
                        "goto": state.get("goto", "END"),
                        "rag_status": "ERROR",
                        "error": "No directory provided for saving"
                    }
                
                success = await self.save_vector_store(directory)
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "SAVED" if success else "ERROR",
                    "saved": success,
                    "error": None if success else f"Failed to save to directory: {directory}"
                }
                
            elif command == "info":
                # Get information about the vector store
                info = await self.get_vector_store_info()
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "INFO",
                    "vector_store_info": info
                }
                
            elif command == "query":
                # Process a query and generate a response
                query = state.get("query")
                if not query:
                    return {
                        **state,
                        "goto": state.get("goto", "END"),
                        "rag_status": "ERROR",
                        "error": "No query provided"
                    }
                
                # Initialize if needed
                if not self.initialized:
                    await self.initialize()
                
                # Check if we have documents
                if not self.loaded_documents:
                    return {
                        **state,
                        "goto": state.get("goto", "END"),
                        "rag_status": "ERROR",
                        "error": "No documents loaded. Please add documents first."
                    }
                
                # Get the retrieval results
                retrieval_results = await self.answer_query(
                    query,
                    k=state.get("k", self.retrieval_k)
                )
                
                # Generate formatted response
                response = await self.generate_response(query, retrieval_results)
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "RESPONSE_READY",
                    "query": query,
                    "retrieval_results": retrieval_results,
                    "response": response
                }
                
            else:
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "ERROR",
                    "error": f"Unknown command: {command}"
                }
                
        except Exception as e:
            self.logger.error(f"Error in RAG agent: {str(e)}")
            return {
                **state,
                "goto": state.get("goto", "END"),
                "rag_status": "ERROR",
                "error": f"Error in RAG agent: {str(e)}"
            }
        finally:
            self._log_completion(state)