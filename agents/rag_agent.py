# agents/rag_agent.py
from typing import Dict, List, Any, Optional
import json
import asyncio
import os
from datetime import datetime
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
        self.prompt_manager = get_prompt_manager()
        
        self.vector_store_tool = OCRVectorStoreTool(config)
        
        self.retrieval_k = config.get("rag_agent", {}).get("retrieval_k", 5)
        self.reranking_enabled = config.get("rag_agent", {}).get("reranking_enabled", False)
        self.max_input_tokens = config.get("rag_agent", {}).get("max_input_tokens", 4000)
        
        self.initialized = False
        self.loaded_documents = []
        
        self.document_collection = {}
        self.document_topics = {}
        self.last_session_id = None
        self.session_questions = {}
    
    async def initialize(self, vector_store_dir: Optional[str] = None) -> bool:
        if self.initialized and vector_store_dir is None:
            self.logger.info("RAG agent already initialized")
            return True
            
        if vector_store_dir and os.path.exists(vector_store_dir):
            self.logger.info(f"Loading vector store from {vector_store_dir}")
            result = await self.vector_store_tool.run(
                command="load",
                directory=vector_store_dir
            )
            if result.success:
                self.initialized = True
                self.logger.info(f"Successfully loaded vector store with {result.data.get('chunks', 0)} chunks")
                
                metadata_path = os.path.join(vector_store_dir, "document_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.document_collection = metadata.get("documents", {})
                        self.document_topics = metadata.get("topics", {})
                        self.loaded_documents = list(self.document_collection.keys())
                
                return True
            else:
                self.logger.error(f"Failed to load vector store: {result.error}")
                return False
        else:
            self.logger.info("Initializing empty RAG agent")
            self.initialized = True
            return True
    
    async def add_document(self, pdf_path: str, topics: Optional[List[str]] = None) -> bool:
        if not os.path.exists(pdf_path):
            self.logger.error(f"Document not found: {pdf_path}")
            return False
            
        self.logger.info(f"Adding document to vector store: {pdf_path}")
        result = await self.vector_store_tool.run(
            command="add_document",
            pdf_path=pdf_path
        )
        
        if result.success:
            document_id = os.path.basename(pdf_path)
            document_size = os.path.getsize(pdf_path)
            
            self.document_collection[document_id] = {
                "path": pdf_path,
                "added_at": datetime.now().isoformat(),
                "size_bytes": document_size,
                "topics": topics or ["unclassified"]
            }
            
            for topic in (topics or ["unclassified"]):
                if topic not in self.document_topics:
                    self.document_topics[topic] = []
                self.document_topics[topic].append(document_id)
            
            self.loaded_documents.append(document_id)
            self.logger.info(f"Successfully added document: {document_id}")
            return True
        else:
            self.logger.error(f"Failed to add document: {result.error}")
            return False
    
    async def save_vector_store(self, directory: str) -> bool:
        if not self.initialized:
            self.logger.warning("Cannot save vector store: RAG agent not initialized")
            return False
            
        self.logger.info(f"Saving vector store to {directory}")
        result = await self.vector_store_tool.run(
            command="save",
            directory=directory
        )
        
        if result.success:
            metadata = {
                "documents": self.document_collection,
                "topics": self.document_topics,
                "last_updated": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(directory, "document_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Successfully saved vector store to {directory}")
            return True
        else:
            self.logger.error(f"Failed to save vector store: {result.error}")
            return False
    
    @retry(stop_after_attempt=3, wait=wait_exponential(multiplier=1, min=2, max=10))
    async def answer_query(self, query: str, session_id: Optional[str] = None, 
                        filter_topics: Optional[List[str]] = None, k: Optional[int] = None) -> Dict[str, Any]:
        if not self.initialized:
            self.logger.warning("RAG agent not initialized")
            return {
                "success": False,
                "error": "RAG agent not initialized. Please add documents first."
            }
            
        k_value = k if k is not None else self.retrieval_k
        
        self.logger.info(f"Processing query: {query}")
        
        if session_id:
            self.last_session_id = session_id
            if session_id not in self.session_questions:
                self.session_questions[session_id] = []
            self.session_questions[session_id].append({
                "query": query,
                "timestamp": datetime.now().isoformat()
            })
        
        retrieval_params = {"k": k_value}
        if filter_topics:
            filtered_docs = []
            for topic in filter_topics:
                if topic in self.document_topics:
                    filtered_docs.extend(self.document_topics[topic])
                    
            if filtered_docs:
                retrieval_params["filter_docs"] = filtered_docs
                
            self.logger.info(f"Filtering retrieval to {len(filtered_docs)} documents from topics: {filter_topics}")
        
        result = await self.vector_store_tool.run(
            command="answer_question", 
            question=query,
            **retrieval_params
        )
        
        if result.success:
            return result.data
        else:
            self.logger.error(f"Query processing failed: {result.error}")
            return {
                "success": False,
                "error": f"Query processing failed: {result.error}"
            }
    
    @retry(stop_after_attempt=3, wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_response(self, query: str, retrieval_results: Dict[str, Any], 
                            session_id: Optional[str] = None) -> str:
        if not retrieval_results.get("success", False):
            return f"I couldn't find an answer to your question. {retrieval_results.get('error', '')}"
            
        results = retrieval_results.get("results", [])
        if not results:
            return "I couldn't find any relevant information to answer your question."
            
        context_parts = []
        for i, result in enumerate(results):
            score = result.get("score", 0)
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            page = metadata.get("page", "unknown")
            source = metadata.get("source", "unknown")
            
            if score > 0.2:
                context_parts.append(f"[Document: {source}, Page: {page}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        conversation_history = ""
        if session_id and session_id in self.session_questions:
            prev_questions = self.session_questions[session_id][:-1]
            if prev_questions:
                history = [f"Q: {q['query']}" for q in prev_questions[-3:]]
                conversation_history = "\n".join(history)
        
        llm_provider = await get_llm_provider()
        
        variables = {
            "query": query,
            "context": context,
            "num_sources": len(results),
            "conversation_history": conversation_history
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
        
        response = await llm_provider.generate_text(
            input_message, 
            model_name=self.config.get("rag_agent", {}).get("model")
        )
        
        return response.strip()
    
    @retry(stop_after_attempt=3, wait=wait_exponential(multiplier=1, min=2, max=10))
    async def list_topics(self) -> Dict[str, Any]:
        topic_stats = {}
        
        for topic, docs in self.document_topics.items():
            topic_stats[topic] = {
                "document_count": len(docs),
                "documents": docs
            }
            
        return {
            "success": True,
            "topics": topic_stats,
            "total_topics": len(topic_stats)
        }
    
    @retry(stop_after_attempt=3, wait=wait_exponential(multiplier=1, min=2, max=10))
    async def categorize_documents(self) -> Dict[str, Any]:
        if not self.initialized or not self.loaded_documents:
            return {
                "success": False,
                "error": "No documents to categorize"
            }
            
        uncategorized_docs = []
        for doc_id, metadata in self.document_collection.items():
            if "unclassified" in metadata.get("topics", []):
                uncategorized_docs.append(doc_id)
                
        if not uncategorized_docs:
            return {
                "success": True,
                "message": "No uncategorized documents found"
            }
            
        self.logger.info(f"Auto-categorizing {len(uncategorized_docs)} documents")
        
        updated_count = 0
        for doc_id in uncategorized_docs:
            sample_query = f"summarize document {doc_id}"
            result = await self.vector_store_tool.run(
                command="answer_question",
                question=sample_query,
                k=5
            )
            
            if not result.success:
                continue
                
            sample_text = ""
            for chunk in result.data.get("results", [])[:3]:
                sample_text += chunk.get("text", "") + "\n\n"
                
            if not sample_text:
                continue
                
            llm_provider = await get_llm_provider()
            
            variables = {
                "document_id": doc_id,
                "sample_text": sample_text
            }
            
            system_prompt, human_prompt = self.prompt_manager.get_prompt(
                agent_name=self.name,
                operation="document_categorization",
                variables=variables
            )
            
            input_message = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = await llm_provider.generate_text(
                input_message, 
                model_name=self.config.get("rag_agent", {}).get("model")
            )
            
            try:
                topics = json.loads(response)
                if isinstance(topics, list) and topics:
                    self.document_collection[doc_id]["topics"] = topics
                    
                    if "unclassified" in self.document_topics:
                        self.document_topics["unclassified"] = [d for d in self.document_topics["unclassified"] if d != doc_id]
                    
                    for topic in topics:
                        if topic not in self.document_topics:
                            self.document_topics[topic] = []
                        if doc_id not in self.document_topics[topic]:
                            self.document_topics[topic].append(doc_id)
                    
                    updated_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error parsing topics for document {doc_id}: {str(e)}")
                
        return {
            "success": True,
            "categorized_count": updated_count,
            "total_uncategorized": len(uncategorized_docs)
        }
    
    async def get_vector_store_info(self) -> Dict[str, Any]:
        result = await self.vector_store_tool.run(command="info")
        if result.success:
            info = result.data
            info.update({
                "initialized": self.initialized,
                "loaded_documents": self.loaded_documents,
                "document_count": len(self.document_collection),
                "topic_count": len(self.document_topics),
                "topics": list(self.document_topics.keys()),
                "session_count": len(self.session_questions)
            })
            return info
        else:
            return {
                "error": result.error,
                "initialized": self.initialized,
                "loaded_documents": self.loaded_documents
            }
    
    @retry(stop_after_attempt=3, wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_topic_report(self, topic: str) -> Dict[str, Any]:
        if topic not in self.document_topics:
            return {
                "success": False,
                "error": f"Topic '{topic}' not found"
            }
            
        doc_ids = self.document_topics[topic]
        if not doc_ids:
            return {
                "success": False,
                "error": f"No documents found for topic '{topic}'"
            }
            
        question = f"What are the key points about {topic}?"
        result = await self.vector_store_tool.run(
            command="answer_question",
            question=question,
            filter_docs=doc_ids,
            k=10
        )
        
        if not result.success:
            return {
                "success": False,
                "error": f"Failed to retrieve information for topic '{topic}'"
            }
            
        key_chunks = []
        for chunk in result.data.get("results", []):
            key_chunks.append(chunk.get("text", ""))
            
        context = "\n\n".join(key_chunks)
        
        llm_provider = await get_llm_provider()
        
        variables = {
            "topic": topic,
            "document_count": len(doc_ids),
            "context": context
        }
        
        system_prompt, human_prompt = self.prompt_manager.get_prompt(
            agent_name=self.name,
            operation="topic_report",
            variables=variables
        )
        
        input_message = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = await llm_provider.generate_text(
            input_message, 
            model_name=self.config.get("rag_agent", {}).get("model")
        )
        
        try:
            report = json.loads(response)
        except json.JSONDecodeError:
            report = {
                "summary": response,
                "key_points": [f"Error parsing structured report for topic: {topic}"]
            }
            
        return {
            "success": True,
            "topic": topic,
            "document_count": len(doc_ids),
            "report": report
        }
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._log_start(state)
        
        command = state.get("command", "query")
        
        try:
            if command == "initialize":
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
                pdf_path = state.get("pdf_path")
                topics = state.get("topics")
                
                if not pdf_path:
                    return {
                        **state,
                        "goto": state.get("goto", "END"),
                        "rag_status": "ERROR",
                        "error": "No PDF path provided"
                    }
                
                if not self.initialized:
                    await self.initialize()
                
                success = await self.add_document(pdf_path, topics)
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "DOCUMENT_ADDED" if success else "ERROR",
                    "document_added": success,
                    "error": None if success else f"Failed to add document: {pdf_path}"
                }
                
            elif command == "save":
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
                info = await self.get_vector_store_info()
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "INFO",
                    "vector_store_info": info
                }
                
            elif command == "query":
                query = state.get("query")
                session_id = state.get("session_id")
                filter_topics = state.get("filter_topics")
                
                if not query:
                    return {
                        **state,
                        "goto": state.get("goto", "END"),
                        "rag_status": "ERROR",
                        "error": "No query provided"
                    }
                
                if not self.initialized:
                    await self.initialize()
                
                if not self.loaded_documents:
                    return {
                        **state,
                        "goto": state.get("goto", "END"),
                        "rag_status": "ERROR",
                        "error": "No documents loaded. Please add documents first."
                    }
                
                retrieval_results = await self.answer_query(
                    query,
                    session_id=session_id,
                    filter_topics=filter_topics,
                    k=state.get("k", self.retrieval_k)
                )
                
                response = await self.generate_response(
                    query, 
                    retrieval_results,
                    session_id=session_id
                )
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "RESPONSE_READY",
                    "query": query,
                    "retrieval_results": retrieval_results,
                    "response": response
                }
                
            elif command == "list_topics":
                topics_result = await self.list_topics()
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "TOPICS_LISTED",
                    "topics_result": topics_result
                }
                
            elif command == "categorize_documents":
                categorization_result = await self.categorize_documents()
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "DOCUMENTS_CATEGORIZED",
                    "categorization_result": categorization_result
                }
                
            elif command == "generate_topic_report":
                topic = state.get("topic")
                
                if not topic:
                    return {
                        **state,
                        "goto": state.get("goto", "END"),
                        "rag_status": "ERROR",
                        "error": "No topic provided"
                    }
                
                report_result = await self.generate_topic_report(topic)
                
                return {
                    **state,
                    "goto": state.get("goto", "END"),
                    "rag_status": "TOPIC_REPORT_READY",
                    "topic_report": report_result
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