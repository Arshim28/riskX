# tests/test_rag_agent.py
import pytest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

from agents.rag_agent import RAGAgent


@pytest.fixture
def config():
    return {
        "rag_agent": {
            "retrieval_k": 5,
            "reranking_enabled": False,
            "max_input_tokens": 4000
        },
        "ocr_vector_store": {},
        "models": {
            "model": "gemini-2.0-pro"
        }
    }


@pytest.fixture
def state():
    return {
        "command": "query",
        "query": "What are the financial issues with Test Company?",
        "session_id": "test_session"
    }


@pytest.fixture
def retrieval_results():
    return {
        "success": True,
        "question": "What are the financial issues with Test Company?",
        "results": [
            {
                "text": "Test Company had accounting irregularities in 2022.",
                "metadata": {"source": "document1.pdf", "page": "5"},
                "score": 0.85
            },
            {
                "text": "The company faced issues with revenue recognition.",
                "metadata": {"source": "document2.pdf", "page": "12"},
                "score": 0.75
            }
        ]
    }


@pytest.fixture
def agent(config):
    with patch("utils.prompt_manager.get_prompt_manager"), \
         patch("utils.logging.get_logger"), \
         patch("tools.ocr_vector_store_tool.OCRVectorStoreTool"):
        return RAGAgent(config)


@pytest.mark.asyncio
async def test_initialize(agent):
    with patch("os.path.exists", return_value=True), \
         patch.object(agent, "vector_store_tool") as mock_tool, \
         patch("builtins.open", mock_open(read_data='{"documents": {}, "topics": {}}')):
        
        result_mock = MagicMock()
        result_mock.success = True
        result_mock.data = {"chunks": 100}
        mock_tool.run = AsyncMock(return_value=result_mock)
        
        success = await agent.initialize("/path/to/store")
        
        assert success is True
        assert agent.initialized is True
        
        mock_tool.run.assert_called_with(
            command="load",
            directory="/path/to/store"
        )


@pytest.mark.asyncio
async def test_add_document(agent):
    agent.initialized = True
    
    with patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1024), \
         patch.object(agent, "vector_store_tool") as mock_tool:
        
        result_mock = MagicMock()
        result_mock.success = True
        mock_tool.run = AsyncMock(return_value=result_mock)
        
        success = await agent.add_document("/path/to/document.pdf", ["topic1", "topic2"])
        
        assert success is True
        assert "document.pdf" in agent.loaded_documents
        assert "document.pdf" in agent.document_collection
        assert "topic1" in agent.document_topics
        assert "topic2" in agent.document_topics
        assert "document.pdf" in agent.document_topics["topic1"]
        assert "document.pdf" in agent.document_topics["topic2"]
        
        mock_tool.run.assert_called_with(
            command="add_document",
            pdf_path="/path/to/document.pdf"
        )


@pytest.mark.asyncio
async def test_answer_query(agent, retrieval_results):
    agent.initialized = True
    agent.loaded_documents = ["document1.pdf", "document2.pdf"]
    
    with patch.object(agent, "vector_store_tool") as mock_tool:
        mock_tool.run = AsyncMock(return_value=MagicMock(success=True, data=retrieval_results))
        
        result = await agent.answer_query(
            "What are the financial issues?",
            session_id="test_session",
            filter_topics=["topic1"]
        )
        
        assert result == retrieval_results
        assert "test_session" in agent.session_questions
        assert len(agent.session_questions["test_session"]) == 1
        assert agent.session_questions["test_session"][0]["query"] == "What are the financial issues?"
        
        mock_tool.run.assert_called_with(
            command="answer_question", 
            question="What are the financial issues?",
            k=5,
            filter_docs=[]  # Empty because document_topics["topic1"] doesn't exist in this test
        )


@pytest.mark.asyncio
async def test_generate_response(agent, retrieval_results):
    with patch("utils.llm_provider.get_llm_provider") as mock_provider:
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "Based on the documents, Test Company had accounting irregularities in 2022 and issues with revenue recognition."
        mock_provider.return_value = mock_llm
        
        response = await agent.generate_response(
            "What are the financial issues?",
            retrieval_results,
            session_id="test_session"
        )
        
        assert "accounting irregularities" in response
        assert "revenue recognition" in response
        
        mock_llm.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_run_query_command(agent):
    with patch.object(agent, "initialize") as mock_init, \
         patch.object(agent, "answer_query") as mock_answer, \
         patch.object(agent, "generate_response") as mock_generate:
        
        mock_init.return_value = True
        mock_answer.return_value = {
            "success": True,
            "results": [{"text": "Test result", "score": 0.8}]
        }
        mock_generate.return_value = "This is the response."
        
        state = {
            "command": "query",
            "query": "What are the issues?",
            "session_id": "test_session"
        }
        
        result = await agent.run(state)
        
        assert result["goto"] == "END"
        assert result["rag_status"] == "RESPONSE_READY"
        assert result["query"] == "What are the issues?"
        assert result["response"] == "This is the response."
        assert "retrieval_results" in result
        
        mock_init.assert_called_once()
        mock_answer.assert_called_once()
        mock_generate.assert_called_once()


@pytest.mark.asyncio
async def test_run_add_document_command(agent):
    with patch.object(agent, "initialize") as mock_init, \
         patch.object(agent, "add_document") as mock_add:
        
        mock_init.return_value = True
        mock_add.return_value = True
        
        state = {
            "command": "add_document",
            "pdf_path": "/path/to/document.pdf",
            "topics": ["topic1", "topic2"]
        }
        
        result = await agent.run(state)
        
        assert result["goto"] == "END"
        assert result["rag_status"] == "DOCUMENT_ADDED"
        assert result["document_added"] is True
        
        mock_init.assert_called_once()
        mock_add.assert_called_once_with("/path/to/document.pdf", ["topic1", "topic2"])