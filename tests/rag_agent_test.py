import pytest
import pytest_asyncio
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from typing import Dict, List, Any, Tuple, Optional

from agents.rag_agent import RAGAgent
from utils.prompt_manager import PromptManager
from utils.llm_provider import LLMProvider, get_llm_provider
from tools.ocr_vector_store_tool import OCRVectorStoreTool
from base.base_tools import ToolResult


# Mock data for tests
QUERY = "What is the main topic of the document?"
VECTOR_STORE_DIR = "/path/to/vector_store"
PDF_PATH = "/path/to/document.pdf"

MOCK_RETRIEVE_RESULTS = {
    "success": True,
    "question": QUERY,
    "results": [
        {
            "text": "The main topic of this document is artificial intelligence and its applications in business.",
            "metadata": {"source": "document.pdf", "page": 1},
            "score": 0.85
        },
        {
            "text": "Artificial intelligence technologies can transform how businesses operate across various departments.",
            "metadata": {"source": "document.pdf", "page": 2},
            "score": 0.75
        }
    ],
    "result_count": 2
}

MOCK_VECTOR_STORE_INFO = {
    "initialized": True,
    "dimension": 768,
    "chunks": 50,
    "index_type": "Flat",
    "metric": "cosine",
    "config": {"index_type": "Flat", "metric": "cosine"}
}

MOCK_RESPONSE = "Based on the retrieved information, the main topic of the document is artificial intelligence and its applications in business. The document discusses how AI technologies can transform business operations across various departments, as mentioned on pages 1 and 2."


@pytest.fixture(scope="module", autouse=True)
def setup_llm_provider():
    """Initialize the LLM Provider once for all tests"""
    # Store the original provider instance
    import utils.llm_provider
    original_instance = utils.llm_provider._provider_instance
    
    # Create a mock provider
    mock_provider = AsyncMock(spec=LLMProvider)
    
    # Initialize the global provider with our mock
    utils.llm_provider._provider_instance = mock_provider
    
    # This is yielded to tests that need it
    yield mock_provider
    
    # Restore the original provider instance after all tests
    utils.llm_provider._provider_instance = original_instance


@pytest_asyncio.fixture
async def mock_logger():
    """Fixture for mocking the logger"""
    mock_log = MagicMock(spec=logging.Logger)
    
    with patch("utils.logging.get_logger") as mock_get_logger:
        mock_get_logger.return_value = mock_log
        yield mock_log


@pytest_asyncio.fixture
async def mock_vector_store_tool():
    """Fixture for mocking the OCRVectorStoreTool"""
    mock_tool = AsyncMock(spec=OCRVectorStoreTool)
    
    # Set up the mock run method
    add_result = ToolResult(success=True, data={"success": True, "document": "document.pdf"})
    load_result = ToolResult(success=True, data={"success": True, "loaded": True, "chunks": 50})
    save_result = ToolResult(success=True, data={"success": True, "saved": True, "directory": VECTOR_STORE_DIR})
    info_result = ToolResult(success=True, data=MOCK_VECTOR_STORE_INFO)
    retrieve_result = ToolResult(success=True, data=MOCK_RETRIEVE_RESULTS)
    
    async def mock_run(**kwargs):
        command = kwargs.get("command")
        if command == "add_document":
            return add_result
        elif command == "load":
            return load_result
        elif command == "save":
            return save_result
        elif command == "info":
            return info_result
        elif command == "answer_question":
            return retrieve_result
        else:
            return ToolResult(success=False, error=f"Unknown command: {command}")
    
    mock_tool.run = AsyncMock(side_effect=mock_run)
    
    yield mock_tool


@pytest_asyncio.fixture
async def mock_prompt_manager():
    """Fixture for mocking the PromptManager"""
    mock_manager = MagicMock(spec=PromptManager)
    mock_manager.get_prompt = MagicMock(return_value=("mocked system prompt", "mocked human prompt"))
    
    # Store original function to restore later
    import utils.prompt_manager
    original_get_pm = utils.prompt_manager.get_prompt_manager
    
    # Replace with our mock
    utils.prompt_manager.get_prompt_manager = MagicMock(return_value=mock_manager)
    
    yield mock_manager
    
    # Restore original function
    utils.prompt_manager.get_prompt_manager = original_get_pm


@pytest_asyncio.fixture
async def rag_agent(mock_logger, setup_llm_provider, mock_prompt_manager, mock_vector_store_tool):
    """Fixture to create a RAGAgent with mocked dependencies"""
    config = {
        "rag_agent": {
            "retrieval_k": 5,
            "reranking_enabled": False,
            "max_input_tokens": 4000,
            "model": "test-model"
        }
    }
    
    with patch.object(RAGAgent, "__init__", return_value=None):
        agent = RAGAgent(None)
        
        # Set the mocked properties
        agent.config = config
        agent.logger = mock_logger
        agent.prompt_manager = mock_prompt_manager
        agent.vector_store_tool = mock_vector_store_tool
        agent.name = "rag_agent"
        agent.retrieval_k = config["rag_agent"]["retrieval_k"]
        agent.reranking_enabled = config["rag_agent"]["reranking_enabled"]
        agent.max_input_tokens = config["rag_agent"]["max_input_tokens"]
        agent.initialized = False
        agent.loaded_documents = []
        
        # Add _log_start and _log_completion methods for safety
        agent._log_start = MagicMock(return_value=None)
        agent._log_completion = MagicMock(return_value=None)
        
        yield agent


@pytest.mark.asyncio
async def test_initialize(rag_agent, mock_vector_store_tool):
    """Test initialization of RAG agent with vector store directory"""
    # Initialize mock_vector_store_tool.run
    mock_vector_store_tool.run.reset_mock()
    
    # Mock os.path.exists
    with patch("os.path.exists", return_value=True):
        # Call the method
        result = await rag_agent.initialize(VECTOR_STORE_DIR)
        
        # Verify results
        assert result is True
        assert rag_agent.initialized is True
        mock_vector_store_tool.run.assert_called_once()
        assert mock_vector_store_tool.run.call_args[1]["command"] == "load"
        assert mock_vector_store_tool.run.call_args[1]["directory"] == VECTOR_STORE_DIR


@pytest.mark.asyncio
async def test_initialize_empty(rag_agent, mock_vector_store_tool):
    """Test initialization of RAG agent without vector store directory"""
    # Reset agent state
    rag_agent.initialized = False
    
    # Reset mock
    mock_vector_store_tool.run.reset_mock()
    
    # Call the method
    result = await rag_agent.initialize()
    
    # Verify results
    assert result is True
    assert rag_agent.initialized is True
    # Should not call vector_store_tool.run for empty initialization
    mock_vector_store_tool.run.assert_not_called()


@pytest.mark.asyncio
async def test_add_document(rag_agent, mock_vector_store_tool):
    """Test adding a document to the vector store"""
    # Reset mock
    mock_vector_store_tool.run.reset_mock()
    
    # Mock os.path.exists
    with patch("os.path.exists", return_value=True):
        # Call the method
        result = await rag_agent.add_document(PDF_PATH)
        
        # Verify results
        assert result is True
        mock_vector_store_tool.run.assert_called_once()
        assert mock_vector_store_tool.run.call_args[1]["command"] == "add_document"
        assert mock_vector_store_tool.run.call_args[1]["pdf_path"] == PDF_PATH
        assert "document.pdf" in rag_agent.loaded_documents


@pytest.mark.asyncio
async def test_add_document_not_found(rag_agent, mock_vector_store_tool):
    """Test adding a document that doesn't exist"""
    # Reset mock
    mock_vector_store_tool.run.reset_mock()
    
    # Mock os.path.exists
    with patch("os.path.exists", return_value=False):
        # Call the method
        result = await rag_agent.add_document(PDF_PATH)
        
        # Verify results
        assert result is False
        mock_vector_store_tool.run.assert_not_called()


@pytest.mark.asyncio
async def test_save_vector_store(rag_agent, mock_vector_store_tool):
    """Test saving the vector store"""
    # Set initialized to True
    rag_agent.initialized = True
    
    # Reset mock
    mock_vector_store_tool.run.reset_mock()
    
    # Call the method
    result = await rag_agent.save_vector_store(VECTOR_STORE_DIR)
    
    # Verify results
    assert result is True
    mock_vector_store_tool.run.assert_called_once()
    assert mock_vector_store_tool.run.call_args[1]["command"] == "save"
    assert mock_vector_store_tool.run.call_args[1]["directory"] == VECTOR_STORE_DIR


@pytest.mark.asyncio
async def test_save_vector_store_not_initialized(rag_agent, mock_vector_store_tool):
    """Test saving the vector store when not initialized"""
    # Set initialized to False
    rag_agent.initialized = False
    
    # Reset mock
    mock_vector_store_tool.run.reset_mock()
    
    # Call the method
    result = await rag_agent.save_vector_store(VECTOR_STORE_DIR)
    
    # Verify results
    assert result is False
    mock_vector_store_tool.run.assert_not_called()


@pytest.mark.asyncio
async def test_answer_query(rag_agent, mock_vector_store_tool):
    """Test answering a query"""
    # Set initialized to True
    rag_agent.initialized = True
    
    # Reset mock
    mock_vector_store_tool.run.reset_mock()
    
    # Call the method
    result = await rag_agent.answer_query(QUERY)
    
    # Verify results
    assert result == MOCK_RETRIEVE_RESULTS
    mock_vector_store_tool.run.assert_called_once()
    assert mock_vector_store_tool.run.call_args[1]["command"] == "answer_question"
    assert mock_vector_store_tool.run.call_args[1]["question"] == QUERY
    assert mock_vector_store_tool.run.call_args[1]["k"] == rag_agent.retrieval_k


@pytest.mark.asyncio
async def test_answer_query_not_initialized(rag_agent, mock_vector_store_tool):
    """Test answering a query when not initialized"""
    # Set initialized to False
    rag_agent.initialized = False
    
    # Reset mock
    mock_vector_store_tool.run.reset_mock()
    
    # Call the method
    result = await rag_agent.answer_query(QUERY)
    
    # Verify results
    assert result["success"] is False
    assert "RAG agent not initialized" in result["error"]
    mock_vector_store_tool.run.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response(rag_agent, setup_llm_provider):
    """Test generating a response from retrieval results"""
    # Set up mock LLM response
    setup_llm_provider.generate_text.reset_mock()
    setup_llm_provider.generate_text.return_value = MOCK_RESPONSE
    
    # Call the method
    result = await rag_agent.generate_response(QUERY, MOCK_RETRIEVE_RESULTS)
    
    # Verify results
    assert result == MOCK_RESPONSE
    setup_llm_provider.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_generate_response_no_results(rag_agent, setup_llm_provider):
    """Test generating a response with no results"""
    # Reset mock
    setup_llm_provider.generate_text.reset_mock()
    
    # Call the method with empty results
    result = await rag_agent.generate_response(QUERY, {"success": True, "results": []})
    
    # Verify results
    assert "couldn't find any relevant information" in result
    setup_llm_provider.generate_text.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_error(rag_agent, setup_llm_provider):
    """Test generating a response with retrieval error"""
    # Reset mock
    setup_llm_provider.generate_text.reset_mock()
    
    # Call the method with error
    result = await rag_agent.generate_response(QUERY, {"success": False, "error": "Test error"})
    
    # Verify results
    assert "couldn't find an answer" in result
    assert "Test error" in result
    setup_llm_provider.generate_text.assert_not_called()


@pytest.mark.asyncio
async def test_get_vector_store_info(rag_agent, mock_vector_store_tool):
    """Test getting vector store info"""
    # Set initialized and loaded_documents
    rag_agent.initialized = True
    rag_agent.loaded_documents = ["document.pdf"]
    
    # Reset mock
    mock_vector_store_tool.run.reset_mock()
    
    # Call the method
    result = await rag_agent.get_vector_store_info()
    
    # Verify results
    assert result["initialized"] is True
    assert "document.pdf" in result["loaded_documents"]
    assert result["dimension"] == MOCK_VECTOR_STORE_INFO["dimension"]
    assert result["chunks"] == MOCK_VECTOR_STORE_INFO["chunks"]
    mock_vector_store_tool.run.assert_called_once()
    assert mock_vector_store_tool.run.call_args[1]["command"] == "info"


@pytest.mark.asyncio
async def test_run_initialize(rag_agent):
    """Test run method with initialize command"""
    # Mock the initialize method
    with patch.object(rag_agent, "initialize", return_value=True) as mock_initialize:
        # Create state
        state = {
            "command": "initialize",
            "vector_store_dir": VECTOR_STORE_DIR
        }
        
        # Call the method
        result = await rag_agent.run(state)
        
        # Verify results
        assert result["rag_status"] == "INITIALIZED"
        assert result["initialized"] is True
        assert result["error"] is None
        mock_initialize.assert_called_once_with(VECTOR_STORE_DIR)


@pytest.mark.asyncio
async def test_run_add_document(rag_agent):
    """Test run method with add_document command"""
    # Mock the add_document and initialize methods
    with patch.object(rag_agent, "add_document", return_value=True) as mock_add_document, \
         patch.object(rag_agent, "initialize", return_value=True) as mock_initialize:
        
        # Set initialized to False to test auto-initialization
        rag_agent.initialized = False
        
        # Create state
        state = {
            "command": "add_document",
            "pdf_path": PDF_PATH
        }
        
        # Call the method
        result = await rag_agent.run(state)
        
        # Verify results
        assert result["rag_status"] == "DOCUMENT_ADDED"
        assert result["document_added"] is True
        assert result["error"] is None
        mock_initialize.assert_called_once()
        mock_add_document.assert_called_once_with(PDF_PATH)


@pytest.mark.asyncio
async def test_run_query(rag_agent):
    """Test run method with query command"""
    # Mock the answer_query and generate_response methods
    with patch.object(rag_agent, "answer_query", return_value=MOCK_RETRIEVE_RESULTS) as mock_answer_query, \
         patch.object(rag_agent, "generate_response", return_value=MOCK_RESPONSE) as mock_generate_response:
        
        # Set initialized to True and add a document
        rag_agent.initialized = True
        rag_agent.loaded_documents = ["document.pdf"]
        
        # Create state
        state = {
            "command": "query",
            "query": QUERY
        }
        
        # Call the method
        result = await rag_agent.run(state)
        
        # Verify results
        assert result["rag_status"] == "RESPONSE_READY"
        assert result["query"] == QUERY
        assert result["retrieval_results"] == MOCK_RETRIEVE_RESULTS
        assert result["response"] == MOCK_RESPONSE
        mock_answer_query.assert_called_once_with(QUERY, k=rag_agent.retrieval_k)
        mock_generate_response.assert_called_once_with(QUERY, MOCK_RETRIEVE_RESULTS)


@pytest.mark.asyncio
async def test_run_query_no_documents(rag_agent):
    """Test run method with query command but no documents loaded"""
    # Set initialized to True but no documents
    rag_agent.initialized = True
    rag_agent.loaded_documents = []
    
    # Create state
    state = {
        "command": "query",
        "query": QUERY
    }
    
    # Call the method
    result = await rag_agent.run(state)
    
    # Verify results
    assert result["rag_status"] == "ERROR"
    assert "No documents loaded" in result["error"]


@pytest.mark.asyncio
async def test_run_unknown_command(rag_agent):
    """Test run method with unknown command"""
    # Create state with unknown command
    state = {
        "command": "unknown_command"
    }
    
    # Call the method
    result = await rag_agent.run(state)
    
    # Verify results
    assert result["rag_status"] == "ERROR"
    assert "Unknown command" in result["error"]