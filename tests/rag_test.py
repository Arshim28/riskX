#!/usr/bin/env python
import os
import asyncio
import argparse
from pprint import pprint
import sys
import json
import time

# Add the project root to the Python path if needed
# Adjust for being in the tests directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and set up logging first
from utils.logging import setup_logging, get_logger
# Initialize logging early
setup_logging("rag_test", level="INFO")
logger = get_logger("rag_test")

# Now import the rest
from tools.ocr_vector_store_tool import OCRVectorStoreTool
from utils.llm_provider import init_llm_provider
from utils.configuration import load_config, validate_config, GOOGLE_API_KEY, MISTRAL_API_KEY


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test RAG capabilities with OCRVectorStoreTool")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to process")
    parser.add_argument("--question", type=str, help="Question to ask about the document")
    parser.add_argument("--save_dir", type=str, help="Directory to save vector store")
    parser.add_argument("--load_dir", type=str, help="Directory to load vector store from")
    parser.add_argument("--k", type=int, help="Number of results to return", default=5)
    parser.add_argument("--config", type=str, help="Path to config file", default="config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config_data = load_config(args.config)
    
    # Validate API keys
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error("Please set the required API keys in your environment.")
        return
    
    # Update logging level if verbose
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info(f"Loaded configuration with chunk_size={config_data['chunk_size']}, "
               f"chunk_overlap={config_data['chunk_overlap']}, "
               f"index_type={config_data['index_type']}")
    
    # Make sure we have either a PDF to process or a vector store to load
    if not args.pdf and not args.load_dir:
        logger.error("Error: Either --pdf or --load_dir must be specified")
        return
    
    # Build tool configuration from our config data
    tool_config = {
        "ocr_vector_store": {
            "index_type": config_data["index_type"],
            "chunk_size": config_data["chunk_size"],
            "chunk_overlap": config_data["chunk_overlap"],
            "max_chunks_per_batch": 10  # Small batch size for testing
        },
        "ocr": {
            "api_key": MISTRAL_API_KEY
        },
        "embedding": {
            "api_key": GOOGLE_API_KEY,
            "model": "gemini-embedding-exp-03-07",
            "dimension": config_data["embedding_dimension"],
            "request_delay": config_data["request_delay"],
            "retry_max_attempts": config_data["retry_max_attempts"],
            "retry_base_delay": config_data["retry_base_delay"],
            "max_tokens": config_data["max_tokens"]
        },
        "vector_store": {
            "metric": "cosine",  # Use cosine similarity for better semantic matching
            "index_type": config_data["index_type"]
        }
    }
    
    # Create the OCRVectorStoreTool
    logger.info("Creating OCRVectorStoreTool...")
    start_time = time.time()
    tool = OCRVectorStoreTool(tool_config)
    
    try:
        # Load vector store if specified
        if args.load_dir:
            logger.info(f"Loading vector store from {args.load_dir}")
            load_result = await tool.run(command="load", directory=args.load_dir)
            if not load_result.success:
                logger.error(f"Failed to load vector store: {load_result.error}")
                return
            logger.info(f"Successfully loaded vector store: {load_result.data}")
        
        # Add document if specified
        if args.pdf:
            if not os.path.exists(args.pdf):
                logger.error(f"PDF file not found: {args.pdf}")
                return
                
            logger.info(f"Processing document: {args.pdf}")
            add_start_time = time.time()
            add_result = await tool.run(command="add_document", pdf_path=args.pdf)
            add_duration = time.time() - add_start_time
            
            if not add_result.success:
                logger.error(f"Failed to add document: {add_result.error}")
                return
                
            logger.info(f"Successfully processed document in {add_duration:.2f}s")
            logger.info(f"Document info: {json.dumps(add_result.data, indent=2)}")
        
        # Save vector store if specified
        if args.save_dir:
            logger.info(f"Saving vector store to {args.save_dir}")
            save_result = await tool.run(command="save", directory=args.save_dir)
            if not save_result.success:
                logger.error(f"Failed to save vector store: {save_result.error}")
            else:
                logger.info(f"Successfully saved vector store: {save_result.data}")
        
        # Answer question if specified
        if args.question:
            logger.info(f"Answering question: '{args.question}'")
            question_start_time = time.time()
            answer_result = await tool.run(
                command="answer_question", 
                question=args.question,
                k=args.k
            )
            question_duration = time.time() - question_start_time
            
            if not answer_result.success:
                logger.error(f"Failed to answer question: {answer_result.error}")
                return
            
            logger.info(f"Question answered in {question_duration:.2f}s")
            
            # Display results
            print("\n" + "="*80)
            print(f"QUESTION: {args.question}")
            print("="*80)
            
            results = answer_result.data.get("results", [])
            if not results:
                print("No relevant information found.")
            else:
                for i, result in enumerate(results):
                    print(f"\nRESULT {i+1} (score: {result['score']:.4f}):")
                    print(f"SOURCE: Page {result['metadata'].get('page', 'unknown')}")
                    print("-"*80)
                    print(result['text'][:500] + ("..." if len(result['text']) > 500 else ""))
                    print("-"*80)
        
        # Get and display info
        info_result = await tool.run(command="info")
        if info_result.success:
            logger.info("Vector store info:")
            pprint(info_result.data)
            
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
    finally:
        total_duration = time.time() - start_time
        logger.info(f"Total execution time: {total_duration:.2f}s")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())