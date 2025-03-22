import os
import sys
import json
import argparse
import asyncio
import yaml
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

from utils.logging import setup_logging, get_logger
from utils.llm_provider import init_llm_provider
from utils.prompt_manager import init_prompt_manager
from langgraph_workflow import ForensicWorkflow, create_and_run_workflow


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial Forensic Analysis System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a company")
    analyze_parser.add_argument("--company", required=True, help="Name of the company to analyze")
    analyze_parser.add_argument("--industry", help="Industry of the company")
    analyze_parser.add_argument("--config", help="Path to config file")
    analyze_parser.add_argument("--output", help="Output file for report")
    analyze_parser.add_argument("--mode", choices=["full", "quick"], default="full", 
                               help="Analysis mode (full or quick)")
    
    # RAG commands
    rag_parser = subparsers.add_parser("rag", help="RAG operations")
    rag_subparsers = rag_parser.add_subparsers(dest="rag_command", help="RAG command to execute")
    
    # RAG add command
    rag_add_parser = rag_subparsers.add_parser("add", help="Add a document to RAG")
    rag_add_parser.add_argument("--file", required=True, help="Path to PDF file")
    rag_add_parser.add_argument("--topics", help="Comma-separated list of topics")
    
    # RAG query command
    rag_query_parser = rag_subparsers.add_parser("query", help="Query RAG system")
    rag_query_parser.add_argument("--query", required=True, help="Query to execute")
    rag_query_parser.add_argument("--topics", help="Comma-separated list of topics to filter by")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    
    # Version command
    subparsers.add_parser("version", help="Print version information")
    
    return parser.parse_args()


async def run_analyze(args):
    """Run analysis on a company."""
    logger = get_logger("main")
    logger.info(f"Starting analysis for company: {args.company}")
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return 1
    
    # Adjust configuration based on mode
    if args.mode == "quick":
        config["max_iterations"] = 1
        config["max_event_iterations"] = 1
        config["forensic_analysis"] = config.get("forensic_analysis", {})
        config["forensic_analysis"]["max_workers"] = config["forensic_analysis"].get("max_workers", 3)
        logger.info("Running in quick mode with reduced iterations")
    
    # Run the workflow
    try:
        result = create_and_run_workflow(
            company=args.company,
            industry=args.industry,
            config_path=args.config
        )
        
        if result.get("error"):
            logger.error(f"Analysis failed: {result['error']}")
            return 1
        
        # Output report
        if result.get("final_report"):
            output_file = args.output
            if not output_file:
                sanitized_company = args.company.replace(" ", "_")
                output_file = f"{sanitized_company}_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
            
            with open(output_file, 'w') as f:
                f.write(result["final_report"])
            
            logger.info(f"Report saved to: {output_file}")
            
            # Print summary
            print(f"\nAnalysis completed for: {args.company}")
            print(f"Report saved to: {output_file}")
            print(f"Found {len(result.get('top_events', []))} significant events")
            print(f"Identified {len(result.get('analysis_results', {}).get('red_flags', []))} red flags")
        else:
            logger.error("No report generated")
            return 1
        
        return 0
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return 1


async def run_rag_add(args):
    """Add a document to the RAG system."""
    logger = get_logger("main")
    logger.info(f"Adding document to RAG: {args.file}")
    
    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return 1
    
    # Parse topics
    topics = None
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",")]
    
    # Create workflow
    workflow = ForensicWorkflow({})
    
    # Initialize RAG agent
    init_result = await workflow.rag_agent.run({"command": "initialize"})
    if not init_result.get("initialized", False):
        logger.error(f"Failed to initialize RAG agent: {init_result.get('error', 'Unknown error')}")
        return 1
    
    # Add document
    add_result = await workflow.rag_agent.run({
        "command": "add_document",
        "pdf_path": args.file,
        "topics": topics
    })
    
    if not add_result.get("document_added", False):
        logger.error(f"Failed to add document: {add_result.get('error', 'Unknown error')}")
        return 1
    
    # Save vector store
    save_result = await workflow.rag_agent.run({
        "command": "save",
        "directory": "vector_store"
    })
    
    if not save_result.get("saved", False):
        logger.warning(f"Failed to save vector store: {save_result.get('error', 'Unknown error')}")
    
    logger.info(f"Successfully added document: {os.path.basename(args.file)}")
    print(f"Successfully added document: {os.path.basename(args.file)}")
    if topics:
        print(f"Topics: {', '.join(topics)}")
    
    return 0


async def run_rag_query(args):
    """Query the RAG system."""
    logger = get_logger("main")
    logger.info(f"Querying RAG system: {args.query}")
    
    # Parse topics
    filter_topics = None
    if args.topics:
        filter_topics = [t.strip() for t in args.topics.split(",")]
    
    # Create workflow
    workflow = ForensicWorkflow({})
    
    # Initialize RAG agent and load vector store
    init_result = await workflow.rag_agent.run({
        "command": "initialize",
        "vector_store_dir": "vector_store"
    })
    
    if not init_result.get("initialized", False):
        logger.error(f"Failed to initialize RAG agent: {init_result.get('error', 'Unknown error')}")
        return 1
    
    # Execute query
    query_result = await workflow.rag_agent.run({
        "command": "query",
        "query": args.query,
        "filter_topics": filter_topics
    })
    
    if query_result.get("rag_status") != "RESPONSE_READY":
        logger.error(f"Query failed: {query_result.get('error', 'Unknown error')}")
        return 1
    
    # Print response
    print("\n" + "="*80)
    print(f"QUERY: {args.query}")
    if filter_topics:
        print(f"TOPICS: {', '.join(filter_topics)}")
    print("="*80 + "\n")
    
    print(query_result.get("response", "No response generated"))
    
    print("\n" + "="*80)
    print("SOURCES:")
    print("="*80)
    
    for i, result in enumerate(query_result.get("retrieval_results", {}).get("results", [])):
        print(f"\n[{i+1}] Source: {result.get('metadata', {}).get('source', 'Unknown')}, "
              f"Page: {result.get('metadata', {}).get('page', 'Unknown')}")
        print("-"*80)
        text = result.get("text", "")
        if len(text) > 300:
            text = text[:300] + "..."
        print(text)
    
    return 0


async def start_server(args):
    """Start the API server."""
    logger = get_logger("main")
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    try:
        import uvicorn
        from fastapi_app import app
        
        print(f"Starting Financial Forensic Analysis API server on http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        
        await uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except ImportError:
        logger.error("FastAPI and uvicorn are required to run the server")
        print("Error: FastAPI and uvicorn are required to run the server")
        print("Please install with: pip install fastapi uvicorn")
        return 1
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return 1


def print_version():
    """Print version information."""
    print("Financial Forensic Analysis System v1.0.0")
    print("Copyright (c) 2025 Meta")
    return 0


async def main():
    """Main entry point."""
    # Set up logging
    setup_logging("forensic_system")
    logger = get_logger("main")
    
    # Parse arguments
    args = parse_args()
    
    # Initialize providers
    init_llm_provider({})
    init_prompt_manager()
    
    # Execute command
    if args.command == "analyze":
        return await run_analyze(args)
    elif args.command == "rag":
        if args.rag_command == "add":
            return await run_rag_add(args)
        elif args.rag_command == "query":
            return await run_rag_query(args)
        else:
            logger.error(f"Unknown RAG command: {args.rag_command}")
            return 1
    elif args.command == "server":
        return await start_server(args)
    elif args.command == "version":
        return print_version()
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Unhandled exception: {e}")
        sys.exit(1)