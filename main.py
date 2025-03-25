import os
import sys
import json
import argparse
import asyncio
import yaml
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
from dotenv import load_dotenv

from utils.logging import setup_logging, get_logger
from utils.llm_provider import init_llm_provider
from utils.prompt_manager import init_prompt_manager
from langgraph_workflow import EnhancedForensicWorkflow, create_and_run_workflow

# Load environment variables
load_dotenv()

# Check required API keys
required_env_vars = [
    ("GOOGLE_API_KEY", "Google API key for Gemini"),
    ("MISTRAL_API_KEY", "Mistral API key for OCR")
]

def check_environment():
    """Check required environment variables."""
    missing_vars = []
    for var_name, description in required_env_vars:
        if not os.environ.get(var_name):
            missing_vars.append(f"{var_name} ({description})")
            
    if missing_vars:
        print("ERROR: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these environment variables in a .env file or in your environment.")
        print("Example .env file format:")
        print("GOOGLE_API_KEY=your_google_api_key_here")
        print("MISTRAL_API_KEY=your_mistral_api_key_here")
        return False
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial Forensic Analysis System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a company")
    analyze_parser.add_argument("--company", required=True, help="Name of the company to analyze")
    analyze_parser.add_argument("--industry", help="Industry of the company")
    analyze_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    analyze_parser.add_argument("--output", help="Output file for report")
    analyze_parser.add_argument("--mode", choices=["full", "quick"], default="full", 
                               help="Analysis mode (full or quick)")
    analyze_parser.add_argument("--enable-rag", dest="enable_rag", action="store_true", 
                               help="Enable RAG for analysis")
    analyze_parser.add_argument("--disable-rag", dest="enable_rag", action="store_false", 
                               help="Disable RAG for analysis")
    analyze_parser.add_argument("--vector-store", dest="vector_store_dir", default="vector_store",
                               help="Vector store directory for RAG")
    analyze_parser.set_defaults(enable_rag=True)
    
    # RAG commands
    rag_parser = subparsers.add_parser("rag", help="RAG operations")
    rag_subparsers = rag_parser.add_subparsers(dest="rag_command", help="RAG command to execute")
    
    # RAG add command
    rag_add_parser = rag_subparsers.add_parser("add", help="Add a document to RAG")
    rag_add_parser.add_argument("--file", required=True, help="Path to PDF file")
    rag_add_parser.add_argument("--topics", help="Comma-separated list of topics")
    rag_add_parser.add_argument("--vector-store", default="vector_store", help="Vector store directory")
    
    # RAG query command
    rag_query_parser = rag_subparsers.add_parser("query", help="Query RAG system")
    rag_query_parser.add_argument("--query", required=True, help="Query to execute")
    rag_query_parser.add_argument("--topics", help="Comma-separated list of topics to filter by")
    rag_query_parser.add_argument("--vector-store", default="vector_store", help="Vector store directory")
    
    # RAG list command
    rag_list_parser = rag_subparsers.add_parser("list", help="List documents in RAG system")
    rag_list_parser.add_argument("--vector-store", default="vector_store", help="Vector store directory")
    
    # RAG topics command
    rag_topics_parser = rag_subparsers.add_parser("topics", help="List topics in RAG system")
    rag_topics_parser.add_argument("--vector-store", default="vector_store", help="Vector store directory")
    
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
        config["workflow"] = config.get("workflow", {})
        config["workflow"]["max_parallel_agents"] = 2
        config["workflow"]["analyst_pool_size"] = 2
        config["workflow"]["require_plan_approval"] = False
        logger.info("Running in quick mode with simplified workflow")
    
    # Configure RAG settings
    logger.info(f"RAG enabled: {args.enable_rag}, vector store directory: {args.vector_store_dir}")
    
    # Create initial state with RAG configuration
    initial_state = {
        "company": args.company,
        "industry": args.industry,
        "enable_rag": args.enable_rag,
        "vector_store_dir": args.vector_store_dir,
        "rag_initialized": False
    }
    
    # Run the workflow
    try:
        result = create_and_run_workflow(
            company=args.company,
            industry=args.industry,
            config_path=args.config,
            initial_state=initial_state
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
            
            # Get final report content
            report_content = result["final_report"]
            
            # Add RAG insights if enabled
            if args.enable_rag and "analysis_results" in result and "rag_insights" in result["analysis_results"]:
                rag_insights = result["analysis_results"]["rag_insights"]
                
                # Format RAG insights section
                rag_section = "\n\n## Document-Based Insights\n\n"
                rag_section += rag_insights.get("response", "No document-based insights available.")
                
                if rag_insights.get("sources"):
                    rag_section += "\n\n### Sources\n\n"
                    for i, source in enumerate(rag_insights["sources"], 1):
                        rag_section += f"{i}. **{source.get('source', 'Unknown')}** (Page: {source.get('page', 'N/A')})\n"
                
                # Append RAG insights to report
                report_content += rag_section
            
            # Write report to file
            with open(output_file, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Report saved to: {output_file}")
            
            # Print summary
            print(f"\nAnalysis completed for: {args.company}")
            print(f"Report saved to: {output_file}")
            print(f"Found {len(result.get('top_events', []))} significant events")
            print(f"Identified {len(result.get('analysis_results', {}).get('red_flags', []))} red flags")
            
            # Print RAG summary if enabled
            if args.enable_rag:
                rag_status = "Used" if "rag_insights" in result.get("analysis_results", {}) else "Not used"
                print(f"Document-based insights (RAG): {rag_status}")
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
    
    # Create RAG agent
    from agents.rag_agent import RAGAgent
    rag_agent = RAGAgent({})
    
    # Initialize RAG agent with specified vector store
    vector_store_dir = args.vector_store
    logger.info(f"Using vector store directory: {vector_store_dir}")
    
    init_result = await rag_agent.run({
        "command": "initialize",
        "vector_store_dir": vector_store_dir
    })
    
    if not init_result.get("initialized", False):
        logger.error(f"Failed to initialize RAG agent: {init_result.get('error', 'Unknown error')}")
        return 1
    
    # Add document
    add_result = await rag_agent.run({
        "command": "add_document",
        "pdf_path": args.file,
        "topics": topics
    })
    
    if not add_result.get("document_added", False):
        logger.error(f"Failed to add document: {add_result.get('error', 'Unknown error')}")
        return 1
    
    # Save vector store
    save_result = await rag_agent.run({
        "command": "save",
        "directory": vector_store_dir
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
    
    # Create RAG agent
    from agents.rag_agent import RAGAgent
    rag_agent = RAGAgent({})
    
    # Use specified vector store
    vector_store_dir = args.vector_store
    logger.info(f"Using vector store directory: {vector_store_dir}")
    
    # Initialize RAG agent and load vector store
    init_result = await rag_agent.run({
        "command": "initialize",
        "vector_store_dir": vector_store_dir
    })
    
    if not init_result.get("initialized", False):
        logger.error(f"Failed to initialize RAG agent: {init_result.get('error', 'Unknown error')}")
        return 1
    
    # Execute query
    query_result = await rag_agent.run({
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
        from app import app
        
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
    print("Copyright (c) 2025 riskX")
    return 0


async def main():
    """Main entry point."""
    # Check environment variables first
    if not check_environment():
        return 1
        
    # Set up logging
    setup_logging("forensic_system")
    logger = get_logger("main")
    
    # Parse arguments
    args = parse_args()
    
    # Initialize providers
    try:
        # Load configuration
        config_path = args.config if hasattr(args, 'config') else "config.yaml"
        config = {}
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}. Using default configuration.")
        
        # Initialize LLM provider with loaded config
        provider_config = {
            "default_provider": "google",
            "providers": {
                "google": {
                    "api_key": os.environ.get("GOOGLE_API_KEY"),
                    "default_model": "gemini-1.5-pro"
                },
                "mistralai": {
                    "api_key": os.environ.get("MISTRAL_API_KEY"),
                    "default_model": "mistral-large-latest"
                }
            }
        }
        
        init_llm_provider(provider_config)
        init_prompt_manager()
        
    except Exception as e:
        logger.error(f"Error initializing providers: {e}")
        return 1
    
    # Add new functions for additional RAG commands
async def run_rag_list(args):
    """List documents in the RAG system."""
    logger = get_logger("main")
    logger.info(f"Listing documents in vector store: {args.vector_store}")
    
    # Create RAG agent
    from agents.rag_agent import RAGAgent
    rag_agent = RAGAgent({})
    
    # Initialize RAG agent with specified vector store
    init_result = await rag_agent.run({
        "command": "initialize",
        "vector_store_dir": args.vector_store
    })
    
    if not init_result.get("initialized", False):
        logger.error(f"Failed to initialize RAG agent: {init_result.get('error', 'Unknown error')}")
        return 1
    
    # Get vector store info
    info_result = await rag_agent.run({"command": "info"})
    vector_store_info = info_result.get("vector_store_info", {})
    
    if not vector_store_info:
        logger.error("Failed to get vector store information")
        return 1
    
    # Print document information
    print("\n" + "="*80)
    print(f"DOCUMENTS IN VECTOR STORE: {args.vector_store}")
    print("="*80)
    
    document_collection = vector_store_info.get("document_collection", {})
    if not document_collection:
        print("\nNo documents found in the vector store.")
        return 0
    
    print(f"\nFound {len(document_collection)} document(s):\n")
    
    for doc_id, metadata in document_collection.items():
        print(f"Document: {doc_id}")
        print(f"  Path: {metadata.get('path', 'Unknown')}")
        print(f"  Added: {metadata.get('added_at', 'Unknown')}")
        print(f"  Size: {metadata.get('size_bytes', 0)} bytes")
        print(f"  Topics: {', '.join(metadata.get('topics', ['unclassified']))}")
        print()
    
    return 0

async def run_rag_topics(args):
    """List topics in the RAG system."""
    logger = get_logger("main")
    logger.info(f"Listing topics in vector store: {args.vector_store}")
    
    # Create RAG agent
    from agents.rag_agent import RAGAgent
    rag_agent = RAGAgent({})
    
    # Initialize RAG agent with specified vector store
    init_result = await rag_agent.run({
        "command": "initialize",
        "vector_store_dir": args.vector_store
    })
    
    if not init_result.get("initialized", False):
        logger.error(f"Failed to initialize RAG agent: {init_result.get('error', 'Unknown error')}")
        return 1
    
    # List topics
    topics_result = await rag_agent.run({"command": "list_topics"})
    
    if not topics_result.get("topics_result", {}).get("success", False):
        logger.error("Failed to list topics")
        return 1
    
    # Print topic information
    print("\n" + "="*80)
    print(f"TOPICS IN VECTOR STORE: {args.vector_store}")
    print("="*80)
    
    topics = topics_result.get("topics_result", {}).get("topics", {})
    if not topics:
        print("\nNo topics found in the vector store.")
        return 0
    
    print(f"\nFound {len(topics)} topic(s):\n")
    
    for topic_name, topic_data in topics.items():
        print(f"Topic: {topic_name}")
        print(f"  Documents: {topic_data.get('document_count', 0)}")
        if topic_data.get("documents"):
            for doc in topic_data.get("documents", []):
                print(f"    - {doc}")
        print()
    
    return 0

# Execute command
    if args.command == "analyze":
        return await run_analyze(args)
    elif args.command == "rag":
        if args.rag_command == "add":
            return await run_rag_add(args)
        elif args.rag_command == "query":
            return await run_rag_query(args)
        elif args.rag_command == "list":
            return await run_rag_list(args)
        elif args.rag_command == "topics":
            return await run_rag_topics(args)
        else:
            logger.error(f"Unknown RAG command: {args.rag_command}")
            return 1
    elif args.command == "server":
        return await start_server(args)
    elif args.command == "version":
        return print_version()
    else:
        print("Please specify a command. Use --help for more information.")
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