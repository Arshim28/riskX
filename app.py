import os
import json
import asyncio
import uvicorn
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile, Form, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from utils.logging import setup_logging, get_logger
from utils.llm_provider import init_llm_provider
from utils.prompt_manager import init_prompt_manager
from langgraph_workflow import EnhancedForensicWorkflow, WorkflowState

# Initialize FastAPI app
app = FastAPI(
    title="Financial Forensic Analysis API",
    description="API for financial forensic analysis of companies",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logging
setup_logging("forensic_api")
logger = get_logger("forensic_api")

# Load configuration
config_path = os.getenv("CONFIG_PATH", "config.json")
config = {}
if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")

# Initialize LLM provider and prompt manager
init_llm_provider(config)
init_prompt_manager()

# Create workflow instance
workflow = EnhancedForensicWorkflow(config)

# Create storage directory for reports and uploads
os.makedirs("reports", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Dictionary to track active workflows
active_workflows = {}
workflow_results = {}


# Input models
class CompanyInput(BaseModel):
    company: str
    industry: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    enable_rag: Optional[bool] = True
    vector_store_dir: Optional[str] = "vector_store"
    

class QueryInput(BaseModel):
    query: str
    company: str
    filter_topics: Optional[List[str]] = None
    session_id: Optional[str] = None


class DocumentCategoryInput(BaseModel):
    topics: List[str]


class FeedbackInput(BaseModel):
    feedback: str
    section: Optional[str] = None


# Result models
class WorkflowStatus(BaseModel):
    workflow_id: str
    company: str
    status: str
    progress: int = 0
    current_phase: Optional[str] = None
    active_agents: List[str] = []
    completed_agents: List[str] = []
    error: Optional[str] = None
    started_at: str
    updated_at: str


class DocumentInfo(BaseModel):
    document_id: str
    name: str
    size: int
    upload_date: str
    topics: List[str]


# Background task to run workflow
async def run_workflow_task(workflow_id: str, company: str, industry: Optional[str], config_overrides: Optional[Dict[str, Any]]):
    try:
        logger.info(f"Starting workflow {workflow_id} for company: {company}")
        
        # Prepare initial state
        initial_state = {
            "company": company,
            "industry": industry,
            "workflow_id": workflow_id,
            "enable_rag": config_overrides.get("enable_rag", True) if config_overrides else True,
            "vector_store_dir": config_overrides.get("vector_store_dir", "vector_store") if config_overrides else "vector_store",
            "rag_initialized": False
        }
        
        # Apply any config overrides
        workflow_config = config.copy()
        if config_overrides:
            for key, value in config_overrides.items():
                workflow_config[key] = value
        
        # Update workflow status
        active_workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "company": company,
            "status": "RUNNING",
            "progress": 0,
            "current_phase": "Initialization",
            "active_agents": ["meta_agent"],
            "completed_agents": [],
            "error": None,
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Run workflow
        result = await workflow.run(initial_state)
        
        # Store result
        workflow_results[workflow_id] = result
        
        # Update workflow status
        active_workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "company": company,
            "status": "COMPLETED" if not result.get("error") else "ERROR",
            "progress": 100,
            "current_phase": "Completed",
            "active_agents": [],
            "completed_agents": list(workflow.agents.keys()),
            "error": result.get("error"),
            "started_at": active_workflows[workflow_id]["started_at"],
            "updated_at": datetime.now().isoformat()
        }
        
        # Save report to file if generated
        if result.get("final_report"):
            report_file = f"reports/{workflow_id}_report.md"
            with open(report_file, 'w') as f:
                f.write(result["final_report"])
            logger.info(f"Saved report to {report_file}")
            
        logger.info(f"Workflow {workflow_id} completed for company: {company}")
        
    except Exception as e:
        logger.error(f"Error in workflow {workflow_id}: {e}")
        
        # Update workflow status
        if workflow_id in active_workflows:
            active_workflows[workflow_id].update({
                "status": "ERROR",
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            })


# Routes for workflow management
@app.post("/workflow/start", response_model=WorkflowStatus)
async def start_workflow(
    background_tasks: BackgroundTasks,
    input_data: CompanyInput
):
    """Start a new analysis workflow for a company."""
    workflow_id = f"{input_data.company.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Prepare config overrides including RAG settings
    config_overrides = input_data.config_overrides or {}
    config_overrides["enable_rag"] = input_data.enable_rag
    config_overrides["vector_store_dir"] = input_data.vector_store_dir
    
    # Start workflow as background task
    background_tasks.add_task(
        run_workflow_task,
        workflow_id=workflow_id,
        company=input_data.company,
        industry=input_data.industry,
        config_overrides=config_overrides
    )
    
    # Return initial status
    status = WorkflowStatus(
        workflow_id=workflow_id,
        company=input_data.company,
        status="STARTING",
        progress=0,
        current_phase="Initialization",
        active_agents=[],
        completed_agents=[],
        error=None,
        started_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    # Store in active workflows
    active_workflows[workflow_id] = status.dict()
    
    return status


@app.get("/workflow/{workflow_id}/status", response_model=WorkflowStatus)
async def get_workflow_status(workflow_id: str):
    """Get the status of a workflow."""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    return active_workflows[workflow_id]


@app.get("/workflow/{workflow_id}/report")
async def get_workflow_report(workflow_id: str, include_rag_insights: bool = True):
    """Get the final report from a workflow."""
    # Check if workflow exists
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    # Check if workflow is completed
    status = active_workflows[workflow_id]
    if status["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Workflow {workflow_id} is not completed")
    
    # Check if report exists
    report_file = f"reports/{workflow_id}_report.md"
    if not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail=f"Report for workflow {workflow_id} not found")
    
    # Add RAG insights if requested and available
    if include_rag_insights and workflow_id in workflow_results:
        result = workflow_results[workflow_id]
        if "analysis_results" in result and "rag_insights" in result["analysis_results"]:
            # Append RAG insights to the report
            with open(report_file, 'r') as f:
                report_content = f.read()
            
            rag_insights = result["analysis_results"]["rag_insights"]
            
            # Format RAG insights section
            rag_section = "\n\n## Document-Based Insights\n\n"
            rag_section += rag_insights.get("response", "No document-based insights available.")
            
            if rag_insights.get("sources"):
                rag_section += "\n\n### Sources\n\n"
                for i, source in enumerate(rag_insights["sources"], 1):
                    rag_section += f"{i}. **{source.get('source', 'Unknown')}** (Page: {source.get('page', 'N/A')})\n"
            
            # Write updated report
            with open(report_file, 'w') as f:
                f.write(report_content + rag_section)
    
    # Return report file
    return FileResponse(report_file, media_type="text/markdown")


@app.get("/workflow/{workflow_id}/executive-briefing")
async def get_executive_briefing(workflow_id: str):
    """Get the executive briefing from a workflow."""
    # Check if workflow exists and completed
    if workflow_id not in workflow_results:
        raise HTTPException(status_code=404, detail=f"Results for workflow {workflow_id} not found")
    
    # Check if executive briefing exists
    result = workflow_results[workflow_id]
    if not result.get("executive_briefing"):
        raise HTTPException(status_code=404, detail=f"Executive briefing for workflow {workflow_id} not found")
    
    return {"executive_briefing": result["executive_briefing"]}


@app.get("/workflows", response_model=List[WorkflowStatus])
async def list_workflows(
    status: Optional[str] = None,
    company: Optional[str] = None
):
    """List all workflows with optional filtering."""
    workflows = list(active_workflows.values())
    
    # Apply filters
    if status:
        workflows = [w for w in workflows if w["status"] == status]
    
    if company:
        workflows = [w for w in workflows if w["company"] == company]
    
    return workflows


# Routes for RAG functionality
@app.post("/documents/upload")
async def upload_document(
    document: UploadFile = File(...),
    topics: Optional[str] = Form(None)
):
    """Upload a document for RAG processing."""
    # Save uploaded file
    file_path = f"uploads/{document.filename}"
    with open(file_path, "wb") as f:
        f.write(await document.read())
    
    # Parse topics
    topic_list = []
    if topics:
        try:
            topic_list = json.loads(topics)
        except:
            # Try comma-separated string
            topic_list = [t.strip() for t in topics.split(",")]
    
    # Ensure RAG agent is initialized first
    init_result = await workflow.rag_agent.run({
        "command": "initialize",
        "vector_store_dir": "vector_store"
    })
    
    if not init_result.get("initialized", False):
        raise HTTPException(status_code=500, detail="Failed to initialize RAG system")
    
    # Add document to RAG agent
    result = await workflow.rag_agent.run({
        "command": "add_document",
        "pdf_path": file_path,
        "topics": topic_list
    })
    
    if not result.get("document_added", False):
        raise HTTPException(status_code=400, detail=f"Failed to add document: {result.get('error', 'Unknown error')}")
    
    # Return success response
    return {
        "success": True,
        "document_id": os.path.basename(file_path),
        "file_name": document.filename,
        "topics": topic_list
    }


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all documents available in the RAG system."""
    # Get vector store info from RAG agent
    result = await workflow.rag_agent.run({"command": "info"})
    
    if not result.get("vector_store_info"):
        return []
    
    # Extract document information
    info = result["vector_store_info"]
    documents = []
    
    if "document_collection" in info:
        for doc_id, metadata in info["document_collection"].items():
            documents.append(DocumentInfo(
                document_id=doc_id,
                name=doc_id,
                size=metadata.get("size_bytes", 0),
                upload_date=metadata.get("added_at", datetime.now().isoformat()),
                topics=metadata.get("topics", ["unclassified"])
            ))
    
    return documents


@app.get("/documents/topics")
async def list_topics():
    """List all topics available in the RAG system."""
    # Get topic information from RAG agent
    result = await workflow.rag_agent.run({"command": "list_topics"})
    
    if not result.get("topics_result", {}).get("success", False):
        return {"topics": []}
    
    return {"topics": result["topics_result"]["topics"]}


@app.post("/documents/{document_id}/categorize")
async def categorize_document(
    document_id: str,
    categories: DocumentCategoryInput
):
    """Update the topics/categories for a document."""
    # Ensure RAG agent is initialized
    init_result = await workflow.rag_agent.run({
        "command": "initialize",
        "vector_store_dir": "vector_store"
    })
    
    if not init_result.get("initialized", False):
        raise HTTPException(status_code=500, detail="Failed to initialize RAG system")
    
    # Get current document information
    result = await workflow.rag_agent.run({"command": "info"})
    
    if not result.get("vector_store_info"):
        raise HTTPException(status_code=404, detail="RAG system not initialized")
    
    # Check if document exists
    info = result["vector_store_info"]
    if "document_collection" not in info or document_id not in info["document_collection"]:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    # Update document topics
    document_path = f"uploads/{document_id}"
    update_result = await workflow.rag_agent.run({
        "command": "add_document",
        "pdf_path": document_path,
        "topics": categories.topics
    })
    
    if not update_result.get("document_added", False):
        raise HTTPException(status_code=400, detail=f"Failed to update document: {update_result.get('error', 'Unknown error')}")
    
    return {
        "success": True,
        "document_id": document_id,
        "topics": categories.topics
    }


@app.post("/query")
async def query_documents(query_input: QueryInput):
    """Query the RAG system."""
    # Initialize RAG agent if needed
    init_result = await workflow.rag_agent.run({
        "command": "initialize",
        "vector_store_dir": "vector_store"
    })
    
    if not init_result.get("initialized", False):
        raise HTTPException(status_code=500, detail="Failed to initialize RAG system")
    
    # Process query through RAG agent
    result = await workflow.rag_agent.run({
        "command": "query",
        "query": query_input.query,
        "session_id": query_input.session_id,
        "filter_topics": query_input.filter_topics,
        "company": query_input.company
    })
    
    if not result.get("rag_status") == "RESPONSE_READY":
        raise HTTPException(status_code=400, detail=f"Query failed: {result.get('error', 'Unknown error')}")
    
    return {
        "query": query_input.query,
        "response": result.get("response"),
        "sources": [
            {
                "text": r.get("text", ""),
                "source": r.get("metadata", {}).get("source", "Unknown"),
                "page": r.get("metadata", {}).get("page", "Unknown"),
                "score": r.get("score", 0)
            }
            for r in result.get("retrieval_results", {}).get("results", [])
        ]
    }


@app.post("/documents/auto-categorize")
async def auto_categorize_documents():
    """Automatically categorize documents in the RAG system."""
    result = await workflow.rag_agent.run({"command": "categorize_documents"})
    
    if not result.get("categorization_result", {}).get("success", False):
        raise HTTPException(status_code=400, detail=f"Categorization failed: {result.get('error', 'Unknown error')}")
    
    return result["categorization_result"]


@app.post("/topics/{topic}/report")
async def generate_topic_report(topic: str):
    """Generate a report for a specific topic."""
    result = await workflow.rag_agent.run({
        "command": "generate_topic_report",
        "topic": topic
    })
    
    if not result.get("topic_report", {}).get("success", False):
        raise HTTPException(status_code=400, detail=f"Topic report generation failed: {result.get('error', 'Unknown error')}")
    
    return result["topic_report"]


# Routes for report feedback and improvement
@app.post("/workflow/{workflow_id}/feedback")
async def submit_feedback(
    workflow_id: str,
    feedback_data: FeedbackInput
):
    """Submit feedback for a workflow report or specific section."""
    # Check if workflow exists and completed
    if workflow_id not in workflow_results:
        raise HTTPException(status_code=404, detail=f"Results for workflow {workflow_id} not found")
    
    result = workflow_results[workflow_id]
    company = result.get("company", "Unknown")
    
    # Apply feedback to improve report
    if feedback_data.section:
        # Improve specific section
        if feedback_data.section not in result.get("report_sections", {}):
            raise HTTPException(status_code=404, detail=f"Section {feedback_data.section} not found")
        
        section_content = result["report_sections"][feedback_data.section]
        
        # Use writer agent to revise the section
        revised_content = await workflow.writer_agent.revise_section(
            company, 
            feedback_data.section, 
            section_content, 
            feedback_data.feedback
        )
        
        # Update result
        result["report_sections"][feedback_data.section] = revised_content
        
        # Regenerate full report
        section_order = ["header", "executive_summary", "key_events", "other_events", 
                         "pattern_recognition", "recommendations"]
        
        full_report_parts = []
        for section_name in section_order:
            if section_name in result["report_sections"]:
                full_report_parts.append(result["report_sections"][section_name])
                
        result["final_report"] = "\n".join(full_report_parts)
        
        # Save updated report
        report_file = f"reports/{workflow_id}_report.md"
        with open(report_file, 'w') as f:
            f.write(result["final_report"])
        
        return {
            "success": True,
            "message": f"Updated section {feedback_data.section} based on feedback",
            "updated_section": feedback_data.section
        }
        
    else:
        # General feedback - log but don't modify report
        return {
            "success": True,
            "message": "Feedback received",
            "feedback_id": f"feedback_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)