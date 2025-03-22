import streamlit as st
import requests
import json
import time
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
REFRESH_INTERVAL = 5  # seconds


# Initialize session state
def init_session_state():
    if "active_workflows" not in st.session_state:
        st.session_state.active_workflows = {}
    if "current_workflow_id" not in st.session_state:
        st.session_state.current_workflow_id = None
    if "company_name" not in st.session_state:
        st.session_state.company_name = ""
    if "industry" not in st.session_state:
        st.session_state.industry = ""
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "topics" not in st.session_state:
        st.session_state.topics = []
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "Start Analysis"


# API functions
def start_workflow(company, industry=None, config_overrides=None):
    try:
        response = requests.post(
            f"{API_URL}/workflow/start",
            json={
                "company": company,
                "industry": industry,
                "config_overrides": config_overrides
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error starting workflow: {e}")
        return None


def get_workflow_status(workflow_id):
    try:
        response = requests.get(f"{API_URL}/workflow/{workflow_id}/status")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting workflow status: {e}")
        return None


def get_all_workflows():
    try:
        response = requests.get(f"{API_URL}/workflows")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting workflows: {e}")
        return []


def get_workflow_report(workflow_id):
    try:
        response = requests.get(f"{API_URL}/workflow/{workflow_id}/report")
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        st.error(f"Error getting report: {e}")
        return None


def get_executive_briefing(workflow_id):
    try:
        response = requests.get(f"{API_URL}/workflow/{workflow_id}/executive-briefing")
        if response.status_code == 200:
            return response.json()["executive_briefing"]
        else:
            return None
    except Exception as e:
        st.error(f"Error getting executive briefing: {e}")
        return None


def upload_document(file, topics=None):
    try:
        files = {"document": file}
        data = {}
        if topics:
            data["topics"] = json.dumps(topics)
        
        response = requests.post(f"{API_URL}/documents/upload", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return None


def list_documents():
    try:
        response = requests.get(f"{API_URL}/documents")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error listing documents: {e}")
        return []


def list_topics():
    try:
        response = requests.get(f"{API_URL}/documents/topics")
        response.raise_for_status()
        return response.json()["topics"]
    except Exception as e:
        st.error(f"Error listing topics: {e}")
        return {}


def query_documents(query, company, filter_topics=None, session_id=None):
    try:
        payload = {
            "query": query,
            "company": company,
            "filter_topics": filter_topics,
            "session_id": session_id
        }
        response = requests.post(f"{API_URL}/query", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying documents: {e}")
        return None


def auto_categorize_documents():
    try:
        response = requests.post(f"{API_URL}/documents/auto-categorize")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error auto-categorizing documents: {e}")
        return None


def categorize_document(document_id, topics):
    try:
        response = requests.post(
            f"{API_URL}/documents/{document_id}/categorize",
            json={"topics": topics}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error categorizing document: {e}")
        return None


def submit_feedback(workflow_id, feedback, section=None):
    try:
        payload = {
            "feedback": feedback,
            "section": section
        }
        response = requests.post(f"{API_URL}/workflow/{workflow_id}/feedback", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error submitting feedback: {e}")
        return None


def generate_topic_report(topic):
    try:
        response = requests.post(f"{API_URL}/topics/{topic}/report")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error generating topic report: {e}")
        return None


# UI Components
def render_start_analysis():
    st.header("Start Financial Forensic Analysis")
    
    with st.form("analysis_form"):
        company = st.text_input("Company Name", value=st.session_state.company_name)
        industry = st.text_input("Industry (Optional)", value=st.session_state.industry)
        
        advanced_options = st.expander("Advanced Configuration")
        with advanced_options:
            parallel_execution = st.checkbox("Enable Parallel Execution", value=True)
            max_workers = st.slider("Max Workers", min_value=1, max_value=10, value=5)
            max_events = st.slider("Max Events to Analyze", min_value=1, max_value=20, value=6)
        
        submit_button = st.form_submit_button("Start Analysis")
        
        if submit_button:
            if not company:
                st.error("Company name is required")
            else:
                config_overrides = {
                    "meta_agent": {
                        "parallel_execution": parallel_execution,
                        "max_parallel_agents": 3
                    },
                    "forensic_analysis": {
                        "max_workers": max_workers
                    },
                    "writer": {
                        "enable_iterative_improvement": True,
                        "max_concurrent_sections": 3
                    }
                }
                
                st.session_state.company_name = company
                st.session_state.industry = industry
                
                with st.spinner("Starting analysis..."):
                    result = start_workflow(company, industry, config_overrides)
                    
                    if result:
                        workflow_id = result["workflow_id"]
                        st.session_state.current_workflow_id = workflow_id
                        st.session_state.active_workflows[workflow_id] = result
                        st.success(f"Analysis started for {company}!")
                        
                        # Automatically switch to the status tab
                        st.session_state.current_tab = "Workflow Status"
                        st.experimental_rerun()


def render_workflow_status():
    st.header("Workflow Status")
    
    # Refresh all workflow statuses
    all_workflows = get_all_workflows()
    for workflow in all_workflows:
        workflow_id = workflow["workflow_id"]
        st.session_state.active_workflows[workflow_id] = workflow
    
    # Workflow selector
    workflow_ids = list(st.session_state.active_workflows.keys())
    
    if not workflow_ids:
        st.info("No active workflows. Start a new analysis on the 'Start Analysis' tab.")
        return
    
    # Default to the current workflow or the most recent one
    default_index = 0
    if st.session_state.current_workflow_id in workflow_ids:
        default_index = workflow_ids.index(st.session_state.current_workflow_id)
    
    selected_workflow_id = st.selectbox(
        "Select Workflow", 
        workflow_ids,
        index=default_index,
        format_func=lambda x: f"{st.session_state.active_workflows[x]['company']} ({x.split('_')[-1]})"
    )
    
    st.session_state.current_workflow_id = selected_workflow_id
    workflow = st.session_state.active_workflows[selected_workflow_id]
    
    # Update status if workflow is still running
    if workflow["status"] in ["STARTING", "RUNNING"]:
        latest = get_workflow_status(selected_workflow_id)
        if latest:
            st.session_state.active_workflows[selected_workflow_id] = latest
            workflow = latest
    
    # Display status card
    status_color = {
        "STARTING": "blue",
        "RUNNING": "blue",
        "COMPLETED": "green",
        "ERROR": "red"
    }.get(workflow["status"], "gray")
    
    st.markdown(
        f"""
        <div style="border:1px solid {status_color}; border-radius:5px; padding:15px; margin-bottom:20px;">
            <h3 style="color:{status_color};">{workflow["status"]}</h3>
            <p><strong>Company:</strong> {workflow["company"]}</p>
            <p><strong>Started:</strong> {workflow["started_at"]}</p>
            <p><strong>Updated:</strong> {workflow["updated_at"]}</p>
            <p><strong>Current Phase:</strong> {workflow["current_phase"]}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Progress bar
    st.progress(workflow["progress"] / 100)
    st.write(f"Progress: {workflow['progress']}%")
    
    # Agent status
    st.subheader("Agent Status")
    
    agent_cols = st.columns(3)
    for i, agent in enumerate(["meta_agent", "research_agent", "youtube_agent", "corporate_agent", "analyst_agent", "rag_agent", "writer_agent"]):
        with agent_cols[i % 3]:
            agent_status = "N/A"
            if agent in workflow["active_agents"]:
                agent_status = "RUNNING"
            elif agent in workflow["completed_agents"]:
                agent_status = "COMPLETED"
            
            status_emoji = {
                "RUNNING": "🔄",
                "COMPLETED": "✅",
                "ERROR": "❌",
                "N/A": "⏹️"
            }.get(agent_status, "⏹️")
            
            st.write(f"{status_emoji} {agent.replace('_', ' ').title()}: {agent_status}")
    
    # Error display
    if workflow["error"]:
        st.error(f"Error: {workflow['error']}")
    
    # Auto-refresh if still running
    if workflow["status"] in ["STARTING", "RUNNING"]:
        st.info(f"Status will refresh automatically every {REFRESH_INTERVAL} seconds...")
        time.sleep(REFRESH_INTERVAL)
        st.experimental_rerun()
    
    # View report button if completed
    if workflow["status"] == "COMPLETED":
        if st.button("View Report"):
            st.session_state.current_tab = "View Reports"
            st.experimental_rerun()


def render_view_reports():
    st.header("Analysis Reports")
    
    # Select workflow
    workflow_ids = [wid for wid, w in st.session_state.active_workflows.items() 
                   if w["status"] == "COMPLETED"]
    
    if not workflow_ids:
        st.info("No completed workflows. Reports will appear here when analyses are completed.")
        return
    
    # Default to the current workflow or the most recent one
    default_index = 0
    if st.session_state.current_workflow_id in workflow_ids:
        default_index = workflow_ids.index(st.session_state.current_workflow_id)
    
    selected_workflow_id = st.selectbox(
        "Select Workflow", 
        workflow_ids,
        index=default_index,
        format_func=lambda x: f"{st.session_state.active_workflows[x]['company']} ({x.split('_')[-1]})"
    )
    
    st.session_state.current_workflow_id = selected_workflow_id
    workflow = st.session_state.active_workflows[selected_workflow_id]
    
    # Tabs for different report views
    report_tab, briefing_tab, feedback_tab = st.tabs(["Full Report", "Executive Briefing", "Provide Feedback"])
    
    with report_tab:
        report = get_workflow_report(selected_workflow_id)
        if report:
            st.markdown(report)
            
            # Download button
            company_name = workflow["company"].replace(" ", "_")
            report_date = datetime.now().strftime("%Y%m%d")
            filename = f"{company_name}_forensic_report_{report_date}.md"
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=filename,
                mime="text/markdown"
            )
        else:
            st.info("Report not available or still being generated.")
    
    with briefing_tab:
        briefing = get_executive_briefing(selected_workflow_id)
        if briefing:
            st.markdown(briefing)
        else:
            st.info("Executive briefing not available or still being generated.")
    
    with feedback_tab:
        feedback = st.text_area("Your Feedback", height=150, 
                               placeholder="Provide feedback on the report quality or suggest improvements...")
        section_options = ["All Report", "Executive Summary", "Key Events", "Other Events", 
                          "Pattern Recognition", "Recommendations"]
        section = st.selectbox("Section to Improve", section_options)
        
        section_mapping = {
            "All Report": None,
            "Executive Summary": "executive_summary",
            "Key Events": "key_events",
            "Other Events": "other_events",
            "Pattern Recognition": "pattern_recognition",
            "Recommendations": "recommendations"
        }
        
        if st.button("Submit Feedback"):
            if feedback:
                with st.spinner("Processing feedback..."):
                    result = submit_feedback(
                        selected_workflow_id, 
                        feedback, 
                        section=section_mapping[section]
                    )
                    
                    if result and result.get("success"):
                        st.success("Feedback submitted successfully!")
                        if section != "All Report":
                            st.info("The report has been updated based on your feedback. Refresh the page to see changes.")
                    else:
                        st.error("Failed to submit feedback")
            else:
                st.warning("Please enter feedback before submitting")


def render_document_management():
    st.header("Document Management")
    
    upload_tab, list_tab, categorize_tab = st.tabs(["Upload Documents", "Document Library", "Categorize Documents"])
    
    with upload_tab:
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
        
        topic_input = st.text_input("Topics (comma separated)", 
                                  placeholder="corporate, financial, regulatory")
        
        if uploaded_file is not None:
            if st.button("Upload Document"):
                with st.spinner("Uploading and processing document..."):
                    topics = [t.strip() for t in topic_input.split(",")] if topic_input else None
                    result = upload_document(uploaded_file, topics)
                    
                    if result and result.get("success"):
                        st.success(f"Document uploaded successfully: {result['file_name']}")
                    else:
                        st.error("Failed to upload document")
    
    with list_tab:
        if st.button("Refresh Document List"):
            st.session_state.documents = list_documents()
        
        if not hasattr(st.session_state, "documents") or not st.session_state.documents:
            st.session_state.documents = list_documents()
        
        if st.session_state.documents:
            # Convert to dataframe
            df = pd.DataFrame(st.session_state.documents)
            
            # Format size as KB/MB
            df["size_formatted"] = df["size"].apply(lambda x: f"{x/1024:.1f} KB" if x < 1024*1024 else f"{x/1024/1024:.1f} MB")
            
            # Format topics as comma-joined string
            df["topics_str"] = df["topics"].apply(lambda x: ", ".join(x))
            
            # Display table
            st.dataframe(
                df[["document_id", "name", "size_formatted", "upload_date", "topics_str"]],
                column_config={
                    "document_id": "ID",
                    "name": "Name",
                    "size_formatted": "Size",
                    "upload_date": "Upload Date",
                    "topics_str": "Topics"
                },
                hide_index=True
            )
        else:
            st.info("No documents found. Upload a document first.")
    
    with categorize_tab:
        st.subheader("Auto-Categorize Documents")
        
        if st.button("Run Auto-Categorization"):
            with st.spinner("Analyzing and categorizing documents..."):
                result = auto_categorize_documents()
                
                if result and result.get("success"):
                    st.success(f"Successfully categorized {result.get('categorized_count', 0)} documents")
                    # Refresh document list
                    st.session_state.documents = list_documents()
                else:
                    st.error("Failed to auto-categorize documents")
        
        st.subheader("Manual Categorization")
        
        if not hasattr(st.session_state, "documents") or not st.session_state.documents:
            st.session_state.documents = list_documents()
        
        if st.session_state.documents:
            doc_options = [f"{doc['name']} ({', '.join(doc['topics'])})" for doc in st.session_state.documents]
            doc_indices = {f"{doc['name']} ({', '.join(doc['topics'])})": i for i, doc in enumerate(st.session_state.documents)}
            
            selected_doc = st.selectbox("Select Document", doc_options)
            doc_index = doc_indices[selected_doc]
            doc_id = st.session_state.documents[doc_index]["document_id"]
            
            new_topics = st.text_input("New Topics (comma separated)", 
                                     placeholder="corporate, financial, regulatory")
            
            if st.button("Update Categories"):
                if new_topics:
                    with st.spinner("Updating document categories..."):
                        topics_list = [t.strip() for t in new_topics.split(",")]
                        result = categorize_document(doc_id, topics_list)
                        
                        if result and result.get("success"):
                            st.success(f"Document categories updated successfully")
                            # Refresh document list
                            st.session_state.documents = list_documents()
                        else:
                            st.error("Failed to update document categories")
                else:
                    st.warning("Please enter at least one topic")
        else:
            st.info("No documents found. Upload a document first.")


def render_document_query():
    st.header("Query Documents")
    
    # Get available topics for filtering
    if not hasattr(st.session_state, "topics") or not st.session_state.topics:
        st.session_state.topics = list_topics()
    
    # Query interface
    query = st.text_input("Your Question", placeholder="What are the main red flags for this company?")
    
    # Topic filtering
    topic_options = list(st.session_state.topics.keys()) if st.session_state.topics else []
    selected_topics = st.multiselect("Filter by Topics (Optional)", topic_options)
    
    # Company context
    company = st.text_input("Company Context", value=st.session_state.company_name)
    
    # Create a unique session ID if one doesn't exist
    if "query_session_id" not in st.session_state:
        st.session_state.query_session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Submit button
    if st.button("Submit Query"):
        if query and company:
            with st.spinner("Searching for answer..."):
                result = query_documents(
                    query=query,
                    company=company,
                    filter_topics=selected_topics if selected_topics else None,
                    session_id=st.session_state.query_session_id
                )
                
                if result:
                    st.subheader("Answer")
                    st.markdown(result["response"])
                    
                    with st.expander("View Sources"):
                        for i, source in enumerate(result.get("sources", [])):
                            st.markdown(f"**Source {i+1}** (Relevance: {source['score']:.2f})")
                            st.markdown(f"**Document:** {source['source']}, **Page:** {source['page']}")
                            st.markdown(f"```\n{source['text'][:500]}{'...' if len(source['text']) > 500 else ''}\n```")
                else:
                    st.error("Failed to process query")
        else:
            st.warning("Please enter both a query and company context")
    
    # Topic reports
    st.subheader("Topic Reports")
    
    topic_for_report = st.selectbox("Generate report for topic", 
                                   [""] + topic_options,
                                   index=0)
    
    if topic_for_report and st.button("Generate Topic Report"):
        with st.spinner(f"Generating report for topic: {topic_for_report}..."):
            report = generate_topic_report(topic_for_report)
            
            if report and report.get("success"):
                st.subheader(f"Report for topic: {topic_for_report}")
                
                report_data = report.get("report", {})
                
                if "summary" in report_data:
                    st.markdown("### Summary")
                    st.markdown(report_data["summary"])
                
                if "key_points" in report_data:
                    st.markdown("### Key Points")
                    for point in report_data["key_points"]:
                        st.markdown(f"- {point}")
                
                if "documents" in report_data:
                    st.markdown("### Key Documents")
                    for doc in report_data["documents"]:
                        st.markdown(f"- {doc}")
            else:
                st.error("Failed to generate topic report")


def render_visualizations():
    st.header("Forensic Analysis Visualizations")
    
    # Select workflow
    workflow_ids = [wid for wid, w in st.session_state.active_workflows.items() 
                   if w["status"] == "COMPLETED"]
    
    if not workflow_ids:
        st.info("No completed workflows. Visualizations will appear here when analyses are completed.")
        return
    
    # Default to the current workflow or the most recent one
    default_index = 0
    if st.session_state.current_workflow_id in workflow_ids:
        default_index = workflow_ids.index(st.session_state.current_workflow_id)
    
    selected_workflow_id = st.selectbox(
        "Select Workflow", 
        workflow_ids,
        index=default_index,
        format_func=lambda x: f"{st.session_state.active_workflows[x]['company']} ({x.split('_')[-1]})"
    )
    
    st.session_state.current_workflow_id = selected_workflow_id
    workflow = st.session_state.active_workflows[selected_workflow_id]
    
    # Load report and data
    report = get_workflow_report(selected_workflow_id)
    
    # Placeholder for actual result data that would come from the API
    # In a real implementation, this would be retrieved from the backend
    
    # Sample visualization 1: Red Flag Severity
    st.subheader("Red Flag Severity Analysis")
    
    # Sample data - in real implementation, this would come from the API
    red_flags = [
        {"name": "Regulatory non-compliance", "severity": 8, "category": "Regulatory"},
        {"name": "Unusual trading patterns", "severity": 7, "category": "Market"},
        {"name": "Management turnover", "severity": 5, "category": "Management"},
        {"name": "Delayed filings", "severity": 6, "category": "Financial"},
        {"name": "Auditor concerns", "severity": 9, "category": "Financial"}
    ]
    
    df_flags = pd.DataFrame(red_flags)
    
    fig = px.bar(
        df_flags,
        y="name",
        x="severity",
        color="category",
        title="Red Flag Severity (0-10 scale)",
        orientation='h',
        labels={"name": "Red Flag", "severity": "Severity Score", "category": "Category"},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    fig.update_layout(xaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample visualization 2: Event Timeline
    st.subheader("Event Timeline")
    
    # Sample data
    events = [
        {"date": "2024-01-15", "event": "Quarterly Report Release", "importance": 5},
        {"date": "2024-02-10", "event": "CEO Resignation", "importance": 9},
        {"date": "2024-03-05", "event": "Regulatory Investigation Announced", "importance": 8},
        {"date": "2024-03-20", "event": "Stock Price Decline (15%)", "importance": 7},
        {"date": "2024-04-12", "event": "Delayed Financial Filing", "importance": 6},
        {"date": "2024-05-03", "event": "Whistleblower Allegations", "importance": 9}
    ]
    
    df_events = pd.DataFrame(events)
    df_events["date"] = pd.to_datetime(df_events["date"])
    df_events = df_events.sort_values("date")
    
    fig = px.timeline(
        df_events,
        x_start="date",
        y="event",
        color="importance",
        title="Key Events Timeline",
        labels={"event": "Event", "date": "Date", "importance": "Importance"},
        color_continuous_scale=px.colors.sequential.Reds
    )
    
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample visualization 3: Entity Network
    st.subheader("Entity Relationship Network")
    
    # Display a pre-generated graph image as placeholder
    # In a real implementation, this would be generated dynamically
    
    st.markdown("""
    *This visualization would show connections between entities identified in the analysis.*
    
    Entity networks help identify:
    - Key individuals involved in potential misconduct
    - Relationships between companies and regulatory bodies
    - Patterns of interaction between related parties
    """)
    
    # Sample visualization 4: Risk Assessment
    st.subheader("Risk Assessment")
    
    # Sample data
    risk_categories = ["Financial", "Regulatory", "Operational", "Reputational", "Legal"]
    risk_scores = [7, 9, 5, 8, 9]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=risk_scores,
        theta=risk_categories,
        fill='toself',
        name='Risk Assessment',
        line_color='indianred'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False,
        title="Risk Assessment by Category (0-10 scale)"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Financial Forensic Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("Financial Forensic Analysis")
    
    tabs = [
        "Start Analysis",
        "Workflow Status", 
        "View Reports",
        "Document Management",
        "Query Documents",
        "Visualizations"
    ]
    
    st.sidebar.subheader("Navigation")
    selected_tab = st.sidebar.radio("Go to", tabs, index=tabs.index(st.session_state.current_tab))
    st.session_state.current_tab = selected_tab
    
    # Render selected tab
    if selected_tab == "Start Analysis":
        render_start_analysis()
    elif selected_tab == "Workflow Status":
        render_workflow_status()
    elif selected_tab == "View Reports":
        render_view_reports()
    elif selected_tab == "Document Management":
        render_document_management()
    elif selected_tab == "Query Documents":
        render_document_query()
    elif selected_tab == "Visualizations":
        render_visualizations()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Financial Forensic Analysis Platform**
        
        This platform helps identify potential red flags and concerns in corporate financial reporting and behavior.
        
        Start by entering a company name and analyzing news and regulatory filings automatically.
        """
    )


if __name__ == "__main__":
    main()