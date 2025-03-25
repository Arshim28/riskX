# RiskX - Financial Forensic Analysis System

A comprehensive multi-agent system for corporate financial forensic analysis.

## Overview

RiskX is a powerful financial forensic analysis platform that leverages multiple specialized AI agents to analyze companies, identify red flags, and generate detailed risk reports. The system combines web research, corporate information analysis, and document-based insights to provide a comprehensive view of company risks.

## Features

- **Multi-Agent Architecture**: Utilizes specialized agents for research, analysis, corporate data, and report writing
- **Document Analysis (RAG)**: Incorporates document-based insights through Retrieval-Augmented Generation
- **Event Timeline Construction**: Builds chronological timelines of significant company events
- **Red Flag Identification**: Automatically identifies potential concerns in corporate behavior
- **Comprehensive Reporting**: Generates detailed reports with executive summaries
- **Interactive UI**: Streamlit-based interface for easy interaction with the system

## Getting Started

### Prerequisites

- Python 3.10+
- Required API keys:
  - Google API key for Gemini models
  - Mistral API key for OCR services

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/riskX.git
cd riskX
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your API keys
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
echo "MISTRAL_API_KEY=your_mistral_api_key_here" >> .env
```

## Usage

### Command Line Interface

Analyze a company:
```bash
python main.py analyze --company "Company Name" --industry "Industry Name"
```

Query the RAG system:
```bash
python main.py rag query --query "What are the regulatory issues for this company?" --vector-store vector_store
```

Add documents to the RAG system:
```bash
python main.py rag add --file /path/to/document.pdf --topics "finance,regulation"
```

List documents in RAG:
```bash
python main.py rag list
```

Start the API server:
```bash
python main.py server
```

### Streamlit Interface

Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

## Architecture

The system implements a true agent-based architecture with centralized orchestration:

1. **Meta-Agent**: Central orchestrator that manages workflow phases and agent coordination
2. **Research Pool**: Manages parallel execution of research, YouTube, and corporate agents
3. **Analyst Pool**: Processes and analyzes research results with forensic techniques
4. **Writer Agent**: Generates comprehensive reports based on analysis results

The workflow follows these phases:
1. RESEARCH: Collect data about the company
2. ANALYSIS: Process and analyze the gathered information
3. REPORT_GENERATION: Create reports and executive briefings
4. REPORT_REVIEW: Final quality assessment
5. COMPLETE: Workflow completed

## Configuration

Configuration is managed through `config.yaml`. Key settings include:

- Embedding and vector store settings
- Workflow execution parameters
- Agent-specific configurations
- RAG system settings

## License

[Specify your license here]

## Acknowledgments

- Built with LangGraph, FastAPI, and Streamlit
- Powered by Google Gemini and Mistral AI models