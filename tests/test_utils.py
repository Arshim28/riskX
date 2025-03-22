import os
import json
import asyncio
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from utils.logging import setup_logging

setup_logging("test_logs")

def async_test(test_case):
    def wrapper(*args, **kwargs):
        coroutine = test_case(*args, **kwargs)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coroutine)
    return wrapper

TEST_CONFIG = {
    "models": {
        "planning": "gemini-2.0-flash",
        "analysis": "gemini-2.0-pro",
        "report": "gemini-2.0-pro",
        "evaluation": "gemini-2.0-pro",
        "lookup": "gemini-2.0-flash",
        "summary": "gemini-2.0-pro"
    },
    "forensic_analysis": {
        "max_workers": 2,
        "batch_size": 2,
        "concurrent_events": 1,
        "task_timeout": 10,
        "evidence_strength": 3
    },
    "writer": {
        "max_concurrent_sections": 2,
        "enable_iterative_improvement": True
    },
    "rag_agent": {
        "retrieval_k": 3,
        "reranking_enabled": False,
        "max_input_tokens": 4000
    },
    "meta_agent": {
        "max_parallel_agents": 2,
        "parallel_execution": True
    }
}

SAMPLE_COMPANY = "Test Company Inc."
SAMPLE_INDUSTRY = "Technology"

SAMPLE_RESEARCH_PLAN = {
    "objective": "Investigate Test Company Inc. for potential financial misconduct",
    "key_areas_of_focus": ["Financial reporting", "Regulatory compliance", "Executive behavior"],
    "query_categories": {
        "financial": "Test Company Inc. financial reporting issues",
        "regulatory": "Test Company Inc. regulatory violations"
    },
    "query_generation_guidelines": "Focus on factual information with emphasis on potential issues"
}

SAMPLE_ARTICLES = [
    {
        "title": "Test Company Faces Investigation",
        "link": "https://example.com/news1",
        "snippet": "Test Company Inc. is under investigation for financial irregularities.",
        "source": "Test News",
        "date": "2023-01-15",
        "category": "regulatory",
        "is_quarterly_report": False
    },
    {
        "title": "Test Company Reports Q2 Earnings",
        "link": "https://example.com/news2",
        "snippet": "Test Company Inc. reports quarterly earnings below analyst expectations.",
        "source": "Financial Times",
        "date": "2023-04-15",
        "category": "financial",
        "is_quarterly_report": True
    }
]

MOCK_LLM_RESPONSES = {
    "query_generation": json.dumps({
        "financial": ["Test Company Inc. accounting issues", "Test Company Inc. financial fraud"],
        "regulatory": ["Test Company Inc. SEC investigation", "Test Company Inc. compliance violations"]
    }),
    "article_clustering": json.dumps({
        "Financial Irregularities Investigation (2023) - High": [0],
        "Quarterly Financial Results (Q2 2023) - Low": [1]
    }),
    "forensic_insights": json.dumps({
        "ALLEGATIONS": "Financial irregularities and misreporting",
        "ENTITIES": "Test Company Inc., SEC",
        "TIMELINE": "January 2023",
        "MAGNITUDE": "Unknown",
        "EVIDENCE": "Ongoing investigation",
        "RESPONSE": "Company denies allegations",
        "STATUS": "Investigation ongoing",
        "CREDIBILITY": "7/10"
    }),
    "event_synthesis": json.dumps({
        "cross_validation": "Multiple sources confirm investigation",
        "timeline": [{"date": "2023-01-15", "description": "Investigation announced"}],
        "key_entities": [{"name": "Test Company", "role": "Subject"}],
        "evidence_assessment": "Moderate evidence of misconduct",
        "severity_assessment": "Potentially severe",
        "credibility_score": 7,
        "red_flags": ["Unusual accounting practices"],
        "narrative": "Test Company is under investigation for financial irregularities."
    }),
    "company_analysis": json.dumps({
        "executive_summary": "Test Company has several concerning financial practices.",
        "risk_assessment": {
            "financial_integrity_risk": "High",
            "legal_regulatory_risk": "Medium",
            "reputational_risk": "High",
            "operational_risk": "Low"
        },
        "key_patterns": ["Consistent misreporting"],
        "critical_entities": [{"name": "Test Company", "role": "Subject"}],
        "red_flags": ["Unusual accounting", "Regulatory issues"],
        "timeline": [{"date": "2023-01-15", "description": "Investigation announced"}],
        "forensic_assessment": "Significant concerns about financial reporting",
        "report_markdown": "# Forensic Analysis of Test Company\n\nSignificant concerns identified."
    }),
    "detailed_event_analysis": "## Test Event\n\nThis event involves suspected financial irregularities at Test Company.",
    "executive_summary": "# Executive Summary\n\nTest Company has several concerning practices that warrant further investigation.",
    "recommendations": "# Recommendations\n\n1. Conduct forensic audit\n2. Review financial controls\n3. Enhance compliance measures",
    "company_lookup": json.dumps({
        "name": "Test Company Inc.",
        "industry": "Technology",
        "founded": "2000",
        "headquarters": "New York",
        "ceo": "John Doe",
        "stock_symbol": "TSTI"
    }),
    "financial_analysis": json.dumps({
        "revenue_trend": "Declining",
        "profit_margins": "Inconsistent",
        "debt_ratio": "High",
        "cash_flow": "Negative",
        "red_flags": ["Unusual revenue recognition", "Inconsistent reporting"]
    }),
    "regulatory_analysis": json.dumps({
        "compliance_status": "Multiple issues",
        "recent_violations": ["SEC reporting violation", "Disclosure failures"],
        "penalties": "$5M in fines",
        "red_flags": ["Repeat violations", "Delayed filings"]
    }),
    "video_analysis": json.dumps({
        "forensic_relevance": "medium",
        "red_flags": ["Executive statements inconsistent with filings"],
        "summary": "Video shows executives making claims not supported by financial data."
    }),
    "video_summary": json.dumps({
        "overall_assessment": "Concerning statements by executives",
        "key_insights": ["Pattern of misleading public statements"],
        "red_flags": ["Contradictory financial claims", "Evasive answers to direct questions"],
        "notable_videos": ["Executive interview on Financial News"],
        "summary": "Analysis of videos reveals inconsistencies in public statements by executives."
    }),
    "rag_response": "Based on the documents, Test Company has had several instances of irregular financial reporting."
}

class MockLLMProvider:
    async def generate_text(self, prompt, model_name=None, temperature=None):
        prompt_lower = str(prompt).lower()
        
        if "query_generation" in prompt_lower:
            return MOCK_LLM_RESPONSES["query_generation"]
        elif "article_clustering" in prompt_lower:
            return MOCK_LLM_RESPONSES["article_clustering"]
        elif "extract_forensic_insight" in prompt_lower:
            return "Extracted forensic content"
        elif "analyze_forensic_content" in prompt_lower:
            return MOCK_LLM_RESPONSES["forensic_insights"]
        elif "synthesize_event_insights" in prompt_lower:
            return MOCK_LLM_RESPONSES["event_synthesis"]
        elif "company_analysis" in prompt_lower:
            return MOCK_LLM_RESPONSES["company_analysis"]
        elif "detailed_event_analysis" in prompt_lower:
            return MOCK_LLM_RESPONSES["detailed_event_analysis"]
        elif "executive_summary" in prompt_lower:
            return MOCK_LLM_RESPONSES["executive_summary"]
        elif "recommendations" in prompt_lower:
            return MOCK_LLM_RESPONSES["recommendations"]
        elif "company_lookup" in prompt_lower:
            return MOCK_LLM_RESPONSES["company_lookup"]
        elif "financial_analysis" in prompt_lower:
            return MOCK_LLM_RESPONSES["financial_analysis"]
        elif "regulatory_analysis" in prompt_lower:
            return MOCK_LLM_RESPONSES["regulatory_analysis"]
        elif "analyze_transcript" in prompt_lower:
            return MOCK_LLM_RESPONSES["video_analysis"]
        elif "generate_summary" in prompt_lower:
            return MOCK_LLM_RESPONSES["video_summary"]
        elif "qa_template" in prompt_lower:
            return MOCK_LLM_RESPONSES["rag_response"]
        else:
            # Default response
            return json.dumps({"result": "default mock response"})