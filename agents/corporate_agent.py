# agents/corporate_agent.py
from typing import Dict, List, Any, Optional, Tuple
import json
import asyncio
from datetime import datetime
import traceback
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    RetryError,
    before_sleep_log
)
import logging
import aiohttp

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.postgres_tool import PostgresTool
from tools.nse_tool import NSETool, NSEConnectionError, NSERateLimitError, NSEDataError


class APIError(Exception):
    """Base exception for API-related errors."""
    pass

class ConnectionError(APIError):
    """Exception raised when there's a connection issue with an external API."""
    pass

class RateLimitError(APIError):
    """Exception raised when an API rate limit is exceeded."""
    pass

class DataProcessingError(APIError):
    """Exception raised when there's an issue processing the data."""
    pass

class ValidationError(APIError):
    """Exception raised when data validation fails."""
    pass


class CorporateAgent(BaseAgent):
    name = "corporate_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager()
        
        self.postgres_tool = PostgresTool(config.get("postgres", {}))
        self.nse_tool = NSETool(config.get("nse", {}))
        
        self.company_data = {}
        self.financial_data = {}
        self.regulatory_data = {}
        self.management_data = {}
        self.market_data = {}
        
        # Set default retry parameters
        self.retry_attempts = config.get("retry", {}).get("max_attempts", 3)
        self.retry_multiplier = config.get("retry", {}).get("multiplier", 1)
        self.retry_min_wait = config.get("retry", {}).get("min_wait", 2)
        self.retry_max_wait = config.get("retry", {}).get("max_wait", 10)
        
    def _get_retry_decorator(self, operation_name: str):
        """Create a standardized retry decorator with logging."""
        return retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=self.retry_multiplier, min=self.retry_min_wait, max=self.retry_max_wait),
            retry=(
                retry_if_exception_type(ConnectionError) | 
                retry_if_exception_type(RateLimitError) |
                retry_if_exception_type(aiohttp.ClientError) |
                retry_if_exception_type(NSEConnectionError) |
                retry_if_exception_type(NSERateLimitError)
            ),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
            reraise=True
        )
    
    def _parse_json_response(self, response: str, context: str = "response") -> Dict[str, Any]:
        """Parse JSON response with better error handling."""
        try:
            # Clean response if it contains markdown code blocks
            if "```json" in response:
                response = response.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in response:
                response = response.split("```", 1)[1].split("```", 1)[0]
                
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse {context} JSON: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg)
    
    def _validate_result(self, result: Any, required_fields: List[str], context: str = "result") -> None:
        """Validate that the result contains all required fields."""
        if not result:
            raise ValidationError(f"Empty {context} received")
            
        if not isinstance(result, dict):
            raise ValidationError(f"{context} must be a dictionary")
            
        for field in required_fields:
            if field not in result:
                raise ValidationError(f"Missing required field '{field}' in {context}")
    
    def _map_nse_error(self, error: Exception, operation: str) -> Exception:
        """Map NSE tool errors to standardized exception types."""
        if isinstance(error, NSEConnectionError):
            return ConnectionError(f"Connection error during {operation}: {str(error)}")
        elif isinstance(error, NSERateLimitError):
            return RateLimitError(f"Rate limit exceeded during {operation}: {str(error)}")
        elif isinstance(error, NSEDataError):
            return DataProcessingError(f"Data error during {operation}: {str(error)}")
        return error
    
    async def fetch_company_info(self, company: str) -> Dict[str, Any]:
        """Fetch company information with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("fetch_company_info")
        
        @retry_decorator
        async def _fetch_with_retry():
            self.logger.info(f"Fetching company information for {company}")
            
            try:
                db_result = await self.postgres_tool.run(
                    command="execute_query",
                    query="SELECT * FROM companies WHERE name = $1",
                    params=[company]
                )
                
                company_info = {}
                
                if db_result.success and db_result.data:
                    company_info = db_result.data
                    self.logger.info(f"Found company information in database for {company}")
                else:
                    self.logger.info(f"No company information found in database for {company}, collecting from external sources")
                    
                    llm_provider = await get_llm_provider()
                    
                    variables = {
                        "company": company,
                    }
                    
                    system_prompt, human_prompt = self.prompt_manager.get_prompt(
                        agent_name=self.name,
                        operation="company_lookup",
                        variables=variables
                    )
                    
                    input_message = [
                        ("system", system_prompt),
                        ("human", human_prompt)
                    ]
                    
                    response = await llm_provider.generate_text(
                        input_message, 
                        model_name=self.config.get("models", {}).get("lookup")
                    )
                    
                    company_info = self._parse_json_response(response, "company information")
                    
                    # Validate required fields
                    self._validate_result(
                        company_info, 
                        ["name", "industry", "headquarters"], 
                        "company information"
                    )
                    
                    if company_info and 'name' in company_info:
                        await self.postgres_tool.run(
                            command="execute_query",
                            query="INSERT INTO companies (name, data) VALUES ($1, $2) ON CONFLICT (name) DO UPDATE SET data = $2, updated_at = CURRENT_TIMESTAMP",
                            params=[company, json.dumps(company_info)]
                        )
                
                self.company_data = company_info
                return company_info
                
            except NSEConnectionError as e:
                raise ConnectionError(f"Connection error fetching company info: {str(e)}")
            except NSERateLimitError as e:
                raise RateLimitError(f"Rate limit exceeded fetching company info: {str(e)}")
            except NSEDataError as e:
                raise DataProcessingError(f"Data error fetching company info: {str(e)}")
            except json.JSONDecodeError as e:
                raise DataProcessingError(f"Error parsing company info JSON: {str(e)}")
            except ValidationError as e:
                # Re-raise validation errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error fetching company info: {str(e)}")
                raise
        
        try:
            return await _fetch_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise ConnectionError(f"Failed to fetch company information after {self.retry_attempts} attempts")
        
    async def analyze_financial_statements(self, company: str) -> Dict[str, Any]:
        """Analyze financial statements with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("analyze_financial_statements")
        
        @retry_decorator
        async def _analyze_with_retry():
            self.logger.info(f"Analyzing financial statements for {company}")
            
            try:
                financial_result = await self.nse_tool.run(
                    command="get_financial_results",
                    company=company,
                    quarters=4
                )
                
                if not financial_result.success:
                    error_msg = f"Failed to fetch financial results for {company}: {financial_result.error}"
                    self.logger.error(error_msg)
                    
                    # Map NSE errors to standardized exceptions
                    if "rate limit" in str(financial_result.error).lower():
                        raise RateLimitError(error_msg)
                    elif "connection" in str(financial_result.error).lower():
                        raise ConnectionError(error_msg)
                    else:
                        raise DataProcessingError(error_msg)
                    
                financial_data = financial_result.data
                
                # Validate financial data
                if not financial_data or not isinstance(financial_data, (dict, list)):
                    raise ValidationError(f"Invalid financial data received for {company}")
                
                llm_provider = await get_llm_provider()
                
                variables = {
                    "company": company,
                    "financial_data": json.dumps(financial_data)
                }
                
                system_prompt, human_prompt = self.prompt_manager.get_prompt(
                    agent_name=self.name,
                    operation="financial_analysis",
                    variables=variables
                )
                
                input_message = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                response = await llm_provider.generate_text(
                    input_message, 
                    model_name=self.config.get("models", {}).get("analysis")
                )
                
                analysis_result = self._parse_json_response(response, "financial analysis")
                
                # Validate required fields in analysis result
                self._validate_result(
                    analysis_result, 
                    ["summary", "insights", "red_flags"], 
                    "financial analysis"
                )
                
                self.financial_data = {
                    "raw_data": financial_data,
                    "analysis": analysis_result
                }
                
                return analysis_result
                
            except NSEConnectionError as e:
                raise ConnectionError(f"Connection error analyzing financial statements: {str(e)}")
            except NSERateLimitError as e:
                raise RateLimitError(f"Rate limit exceeded analyzing financial statements: {str(e)}")
            except NSEDataError as e:
                raise DataProcessingError(f"Data error analyzing financial statements: {str(e)}")
            except ValidationError as e:
                # Re-raise validation errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error analyzing financial statements: {str(e)}")
                raise
        
        try:
            return await _analyze_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise ConnectionError(f"Failed to analyze financial statements after {self.retry_attempts} attempts")
    
    async def check_regulatory_filings(self, company: str) -> Dict[str, Any]:
        """Check regulatory filings with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("check_regulatory_filings")
        
        @retry_decorator
        async def _check_with_retry():
            self.logger.info(f"Checking regulatory filings for {company}")
            
            try:
                db_result = await self.postgres_tool.run(
                    command="execute_query",
                    query="SELECT * FROM regulatory_filings WHERE company = $1 ORDER BY filing_date DESC LIMIT 20",
                    params=[company]
                )
                
                regulatory_filings = []
                
                if db_result.success and db_result.data:
                    regulatory_filings = db_result.data
                    self.logger.info(f"Found {len(regulatory_filings)} regulatory filings in database for {company}")
                
                if len(regulatory_filings) < 5:
                    self.logger.info(f"Insufficient regulatory filings in database, fetching from NSE for {company}")
                    
                    nse_result = await self.nse_tool.run(
                        command="get_company_announcements",
                        company=company,
                        filing_type="regulatory",
                        limit=20
                    )
                    
                    if not nse_result.success:
                        error_msg = f"Failed to fetch regulatory filings from NSE for {company}: {nse_result.error}"
                        self.logger.error(error_msg)
                        
                        # Map NSE errors to standardized exceptions
                        if "rate limit" in str(nse_result.error).lower():
                            raise RateLimitError(error_msg)
                        elif "connection" in str(nse_result.error).lower():
                            raise ConnectionError(error_msg)
                        else:
                            raise DataProcessingError(error_msg)
                    
                    if nse_result.success and nse_result.data:
                        regulatory_filings = nse_result.data
                        
                        # Save filings to database
                        for filing in regulatory_filings:
                            await self.postgres_tool.run(
                                command="execute_query",
                                query="INSERT INTO regulatory_filings (company, filing_date, filing_type, filing_data) VALUES ($1, $2, $3, $4) ON CONFLICT (company, filing_date, filing_type) DO NOTHING",
                                params=[company, filing.get("date"), filing.get("type"), json.dumps(filing)]
                            )
                
                # Validate regulatory filings
                if not isinstance(regulatory_filings, list):
                    raise ValidationError(f"Invalid regulatory filings data format for {company}")
                
                llm_provider = await get_llm_provider()
                
                variables = {
                    "company": company,
                    "regulatory_filings": json.dumps(regulatory_filings)
                }
                
                system_prompt, human_prompt = self.prompt_manager.get_prompt(
                    agent_name=self.name,
                    operation="regulatory_analysis",
                    variables=variables
                )
                
                input_message = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                response = await llm_provider.generate_text(
                    input_message, 
                    model_name=self.config.get("models", {}).get("analysis")
                )
                
                analysis_result = self._parse_json_response(response, "regulatory analysis")
                
                # Validate required fields in analysis result
                self._validate_result(
                    analysis_result, 
                    ["summary", "compliance_status", "red_flags"], 
                    "regulatory analysis"
                )
                
                self.regulatory_data = {
                    "filings": regulatory_filings,
                    "analysis": analysis_result
                }
                
                return analysis_result
                
            except NSEConnectionError as e:
                raise ConnectionError(f"Connection error checking regulatory filings: {str(e)}")
            except NSERateLimitError as e:
                raise RateLimitError(f"Rate limit exceeded checking regulatory filings: {str(e)}")
            except NSEDataError as e:
                raise DataProcessingError(f"Data error checking regulatory filings: {str(e)}")
            except ValidationError as e:
                # Re-raise validation errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error checking regulatory filings: {str(e)}")
                raise
        
        try:
            return await _check_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise ConnectionError(f"Failed to check regulatory filings after {self.retry_attempts} attempts")
    
    async def analyze_management_team(self, company: str) -> Dict[str, Any]:
        """Analyze management team with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("analyze_management_team")
        
        @retry_decorator
        async def _analyze_with_retry():
            self.logger.info(f"Analyzing management team for {company}")
            
            try:
                db_result = await self.postgres_tool.run(
                    command="execute_query",
                    query="SELECT * FROM company_management WHERE company = $1",
                    params=[company]
                )
                
                management_info = []
                
                if db_result.success and db_result.data:
                    management_info = db_result.data
                    self.logger.info(f"Found management information in database for {company}")
                else:
                    self.logger.info(f"No management information found in database, collecting from external sources for {company}")
                    
                    llm_provider = await get_llm_provider()
                    
                    variables = {
                        "company": company
                    }
                    
                    system_prompt, human_prompt = self.prompt_manager.get_prompt(
                        agent_name=self.name,
                        operation="management_lookup",
                        variables=variables
                    )
                    
                    input_message = [
                        ("system", system_prompt),
                        ("human", human_prompt)
                    ]
                    
                    response = await llm_provider.generate_text(
                        input_message, 
                        model_name=self.config.get("models", {}).get("lookup")
                    )
                    
                    management_info = self._parse_json_response(response, "management information")
                    
                    # Validate management info
                    if not isinstance(management_info, list):
                        raise ValidationError(f"Invalid management information format for {company}")
                    
                    if management_info:
                        for manager in management_info:
                            if not all(k in manager for k in ["name", "position"]):
                                continue  # Skip invalid entries
                                
                            await self.postgres_tool.run(
                                command="execute_query",
                                query="INSERT INTO company_management (company, name, position, data) VALUES ($1, $2, $3, $4) ON CONFLICT (company, name, position) DO UPDATE SET data = $4, updated_at = CURRENT_TIMESTAMP",
                                params=[company, manager.get("name"), manager.get("position"), json.dumps(manager)]
                            )
                
                analysis_result = {}
                
                if management_info:
                    llm_provider = await get_llm_provider()
                    
                    variables = {
                        "company": company,
                        "management_info": json.dumps(management_info)
                    }
                    
                    system_prompt, human_prompt = self.prompt_manager.get_prompt(
                        agent_name=self.name,
                        operation="management_analysis",
                        variables=variables
                    )
                    
                    input_message = [
                        ("system", system_prompt),
                        ("human", human_prompt)
                    ]
                    
                    response = await llm_provider.generate_text(
                        input_message, 
                        model_name=self.config.get("models", {}).get("analysis")
                    )
                    
                    analysis_result = self._parse_json_response(response, "management analysis")
                    
                    # Validate required fields in analysis result
                    self._validate_result(
                        analysis_result, 
                        ["leadership_assessment", "red_flags"], 
                        "management analysis"
                    )
                else:
                    analysis_result = {
                        "error": "No management information available",
                        "leadership_assessment": "Unknown - insufficient data",
                        "red_flags": ["No management information available"],
                        "summary": "Unable to analyze management due to lack of data"
                    }
                
                self.management_data = {
                    "management": management_info,
                    "analysis": analysis_result
                }
                
                return analysis_result
                
            except ValidationError as e:
                # Re-raise validation errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error analyzing management team: {str(e)}")
                raise
        
        try:
            return await _analyze_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise ConnectionError(f"Failed to analyze management team after {self.retry_attempts} attempts")
    
    async def analyze_market_data(self, company: str) -> Dict[str, Any]:
        """Analyze market data with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("analyze_market_data")
        
        @retry_decorator
        async def _analyze_with_retry():
            self.logger.info(f"Analyzing market data for {company}")
            
            try:
                price_result = await self.nse_tool.run(
                    command="get_stock_price_history",
                    company=company,
                    days=180
                )
                
                if not price_result.success:
                    error_msg = f"Failed to fetch stock price history for {company}: {price_result.error}"
                    self.logger.error(error_msg)
                    
                    # Map NSE errors to standardized exceptions
                    if "rate limit" in str(price_result.error).lower():
                        raise RateLimitError(error_msg)
                    elif "connection" in str(price_result.error).lower():
                        raise ConnectionError(error_msg)
                    else:
                        raise DataProcessingError(error_msg)
                    
                price_data = price_result.data
                
                # Validate price data
                if not price_data or not isinstance(price_data, (dict, list)):
                    raise ValidationError(f"Invalid stock price data for {company}")
                
                pattern_result = await self.nse_tool.run(
                    command="detect_unusual_patterns",
                    company=company,
                    price_data=price_data
                )
                
                unusual_patterns = []
                if pattern_result.success:
                    unusual_patterns = pattern_result.data
                else:
                    self.logger.warning(f"Failed to detect unusual patterns for {company}: {pattern_result.error}")
                    
                llm_provider = await get_llm_provider()
                
                variables = {
                    "company": company,
                    "price_data": json.dumps(price_data),
                    "unusual_patterns": json.dumps(unusual_patterns)
                }
                
                system_prompt, human_prompt = self.prompt_manager.get_prompt(
                    agent_name=self.name,
                    operation="market_analysis",
                    variables=variables
                )
                
                input_message = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                response = await llm_provider.generate_text(
                    input_message, 
                    model_name=self.config.get("models", {}).get("analysis")
                )
                
                analysis_result = self._parse_json_response(response, "market analysis")
                
                # Validate required fields in analysis result
                self._validate_result(
                    analysis_result, 
                    ["market_summary", "volatility_assessment", "red_flags"], 
                    "market analysis"
                )
                
                self.market_data = {
                    "price_data": price_data,
                    "unusual_patterns": unusual_patterns,
                    "analysis": analysis_result
                }
                
                return analysis_result
                
            except NSEConnectionError as e:
                raise ConnectionError(f"Connection error analyzing market data: {str(e)}")
            except NSERateLimitError as e:
                raise RateLimitError(f"Rate limit exceeded analyzing market data: {str(e)}")
            except NSEDataError as e:
                raise DataProcessingError(f"Data error analyzing market data: {str(e)}")
            except ValidationError as e:
                # Re-raise validation errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error analyzing market data: {str(e)}")
                raise
        
        try:
            return await _analyze_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise ConnectionError(f"Failed to analyze market data after {self.retry_attempts} attempts")
    
    async def generate_corporate_report(self, company: str) -> Dict[str, Any]:
        """Generate corporate report with enhanced error handling and retries."""
        retry_decorator = self._get_retry_decorator("generate_corporate_report")
        
        @retry_decorator
        async def _generate_with_retry():
            self.logger.info(f"Generating corporate report for {company}")
            
            try:
                combined_data = {
                    "company_info": self.company_data,
                    "financial_data": self.financial_data,
                    "regulatory_data": self.regulatory_data,
                    "management_data": self.management_data,
                    "market_data": self.market_data,
                }
                
                # Check if we have enough data to generate a meaningful report
                missing_data = [k for k, v in combined_data.items() if not v]
                if missing_data:
                    self.logger.warning(f"Missing data for {company}: {missing_data}")
                
                llm_provider = await get_llm_provider()
                
                variables = {
                    "company": company,
                    "combined_data": json.dumps(combined_data)
                }
                
                system_prompt, human_prompt = self.prompt_manager.get_prompt(
                    agent_name=self.name,
                    operation="corporate_report",
                    variables=variables
                )
                
                input_message = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                response = await llm_provider.generate_text(
                    input_message, 
                    model_name=self.config.get("models", {}).get("report")
                )
                
                report = self._parse_json_response(response, "corporate report")
                
                # Validate required fields in the report
                self._validate_result(
                    report, 
                    ["title", "executive_summary", "findings", "red_flags"], 
                    "corporate report"
                )
                
                # Add timestamp and source information
                report["company"] = company
                report["timestamp"] = datetime.now().isoformat()
                report["data_sources"] = [k for k, v in combined_data.items() if v]
                
                # Store report in database
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO corporate_reports (company, report_date, report_data) VALUES ($1, $2, $3)",
                    params=[company, datetime.now().isoformat(), json.dumps(report)]
                )
                
                return report
                
            except ValidationError as e:
                # Re-raise validation errors
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error generating corporate report: {str(e)}")
                raise
        
        try:
            return await _generate_with_retry()
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise ConnectionError(f"Failed to generate corporate report after {self.retry_attempts} attempts")
    
    async def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await self.run(state)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the corporate agent with enhanced error handling and logging."""
        try:
            self._log_start(state)
            
            company = state.get("company", "")
            
            if not company:
                self.logger.error("Company name is missing!")
                return {**state, "goto": "meta_agent", "corporate_status": "ERROR", "error": "Company name is missing"}
            
            self.logger.info(f"Starting corporate analysis for {company}")
            
            corporate_results = {
                "company_info": {},
                "financial_analysis": {},
                "regulatory_analysis": {},
                "management_analysis": {},
                "market_analysis": {},
                "red_flags": [],
                "corporate_report": {}
            }
            
            try:
                # Create and execute tasks for all analysis components
                company_info_task = self.fetch_company_info(company)
                financial_analysis_task = self.analyze_financial_statements(company)
                regulatory_analysis_task = self.check_regulatory_filings(company)
                management_analysis_task = self.analyze_management_team(company)
                market_analysis_task = self.analyze_market_data(company)
                
                # Wait for all tasks to complete, with proper error handling
                try:
                    company_info = await company_info_task
                    corporate_results["company_info"] = company_info
                except Exception as e:
                    self.logger.error(f"Error fetching company info: {str(e)}")
                    corporate_results["company_info"] = {"error": str(e)}
                
                try:
                    financial_analysis = await financial_analysis_task
                    corporate_results["financial_analysis"] = financial_analysis
                except Exception as e:
                    self.logger.error(f"Error analyzing financial statements: {str(e)}")
                    corporate_results["financial_analysis"] = {"error": str(e)}
                
                try:
                    regulatory_analysis = await regulatory_analysis_task
                    corporate_results["regulatory_analysis"] = regulatory_analysis
                except Exception as e:
                    self.logger.error(f"Error checking regulatory filings: {str(e)}")
                    corporate_results["regulatory_analysis"] = {"error": str(e)}
                
                try: 
                    management_analysis = await management_analysis_task
                    corporate_results["management_analysis"] = management_analysis
                except Exception as e:
                    self.logger.error(f"Error analyzing management team: {str(e)}")
                    corporate_results["management_analysis"] = {"error": str(e)}
                
                try:
                    market_analysis = await market_analysis_task
                    corporate_results["market_analysis"] = market_analysis
                except Exception as e:
                    self.logger.error(f"Error analyzing market data: {str(e)}")
                    corporate_results["market_analysis"] = {"error": str(e)}
                
                # Collect red flags from each analysis component
                for analysis_type, analysis in corporate_results.items():
                    if analysis_type == "red_flags":
                        continue
                    
                    if analysis and isinstance(analysis, dict) and "red_flags" in analysis and isinstance(analysis["red_flags"], list):
                        # Add source to each red flag for better context
                        prefixed_flags = [f"[{analysis_type}] {flag}" for flag in analysis["red_flags"]]
                        corporate_results["red_flags"].extend(prefixed_flags)
                
                # Generate the final corporate report
                try:
                    corporate_report = await self.generate_corporate_report(company)
                    corporate_results["corporate_report"] = corporate_report
                except Exception as e:
                    self.logger.error(f"Error generating corporate report: {str(e)}")
                    corporate_results["corporate_report"] = {
                        "error": str(e),
                        "title": f"Corporate Analysis for {company}",
                        "executive_summary": "Error generating complete report. See individual analysis sections.",
                        "findings": "Error generating findings.",
                        "red_flags": corporate_results["red_flags"]
                    }
                
                self.logger.info(f"Completed corporate analysis for {company} with {len(corporate_results['red_flags'])} red flags")
                
            except Exception as e:
                tb = traceback.format_exc()
                error_msg = f"Error during corporate analysis for {company}: {str(e)}\n{tb}"
                self.logger.error(error_msg)
                return {
                    **state, 
                    "goto": "meta_agent", 
                    "error": error_msg,
                    "corporate_status": "ERROR",
                    "corporate_results": corporate_results  # Include partial results
                }
            
            state["corporate_results"] = corporate_results
            state["corporate_status"] = "DONE"
            
            goto = "meta_agent"
            if state.get("synchronous_pipeline", False):
                goto = state.get("next_agent", "meta_agent")
                
            self._log_completion({**state, "goto": goto})
            return {**state, "goto": goto}
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Unhandled error in corporate_agent run method: {str(e)}\n{tb}"
            self.logger.error(error_msg)
            return {
                **state, 
                "goto": "meta_agent", 
                "error": error_msg,
                "corporate_status": "ERROR"
            }