from typing import Dict, List, Any, Optional, Tuple
import json
import asyncio
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from base.base_agents import BaseAgent
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.postgres_tool import PostgresTool
from tools.nse_tool import NSETool


class CorporateAgent(BaseAgent):
    name = "corporate_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager(self.name)
        
        # Initialize tools
        self.postgres_tool = PostgresTool(config.get("postgres", {}))
        self.nse_tool = NSETool(config.get("nse", {}))
        
        # Initialize state tracking
        self.company_data = {}
        self.financial_data = {}
        self.regulatory_data = {}
        self.management_data = {}
        self.market_data = {}
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_company_info(self, company: str) -> Dict[str, Any]:
        """Fetch comprehensive company information from database and external sources"""
        self.logger.info(f"Fetching company information for {company}")
        
        # First check if we have the data in our database
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
            
            # Collect basic company information using LLM
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
                prompt=input_message,
                model_name=self.config.get("models", {}).get("lookup")
            )
            
            try:
                company_info = json.loads(response)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse company information JSON for {company}")
                company_info = {"name": company, "error": "Failed to parse information"}
            
            # Store in database for future use
            if company_info and 'name' in company_info:
                await self.postgres_tool.run(
                    command="execute_query",
                    query="INSERT INTO companies (name, data) VALUES ($1, $2) ON CONFLICT (name) DO UPDATE SET data = $2",
                    params=[company, json.dumps(company_info)]
                )
        
        self.company_data = company_info
        return company_info
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_financial_statements(self, company: str) -> Dict[str, Any]:
        """Analyze financial statements for warning signs and forensic indicators"""
        self.logger.info(f"Analyzing financial statements for {company}")
        
        # Fetch financial statements from NSE
        financial_result = await self.nse_tool.run(
            command="get_financial_results",
            company=company,
            quarters=4  # Last 4 quarters
        )
        
        if not financial_result.success:
            self.logger.error(f"Failed to fetch financial results for {company}: {financial_result.error}")
            return {"error": f"Failed to fetch financial data: {financial_result.error}"}
            
        financial_data = financial_result.data
        
        # Perform financial analysis using LLM
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
            prompt=input_message,
            model_name=self.config.get("models", {}).get("analysis")
        )
        
        try:
            analysis_result = json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse financial analysis JSON for {company}")
            analysis_result = {"error": "Failed to parse financial analysis"}
        
        self.financial_data = {
            "raw_data": financial_data,
            "analysis": analysis_result
        }
        
        return analysis_result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def check_regulatory_filings(self, company: str) -> Dict[str, Any]:
        """Check regulatory filings for compliance issues and red flags"""
        self.logger.info(f"Checking regulatory filings for {company}")
        
        # Fetch regulatory filings from database
        db_result = await self.postgres_tool.run(
            command="execute_query",
            query="SELECT * FROM regulatory_filings WHERE company = $1 ORDER BY filing_date DESC LIMIT 20",
            params=[company]
        )
        
        regulatory_filings = []
        
        if db_result.success and db_result.data:
            regulatory_filings = db_result.data
            self.logger.info(f"Found {len(regulatory_filings)} regulatory filings in database for {company}")
        
        # If insufficient data in database, fetch from NSE
        if len(regulatory_filings) < 5:
            self.logger.info(f"Insufficient regulatory filings in database, fetching from NSE for {company}")
            
            nse_result = await self.nse_tool.run(
                command="get_company_announcements",
                company=company,
                filing_type="regulatory",
                limit=20
            )
            
            if nse_result.success and nse_result.data:
                regulatory_filings = nse_result.data
                
                # Store in database for future use
                for filing in regulatory_filings:
                    await self.postgres_tool.run(
                        command="execute_query",
                        query="INSERT INTO regulatory_filings (company, filing_date, filing_type, filing_data) VALUES ($1, $2, $3, $4) ON CONFLICT (company, filing_date, filing_type) DO NOTHING",
                        params=[company, filing.get("date"), filing.get("type"), json.dumps(filing)]
                    )
        
        # Analyze regulatory filings using LLM
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
            prompt=input_message,
            model_name=self.config.get("models", {}).get("analysis")
        )
        
        try:
            analysis_result = json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse regulatory analysis JSON for {company}")
            analysis_result = {"error": "Failed to parse regulatory analysis"}
        
        self.regulatory_data = {
            "filings": regulatory_filings,
            "analysis": analysis_result
        }
        
        return analysis_result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_management_team(self, company: str) -> Dict[str, Any]:
        """Analyze management team for potential conflicts or integrity issues"""
        self.logger.info(f"Analyzing management team for {company}")
        
        # Fetch management information
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
            
            # Collect management information using LLM
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
                prompt=input_message,
                model_name=self.config.get("models", {}).get("lookup")
            )
            
            try:
                management_info = json.loads(response)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse management information JSON for {company}")
                management_info = []
            
            # Store in database for future use
            if management_info:
                for manager in management_info:
                    await self.postgres_tool.run(
                        command="execute_query",
                        query="INSERT INTO company_management (company, name, position, data) VALUES ($1, $2, $3, $4) ON CONFLICT (company, name, position) DO UPDATE SET data = $4",
                        params=[company, manager.get("name"), manager.get("position"), json.dumps(manager)]
                    )
        
        # Analyze management team using LLM
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
                prompt=input_message,
                model_name=self.config.get("models", {}).get("analysis")
            )
            
            try:
                analysis_result = json.loads(response)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse management analysis JSON for {company}")
                analysis_result = {"error": "Failed to parse management analysis"}
        else:
            analysis_result = {"error": "No management information available"}
        
        self.management_data = {
            "management": management_info,
            "analysis": analysis_result
        }
        
        return analysis_result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_market_data(self, company: str) -> Dict[str, Any]:
        """Analyze market data for unusual patterns or irregularities"""
        self.logger.info(f"Analyzing market data for {company}")
        
        # Fetch stock price history from NSE
        price_result = await self.nse_tool.run(
            command="get_stock_price_history",
            company=company,
            days=180  # Last 180 days
        )
        
        if not price_result.success:
            self.logger.error(f"Failed to fetch stock price history for {company}: {price_result.error}")
            return {"error": f"Failed to fetch market data: {price_result.error}"}
            
        price_data = price_result.data
        
        # Fetch unusual patterns
        pattern_result = await self.nse_tool.run(
            command="detect_unusual_patterns",
            company=company,
            price_data=price_data
        )
        
        unusual_patterns = []
        if pattern_result.success:
            unusual_patterns = pattern_result.data
            
        # Analyze market data using LLM
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
            prompt=input_message,
            model_name=self.config.get("models", {}).get("analysis")
        )
        
        try:
            analysis_result = json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse market analysis JSON for {company}")
            analysis_result = {"error": "Failed to parse market analysis"}
        
        self.market_data = {
            "price_data": price_data,
            "unusual_patterns": unusual_patterns,
            "analysis": analysis_result
        }
        
        return analysis_result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_corporate_report(self, company: str) -> Dict[str, Any]:
        """Generate comprehensive corporate report based on all collected data"""
        self.logger.info(f"Generating corporate report for {company}")
        
        # Combine all data
        combined_data = {
            "company_info": self.company_data,
            "financial_data": self.financial_data,
            "regulatory_data": self.regulatory_data,
            "management_data": self.management_data,
            "market_data": self.market_data,
        }
        
        # Generate report using LLM
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
            prompt=input_message,
            model_name=self.config.get("models", {}).get("report")
        )
        
        try:
            report = json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse corporate report JSON for {company}")
            report = {
                "error": "Failed to parse corporate report",
                "company": company,
                "timestamp": datetime.now().isoformat()
            }
        
        # Store report in database
        await self.postgres_tool.run(
            command="execute_query",
            query="INSERT INTO corporate_reports (company, report_date, report_data) VALUES ($1, $2, $3)",
            params=[company, datetime.now().isoformat(), json.dumps(report)]
        )
        
        return report
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the corporate agent workflow"""
        self._log_start(state)
        
        company = state.get("company", "")
        
        if not company:
            self.logger.error("Company name is missing!")
            return {**state, "goto": "meta_agent", "error": "Company name is missing", "corporate_status": "ERROR"}
        
        self.logger.info(f"Starting corporate analysis for {company}")
        
        # Initialize result structure
        corporate_results = {
            "company_info": {},
            "financial_analysis": {},
            "regulatory_analysis": {},
            "management_analysis": {},
            "market_analysis": {},
            "red_flags": [],
            "corporate_report": {}
        }
        
        # Fetch and analyze company information
        try:
            # Run tasks concurrently for efficiency
            company_info_task = self.fetch_company_info(company)
            financial_analysis_task = self.analyze_financial_statements(company)
            regulatory_analysis_task = self.check_regulatory_filings(company)
            management_analysis_task = self.analyze_management_team(company)
            market_analysis_task = self.analyze_market_data(company)
            
            # Wait for all tasks to complete
            company_info = await company_info_task
            financial_analysis = await financial_analysis_task
            regulatory_analysis = await regulatory_analysis_task
            management_analysis = await management_analysis_task
            market_analysis = await market_analysis_task
            
            # Store results
            corporate_results["company_info"] = company_info
            corporate_results["financial_analysis"] = financial_analysis
            corporate_results["regulatory_analysis"] = regulatory_analysis
            corporate_results["management_analysis"] = management_analysis
            corporate_results["market_analysis"] = market_analysis
            
            # Collect red flags from all analyses
            for analysis in [financial_analysis, regulatory_analysis, management_analysis, market_analysis]:
                if analysis and "red_flags" in analysis and isinstance(analysis["red_flags"], list):
                    corporate_results["red_flags"].extend(analysis["red_flags"])
            
            # Generate comprehensive corporate report
            corporate_report = await self.generate_corporate_report(company)
            corporate_results["corporate_report"] = corporate_report
            
            self.logger.info(f"Completed corporate analysis for {company} with {len(corporate_results['red_flags'])} red flags")
            
        except Exception as e:
            self.logger.error(f"Error during corporate analysis for {company}: {str(e)}")
            return {
                **state, 
                "goto": "meta_agent", 
                "error": f"Corporate analysis error: {str(e)}",
                "corporate_status": "ERROR"
            }
        
        # Update state with results
        state["corporate_results"] = corporate_results
        state["corporate_status"] = "DONE"
        
        # Determine next step based on configuration or results
        goto = "meta_agent"  # Default to meta_agent for orchestration
        if state.get("synchronous_pipeline", False):  # If running in synchronous mode
            goto = state.get("next_agent", "meta_agent")
            
        self._log_completion({**state, "goto": goto})
        return {**state, "goto": goto}