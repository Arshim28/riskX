# agents/corporate_agent.py
from typing import Dict, List, Any, Optional, Tuple
import json
import asyncio
from datetime import datetime
import traceback
import logging
import os
import yaml
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type,
    RetryError,
    before_sleep_log
)

from base.base_agents import BaseAgent, AgentState
from utils.llm_provider import get_llm_provider
from utils.prompt_manager import get_prompt_manager
from utils.logging import get_logger
from tools.postgres_tool import PostgresTool
from tools.nse_tool import NSETool, NSEConnectionError, NSERateLimitError, NSEDataError


class CorporateGovernanceError(Exception):
    pass


class CorporateAgent(BaseAgent):
    name = "corporate_agent"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.prompt_manager = get_prompt_manager()
        
        self.postgres_tool = PostgresTool(config.get("postgres", {}))
        self.nse_tool = NSETool(config.get("nse", {}))
        
        self.default_date_params = {
            "from_date": (datetime.now().replace(year=datetime.now().year-1)).strftime("%d-%m-%Y"),
            "to_date": datetime.now().strftime("%d-%m-%Y")
        }
        
        self.retry_attempts = config.get("retry", {}).get("max_attempts", 3)
        self.retry_multiplier = config.get("retry", {}).get("multiplier", 1)
        self.retry_min_wait = config.get("retry", {}).get("min_wait", 2)
        self.retry_max_wait = config.get("retry", {}).get("max_wait", 10)
    
    async def _get_symbol_from_config(self) -> Optional[str]:
        """Get symbol from NSE tool config if available"""
        if (hasattr(self.nse_tool, 'config') and 
            hasattr(self.nse_tool.config, 'symbol') and 
            self.nse_tool.config.symbol):
            return self.nse_tool.config.symbol
        return None
        
    async def _get_symbol_from_database(self, company_name: str) -> Optional[str]:
        """Try to retrieve company symbol from database"""
        try:
            query = "SELECT symbol FROM nse_metadata WHERE name = $1"
            result = await self.postgres_tool.run(
                command="execute_query",
                query=query,
                params=[company_name]
            )
            
            if result.success and result.data and len(result.data) > 0:
                symbol = result.data[0].get('symbol')
                if symbol:
                    return symbol
            return None
        except Exception as e:
            self.logger.error(f"Database error retrieving symbol for {company_name}: {str(e)}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_company_symbol(self, company_name: str) -> str:
        """Get company symbol with multiple fallback mechanisms"""
        # Try to get symbol from config first
        symbol = await self._get_symbol_from_config()
        if symbol:
            self.logger.info(f"Using symbol {symbol} from NSE tool config for company: {company_name}")
            return symbol
        
        self.logger.info(f"Looking up symbol for company: {company_name}")
        
        # Try to get symbol from database
        symbol = await self._get_symbol_from_database(company_name)
        if symbol:
            self.logger.info(f"Found symbol {symbol} for company: {company_name}")
            return symbol
        
        # Fallback to using company name as symbol
        self.logger.warning(f"Symbol not found for company: {company_name}, using company name as symbol")
        return company_name
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_corporate_governance_data(self, symbol: str) -> Dict[str, Any]:
        try:
            governance_file = os.path.join("assets", "corporate_governance.json")
            if not os.path.exists(governance_file):
                self.logger.error(f"Corporate governance file not found at {governance_file}")
                return {}
                
            with open(governance_file, 'r') as file:
                data = json.load(file)
                
            company_data = data.get(symbol, {})
            if not company_data:
                self.logger.warning(f"No corporate governance data found for symbol: {symbol}")
                
            return company_data
        except Exception as e:
            error_msg = f"Unable to get corporate governance data for company with symbol: {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise CorporateGovernanceError(error_msg)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def collect_corporate_data(self, company: str, stream_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.info(f"Collecting corporate governance data for {company} using {len(stream_config)} streams")
        
        try:
            company_symbol = await self.get_company_symbol(company)
            
            governance_data = await self.get_corporate_governance_data(company_symbol)
            
            result = await self.nse_tool.run(
                command="fetch_multiple",
                stream_dict=stream_config
            )
            
            if not result.success:
                error_msg = f"Failed to collect corporate data for {company}: {result.error}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "company": company,
                    "symbol": company_symbol,
                    "governance": governance_data,
                    "timestamp": datetime.now().isoformat()
                }
            
            corporate_data = {
                "success": True,
                "company": company,
                "symbol": company_symbol,
                "timestamp": datetime.now().isoformat(),
                "governance": governance_data,
                "data": result.data,
                "summary": {
                    "total_streams": len(stream_config),
                    "stream_counts": {stream: len(data) for stream, data in result.data.items()}
                }
            }
            
            self.logger.info(f"Successfully collected corporate data for {company} from {len(result.data)} streams")
            for stream, items in corporate_data["summary"]["stream_counts"].items():
                self.logger.info(f"  - {stream}: {items} items")
            
            return corporate_data
            
        except Exception as e:
            error_msg = f"Error collecting corporate data for {company}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "company": company,
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_stream_config(self, config: Dict[str, Dict[str, Any]]) -> bool:
        """Validate if the stream config has the required fields"""
        if not config:
            return False
            
        # Check if config has at least one stream with required fields
        for stream, stream_config in config.items():
            if not isinstance(stream_config, dict):
                self.logger.warning(f"Invalid stream config for {stream}: not a dictionary")
                continue
                
            # Check if stream has 'active' field as minimum requirement
            if 'active' not in stream_config:
                self.logger.warning(f"Stream config for {stream} missing 'active' field")
                continue
                
            # Found at least one valid stream
            return True
            
        return False
        
    def _apply_date_params(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Apply default date parameters to streams that need them"""
        for stream, stream_config in config.items():
            # Check if stream config has input_params
            if 'input_params' in stream_config and stream_config['input_params']:
                # Check if any date parameters are needed
                params = stream_config['input_params']
                if any(param in params for param in ['from_date', 'to_date']):
                    # Apply default date parameters
                    stream_config['input_params'] = self.default_date_params
                    
        return config
    
    def _get_default_stream_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default stream configuration with validation and date parameter handling"""
        default_config_path = 'assets/default_stream_config.yaml'
        
        if not os.path.exists(default_config_path):
            self.logger.error(f"Default stream config file not found at {default_config_path}")
            return {}
            
        try:
            with open(default_config_path, 'r') as file:
                config = yaml.safe_load(file)
                
                # Validate the config
                if not self._validate_stream_config(config):
                    self.logger.error("Stream config validation failed, using empty config")
                    return {}
                
                # Apply date parameters to streams that need them
                config = self._apply_date_params(config)
                
                self.logger.info(f"Loaded default stream config from {default_config_path} with {len(config)} streams")
                return config
                
        except Exception as e:
            self.logger.error(f"Failed to load default stream config from file: {str(e)}")
            return {}
    
    async def _execute(self, state: AgentState) -> Dict[str, Any]:
        return await self.run(state.to_dict())

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._log_start(state)
            
            company = state.get("company", "")
            
            if not company:
                self.logger.error("Company name is missing!")
                return {**state, "goto": "meta_agent", "corporate_status": "ERROR", "error": "Company name is missing"}
            
            self.logger.info(f"Starting corporate governance data collection for {company}")
            
            if hasattr(self.nse_tool, 'config'):
                self.nse_tool.config.company = company
                
                symbol = state.get("symbol")
                if symbol:
                    self.nse_tool.config.symbol = symbol
            
            stream_config = state.get("corporate_stream_config", self._get_default_stream_config())
            
            corporate_results = await self.collect_corporate_data(company, stream_config)
            
            state["corporate_results"] = corporate_results
            state["corporate_status"] = "DONE" if corporate_results.get("success", False) else "ERROR"
            
            if not corporate_results.get("success", False):
                state["error"] = corporate_results.get("error", "Unknown error in corporate data collection")
            
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