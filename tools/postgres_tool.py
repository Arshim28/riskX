import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncpg

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "forensic_db"
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: float = 10.0


class PostgresPool:
    """Singleton for managing PostgreSQL connection pool"""
    _instance = None
    _pool = None
    _config = None
    _lock = asyncio.Lock()
    _logger = get_logger("postgres_pool")

    @classmethod
    async def get_instance(cls, config: PostgresConfig) -> 'PostgresPool':
        """Get PostgresPool singleton instance"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
                    await cls._instance._initialize()
        return cls._instance

    def __init__(self, config: PostgresConfig):
        self._config = config

    async def _initialize(self) -> None:
        """Initialize the connection pool"""
        try:
            self._logger.info(f"Initializing PostgreSQL connection pool to {self._config.host}:{self._config.port}/{self._config.database}")
            
            # Get database password from environment if not provided in config
            password = self._config.password or os.environ.get("POSTGRES_PASSWORD", "")
            
            self._pool = await asyncpg.create_pool(
                host=self._config.host,
                port=self._config.port,
                user=self._config.user,
                password=password,
                database=self._config.database,
                min_size=self._config.min_connections,
                max_size=self._config.max_connections,
                timeout=self._config.connection_timeout,
                command_timeout=self._config.connection_timeout
            )
            
            self._logger.info("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            self._logger.error(f"Failed to initialize PostgreSQL connection pool: {str(e)}")
            raise

    @property
    def pool(self) -> asyncpg.Pool:
        """Get the connection pool"""
        if self._pool is None:
            raise RuntimeError("Connection pool not initialized. Call _initialize() first.")
        return self._pool

    async def close(self) -> None:
        """Close the connection pool"""
        if self._pool:
            self._logger.info("Closing PostgreSQL connection pool")
            await self._pool.close()
            self._pool = None
            self._instance = None


class PostgresTool(BaseTool):
    name = "postgres_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = PostgresConfig(**config)
        self.logger = get_logger(self.name)
        self.pool = None
        self.initialized = False
        
        # Track active transactions
        self._active_transactions = {}
        
        # Initialize standard schema if requested
        self._init_schema = config.get("init_schema", False)
    
    async def _ensure_initialized(self) -> None:
        """Ensure the database connection is initialized"""
        if not self.initialized:
            try:
                pool_instance = await PostgresPool.get_instance(self.config)
                self.pool = pool_instance.pool
                self.initialized = True
                
                if self._init_schema:
                    await self._initialize_schema()
            except Exception as e:
                self.logger.error(f"Failed to initialize PostgreSQL connection: {str(e)}")
                raise
    
    async def _initialize_schema(self) -> None:
        """Initialize the standard schema for forensic analysis"""
        schema_queries = [
            # Companies table
            """
            CREATE TABLE IF NOT EXISTS companies (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Regulatory filings
            """
            CREATE TABLE IF NOT EXISTS regulatory_filings (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                filing_date TIMESTAMP WITH TIME ZONE,
                filing_type TEXT,
                filing_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, filing_date, filing_type)
            )
            """,
            
            # Company management
            """
            CREATE TABLE IF NOT EXISTS company_management (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                name TEXT NOT NULL,
                position TEXT NOT NULL,
                data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, name, position)
            )
            """,
            
            # Forensic insights
            """
            CREATE TABLE IF NOT EXISTS forensic_insights (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                event_name TEXT,
                article_title TEXT NOT NULL,
                insights_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, article_title)
            )
            """,
            
            # Event synthesis
            """
            CREATE TABLE IF NOT EXISTS event_synthesis (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                event_name TEXT NOT NULL,
                synthesis_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, event_name)
            )
            """,
            
            # Company analysis
            """
            CREATE TABLE IF NOT EXISTS company_analysis (
                id SERIAL PRIMARY KEY,
                company TEXT UNIQUE NOT NULL,
                analysis_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Analysis results
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                report_date TIMESTAMP WITH TIME ZONE,
                analysis_data JSONB,
                red_flags JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Report templates
            """
            CREATE TABLE IF NOT EXISTS report_templates (
                id SERIAL PRIMARY KEY,
                template_name TEXT UNIQUE NOT NULL,
                sections JSONB NOT NULL,
                variables JSONB,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Report sections
            """
            CREATE TABLE IF NOT EXISTS report_sections (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                section_name TEXT NOT NULL,
                section_content TEXT,
                event_name TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, section_name, COALESCE(event_name, ''))
            )
            """,
            
            # Report revisions
            """
            CREATE TABLE IF NOT EXISTS report_revisions (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                section_name TEXT NOT NULL,
                revision_id TEXT NOT NULL,
                original_content TEXT,
                revised_content TEXT,
                feedback TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Final reports
            """
            CREATE TABLE IF NOT EXISTS final_reports (
                id SERIAL PRIMARY KEY,
                company TEXT UNIQUE NOT NULL,
                report_date TIMESTAMP WITH TIME ZONE,
                report_content TEXT,
                filename TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Executive briefings
            """
            CREATE TABLE IF NOT EXISTS executive_briefings (
                id SERIAL PRIMARY KEY,
                company TEXT UNIQUE NOT NULL,
                briefing_date TIMESTAMP WITH TIME ZONE,
                briefing_content TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Report feedback
            """
            CREATE TABLE IF NOT EXISTS report_feedback (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                feedback_date TIMESTAMP WITH TIME ZONE,
                feedback_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Corporate reports
            """
            CREATE TABLE IF NOT EXISTS corporate_reports (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                report_date TIMESTAMP WITH TIME ZONE,
                report_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Workflow status
            """
            CREATE TABLE IF NOT EXISTS workflow_status (
                id SERIAL PRIMARY KEY,
                company TEXT UNIQUE NOT NULL,
                status_data JSONB,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        async with self.pool.acquire() as conn:
            for query in schema_queries:
                try:
                    await conn.execute(query)
                except Exception as e:
                    self.logger.error(f"Error creating schema: {str(e)}")
                    raise
                    
        self.logger.info("Schema initialization completed successfully")
    
    async def connect(self) -> None:
        """Connect to the PostgreSQL database"""
        await self._ensure_initialized()
    
    async def execute_query(self, query: str, params: Optional[List[Any]] = None) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """Execute a SQL query with parameters and return results"""
        await self._ensure_initialized()
        
        try:
            async with self.pool.acquire() as conn:
                if query.strip().upper().startswith(("SELECT", "WITH")):
                    # It's a SELECT query, return results
                    if params:
                        rows = await conn.fetch(query, *params)
                    else:
                        rows = await conn.fetch(query)
                        
                    # Convert to list of dicts
                    result = [dict(row) for row in rows]
                    return True, result, None
                else:
                    # It's a non-SELECT query (INSERT, UPDATE, DELETE, etc.)
                    if params:
                        await conn.execute(query, *params)
                    else:
                        await conn.execute(query)
                    return True, None, None
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    async def fetch_one(self, query: str, params: Optional[List[Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Fetch a single row from the database"""
        await self._ensure_initialized()
        
        try:
            async with self.pool.acquire() as conn:
                if params:
                    row = await conn.fetchrow(query, *params)
                else:
                    row = await conn.fetchrow(query)
                    
                if row:
                    result = dict(row)
                    return True, result, None
                else:
                    return True, None, None
        except Exception as e:
            error_msg = f"Error fetching row: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    async def fetch_all(self, query: str, params: Optional[List[Any]] = None) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """Fetch all rows from the database"""
        return await self.execute_query(query, params)
    
    async def execute_batch(self, query: str, params_list: List[List[Any]]) -> Tuple[bool, Optional[List[int]], Optional[str]]:
        """Execute a batch of queries with different parameters"""
        await self._ensure_initialized()
        
        try:
            results = []
            async with self.pool.acquire() as conn:
                # Create a prepared statement for efficiency
                stmt = await conn.prepare(query)
                
                for params in params_list:
                    result = await stmt.execute(*params)
                    if isinstance(result, str) and result.startswith("INSERT") and "RETURNING" in query.upper():
                        # Extract the returned ID from the result string
                        try:
                            id_str = result.split()[-1]
                            results.append(int(id_str))
                        except (IndexError, ValueError):
                            results.append(None)
                
            return True, results, None
        except Exception as e:
            error_msg = f"Error executing batch query: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    async def begin_transaction(self, transaction_id: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """Begin a new transaction"""
        await self._ensure_initialized()
        
        # Generate a random transaction ID if not provided
        if not transaction_id:
            import uuid
            transaction_id = str(uuid.uuid4())
        
        try:
            if transaction_id in self._active_transactions:
                return False, None, f"Transaction {transaction_id} already exists"
                
            conn = await self.pool.acquire()
            tr = conn.transaction()
            await tr.start()
            
            self._active_transactions[transaction_id] = {
                "connection": conn,
                "transaction": tr
            }
            
            return True, transaction_id, None
        except Exception as e:
            error_msg = f"Error beginning transaction: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    async def commit(self, transaction_id: str) -> Tuple[bool, Optional[str]]:
        """Commit an active transaction"""
        if transaction_id not in self._active_transactions:
            return False, f"Transaction {transaction_id} not found"
            
        try:
            tr_data = self._active_transactions[transaction_id]
            await tr_data["transaction"].commit()
            await self.pool.release(tr_data["connection"])
            del self._active_transactions[transaction_id]
            return True, None
        except Exception as e:
            error_msg = f"Error committing transaction: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def rollback(self, transaction_id: str) -> Tuple[bool, Optional[str]]:
        """Rollback an active transaction"""
        if transaction_id not in self._active_transactions:
            return False, f"Transaction {transaction_id} not found"
            
        try:
            tr_data = self._active_transactions[transaction_id]
            await tr_data["transaction"].rollback()
            await self.pool.release(tr_data["connection"])
            del self._active_transactions[transaction_id]
            return True, None
        except Exception as e:
            error_msg = f"Error rolling back transaction: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def close(self) -> None:
        """Close all connections"""
        if self.initialized:
            # Rollback any active transactions
            for transaction_id in list(self._active_transactions.keys()):
                await self.rollback(transaction_id)
                
            # Close the pool
            self.initialized = False
            await PostgresPool.get_instance(self.config).close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def run(self, command: str, **kwargs) -> ToolResult[Any]:
        """Run a PostgreSQL operation"""
        try:
            await self._ensure_initialized()
            
            result = None
            
            if command == "execute_query":
                query = kwargs.get("query")
                params = kwargs.get("params")
                
                if not query:
                    return ToolResult(success=False, error="Query is required for execute_query command")
                
                success, data, error = await self.execute_query(query, params)
                if success:
                    result = data
                else:
                    return ToolResult(success=False, error=error)
                    
            elif command == "fetch_one":
                query = kwargs.get("query")
                params = kwargs.get("params")
                
                if not query:
                    return ToolResult(success=False, error="Query is required for fetch_one command")
                
                success, data, error = await self.fetch_one(query, params)
                if success:
                    result = data
                else:
                    return ToolResult(success=False, error=error)
                    
            elif command == "fetch_all":
                query = kwargs.get("query")
                params = kwargs.get("params")
                
                if not query:
                    return ToolResult(success=False, error="Query is required for fetch_all command")
                
                success, data, error = await self.fetch_all(query, params)
                if success:
                    result = data
                else:
                    return ToolResult(success=False, error=error)
                    
            elif command == "execute_batch":
                query = kwargs.get("query")
                params_list = kwargs.get("params_list")
                
                if not query or not params_list:
                    return ToolResult(success=False, error="Query and params_list are required for execute_batch command")
                
                success, data, error = await self.execute_batch(query, params_list)
                if success:
                    result = data
                else:
                    return ToolResult(success=False, error=error)
                    
            elif command == "begin_transaction":
                transaction_id = kwargs.get("transaction_id")
                
                success, tx_id, error = await self.begin_transaction(transaction_id)
                if success:
                    result = {"transaction_id": tx_id}
                else:
                    return ToolResult(success=False, error=error)
                    
            elif command == "commit":
                transaction_id = kwargs.get("transaction_id")
                
                if not transaction_id:
                    return ToolResult(success=False, error="Transaction ID is required for commit command")
                
                success, error = await self.commit(transaction_id)
                if success:
                    result = {"committed": True}
                else:
                    return ToolResult(success=False, error=error)
                    
            elif command == "rollback":
                transaction_id = kwargs.get("transaction_id")
                
                if not transaction_id:
                    return ToolResult(success=False, error="Transaction ID is required for rollback command")
                
                success, error = await self.rollback(transaction_id)
                if success:
                    result = {"rolled_back": True}
                else:
                    return ToolResult(success=False, error=error)
                    
            elif command == "close":
                await self.close()
                result = {"closed": True}
                
            elif command == "ping":
                # Simple ping to check if database is accessible
                success, _, error = await self.execute_query("SELECT 1")
                if success:
                    result = {"ping": "successful"}
                else:
                    return ToolResult(success=False, error=error)
                    
            elif command == "init_schema":
                # Force schema initialization
                await self._initialize_schema()
                result = {"schema_initialized": True}
                
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
                
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            error_msg = f"PostgreSQL tool error: {str(e)}"
            self.logger.error(error_msg)
            return await self._handle_error(e)