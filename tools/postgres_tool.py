import os
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncpg
import hashlib

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "forensic_db"
    min_connections: int = 2
    max_connections: int = 20
    connection_timeout: float = 10.0
    query_timeout: float = 30.0
    prepared_statement_cache_size: int = 100
    health_check_interval: float = 30.0


class QueryStats:
    """Track query performance statistics"""
    
    def __init__(self):
        self.query_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.errors = 0
        self.query_types = {"SELECT": 0, "INSERT": 0, "UPDATE": 0, "DELETE": 0, "OTHER": 0}
        
    def record_query(self, query: str, execution_time: float, success: bool = True):
        self.query_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time) if success else self.min_time
        self.max_time = max(self.max_time, execution_time) if success else self.max_time
        
        if not success:
            self.errors += 1
        
        # Categorize query type
        query_type = query.strip().split(' ')[0].upper() if query.strip() else "OTHER"
        if query_type in self.query_types:
            self.query_types[query_type] += 1
        else:
            self.query_types["OTHER"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "query_count": self.query_count,
            "avg_time": self.total_time / max(1, self.query_count),
            "min_time": self.min_time if self.min_time != float('inf') else 0,
            "max_time": self.max_time,
            "error_rate": self.errors / max(1, self.query_count),
            "query_types": self.query_types
        }


class PreparedStatementCache:
    """Cache for prepared statements to avoid repeated preparation"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[Tuple[asyncpg.Connection, str], Any] = {}
        self.usage_count: Dict[Tuple[asyncpg.Connection, str], int] = {}
        self.last_used: Dict[Tuple[asyncpg.Connection, str], float] = {}
        self.logger = get_logger("prepared_stmt_cache")
    
    def get(self, connection: asyncpg.Connection, query: str) -> Optional[Any]:
        key = (connection, query)
        if key in self.cache:
            self.usage_count[key] += 1
            self.last_used[key] = time.time()
            return self.cache[key]
        return None
    
    async def add(self, connection: asyncpg.Connection, query: str, statement: Any) -> None:
        key = (connection, query)
        
        # If cache is full, remove least recently used entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.last_used.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.usage_count[oldest_key]
            del self.last_used[oldest_key]
            self.logger.debug(f"Cache full, removed oldest statement from cache")
        
        self.cache[key] = statement
        self.usage_count[key] = 1
        self.last_used[key] = time.time()
    
    def remove_for_connection(self, connection: asyncpg.Connection) -> None:
        """Remove all cached statements for a specific connection"""
        keys_to_remove = [key for key in self.cache if key[0] == connection]
        for key in keys_to_remove:
            del self.cache[key]
            del self.usage_count[key]
            del self.last_used[key]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": sum(self.usage_count.values()) / max(1, len(self.cache)),
            "most_used_queries": sorted(
                [(query, count) for (_, query), count in self.usage_count.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


class PostgresPool:
    """Enhanced connection pool for PostgreSQL with health checking and statistics"""
    
    def __init__(self, config: PostgresConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger("postgres_pool")
        self.pool = None
        self._lock = asyncio.Lock()
        self._health_check_task = None
        self._terminating = False
        
        # Statistics tracking
        self.stats = {
            "connections_created": 0,
            "connections_closed": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "acquire_count": 0,
            "acquire_wait_time": 0.0,
            "max_wait_time": 0.0
        }
        
        # Track active transactions
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        
        # Create prepared statement cache
        self.stmt_cache = PreparedStatementCache(max_size=config.prepared_statement_cache_size)
    
    async def initialize(self) -> None:
        """Initialize the connection pool"""
        if self.pool:
            self.logger.info("Pool already initialized")
            return
            
        async with self._lock:
            if self.pool:  # Check again in case another task initialized it
                return
                
            try:
                self.logger.info(f"Initializing PostgreSQL connection pool to {self.config.host}:{self.config.port}/{self.config.database}")
                
                # Get database password from environment if not provided in config
                password = self.config.password or os.environ.get("POSTGRES_PASSWORD", "")
                
                # Define connection setup function to configure each new connection
                async def setup_connection(conn):
                    # Set session parameters for improved performance
                    await conn.execute("SET statement_timeout = $1", int(self.config.query_timeout * 1000))
                    
                    # Track connection creation
                    self.stats["connections_created"] += 1
                    self.stats["active_connections"] += 1
                
                # Create the connection pool with connection lifecycle hooks
                self.pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.user,
                    password=password,
                    database=self.config.database,
                    min_size=self.config.min_connections,
                    max_size=self.config.max_connections,
                    timeout=self.config.connection_timeout,
                    command_timeout=self.config.query_timeout,
                    setup=setup_connection,
                    max_inactive_connection_lifetime=300.0  # 5 minutes
                )
                
                # Start health check task
                self._start_health_check()
                
                self.logger.info("PostgreSQL connection pool initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize PostgreSQL connection pool: {str(e)}")
                raise
    
    async def _health_check(self) -> None:
        """Periodic health check of the connection pool"""
        while not self._terminating and self.pool:
            try:
                # Test a connection from the pool
                async with self.pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                    
                    # Update pool statistics
                    self.stats["idle_connections"] = self.pool.get_idle_size()
                    self.stats["active_connections"] = self.pool.get_size() - self.pool.get_idle_size()
                    
                    # Log long-running transactions
                    now = time.time()
                    long_running = {
                        tx_id: data for tx_id, data in self.active_transactions.items() 
                        if now - data["start_time"] > 60  # Transactions running > 60 seconds
                    }
                    if long_running:
                        self.logger.warning(f"Long-running transactions detected: {len(long_running)}")
                        for tx_id, data in long_running.items():
                            duration = now - data["start_time"]
                            self.logger.warning(f"Transaction {tx_id} running for {duration:.1f}s")
                
                self.logger.debug(f"Health check passed. Connections: active={self.stats['active_connections']}, idle={self.stats['idle_connections']}")
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                
                # Try to recreate the pool if health check consistently fails
                if str(e).lower().find("pool is closed") >= 0:
                    self.logger.warning("Pool is closed, attempting to reinitialize")
                    self.pool = None
                    try:
                        await self.initialize()
                        self.logger.info("Pool reinitialized successfully")
                    except Exception as init_error:
                        self.logger.error(f"Failed to reinitialize pool: {str(init_error)}")
            
            # Wait for next health check
            await asyncio.sleep(self.config.health_check_interval)
    
    def _start_health_check(self) -> None:
        """Start the health check background task"""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check())
    
    async def acquire(self) -> asyncpg.Connection:
        """Acquire a connection from the pool with statistics tracking"""
        if not self.pool:
            await self.initialize()
            
        start_time = time.time()
        conn = await self.pool.acquire()
        acquire_time = time.time() - start_time
        
        # Update statistics
        self.stats["acquire_count"] += 1
        self.stats["acquire_wait_time"] += acquire_time
        self.stats["max_wait_time"] = max(self.stats["max_wait_time"], acquire_time)
        
        if acquire_time > 1.0:  # Log slow acquisitions
            self.logger.warning(f"Slow connection acquisition: {acquire_time:.3f}s")
        
        return conn
    
    async def release(self, conn: asyncpg.Connection) -> None:
        """Release a connection back to the pool"""
        if self.pool:
            try:
                # Remove any cached prepared statements for this connection
                self.stmt_cache.remove_for_connection(conn)
                
                # Release the connection
                await self.pool.release(conn)
                
                # Update stats
                self.stats["idle_connections"] = self.pool.get_idle_size()
                self.stats["active_connections"] = self.pool.get_size() - self.pool.get_idle_size()
                
            except Exception as e:
                self.logger.error(f"Error releasing connection: {str(e)}")
    
    async def close(self) -> None:
        """Close the connection pool"""
        if self.pool:
            self._terminating = True
            
            # Cancel health check task
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Close the pool
            await self.pool.close()
            self.pool = None
            self._terminating = False
            
            # Update stats
            self.stats["connections_closed"] += self.stats["active_connections"] + self.stats["idle_connections"]
            self.stats["active_connections"] = 0
            self.stats["idle_connections"] = 0
            
            self.logger.info("PostgreSQL connection pool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection pool"""
        if self.pool:
            self.stats["idle_connections"] = self.pool.get_idle_size()
            self.stats["active_connections"] = self.pool.get_size() - self.pool.get_idle_size()
            
            # Calculate average acquisition time
            if self.stats["acquire_count"] > 0:
                self.stats["avg_acquire_time"] = self.stats["acquire_wait_time"] / self.stats["acquire_count"]
            else:
                self.stats["avg_acquire_time"] = 0
                
            # Add prepared statement cache stats
            self.stats["stmt_cache"] = self.stmt_cache.get_stats()
        
        return self.stats


class PostgresTool(BaseTool):
    name = "postgres_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = PostgresConfig(**config)
        self.logger = get_logger(self.name)
        self.pool = None
        self.initialized = False
        
        # Track transactions
        self.active_transactions = {}
        
        # Query statistics
        self.query_stats = QueryStats()
        
        # Query plan cache
        self.query_plan_cache = {}
        
        # Initialize standard schema if requested
        self._init_schema = config.get("init_schema", False)
    
    async def _ensure_initialized(self) -> None:
        """Ensure the database connection is initialized"""
        if not self.initialized:
            try:
                self.pool = PostgresPool(self.config, self.logger)
                await self.pool.initialize()
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
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Add indexes on commonly queried columns
            """
            CREATE INDEX IF NOT EXISTS idx_companies_name ON companies (name);
            """,
            
            # Create updated_at trigger function
            """
            CREATE OR REPLACE FUNCTION update_modified_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = now();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
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
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, filing_date, filing_type)
            )
            """,
            
            # Add indexes for regulatory filings
            """
            CREATE INDEX IF NOT EXISTS idx_regulatory_filings_company ON regulatory_filings (company);
            CREATE INDEX IF NOT EXISTS idx_regulatory_filings_date ON regulatory_filings (filing_date);
            CREATE INDEX IF NOT EXISTS idx_regulatory_filings_type ON regulatory_filings (filing_type);
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
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, name, position)
            )
            """,
            
            # Add indexes for company management
            """
            CREATE INDEX IF NOT EXISTS idx_company_management_company ON company_management (company);
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
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, article_title)
            )
            """,
            
            # Add indexes for forensic insights
            """
            CREATE INDEX IF NOT EXISTS idx_forensic_insights_company ON forensic_insights (company);
            CREATE INDEX IF NOT EXISTS idx_forensic_insights_event ON forensic_insights (event_name);
            """,
            
            # Event synthesis
            """
            CREATE TABLE IF NOT EXISTS event_synthesis (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                event_name TEXT NOT NULL,
                synthesis_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, event_name)
            )
            """,
            
            # Add indexes for event synthesis
            """
            CREATE INDEX IF NOT EXISTS idx_event_synthesis_company ON event_synthesis (company);
            """,
            
            # Company analysis
            """
            CREATE TABLE IF NOT EXISTS company_analysis (
                id SERIAL PRIMARY KEY,
                company TEXT UNIQUE NOT NULL,
                analysis_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
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
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Add indexes for analysis results
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_results_company ON analysis_results (company);
            """,
            
            # Report templates
            """
            CREATE TABLE IF NOT EXISTS report_templates (
                id SERIAL PRIMARY KEY,
                template_name TEXT UNIQUE NOT NULL,
                sections JSONB NOT NULL,
                variables JSONB,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
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
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, section_name, COALESCE(event_name, ''))
            )
            """,
            
            # Add indexes for report sections
            """
            CREATE INDEX IF NOT EXISTS idx_report_sections_company ON report_sections (company);
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
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Executive briefings
            """
            CREATE TABLE IF NOT EXISTS executive_briefings (
                id SERIAL PRIMARY KEY,
                company TEXT UNIQUE NOT NULL,
                briefing_date TIMESTAMP WITH TIME ZONE,
                briefing_content TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
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
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Add indexes for corporate reports
            """
            CREATE INDEX IF NOT EXISTS idx_corporate_reports_company ON corporate_reports (company);
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
    
    def _generate_query_hash(self, query: str, params: Optional[List[Any]] = None) -> str:
        """Generate a unique hash for a query and its parameters"""
        param_str = json.dumps(params) if params else ""
        hash_input = f"{query}:{param_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    async def _prepare_statement(self, conn: asyncpg.Connection, query: str) -> Any:
        """Get or create a prepared statement"""
        # Check if we have this statement in cache
        stmt = self.pool.stmt_cache.get(conn, query)
        if stmt:
            return stmt
            
        # Prepare the statement and add to cache
        stmt = await conn.prepare(query)
        await self.pool.stmt_cache.add(conn, query, stmt)
        return stmt
    
    async def _optimize_query(self, query: str) -> str:
        """Apply simple query optimizations"""
        # This is a simplified optimization - in a real system, you would
        # implement more sophisticated query analysis and optimization
        
        # Convert to lowercase for easier pattern matching
        lowercase_query = query.lower()
        
        # Add simple optimizations
        if "select" in lowercase_query and "where" in lowercase_query:
            # If selecting all columns without a clear need, suggest optimizing
            if "select *" in lowercase_query and not "group by" in lowercase_query:
                self.logger.info("Query optimization: 'SELECT *' detected, consider selecting only needed columns")
                
            # Check for missing indexes in WHERE clause
            if "where" in lowercase_query and not "index" in self.query_plan_cache.get(query, ""):
                self.logger.debug("Query may benefit from an index on WHERE clause columns")
        
        # For large result sets, add LIMIT if not present
        if "select" in lowercase_query and not any(x in lowercase_query for x in ["limit", "fetch first"]):
            if not "where" in lowercase_query or "like '%%" in lowercase_query:
                self.logger.debug("Adding LIMIT to potentially large result set query")
                if ";" in query:
                    return query.replace(";", " LIMIT 1000;")
                else:
                    return query + " LIMIT 1000"
        
        return query
    
    async def _get_query_plan(self, conn: asyncpg.Connection, query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Get the execution plan for a query"""
        query_hash = self._generate_query_hash(query, params)
        
        # Check if we have this plan cached
        if query_hash in self.query_plan_cache:
            return self.query_plan_cache[query_hash]
        
        # If it's a SELECT query, get EXPLAIN output
        if query.strip().upper().startswith("SELECT"):
            try:
                # Create EXPLAIN query
                explain_query = f"EXPLAIN (FORMAT JSON) {query}"
                
                if params:
                    plan_rows = await conn.fetch(explain_query, *params)
                else:
                    plan_rows = await conn.fetch(explain_query)
                
                if plan_rows and plan_rows[0]:
                    # Parse explain output
                    plan = plan_rows[0][0][0]
                    
                    # Cache the plan
                    self.query_plan_cache[query_hash] = plan
                    
                    return plan
            except Exception as e:
                self.logger.warning(f"Error getting query plan: {str(e)}")
        
        return {}
    
    async def connect(self) -> None:
        """Connect to the PostgreSQL database"""
        await self._ensure_initialized()
    
    async def execute_query(self, query: str, params: Optional[List[Any]] = None) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """Execute a SQL query with parameters and return results"""
        await self._ensure_initialized()
        
        start_time = time.time()
        success = False
        
        try:
            # Optimize query if possible
            optimized_query = await self._optimize_query(query)
            
            async with self.pool.acquire() as conn:
                # Get query plan for monitoring and optimization
                await self._get_query_plan(conn, optimized_query, params)
                
                # Use prepared statement for efficiency
                stmt = await self._prepare_statement(conn, optimized_query)
                
                if optimized_query.strip().upper().startswith(("SELECT", "WITH")):
                    # It's a SELECT query, return results
                    if params:
                        rows = await stmt.fetch(*params)
                    else:
                        rows = await stmt.fetch()
                        
                    # Convert to list of dicts
                    result = [dict(row) for row in rows]
                    success = True
                    return True, result, None
                else:
                    # It's a non-SELECT query (INSERT, UPDATE, DELETE, etc.)
                    if params:
                        await stmt.execute(*params)
                    else:
                        await stmt.execute()
                    success = True
                    return True, None, None
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
        finally:
            # Record query statistics
            execution_time = time.time() - start_time
            self.query_stats.record_query(query, execution_time, success)
            
            # Log slow queries
            if execution_time > 1.0:  # Log queries taking more than 1 second
                self.logger.warning(f"Slow query detected: {execution_time:.3f}s for {query[:100]}...")
    
    async def fetch_one(self, query: str, params: Optional[List[Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Fetch a single row from the database"""
        await self._ensure_initialized()
        
        start_time = time.time()
        success = False
        
        try:
            # Optimize query if possible
            optimized_query = await self._optimize_query(query)
            
            async with self.pool.acquire() as conn:
                # Use prepared statement for efficiency
                stmt = await self._prepare_statement(conn, optimized_query)
                
                if params:
                    row = await stmt.fetchrow(*params)
                else:
                    row = await stmt.fetchrow()
                    
                if row:
                    result = dict(row)
                    success = True
                    return True, result, None
                else:
                    success = True
                    return True, None, None
        except Exception as e:
            error_msg = f"Error fetching row: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
        finally:
            # Record query statistics
            execution_time = time.time() - start_time
            self.query_stats.record_query(query, execution_time, success)
    
    async def fetch_all(self, query: str, params: Optional[List[Any]] = None) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """Fetch all rows from the database"""
        return await self.execute_query(query, params)
    
    async def execute_batch(self, query: str, params_list: List[List[Any]]) -> Tuple[bool, Optional[List[int]], Optional[str]]:
        """Execute a batch of queries with different parameters"""
        await self._ensure_initialized()
        
        start_time = time.time()
        success = False
        
        try:
            results = []
            async with self.pool.acquire() as conn:
                # Create a prepared statement for efficiency
                stmt = await self._prepare_statement(conn, query)
                
                # Start a transaction for the batch
                async with conn.transaction():
                    for params in params_list:
                        result = await stmt.execute(*params)
                        if isinstance(result, str) and result.startswith("INSERT") and "RETURNING" in query.upper():
                            # Extract the returned ID from the result string
                            try:
                                id_str = result.split()[-1]
                                results.append(int(id_str))
                            except (IndexError, ValueError):
                                results.append(None)
                
            success = True
            return True, results, None
        except Exception as e:
            error_msg = f"Error executing batch query: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
        finally:
            # Record query statistics
            execution_time = time.time() - start_time
            self.query_stats.record_query(f"BATCH: {query}", execution_time, success)
    
    async def begin_transaction(self, transaction_id: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """Begin a new transaction"""
        await self._ensure_initialized()
        
        # Generate a random transaction ID if not provided
        if not transaction_id:
            import uuid
            transaction_id = str(uuid.uuid4())
        
        try:
            if transaction_id in self.active_transactions:
                return False, None, f"Transaction {transaction_id} already exists"
                
            conn = await self.pool.acquire()
            tr = conn.transaction()
            await tr.start()
            
            # Record transaction start time and details
            start_time = time.time()
            self.active_transactions[transaction_id] = {
                "connection": conn,
                "transaction": tr,
                "start_time": start_time,
                "queries": [],
                "is_readonly": False
            }
            
            # Add transaction to pool's tracking
            self.pool.active_transactions[transaction_id] = {
                "start_time": start_time,
                "connection_id": id(conn)
            }
            
            self.logger.debug(f"Transaction {transaction_id} started")
            return True, transaction_id, None
        except Exception as e:
            error_msg = f"Error beginning transaction: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    async def execute_in_transaction(self, transaction_id: str, query: str, params: Optional[List[Any]] = None) -> Tuple[bool, Optional[Any], Optional[str]]:
        """Execute a query within an existing transaction"""
        if transaction_id not in self.active_transactions:
            return False, None, f"Transaction {transaction_id} not found"
            
        start_time = time.time()
        success = False
        
        try:
            tr_data = self.active_transactions[transaction_id]
            conn = tr_data["connection"]
            
            # Optimize query if possible
            optimized_query = await self._optimize_query(query)
            
            # Prepare statement
            stmt = await self._prepare_statement(conn, optimized_query)
            
            # Record query in transaction history
            tr_data["queries"].append({
                "query": query,
                "params": params,
                "time": time.time()
            })
            
            # Check if this is a read-only operation
            is_read = optimized_query.strip().upper().startswith(("SELECT", "WITH"))
            if not is_read:
                tr_data["is_readonly"] = False
            
            # Execute query
            if is_read:
                if params:
                    rows = await stmt.fetch(*params)
                else:
                    rows = await stmt.fetch()
                    
                # Convert to list of dicts
                result = [dict(row) for row in rows]
                success = True
                return True, result, None
            else:
                if params:
                    await stmt.execute(*params)
                else:
                    await stmt.execute()
                success = True
                return True, None, None
                
        except Exception as e:
            error_msg = f"Error executing query in transaction: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
        finally:
            # Record query statistics
            execution_time = time.time() - start_time
            self.query_stats.record_query(query, execution_time, success)
    
    async def commit(self, transaction_id: str) -> Tuple[bool, Optional[str]]:
        """Commit an active transaction"""
        if transaction_id not in self.active_transactions:
            return False, f"Transaction {transaction_id} not found"
            
        try:
            tr_data = self.active_transactions[transaction_id]
            
            # Log transaction details before commit
            duration = time.time() - tr_data["start_time"]
            query_count = len(tr_data["queries"])
            self.logger.debug(f"Committing transaction {transaction_id} after {duration:.3f}s with {query_count} queries")
            
            await tr_data["transaction"].commit()
            await self.pool.release(tr_data["connection"])
            
            # Remove from active transactions
            del self.active_transactions[transaction_id]
            if transaction_id in self.pool.active_transactions:
                del self.pool.active_transactions[transaction_id]
                
            return True, None
        except Exception as e:
            error_msg = f"Error committing transaction: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def rollback(self, transaction_id: str) -> Tuple[bool, Optional[str]]:
        """Rollback an active transaction"""
        if transaction_id not in self.active_transactions:
            return False, f"Transaction {transaction_id} not found"
            
        try:
            tr_data = self.active_transactions[transaction_id]
            
            # Log transaction details before rollback
            duration = time.time() - tr_data["start_time"]
            query_count = len(tr_data["queries"])
            self.logger.debug(f"Rolling back transaction {transaction_id} after {duration:.3f}s with {query_count} queries")
            
            await tr_data["transaction"].rollback()
            await self.pool.release(tr_data["connection"])
            
            # Remove from active transactions
            del self.active_transactions[transaction_id]
            if transaction_id in self.pool.active_transactions:
                del self.pool.active_transactions[transaction_id]
                
            return True, None
        except Exception as e:
            error_msg = f"Error rolling back transaction: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    async def close(self) -> None:
        """Close all connections"""
        if self.initialized:
            # Rollback any active transactions
            for transaction_id in list(self.active_transactions.keys()):
                await self.rollback(transaction_id)
                
            # Close the pool
            self.initialized = False
            if self.pool:
                await self.pool.close()
                self.pool = None
    
    async def _execute(self, command: str, **kwargs) -> ToolResult[Any]:
        return await self.run(command, **kwargs)

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
                    
            elif command == "execute_in_transaction":
                transaction_id = kwargs.get("transaction_id")
                query = kwargs.get("query")
                params = kwargs.get("params")
                
                if not transaction_id or not query:
                    return ToolResult(success=False, error="Transaction ID and query are required for execute_in_transaction command")
                
                success, data, error = await self.execute_in_transaction(transaction_id, query, params)
                if success:
                    result = data
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
                
            elif command == "get_stats":
                # Get database statistics
                result = {
                    "query_stats": self.query_stats.get_stats(),
                    "pool_stats": self.pool.get_stats() if self.pool else {},
                    "active_transactions": len(self.active_transactions),
                    "plan_cache_size": len(self.query_plan_cache)
                }
                
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
                
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            error_msg = f"PostgreSQL tool error: {str(e)}"
            self.logger.error(error_msg)
            return await self._handle_error(e)