"""
PostgresTool Requirements

The PostgresTool should provide a robust interface to the database with the following functionality:

1. Connection Management:
   - Connect to PostgreSQL database using asyncpg
   - Support connection pooling for efficient resource usage
   - Handle connection errors gracefully

2. Query Execution:
   - Execute parameterized queries to prevent SQL injection
   - Support both single queries and batch operations
   - Handle transactions with commit/rollback support

3. Data Operations:
   - Query corporate information
   - Store analysis results
   - Cache research data
   - Persist report templates and generated reports
   - Track agent interactions and workflow state

4. Schema Operations:
   - Create schema for storing:
     - Company information (financials, management, regulatory history)
     - Analysis results (forensic insights, red flags, timeline events)
     - Research data (articles, events, evidence)
     - Reports (templates, sections, final reports)

5. Required Methods:
   - connect() - Establish connection to database
   - execute_query() - Execute SQL query with parameters
   - fetch_one() - Fetch single row result
   - fetch_all() - Fetch all results
   - execute_batch() - Execute batch operations
   - begin_transaction() - Start transaction
   - commit() - Commit transaction
   - rollback() - Rollback transaction
   - close() - Close connection

6. Integration with Agents:
   - Provide methods for Corporate Agent to fetch financial data
   - Store analysis results from Analyst Agent
   - Store research data from Research Agent
   - Support Writer Agent with report templates and storage
"""