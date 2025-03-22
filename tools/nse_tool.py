"""
NSETool Requirements

The NSETool should provide functionality to interact with National Stock Exchange (NSE) data for Indian companies with the following capabilities:

1. Data Retrieval:
   - Fetch historical stock prices and trading volumes
   - Retrieve company announcements and disclosures
   - Get corporate actions (dividends, splits, bonus issues)
   - Access quarterly and annual financial results
   - Download shareholding patterns and insider trading data

2. Data Analysis:
   - Calculate key financial ratios and metrics
   - Identify unusual trading patterns
   - Track price movements around key events
   - Compare performance against sector indices
   - Detect potential insider trading patterns

3. API Integration:
   - Connect to NSE official APIs where available
   - Support alternative data sources when official APIs are limited
   - Handle rate limiting and authentication
   - Cache responses to minimize API calls

4. Required Methods:
   - get_stock_price_history() - Fetch historical stock prices
   - get_company_announcements() - Fetch company announcements
   - get_financial_results() - Fetch quarterly/annual financial results
   - get_shareholding_pattern() - Fetch shareholding patterns
   - get_insider_trading() - Fetch insider trading data
   - get_corporate_actions() - Fetch corporate actions
   - calculate_financial_ratios() - Calculate key financial ratios
   - detect_unusual_patterns() - Identify unusual trading patterns

5. Integration with Corporate Agent:
   - Provide market data for corporate analysis
   - Supply event timeline information
   - Support forensic analysis with trading pattern detection
"""