import os
import json
import asyncio
import random
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


class FinancialResult(BaseModel):
    quarter: str
    period_end_date: str
    revenue: float
    operating_profit: float
    net_profit: float
    eps: float
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None
    return_on_equity: Optional[float] = None


class Announcement(BaseModel):
    date: str
    title: str
    type: str
    summary: str
    url: Optional[str] = None


class ShareholdingPattern(BaseModel):
    date: str
    promoters: float
    fii: float
    dii: float
    public: float
    others: float


class InsiderTrade(BaseModel):
    date: str
    person_name: str
    person_category: str
    transaction_type: str
    quantity: int
    price: float
    value: float


class CorporateAction(BaseModel):
    date: str
    ex_date: str
    action_type: str
    value: float
    details: Optional[str] = None


class PriceData(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class UnusualPattern(BaseModel):
    date: str
    pattern_type: str
    severity: float  # 0-1
    description: str
    supporting_metrics: Dict[str, float]


class NSETool(BaseTool):
    name = "nse_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.api_key = config.get("api_key", os.environ.get("NSE_API_KEY", "mock_key"))
        self.base_url = config.get("base_url", "https://api.example-nse.com/v1")
        
        # Cache for mock data
        self.company_cache = {}
    
    def _generate_mock_price_data(self, days: int, seed: Optional[str] = None) -> List[PriceData]:
        """Generate mock price data for testing"""
        if seed:
            random.seed(seed)
            np.random.seed(int.from_bytes(seed.encode(), 'big') % (2**32))
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate a realistic price series
        initial_price = random.uniform(100, 1000)
        volatility = random.uniform(0.01, 0.05)
        drift = random.uniform(-0.0002, 0.0004)
        
        # Generate returns
        daily_returns = np.random.normal(drift, volatility, days)
        
        # Calculate price series
        prices = [initial_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate price data for each day
        price_data = []
        current_date = start_date
        
        for i in range(days):
            if current_date.weekday() < 5:  # Skip weekends
                base_price = prices[i]
                daily_vol = base_price * volatility
                
                # Generate OHLC
                open_price = base_price
                high_price = base_price + random.uniform(0, daily_vol * 2)
                low_price = base_price - random.uniform(0, daily_vol * 1.5)
                close_price = prices[i+1]
                
                # Ensure high is highest and low is lowest
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                # Volume: higher on volatile days
                price_change_pct = abs(close_price - open_price) / open_price
                volume = int(random.uniform(100000, 1000000) * (1 + price_change_pct * 10))
                
                price_data.append(PriceData(
                    date=current_date.strftime("%Y-%m-%d"),
                    open=round(open_price, 2),
                    high=round(high_price, 2),
                    low=round(low_price, 2),
                    close=round(close_price, 2),
                    volume=volume
                ))
            
            current_date += timedelta(days=1)
        
        return price_data
    
    def _generate_mock_financial_results(self, company: str, quarters: int) -> List[FinancialResult]:
        """Generate mock financial results for testing"""
        if company in self.company_cache and "financial_results" in self.company_cache[company]:
            return self.company_cache[company]["financial_results"][:quarters]
        
        # Create a seed from company name for consistency
        seed = f"{company}_financials"
        random.seed(seed)
        
        # Base financials - will grow or decline over quarters
        base_revenue = random.uniform(1000, 10000)
        base_margin = random.uniform(0.08, 0.25)
        base_shares = random.uniform(10, 100) * 1000000
        
        # Growth or decline trend (some randomness but with a trend)
        revenue_growth_trend = random.uniform(-0.05, 0.15)
        margin_trend = random.uniform(-0.02, 0.05)
        
        # Generate quarterly results
        results = []
        current_date = datetime.now()
        
        # Start with the most recent quarter and go backwards
        for i in range(quarters):
            # Calculate quarter and year
            quarter_month = ((current_date.month - 1) // 3) * 3 + 1
            quarter_num = (quarter_month - 1) // 3 + 1
            quarter_str = f"Q{quarter_num} {current_date.year}"
            
            # Add randomness to the trend
            quarter_growth = revenue_growth_trend + random.uniform(-0.05, 0.05)
            quarter_margin_change = margin_trend + random.uniform(-0.01, 0.01)
            
            # Calculate financials with compounding growth/decline
            revenue = base_revenue * (1 + quarter_growth) ** i
            operating_margin = max(0.01, min(0.40, base_margin + quarter_margin_change * i))
            net_margin = operating_margin * random.uniform(0.6, 0.85)
            
            operating_profit = revenue * operating_margin
            net_profit = revenue * net_margin
            eps = net_profit / base_shares
            
            # Generate some ratios
            debt_to_equity = random.uniform(0.1, 2.0)
            current_ratio = random.uniform(0.8, 3.0)
            quick_ratio = current_ratio * random.uniform(0.6, 0.9)
            interest_coverage = random.uniform(1.5, 12.0)
            return_on_equity = random.uniform(0.05, 0.25)
            
            # Quarter end date
            quarter_end = datetime(
                current_date.year, 
                quarter_month + 2 if quarter_month <= 10 else 12,
                28 if quarter_month + 2 != 2 else 29
            )
            
            results.append(FinancialResult(
                quarter=quarter_str,
                period_end_date=quarter_end.strftime("%Y-%m-%d"),
                revenue=round(revenue, 2),
                operating_profit=round(operating_profit, 2),
                net_profit=round(net_profit, 2),
                eps=round(eps, 2),
                debt_to_equity=round(debt_to_equity, 2),
                current_ratio=round(current_ratio, 2),
                quick_ratio=round(quick_ratio, 2),
                interest_coverage=round(interest_coverage, 2),
                return_on_equity=round(return_on_equity, 2)
            ))
            
            # Move to previous quarter
            current_date = current_date - timedelta(days=90)
        
        # Cache the results
        if company not in self.company_cache:
            self.company_cache[company] = {}
        self.company_cache[company]["financial_results"] = results
        
        return results
    
    def _generate_mock_announcements(self, company: str, filing_type: Optional[str] = None, limit: int = 20) -> List[Announcement]:
        """Generate mock company announcements for testing"""
        seed = f"{company}_announcements"
        random.seed(seed)
        
        # Lists of possible announcement types and templates
        announcement_types = [
            "regulatory", "financial", "board_meeting", "dividend", 
            "acquisition", "expansion", "management_change"
        ]
        
        # Filter by type if specified
        if filing_type and filing_type in announcement_types:
            filtered_types = [filing_type]
        else:
            filtered_types = announcement_types
        
        title_templates = {
            "regulatory": [
                "Compliance with {regulation} Regulations",
                "Disclosure under {regulation} Requirements",
                "Statement of Investor Complaints",
                "Outcome of Board Meeting - {regulation} Compliance"
            ],
            "financial": [
                "Financial Results for {quarter}",
                "Audited Financial Results for {period}",
                "Un-audited Financial Results for {quarter}",
                "Investor Presentation for {quarter}"
            ],
            "board_meeting": [
                "Notice of Board Meeting",
                "Outcome of Board Meeting",
                "Board Meeting Scheduled on {date}",
                "Change in Date of Board Meeting"
            ],
            "dividend": [
                "Declaration of {dividend_type} Dividend",
                "Record Date for {dividend_type} Dividend",
                "Dividend of Rs. {amount} per share"
            ],
            "acquisition": [
                "Acquisition of {target_company}",
                "Update on Acquisition of {target_company}",
                "Completion of Acquisition Transaction"
            ],
            "expansion": [
                "Expansion of {facility_type} in {location}",
                "New Project Announcement",
                "Capacity Addition at {location}"
            ],
            "management_change": [
                "Appointment of {position}",
                "Resignation of {position}",
                "Change in Key Management Personnel"
            ]
        }
        
        regulations = ["SEBI", "LODR", "PIT", "Takeover", "ICDR"]
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        years = [str(datetime.now().year - i) for i in range(3)]
        dividend_types = ["Interim", "Final", "Special"]
        positions = ["CEO", "CFO", "Director", "Managing Director", "CTO", "COO"]
        locations = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune"]
        facility_types = ["Manufacturing Plant", "R&D Center", "Office", "Warehouse"]
        target_companies = [f"XYZ Ltd", "ABC Corp", "PQR Industries", "LMN Technologies"]
        
        # Generate announcements
        announcements = []
        current_date = datetime.now()
        
        for i in range(limit):
            # Select a random type from filtered types
            ann_type = random.choice(filtered_types)
            
            # Select a random template
            template = random.choice(title_templates[ann_type])
            
            # Fill in the template
            if ann_type == "regulatory":
                regulation = random.choice(regulations)
                title = template.replace("{regulation}", regulation)
            elif ann_type == "financial":
                quarter = f"{random.choice(quarters)} {random.choice(years)}"
                period = f"FY {random.choice(years)}"
                title = template.replace("{quarter}", quarter).replace("{period}", period)
            elif ann_type == "board_meeting":
                meeting_date = (current_date - timedelta(days=random.randint(1, 30))).strftime("%d-%m-%Y")
                title = template.replace("{date}", meeting_date)
            elif ann_type == "dividend":
                dividend_type = random.choice(dividend_types)
                amount = round(random.uniform(0.5, 10.0), 1)
                title = template.replace("{dividend_type}", dividend_type).replace("{amount}", str(amount))
            elif ann_type == "acquisition":
                target = random.choice(target_companies)
                title = template.replace("{target_company}", target)
            elif ann_type == "expansion":
                facility = random.choice(facility_types)
                location = random.choice(locations)
                title = template.replace("{facility_type}", facility).replace("{location}", location)
            elif ann_type == "management_change":
                position = random.choice(positions)
                title = template.replace("{position}", position)
            
            # Generate date (spread over last year with more recent ones first)
            days_ago = int(np.random.exponential(scale=60))
            ann_date = current_date - timedelta(days=days_ago)
            
            # Generate a summary
            summary = f"This is a mock summary for the announcement: {title}"
            
            announcements.append(Announcement(
                date=ann_date.strftime("%Y-%m-%d"),
                title=title,
                type=ann_type,
                summary=summary,
                url=f"https://example.com/announcements/{company}/{ann_date.strftime('%Y%m%d')}"
            ))
            
        return announcements
    
    def _generate_mock_shareholding_pattern(self, company: str, quarters: int = 4) -> List[ShareholdingPattern]:
        """Generate mock shareholding pattern data for testing"""
        seed = f"{company}_shareholding"
        random.seed(seed)
        
        # Initial shareholding pattern
        initial_promoters = random.uniform(35, 75)
        initial_fii = random.uniform(5, 25)
        initial_dii = random.uniform(5, 20)
        initial_public = random.uniform(10, 40)
        
        # Ensure the total is 100%
        total = initial_promoters + initial_fii + initial_dii + initial_public
        initial_promoters = initial_promoters * 100 / total
        initial_fii = initial_fii * 100 / total
        initial_dii = initial_dii * 100 / total
        initial_public = initial_public * 100 / total
        
        # Generate quarterly data with small changes
        patterns = []
        current_date = datetime.now()
        
        promoters = initial_promoters
        fii = initial_fii
        dii = initial_dii
        public = initial_public
        
        for i in range(quarters):
            # Calculate quarter end date
            quarter_month = ((current_date.month - 1) // 3) * 3 + 1
            quarter_end = datetime(
                current_date.year, 
                quarter_month + 2 if quarter_month <= 10 else 12,
                28 if quarter_month + 2 != 2 else 29
            )
            
            # Small random changes
            promoters_change = random.uniform(-1.5, 1.0)
            fii_change = random.uniform(-1.0, 1.5)
            dii_change = random.uniform(-0.8, 1.2)
            
            # Apply changes
            promoters += promoters_change
            fii += fii_change
            dii += dii_change
            
            # Calculate public to ensure total is 100%
            public = 100 - (promoters + fii + dii)
            
            # Ensure all values are positive
            promoters = max(0, min(100, promoters))
            fii = max(0, min(100, fii))
            dii = max(0, min(100, dii))
            public = max(0, min(100, public))
            
            # Recalculate to ensure exactly 100%
            total = promoters + fii + dii + public
            promoters = promoters * 100 / total
            fii = fii * 100 / total
            dii = dii * 100 / total
            public = public * 100 / total
            
            patterns.append(ShareholdingPattern(
                date=quarter_end.strftime("%Y-%m-%d"),
                promoters=round(promoters, 2),
                fii=round(fii, 2),
                dii=round(dii, 2),
                public=round(public, 2),
                others=round(100 - (promoters + fii + dii + public), 2)  # Should be close to 0
            ))
            
            # Move to previous quarter
            current_date = current_date - timedelta(days=90)
            
        return patterns
    
    def _generate_mock_insider_trades(self, company: str, days: int = 180) -> List[InsiderTrade]:
        """Generate mock insider trading data for testing"""
        seed = f"{company}_insider"
        random.seed(seed)
        
        # List of potential insiders
        # Format: (name, category)
        insiders = [
            ("John Smith", "Director"),
            ("Alice Johnson", "CFO"),
            ("Robert Williams", "CEO"),
            ("Sarah Brown", "Director"),
            ("Michael Davis", "CTO"),
            ("Jennifer Wilson", "Promoter"),
            ("David Miller", "Director"),
            ("Linda Garcia", "Company Secretary"),
            ("James Anderson", "Promoter"),
            ("Patricia Martinez", "COO")
        ]
        
        # Generate stock price for reference
        price_data = self._generate_mock_price_data(days, seed=f"{company}_price_for_insider")
        price_by_date = {pd.date: pd.close for pd in price_data}
        
        # Generate trades
        trades = []
        
        # Number of trades to generate (random but proportional to days)
        num_trades = random.randint(days // 30, days // 10)
        
        for _ in range(num_trades):
            # Select random insider
            insider_name, insider_category = random.choice(insiders)
            
            # Select random date from the price data
            trade_date_str = random.choice(list(price_by_date.keys()))
            base_price = price_by_date[trade_date_str]
            
            # Determine transaction type (more buys than sells)
            transaction_type = "Buy" if random.random() < 0.6 else "Sell"
            
            # Generate quantity and price
            quantity = random.randint(1000, 50000)
            price = base_price * random.uniform(0.98, 1.02)  # Small variation around market price
            value = quantity * price
            
            trades.append(InsiderTrade(
                date=trade_date_str,
                person_name=insider_name,
                person_category=insider_category,
                transaction_type=transaction_type,
                quantity=quantity,
                price=round(price, 2),
                value=round(value, 2)
            ))
        
        # Sort by date (most recent first)
        trades.sort(key=lambda x: x.date, reverse=True)
        
        return trades
    
    def _generate_mock_corporate_actions(self, company: str, days: int = 365) -> List[CorporateAction]:
        """Generate mock corporate actions for testing"""
        seed = f"{company}_corporate"
        random.seed(seed)
        
        # Generate a realistic number of corporate actions
        num_actions = random.randint(1, 5)
        
        actions = []
        current_date = datetime.now()
        
        action_types = ["Dividend", "Bonus", "Split", "Rights", "Buyback"]
        action_weights = [0.6, 0.1, 0.1, 0.1, 0.1]  # Dividends are more common
        
        for _ in range(num_actions):
            # Select a random action type based on weights
            action_type = random.choices(action_types, weights=action_weights, k=1)[0]
            
            # Random date in the past year
            days_ago = random.randint(1, days)
            action_date = current_date - timedelta(days=days_ago)
            
            # Ex-date is typically a few days before record date
            ex_date = action_date - timedelta(days=random.randint(2, 7))
            
            # Action details
            if action_type == "Dividend":
                value = round(random.uniform(0.5, 10), 2)
                details = f"Dividend of Rs {value} per share"
            elif action_type == "Bonus":
                ratio_num = random.randint(1, 5)
                ratio_den = 10
                value = ratio_num / ratio_den
                details = f"Bonus issue in the ratio of {ratio_num}:{ratio_den}"
            elif action_type == "Split":
                from_face = 10
                to_face = random.choice([1, 2, 5])
                value = from_face / to_face
                details = f"Stock split from face value of Rs {from_face} to Rs {to_face}"
            elif action_type == "Rights":
                ratio_num = random.randint(1, 3)
                ratio_den = 10
                price = random.randint(50, 500)
                value = ratio_num / ratio_den
                details = f"Rights issue in the ratio of {ratio_num}:{ratio_den} at Rs {price} per share"
            elif action_type == "Buyback":
                percentage = round(random.uniform(1, 10), 2)
                price = random.randint(100, 1000)
                value = percentage
                details = f"Buyback of {percentage}% of equity shares at Rs {price} per share"
            
            actions.append(CorporateAction(
                date=action_date.strftime("%Y-%m-%d"),
                ex_date=ex_date.strftime("%Y-%m-%d"),
                action_type=action_type,
                value=value,
                details=details
            ))
        
        # Sort by date (most recent first)
        actions.sort(key=lambda x: x.date, reverse=True)
        
        return actions
    
    def _detect_unusual_patterns(self, price_data: List[PriceData]) -> List[UnusualPattern]:
        """Detect potentially unusual patterns in price/volume data for testing"""
        if not price_data or len(price_data) < 10:
            return []
        
        # Convert to numpy arrays for analysis
        dates = [pd.date for pd in price_data]
        prices = np.array([pd.close for pd in price_data])
        volumes = np.array([pd.volume for pd in price_data])
        
        # Calculate returns
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        
        # Calculate moving averages and volatility
        window = min(20, len(prices) // 4)
        moving_avg = np.zeros_like(prices)
        volatility = np.zeros_like(prices)
        
        for i in range(len(prices)):
            start = max(0, i - window)
            moving_avg[i] = np.mean(prices[start:i+1])
            if i >= window:
                volatility[i] = np.std(returns[i-window+1:i+1])
        
        # Relative volume
        avg_volume = np.mean(volumes)
        relative_volume = volumes / avg_volume
        
        # Find unusual patterns
        patterns = []
        
        # 1. Unusual price movements (price shocks)
        for i in range(5, len(prices)):
            daily_return = returns[i]
            recent_volatility = volatility[i]
            
            if recent_volatility > 0 and abs(daily_return) > 2.5 * recent_volatility:
                direction = "positive" if daily_return > 0 else "negative"
                severity = min(1.0, abs(daily_return) / (3 * recent_volatility))
                
                patterns.append(UnusualPattern(
                    date=dates[i],
                    pattern_type=f"Price Shock ({direction})",
                    severity=round(severity, 2),
                    description=f"Abnormal {direction} price movement of {abs(daily_return)*100:.1f}%",
                    supporting_metrics={
                        "return": round(daily_return * 100, 2),
                        "recent_volatility": round(recent_volatility * 100, 2),
                        "z_score": round(daily_return / (recent_volatility if recent_volatility > 0 else 0.01), 2)
                    }
                ))
        
        # 2. Unusual volume spikes
        for i in range(5, len(volumes)):
            if relative_volume[i] > 3.0:
                severity = min(1.0, (relative_volume[i] - 2) / 8)
                
                patterns.append(UnusualPattern(
                    date=dates[i],
                    pattern_type="Volume Spike",
                    severity=round(severity, 2),
                    description=f"Volume {relative_volume[i]:.1f}x higher than average",
                    supporting_metrics={
                        "relative_volume": round(relative_volume[i], 2),
                        "price_change": round(returns[i] * 100, 2),
                        "avg_volume": int(avg_volume)
                    }
                ))
        
        # 3. Price-volume divergence
        for i in range(5, len(prices)):
            if abs(returns[i]) > 0.02 and relative_volume[i] > 2.0:
                if (returns[i] > 0 and prices[i] < moving_avg[i]) or (returns[i] < 0 and prices[i] > moving_avg[i]):
                    severity = min(1.0, (relative_volume[i] * abs(returns[i]) * 100) / 10)
                    
                    patterns.append(UnusualPattern(
                        date=dates[i],
                        pattern_type="Price-Volume Divergence",
                        severity=round(severity, 2),
                        description="Unusual price-volume relationship suggests potential manipulation",
                        supporting_metrics={
                            "price_vs_moving_avg": round((prices[i] / moving_avg[i] - 1) * 100, 2),
                            "return": round(returns[i] * 100, 2),
                            "relative_volume": round(relative_volume[i], 2)
                        }
                    ))
        
        # Sort by date and limit to avoid too many patterns
        patterns.sort(key=lambda x: x.date, reverse=True)
        
        # Filter to unique dates and keep most severe
        date_severity = {}
        for pattern in patterns:
            if pattern.date not in date_severity or pattern.severity > date_severity[pattern.date][1]:
                date_severity[pattern.date] = (pattern, pattern.severity)
        
        unique_patterns = [v[0] for v in date_severity.values()]
        unique_patterns.sort(key=lambda x: x.date, reverse=True)
        
        return unique_patterns[:5]  # Limit to 5 most recent
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_stock_price_history(self, company: str, days: int = 365) -> Dict[str, Any]:
        """Get historical stock prices for a company"""
        self.logger.info(f"Fetching stock price history for {company} (last {days} days)")
        
        # For testing, generate mock data
        price_data = self._generate_mock_price_data(days, seed=company)
        
        return {
            "company": company,
            "days": days,
            "price_data": [pd.model_dump() for pd in price_data]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_company_announcements(self, company: str, filing_type: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Get company announcements and disclosures"""
        self.logger.info(f"Fetching announcements for {company} (type: {filing_type}, limit: {limit})")
        
        # For testing, generate mock data
        announcements = self._generate_mock_announcements(company, filing_type, limit)
        
        return {
            "company": company,
            "filing_type": filing_type,
            "count": len(announcements),
            "announcements": [ann.model_dump() for ann in announcements]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_financial_results(self, company: str, quarters: int = 4) -> Dict[str, Any]:
        """Get quarterly/annual financial results"""
        self.logger.info(f"Fetching financial results for {company} (last {quarters} quarters)")
        
        # For testing, generate mock data
        results = self._generate_mock_financial_results(company, quarters)
        
        return {
            "company": company,
            "quarters": quarters,
            "results": [res.model_dump() for res in results]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_shareholding_pattern(self, company: str, quarters: int = 4) -> Dict[str, Any]:
        """Get shareholding patterns"""
        self.logger.info(f"Fetching shareholding pattern for {company} (last {quarters} quarters)")
        
        # For testing, generate mock data
        patterns = self._generate_mock_shareholding_pattern(company, quarters)
        
        return {
            "company": company,
            "quarters": quarters,
            "patterns": [pattern.model_dump() for pattern in patterns]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_insider_trading(self, company: str, days: int = 180) -> Dict[str, Any]:
        """Get insider trading data"""
        self.logger.info(f"Fetching insider trading data for {company} (last {days} days)")
        
        # For testing, generate mock data
        trades = self._generate_mock_insider_trades(company, days)
        
        return {
            "company": company,
            "days": days,
            "trades": [trade.model_dump() for trade in trades]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_corporate_actions(self, company: str, days: int = 365) -> Dict[str, Any]:
        """Get corporate actions (dividends, splits, bonus, etc.)"""
        self.logger.info(f"Fetching corporate actions for {company} (last {days} days)")
        
        # For testing, generate mock data
        actions = self._generate_mock_corporate_actions(company, days)
        
        return {
            "company": company,
            "days": days,
            "actions": [action.model_dump() for action in actions]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def calculate_financial_ratios(self, company: str) -> Dict[str, Any]:
        """Calculate key financial ratios and metrics"""
        self.logger.info(f"Calculating financial ratios for {company}")
        
        # Get recent financial results
        financial_data = await self.get_financial_results(company, quarters=4)
        results = [FinancialResult(**res) for res in financial_data.get("results", [])]
        
        if not results:
            return {
                "company": company,
                "error": "No financial data available"
            }
        
        # Get most recent quarter
        latest = results[0]
        
        # Get previous year same quarter (for YoY comparisons)
        prev_year_same_quarter = next((r for r in results if r.quarter[-4:] == str(int(latest.quarter[-4:]) - 1)), None)
        
        # Calculate ratios
        ratios = {
            "profitability": {
                "operating_margin": round(latest.operating_profit / latest.revenue * 100, 2),
                "net_margin": round(latest.net_profit / latest.revenue * 100, 2),
                "return_on_equity": latest.return_on_equity * 100 if latest.return_on_equity else None
            },
            "liquidity": {
                "current_ratio": latest.current_ratio,
                "quick_ratio": latest.quick_ratio
            },
            "leverage": {
                "debt_to_equity": latest.debt_to_equity,
                "interest_coverage": latest.interest_coverage
            },
            "growth": {}
        }
        
        # Add YoY growth metrics if available
        if prev_year_same_quarter:
            revenue_growth = (latest.revenue / prev_year_same_quarter.revenue - 1) * 100
            profit_growth = (latest.net_profit / prev_year_same_quarter.net_profit - 1) * 100
            
            ratios["growth"] = {
                "revenue_growth_yoy": round(revenue_growth, 2),
                "profit_growth_yoy": round(profit_growth, 2)
            }
        
        return {
            "company": company,
            "latest_quarter": latest.quarter,
            "ratios": ratios
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def detect_unusual_patterns(self, company: str, price_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Identify unusual trading patterns"""
        self.logger.info(f"Detecting unusual patterns for {company}")
        
        if not price_data:
            # Get price history if not provided
            price_history = await self.get_stock_price_history(company, days=180)
            price_data = price_history.get("price_data", [])
        
        # Convert price data to PriceData model
        prices = [PriceData(**pd) if isinstance(pd, dict) else pd for pd in price_data]
        
        # Detect patterns
        patterns = self._detect_unusual_patterns(prices)
        
        return {
            "company": company,
            "pattern_count": len(patterns),
            "patterns": [pattern.model_dump() for pattern in patterns]
        }
    
    async def run(self, command: str, **kwargs) -> ToolResult[Dict[str, Any]]:
        """Execute various NSE data operations"""
        try:
            self.logger.info(f"Running NSE command: {command}")
            
            result = {}
            
            if command == "get_stock_price_history":
                company = kwargs.get("company")
                days = kwargs.get("days", 365)
                
                if not company:
                    return ToolResult(success=False, error="Company parameter is required")
                
                result = await self.get_stock_price_history(company, days)
                
            elif command == "get_company_announcements":
                company = kwargs.get("company")
                filing_type = kwargs.get("filing_type")
                limit = kwargs.get("limit", 20)
                
                if not company:
                    return ToolResult(success=False, error="Company parameter is required")
                
                result = await self.get_company_announcements(company, filing_type, limit)
                
            elif command == "get_financial_results":
                company = kwargs.get("company")
                quarters = kwargs.get("quarters", 4)
                
                if not company:
                    return ToolResult(success=False, error="Company parameter is required")
                
                result = await self.get_financial_results(company, quarters)
                
            elif command == "get_shareholding_pattern":
                company = kwargs.get("company")
                quarters = kwargs.get("quarters", 4)
                
                if not company:
                    return ToolResult(success=False, error="Company parameter is required")
                
                result = await self.get_shareholding_pattern(company, quarters)
                
            elif command == "get_insider_trading":
                company = kwargs.get("company")
                days = kwargs.get("days", 180)
                
                if not company:
                    return ToolResult(success=False, error="Company parameter is required")
                
                result = await self.get_insider_trading(company, days)
                
            elif command == "get_corporate_actions":
                company = kwargs.get("company")
                days = kwargs.get("days", 365)
                
                if not company:
                    return ToolResult(success=False, error="Company parameter is required")
                
                result = await self.get_corporate_actions(company, days)
                
            elif command == "calculate_financial_ratios":
                company = kwargs.get("company")
                
                if not company:
                    return ToolResult(success=False, error="Company parameter is required")
                
                result = await self.calculate_financial_ratios(company)
                
            elif command == "detect_unusual_patterns":
                company = kwargs.get("company")
                price_data = kwargs.get("price_data")
                
                if not company:
                    return ToolResult(success=False, error="Company parameter is required")
                
                result = await self.detect_unusual_patterns(company, price_data)
                
            else:
                return ToolResult(success=False, error=f"Unknown command: {command}")
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"NSE tool error: {str(e)}")
            return await self._handle_error(e)