"""
Stock Screening Engine

This module provides a comprehensive stock screening system that combines
multiple factors to rank and select the best stocks for trading. It integrates
with the sector management system and factor calculation modules to provide
intelligent stock selection capabilities.

Features:
- Multi-factor scoring algorithm
- Stock ranking and Top-N selection
- Customizable screening criteria
- Integration with existing factor modules
- Backtesting and validation functionality
- Real-time screening capabilities
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .config import SETTINGS
from .data import fetch_batch_history, get_batch_latest_data
from .factors.valuation import valuation_score
from .factors.volume_factors import volume_features, cross_section_volume_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScreeningCriteria:
    """Configuration for stock screening criteria."""
    # Basic filters
    min_market_cap: float = 1e9  # 1B minimum
    max_market_cap: float = 1e12  # 1T maximum
    min_price: float = 5.0  # Minimum stock price
    max_price: float = 1000.0  # Maximum stock price
    min_volume: int = 100000  # Minimum daily volume
    
    # Technical criteria
    min_rsi: float = 20.0  # Oversold threshold
    max_rsi: float = 80.0  # Overbought threshold
    min_volatility: float = 0.1  # Minimum volatility
    max_volatility: float = 2.0  # Maximum volatility
    
    # Factor weights
    valuation_weight: float = 0.3
    volume_weight: float = 0.2
    momentum_weight: float = 0.2
    quality_weight: float = 0.15
    technical_weight: float = 0.15
    
    # Selection criteria
    top_n: int = 20  # Number of stocks to select
    sectors: List[str] = None  # Specific sectors to screen
    exclude_symbols: List[str] = None  # Symbols to exclude
    
    def __post_init__(self):
        if self.sectors is None:
            self.sectors = []
        if self.exclude_symbols is None:
            self.exclude_symbols = []


@dataclass
class StockScore:
    """Individual stock score with component breakdown."""
    symbol: str
    final_score: float
    rank: int = 0
    
    # Component scores
    valuation_score: float = 0.0
    volume_score: float = 0.0
    momentum_score: float = 0.0
    quality_score: float = 0.0
    technical_score: float = 0.0
    
    # Market data
    current_price: float = 0.0
    market_cap: float = 0.0
    volume: int = 0
    sector: str = "Unknown"
    
    # Technical indicators
    rsi: float = 50.0
    volatility: float = 0.0
    
    # Metadata
    last_updated: str = ""


class StockScreener:
    """
    Advanced stock screening engine with multi-factor analysis.
    
    This class provides comprehensive stock screening capabilities:
    - Multi-factor scoring and ranking
    - Customizable screening criteria
    - Real-time and historical screening
    - Backtesting and validation
    - Integration with sector management
    """
    
    def __init__(self, criteria: Optional[ScreeningCriteria] = None):
        """
        Initialize the stock screener.
        
        Args:
            criteria: Screening criteria configuration. Uses defaults if None.
        """
        self.criteria = criteria or ScreeningCriteria()
        self.last_screening_results: List[StockScore] = []
        self.screening_cache: Dict[str, Any] = {}
        
        logger.info(f"[stock_screener] Initialized with top_n={self.criteria.top_n}")
    
    def screen_stocks(
        self,
        quote_client,
        symbols: Optional[List[str]] = None,
        dry_run: bool = False,
        use_cache: bool = True
    ) -> List[StockScore]:
        """
        Perform comprehensive stock screening.
        
        Args:
            quote_client: Tiger SDK quote client instance
            symbols: List of symbols to screen. If None, screens all sector stocks.
            dry_run: If True, use placeholder data
            use_cache: If True, use cached data when available
            
        Returns:
            List of StockScore objects sorted by score (best first)
        """
        logger.info(f"[stock_screener] Starting stock screening")
        
        # Get symbols to screen
        if symbols is None:
            symbols = self._get_screening_universe()
        
        if not symbols:
            logger.warning(f"[stock_screener] No symbols to screen")
            return []
        
        logger.info(f"[stock_screener] Screening {len(symbols)} symbols")
        
        # Fetch market data
        market_data = self._fetch_screening_data(quote_client, symbols, dry_run)
        
        # Apply basic filters
        filtered_symbols = self._apply_basic_filters(symbols, market_data)
        
        if not filtered_symbols:
            logger.warning(f"[stock_screener] No symbols passed basic filters")
            return []
        
        logger.info(f"[stock_screener] {len(filtered_symbols)} symbols passed basic filters")
        
        # Calculate factor scores
        factor_scores = self._calculate_factor_scores(filtered_symbols, market_data)
        
        # Combine scores and rank
        final_scores = self._combine_scores(filtered_symbols, factor_scores, market_data)
        
        # Sort by score and apply top-N selection
        final_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign ranks
        for i, score in enumerate(final_scores):
            score.rank = i + 1
        
        # Apply top-N selection
        selected_stocks = final_scores[:self.criteria.top_n]
        
        self.last_screening_results = selected_stocks
        
        logger.info(f"[stock_screener] Screening completed: selected {len(selected_stocks)} stocks")
        
        return selected_stocks
    
    def screen_sector(
        self,
        quote_client,
        sector_name: str,
        dry_run: bool = False
    ) -> List[StockScore]:
        """
        Screen stocks within a specific sector.
        
        Args:
            quote_client: Tiger SDK quote client instance
            sector_name: Name of the sector to screen
            dry_run: If True, use placeholder data
            
        Returns:
            List of StockScore objects for the sector
        """
        try:
            from .sector_manager import get_sector_stocks
            
            symbols = get_sector_stocks(sector_name, validate=True)
            
            if not symbols:
                logger.warning(f"[stock_screener] No valid stocks in sector '{sector_name}'")
                return []
            
            logger.info(f"[stock_screener] Screening {len(symbols)} stocks in sector '{sector_name}'")
            
            return self.screen_stocks(quote_client, symbols, dry_run)
            
        except ImportError:
            logger.error(f"[stock_screener] Sector manager not available")
            return []
        except Exception as e:
            logger.error(f"[stock_screener] Failed to screen sector '{sector_name}': {e}")
            return []
    
    def get_top_stocks(self, n: Optional[int] = None) -> List[StockScore]:
        """
        Get top N stocks from last screening results.
        
        Args:
            n: Number of top stocks to return. Uses criteria.top_n if None.
            
        Returns:
            List of top StockScore objects
        """
        if not self.last_screening_results:
            logger.warning(f"[stock_screener] No screening results available")
            return []
        
        top_n = n or self.criteria.top_n
        return self.last_screening_results[:top_n]
    
    def get_screening_summary(self) -> Dict[str, Any]:
        """
        Get summary of last screening results.
        
        Returns:
            Dictionary with screening summary statistics
        """
        if not self.last_screening_results:
            return {"error": "No screening results available"}
        
        scores = [stock.final_score for stock in self.last_screening_results]
        
        return {
            "total_stocks": len(self.last_screening_results),
            "top_n": self.criteria.top_n,
            "score_stats": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores)
            },
            "sector_breakdown": self._get_sector_breakdown(),
            "top_stocks": [stock.symbol for stock in self.last_screening_results[:10]]
        }
    
    def backtest_screening(
        self,
        quote_client,
        start_date: str,
        end_date: str,
        rebalance_frequency: str = "monthly",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Backtest the screening strategy over a historical period.
        
        Args:
            quote_client: Tiger SDK quote client instance
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            rebalance_frequency: How often to rebalance ("daily", "weekly", "monthly")
            dry_run: If True, use placeholder data
            
        Returns:
            Dictionary with backtesting results
        """
        logger.info(f"[stock_screener] Starting backtest from {start_date} to {end_date}")
        
        # This is a simplified backtest framework
        # In a production system, you would want more sophisticated backtesting
        
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Generate rebalance dates
            rebalance_dates = self._generate_rebalance_dates(start_dt, end_dt, rebalance_frequency)
            
            backtest_results = {
                "period": f"{start_date} to {end_date}",
                "rebalance_frequency": rebalance_frequency,
                "rebalance_dates": len(rebalance_dates),
                "portfolios": [],
                "performance": {}
            }
            
            for date in rebalance_dates:
                logger.info(f"[stock_screener] Backtesting for date: {date.strftime('%Y-%m-%d')}")
                
                # For demonstration, use current screening logic
                # In practice, you would fetch historical data for the specific date
                selected_stocks = self.screen_stocks(quote_client, dry_run=dry_run)
                
                portfolio = {
                    "date": date.strftime('%Y-%m-%d'),
                    "stocks": [stock.symbol for stock in selected_stocks[:10]],
                    "scores": [stock.final_score for stock in selected_stocks[:10]]
                }
                backtest_results["portfolios"].append(portfolio)
            
            # Calculate simple performance metrics
            if backtest_results["portfolios"]:
                all_selected_stocks = set()
                for portfolio in backtest_results["portfolios"]:
                    all_selected_stocks.update(portfolio["stocks"])
                
                backtest_results["performance"] = {
                    "total_unique_stocks": len(all_selected_stocks),
                    "avg_portfolio_size": np.mean([len(p["stocks"]) for p in backtest_results["portfolios"]]),
                    "turnover_rate": self._calculate_turnover(backtest_results["portfolios"])
                }
            
            logger.info(f"[stock_screener] Backtest completed")
            return backtest_results
            
        except Exception as e:
            logger.error(f"[stock_screener] Backtest failed: {e}")
            return {"error": str(e)}
    
    def save_screening_results(self, filepath: str) -> bool:
        """
        Save last screening results to file.
        
        Args:
            filepath: Path to save results
            
        Returns:
            True if saved successfully
        """
        try:
            if not self.last_screening_results:
                logger.warning(f"[stock_screener] No results to save")
                return False
            
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "criteria": asdict(self.criteria),
                "results": [asdict(stock) for stock in self.last_screening_results]
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"[stock_screener] Results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[stock_screener] Failed to save results: {e}")
            return False
    
    def load_screening_results(self, filepath: str) -> bool:
        """
        Load screening results from file.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct StockScore objects
            self.last_screening_results = [
                StockScore(**result) for result in data["results"]
            ]
            
            logger.info(f"[stock_screener] Results loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[stock_screener] Failed to load results: {e}")
            return False
    
    def _get_screening_universe(self) -> List[str]:
        """Get universe of stocks to screen."""
        try:
            from .sector_manager import get_all_stocks, get_sector_stocks
            
            if self.criteria.sectors:
                # Screen specific sectors
                universe = set()
                for sector in self.criteria.sectors:
                    sector_stocks = get_sector_stocks(sector, validate=True)
                    universe.update(sector_stocks)
                return list(universe)
            else:
                # Screen all sectors
                return get_all_stocks(validate=True)
                
        except ImportError:
            logger.warning(f"[stock_screener] Sector manager not available, using default symbols")
            # Fallback to a default list
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]
    
    def _fetch_screening_data(
        self,
        quote_client,
        symbols: List[str],
        dry_run: bool
    ) -> Dict[str, pd.DataFrame]:
        """Fetch market data for screening."""
        # Get recent data for factor calculation
        return get_batch_latest_data(quote_client, symbols, dry_run)
    
    def _apply_basic_filters(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Apply basic filtering criteria."""
        filtered_symbols = []
        
        for symbol in symbols:
            # Skip excluded symbols
            if symbol in self.criteria.exclude_symbols:
                continue
                
            df = market_data.get(symbol)
            if df is None or df.empty:
                continue
            
            latest = df.iloc[-1]
            
            # Price filters
            price = float(latest['close'])
            if price < self.criteria.min_price or price > self.criteria.max_price:
                continue
            
            # Volume filter
            volume = int(latest['volume'])
            if volume < self.criteria.min_volume:
                continue
            
            # Get additional market data for market cap filter
            market_cap = self._estimate_market_cap(symbol, price)
            if market_cap < self.criteria.min_market_cap or market_cap > self.criteria.max_market_cap:
                continue
            
            filtered_symbols.append(symbol)
        
        return filtered_symbols
    
    def _calculate_factor_scores(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate factor scores for filtered symbols."""
        factor_scores = {}
        
        for symbol in symbols:
            df = market_data.get(symbol)
            if df is None or df.empty:
                continue
            
            scores = {
                'valuation': self._calculate_valuation_score(symbol, df),
                'volume': self._calculate_volume_score(symbol, df),
                'momentum': self._calculate_momentum_score(symbol, df),
                'quality': self._calculate_quality_score(symbol, df),
                'technical': self._calculate_technical_score(symbol, df)
            }
            
            factor_scores[symbol] = scores
        
        return factor_scores
    
    def _combine_scores(
        self,
        symbols: List[str],
        factor_scores: Dict[str, Dict[str, float]],
        market_data: Dict[str, pd.DataFrame]
    ) -> List[StockScore]:
        """Combine factor scores into final scores."""
        stock_scores = []
        
        for symbol in symbols:
            scores = factor_scores.get(symbol, {})
            df = market_data.get(symbol)
            
            if not scores or df is None or df.empty:
                continue
            
            latest = df.iloc[-1]
            
            # Calculate final score as weighted average
            final_score = (
                scores.get('valuation', 0) * self.criteria.valuation_weight +
                scores.get('volume', 0) * self.criteria.volume_weight +
                scores.get('momentum', 0) * self.criteria.momentum_weight +
                scores.get('quality', 0) * self.criteria.quality_weight +
                scores.get('technical', 0) * self.criteria.technical_weight
            )
            
            # Create StockScore object
            stock_score = StockScore(
                symbol=symbol,
                final_score=final_score,
                valuation_score=scores.get('valuation', 0),
                volume_score=scores.get('volume', 0),
                momentum_score=scores.get('momentum', 0),
                quality_score=scores.get('quality', 0),
                technical_score=scores.get('technical', 0),
                current_price=float(latest['close']),
                volume=int(latest['volume']),
                rsi=self._calculate_rsi(df),
                volatility=self._calculate_volatility(df),
                last_updated=datetime.now().isoformat()
            )
            
            stock_scores.append(stock_score)
        
        return stock_scores
    
    def _calculate_valuation_score(self, symbol: str, df: pd.DataFrame) -> float:
        """
        Calculate valuation factor score using fundamental data.

        Uses P/E ratio and P/B ratio from yfinance when available.
        Lower valuation multiples relative to sector peers = higher score.
        """
        try:
            import yfinance as yf

            # Fetch fundamental data
            ticker = yf.Ticker(symbol)
            info = ticker.info

            pe_ratio = info.get('trailingPE') or info.get('forwardPE')
            pb_ratio = info.get('priceToBook')
            ps_ratio = info.get('priceToSalesTrailing12Months')

            score = 0.0
            score_components = 0

            # P/E Score: lower P/E = higher score (capped at reasonable range)
            if pe_ratio and 0 < pe_ratio < 100:
                # Score inversely proportional to P/E
                # P/E of 10 -> score 0.8, P/E of 25 -> score 0.4, P/E of 50 -> score 0.2
                pe_score = max(0, min(1.0, 1.0 - (pe_ratio - 5) / 50))
                score += pe_score * 0.5  # 50% weight to P/E
                score_components += 0.5

            # P/B Score: lower P/B = higher score
            if pb_ratio and 0 < pb_ratio < 20:
                # P/B of 1 -> score 0.9, P/B of 5 -> score 0.5, P/B of 10 -> score 0.2
                pb_score = max(0, min(1.0, 1.0 - (pb_ratio - 0.5) / 10))
                score += pb_score * 0.3  # 30% weight to P/B
                score_components += 0.3

            # P/S Score: lower P/S = higher score
            if ps_ratio and 0 < ps_ratio < 30:
                ps_score = max(0, min(1.0, 1.0 - (ps_ratio - 1) / 15))
                score += ps_score * 0.2  # 20% weight to P/S
                score_components += 0.2

            # Normalize score if we have components
            if score_components > 0:
                normalized_score = score / score_components
                return round(normalized_score, 4)

            # Fallback: use price relative to 52-week range
            # Stocks near 52-week low may be more attractively valued
            high_52w = info.get('fiftyTwoWeekHigh', 0)
            low_52w = info.get('fiftyTwoWeekLow', 0)
            current_price = df['close'].iloc[-1] if len(df) > 0 else 0

            if high_52w > low_52w > 0 and current_price > 0:
                price_range = high_52w - low_52w
                price_position = (current_price - low_52w) / price_range
                # Lower position in range = higher valuation score (more attractive)
                return round(max(0, 1.0 - price_position), 4)

            return 0.5  # Neutral score if no data available

        except Exception as e:
            # Fallback: use simple price position in recent range
            try:
                if len(df) >= 20:
                    high_20d = df['close'].tail(20).max()
                    low_20d = df['close'].tail(20).min()
                    current = df['close'].iloc[-1]
                    if high_20d > low_20d:
                        position = (current - low_20d) / (high_20d - low_20d)
                        # Lower position = potentially better value
                        return round(max(0, 0.8 - position * 0.6), 4)
            except Exception:
                pass
            return 0.5  # Default neutral score
    
    def _calculate_volume_score(self, symbol: str, df: pd.DataFrame) -> float:
        """Calculate volume factor score using existing volume_factors module."""
        try:
            # Use existing volume_features function
            df_with_features = volume_features(df)
            if 'vol_score' in df_with_features.columns:
                return float(df_with_features['vol_score'].iloc[-1])
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_momentum_score(self, symbol: str, df: pd.DataFrame) -> float:
        """Calculate momentum factor score."""
        try:
            # Calculate multiple timeframe momentum
            returns = df['close'].pct_change().fillna(0)
            
            # Short-term momentum (5 days)
            short_momentum = returns.tail(5).sum()
            
            # Medium-term momentum (20 days) 
            medium_momentum = returns.tail(20).sum() if len(returns) >= 20 else short_momentum
            
            # Combine with more weight on recent momentum
            momentum_score = 0.7 * short_momentum + 0.3 * medium_momentum
            
            return float(momentum_score * 100)  # Scale up
        except Exception:
            return 0.0
    
    def _calculate_quality_score(self, symbol: str, df: pd.DataFrame) -> float:
        """Calculate quality factor score."""
        try:
            # Use price stability and volume consistency as quality proxies
            returns = df['close'].pct_change().fillna(0)
            volume_changes = df['volume'].pct_change().fillna(0)
            
            # Lower volatility and volume volatility indicate higher quality
            price_stability = 1 / (returns.std() + 0.01)  # Avoid division by zero
            volume_stability = 1 / (volume_changes.std() + 0.01)
            
            quality_score = 0.6 * price_stability + 0.4 * volume_stability
            
            return float(min(10, quality_score))  # Cap at 10
        except Exception:
            return 0.0
    
    def _calculate_technical_score(self, symbol: str, df: pd.DataFrame) -> float:
        """Calculate technical analysis score."""
        try:
            # Combine multiple technical indicators
            rsi = self._calculate_rsi(df)
            
            # RSI scoring: prefer moderate RSI values (not too high/low)
            if 30 <= rsi <= 70:
                rsi_score = 1.0
            elif rsi < 30:
                rsi_score = 0.5  # Oversold, potential reversal
            else:
                rsi_score = 0.2  # Overbought, risky
            
            # Moving average crossover
            if len(df) >= 20:
                sma_5 = df['close'].rolling(5).mean().iloc[-1]
                sma_20 = df['close'].rolling(20).mean().iloc[-1]
                ma_score = 1.0 if sma_5 > sma_20 else 0.5
            else:
                ma_score = 0.5
            
            # Combine technical scores
            technical_score = 0.6 * rsi_score + 0.4 * ma_score
            
            return float(technical_score * 10)  # Scale to 0-10 range
        except Exception:
            return 5.0  # Neutral score on error
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        try:
            if len(df) < period + 1:
                return 50.0  # Neutral RSI
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
        except Exception:
            return 50.0
    
    def _calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate price volatility."""
        try:
            if len(df) < period:
                return 0.0
            
            returns = df['close'].pct_change().fillna(0)
            volatility = returns.tail(period).std()
            
            return float(volatility * np.sqrt(252))  # Annualized
        except Exception:
            return 0.0
    
    def _estimate_market_cap(self, symbol: str, price: float) -> float:
        """
        Estimate market cap using real shares outstanding from yfinance.
        
        CRITICAL FIX #8: Replaced hardcoded 1B shares default with real data lookup.
        Falls back to yfinance marketCap directly if shares not available.
        """
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Priority 1: Use marketCap directly if available (most accurate)
            market_cap = info.get('marketCap')
            if market_cap and market_cap > 0:
                return float(market_cap)
            
            # Priority 2: Calculate from shares outstanding
            shares_outstanding = info.get('sharesOutstanding')
            if shares_outstanding and shares_outstanding > 0 and price > 0:
                return float(price * shares_outstanding)
            
            # Priority 3: Use implied shares from market cap
            implied_shares = info.get('impliedSharesOutstanding')
            if implied_shares and implied_shares > 0 and price > 0:
                return float(price * implied_shares)
            
            # Priority 4: Fallback to common known shares (frequently traded stocks)
            common_shares = {
                'AAPL': 15.3e9, 'MSFT': 7.4e9, 'GOOGL': 5.9e9, 'GOOG': 5.9e9,
                'AMZN': 10.5e9, 'TSLA': 3.2e9, 'META': 2.6e9, 'NVDA': 24.5e9,
                'BRK.B': 1.3e9, 'JPM': 2.9e9, 'V': 1.6e9, 'JNJ': 2.4e9
            }
            if symbol in common_shares:
                return float(price * common_shares[symbol])
            
            # Priority 5: If all else fails, log warning and return 0 (will be filtered out)
            logger.warning(f"[MARKET_CAP] Cannot determine market cap for {symbol} - will be filtered")
            return 0.0
            
        except Exception as e:
            logger.warning(f"[MARKET_CAP] Error fetching market cap for {symbol}: {e}")
            # Conservative fallback - return 0 so stock gets filtered out
            return 0.0
    
    def _get_sector_breakdown(self) -> Dict[str, int]:
        """Get sector breakdown of screening results."""
        sector_counts = {}
        for stock in self.last_screening_results:
            sector = stock.sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return sector_counts
    
    def _generate_rebalance_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> List[datetime]:
        """Generate rebalancing dates for backtesting."""
        dates = []
        current_date = start_date
        
        if frequency == "daily":
            delta = timedelta(days=1)
        elif frequency == "weekly":
            delta = timedelta(weeks=1)
        elif frequency == "monthly":
            delta = timedelta(days=30)  # Approximate
        else:
            delta = timedelta(days=30)  # Default to monthly
        
        while current_date <= end_date:
            dates.append(current_date)
            current_date += delta
        
        return dates
    
    def _calculate_turnover(self, portfolios: List[Dict]) -> float:
        """Calculate portfolio turnover rate."""
        if len(portfolios) < 2:
            return 0.0
        
        turnover_rates = []
        
        for i in range(1, len(portfolios)):
            prev_stocks = set(portfolios[i-1]["stocks"])
            curr_stocks = set(portfolios[i]["stocks"])
            
            # Turnover = (stocks added + stocks removed) / total positions
            added = len(curr_stocks - prev_stocks)
            removed = len(prev_stocks - curr_stocks)
            total_positions = len(prev_stocks)
            
            if total_positions > 0:
                turnover = (added + removed) / total_positions
                turnover_rates.append(turnover)
        
        return np.mean(turnover_rates) if turnover_rates else 0.0


# Convenience functions for easy usage

def screen_top_stocks(
    quote_client,
    top_n: int = 20,
    sectors: Optional[List[str]] = None,
    dry_run: bool = False
) -> List[StockScore]:
    """
    Convenience function to screen top stocks.
    
    Args:
        quote_client: Tiger SDK quote client instance
        top_n: Number of top stocks to return
        sectors: Specific sectors to screen
        dry_run: If True, use placeholder data
        
    Returns:
        List of top StockScore objects
    """
    criteria = ScreeningCriteria(top_n=top_n, sectors=sectors or [])
    screener = StockScreener(criteria)
    return screener.screen_stocks(quote_client, dry_run=dry_run)


def quick_sector_screen(
    quote_client,
    sector_name: str,
    top_n: int = 10,
    dry_run: bool = False
) -> List[StockScore]:
    """
    Convenience function to quickly screen a sector.
    
    Args:
        quote_client: Tiger SDK quote client instance
        sector_name: Name of the sector to screen
        top_n: Number of top stocks to return
        dry_run: If True, use placeholder data
        
    Returns:
        List of top StockScore objects from the sector
    """
    criteria = ScreeningCriteria(top_n=top_n)
    screener = StockScreener(criteria)
    return screener.screen_sector(quote_client, sector_name, dry_run)


def get_screening_candidates(
    quote_client,
    min_score: float = 0.0,
    dry_run: bool = False
) -> List[str]:
    """
    Get list of stock symbols that meet screening criteria.
    
    Args:
        quote_client: Tiger SDK quote client instance
        min_score: Minimum score threshold
        dry_run: If True, use placeholder data
        
    Returns:
        List of stock symbols meeting criteria
    """
    screener = StockScreener()
    results = screener.screen_stocks(quote_client, dry_run=dry_run)
    
    return [stock.symbol for stock in results if stock.final_score >= min_score]