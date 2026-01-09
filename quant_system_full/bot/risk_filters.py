"""
Comprehensive Risk Control and Filtering System

This module provides advanced risk management filters for the quantitative trading system:
- Volatility-based risk filtering
- Liquidity and market impact analysis
- Industry concentration limits
- Position sizing constraints
- Correlation-based risk management
- Market cap and fundamental screening

Integrates with the scoring engine, stock screener, and sector management systems
to provide comprehensive risk oversight for portfolio construction.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import json

# Optional statistical imports
try:
    from scipy.stats import pearsonr
    from scipy.stats import zscore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available, some statistical features disabled")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available, some clustering features disabled")

from .config import SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Configuration class for risk limits and thresholds."""
    
    # Volatility filters
    max_volatility: float = 2.0  # Maximum annualized volatility
    min_volatility: float = 0.05  # Minimum annualized volatility
    volatility_lookback: int = 60  # Days for volatility calculation
    volatility_percentile_filter: float = 0.95  # Remove top 5% most volatile
    
    # Liquidity filters
    min_avg_volume: int = 500000  # Minimum 20-day average volume
    min_dollar_volume: float = 10000000  # $10M minimum daily dollar volume
    max_bid_ask_spread: float = 0.03  # 3% maximum bid-ask spread
    liquidity_lookback: int = 20  # Days for liquidity calculation
    
    # Position sizing constraints
    max_single_position: float = 0.10  # 10% maximum single position
    max_sector_allocation: float = 0.25  # 25% maximum sector allocation
    max_concentration_top5: float = 0.40  # 40% maximum top 5 positions
    min_position_size: float = 0.01  # 1% minimum position size
    
    # Correlation filters
    max_pairwise_correlation: float = 0.8  # Maximum correlation between holdings
    correlation_lookback: int = 60  # Days for correlation calculation
    correlation_window: int = 252  # Rolling window for correlation
    
    # Market cap filters
    min_market_cap: float = 1e9  # $1B minimum market cap
    max_market_cap: float = 1e13  # $10T maximum market cap
    market_cap_concentration: float = 0.6  # Max 60% in single market cap tier
    
    # Fundamental screening filters
    max_price_to_book: float = 10.0  # Maximum P/B ratio
    min_price_to_book: float = 0.1   # Minimum P/B ratio
    max_debt_to_equity: float = 5.0   # Maximum debt-to-equity ratio
    min_current_ratio: float = 0.5    # Minimum current ratio
    
    # Sector diversification
    max_sector_positions: int = 8     # Maximum positions per sector
    min_sectors_represented: int = 3  # Minimum number of sectors
    sector_weight_limit: Dict[str, float] = field(default_factory=lambda: {
        "Technology": 0.30,
        "Healthcare": 0.25,
        "Financial": 0.25,
        "Consumer": 0.20,
        "Industrial": 0.20,
        "Energy": 0.15,
        "Utilities": 0.10,
        "Materials": 0.15,
        "Telecom": 0.10,
        "RealEstate": 0.10
    })
    
    # Risk budget allocation
    total_risk_budget: float = 0.20   # 20% maximum portfolio volatility
    max_component_risk: float = 0.05  # 5% maximum risk from single position
    var_confidence: float = 0.95      # 95% VaR confidence level
    
    # Market regime filters
    market_stress_threshold: float = 0.25  # VIX threshold for stress regime
    recession_filter_enabled: bool = True  # Enable recession indicator filtering
    bear_market_max_beta: float = 0.8     # Maximum beta during bear markets


@dataclass
class RiskMetrics:
    """Container for calculated risk metrics."""
    
    symbol: str
    
    # Volatility metrics
    volatility: float = 0.0
    volatility_percentile: float = 0.0
    
    # Liquidity metrics
    avg_volume: float = 0.0
    dollar_volume: float = 0.0
    bid_ask_spread: float = 0.0
    liquidity_score: float = 0.0
    
    # Market metrics
    market_cap: float = 0.0
    market_cap_tier: str = "Unknown"
    
    # Correlation metrics
    max_correlation: float = 0.0
    avg_correlation: float = 0.0
    
    # Fundamental metrics
    price_to_book: float = 0.0
    debt_to_equity: float = 0.0
    current_ratio: float = 0.0
    
    # Risk scores
    overall_risk_score: float = 0.0
    risk_tier: str = "Medium"
    
    # Flags
    passes_filters: bool = False
    filter_failures: List[str] = field(default_factory=list)


class RiskFilterEngine:
    """
    Comprehensive risk filtering engine for portfolio construction.
    
    This engine applies multiple layers of risk controls:
    1. Individual security risk filters
    2. Portfolio-level concentration limits
    3. Correlation-based position sizing
    4. Dynamic risk budget allocation
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize the risk filter engine.
        
        Args:
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()
        self.risk_history: List[Dict[str, RiskMetrics]] = []
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.sector_allocations: Dict[str, float] = {}
        
        logger.info("[risk_filters] Risk filter engine initialized")
    
    def apply_risk_filters(self,
                          symbols: List[str],
                          market_data: Dict[str, pd.DataFrame],
                          sector_mapping: Optional[Dict[str, str]] = None,
                          fundamental_data: Optional[Dict[str, Dict]] = None,
                          current_portfolio: Optional[Dict[str, float]] = None) -> Tuple[List[str], Dict[str, RiskMetrics]]:
        """
        Apply comprehensive risk filters to symbol universe.
        
        Args:
            symbols: List of symbols to filter
            market_data: Dictionary of symbol -> OHLCV DataFrame
            sector_mapping: Optional sector classification mapping
            fundamental_data: Optional fundamental data dictionary
            current_portfolio: Optional current portfolio weights
            
        Returns:
            Tuple of (filtered_symbols, risk_metrics)
        """
        logger.info(f"[risk_filters] Applying risk filters to {len(symbols)} symbols")
        
        # Calculate risk metrics for all symbols
        risk_metrics = self._calculate_risk_metrics(
            symbols, market_data, sector_mapping, fundamental_data
        )
        
        # Apply individual security filters
        filtered_symbols = self._apply_individual_filters(symbols, risk_metrics)
        
        logger.info(f"[risk_filters] {len(filtered_symbols)} symbols passed individual filters")
        
        # Apply correlation filters
        filtered_symbols = self._apply_correlation_filters(
            filtered_symbols, risk_metrics, market_data
        )
        
        logger.info(f"[risk_filters] {len(filtered_symbols)} symbols passed correlation filters")
        
        # Apply portfolio-level concentration limits
        if current_portfolio:
            filtered_symbols = self._apply_portfolio_filters(
                filtered_symbols, risk_metrics, current_portfolio, sector_mapping
            )
            
            logger.info(f"[risk_filters] {len(filtered_symbols)} symbols passed portfolio filters")
        
        # Store history
        self.risk_history.append(risk_metrics)
        if len(self.risk_history) > 60:  # Keep 60 periods of history
            self.risk_history = self.risk_history[-60:]
        
        logger.info(f"[risk_filters] Risk filtering completed: {len(filtered_symbols)} symbols selected")
        
        return filtered_symbols, risk_metrics
    
    def calculate_position_sizes(self,
                               symbols: List[str],
                               scores: Dict[str, float],
                               risk_metrics: Dict[str, RiskMetrics],
                               sector_mapping: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        Calculate risk-adjusted position sizes.
        
        Args:
            symbols: List of filtered symbols
            scores: Dictionary of symbol -> composite score
            risk_metrics: Dictionary of symbol -> risk metrics
            sector_mapping: Optional sector mapping
            
        Returns:
            Dictionary of symbol -> position weight
        """
        logger.info(f"[risk_filters] Calculating position sizes for {len(symbols)} symbols")
        
        if not symbols:
            return {}
        
        # Initialize position weights
        position_weights = {}
        total_score = sum(scores.get(symbol, 0) for symbol in symbols)
        
        if total_score <= 0:
            # Equal weight if no valid scores
            equal_weight = 1.0 / len(symbols)
            return {symbol: equal_weight for symbol in symbols}
        
        # Start with score-weighted positions
        for symbol in symbols:
            score_weight = scores.get(symbol, 0) / total_score
            position_weights[symbol] = score_weight
        
        # Apply risk-based position sizing adjustments
        position_weights = self._apply_volatility_scaling(position_weights, risk_metrics)
        position_weights = self._apply_concentration_limits(position_weights, sector_mapping)
        position_weights = self._apply_correlation_scaling(position_weights, risk_metrics)
        
        # Final normalization and validation
        position_weights = self._normalize_and_validate_weights(position_weights)
        
        logger.info("[risk_filters] Position sizing completed")
        
        return position_weights
    
    def validate_portfolio_risk(self,
                               positions: Dict[str, float],
                               market_data: Dict[str, pd.DataFrame],
                               risk_metrics: Dict[str, RiskMetrics],
                               sector_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Validate overall portfolio risk metrics.
        
        Args:
            positions: Dictionary of symbol -> weight
            market_data: Market data for symbols
            risk_metrics: Risk metrics for symbols  
            sector_mapping: Optional sector mapping
            
        Returns:
            Dictionary with portfolio risk validation results
        """
        logger.info("[risk_filters] Validating portfolio risk metrics")
        
        validation_results = {
            "passes_validation": True,
            "risk_violations": [],
            "risk_metrics": {},
            "recommendations": []
        }
        
        # Calculate portfolio volatility
        portfolio_vol = self._calculate_portfolio_volatility(positions, market_data)
        validation_results["risk_metrics"]["portfolio_volatility"] = portfolio_vol
        
        if portfolio_vol > self.limits.total_risk_budget:
            validation_results["passes_validation"] = False
            validation_results["risk_violations"].append(
                f"Portfolio volatility {portfolio_vol:.3f} exceeds limit {self.limits.total_risk_budget:.3f}"
            )
        
        # Check concentration limits
        concentration_results = self._check_concentration_limits(positions, sector_mapping)
        validation_results["risk_metrics"].update(concentration_results["metrics"])
        
        if concentration_results["violations"]:
            validation_results["passes_validation"] = False
            validation_results["risk_violations"].extend(concentration_results["violations"])
        
        # Check correlation risk
        correlation_risk = self._check_correlation_risk(positions, risk_metrics)
        validation_results["risk_metrics"]["correlation_risk"] = correlation_risk
        
        if correlation_risk > 0.5:  # High correlation risk threshold
            validation_results["risk_violations"].append(
                f"High correlation risk {correlation_risk:.3f} detected"
            )
        
        # Generate recommendations
        if validation_results["risk_violations"]:
            validation_results["recommendations"] = self._generate_risk_recommendations(
                validation_results["risk_violations"], positions
            )
        
        logger.info(f"[risk_filters] Portfolio risk validation completed: {'PASSED' if validation_results['passes_validation'] else 'FAILED'}")
        
        return validation_results
    
    def get_risk_report(self, 
                       positions: Dict[str, float],
                       risk_metrics: Dict[str, RiskMetrics]) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Args:
            positions: Current portfolio positions
            risk_metrics: Risk metrics for positions
            
        Returns:
            Dictionary with comprehensive risk analysis
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_summary": self._get_portfolio_summary(positions, risk_metrics),
            "risk_breakdown": self._get_risk_breakdown(positions, risk_metrics),
            "concentration_analysis": self._get_concentration_analysis(positions),
            "liquidity_analysis": self._get_liquidity_analysis(positions, risk_metrics),
            "stress_test_results": self._perform_stress_tests(positions, risk_metrics)
        }
        
        return report
    
    def _calculate_risk_metrics(self,
                              symbols: List[str],
                              market_data: Dict[str, pd.DataFrame],
                              sector_mapping: Optional[Dict[str, str]],
                              fundamental_data: Optional[Dict[str, Dict]]) -> Dict[str, RiskMetrics]:
        """Calculate comprehensive risk metrics for all symbols."""
        risk_metrics = {}
        
        for symbol in symbols:
            df = market_data.get(symbol)
            if df is None or df.empty:
                continue
            
            try:
                metrics = RiskMetrics(symbol=symbol)
                
                # Calculate volatility metrics
                self._calculate_volatility_metrics(metrics, df)
                
                # Calculate liquidity metrics
                self._calculate_liquidity_metrics(metrics, df)
                
                # Calculate market cap metrics
                self._calculate_market_cap_metrics(metrics, df)
                
                # Calculate fundamental metrics if available
                if fundamental_data and symbol in fundamental_data:
                    self._calculate_fundamental_metrics(metrics, fundamental_data[symbol])
                
                # Calculate overall risk score
                self._calculate_overall_risk_score(metrics)
                
                # Apply individual filters
                self._apply_individual_security_filters(metrics)
                
                risk_metrics[symbol] = metrics
                
            except Exception as e:
                logger.warning(f"[risk_filters] Failed to calculate metrics for {symbol}: {e}")
                continue
        
        # Calculate cross-sectional correlation metrics
        if len(risk_metrics) > 1:
            self._calculate_correlation_metrics(risk_metrics, market_data)
        
        return risk_metrics
    
    def _calculate_volatility_metrics(self, metrics: RiskMetrics, df: pd.DataFrame):
        """Calculate volatility-based risk metrics."""
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < self.limits.volatility_lookback:
            metrics.volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
        else:
            # Use specified lookback period
            recent_returns = returns.tail(self.limits.volatility_lookback)
            metrics.volatility = recent_returns.std() * np.sqrt(252)
        
        # Calculate volatility percentile relative to history
        if len(returns) > 60:  # Need sufficient history
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else metrics.volatility
            vol_history = rolling_vol.dropna()
            if len(vol_history) > 0:
                metrics.volatility_percentile = (vol_history <= current_vol).mean()
    
    def _calculate_liquidity_metrics(self, metrics: RiskMetrics, df: pd.DataFrame):
        """Calculate liquidity-based risk metrics."""
        # Average volume
        volumes = df['volume'].tail(self.limits.liquidity_lookback)
        metrics.avg_volume = volumes.mean()
        
        # Dollar volume
        prices = df['close'].tail(self.limits.liquidity_lookback)
        dollar_volumes = prices * volumes
        metrics.dollar_volume = dollar_volumes.mean()
        
        # Simple bid-ask spread estimate (using high-low as proxy)
        if 'high' in df.columns and 'low' in df.columns:
            spreads = (df['high'] - df['low']) / df['close']
            metrics.bid_ask_spread = spreads.tail(self.limits.liquidity_lookback).mean()
        
        # Liquidity score (normalized)
        volume_score = min(1.0, metrics.avg_volume / 1e6)  # Normalize by 1M shares
        dollar_score = min(1.0, metrics.dollar_volume / 50e6)  # Normalize by $50M
        spread_score = max(0.0, 1.0 - metrics.bid_ask_spread / 0.02)  # Penalize spreads > 2%
        
        metrics.liquidity_score = (volume_score + dollar_score + spread_score) / 3
    
    def _calculate_market_cap_metrics(self, metrics: RiskMetrics, df: pd.DataFrame):
        """
        Calculate market cap related metrics using real data.
        
        CRITICAL FIX #8: Replaced hardcoded 1B shares with yfinance lookup.
        """
        current_price = df['close'].iloc[-1]
        
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(metrics.symbol)
            info = ticker.info
            
            # Priority 1: Use marketCap directly if available
            market_cap = info.get('marketCap')
            if market_cap and market_cap > 0:
                metrics.market_cap = float(market_cap)
            else:
                # Priority 2: Calculate from shares outstanding
                shares_outstanding = info.get('sharesOutstanding')
                if shares_outstanding and shares_outstanding > 0:
                    metrics.market_cap = float(current_price * shares_outstanding)
                else:
                    # Fallback for common stocks
                    common_shares = {
                        'AAPL': 15.3e9, 'MSFT': 7.4e9, 'GOOGL': 5.9e9, 'GOOG': 5.9e9,
                        'AMZN': 10.5e9, 'TSLA': 3.2e9, 'META': 2.6e9, 'NVDA': 24.5e9,
                        'JPM': 2.9e9, 'V': 1.6e9, 'JNJ': 2.4e9
                    }
                    shares = common_shares.get(metrics.symbol)
                    if shares:
                        metrics.market_cap = float(current_price * shares)
                    else:
                        # Unknown stock - mark as 0 for filtering
                        metrics.market_cap = 0.0
                        
        except Exception as e:
            # On error, use 0 to be conservative
            metrics.market_cap = 0.0
        
        # Classify market cap tier
        if metrics.market_cap >= 200e9:  # $200B+
            metrics.market_cap_tier = "Mega"
        elif metrics.market_cap >= 50e9:  # $50B-$200B
            metrics.market_cap_tier = "Large"
        elif metrics.market_cap >= 10e9:  # $10B-$50B
            metrics.market_cap_tier = "Mid"
        elif metrics.market_cap >= 2e9:   # $2B-$10B
            metrics.market_cap_tier = "Small"
        else:
            metrics.market_cap_tier = "Micro"
    
    def _calculate_fundamental_metrics(self, metrics: RiskMetrics, fundamental_data: Dict):
        """Calculate fundamental risk metrics."""
        metrics.price_to_book = fundamental_data.get('price_to_book', 0.0)
        metrics.debt_to_equity = fundamental_data.get('debt_to_equity', 0.0) 
        metrics.current_ratio = fundamental_data.get('current_ratio', 1.0)
    
    def _calculate_overall_risk_score(self, metrics: RiskMetrics):
        """Calculate composite risk score."""
        # Volatility component (higher vol = higher risk)
        vol_risk = min(1.0, metrics.volatility / 2.0)  # Normalize by 200% vol
        
        # Liquidity component (lower liquidity = higher risk)
        liquidity_risk = 1.0 - metrics.liquidity_score
        
        # Market cap component (smaller cap = higher risk)
        if metrics.market_cap_tier == "Mega":
            cap_risk = 0.1
        elif metrics.market_cap_tier == "Large":
            cap_risk = 0.3
        elif metrics.market_cap_tier == "Mid":
            cap_risk = 0.5
        elif metrics.market_cap_tier == "Small":
            cap_risk = 0.7
        else:  # Micro
            cap_risk = 0.9
        
        # Combine components
        metrics.overall_risk_score = (0.4 * vol_risk + 0.4 * liquidity_risk + 0.2 * cap_risk)
        
        # Assign risk tier
        if metrics.overall_risk_score >= 0.7:
            metrics.risk_tier = "High"
        elif metrics.overall_risk_score >= 0.4:
            metrics.risk_tier = "Medium"
        else:
            metrics.risk_tier = "Low"
    
    def _calculate_correlation_metrics(self, risk_metrics: Dict[str, RiskMetrics], 
                                     market_data: Dict[str, pd.DataFrame]):
        """Calculate cross-sectional correlation metrics."""
        symbols = list(risk_metrics.keys())
        
        if len(symbols) < 2:
            return
        
        # Build returns matrix
        returns_data = {}
        
        for symbol in symbols:
            df = market_data.get(symbol)
            if df is not None and not df.empty:
                returns = df['close'].pct_change().dropna()
                if len(returns) >= self.limits.correlation_lookback:
                    returns_data[symbol] = returns.tail(self.limits.correlation_lookback)
        
        if len(returns_data) < 2:
            return
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()  # Remove dates with missing data
        
        if returns_df.empty or len(returns_df) < 20:  # Need minimum data
            return
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        self.correlation_matrix = correlation_matrix
        
        # Calculate correlation metrics for each symbol
        for symbol in symbols:
            if symbol not in correlation_matrix.index:
                continue
                
            correlations = correlation_matrix.loc[symbol].drop(symbol)
            if not correlations.empty:
                risk_metrics[symbol].max_correlation = correlations.abs().max()
                risk_metrics[symbol].avg_correlation = correlations.abs().mean()
    
    def _apply_individual_security_filters(self, metrics: RiskMetrics):
        """Apply individual security risk filters."""
        metrics.passes_filters = True
        metrics.filter_failures = []
        
        # Volatility filters
        if metrics.volatility > self.limits.max_volatility:
            metrics.passes_filters = False
            metrics.filter_failures.append(f"Volatility {metrics.volatility:.3f} > {self.limits.max_volatility}")
        
        if metrics.volatility < self.limits.min_volatility:
            metrics.passes_filters = False  
            metrics.filter_failures.append(f"Volatility {metrics.volatility:.3f} < {self.limits.min_volatility}")
        
        # Liquidity filters
        if metrics.avg_volume < self.limits.min_avg_volume:
            metrics.passes_filters = False
            metrics.filter_failures.append(f"Volume {metrics.avg_volume:.0f} < {self.limits.min_avg_volume}")
        
        if metrics.dollar_volume < self.limits.min_dollar_volume:
            metrics.passes_filters = False
            metrics.filter_failures.append(f"Dollar volume {metrics.dollar_volume:.0f} < {self.limits.min_dollar_volume}")
        
        if metrics.bid_ask_spread > self.limits.max_bid_ask_spread:
            metrics.passes_filters = False
            metrics.filter_failures.append(f"Spread {metrics.bid_ask_spread:.4f} > {self.limits.max_bid_ask_spread}")
        
        # Market cap filters  
        if metrics.market_cap < self.limits.min_market_cap:
            metrics.passes_filters = False
            metrics.filter_failures.append(f"Market cap {metrics.market_cap:.0f} < {self.limits.min_market_cap}")
        
        if metrics.market_cap > self.limits.max_market_cap:
            metrics.passes_filters = False
            metrics.filter_failures.append(f"Market cap {metrics.market_cap:.0f} > {self.limits.max_market_cap}")
        
        # Fundamental filters (if data available)
        if metrics.price_to_book > 0:
            if metrics.price_to_book > self.limits.max_price_to_book:
                metrics.passes_filters = False
                metrics.filter_failures.append(f"P/B {metrics.price_to_book:.2f} > {self.limits.max_price_to_book}")
            
            if metrics.price_to_book < self.limits.min_price_to_book:
                metrics.passes_filters = False
                metrics.filter_failures.append(f"P/B {metrics.price_to_book:.2f} < {self.limits.min_price_to_book}")
        
        if metrics.debt_to_equity > self.limits.max_debt_to_equity:
            metrics.passes_filters = False
            metrics.filter_failures.append(f"D/E {metrics.debt_to_equity:.2f} > {self.limits.max_debt_to_equity}")
    
    def _apply_individual_filters(self, symbols: List[str], 
                                risk_metrics: Dict[str, RiskMetrics]) -> List[str]:
        """Filter symbols based on individual security risk criteria."""
        filtered = []
        
        for symbol in symbols:
            metrics = risk_metrics.get(symbol)
            if metrics and metrics.passes_filters:
                filtered.append(symbol)
            elif metrics and not metrics.passes_filters:
                logger.debug(f"[risk_filters] {symbol} filtered: {', '.join(metrics.filter_failures)}")
        
        return filtered
    
    def _apply_correlation_filters(self, symbols: List[str],
                                 risk_metrics: Dict[str, RiskMetrics],
                                 market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Apply correlation-based filters."""
        if not self.correlation_matrix or self.correlation_matrix.empty:
            return symbols  # No correlation data available
        
        filtered = []
        selected_symbols = set()
        
        # Sort symbols by overall quality (inverse risk score)
        symbols_by_quality = sorted(symbols, 
                                  key=lambda s: risk_metrics.get(s, RiskMetrics(s)).overall_risk_score)
        
        for symbol in symbols_by_quality:
            if symbol not in self.correlation_matrix.index:
                filtered.append(symbol)
                selected_symbols.add(symbol)
                continue
            
            # Check correlation with already selected symbols
            high_correlation = False
            
            for selected in selected_symbols:
                if selected in self.correlation_matrix.index:
                    correlation = abs(self.correlation_matrix.loc[symbol, selected])
                    
                    if correlation > self.limits.max_pairwise_correlation:
                        high_correlation = True
                        logger.debug(f"[risk_filters] {symbol} filtered: high correlation {correlation:.3f} with {selected}")
                        break
            
            if not high_correlation:
                filtered.append(symbol)
                selected_symbols.add(symbol)
        
        return filtered
    
    def _apply_portfolio_filters(self, symbols: List[str],
                               risk_metrics: Dict[str, RiskMetrics],
                               current_portfolio: Dict[str, float],
                               sector_mapping: Optional[Dict[str, str]]) -> List[str]:
        """Apply portfolio-level concentration filters."""
        if not sector_mapping:
            return symbols  # Can't apply sector filters without mapping
        
        # Calculate current sector allocations
        current_sector_weights = {}
        for symbol, weight in current_portfolio.items():
            sector = sector_mapping.get(symbol, "Unknown")
            current_sector_weights[sector] = current_sector_weights.get(sector, 0) + weight
        
        # Filter based on sector concentration limits
        filtered = []
        
        for symbol in symbols:
            sector = sector_mapping.get(symbol, "Unknown")
            sector_limit = self.limits.sector_weight_limit.get(sector, self.limits.max_sector_allocation)
            current_sector_weight = current_sector_weights.get(sector, 0)
            
            # Allow symbol if it won't violate sector limits
            if current_sector_weight < sector_limit:
                filtered.append(symbol)
            else:
                logger.debug(f"[risk_filters] {symbol} filtered: sector {sector} at limit {current_sector_weight:.3f}")
        
        return filtered
    
    def _apply_volatility_scaling(self, weights: Dict[str, float],
                                risk_metrics: Dict[str, RiskMetrics]) -> Dict[str, float]:
        """Scale position sizes based on volatility."""
        scaled_weights = {}
        
        for symbol, weight in weights.items():
            metrics = risk_metrics.get(symbol)
            if not metrics:
                scaled_weights[symbol] = weight
                continue
            
            # Inverse volatility scaling (higher vol = smaller position)
            if metrics.volatility > 0:
                vol_scalar = min(1.0, 0.15 / metrics.volatility)  # Target 15% vol
                scaled_weight = weight * vol_scalar
                scaled_weights[symbol] = max(scaled_weight, self.limits.min_position_size)
            else:
                scaled_weights[symbol] = weight
        
        return scaled_weights
    
    def _apply_concentration_limits(self, weights: Dict[str, float],
                                  sector_mapping: Optional[Dict[str, str]]) -> Dict[str, float]:
        """Apply concentration limits to position weights."""
        # Apply single position limit
        for symbol in weights:
            if weights[symbol] > self.limits.max_single_position:
                weights[symbol] = self.limits.max_single_position
        
        # Apply sector concentration limits if mapping available
        if sector_mapping:
            sector_weights = {}
            
            # Calculate current sector weights
            for symbol, weight in weights.items():
                sector = sector_mapping.get(symbol, "Unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            # Scale down if sector limits exceeded
            for sector, total_weight in sector_weights.items():
                sector_limit = self.limits.sector_weight_limit.get(sector, self.limits.max_sector_allocation)
                
                if total_weight > sector_limit:
                    scale_factor = sector_limit / total_weight
                    
                    # Apply scaling to all symbols in the sector
                    for symbol, weight in weights.items():
                        if sector_mapping.get(symbol) == sector:
                            weights[symbol] *= scale_factor
        
        return weights
    
    def _apply_correlation_scaling(self, weights: Dict[str, float],
                                 risk_metrics: Dict[str, RiskMetrics]) -> Dict[str, float]:
        """Scale positions based on correlation risk."""
        if not self.correlation_matrix or self.correlation_matrix.empty:
            return weights
        
        # Calculate correlation-adjusted weights
        adjusted_weights = weights.copy()
        
        for symbol in weights:
            if symbol not in self.correlation_matrix.index:
                continue
            
            # Get correlations with other holdings
            symbol_correlations = []
            for other_symbol in weights:
                if other_symbol != symbol and other_symbol in self.correlation_matrix.index:
                    correlation = abs(self.correlation_matrix.loc[symbol, other_symbol])
                    weight_product = weights[symbol] * weights[other_symbol]
                    symbol_correlations.append(correlation * weight_product)
            
            # Scale down if high correlation risk
            if symbol_correlations:
                avg_correlation_risk = np.mean(symbol_correlations)
                
                if avg_correlation_risk > 0.1:  # High correlation risk threshold
                    scale_factor = max(0.5, 1.0 - avg_correlation_risk)
                    adjusted_weights[symbol] *= scale_factor
        
        return adjusted_weights
    
    def _normalize_and_validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Final normalization and validation of position weights."""
        # Remove positions below minimum size
        filtered_weights = {
            symbol: weight for symbol, weight in weights.items() 
            if weight >= self.limits.min_position_size
        }
        
        if not filtered_weights:
            return {}
        
        # Normalize to sum to 1.0
        total_weight = sum(filtered_weights.values())
        if total_weight > 0:
            normalized_weights = {
                symbol: weight / total_weight 
                for symbol, weight in filtered_weights.items()
            }
        else:
            normalized_weights = {}
        
        # Final validation
        validated_weights = {}
        
        for symbol, weight in normalized_weights.items():
            # Ensure within bounds
            validated_weight = np.clip(weight, self.limits.min_position_size, self.limits.max_single_position)
            validated_weights[symbol] = validated_weight
        
        # Re-normalize after clipping
        total_validated = sum(validated_weights.values())
        if total_validated > 0:
            final_weights = {
                symbol: weight / total_validated 
                for symbol, weight in validated_weights.items()
            }
        else:
            final_weights = {}
        
        return final_weights
    
    def _calculate_portfolio_volatility(self, positions: Dict[str, float],
                                      market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate portfolio volatility."""
        if not positions or not self.correlation_matrix or self.correlation_matrix.empty:
            return 0.0
        
        # Calculate individual volatilities
        volatilities = {}
        for symbol, weight in positions.items():
            df = market_data.get(symbol)
            if df is not None and not df.empty:
                returns = df['close'].pct_change().dropna()
                if len(returns) > 20:
                    vol = returns.std() * np.sqrt(252)
                    volatilities[symbol] = vol
        
        if not volatilities:
            return 0.0
        
        # Calculate portfolio variance using correlation matrix
        portfolio_variance = 0.0
        
        for symbol1, weight1 in positions.items():
            for symbol2, weight2 in positions.items():
                vol1 = volatilities.get(symbol1, 0)
                vol2 = volatilities.get(symbol2, 0)
                
                if symbol1 == symbol2:
                    correlation = 1.0
                elif (symbol1 in self.correlation_matrix.index and 
                      symbol2 in self.correlation_matrix.columns):
                    correlation = self.correlation_matrix.loc[symbol1, symbol2]
                else:
                    correlation = 0.0
                
                portfolio_variance += weight1 * weight2 * vol1 * vol2 * correlation
        
        portfolio_volatility = np.sqrt(max(0, portfolio_variance))
        return portfolio_volatility
    
    def _check_concentration_limits(self, positions: Dict[str, float],
                                  sector_mapping: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Check portfolio concentration limits."""
        results = {
            "metrics": {},
            "violations": []
        }
        
        # Single position concentration
        max_position = max(positions.values()) if positions else 0
        results["metrics"]["max_single_position"] = max_position
        
        if max_position > self.limits.max_single_position:
            results["violations"].append(
                f"Single position {max_position:.3f} exceeds limit {self.limits.max_single_position:.3f}"
            )
        
        # Top 5 concentration
        sorted_positions = sorted(positions.values(), reverse=True)
        top5_concentration = sum(sorted_positions[:5]) if len(sorted_positions) >= 5 else sum(sorted_positions)
        results["metrics"]["top5_concentration"] = top5_concentration
        
        if top5_concentration > self.limits.max_concentration_top5:
            results["violations"].append(
                f"Top 5 concentration {top5_concentration:.3f} exceeds limit {self.limits.max_concentration_top5:.3f}"
            )
        
        # Sector concentration if mapping available
        if sector_mapping:
            sector_weights = {}
            for symbol, weight in positions.items():
                sector = sector_mapping.get(symbol, "Unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            results["metrics"]["sector_weights"] = sector_weights
            
            for sector, weight in sector_weights.items():
                sector_limit = self.limits.sector_weight_limit.get(sector, self.limits.max_sector_allocation)
                if weight > sector_limit:
                    results["violations"].append(
                        f"Sector {sector} weight {weight:.3f} exceeds limit {sector_limit:.3f}"
                    )
        
        return results
    
    def _check_correlation_risk(self, positions: Dict[str, float],
                              risk_metrics: Dict[str, RiskMetrics]) -> float:
        """Calculate overall correlation risk of portfolio."""
        if not self.correlation_matrix or self.correlation_matrix.empty or len(positions) < 2:
            return 0.0
        
        total_correlation_risk = 0.0
        position_pairs = 0
        
        for symbol1, weight1 in positions.items():
            for symbol2, weight2 in positions.items():
                if (symbol1 != symbol2 and 
                    symbol1 in self.correlation_matrix.index and 
                    symbol2 in self.correlation_matrix.columns):
                    
                    correlation = abs(self.correlation_matrix.loc[symbol1, symbol2])
                    weight_product = weight1 * weight2
                    correlation_contribution = correlation * weight_product
                    total_correlation_risk += correlation_contribution
                    position_pairs += 1
        
        if position_pairs > 0:
            avg_correlation_risk = total_correlation_risk / position_pairs
            return avg_correlation_risk
        
        return 0.0
    
    def _generate_risk_recommendations(self, violations: List[str], 
                                     positions: Dict[str, float]) -> List[str]:
        """Generate recommendations to address risk violations."""
        recommendations = []
        
        for violation in violations:
            if "Portfolio volatility" in violation:
                recommendations.append("Consider reducing position sizes or adding lower volatility assets")
            
            elif "Single position" in violation:
                largest_position = max(positions, key=positions.get)
                recommendations.append(f"Reduce position size in {largest_position}")
            
            elif "Top 5 concentration" in violation:
                recommendations.append("Diversify portfolio by adding more positions or reducing top holdings")
            
            elif "Sector" in violation:
                recommendations.append("Rebalance sector allocations to meet concentration limits")
            
            elif "correlation risk" in violation:
                recommendations.append("Add uncorrelated assets or reduce correlated position sizes")
        
        return recommendations
    
    def _get_portfolio_summary(self, positions: Dict[str, float], 
                             risk_metrics: Dict[str, RiskMetrics]) -> Dict[str, Any]:
        """Generate portfolio summary statistics."""
        return {
            "total_positions": len(positions),
            "total_weight": sum(positions.values()),
            "average_risk_score": np.mean([risk_metrics.get(s, RiskMetrics(s)).overall_risk_score 
                                         for s in positions]),
            "risk_tier_distribution": self._get_risk_tier_distribution(positions, risk_metrics)
        }
    
    def _get_risk_breakdown(self, positions: Dict[str, float], 
                          risk_metrics: Dict[str, RiskMetrics]) -> Dict[str, Any]:
        """Generate detailed risk breakdown."""
        breakdown = {}
        
        for symbol, weight in positions.items():
            metrics = risk_metrics.get(symbol, RiskMetrics(symbol))
            breakdown[symbol] = {
                "weight": weight,
                "volatility": metrics.volatility,
                "liquidity_score": metrics.liquidity_score,
                "risk_tier": metrics.risk_tier,
                "risk_contribution": weight * metrics.overall_risk_score
            }
        
        return breakdown
    
    def _get_concentration_analysis(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Generate concentration analysis."""
        sorted_positions = sorted(positions.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "top_5_holdings": sorted_positions[:5],
            "top_5_weight": sum([w for _, w in sorted_positions[:5]]),
            "herfindahl_index": sum([w**2 for w in positions.values()])  # Concentration measure
        }
    
    def _get_liquidity_analysis(self, positions: Dict[str, float], 
                              risk_metrics: Dict[str, RiskMetrics]) -> Dict[str, Any]:
        """Generate liquidity analysis."""
        liquidity_scores = []
        weighted_liquidity = 0.0
        
        for symbol, weight in positions.items():
            metrics = risk_metrics.get(symbol, RiskMetrics(symbol))
            liquidity_scores.append(metrics.liquidity_score)
            weighted_liquidity += weight * metrics.liquidity_score
        
        return {
            "weighted_liquidity_score": weighted_liquidity,
            "min_liquidity_score": min(liquidity_scores) if liquidity_scores else 0,
            "avg_liquidity_score": np.mean(liquidity_scores) if liquidity_scores else 0
        }
    
    def _get_risk_tier_distribution(self, positions: Dict[str, float], 
                                   risk_metrics: Dict[str, RiskMetrics]) -> Dict[str, float]:
        """Get distribution of positions by risk tier."""
        tier_weights = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
        
        for symbol, weight in positions.items():
            metrics = risk_metrics.get(symbol, RiskMetrics(symbol))
            tier_weights[metrics.risk_tier] += weight
        
        return tier_weights
    
    def _perform_stress_tests(self, positions: Dict[str, float], 
                            risk_metrics: Dict[str, RiskMetrics]) -> Dict[str, Any]:
        """Perform simple stress tests on portfolio."""
        # This is a simplified stress testing framework
        # In production, you would use more sophisticated scenarios
        
        stress_results = {}
        
        # Volatility stress test: assume 2x volatility increase
        vol_stress_impact = 0.0
        for symbol, weight in positions.items():
            metrics = risk_metrics.get(symbol, RiskMetrics(symbol))
            vol_stress_impact += weight * metrics.volatility * 2.0
        
        stress_results["volatility_stress_2x"] = vol_stress_impact
        
        # Liquidity stress test: assume liquidity drops by 50%
        liquidity_stress = 0.0
        for symbol, weight in positions.items():
            metrics = risk_metrics.get(symbol, RiskMetrics(symbol))
            liquidity_stress += weight * max(0, metrics.liquidity_score - 0.5)
        
        stress_results["liquidity_stress"] = liquidity_stress
        
        return stress_results
    
    def save_risk_configuration(self, filepath: str):
        """Save risk limits configuration to file."""
        config = {
            "max_volatility": self.limits.max_volatility,
            "min_volatility": self.limits.min_volatility,
            "volatility_lookback": self.limits.volatility_lookback,
            "min_avg_volume": self.limits.min_avg_volume,
            "min_dollar_volume": self.limits.min_dollar_volume,
            "max_bid_ask_spread": self.limits.max_bid_ask_spread,
            "max_single_position": self.limits.max_single_position,
            "max_sector_allocation": self.limits.max_sector_allocation,
            "max_pairwise_correlation": self.limits.max_pairwise_correlation,
            "min_market_cap": self.limits.min_market_cap,
            "max_market_cap": self.limits.max_market_cap,
            "sector_weight_limits": self.limits.sector_weight_limit,
            "total_risk_budget": self.limits.total_risk_budget
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"[risk_filters] Configuration saved to {filepath}")
    
    @classmethod
    def load_risk_configuration(cls, filepath: str) -> 'RiskFilterEngine':
        """Load risk configuration from file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        limits = RiskLimits()
        
        # Update limits with loaded config
        for key, value in config.items():
            if hasattr(limits, key):
                setattr(limits, key, value)
        
        logger.info(f"[risk_filters] Configuration loaded from {filepath}")
        
        return cls(limits)


# Utility functions for integration with existing systems

def apply_basic_risk_filters(symbols: List[str], 
                           market_data: Dict[str, pd.DataFrame],
                           limits: Optional[RiskLimits] = None) -> List[str]:
    """
    Apply basic risk filters to a list of symbols.
    
    Args:
        symbols: List of symbols to filter
        market_data: Market data for symbols
        limits: Optional risk limits
        
    Returns:
        List of symbols passing basic risk filters
    """
    engine = RiskFilterEngine(limits)
    filtered_symbols, _ = engine.apply_risk_filters(symbols, market_data)
    return filtered_symbols


def calculate_risk_adjusted_weights(symbols: List[str],
                                  scores: Dict[str, float],
                                  market_data: Dict[str, pd.DataFrame],
                                  sector_mapping: Optional[Dict[str, str]] = None,
                                  limits: Optional[RiskLimits] = None) -> Dict[str, float]:
    """
    Calculate risk-adjusted position weights.
    
    Args:
        symbols: List of symbols
        scores: Composite scores for symbols
        market_data: Market data
        sector_mapping: Optional sector mapping
        limits: Optional risk limits
        
    Returns:
        Dictionary of symbol -> risk-adjusted weight
    """
    engine = RiskFilterEngine(limits)
    
    # Calculate risk metrics
    risk_metrics = engine._calculate_risk_metrics(symbols, market_data, sector_mapping, None)
    
    # Calculate risk-adjusted weights
    weights = engine.calculate_position_sizes(symbols, scores, risk_metrics, sector_mapping)
    
    return weights


def validate_portfolio_risk_limits(positions: Dict[str, float],
                                 market_data: Dict[str, pd.DataFrame],
                                 limits: Optional[RiskLimits] = None) -> bool:
    """
    Validate if portfolio meets risk limits.
    
    Args:
        positions: Portfolio positions
        market_data: Market data
        limits: Optional risk limits
        
    Returns:
        True if portfolio passes risk validation
    """
    engine = RiskFilterEngine(limits)
    
    symbols = list(positions.keys())
    risk_metrics = engine._calculate_risk_metrics(symbols, market_data, None, None)
    
    validation_result = engine.validate_portfolio_risk(positions, market_data, risk_metrics)
    
    return validation_result["passes_validation"]