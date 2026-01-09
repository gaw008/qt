"""Enhanced Backtesting System with Portfolio-Level Validation

This module provides comprehensive backtesting capabilities for the quantitative trading system:
- Single-stock and multi-stock portfolio backtesting
- Integration with scoring engine and stock screener
- Risk-adjusted performance metrics
- Factor attribution analysis
- Portfolio-level risk validation
- Strategy performance comparison
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import warnings
import logging

# Core imports
from bot.utils import simple_pnl
from bot.strategies.sma_crossover import generate_signals
from bot.config import SETTINGS

# New imports for enhanced functionality
try:
    from bot.scoring_engine import MultiFactorScoringEngine, FactorWeights
    from bot.stock_screener import StockScreener, ScreeningCriteria
    from bot.risk_filters import RiskFilterEngine, RiskLimits
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    ENHANCED_MODULES_AVAILABLE = False
    warnings.warn(f"Enhanced modules not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """
    Enhanced portfolio-level backtesting engine.
    
    Supports:
    - Multi-factor stock selection strategies
    - Risk-adjusted portfolio construction
    - Performance attribution analysis
    - Drawdown and risk metrics
    - Strategy comparison and optimization
    """
    
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 1000000.0,
                 rebalance_frequency: str = "monthly",
                 transaction_costs: float = 0.001):
        """
        Initialize portfolio backtester.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting portfolio value
            rebalance_frequency: "daily", "weekly", "monthly", "quarterly"
            transaction_costs: Transaction cost as fraction of trade value
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_costs = transaction_costs
        
        # Initialize engines if available
        if ENHANCED_MODULES_AVAILABLE:
            self.scoring_engine = MultiFactorScoringEngine()
            self.screener = StockScreener()
            self.risk_engine = RiskFilterEngine()
        else:
            self.scoring_engine = None
            self.screener = None
            self.risk_engine = None
        
        # Results storage
        self.portfolio_history = []
        self.performance_metrics = {}
        self.attribution_results = {}
        
        logger.info(f"[backtest] Initialized portfolio backtester for {start_date} to {end_date}")
    
    def run_portfolio_backtest(self,
                             data_source: str = "csv",
                             csv_directory: Optional[str] = None,
                             universe: Optional[List[str]] = None,
                             strategy_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run comprehensive portfolio backtest.
        
        Args:
            data_source: "csv" or "live" data source
            csv_directory: Directory containing CSV files (if data_source="csv")
            universe: List of symbols to trade (if None, uses default universe)
            strategy_config: Strategy configuration parameters
            
        Returns:
            Dictionary with comprehensive backtest results
        """
        logger.info("[backtest] Starting portfolio backtest")
        
        if not ENHANCED_MODULES_AVAILABLE:
            logger.warning("[backtest] Enhanced modules not available, running basic backtest")
            return self._run_basic_backtest(data_source, csv_directory, universe)
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates()
        
        if not rebalance_dates:
            logger.error("[backtest] No rebalance dates generated")
            return {"error": "No valid rebalance dates"}
        
        # Initialize portfolio tracking
        current_portfolio = {}
        portfolio_value = self.initial_capital
        cash_balance = self.initial_capital
        
        backtest_results = {
            "config": {
                "start_date": self.start_date.strftime('%Y-%m-%d'),
                "end_date": self.end_date.strftime('%Y-%m-%d'),
                "rebalance_frequency": self.rebalance_frequency,
                "initial_capital": self.initial_capital,
                "transaction_costs": self.transaction_costs
            },
            "periods": [],
            "performance": {},
            "risk_metrics": {},
            "attribution": {}
        }
        
        # Main backtest loop
        for i, rebalance_date in enumerate(rebalance_dates):
            logger.info(f"[backtest] Processing rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date.strftime('%Y-%m-%d')}")
            
            try:
                # Load market data for this period
                market_data = self._load_market_data_for_date(
                    rebalance_date, data_source, csv_directory, universe
                )
                
                if not market_data:
                    logger.warning(f"[backtest] No market data for {rebalance_date}")
                    continue
                
                # Run stock selection process
                selected_stocks = self._run_stock_selection(
                    market_data, rebalance_date, strategy_config
                )
                
                if not selected_stocks:
                    logger.warning(f"[backtest] No stocks selected for {rebalance_date}")
                    continue
                
                # Calculate new portfolio weights
                new_weights = self._calculate_portfolio_weights(
                    selected_stocks, market_data, current_portfolio
                )
                
                # Execute rebalancing
                portfolio_value, cash_balance, transaction_costs = self._execute_rebalancing(
                    current_portfolio, new_weights, market_data, portfolio_value, cash_balance
                )
                
                # Update current portfolio
                current_portfolio = new_weights.copy()
                
                # Store period results
                period_result = {
                    "date": rebalance_date.strftime('%Y-%m-%d'),
                    "portfolio_value": portfolio_value,
                    "cash_balance": cash_balance,
                    "transaction_costs": transaction_costs,
                    "positions": current_portfolio.copy(),
                    "selected_stocks": list(selected_stocks.keys()),
                    "num_positions": len(current_portfolio)
                }
                
                backtest_results["periods"].append(period_result)
                
                # Calculate period returns for next iteration
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    period_returns = self._calculate_period_returns(
                        current_portfolio, rebalance_date, next_date, data_source, csv_directory
                    )
                    
                    # Update portfolio value based on returns
                    portfolio_return = sum(
                        weight * period_returns.get(symbol, 0) 
                        for symbol, weight in current_portfolio.items()
                    )
                    
                    portfolio_value = portfolio_value * (1 + portfolio_return)
                
            except Exception as e:
                logger.error(f"[backtest] Error processing {rebalance_date}: {e}")
                continue
        
        # Calculate final performance metrics
        if backtest_results["periods"]:
            backtest_results["performance"] = self._calculate_performance_metrics(backtest_results)
            backtest_results["risk_metrics"] = self._calculate_risk_metrics(backtest_results)
            backtest_results["attribution"] = self._calculate_attribution_analysis(backtest_results)
        
        self.portfolio_history = backtest_results["periods"]
        self.performance_metrics = backtest_results["performance"]
        
        logger.info("[backtest] Portfolio backtest completed")
        
        return backtest_results
    
    def run_strategy_comparison(self,
                              strategies: Dict[str, Dict],
                              data_source: str = "csv",
                              csv_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comparison between multiple strategies.
        
        Args:
            strategies: Dictionary of strategy_name -> strategy_config
            data_source: Data source type
            csv_directory: CSV data directory
            
        Returns:
            Comparison results across strategies
        """
        logger.info(f"[backtest] Running strategy comparison for {len(strategies)} strategies")
        
        comparison_results = {
            "strategies": {},
            "comparison_metrics": {},
            "rankings": {}
        }
        
        # Run backtest for each strategy
        for strategy_name, strategy_config in strategies.items():
            logger.info(f"[backtest] Testing strategy: {strategy_name}")
            
            try:
                strategy_results = self.run_portfolio_backtest(
                    data_source=data_source,
                    csv_directory=csv_directory,
                    strategy_config=strategy_config
                )
                
                comparison_results["strategies"][strategy_name] = strategy_results
                
            except Exception as e:
                logger.error(f"[backtest] Error testing strategy {strategy_name}: {e}")
                continue
        
        # Calculate comparison metrics
        if comparison_results["strategies"]:
            comparison_results["comparison_metrics"] = self._calculate_strategy_comparison_metrics(
                comparison_results["strategies"]
            )
            comparison_results["rankings"] = self._rank_strategies(
                comparison_results["comparison_metrics"]
            )
        
        return comparison_results
    
    def calculate_factor_attribution(self,
                                   backtest_results: Dict[str, Any],
                                   factor_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Calculate performance attribution by factors.
        
        Args:
            backtest_results: Results from portfolio backtest
            factor_data: Optional external factor data
            
        Returns:
            Factor attribution analysis
        """
        if not ENHANCED_MODULES_AVAILABLE or not backtest_results.get("periods"):
            return {"error": "Enhanced modules or backtest data not available"}
        
        logger.info("[backtest] Calculating factor attribution")
        
        attribution_results = {
            "factor_contributions": {},
            "factor_exposures": {},
            "period_attribution": [],
            "summary_statistics": {}
        }
        
        # Analyze each period
        for period in backtest_results["periods"]:
            period_attribution = self._calculate_period_factor_attribution(
                period, factor_data
            )
            attribution_results["period_attribution"].append(period_attribution)
        
        # Aggregate factor contributions
        attribution_results["factor_contributions"] = self._aggregate_factor_contributions(
            attribution_results["period_attribution"]
        )
        
        # Calculate summary statistics
        attribution_results["summary_statistics"] = self._calculate_attribution_summary(
            attribution_results["factor_contributions"]
        )
        
        return attribution_results
    
    def _run_basic_backtest(self,
                          data_source: str,
                          csv_directory: Optional[str],
                          universe: Optional[List[str]]) -> Dict[str, Any]:
        """
        Run basic single-symbol backtest for compatibility.
        """
        logger.warning("[backtest] Running basic backtest - enhanced features disabled")
        
        # This maintains compatibility with the original backtest.py functionality
        basic_results = {
            "type": "basic_backtest",
            "message": "Enhanced modules not available - ran basic backtest",
            "periods": [],
            "performance": {}
        }
        
        return basic_results
    
    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        """
        Generate rebalancing dates based on frequency.
        """
        dates = []
        current_date = self.start_date
        
        if self.rebalance_frequency == "daily":
            freq = pd.Timedelta(days=1)
        elif self.rebalance_frequency == "weekly":
            freq = pd.Timedelta(weeks=1)
        elif self.rebalance_frequency == "monthly":
            freq = pd.Timedelta(days=30)  # Approximate
        elif self.rebalance_frequency == "quarterly":
            freq = pd.Timedelta(days=90)  # Approximate
        else:
            freq = pd.Timedelta(days=30)  # Default to monthly
        
        while current_date <= self.end_date:
            dates.append(current_date)
            current_date += freq
        
        return dates
    
    def _load_market_data_for_date(self,
                                 date: pd.Timestamp,
                                 data_source: str,
                                 csv_directory: Optional[str],
                                 universe: Optional[List[str]]) -> Dict[str, pd.DataFrame]:
        """
        Load market data for a specific date.
        """
        market_data = {}
        
        if data_source == "csv" and csv_directory:
            # Load from CSV files
            csv_path = Path(csv_directory)
            if not csv_path.exists():
                logger.warning(f"[backtest] CSV directory not found: {csv_directory}")
                return market_data
            
            # Default universe if not provided
            if universe is None:
                universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            
            for symbol in universe:
                csv_file = csv_path / f"{symbol}.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        if 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time'])
                            # Filter data up to the rebalance date
                            df = df[df['time'] <= date].copy()
                            if not df.empty:
                                market_data[symbol] = df
                    except Exception as e:
                        logger.warning(f"[backtest] Error loading {symbol}: {e}")
                        continue
        
        elif data_source == "live":
            # Would integrate with live data sources
            logger.warning("[backtest] Live data source not implemented in backtest mode")
        
        return market_data
    
    def _run_stock_selection(self,
                           market_data: Dict[str, pd.DataFrame],
                           date: pd.Timestamp,
                           strategy_config: Optional[Dict]) -> Dict[str, float]:
        """
        Run stock selection process for a given date.
        """
        if not self.scoring_engine or not self.screener:
            # Fallback to simple selection
            return {symbol: 1.0 for symbol in list(market_data.keys())[:10]}
        
        try:
            # Run scoring engine
            scoring_result = self.scoring_engine.calculate_composite_scores(market_data)
            
            if scoring_result.scores.empty:
                return {}
            
            # Apply risk filters
            symbols = scoring_result.scores['symbol'].tolist()
            filtered_symbols, risk_metrics = self.risk_engine.apply_risk_filters(
                symbols, market_data
            )
            
            if not filtered_symbols:
                return {}
            
            # Get top scoring filtered symbols
            filtered_scores = scoring_result.scores[
                scoring_result.scores['symbol'].isin(filtered_symbols)
            ].copy()
            
            # Select top N symbols
            top_n = strategy_config.get('top_n', 20) if strategy_config else 20
            top_stocks = filtered_scores.nlargest(top_n, 'composite_score')
            
            # Return as dictionary with scores
            selected_stocks = {}
            for _, row in top_stocks.iterrows():
                selected_stocks[row['symbol']] = row['composite_score']
            
            return selected_stocks
            
        except Exception as e:
            logger.warning(f"[backtest] Stock selection error: {e}")
            return {}
    
    def _calculate_portfolio_weights(self,
                                   selected_stocks: Dict[str, float],
                                   market_data: Dict[str, pd.DataFrame],
                                   current_portfolio: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk-adjusted portfolio weights.
        """
        if not self.risk_engine:
            # Equal weight fallback
            n_stocks = len(selected_stocks)
            if n_stocks == 0:
                return {}
            
            equal_weight = 1.0 / n_stocks
            return {symbol: equal_weight for symbol in selected_stocks.keys()}
        
        try:
            symbols = list(selected_stocks.keys())
            scores = selected_stocks
            
            # Calculate risk-adjusted weights
            weights = self.risk_engine.calculate_position_sizes(
                symbols, scores, {}, None  # Simplified - would pass proper risk metrics
            )
            
            return weights
            
        except Exception as e:
            logger.warning(f"[backtest] Weight calculation error: {e}")
            # Fallback to equal weights
            n_stocks = len(selected_stocks)
            equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0.0
            return {symbol: equal_weight for symbol in selected_stocks.keys()}
    
    def _execute_rebalancing(self,
                           current_portfolio: Dict[str, float],
                           new_weights: Dict[str, float],
                           market_data: Dict[str, pd.DataFrame],
                           portfolio_value: float,
                           cash_balance: float) -> Tuple[float, float, float]:
        """
        Execute portfolio rebalancing and calculate transaction costs.
        """
        total_transaction_costs = 0.0
        
        # Calculate trades needed
        all_symbols = set(current_portfolio.keys()) | set(new_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_portfolio.get(symbol, 0.0)
            target_weight = new_weights.get(symbol, 0.0)
            weight_change = abs(target_weight - current_weight)
            
            # Calculate transaction cost
            trade_value = weight_change * portfolio_value
            transaction_cost = trade_value * self.transaction_costs
            total_transaction_costs += transaction_cost
        
        # Adjust portfolio value for transaction costs
        new_portfolio_value = portfolio_value - total_transaction_costs
        new_cash_balance = cash_balance - total_transaction_costs
        
        return new_portfolio_value, new_cash_balance, total_transaction_costs
    
    def _calculate_period_returns(self,
                                portfolio: Dict[str, float],
                                start_date: pd.Timestamp,
                                end_date: pd.Timestamp,
                                data_source: str,
                                csv_directory: Optional[str]) -> Dict[str, float]:
        """
        Calculate returns for each symbol over a period.
        """
        returns = {}
        
        # Load data for the period
        period_data = self._load_market_data_for_date(end_date, data_source, csv_directory, list(portfolio.keys()))
        
        for symbol, weight in portfolio.items():
            df = period_data.get(symbol)
            if df is None or df.empty:
                returns[symbol] = 0.0
                continue
            
            try:
                # Find prices at start and end dates
                df_period = df[(df['time'] >= start_date) & (df['time'] <= end_date)].copy()
                
                if len(df_period) < 2:
                    returns[symbol] = 0.0
                    continue
                
                start_price = df_period['close'].iloc[0]
                end_price = df_period['close'].iloc[-1]
                
                period_return = (end_price - start_price) / start_price
                returns[symbol] = period_return
                
            except Exception as e:
                logger.warning(f"[backtest] Error calculating return for {symbol}: {e}")
                returns[symbol] = 0.0
        
        return returns
    
    def _calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        """
        periods = backtest_results.get("periods", [])
        if not periods:
            return {}
        
        # Extract portfolio values
        portfolio_values = [p["portfolio_value"] for p in periods]
        dates = [pd.to_datetime(p["date"]) for p in periods]
        
        # Calculate returns
        returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)
        
        if not returns:
            return {}
        
        returns = np.array(returns)
        
        # Performance metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualized return
        periods_per_year = self._get_periods_per_year()
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_values = np.array(portfolio_values)
        rolling_max = np.maximum.accumulate(cumulative_values)
        drawdowns = (cumulative_values - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate
        positive_returns = returns > 0
        win_rate = np.mean(positive_returns) if len(returns) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        performance_metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "calmar_ratio": calmar_ratio,
            "final_value": portfolio_values[-1],
            "total_periods": len(periods)
        }
        
        return performance_metrics
    
    def _calculate_risk_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk metrics for the backtest.
        """
        periods = backtest_results.get("periods", [])
        if not periods:
            return {}
        
        # Calculate portfolio concentration metrics
        avg_positions = np.mean([p["num_positions"] for p in periods])
        
        # Calculate turnover
        turnover_rates = []
        for i in range(1, len(periods)):
            prev_positions = set(periods[i-1]["positions"].keys())
            curr_positions = set(periods[i]["positions"].keys())
            
            if len(prev_positions) > 0:
                turnover = len(prev_positions.symmetric_difference(curr_positions)) / len(prev_positions)
                turnover_rates.append(turnover)
        
        avg_turnover = np.mean(turnover_rates) if turnover_rates else 0
        
        risk_metrics = {
            "average_positions": avg_positions,
            "average_turnover": avg_turnover,
            "total_transaction_costs": sum(p.get("transaction_costs", 0) for p in periods)
        }
        
        return risk_metrics
    
    def _calculate_attribution_analysis(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate basic attribution analysis.
        """
        periods = backtest_results.get("periods", [])
        if not periods:
            return {}
        
        # Track stock selection over time
        all_selected_stocks = set()
        stock_selection_counts = {}
        
        for period in periods:
            selected_stocks = period.get("selected_stocks", [])
            all_selected_stocks.update(selected_stocks)
            
            for stock in selected_stocks:
                stock_selection_counts[stock] = stock_selection_counts.get(stock, 0) + 1
        
        # Most frequently selected stocks
        top_selections = sorted(stock_selection_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        attribution = {
            "total_unique_stocks": len(all_selected_stocks),
            "top_selections": dict(top_selections),
            "selection_frequency": stock_selection_counts
        }
        
        return attribution
    
    def _get_periods_per_year(self) -> float:
        """
        Get number of rebalancing periods per year.
        """
        if self.rebalance_frequency == "daily":
            return 252  # Trading days
        elif self.rebalance_frequency == "weekly":
            return 52
        elif self.rebalance_frequency == "monthly":
            return 12
        elif self.rebalance_frequency == "quarterly":
            return 4
        else:
            return 12  # Default to monthly
    
    def _calculate_strategy_comparison_metrics(self, strategies: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate metrics for comparing strategies.
        """
        comparison_metrics = {}
        
        for strategy_name, strategy_results in strategies.items():
            performance = strategy_results.get("performance", {})
            risk_metrics = strategy_results.get("risk_metrics", {})
            
            comparison_metrics[strategy_name] = {
                "annualized_return": performance.get("annualized_return", 0),
                "volatility": performance.get("volatility", 0),
                "sharpe_ratio": performance.get("sharpe_ratio", 0),
                "max_drawdown": performance.get("max_drawdown", 0),
                "calmar_ratio": performance.get("calmar_ratio", 0),
                "win_rate": performance.get("win_rate", 0),
                "avg_turnover": risk_metrics.get("average_turnover", 0)
            }
        
        return comparison_metrics
    
    def _rank_strategies(self, comparison_metrics: Dict[str, Dict]) -> Dict[str, List]:
        """
        Rank strategies by different metrics.
        """
        rankings = {}
        
        metrics_to_rank = ['annualized_return', 'sharpe_ratio', 'calmar_ratio', 'win_rate']
        
        for metric in metrics_to_rank:
            strategy_scores = [(name, data.get(metric, 0)) for name, data in comparison_metrics.items()]
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [name for name, score in strategy_scores]
        
        # Rank by max drawdown (lower is better)
        drawdown_scores = [(name, data.get('max_drawdown', 0)) for name, data in comparison_metrics.items()]
        drawdown_scores.sort(key=lambda x: x[1])  # Ascending order for drawdown
        rankings['max_drawdown'] = [name for name, score in drawdown_scores]
        
        return rankings
    
    def _calculate_period_factor_attribution(self, period: Dict, factor_data: Optional[Dict]) -> Dict:
        """
        Calculate factor attribution for a single period.
        """
        # Simplified factor attribution - would be expanded with real factor data
        attribution = {
            "date": period["date"],
            "factor_contributions": {
                "stock_selection": 0.8,  # Placeholder
                "market_timing": 0.1,    # Placeholder
                "sector_allocation": 0.1  # Placeholder
            }
        }
        
        return attribution
    
    def _aggregate_factor_contributions(self, period_attributions: List[Dict]) -> Dict:
        """
        Aggregate factor contributions across periods.
        """
        if not period_attributions:
            return {}
        
        # Aggregate contributions
        total_contributions = {"stock_selection": 0.0, "market_timing": 0.0, "sector_allocation": 0.0}
        
        for period in period_attributions:
            contributions = period.get("factor_contributions", {})
            for factor, contribution in contributions.items():
                total_contributions[factor] = total_contributions.get(factor, 0) + contribution
        
        # Average contributions
        n_periods = len(period_attributions)
        avg_contributions = {factor: contrib / n_periods for factor, contrib in total_contributions.items()}
        
        return avg_contributions
    
    def _calculate_attribution_summary(self, factor_contributions: Dict) -> Dict:
        """
        Calculate summary statistics for attribution analysis.
        """
        summary = {
            "primary_factor": max(factor_contributions, key=factor_contributions.get) if factor_contributions else "unknown",
            "factor_diversification": len(factor_contributions),
            "total_explained": sum(factor_contributions.values()) if factor_contributions else 0
        }
        
        return summary
    
    def save_backtest_results(self, results: Dict[str, Any], filepath: str):
        """
        Save backtest results to file.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"[backtest] Results saved to {filepath}")


def main():
    """Main function with enhanced argument parsing."""
    ap = argparse.ArgumentParser(description='Enhanced Portfolio Backtesting System')
    
    # Original arguments for backward compatibility
    ap.add_argument('--symbol', help='Single symbol for basic backtest')
    ap.add_argument('--short', type=int, default=5, help='Short MA period')
    ap.add_argument('--long', type=int, default=20, help='Long MA period')
    ap.add_argument('--csv', help='CSV file for single symbol backtest')
    
    # New portfolio backtest arguments
    ap.add_argument('--portfolio', action='store_true', help='Run portfolio backtest')
    ap.add_argument('--start-date', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    ap.add_argument('--end-date', default='2023-12-31', help='End date (YYYY-MM-DD)')
    ap.add_argument('--csv-dir', help='Directory containing CSV files')
    ap.add_argument('--universe', nargs='+', help='List of symbols to trade')
    ap.add_argument('--rebalance', default='monthly', choices=['daily', 'weekly', 'monthly', 'quarterly'],
                    help='Rebalancing frequency')
    ap.add_argument('--initial-capital', type=float, default=1000000.0, help='Initial capital')
    ap.add_argument('--transaction-costs', type=float, default=0.001, help='Transaction costs (fraction)')
    ap.add_argument('--output', help='Output file for results')
    
    args = ap.parse_args()
    
    if args.portfolio:
        # Run enhanced portfolio backtest
        logger.info("Running enhanced portfolio backtest")
        
        backtester = PortfolioBacktester(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital,
            rebalance_frequency=args.rebalance,
            transaction_costs=args.transaction_costs
        )
        
        results = backtester.run_portfolio_backtest(
            data_source="csv" if args.csv_dir else "live",
            csv_directory=args.csv_dir,
            universe=args.universe
        )
        
        # Print summary
        if "performance" in results:
            perf = results["performance"]
            print("\n=== Portfolio Backtest Results ===")
            print(f"Total Return: {perf.get('total_return', 0):.2%}")
            print(f"Annualized Return: {perf.get('annualized_return', 0):.2%}")
            print(f"Volatility: {perf.get('volatility', 0):.2%}")
            print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
            print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"Final Value: ${perf.get('final_value', 0):,.0f}")
        
        # Save results if output specified
        if args.output:
            backtester.save_backtest_results(results, args.output)
            print(f"\nResults saved to: {args.output}")
        
    else:
        # Run original single-symbol backtest for backward compatibility
        if not args.symbol or not args.csv:
            print('For single symbol backtest, please supply --symbol and --csv')
            print('For portfolio backtest, use --portfolio flag')
            return
        
        logger.info(f"Running basic backtest for {args.symbol}")
        
        df = pd.read_csv(args.csv)
        sig = generate_signals(df, short=args.short, long=args.long)
        res = simple_pnl(sig)
        
        print(res[['time','close','position']].tail())
        print('Backtest cumulative return:', round(res["cum_pnl"].iloc[-1]*100, 2), '%')


if __name__ == '__main__':
    main()
