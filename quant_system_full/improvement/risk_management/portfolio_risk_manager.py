#!/usr/bin/env python3
"""
Advanced Portfolio Risk Management System

This module implements sophisticated portfolio risk management including:
- Ledoit-Wolf covariance matrix shrinkage estimation
- Risk budget allocation and monitoring
- VaR and CVaR calculations
- Correlation and concentration risk analysis
- Real-time risk monitoring and alerts

Based on modern portfolio theory and risk management best practices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy import linalg
from sklearn.covariance import LedoitWolf
import warnings

logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Types of risk metrics"""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    DRAWDOWN = "drawdown"


@dataclass
class RiskBudget:
    """Risk budget allocation"""
    total_budget: float
    sector_limits: Dict[str, float]
    position_limits: Dict[str, float]
    var_limit: float
    correlation_limit: float
    concentration_limit: float


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    portfolio_var_95: float
    portfolio_var_99: float
    portfolio_cvar_95: float
    portfolio_cvar_99: float
    portfolio_volatility: float
    portfolio_beta: float
    max_correlation: float
    concentration_hhi: float
    max_position_weight: float
    risk_budget_utilization: float
    timestamp: datetime


@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    weight: float
    contribution_to_var: float
    marginal_var: float
    component_var: float
    beta: float
    volatility: float
    correlation_with_market: float


class PortfolioRiskManager:
    """
    Advanced portfolio risk management system

    Implements modern risk management techniques including:
    - Robust covariance estimation
    - Multi-horizon risk calculation
    - Real-time risk monitoring
    - Risk budget enforcement
    """

    def __init__(self,
                 confidence_levels: List[float] = [0.95, 0.99],
                 risk_horizon_days: int = 1,
                 lookback_days: int = 252,
                 shrinkage_method: str = 'ledoit_wolf',
                 min_observations: int = 60):
        """
        Initialize portfolio risk manager

        Args:
            confidence_levels: VaR confidence levels
            risk_horizon_days: Risk measurement horizon
            lookback_days: Historical data lookback period
            shrinkage_method: Covariance shrinkage method
            min_observations: Minimum observations required
        """
        self.confidence_levels = confidence_levels
        self.risk_horizon_days = risk_horizon_days
        self.lookback_days = lookback_days
        self.shrinkage_method = shrinkage_method
        self.min_observations = min_observations

        # Risk model parameters
        self.market_index = 'SPY'  # Market benchmark
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

        # Data storage
        self._price_data = {}
        self._returns_data = {}
        self._covariance_matrix = None
        self._last_update = None

        logger.info("Portfolio Risk Manager initialized")

    def update_price_data(self, prices: pd.DataFrame):
        """
        Update price data for risk calculations

        Args:
            prices: DataFrame with symbols as columns, dates as index
        """
        try:
            # Store price data
            self._price_data = prices.copy()

            # Calculate returns
            self._returns_data = prices.pct_change().dropna()

            # Update covariance matrix
            self._update_covariance_matrix()

            self._last_update = datetime.now()

            logger.info(f"Updated price data for {len(prices.columns)} symbols")

        except Exception as e:
            logger.error(f"Error updating price data: {e}")
            raise

    def _update_covariance_matrix(self):
        """Update covariance matrix using shrinkage estimation"""
        try:
            if self._returns_data.empty:
                logger.warning("No returns data available for covariance estimation")
                return

            returns = self._returns_data.fillna(0)

            if len(returns) < self.min_observations:
                logger.warning(f"Insufficient data: {len(returns)} < {self.min_observations}")
                # Use simple covariance for small samples
                self._covariance_matrix = returns.cov()
                return

            if self.shrinkage_method == 'ledoit_wolf':
                # Ledoit-Wolf shrinkage estimator
                lw = LedoitWolf()
                cov_matrix, _ = lw.fit(returns).covariance_, lw.shrinkage_

                # Convert to DataFrame
                self._covariance_matrix = pd.DataFrame(
                    cov_matrix,
                    index=returns.columns,
                    columns=returns.columns
                )

                logger.debug(f"Covariance matrix updated using Ledoit-Wolf shrinkage")

            else:
                # Sample covariance
                self._covariance_matrix = returns.cov()

        except Exception as e:
            logger.error(f"Error updating covariance matrix: {e}")
            # Fallback to identity matrix
            n = len(self._returns_data.columns)
            self._covariance_matrix = pd.DataFrame(
                np.eye(n) * 0.01,  # 1% volatility assumption
                index=self._returns_data.columns,
                columns=self._returns_data.columns
            )

    def calculate_portfolio_var(self, weights: Union[Dict[str, float], pd.Series],
                              confidence_level: float = 0.95,
                              method: str = 'parametric') -> float:
        """
        Calculate portfolio Value at Risk

        Args:
            weights: Portfolio weights (symbol -> weight)
            confidence_level: Confidence level for VaR
            method: VaR calculation method ('parametric', 'historical', 'montecarlo')

        Returns:
            Portfolio VaR as a positive number
        """
        try:
            if isinstance(weights, dict):
                weights = pd.Series(weights)

            # Align weights with available data
            available_symbols = set(weights.index) & set(self._covariance_matrix.index)
            if not available_symbols:
                logger.warning("No symbols overlap between weights and covariance matrix")
                return 0.0

            # Filter weights and covariance matrix
            aligned_weights = weights.reindex(available_symbols).fillna(0)
            aligned_cov = self._covariance_matrix.loc[available_symbols, available_symbols]

            if method == 'parametric':
                return self._calculate_parametric_var(aligned_weights, aligned_cov, confidence_level)
            elif method == 'historical':
                return self._calculate_historical_var(aligned_weights, confidence_level)
            else:
                raise ValueError(f"Unsupported VaR method: {method}")

        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0

    def _calculate_parametric_var(self, weights: pd.Series, cov_matrix: pd.DataFrame,
                                confidence_level: float) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        try:
            # Portfolio variance
            portfolio_variance = np.dot(weights.values, np.dot(cov_matrix.values, weights.values))
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Scale to risk horizon
            portfolio_volatility *= np.sqrt(self.risk_horizon_days)

            # Calculate VaR (normal distribution assumption)
            from scipy.stats import norm
            var_multiplier = norm.ppf(confidence_level)
            portfolio_var = portfolio_volatility * var_multiplier

            return portfolio_var

        except Exception as e:
            logger.error(f"Error in parametric VaR calculation: {e}")
            return 0.0

    def _calculate_historical_var(self, weights: pd.Series, confidence_level: float) -> float:
        """Calculate historical VaR using empirical distribution"""
        try:
            if self._returns_data.empty:
                return 0.0

            # Get aligned returns
            aligned_returns = self._returns_data.reindex(columns=weights.index).fillna(0)

            # Calculate portfolio returns
            portfolio_returns = (aligned_returns * weights).sum(axis=1)

            # Scale to risk horizon
            if self.risk_horizon_days > 1:
                portfolio_returns = portfolio_returns.rolling(self.risk_horizon_days).sum().dropna()

            # Calculate VaR as negative quantile
            var_quantile = 1 - confidence_level
            portfolio_var = -portfolio_returns.quantile(var_quantile)

            return portfolio_var

        except Exception as e:
            logger.error(f"Error in historical VaR calculation: {e}")
            return 0.0

    def calculate_conditional_var(self, weights: Union[Dict[str, float], pd.Series],
                                confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)

        Args:
            weights: Portfolio weights
            confidence_level: Confidence level

        Returns:
            Portfolio CVaR
        """
        try:
            if isinstance(weights, dict):
                weights = pd.Series(weights)

            if self._returns_data.empty:
                return 0.0

            # Get aligned returns
            aligned_returns = self._returns_data.reindex(columns=weights.index).fillna(0)

            # Calculate portfolio returns
            portfolio_returns = (aligned_returns * weights).sum(axis=1)

            # Scale to risk horizon
            if self.risk_horizon_days > 1:
                portfolio_returns = portfolio_returns.rolling(self.risk_horizon_days).sum().dropna()

            # Calculate CVaR
            var_quantile = 1 - confidence_level
            var_threshold = portfolio_returns.quantile(var_quantile)

            # Expected value of returns below VaR threshold
            tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
            cvar = -tail_returns.mean() if len(tail_returns) > 0 else 0.0

            return cvar

        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0

    def calculate_position_risk_contributions(self, weights: Union[Dict[str, float], pd.Series]) -> List[PositionRisk]:
        """
        Calculate risk contributions for each position

        Args:
            weights: Portfolio weights

        Returns:
            List of PositionRisk objects
        """
        try:
            if isinstance(weights, dict):
                weights = pd.Series(weights)

            position_risks = []

            for symbol in weights.index:
                if weights[symbol] == 0:
                    continue

                try:
                    # Calculate position risk metrics
                    pos_risk = self._calculate_position_risk(symbol, weights)
                    position_risks.append(pos_risk)

                except Exception as e:
                    logger.warning(f"Error calculating risk for {symbol}: {e}")
                    continue

            return position_risks

        except Exception as e:
            logger.error(f"Error calculating position risk contributions: {e}")
            return []

    def _calculate_position_risk(self, symbol: str, weights: pd.Series) -> PositionRisk:
        """Calculate risk metrics for individual position"""
        try:
            # Position weight
            weight = weights.get(symbol, 0.0)

            # Position volatility
            if symbol in self._covariance_matrix.index:
                position_vol = np.sqrt(self._covariance_matrix.loc[symbol, symbol])
            else:
                position_vol = 0.0

            # Beta calculation (if market data available)
            beta = self._calculate_beta(symbol)

            # Correlation with market
            market_corr = self._calculate_market_correlation(symbol)

            # Marginal VaR (simplified)
            marginal_var = self._calculate_marginal_var(symbol, weights)

            # Component VaR
            component_var = weight * marginal_var

            # Contribution to portfolio VaR
            portfolio_var = self.calculate_portfolio_var(weights)
            contribution_to_var = component_var / portfolio_var if portfolio_var > 0 else 0.0

            return PositionRisk(
                symbol=symbol,
                weight=weight,
                contribution_to_var=contribution_to_var,
                marginal_var=marginal_var,
                component_var=component_var,
                beta=beta,
                volatility=position_vol,
                correlation_with_market=market_corr
            )

        except Exception as e:
            logger.error(f"Error calculating position risk for {symbol}: {e}")
            return PositionRisk(
                symbol=symbol,
                weight=weights.get(symbol, 0.0),
                contribution_to_var=0.0,
                marginal_var=0.0,
                component_var=0.0,
                beta=1.0,
                volatility=0.0,
                correlation_with_market=0.0
            )

    def _calculate_beta(self, symbol: str) -> float:
        """Calculate beta vs market index"""
        try:
            if (symbol not in self._returns_data.columns or
                self.market_index not in self._returns_data.columns):
                return 1.0

            symbol_returns = self._returns_data[symbol].dropna()
            market_returns = self._returns_data[self.market_index].dropna()

            # Align returns
            aligned_data = pd.concat([symbol_returns, market_returns], axis=1).dropna()

            if len(aligned_data) < 30:  # Need minimum observations
                return 1.0

            symbol_aligned = aligned_data.iloc[:, 0]
            market_aligned = aligned_data.iloc[:, 1]

            # Calculate beta
            covariance = np.cov(symbol_aligned, market_aligned)[0, 1]
            market_variance = np.var(market_aligned)

            beta = covariance / market_variance if market_variance > 0 else 1.0

            return beta

        except Exception as e:
            logger.warning(f"Error calculating beta for {symbol}: {e}")
            return 1.0

    def _calculate_market_correlation(self, symbol: str) -> float:
        """Calculate correlation with market index"""
        try:
            if (symbol not in self._returns_data.columns or
                self.market_index not in self._returns_data.columns):
                return 0.0

            symbol_returns = self._returns_data[symbol].dropna()
            market_returns = self._returns_data[self.market_index].dropna()

            # Align returns
            aligned_data = pd.concat([symbol_returns, market_returns], axis=1).dropna()

            if len(aligned_data) < 30:
                return 0.0

            correlation = aligned_data.corr().iloc[0, 1]

            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.warning(f"Error calculating market correlation for {symbol}: {e}")
            return 0.0

    def _calculate_marginal_var(self, symbol: str, weights: pd.Series) -> float:
        """Calculate marginal VaR for a position"""
        try:
            # Simplified marginal VaR calculation
            if symbol not in self._covariance_matrix.index:
                return 0.0

            # Portfolio covariance with the asset
            aligned_weights = weights.reindex(self._covariance_matrix.index).fillna(0)
            asset_cov_with_portfolio = np.dot(
                self._covariance_matrix.loc[symbol, :].values,
                aligned_weights.values
            )

            # Portfolio variance
            portfolio_variance = np.dot(
                aligned_weights.values,
                np.dot(self._covariance_matrix.values, aligned_weights.values)
            )

            if portfolio_variance <= 0:
                return 0.0

            # Marginal VaR (derivative of portfolio risk)
            marginal_var = asset_cov_with_portfolio / np.sqrt(portfolio_variance)

            # Scale for confidence level (95% assumed)
            from scipy.stats import norm
            marginal_var *= norm.ppf(0.95)

            # Scale for risk horizon
            marginal_var *= np.sqrt(self.risk_horizon_days)

            return marginal_var

        except Exception as e:
            logger.warning(f"Error calculating marginal VaR for {symbol}: {e}")
            return 0.0

    def calculate_concentration_risk(self, weights: Union[Dict[str, float], pd.Series]) -> float:
        """
        Calculate portfolio concentration risk using Herfindahl-Hirschman Index

        Args:
            weights: Portfolio weights

        Returns:
            HHI concentration index (0 to 1, higher = more concentrated)
        """
        try:
            if isinstance(weights, dict):
                weights = pd.Series(weights)

            # Remove zero weights
            active_weights = weights[weights != 0]

            if len(active_weights) == 0:
                return 0.0

            # Calculate HHI
            hhi = np.sum(active_weights ** 2)

            return hhi

        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0

    def generate_risk_report(self, weights: Union[Dict[str, float], pd.Series],
                           risk_budget: Optional[RiskBudget] = None) -> RiskMetrics:
        """
        Generate comprehensive risk report

        Args:
            weights: Portfolio weights
            risk_budget: Optional risk budget for compliance checking

        Returns:
            RiskMetrics object with all risk measures
        """
        try:
            if isinstance(weights, dict):
                weights = pd.Series(weights)

            # Calculate core risk metrics
            var_95 = self.calculate_portfolio_var(weights, 0.95)
            var_99 = self.calculate_portfolio_var(weights, 0.99)
            cvar_95 = self.calculate_conditional_var(weights, 0.95)
            cvar_99 = self.calculate_conditional_var(weights, 0.99)

            # Portfolio volatility
            portfolio_vol = self.calculate_portfolio_var(weights, 0.6827)  # 1-sigma

            # Portfolio beta
            portfolio_beta = self._calculate_portfolio_beta(weights)

            # Concentration metrics
            concentration_hhi = self.calculate_concentration_risk(weights)
            max_position_weight = weights.abs().max()

            # Correlation analysis
            max_correlation = self._calculate_max_pairwise_correlation(weights)

            # Risk budget utilization
            risk_budget_utilization = 0.0
            if risk_budget:
                risk_budget_utilization = var_95 / risk_budget.var_limit if risk_budget.var_limit > 0 else 0.0

            return RiskMetrics(
                portfolio_var_95=var_95,
                portfolio_var_99=var_99,
                portfolio_cvar_95=cvar_95,
                portfolio_cvar_99=cvar_99,
                portfolio_volatility=portfolio_vol,
                portfolio_beta=portfolio_beta,
                max_correlation=max_correlation,
                concentration_hhi=concentration_hhi,
                max_position_weight=max_position_weight,
                risk_budget_utilization=risk_budget_utilization,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            # Return empty metrics
            return RiskMetrics(
                portfolio_var_95=0.0,
                portfolio_var_99=0.0,
                portfolio_cvar_95=0.0,
                portfolio_cvar_99=0.0,
                portfolio_volatility=0.0,
                portfolio_beta=1.0,
                max_correlation=0.0,
                concentration_hhi=0.0,
                max_position_weight=0.0,
                risk_budget_utilization=0.0,
                timestamp=datetime.now()
            )

    def _calculate_portfolio_beta(self, weights: pd.Series) -> float:
        """Calculate portfolio beta (weighted average)"""
        try:
            total_beta = 0.0
            total_weight = 0.0

            for symbol, weight in weights.items():
                if weight != 0:
                    beta = self._calculate_beta(symbol)
                    total_beta += weight * beta
                    total_weight += abs(weight)

            return total_beta / total_weight if total_weight > 0 else 1.0

        except Exception as e:
            logger.warning(f"Error calculating portfolio beta: {e}")
            return 1.0

    def _calculate_max_pairwise_correlation(self, weights: pd.Series) -> float:
        """Calculate maximum pairwise correlation among holdings"""
        try:
            active_symbols = weights[weights != 0].index.tolist()

            if len(active_symbols) < 2:
                return 0.0

            max_corr = 0.0

            for i, symbol1 in enumerate(active_symbols):
                for symbol2 in active_symbols[i+1:]:
                    if (symbol1 in self._covariance_matrix.index and
                        symbol2 in self._covariance_matrix.index):

                        vol1 = np.sqrt(self._covariance_matrix.loc[symbol1, symbol1])
                        vol2 = np.sqrt(self._covariance_matrix.loc[symbol2, symbol2])
                        cov = self._covariance_matrix.loc[symbol1, symbol2]

                        if vol1 > 0 and vol2 > 0:
                            corr = cov / (vol1 * vol2)
                            max_corr = max(max_corr, abs(corr))

            return max_corr

        except Exception as e:
            logger.warning(f"Error calculating max correlation: {e}")
            return 0.0

    def check_risk_limits(self, weights: Union[Dict[str, float], pd.Series],
                         risk_budget: RiskBudget) -> Dict[str, Any]:
        """
        Check portfolio against risk limits

        Args:
            weights: Portfolio weights
            risk_budget: Risk budget constraints

        Returns:
            Dictionary with limit check results
        """
        try:
            if isinstance(weights, dict):
                weights = pd.Series(weights)

            results = {
                'compliant': True,
                'violations': [],
                'warnings': []
            }

            # Calculate current risk metrics
            risk_metrics = self.generate_risk_report(weights, risk_budget)

            # Check VaR limit
            if risk_metrics.portfolio_var_95 > risk_budget.var_limit:
                results['compliant'] = False
                results['violations'].append({
                    'type': 'var_limit',
                    'current': risk_metrics.portfolio_var_95,
                    'limit': risk_budget.var_limit,
                    'excess': risk_metrics.portfolio_var_95 - risk_budget.var_limit
                })

            # Check concentration limit
            if risk_metrics.concentration_hhi > risk_budget.concentration_limit:
                results['compliant'] = False
                results['violations'].append({
                    'type': 'concentration_limit',
                    'current': risk_metrics.concentration_hhi,
                    'limit': risk_budget.concentration_limit,
                    'excess': risk_metrics.concentration_hhi - risk_budget.concentration_limit
                })

            # Check correlation limit
            if risk_metrics.max_correlation > risk_budget.correlation_limit:
                results['warnings'].append({
                    'type': 'correlation_warning',
                    'current': risk_metrics.max_correlation,
                    'limit': risk_budget.correlation_limit
                })

            # Check position limits
            for symbol, weight in weights.items():
                if symbol in risk_budget.position_limits:
                    limit = risk_budget.position_limits[symbol]
                    if abs(weight) > limit:
                        results['compliant'] = False
                        results['violations'].append({
                            'type': 'position_limit',
                            'symbol': symbol,
                            'current': abs(weight),
                            'limit': limit,
                            'excess': abs(weight) - limit
                        })

            return results

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {'compliant': False, 'violations': [], 'warnings': [], 'error': str(e)}


def create_risk_manager(config: Optional[Dict[str, Any]] = None) -> PortfolioRiskManager:
    """
    Factory function to create portfolio risk manager

    Args:
        config: Optional configuration parameters

    Returns:
        Configured PortfolioRiskManager
    """
    if config is None:
        config = {}

    return PortfolioRiskManager(
        confidence_levels=config.get('confidence_levels', [0.95, 0.99]),
        risk_horizon_days=config.get('risk_horizon_days', 1),
        lookback_days=config.get('lookback_days', 252),
        shrinkage_method=config.get('shrinkage_method', 'ledoit_wolf'),
        min_observations=config.get('min_observations', 60)
    )