"""
Performance Attribution Analysis

Provides comprehensive performance attribution analysis for the trading system,
breaking down returns by factors, sectors, and individual stocks.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json


@dataclass
class AttributionResult:
    """Performance attribution analysis result."""
    total_return: float
    benchmark_return: float
    total_active_return: float
    factor_attribution: Dict[str, float]
    sector_attribution: Dict[str, float]
    stock_selection_return: float
    asset_allocation_return: float
    interaction_return: float
    unexplained_return: float
    timestamp: datetime


@dataclass
class FactorContribution:
    """Individual factor contribution to performance."""
    factor_name: str
    exposure: float
    factor_return: float
    contribution: float
    attribution: float


class PerformanceAttributor:
    """Analyze and attribute portfolio performance to various factors."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize performance attributor."""
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'factors': ['momentum', 'value', 'quality', 'size', 'volatility', 'profitability'],
            'benchmark': 'SPY',
            'attribution_period': 30,  # days
            'min_factor_exposure': 0.01
        }

    def calculate_attribution(self, returns_data: Dict[str, Any]) -> AttributionResult:
        """
        Calculate comprehensive performance attribution.

        Args:
            returns_data: Dictionary containing:
                - portfolio_return: Portfolio total return
                - benchmark_return: Benchmark return
                - factor_exposures: Dict of factor exposures
                - factor_returns: Dict of factor returns
                - sector_weights: Optional sector weights
                - sector_returns: Optional sector returns

        Returns:
            AttributionResult with detailed breakdown
        """
        portfolio_return = returns_data.get('portfolio_return', 0.0)
        benchmark_return = returns_data.get('benchmark_return', 0.0)
        factor_exposures = returns_data.get('factor_exposures', {})
        factor_returns = returns_data.get('factor_returns', {})

        # Calculate active return
        total_active_return = portfolio_return - benchmark_return

        # Factor attribution
        factor_attribution = self._calculate_factor_attribution(
            factor_exposures, factor_returns
        )

        # Sector attribution (if data available)
        sector_attribution = self._calculate_sector_attribution(
            returns_data.get('sector_weights', {}),
            returns_data.get('sector_returns', {}),
            returns_data.get('benchmark_sector_weights', {}),
            returns_data.get('benchmark_sector_returns', {})
        )

        # Brinson attribution components
        stock_selection, asset_allocation, interaction = self._calculate_brinson_attribution(
            returns_data.get('portfolio_weights', {}),
            returns_data.get('portfolio_returns', {}),
            returns_data.get('benchmark_weights', {}),
            returns_data.get('benchmark_returns', {})
        )

        # Calculate unexplained return
        explained_return = sum(factor_attribution.values()) + sum(sector_attribution.values())
        unexplained_return = total_active_return - explained_return

        return AttributionResult(
            total_return=portfolio_return,
            benchmark_return=benchmark_return,
            total_active_return=total_active_return,
            factor_attribution=factor_attribution,
            sector_attribution=sector_attribution,
            stock_selection_return=stock_selection,
            asset_allocation_return=asset_allocation,
            interaction_return=interaction,
            unexplained_return=unexplained_return,
            timestamp=datetime.now()
        )

    def _calculate_factor_attribution(self, exposures: Dict[str, float],
                                    returns: Dict[str, float]) -> Dict[str, float]:
        """Calculate factor-based attribution."""
        attribution = {}

        for factor in self.config['factors']:
            exposure = exposures.get(factor, 0.0)
            factor_return = returns.get(factor, 0.0)

            # Attribution = exposure * factor_return
            attribution[factor] = exposure * factor_return

        return attribution

    def _calculate_sector_attribution(self, portfolio_weights: Dict[str, float],
                                    sector_returns: Dict[str, float],
                                    benchmark_weights: Dict[str, float],
                                    benchmark_returns: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector-based attribution using Brinson model."""
        attribution = {}

        all_sectors = set(portfolio_weights.keys()) | set(benchmark_weights.keys())

        for sector in all_sectors:
            wp = portfolio_weights.get(sector, 0.0)  # Portfolio weight
            wb = benchmark_weights.get(sector, 0.0)  # Benchmark weight
            rp = sector_returns.get(sector, 0.0)     # Portfolio sector return
            rb = benchmark_returns.get(sector, 0.0)  # Benchmark sector return

            # Allocation effect: (wp - wb) * rb
            allocation_effect = (wp - wb) * rb

            # Selection effect: wb * (rp - rb)
            selection_effect = wb * (rp - rb)

            # Interaction effect: (wp - wb) * (rp - rb)
            interaction_effect = (wp - wb) * (rp - rb)

            # Total sector attribution
            attribution[sector] = allocation_effect + selection_effect + interaction_effect

        return attribution

    def _calculate_brinson_attribution(self, portfolio_weights: Dict[str, float],
                                     portfolio_returns: Dict[str, float],
                                     benchmark_weights: Dict[str, float],
                                     benchmark_returns: Dict[str, float]) -> Tuple[float, float, float]:
        """Calculate Brinson attribution components."""
        if not all([portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns]):
            return 0.0, 0.0, 0.0

        stock_selection = 0.0
        asset_allocation = 0.0
        interaction = 0.0

        all_assets = set(portfolio_weights.keys()) | set(benchmark_weights.keys())

        for asset in all_assets:
            wp = portfolio_weights.get(asset, 0.0)
            wb = benchmark_weights.get(asset, 0.0)
            rp = portfolio_returns.get(asset, 0.0)
            rb = benchmark_returns.get(asset, 0.0)

            # Stock selection: wb * (rp - rb)
            stock_selection += wb * (rp - rb)

            # Asset allocation: (wp - wb) * rb
            asset_allocation += (wp - wb) * rb

            # Interaction: (wp - wb) * (rp - rb)
            interaction += (wp - wb) * (rp - rb)

        return stock_selection, asset_allocation, interaction

    def calculate_rolling_attribution(self, historical_data: pd.DataFrame,
                                    period_days: int = 30) -> pd.DataFrame:
        """Calculate rolling performance attribution over time."""
        results = []

        for i in range(period_days, len(historical_data)):
            period_data = historical_data.iloc[i-period_days:i]

            # Calculate period returns (simplified)
            portfolio_return = period_data['portfolio_value'].iloc[-1] / period_data['portfolio_value'].iloc[0] - 1
            benchmark_return = period_data['benchmark_value'].iloc[-1] / period_data['benchmark_value'].iloc[0] - 1

            # Prepare attribution data
            returns_data = {
                'portfolio_return': portfolio_return,
                'benchmark_return': benchmark_return,
                'factor_exposures': self._extract_factor_exposures(period_data),
                'factor_returns': self._calculate_factor_returns(period_data)
            }

            # Calculate attribution
            attribution = self.calculate_attribution(returns_data)

            result_row = {
                'date': period_data.index[-1],
                'total_return': attribution.total_return,
                'benchmark_return': attribution.benchmark_return,
                'active_return': attribution.total_active_return,
                'stock_selection': attribution.stock_selection_return,
                'asset_allocation': attribution.asset_allocation_return,
                'unexplained': attribution.unexplained_return
            }

            # Add factor attributions
            for factor, contrib in attribution.factor_attribution.items():
                result_row[f'factor_{factor}'] = contrib

            results.append(result_row)

        return pd.DataFrame(results)

    def _extract_factor_exposures(self, period_data: pd.DataFrame) -> Dict[str, float]:
        """Extract factor exposures from period data (simplified)."""
        # This is a simplified implementation
        # In practice, this would extract actual factor exposures from holdings data
        return {
            'momentum': np.random.normal(0, 0.3),
            'value': np.random.normal(0, 0.2),
            'quality': np.random.normal(0, 0.25),
            'size': np.random.normal(0, 0.4),
            'volatility': np.random.normal(0, 0.15)
        }

    def _calculate_factor_returns(self, period_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate factor returns for the period (simplified)."""
        # This is a simplified implementation
        # In practice, this would use actual factor model returns
        return {
            'momentum': np.random.normal(0.02, 0.08),
            'value': np.random.normal(-0.01, 0.06),
            'quality': np.random.normal(0.015, 0.05),
            'size': np.random.normal(0.005, 0.1),
            'volatility': np.random.normal(-0.01, 0.07)
        }

    def generate_attribution_report(self, attribution: AttributionResult) -> Dict[str, Any]:
        """Generate comprehensive attribution report."""
        return {
            'timestamp': attribution.timestamp.isoformat(),
            'summary': {
                'total_return': f"{attribution.total_return:.2%}",
                'benchmark_return': f"{attribution.benchmark_return:.2%}",
                'active_return': f"{attribution.total_active_return:.2%}",
                'information_ratio': self._calculate_information_ratio(attribution)
            },
            'brinson_attribution': {
                'stock_selection': f"{attribution.stock_selection_return:.2%}",
                'asset_allocation': f"{attribution.asset_allocation_return:.2%}",
                'interaction': f"{attribution.interaction_return:.2%}",
                'total_explained': f"{(attribution.stock_selection_return + attribution.asset_allocation_return + attribution.interaction_return):.2%}"
            },
            'factor_attribution': {
                factor: f"{contrib:.2%}"
                for factor, contrib in attribution.factor_attribution.items()
            },
            'sector_attribution': {
                sector: f"{contrib:.2%}"
                for sector, contrib in attribution.sector_attribution.items()
            },
            'analysis': {
                'unexplained_return': f"{attribution.unexplained_return:.2%}",
                'explanation_ratio': f"{(1 - abs(attribution.unexplained_return) / max(abs(attribution.total_active_return), 0.0001)):.1%}",
                'primary_drivers': self._identify_primary_drivers(attribution)
            }
        }

    def _calculate_information_ratio(self, attribution: AttributionResult) -> str:
        """Calculate information ratio (simplified)."""
        # This is simplified - in practice would use time series of active returns
        active_return = attribution.total_active_return
        tracking_error = 0.05  # Assumed tracking error

        if tracking_error > 0:
            ir = active_return / tracking_error
            return f"{ir:.2f}"
        return "N/A"

    def _identify_primary_drivers(self, attribution: AttributionResult) -> List[str]:
        """Identify primary drivers of performance."""
        drivers = []

        # Combine all attribution sources
        all_attributions = {
            'stock_selection': attribution.stock_selection_return,
            'asset_allocation': attribution.asset_allocation_return,
            **attribution.factor_attribution,
            **attribution.sector_attribution
        }

        # Sort by absolute contribution
        sorted_attributions = sorted(
            all_attributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Take top 3 drivers
        for name, contribution in sorted_attributions[:3]:
            if abs(contribution) > 0.001:  # 10bps threshold
                direction = "positive" if contribution > 0 else "negative"
                drivers.append(f"{name.replace('_', ' ').title()}: {direction} {abs(contribution):.2%}")

        return drivers


# Example usage and testing
if __name__ == "__main__":
    # Test the performance attributor
    attributor = PerformanceAttributor()

    # Sample performance data
    returns_data = {
        'portfolio_return': 0.15,
        'benchmark_return': 0.12,
        'factor_exposures': {
            'momentum': 0.3,
            'value': -0.1,
            'quality': 0.2,
            'size': 0.15,
            'volatility': -0.05
        },
        'factor_returns': {
            'momentum': 0.08,
            'value': -0.02,
            'quality': 0.05,
            'size': 0.03,
            'volatility': -0.01
        },
        'sector_weights': {
            'Technology': 0.4,
            'Healthcare': 0.3,
            'Financial': 0.3
        },
        'sector_returns': {
            'Technology': 0.18,
            'Healthcare': 0.12,
            'Financial': 0.15
        }
    }

    # Calculate attribution
    attribution = attributor.calculate_attribution(returns_data)

    # Generate report
    report = attributor.generate_attribution_report(attribution)

    print("Performance Attribution Analysis:")
    print(f"Total Return: {attribution.total_return:.2%}")
    print(f"Benchmark Return: {attribution.benchmark_return:.2%}")
    print(f"Active Return: {attribution.total_active_return:.2%}")

    print("\nFactor Attribution:")
    for factor, contrib in attribution.factor_attribution.items():
        print(f"  {factor.title()}: {contrib:.2%}")

    print("\nBrinson Attribution:")
    print(f"  Stock Selection: {attribution.stock_selection_return:.2%}")
    print(f"  Asset Allocation: {attribution.asset_allocation_return:.2%}")
    print(f"  Interaction: {attribution.interaction_return:.2%}")

    print(f"\nUnexplained Return: {attribution.unexplained_return:.2%}")

    print("\nPrimary Drivers:")
    for driver in report['analysis']['primary_drivers']:
        print(f"  - {driver}")