"""
Risk Exposure Monitor

Monitors portfolio risk exposures including sector concentration, style factors,
and overall risk metrics for the trading system.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class ExposureMetrics:
    """Risk exposure metrics for portfolio monitoring."""
    sector_exposure: Dict[str, float]
    style_exposure: Dict[str, float]
    concentration_ratio: float
    max_position_weight: float
    effective_positions: int
    total_exposure: float
    leverage_ratio: float
    timestamp: datetime


class RiskExposureMonitor:
    """Monitor and analyze portfolio risk exposures."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize risk exposure monitor."""
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'max_sector_exposure': 0.25,
            'max_position_weight': 0.08,
            'concentration_threshold': 0.1,
            'style_factors': ['momentum', 'value', 'quality', 'size', 'volatility'],
            'sectors': ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial', 'Energy', 'Utilities']
        }

    def calculate_exposure_metrics(self, portfolio_data: Dict[str, Dict]) -> ExposureMetrics:
        """
        Calculate comprehensive risk exposure metrics.

        Args:
            portfolio_data: Dictionary with symbol as key and position data as value
                          Each position should have: weight, sector, market_cap, etc.

        Returns:
            ExposureMetrics object with all calculated metrics
        """
        if not portfolio_data:
            return self._empty_metrics()

        # Calculate sector exposures
        sector_exposure = self._calculate_sector_exposure(portfolio_data)

        # Calculate style exposures (simplified)
        style_exposure = self._calculate_style_exposure(portfolio_data)

        # Calculate concentration metrics
        weights = [pos['weight'] for pos in portfolio_data.values()]
        concentration_ratio = self._calculate_concentration_ratio(weights)
        max_position_weight = max(weights) if weights else 0.0

        # Calculate effective number of positions
        effective_positions = self._calculate_effective_positions(weights)

        # Calculate total exposure and leverage
        total_exposure = sum(weights)
        leverage_ratio = total_exposure  # Simplified - assume no shorting

        return ExposureMetrics(
            sector_exposure=sector_exposure,
            style_exposure=style_exposure,
            concentration_ratio=concentration_ratio,
            max_position_weight=max_position_weight,
            effective_positions=effective_positions,
            total_exposure=total_exposure,
            leverage_ratio=leverage_ratio,
            timestamp=datetime.now()
        )

    def _calculate_sector_exposure(self, portfolio_data: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate sector exposure breakdown."""
        sector_weights = {}

        for symbol, position in portfolio_data.items():
            sector = position.get('sector', 'Unknown')
            weight = position.get('weight', 0.0)

            if sector in sector_weights:
                sector_weights[sector] += weight
            else:
                sector_weights[sector] = weight

        return sector_weights

    def _calculate_style_exposure(self, portfolio_data: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate style factor exposures (simplified implementation)."""
        # Simplified style calculation based on market cap and sector
        style_exposure = {factor: 0.0 for factor in self.config['style_factors']}

        total_weight = sum(pos['weight'] for pos in portfolio_data.values())
        if total_weight == 0:
            return style_exposure

        for symbol, position in portfolio_data.items():
            weight = position.get('weight', 0.0)
            market_cap = position.get('market_cap', 1e9)
            sector = position.get('sector', 'Unknown')

            # Simplified style factor calculation
            # Size factor: larger market cap = positive size exposure
            style_exposure['size'] += weight * np.log(market_cap / 1e9) / total_weight

            # Sector-based style proxies
            if sector == 'Technology':
                style_exposure['momentum'] += weight / total_weight * 0.5
            elif sector == 'Financial':
                style_exposure['value'] += weight / total_weight * 0.3
            elif sector == 'Utilities':
                style_exposure['quality'] += weight / total_weight * 0.4
                style_exposure['volatility'] -= weight / total_weight * 0.2

        return style_exposure

    def _calculate_concentration_ratio(self, weights: List[float]) -> float:
        """Calculate concentration ratio (Herfindahl-Hirschman Index)."""
        if not weights:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        # Normalize weights
        normalized_weights = [w / total_weight for w in weights]

        # Calculate HHI
        hhi = sum(w ** 2 for w in normalized_weights)
        return hhi

    def _calculate_effective_positions(self, weights: List[float]) -> int:
        """Calculate effective number of positions."""
        if not weights:
            return 0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0

        # Normalize weights
        normalized_weights = [w / total_weight for w in weights if w > 0]

        # Effective positions = 1 / sum(weight^2)
        sum_squared = sum(w ** 2 for w in normalized_weights)
        effective_positions = 1.0 / sum_squared if sum_squared > 0 else 0

        return int(round(effective_positions))

    def _empty_metrics(self) -> ExposureMetrics:
        """Return empty metrics for empty portfolio."""
        return ExposureMetrics(
            sector_exposure={},
            style_exposure={factor: 0.0 for factor in self.config['style_factors']},
            concentration_ratio=0.0,
            max_position_weight=0.0,
            effective_positions=0,
            total_exposure=0.0,
            leverage_ratio=0.0,
            timestamp=datetime.now()
        )

    def check_risk_limits(self, metrics: ExposureMetrics) -> List[str]:
        """Check if portfolio violates risk limits."""
        violations = []

        # Check sector concentration
        for sector, exposure in metrics.sector_exposure.items():
            if exposure > self.config['max_sector_exposure']:
                violations.append(f"Sector {sector} exposure {exposure:.1%} exceeds limit {self.config['max_sector_exposure']:.1%}")

        # Check position concentration
        if metrics.max_position_weight > self.config['max_position_weight']:
            violations.append(f"Max position weight {metrics.max_position_weight:.1%} exceeds limit {self.config['max_position_weight']:.1%}")

        # Check overall concentration
        if metrics.concentration_ratio > self.config['concentration_threshold']:
            violations.append(f"Portfolio concentration {metrics.concentration_ratio:.3f} exceeds threshold {self.config['concentration_threshold']:.3f}")

        return violations

    def generate_risk_report(self, metrics: ExposureMetrics) -> Dict[str, Any]:
        """Generate comprehensive risk exposure report."""
        violations = self.check_risk_limits(metrics)

        return {
            'timestamp': metrics.timestamp.isoformat(),
            'summary': {
                'total_exposure': metrics.total_exposure,
                'effective_positions': metrics.effective_positions,
                'concentration_ratio': metrics.concentration_ratio,
                'max_position_weight': metrics.max_position_weight,
                'leverage_ratio': metrics.leverage_ratio
            },
            'sector_breakdown': metrics.sector_exposure,
            'style_factors': metrics.style_exposure,
            'risk_violations': violations,
            'risk_status': 'SAFE' if not violations else 'WARNING' if len(violations) <= 2 else 'CRITICAL'
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the risk exposure monitor
    monitor = RiskExposureMonitor()

    # Sample portfolio data
    portfolio_data = {
        'AAPL': {'weight': 0.15, 'sector': 'Technology', 'market_cap': 3e12},
        'MSFT': {'weight': 0.12, 'sector': 'Technology', 'market_cap': 2.8e12},
        'JPM': {'weight': 0.08, 'sector': 'Financial', 'market_cap': 500e9},
        'JNJ': {'weight': 0.06, 'sector': 'Healthcare', 'market_cap': 400e9},
        'XOM': {'weight': 0.05, 'sector': 'Energy', 'market_cap': 300e9}
    }

    # Calculate metrics
    metrics = monitor.calculate_exposure_metrics(portfolio_data)

    # Generate report
    report = monitor.generate_risk_report(metrics)

    print("Risk Exposure Analysis:")
    print(f"Total Exposure: {metrics.total_exposure:.1%}")
    print(f"Effective Positions: {metrics.effective_positions}")
    print(f"Concentration Ratio: {metrics.concentration_ratio:.3f}")
    print(f"Max Position Weight: {metrics.max_position_weight:.1%}")

    print("\nSector Breakdown:")
    for sector, weight in metrics.sector_exposure.items():
        print(f"  {sector}: {weight:.1%}")

    print(f"\nRisk Status: {report['risk_status']}")
    if report['risk_violations']:
        print("Risk Violations:")
        for violation in report['risk_violations']:
            print(f"  - {violation}")