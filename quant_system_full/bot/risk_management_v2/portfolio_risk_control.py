"""Portfolio Risk Control - enforces portfolio-level risk constraints"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class PortfolioRiskControl:
    """Enforces portfolio-level risk constraints."""

    def __init__(self,
                 max_single_position: float = 0.15,
                 max_sector_exposure: float = 0.35,
                 min_stocks: int = 15):
        """
        Initialize portfolio risk control.

        Args:
            max_single_position: Max % per stock (e.g., 0.15 = 15%)
            max_sector_exposure: Max % per sector (e.g., 0.35 = 35%)
            min_stocks: Minimum number of stocks for diversification
        """
        self.max_single_position = max_single_position
        self.max_sector_exposure = max_sector_exposure
        self.min_stocks = min_stocks

        logger.info(f"PortfolioRiskControl initialized: max_position={max_single_position:.1%}, "
                    f"max_sector={max_sector_exposure:.1%}, min_stocks={min_stocks}")

    def validate_portfolio(self, selections: List[Dict[str, Any]]) -> tuple:
        """
        Validate portfolio against risk constraints.

        Args:
            selections: List of selected stocks with scores

        Returns:
            (is_valid, violations[])
        """
        try:
            violations = []

            # Check minimum diversification
            if len(selections) < self.min_stocks:
                violations.append(f"Insufficient diversification: {len(selections)} stocks < {self.min_stocks} minimum")

            # Check sector concentration (if sector info available)
            sector_counts = {}
            for stock in selections:
                sector = stock.get('sector', 'Unknown')
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            for sector, count in sector_counts.items():
                sector_exposure = count / len(selections)
                if sector_exposure > self.max_sector_exposure:
                    violations.append(f"Sector over-concentration: {sector} = {sector_exposure:.1%} > {self.max_sector_exposure:.1%}")

            is_valid = len(violations) == 0

            if not is_valid:
                logger.warning(f"Portfolio validation failed: {len(violations)} violations")
                for violation in violations:
                    logger.warning(f"  - {violation}")

            return is_valid, violations

        except Exception as e:
            logger.error(f"Error validating portfolio: {e}")
            return False, [f"Validation error: {e}"]

    def enforce_position_limits(self, selections: List[Dict[str, Any]], total_capital: float) -> List[Dict[str, Any]]:
        """
        Enforce position size limits.

        Args:
            selections: Selected stocks
            total_capital: Total available capital

        Returns:
            Selections with adjusted position sizes
        """
        try:
            if not selections:
                return selections

            # Equal weight by default
            num_stocks = len(selections)
            equal_weight = 1.0 / num_stocks

            # Cap at max_single_position
            max_weight = min(equal_weight, self.max_single_position)

            for stock in selections:
                stock['recommended_weight'] = max_weight
                stock['recommended_value'] = total_capital * max_weight

            return selections

        except Exception as e:
            logger.error(f"Error enforcing position limits: {e}")
            return selections
