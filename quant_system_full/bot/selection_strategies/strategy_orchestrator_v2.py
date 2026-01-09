"""
Strategy Orchestrator V2 - Independent Selection Engine

Orchestrates improved strategies with risk management and fallback mechanism.
Completely independent from original system.

Key features:
- Style diversification (40% value + 30% momentum + 30% balanced)
- Risk management integration
- Market regime filtering
- Automatic fallback to original strategies on error
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bot.selection_strategies.base_strategy import SelectionCriteria, StrategyResults

logger = logging.getLogger(__name__)


class StrategyOrchestratorV2:
    """
    Improved strategy orchestrator with risk management.

    Manages:
    - Multiple improved strategies
    - Style diversification
    - Risk management integration
    - Fallback to original strategies
    """

    def __init__(self, enable_improved: bool = False, config_path: Optional[str] = None):
        """
        Initialize orchestrator.

        Args:
            enable_improved: Enable improved strategies
            config_path: Path to configuration file
        """
        self.enable_improved = enable_improved
        self.config = self._load_config(config_path)

        # Initialize strategies
        self.strategies = []
        self.risk_manager = None
        self.market_filter = None
        self.portfolio_control = None

        if self.enable_improved:
            self._initialize_improved_strategies()
        else:
            self._initialize_original_strategies()

        logger.info(f"StrategyOrchestratorV2 initialized (improved={enable_improved})")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded config from {config_path}")
                    return config
            else:
                logger.info("Using default configuration")
                return self._get_default_config()

        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "enable_improved_strategies": False,
            "fallback_on_error": True,
            "portfolio_construction": {
                "value_allocation": 0.4,
                "momentum_allocation": 0.3,
                "balanced_allocation": 0.3,
                "force_diversification": True
            },
            "risk_management": {
                "stop_loss_percentage": 0.08,
                "max_single_position": 0.15,
                "min_portfolio_stocks": 15
            }
        }

    def _initialize_improved_strategies(self):
        """Initialize improved strategies with risk management."""
        try:
            from bot.selection_strategies.improved_strategies import (
                ImprovedValueMomentumV2, DefensiveValue, BalancedMomentum
            )
            from bot.risk_management_v2 import (
                StopLossManager, MarketRegimeFilter, PortfolioRiskControl
            )

            # Get strategy configurations from config
            value_momentum_config = self.config.get('improved_value_momentum', {})
            defensive_value_config = self.config.get('defensive_value', {})
            balanced_momentum_config = self.config.get('balanced_momentum', {})

            # Initialize strategies with config parameters
            self.strategies = [
                ImprovedValueMomentumV2(
                    value_weight=value_momentum_config.get('value_weight', 0.6),
                    momentum_weight=value_momentum_config.get('momentum_weight', 0.4),
                    momentum_period_long=value_momentum_config.get('momentum_period_long', 252),
                    momentum_period_skip=value_momentum_config.get('momentum_period_skip', 21),
                    max_rsi_threshold=value_momentum_config.get('max_rsi_threshold', 80),
                    max_acceptable_pe=value_momentum_config.get('max_acceptable_pe', 30)
                ),
                DefensiveValue(
                    pe_weight=defensive_value_config.get('pe_weight', 0.4),
                    pb_weight=defensive_value_config.get('pb_weight', 0.3),
                    dividend_weight=defensive_value_config.get('dividend_yield_weight', 0.2),
                    debt_weight=defensive_value_config.get('debt_weight', 0.1),
                    max_pe=defensive_value_config.get('max_pe', 20),
                    min_dividend_yield=defensive_value_config.get('min_dividend_yield', 0.01)
                ),
                BalancedMomentum(
                    momentum_6m_weight=balanced_momentum_config.get('momentum_6m_weight', 0.5),
                    momentum_12m_weight=balanced_momentum_config.get('momentum_12m_weight', 0.3),
                    momentum_3m_weight=balanced_momentum_config.get('momentum_3m_weight', 0.2),
                    require_sustained_volume=balanced_momentum_config.get('require_sustained_volume', True),
                    volume_consistency_window=balanced_momentum_config.get('volume_consistency_window', 60)
                )
            ]

            logger.info(f"Improved strategies initialized with config values:")
            logger.info(f"  - ImprovedValueMomentumV2: max_pe={value_momentum_config.get('max_acceptable_pe', 30)}, max_rsi={value_momentum_config.get('max_rsi_threshold', 80)}")
            logger.info(f"  - DefensiveValue: max_pe={defensive_value_config.get('max_pe', 20)}, min_div={defensive_value_config.get('min_dividend_yield', 0.01):.2%}")
            logger.info(f"  - BalancedMomentum: 6m={balanced_momentum_config.get('momentum_6m_weight', 0.5):.1%}")

            # Initialize risk management
            risk_config = self.config.get('risk_management', {})
            self.risk_manager = StopLossManager(
                stop_loss_pct=risk_config.get('stop_loss_percentage', 0.08),
                trailing_stop_pct=risk_config.get('trailing_stop_percentage', 0.12),
                profit_lock_pct=risk_config.get('profit_lock_percentage', 0.15)
            )

            self.market_filter = MarketRegimeFilter()

            self.portfolio_control = PortfolioRiskControl(
                max_single_position=risk_config.get('max_single_position', 0.15),
                max_sector_exposure=risk_config.get('max_sector_exposure', 0.35),
                min_stocks=risk_config.get('min_portfolio_stocks', 15)
            )

            logger.info("Improved strategies initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize improved strategies: {e}")
            raise

    def _initialize_original_strategies(self):
        """Initialize original strategies as fallback."""
        try:
            from bot.selection_strategies.value_momentum import ValueMomentumStrategy
            from bot.selection_strategies.technical_breakout import TechnicalBreakoutStrategy
            from bot.selection_strategies.earnings_momentum import EarningsMomentumStrategy

            self.strategies = [
                ValueMomentumStrategy(),
                TechnicalBreakoutStrategy(),
                EarningsMomentumStrategy()
            ]

            logger.info("Original strategies initialized as fallback")

        except Exception as e:
            logger.error(f"Failed to initialize original strategies: {e}")
            raise

    def select_stocks_with_risk_management(
        self,
        universe: List[str],
        criteria: SelectionCriteria
    ) -> List[Dict[str, Any]]:
        """
        Select stocks with risk management.

        Args:
            universe: Stock universe
            criteria: Selection criteria

        Returns:
            Combined and validated selections
        """
        try:
            if not self.enable_improved:
                # Use original strategy combination
                return self._run_original_strategies(universe, criteria)

            # Use improved strategy with risk management
            logger.info("[IMPROVED] Running improved strategy orchestration")

            # 1. Market regime filter
            if self.market_filter and self.market_filter.should_reduce_exposure():
                adjusted_max = self.market_filter.get_recommended_max_stocks(criteria.max_stocks)
                logger.warning(f"[IMPROVED] Market regime suggests reducing exposure: {criteria.max_stocks} -> {adjusted_max}")
                criteria.max_stocks = adjusted_max

            # 2. Run all strategies
            all_results = {}
            for strategy in self.strategies:
                try:
                    logger.info(f"[IMPROVED] Running {strategy.name}")
                    results = strategy.select_stocks(universe, criteria)
                    all_results[strategy.name] = results
                    logger.info(f"[IMPROVED] {strategy.name}: {len(results.selected_stocks)} stocks")
                except Exception as e:
                    logger.error(f"[IMPROVED] Error in {strategy.name}: {e}")
                    continue

            if not all_results:
                raise ValueError("No strategies produced results")

            # 3. Combine with style diversification
            combined = self._combine_with_diversification(all_results, criteria)

            # 4. Validate portfolio
            if self.portfolio_control:
                is_valid, violations = self.portfolio_control.validate_portfolio(combined)
                if not is_valid:
                    logger.warning(f"[IMPROVED] Portfolio validation warnings: {len(violations)}")
                    for v in violations:
                        logger.warning(f"  - {v}")

            logger.info(f"[IMPROVED] Final selection: {len(combined)} stocks")
            return combined

        except Exception as e:
            logger.error(f"[IMPROVED] Improved strategies failed: {e}")

            # Fallback to original
            if self.config.get('fallback_on_error', True):
                logger.warning("[IMPROVED] Falling back to original strategies")
                self._initialize_original_strategies()
                return self._run_original_strategies(universe, criteria)
            else:
                raise

    def _run_original_strategies(
        self,
        universe: List[str],
        criteria: SelectionCriteria
    ) -> List[Dict[str, Any]]:
        """Run original strategies (fallback mode)."""
        try:
            logger.info("[ORIGINAL] Running original strategy combination")

            all_results = {}
            for strategy in self.strategies:
                try:
                    results = strategy.select_stocks(universe, criteria)
                    all_results[strategy.name] = results
                except Exception as e:
                    logger.error(f"[ORIGINAL] Error in {strategy.name}: {e}")
                    continue

            # Simple combination (original logic)
            combined = self._combine_simple(all_results)

            logger.info(f"[ORIGINAL] Final selection: {len(combined)} stocks")
            return combined

        except Exception as e:
            logger.error(f"[ORIGINAL] Original strategies failed: {e}")
            return []

    def _combine_with_diversification(
        self,
        all_results: Dict[str, StrategyResults],
        criteria: SelectionCriteria
    ) -> List[Dict[str, Any]]:
        """
        Combine results with forced style diversification and smart fallback.

        Allocation:
        - 40% from value strategies (ImprovedValueMomentumV2, DefensiveValue)
        - 30% from momentum strategies (BalancedMomentum)
        - 30% balanced/consensus

        Smart Fallback:
        - Warns when strategies return zero results
        - Ensures minimum 2 strategies participate (if available)
        - Redistributes slots if a category has zero qualifying stocks
        """
        try:
            portfolio_config = self.config.get('portfolio_construction', {})
            value_alloc = portfolio_config.get('value_allocation', 0.4)
            momentum_alloc = portfolio_config.get('momentum_allocation', 0.3)
            balanced_alloc = portfolio_config.get('balanced_allocation', 0.3)

            max_stocks = criteria.max_stocks
            value_slots = int(max_stocks * value_alloc)
            momentum_slots = int(max_stocks * momentum_alloc)
            balanced_slots = max_stocks - value_slots - momentum_slots

            logger.info(f"[DIVERSIFICATION] Target allocation: {value_slots} value + {momentum_slots} momentum + {balanced_slots} balanced = {max_stocks} total")

            final_selections = []

            # Get value picks
            value_picks = []
            for strategy_name in ['ImprovedValueMomentumV2', 'DefensiveValue']:
                if strategy_name in all_results:
                    results = all_results[strategy_name]
                    strategy_count = len(results.selected_stocks)
                    logger.info(f"[DIVERSIFICATION] {strategy_name} provided {strategy_count} stocks")

                    for stock in results.selected_stocks[:value_slots]:
                        value_picks.append({
                            'symbol': stock.symbol,
                            'score': stock.score,
                            'avg_score': stock.score,
                            'action': stock.action.value,
                            'reasoning': stock.reasoning,
                            'strategy': strategy_name,
                            'style': 'value'
                        })

            # Sort and take top value picks
            value_picks.sort(key=lambda x: x['score'], reverse=True)
            value_actual = value_picks[:value_slots]
            final_selections.extend(value_actual)

            # SMART FALLBACK: Check if value strategies failed
            if len(value_actual) == 0:
                logger.warning(f"[DIVERSIFICATION] VALUE STRATEGIES RETURNED ZERO STOCKS!")
                logger.warning(f"[DIVERSIFICATION] This indicates overly strict criteria. Check diagnostic logs for details.")
                logger.warning(f"[DIVERSIFICATION] Reallocating {value_slots} value slots to balanced pool")
                balanced_slots += value_slots  # Redistribute value slots to balanced
            elif len(value_actual) < value_slots:
                shortage = value_slots - len(value_actual)
                logger.warning(f"[DIVERSIFICATION] Value strategies only provided {len(value_actual)}/{value_slots} stocks (shortage: {shortage})")
                logger.warning(f"[DIVERSIFICATION] Reallocating {shortage} unfilled value slots to balanced pool")
                balanced_slots += shortage

            # Get momentum picks
            momentum_picks = []
            for strategy_name in ['BalancedMomentum']:
                if strategy_name in all_results:
                    results = all_results[strategy_name]
                    strategy_count = len(results.selected_stocks)
                    logger.info(f"[DIVERSIFICATION] {strategy_name} provided {strategy_count} stocks")

                    for stock in results.selected_stocks[:momentum_slots]:
                        momentum_picks.append({
                            'symbol': stock.symbol,
                            'score': stock.score,
                            'avg_score': stock.score,
                            'action': stock.action.value,
                            'reasoning': stock.reasoning,
                            'strategy': strategy_name,
                            'style': 'momentum'
                        })

            momentum_picks.sort(key=lambda x: x['score'], reverse=True)
            momentum_actual = momentum_picks[:momentum_slots]
            final_selections.extend(momentum_actual)

            # SMART FALLBACK: Check if momentum strategies failed
            if len(momentum_actual) == 0:
                logger.warning(f"[DIVERSIFICATION] MOMENTUM STRATEGIES RETURNED ZERO STOCKS!")
                logger.warning(f"[DIVERSIFICATION] Reallocating {momentum_slots} momentum slots to balanced pool")
                balanced_slots += momentum_slots
            elif len(momentum_actual) < momentum_slots:
                shortage = momentum_slots - len(momentum_actual)
                logger.warning(f"[DIVERSIFICATION] Momentum strategies only provided {len(momentum_actual)}/{momentum_slots} stocks (shortage: {shortage})")
                logger.warning(f"[DIVERSIFICATION] Reallocating {shortage} unfilled momentum slots to balanced pool")
                balanced_slots += shortage

            # Fill remaining with highest scores from any strategy
            used_symbols = {s['symbol'] for s in final_selections}
            all_picks = []
            for strategy_name, results in all_results.items():
                for stock in results.selected_stocks:
                    if stock.symbol not in used_symbols:
                        all_picks.append({
                            'symbol': stock.symbol,
                            'score': stock.score,
                            'avg_score': stock.score,
                            'action': stock.action.value,
                            'reasoning': stock.reasoning,
                            'strategy': strategy_name,
                            'style': 'balanced'
                        })

            all_picks.sort(key=lambda x: x['score'], reverse=True)
            balanced_actual = all_picks[:balanced_slots]
            final_selections.extend(balanced_actual)

            # SMART FALLBACK: Final diversity check
            unique_strategies = len(set(s.get('strategy', 'unknown') for s in final_selections))
            logger.info(f"[DIVERSIFICATION] Final selection: {len(final_selections)} stocks from {unique_strategies} strategies")

            if unique_strategies < 2 and len(all_results) >= 2:
                logger.error(f"[DIVERSIFICATION] CRITICAL: Only {unique_strategies} strategy contributed despite {len(all_results)} available!")
                logger.error(f"[DIVERSIFICATION] Strategy diversification FAILED. Review selection criteria urgently.")

            # Count actual distribution by style
            style_counts = {}
            for stock in final_selections:
                style = stock.get('style', 'unknown')
                style_counts[style] = style_counts.get(style, 0) + 1

            logger.info(f"[DIVERSIFICATION] Style distribution: {style_counts}")
            logger.info(f"[DIVERSIFICATION] Target was {value_slots} value + {momentum_slots} momentum + {balanced_slots - value_slots - momentum_slots if len(value_actual)==0 or len(momentum_actual)==0 else balanced_slots} balanced")

            return final_selections[:max_stocks]

        except Exception as e:
            logger.error(f"Error in diversified combination: {e}")
            return self._combine_simple(all_results)

    def _combine_simple(self, all_results: Dict[str, StrategyResults]) -> List[Dict[str, Any]]:
        """Simple combination with consensus bonus (original logic)."""
        try:
            combined = {}

            for strategy_name, results in all_results.items():
                for stock in results.selected_stocks:
                    symbol = stock.symbol

                    if symbol not in combined:
                        combined[symbol] = {
                            'symbol': symbol,
                            'total_score': 0,
                            'strategy_count': 0,
                            'strategies': []
                        }

                    combined[symbol]['total_score'] += stock.score
                    combined[symbol]['strategy_count'] += 1
                    combined[symbol]['strategies'].append(strategy_name)

            # Calculate average and add consensus bonus
            final = []
            for symbol, data in combined.items():
                avg_score = data['total_score'] / data['strategy_count']
                consensus_bonus = min(10.0, data['strategy_count'] * 2.5)
                final_score = avg_score + consensus_bonus

                final.append({
                    'symbol': symbol,
                    'score': final_score,
                    'avg_score': avg_score,
                    'strategy_count': data['strategy_count']
                })

            final.sort(key=lambda x: x['score'], reverse=True)
            return final[:20]

        except Exception as e:
            logger.error(f"Error in simple combination: {e}")
            return []
