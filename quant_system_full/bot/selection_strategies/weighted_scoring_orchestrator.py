"""
Weighted Scoring Orchestrator

Multi-strategy stock selection using weighted average scoring across:
- Momentum (40%): BalancedMomentum - medium to long-term momentum trends
- Value (30%): ImprovedValueMomentumV2 - value-momentum combination
- Technical (15%): TechnicalBreakoutStrategy - technical breakout patterns
- Earnings (15%): EarningsMomentumStrategy - earnings growth and momentum

Each stock receives a score from each strategy (0-100), and the final score
is calculated as a weighted average based on configured weights.
"""

import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base_strategy import SelectionCriteria, StrategyResults, SelectionResult, SelectionAction
from .improved_strategies.balanced_momentum import BalancedMomentum
from .improved_strategies.improved_value_momentum_v2 import ImprovedValueMomentumV2
from .technical_breakout import TechnicalBreakoutStrategy
from .earnings_momentum import EarningsMomentumStrategy

logger = logging.getLogger(__name__)


class WeightedScoringOrchestrator:
    """
    Orchestrates multiple selection strategies with weighted scoring.

    Runs all configured strategies in parallel and combines their scores
    using weighted averages to produce final stock selections.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize weighted scoring orchestrator.

        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.weights = self.config.get('weighted_scoring', {}).get('weights', {
            'momentum': 0.40,
            'value': 0.30,
            'technical': 0.15,
            'earnings': 0.15
        })

        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum:.3f}, normalizing to 1.0")
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}

        # Initialize strategies
        self.strategies = self._initialize_strategies()

        logger.info(f"Initialized WeightedScoringOrchestrator with weights:")
        for name, weight in self.weights.items():
            logger.info(f"  - {name}: {weight:.1%}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / 'config' / 'selection_config_v2.json'

        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}

    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize all four strategies with config parameters."""
        try:
            strategy_configs = self.config.get('weighted_scoring', {}).get('strategies', {})

            # Momentum Strategy
            momentum_config = strategy_configs.get('momentum', {})
            momentum = BalancedMomentum(
                momentum_6m_weight=momentum_config.get('momentum_6m_weight', 0.5),
                momentum_12m_weight=momentum_config.get('momentum_12m_weight', 0.3),
                momentum_3m_weight=momentum_config.get('momentum_3m_weight', 0.2),
                require_sustained_volume=momentum_config.get('require_sustained_volume', True),
                volume_consistency_window=momentum_config.get('volume_consistency_window', 60)
            )

            # Value Strategy
            value_config = strategy_configs.get('value', {})
            value = ImprovedValueMomentumV2(
                value_weight=value_config.get('value_weight', 0.6),
                momentum_weight=value_config.get('momentum_weight', 0.4),
                max_rsi_threshold=value_config.get('max_rsi_threshold', 80),
                max_acceptable_pe=value_config.get('max_acceptable_pe', 30)
            )

            # Technical Strategy
            technical_config = strategy_configs.get('technical', {})
            technical = TechnicalBreakoutStrategy(
                sma_short=technical_config.get('sma_short', 20),
                sma_long=technical_config.get('sma_long', 50),
                bb_period=technical_config.get('bb_period', 20),
                bb_std=technical_config.get('bb_std', 2.0),
                volume_threshold=technical_config.get('volume_threshold', 1.5)
            )

            # Earnings Strategy
            earnings_config = strategy_configs.get('earnings', {})
            earnings = EarningsMomentumStrategy(
                earnings_growth_weight=earnings_config.get('earnings_growth_weight', 0.4),
                surprise_weight=earnings_config.get('surprise_weight', 0.3),
                revision_weight=earnings_config.get('revision_weight', 0.3),
                min_earnings_growth=earnings_config.get('min_earnings_growth', 0.1)
            )

            strategies = {
                'momentum': momentum,
                'value': value,
                'technical': technical,
                'earnings': earnings
            }

            logger.info(f"Initialized {len(strategies)} strategies:")
            for name, strategy in strategies.items():
                logger.info(f"  - {name}: {strategy.name}")

            return strategies

        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            raise

    def select_stocks(
        self,
        universe: List[str],
        criteria: Optional[SelectionCriteria] = None
    ) -> List[Dict[str, Any]]:
        """
        Select stocks using weighted scoring across all strategies.

        Args:
            universe: List of stock symbols to evaluate
            criteria: Selection criteria (max_stocks, min_score_threshold, etc.)

        Returns:
            List of selected stocks with weighted scores and component scores
        """
        start_time = time.time()

        if criteria is None:
            criteria = SelectionCriteria()

        logger.info(f"[WEIGHTED SCORING] Starting selection on {len(universe)} stocks")
        logger.info(f"[WEIGHTED SCORING] Weights: Momentum={self.weights['momentum']:.1%}, "
                   f"Value={self.weights['value']:.1%}, Technical={self.weights['technical']:.1%}, "
                   f"Earnings={self.weights['earnings']:.1%}")

        # Step 1: Run all strategies and collect scores
        all_strategy_results = {}
        for strategy_name, strategy in self.strategies.items():
            try:
                logger.info(f"[WEIGHTED SCORING] Running {strategy_name} strategy...")
                # Run strategy with relaxed max_stocks to get more candidates
                expanded_criteria = SelectionCriteria(
                    max_stocks=min(100, len(universe)),  # Get up to 100 candidates per strategy
                    min_score_threshold=0.0,  # Strategy-level: no filtering, return all scored stocks
                    min_market_cap=criteria.min_market_cap,
                    min_volume=criteria.min_volume
                )
                results = strategy.select_stocks(universe, expanded_criteria)
                all_strategy_results[strategy_name] = results
                logger.info(f"[WEIGHTED SCORING] {strategy_name} selected {len(results.selected_stocks)} stocks")
            except Exception as e:
                logger.error(f"[WEIGHTED SCORING] Error running {strategy_name} strategy: {e}")
                all_strategy_results[strategy_name] = StrategyResults(
                    strategy_name=strategy_name,
                    selected_stocks=[],
                    total_candidates=0,
                    execution_time=0.0,
                    criteria_used=criteria,
                    errors=[str(e)]
                )

        # Step 2: Collect all scores by symbol
        stock_scores = {}  # {symbol: {strategy_name: score}}
        stock_details = {}  # {symbol: {reasoning, action, metrics, etc.}}

        for strategy_name, results in all_strategy_results.items():
            for stock in results.selected_stocks:
                if stock.symbol not in stock_scores:
                    stock_scores[stock.symbol] = {}
                    stock_details[stock.symbol] = {}

                stock_scores[stock.symbol][strategy_name] = stock.score
                stock_details[stock.symbol][strategy_name] = {
                    'score': stock.score,
                    'action': stock.action,
                    'reasoning': stock.reasoning,
                    'metrics': stock.metrics,
                    'confidence': stock.confidence
                }

        logger.info(f"[WEIGHTED SCORING] Collected scores for {len(stock_scores)} unique stocks")

        # Step 3: Calculate weighted scores
        weighted_selections = []
        for symbol, scores in stock_scores.items():
            weighted_score, strategy_count = self._calculate_weighted_score(scores)

            # Generate combined reasoning
            reasoning = self._generate_combined_reasoning(symbol, scores, stock_details.get(symbol, {}))

            # Determine overall action based on weighted score
            if weighted_score >= 80:
                action = SelectionAction.STRONG_BUY
            elif weighted_score >= 65:
                action = SelectionAction.BUY
            elif weighted_score >= 50:
                action = SelectionAction.WATCH
            else:
                action = SelectionAction.AVOID

            weighted_selections.append({
                'symbol': symbol,
                'score': weighted_score,
                'avg_score': weighted_score,  # For compatibility with status logging
                'action': action.value,
                'reasoning': reasoning,
                'strategy_count': strategy_count,
                'component_scores': scores
            })

        # Step 4: Sort by weighted score and apply selection criteria
        weighted_selections.sort(key=lambda x: x['score'], reverse=True)

        # Save a copy before filtering for logging
        all_selections_before_filter = list(weighted_selections)

        # Log all weighted scores BEFORE filtering (for debugging)
        logger.info(f"[WEIGHTED SCORING] All {len(weighted_selections)} stocks with weighted scores:")
        for i, s in enumerate(weighted_selections[:20], 1):  # Show top 20
            logger.info(f"  {i}. {s['symbol']}: {s['score']:.2f} (from {s['strategy_count']} strategies) "
                       f"- {s['component_scores']}")

        # Filter by minimum score threshold
        before_filter = len(weighted_selections)
        weighted_selections = [
            s for s in weighted_selections
            if s['score'] >= criteria.min_score_threshold
        ]
        after_filter = len(weighted_selections)

        # Log filtering statistics
        if before_filter > after_filter:
            logger.warning(f"[WEIGHTED SCORING] {before_filter - after_filter} stocks filtered out "
                          f"by threshold {criteria.min_score_threshold}")
            # Show highest score that didn't make it
            if after_filter == 0 and len(all_selections_before_filter) > 0:
                highest = all_selections_before_filter[0]
                logger.warning(f"[WEIGHTED SCORING] Highest score (below threshold): "
                              f"{highest['score']:.2f} ({highest['symbol']})")

        # Limit to max_stocks
        final_selections = weighted_selections[:criteria.max_stocks]

        execution_time = time.time() - start_time

        logger.info(f"[WEIGHTED SCORING] Final selection: {len(final_selections)} stocks")
        logger.info(f"[WEIGHTED SCORING] Execution time: {execution_time:.2f}s")

        # Log top 5 selections
        logger.info(f"[WEIGHTED SCORING] Top 5 selections:")
        for i, stock in enumerate(final_selections[:5], 1):
            logger.info(f"  {i}. {stock['symbol']}: score={stock['score']:.1f} "
                       f"(from {stock['strategy_count']} strategies)")
            logger.info(f"     Component scores: {stock['component_scores']}")

        return final_selections

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> tuple[float, int]:
        """
        Calculate weighted average score.

        Args:
            scores: Dictionary of {strategy_name: score}

        Returns:
            (weighted_score, strategy_count) tuple
        """
        weighted_sum = 0.0
        total_weight = 0.0
        strategy_count = 0

        for strategy_name, weight in self.weights.items():
            if strategy_name in scores:
                score = scores[strategy_name]
                # CRITICAL FIX: Normalize value strategy (0-50 scale) to 0-100 scale
                # Value strategy returns 0-50, others return 0-100
                if strategy_name == 'value':
                    score = score * 2.0  # Convert 0-50 to 0-100 scale
                weighted_sum += weight * score
                total_weight += weight
                strategy_count += 1
            else:
                # Missing strategy: use neutral score (50)
                weighted_sum += weight * 50.0
                total_weight += weight

        if total_weight == 0:
            return 0.0, 0

        weighted_score = weighted_sum / total_weight
        return weighted_score, strategy_count

    def _generate_combined_reasoning(
        self,
        symbol: str,
        scores: Dict[str, float],
        details: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate combined reasoning from all contributing strategies.

        Args:
            symbol: Stock symbol
            scores: Dictionary of {strategy_name: score}
            details: Dictionary of {strategy_name: {reasoning, action, etc.}}

        Returns:
            Combined reasoning string
        """
        weighted_score, strategy_count = self._calculate_weighted_score(scores)

        # Start with overall assessment
        reasoning = f"Multi-strategy score: {weighted_score:.1f} (from {strategy_count} strategies)"

        # Add top 2 contributing strategies
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_strategies = sorted_strategies[:2]

        strategy_names = {
            'momentum': 'Momentum',
            'value': 'Value',
            'technical': 'Technical',
            'earnings': 'Earnings'
        }

        for strategy_name, score in top_strategies:
            display_name = strategy_names.get(strategy_name, strategy_name)
            reasoning += f"; {display_name}: {score:.1f}"

        return reasoning


def create_weighted_scoring_orchestrator(config_path: Optional[str] = None) -> WeightedScoringOrchestrator:
    """
    Factory function to create a weighted scoring orchestrator.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Initialized WeightedScoringOrchestrator instance
    """
    return WeightedScoringOrchestrator(config_path=config_path)
