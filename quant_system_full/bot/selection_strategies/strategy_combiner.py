"""
Selection Strategy Combiner

This module combines results from multiple selection strategies to create
a consensus-based stock selection system with enhanced reliability.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import logging

try:
    from .base_strategy import (
        BaseSelectionStrategy, SelectionResult, SelectionCriteria, 
        StrategyResults, SelectionAction
    )
    from .value_momentum import ValueMomentumStrategy
    from .technical_breakout import TechnicalBreakoutStrategy
    from .earnings_momentum import EarningsMomentumStrategy
except ImportError:
    from base_strategy import (
        BaseSelectionStrategy, SelectionResult, SelectionCriteria, 
        StrategyResults, SelectionAction
    )
    from value_momentum import ValueMomentumStrategy
    from technical_breakout import TechnicalBreakoutStrategy
    from earnings_momentum import EarningsMomentumStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyWeight:
    """Weight configuration for individual strategies."""
    strategy_name: str
    weight: float
    min_score_threshold: float = 0.0


@dataclass
class CombinedResult:
    """Combined result from multiple strategies."""
    symbol: str
    combined_score: float
    strategy_scores: Dict[str, float]
    strategy_actions: Dict[str, SelectionAction]
    consensus_action: SelectionAction
    consensus_reasoning: str
    strategy_count: int
    confidence: float
    metrics: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'combined_score': self.combined_score,
            'strategy_scores': self.strategy_scores,
            'strategy_actions': {k: v.value for k, v in self.strategy_actions.items()},
            'consensus_action': self.consensus_action.value,
            'consensus_reasoning': self.consensus_reasoning,
            'strategy_count': self.strategy_count,
            'confidence': self.confidence,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }


class StrategyCombiner:
    """
    Combines results from multiple selection strategies to create
    a consensus-based selection system.
    """
    
    def __init__(self, 
                 strategy_weights: Optional[List[StrategyWeight]] = None,
                 min_strategy_agreement: int = 2,
                 confidence_threshold: float = 0.6):
        """
        Initialize strategy combiner.
        
        Args:
            strategy_weights: List of strategy weights. Uses default if None.
            min_strategy_agreement: Minimum strategies that must agree
            confidence_threshold: Minimum confidence for final selection
        """
        self.strategy_weights = strategy_weights or self._get_default_weights()
        self.min_strategy_agreement = min_strategy_agreement
        self.confidence_threshold = confidence_threshold
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        logger.info(f"Strategy combiner initialized with {len(self.strategies)} strategies")
    
    def _get_default_weights(self) -> List[StrategyWeight]:
        """Get default strategy weights."""
        return [
            StrategyWeight("ValueMomentum", weight=0.4, min_score_threshold=40.0),
            StrategyWeight("TechnicalBreakout", weight=0.35, min_score_threshold=50.0),
            StrategyWeight("EarningsMomentum", weight=0.25, min_score_threshold=55.0)
        ]
    
    def _initialize_strategies(self) -> Dict[str, BaseSelectionStrategy]:
        """Initialize strategy instances."""
        strategies = {}
        
        for weight_config in self.strategy_weights:
            try:
                if weight_config.strategy_name == "ValueMomentum":
                    strategies[weight_config.strategy_name] = ValueMomentumStrategy()
                elif weight_config.strategy_name == "TechnicalBreakout":
                    strategies[weight_config.strategy_name] = TechnicalBreakoutStrategy()
                elif weight_config.strategy_name == "EarningsMomentum":
                    strategies[weight_config.strategy_name] = EarningsMomentumStrategy()
                else:
                    logger.warning(f"Unknown strategy: {weight_config.strategy_name}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {weight_config.strategy_name}: {e}")
        
        return strategies
    
    def run_combined_selection(
        self,
        universe: List[str],
        criteria: Optional[SelectionCriteria] = None
    ) -> StrategyResults:
        """
        Run combined selection across all strategies.
        
        Args:
            universe: List of stock symbols to evaluate
            criteria: Selection criteria
            
        Returns:
            Combined strategy results
        """
        start_time = time.time()
        criteria = criteria or SelectionCriteria()
        
        logger.info(f"Running combined selection on {len(universe)} stocks using {len(self.strategies)} strategies")
        
        # Run each strategy
        strategy_results = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                logger.info(f"Running {strategy_name} strategy...")
                
                # Adjust criteria for this strategy
                strategy_criteria = self._adjust_criteria_for_strategy(criteria, strategy_name)
                
                # Run strategy
                results = strategy.select_stocks(universe, strategy_criteria)
                strategy_results[strategy_name] = results
                
                logger.info(f"{strategy_name}: {len(results.selected_stocks)} stocks selected")
                
            except Exception as e:
                logger.error(f"Error running {strategy_name}: {e}")
                # Create empty results for failed strategy
                strategy_results[strategy_name] = StrategyResults(
                    strategy_name=strategy_name,
                    selected_stocks=[],
                    total_candidates=0,
                    execution_time=0,
                    errors=[str(e)]
                )
        
        # Combine strategy results
        combined_results = self._combine_strategy_results(strategy_results)
        
        # Apply final filters and ranking
        final_selections = self._apply_final_selection(combined_results, criteria)
        
        execution_time = time.time() - start_time
        
        # Convert to SelectionResult format for compatibility
        selection_results = []
        for combined_result in final_selections:
            selection_result = SelectionResult(
                symbol=combined_result.symbol,
                score=combined_result.combined_score,
                action=combined_result.consensus_action,
                reasoning=combined_result.consensus_reasoning,
                metrics=combined_result.metrics,
                confidence=combined_result.confidence
            )
            selection_results.append(selection_result)
        
        # Create final results
        results = StrategyResults(
            strategy_name="CombinedStrategies",
            selected_stocks=selection_results,
            total_candidates=len(universe),
            execution_time=execution_time,
            criteria_used=criteria,
            metadata={
                'strategies_used': list(self.strategies.keys()),
                'strategy_weights': {w.strategy_name: w.weight for w in self.strategy_weights},
                'combined_results_count': len(combined_results),
                'final_selections_count': len(final_selections)
            }
        )
        
        logger.info(f"Combined selection completed: {len(final_selections)} stocks selected in {execution_time:.2f}s")
        return results
    
    def _adjust_criteria_for_strategy(self, base_criteria: SelectionCriteria, strategy_name: str) -> SelectionCriteria:
        """Adjust selection criteria for specific strategy."""
        criteria = SelectionCriteria(
            max_stocks=base_criteria.max_stocks * 2,  # Allow more candidates from each strategy
            min_market_cap=base_criteria.min_market_cap,
            max_market_cap=base_criteria.max_market_cap,
            min_volume=base_criteria.min_volume,
            min_price=base_criteria.min_price,
            max_price=base_criteria.max_price,
            exclude_sectors=base_criteria.exclude_sectors,
            include_sectors=base_criteria.include_sectors,
            exclude_symbols=base_criteria.exclude_symbols,
            custom_params=base_criteria.custom_params.copy()
        )
        
        # Set strategy-specific minimum score threshold
        weight_config = next((w for w in self.strategy_weights if w.strategy_name == strategy_name), None)
        if weight_config:
            criteria.min_score_threshold = weight_config.min_score_threshold
        
        return criteria
    
    def _combine_strategy_results(self, strategy_results: Dict[str, StrategyResults]) -> List[CombinedResult]:
        """Combine results from multiple strategies."""
        # Collect all symbols that appeared in any strategy
        all_symbols = set()
        for results in strategy_results.values():
            for stock in results.selected_stocks:
                all_symbols.add(stock.symbol)
        
        logger.info(f"Combining results for {len(all_symbols)} unique symbols")
        
        combined_results = []
        
        for symbol in all_symbols:
            # Collect data from each strategy for this symbol
            strategy_scores = {}
            strategy_actions = {}
            strategy_metrics = {}
            
            for strategy_name, results in strategy_results.items():
                # Find this symbol in strategy results
                stock_result = next((s for s in results.selected_stocks if s.symbol == symbol), None)
                
                if stock_result:
                    strategy_scores[strategy_name] = stock_result.score
                    strategy_actions[strategy_name] = stock_result.action
                    strategy_metrics[strategy_name] = stock_result.metrics
                else:
                    # Symbol not selected by this strategy - assign neutral score
                    strategy_scores[strategy_name] = 0.0
                    strategy_actions[strategy_name] = SelectionAction.AVOID
            
            # Calculate combined score
            combined_score = self._calculate_combined_score(strategy_scores)
            
            # Determine consensus action
            consensus_action, reasoning = self._determine_consensus_action(
                strategy_actions, strategy_scores, combined_score
            )
            
            # Calculate confidence
            confidence = self._calculate_consensus_confidence(
                strategy_scores, strategy_actions
            )
            
            # Combine metrics
            combined_metrics = self._combine_metrics(strategy_metrics)
            
            # Create combined result
            combined_result = CombinedResult(
                symbol=symbol,
                combined_score=combined_score,
                strategy_scores=strategy_scores,
                strategy_actions=strategy_actions,
                consensus_action=consensus_action,
                consensus_reasoning=reasoning,
                strategy_count=len([s for s in strategy_scores.values() if s > 0]),
                confidence=confidence,
                metrics=combined_metrics,
                timestamp=datetime.now()
            )
            
            combined_results.append(combined_result)
        
        return combined_results
    
    def _calculate_combined_score(self, strategy_scores: Dict[str, float]) -> float:
        """Calculate weighted combined score."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for strategy_name, score in strategy_scores.items():
            weight_config = next((w for w in self.strategy_weights if w.strategy_name == strategy_name), None)
            if weight_config and score > 0:
                total_weighted_score += score * weight_config.weight
                total_weight += weight_config.weight
        
        if total_weight == 0:
            return 0.0
        
        return total_weighted_score / total_weight
    
    def _determine_consensus_action(
        self,
        strategy_actions: Dict[str, SelectionAction],
        strategy_scores: Dict[str, float],
        combined_score: float
    ) -> Tuple[SelectionAction, str]:
        """Determine consensus action from strategy actions."""
        # Count actions (excluding AVOID for stocks with 0 score)
        action_counts = {}
        participating_strategies = []
        
        for strategy_name, action in strategy_actions.items():
            score = strategy_scores.get(strategy_name, 0)
            if score > 0:  # Only count strategies that actually selected this stock
                action_counts[action] = action_counts.get(action, 0) + 1
                participating_strategies.append(strategy_name)
        
        if not participating_strategies:
            return SelectionAction.AVOID, "No strategies recommended this stock"
        
        # Find most common action
        if not action_counts:
            return SelectionAction.AVOID, "No positive recommendations"
        
        most_common_action = max(action_counts.items(), key=lambda x: x[1])
        action, count = most_common_action
        
        # Check if minimum agreement is met
        if count < self.min_strategy_agreement:
            return SelectionAction.WATCH, f"Limited consensus ({count}/{len(participating_strategies)} strategies agree)"
        
        # Create reasoning
        strategy_list = ", ".join(participating_strategies)
        reasoning = f"Consensus from {count}/{len(participating_strategies)} strategies ({strategy_list})"
        
        # Adjust action based on combined score
        if combined_score >= 80 and action in [SelectionAction.BUY, SelectionAction.STRONG_BUY]:
            return SelectionAction.STRONG_BUY, f"{reasoning}; high combined score ({combined_score:.1f})"
        elif combined_score >= 60 and action in [SelectionAction.BUY, SelectionAction.STRONG_BUY]:
            return SelectionAction.BUY, f"{reasoning}; good combined score ({combined_score:.1f})"
        elif combined_score >= 40:
            return SelectionAction.WATCH, f"{reasoning}; moderate combined score ({combined_score:.1f})"
        else:
            return SelectionAction.AVOID, f"{reasoning}; low combined score ({combined_score:.1f})"
    
    def _calculate_consensus_confidence(
        self,
        strategy_scores: Dict[str, float],
        strategy_actions: Dict[str, SelectionAction]
    ) -> float:
        """Calculate confidence level for consensus result."""
        participating_strategies = [name for name, score in strategy_scores.items() if score > 0]
        
        if not participating_strategies:
            return 0.0
        
        # Base confidence on number of strategies
        base_confidence = len(participating_strategies) / len(self.strategies)
        
        # Adjust for score consistency
        scores = [score for score in strategy_scores.values() if score > 0]
        if scores:
            score_std = np.std(scores) if len(scores) > 1 else 0
            score_consistency = max(0, 1 - (score_std / 50))  # Normalize std dev
            base_confidence *= score_consistency
        
        # Adjust for action agreement
        actions = [action for name, action in strategy_actions.items() 
                  if strategy_scores.get(name, 0) > 0]
        if actions:
            most_common_count = max([actions.count(action) for action in set(actions)])
            action_agreement = most_common_count / len(actions)
            base_confidence *= action_agreement
        
        return min(1.0, base_confidence)
    
    def _combine_metrics(self, strategy_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine metrics from different strategies."""
        combined = {}
        
        # Collect all unique metric keys
        all_keys = set()
        for metrics in strategy_metrics.values():
            all_keys.update(metrics.keys())
        
        # For each metric, try to aggregate across strategies
        for key in all_keys:
            values = []
            for strategy_name, metrics in strategy_metrics.items():
                if key in metrics and metrics[key] is not None:
                    try:
                        float(metrics[key])  # Check if numeric
                        values.append(metrics[key])
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric values
            
            if values:
                if len(values) == 1:
                    combined[key] = values[0]
                else:
                    # Take average for multiple values
                    combined[f"avg_{key}"] = sum(values) / len(values)
                    combined[f"max_{key}"] = max(values)
                    combined[f"min_{key}"] = min(values)
        
        return combined
    
    def _apply_final_selection(
        self,
        combined_results: List[CombinedResult],
        criteria: SelectionCriteria
    ) -> List[CombinedResult]:
        """Apply final filters and ranking to combined results."""
        # Filter by confidence threshold
        filtered_results = [
            result for result in combined_results
            if result.confidence >= self.confidence_threshold
        ]
        
        logger.info(f"After confidence filter: {len(filtered_results)}/{len(combined_results)} results")
        
        # Filter by minimum strategy participation
        filtered_results = [
            result for result in filtered_results
            if result.strategy_count >= self.min_strategy_agreement
        ]
        
        logger.info(f"After strategy agreement filter: {len(filtered_results)}/{len(combined_results)} results")
        
        # Sort by combined score
        filtered_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply max_stocks limit
        final_results = filtered_results[:criteria.max_stocks]
        
        return final_results
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of configured strategies."""
        return {
            'total_strategies': len(self.strategies),
            'strategy_weights': {w.strategy_name: w.weight for w in self.strategy_weights},
            'min_strategy_agreement': self.min_strategy_agreement,
            'confidence_threshold': self.confidence_threshold,
            'strategies': [strategy.get_strategy_info() for strategy in self.strategies.values()]
        }


# Convenience function for easy usage
def run_combined_stock_selection(
    universe: List[str],
    max_stocks: int = 20,
    min_strategy_agreement: int = 2,
    confidence_threshold: float = 0.6
) -> StrategyResults:
    """
    Convenience function to run combined stock selection.
    
    Args:
        universe: List of stock symbols to evaluate
        max_stocks: Maximum number of stocks to select
        min_strategy_agreement: Minimum strategies that must agree
        confidence_threshold: Minimum confidence for final selection
        
    Returns:
        Combined strategy results
    """
    criteria = SelectionCriteria(max_stocks=max_stocks)
    combiner = StrategyCombiner(
        min_strategy_agreement=min_strategy_agreement,
        confidence_threshold=confidence_threshold
    )
    
    return combiner.run_combined_selection(universe, criteria)