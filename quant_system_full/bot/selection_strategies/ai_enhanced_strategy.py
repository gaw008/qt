#!/usr/bin/env python3
"""
AI-Enhanced Selection Strategy
Integrates AI training manager recommendations with traditional selection methods
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add dashboard backend to path for AI imports
dashboard_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "backend"))
if dashboard_backend_path not in sys.path:
    sys.path.append(dashboard_backend_path)

try:
    from real_ai_training_manager import real_ai_manager
    from real_ai_recommendations import get_real_ai_recommendations
    AI_AVAILABLE = True
except ImportError as e:
    print(f"[AI_STRATEGY] AI modules not available: {e}")
    AI_AVAILABLE = False

# Import existing strategies for fallback
try:
    from .value_momentum import ValueMomentumStrategy
    from .technical_breakout import TechnicalBreakoutStrategy
    from .earnings_momentum import EarningsMomentumStrategy
    TRADITIONAL_STRATEGIES_AVAILABLE = True
except ImportError:
    print("[AI_STRATEGY] Traditional strategies not available")
    TRADITIONAL_STRATEGIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class AIEnhancedSelectionStrategy:
    """
    AI-Enhanced stock selection strategy that combines traditional factors with AI recommendations
    """

    def __init__(self):
        self.ai_manager = real_ai_manager if AI_AVAILABLE else None
        self.traditional_strategies = {}

        if TRADITIONAL_STRATEGIES_AVAILABLE:
            try:
                self.traditional_strategies = {
                    'value_momentum': ValueMomentumStrategy(),
                    'technical_breakout': TechnicalBreakoutStrategy(),
                    'earnings_momentum': EarningsMomentumStrategy()
                }
            except Exception as e:
                logger.warning(f"Failed to initialize traditional strategies: {e}")

    def select_stocks(self, universe: List[str], market_data: Dict[str, Any],
                     limit: int = 20, ai_weight: float = 0.4) -> List[Dict[str, Any]]:
        """
        Select stocks using AI-enhanced methodology

        Args:
            universe: List of stock symbols to evaluate
            market_data: Market data for the symbols
            limit: Maximum number of stocks to select
            ai_weight: Weight given to AI recommendations (0.0 to 1.0)

        Returns:
            List of selected stocks with scores and reasoning
        """
        try:
            logger.info(f"AI-Enhanced selection started for {len(universe)} symbols")

            # Step 1: Get AI recommendations if available
            ai_scores = {}
            if AI_AVAILABLE and self.ai_manager:
                ai_scores = self._get_ai_scores(universe)
                logger.info(f"AI scores obtained for {len(ai_scores)} symbols")

            # Step 2: Get traditional strategy scores
            traditional_scores = {}
            if TRADITIONAL_STRATEGIES_AVAILABLE and self.traditional_strategies:
                traditional_scores = self._get_traditional_scores(universe, market_data)
                logger.info(f"Traditional scores obtained for {len(traditional_scores)} symbols")

            # Step 3: Combine AI and traditional scores
            combined_scores = self._combine_scores(universe, ai_scores, traditional_scores, ai_weight)

            # Step 4: Rank and select top candidates
            selected_stocks = self._rank_and_select(combined_scores, limit)

            logger.info(f"AI-Enhanced selection completed: {len(selected_stocks)} stocks selected")
            return selected_stocks

        except Exception as e:
            logger.error(f"AI-Enhanced selection failed: {e}")
            # Fallback to traditional selection if available
            return self._fallback_selection(universe, market_data, limit)

    def _get_ai_scores(self, universe: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get AI-based scores for universe symbols"""
        try:
            # Get current training progress
            training_progress = self.ai_manager.get_training_progress()

            # Generate AI recommendations
            ai_recommendations = get_real_ai_recommendations(training_progress)

            # Extract AI scores for our universe
            ai_scores = {}

            if ai_recommendations.get('status') == 'success' or ai_recommendations.get('top_picks'):
                # Use top picks if available
                top_picks = ai_recommendations.get('top_picks', [])

                for pick in top_picks:
                    symbol = pick.get('symbol')
                    if symbol in universe:
                        ai_scores[symbol] = {
                            'ai_score': pick.get('confidence', 0.5),
                            'ai_action': pick.get('action', 'HOLD'),
                            'ai_expected_return': pick.get('expected_return', 0.0),
                            'ai_risk_score': pick.get('risk_score', 0.5),
                            'ai_reasons': pick.get('reasons', [])
                        }

            # For symbols not in AI recommendations, assign neutral scores
            for symbol in universe:
                if symbol not in ai_scores:
                    ai_scores[symbol] = {
                        'ai_score': 0.5,
                        'ai_action': 'HOLD',
                        'ai_expected_return': 0.0,
                        'ai_risk_score': 0.5,
                        'ai_reasons': ['No AI recommendation available']
                    }

            return ai_scores

        except Exception as e:
            logger.error(f"Failed to get AI scores: {e}")
            return {}

    def _get_traditional_scores(self, universe: List[str], market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get traditional strategy-based scores"""
        try:
            traditional_scores = {}

            for symbol in universe:
                symbol_data = market_data.get(symbol)
                if symbol_data is None:
                    continue

                scores = {}

                # Get scores from each traditional strategy
                for strategy_name, strategy in self.traditional_strategies.items():
                    try:
                        # Call strategy's evaluate method (assuming it exists)
                        if hasattr(strategy, 'evaluate'):
                            score = strategy.evaluate(symbol, symbol_data)
                            scores[f'{strategy_name}_score'] = score
                        elif hasattr(strategy, 'select_stocks'):
                            # For strategies that return selections, we'll need to adapt
                            result = strategy.select_stocks([symbol], {symbol: symbol_data}, limit=1)
                            scores[f'{strategy_name}_score'] = 1.0 if result else 0.0
                    except Exception as e:
                        logger.warning(f"Strategy {strategy_name} failed for {symbol}: {e}")
                        scores[f'{strategy_name}_score'] = 0.5

                # Calculate composite traditional score
                if scores:
                    avg_score = sum(scores.values()) / len(scores)
                    traditional_scores[symbol] = {
                        'traditional_score': avg_score,
                        'individual_scores': scores
                    }
                else:
                    traditional_scores[symbol] = {
                        'traditional_score': 0.5,
                        'individual_scores': {}
                    }

            return traditional_scores

        except Exception as e:
            logger.error(f"Failed to get traditional scores: {e}")
            return {}

    def _combine_scores(self, universe: List[str], ai_scores: Dict, traditional_scores: Dict,
                       ai_weight: float) -> Dict[str, Dict[str, Any]]:
        """Combine AI and traditional scores with specified weighting"""
        combined_scores = {}
        traditional_weight = 1.0 - ai_weight

        for symbol in universe:
            ai_data = ai_scores.get(symbol, {})
            traditional_data = traditional_scores.get(symbol, {})

            # Get base scores
            ai_score = ai_data.get('ai_score', 0.5)
            traditional_score = traditional_data.get('traditional_score', 0.5)

            # Calculate weighted composite score
            composite_score = (ai_score * ai_weight) + (traditional_score * traditional_weight)

            # Combine risk assessments
            ai_risk = ai_data.get('ai_risk_score', 0.5)
            composite_risk = ai_risk  # Use AI risk as primary, could be enhanced

            # Combine expected returns
            ai_return = ai_data.get('ai_expected_return', 0.0)
            expected_return = ai_return  # Use AI return estimate

            # Create reasoning
            reasons = []
            if ai_data.get('ai_reasons'):
                reasons.extend([f"AI: {reason}" for reason in ai_data['ai_reasons']])
            if traditional_data.get('individual_scores'):
                for strategy, score in traditional_data['individual_scores'].items():
                    reasons.append(f"{strategy}: {score:.2f}")

            combined_scores[symbol] = {
                'symbol': symbol,
                'composite_score': composite_score,
                'ai_score': ai_score,
                'traditional_score': traditional_score,
                'risk_score': composite_risk,
                'expected_return': expected_return,
                'ai_action': ai_data.get('ai_action', 'HOLD'),
                'reasons': reasons,
                'ai_weight_used': ai_weight,
                'timestamp': datetime.now().isoformat()
            }

        return combined_scores

    def _rank_and_select(self, scores: Dict[str, Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Rank stocks by composite score and select top candidates"""
        try:
            # Convert to list and sort by composite score
            ranked_stocks = list(scores.values())
            ranked_stocks.sort(key=lambda x: x['composite_score'], reverse=True)

            # Apply additional filters
            filtered_stocks = []
            for stock in ranked_stocks:
                # Filter out very low scores
                if stock['composite_score'] < 0.3:
                    continue

                # Filter out high risk unless score is very high
                if stock['risk_score'] > 0.8 and stock['composite_score'] < 0.8:
                    continue

                # Prefer AI buy recommendations
                if stock['ai_action'] == 'BUY' or stock['composite_score'] > 0.7:
                    filtered_stocks.append(stock)

                if len(filtered_stocks) >= limit:
                    break

            # If we don't have enough, fill with highest scoring remaining
            if len(filtered_stocks) < limit:
                remaining = [s for s in ranked_stocks if s not in filtered_stocks]
                filtered_stocks.extend(remaining[:limit - len(filtered_stocks)])

            return filtered_stocks[:limit]

        except Exception as e:
            logger.error(f"Failed to rank and select stocks: {e}")
            return list(scores.values())[:limit]

    def _fallback_selection(self, universe: List[str], market_data: Dict[str, Any],
                           limit: int) -> List[Dict[str, Any]]:
        """Fallback selection when AI is not available"""
        try:
            logger.warning("Using fallback selection - AI not available")

            if TRADITIONAL_STRATEGIES_AVAILABLE and self.traditional_strategies:
                # Use first available traditional strategy
                strategy_name, strategy = next(iter(self.traditional_strategies.items()))
                logger.info(f"Using {strategy_name} as fallback")

                if hasattr(strategy, 'select_stocks'):
                    return strategy.select_stocks(universe, market_data, limit)

            # Ultimate fallback - random selection from universe
            import random
            fallback_selection = random.sample(universe, min(limit, len(universe)))
            return [{'symbol': symbol, 'composite_score': 0.5, 'source': 'random_fallback'}
                   for symbol in fallback_selection]

        except Exception as e:
            logger.error(f"Fallback selection failed: {e}")
            return []

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and capabilities"""
        try:
            status = {
                'ai_available': AI_AVAILABLE,
                'traditional_strategies_available': TRADITIONAL_STRATEGIES_AVAILABLE,
                'strategy_name': 'AI-Enhanced Selection',
                'capabilities': []
            }

            if AI_AVAILABLE:
                # Get AI training status
                training_progress = self.ai_manager.get_training_progress()
                status['ai_training_status'] = training_progress.get('status', 'unknown')
                status['ai_sharpe_ratio'] = training_progress.get('sharpe_ratio', 0.0)
                status['capabilities'].append('AI Recommendations')

            if TRADITIONAL_STRATEGIES_AVAILABLE:
                status['traditional_strategies'] = list(self.traditional_strategies.keys())
                status['capabilities'].extend(['Traditional Factors', 'Multi-Strategy Fusion'])

            return status

        except Exception as e:
            logger.error(f"Failed to get strategy status: {e}")
            return {'error': str(e)}


# Factory function for easy integration
def create_ai_enhanced_strategy():
    """Create and return an AI-enhanced selection strategy instance"""
    return AIEnhancedSelectionStrategy()