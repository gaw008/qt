#!/usr/bin/env python3
"""
AI Integration Module for Quantitative Trading System
Integrates AI learning, strategy optimization, and enhanced selection into the worker
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Add bot path for AI imports
BOT_PATH = str(Path(__file__).parent.parent.parent / "bot")
if BOT_PATH not in sys.path:
    sys.path.insert(0, BOT_PATH)

logger = logging.getLogger(__name__)

# Import AI modules with graceful fallback
try:
    from bot.ai_learning_engine import AILearningEngine
    AI_LEARNING_AVAILABLE = True
    logger.info("[AI_INTEGRATION] AI Learning Engine imported successfully")
except ImportError as e:
    AI_LEARNING_AVAILABLE = False
    logger.warning(f"[AI_INTEGRATION] AI Learning Engine not available: {e}")

try:
    from bot.ai_strategy_optimizer import AIStrategyOptimizer
    AI_OPTIMIZER_AVAILABLE = True
    logger.info("[AI_INTEGRATION] AI Strategy Optimizer imported successfully")
except ImportError as e:
    AI_OPTIMIZER_AVAILABLE = False
    logger.warning(f"[AI_INTEGRATION] AI Strategy Optimizer not available: {e}")

try:
    from bot.selection_strategies.ai_enhanced_strategy import AIEnhancedSelectionStrategy
    AI_STRATEGY_AVAILABLE = True
    logger.info("[AI_INTEGRATION] AI Enhanced Strategy imported successfully")
except ImportError as e:
    AI_STRATEGY_AVAILABLE = False
    logger.warning(f"[AI_INTEGRATION] AI Enhanced Strategy not available: {e}")


class AIIntegrationManager:
    """
    Manages AI/ML integration for the quantitative trading system.
    Coordinates AI learning, strategy optimization, and enhanced stock selection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AI integration manager.

        Args:
            config: Configuration dictionary with AI settings
        """
        self.config = config or {}
        self.ai_learning_engine = None
        self.ai_optimizer = None
        self.ai_strategy = None
        self.enabled = False
        self.last_training_time = None
        self.last_optimization_time = None
        self.training_interval = self.config.get('training_interval', 3600 * 24)  # Daily by default
        self.optimization_interval = self.config.get('optimization_interval', 3600 * 6)  # Every 6 hours

        # Performance tracking
        self.ai_stats = {
            'total_training_runs': 0,
            'total_optimization_runs': 0,
            'total_ai_selections': 0,
            'last_ai_score': None,
            'avg_ai_improvement': 0.0
        }

        self._initialize_ai_components()

    def _initialize_ai_components(self):
        """Initialize AI components if available."""
        try:
            # Initialize AI Learning Engine
            if AI_LEARNING_AVAILABLE:
                self.ai_learning_engine = AILearningEngine()  # Use default parameters
                logger.info("[AI_INTEGRATION] AI Learning Engine initialized")

            # Initialize AI Strategy Optimizer
            if AI_OPTIMIZER_AVAILABLE:
                self.ai_optimizer = AIStrategyOptimizer()  # Use default parameters
                logger.info("[AI_INTEGRATION] AI Strategy Optimizer initialized")

            # Initialize AI Enhanced Strategy
            if AI_STRATEGY_AVAILABLE:
                self.ai_strategy = AIEnhancedSelectionStrategy()
                logger.info("[AI_INTEGRATION] AI Enhanced Strategy initialized")

            # Enable AI if at least one component is available
            self.enabled = any([
                AI_LEARNING_AVAILABLE,
                AI_OPTIMIZER_AVAILABLE,
                AI_STRATEGY_AVAILABLE
            ])

            if self.enabled:
                logger.info("[AI_INTEGRATION] AI capabilities ENABLED")
            else:
                logger.warning("[AI_INTEGRATION] No AI components available - running in traditional mode")

        except Exception as e:
            logger.error(f"[AI_INTEGRATION] Failed to initialize AI components: {e}")
            self.enabled = False

    def is_enabled(self) -> bool:
        """Check if AI integration is enabled."""
        return self.enabled

    def should_run_training(self) -> bool:
        """Check if AI training should run now."""
        if not self.enabled or not AI_LEARNING_AVAILABLE:
            return False

        if self.last_training_time is None:
            return True

        time_since_training = (datetime.now() - self.last_training_time).total_seconds()
        return time_since_training >= self.training_interval

    def should_run_optimization(self) -> bool:
        """Check if strategy optimization should run now."""
        if not self.enabled or not AI_OPTIMIZER_AVAILABLE:
            return False

        if self.last_optimization_time is None:
            return True

        time_since_optimization = (datetime.now() - self.last_optimization_time).total_seconds()
        return time_since_optimization >= self.optimization_interval

    def run_ai_training(self, market_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run AI model training on recent market data.

        Args:
            market_data: Optional market data for training

        Returns:
            Training results dictionary
        """
        if not AI_LEARNING_AVAILABLE or not self.ai_learning_engine:
            return {'success': False, 'error': 'AI Learning Engine not available'}

        try:
            import asyncio
            from bot.ai_learning_engine import ModelType

            logger.info("[AI_TRAINING] Starting AI model training...")

            # Get simulation data from AI engine
            sim_data = self.ai_learning_engine._simulation_state
            features = sim_data['features']
            targets = sim_data['target']
            feature_names = sim_data['feature_names']

            # Train multiple model types asynchronously
            model_types = [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, ModelType.RIDGE_REGRESSION]

            async def train_models_async():
                trained_models = []
                for model_type in model_types:
                    try:
                        model_id = await self.ai_learning_engine.train_model(
                            model_type, features, targets, feature_names
                        )
                        if model_id:
                            trained_models.append(model_id)
                    except Exception as e:
                        logger.warning(f"[AI_TRAINING] Failed to train {model_type.value}: {e}")
                return trained_models

            # Run async training
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            trained_models = loop.run_until_complete(train_models_async())
            loop.close()

            self.last_training_time = datetime.now()
            self.ai_stats['total_training_runs'] += 1

            logger.info(f"[AI_TRAINING] Training complete - Trained {len(trained_models)} models")

            return {
                'success': True,
                'models_trained': len(trained_models),
                'model_ids': trained_models,
                'timestamp': self.last_training_time.isoformat()
            }

        except Exception as e:
            logger.error(f"[AI_TRAINING] Training failed: {e}")
            return {'success': False, 'error': str(e)}

    def run_strategy_optimization(self, performance_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run strategy parameter optimization.

        Args:
            performance_data: Optional performance data for optimization

        Returns:
            Optimization results dictionary
        """
        if not AI_OPTIMIZER_AVAILABLE or not self.ai_optimizer:
            return {'success': False, 'error': 'AI Strategy Optimizer not available'}

        try:
            import asyncio
            from bot.ai_strategy_optimizer import StrategyParameter, OptimizationMethod, ObjectiveType

            logger.info("[AI_OPTIMIZATION] Starting strategy optimization...")

            # Define strategy parameters for optimization
            parameters = [
                StrategyParameter(
                    name="lookback_period",
                    param_type="int",
                    bounds=(5, 50),
                    current_value=20,
                    default_value=20,
                    description="Lookback period for momentum"
                ),
                StrategyParameter(
                    name="threshold",
                    param_type="float",
                    bounds=(0.001, 0.05),
                    current_value=0.02,
                    default_value=0.02,
                    description="Signal threshold"
                ),
                StrategyParameter(
                    name="stop_loss",
                    param_type="float",
                    bounds=(0.01, 0.10),
                    current_value=0.05,
                    default_value=0.05,
                    description="Stop loss percentage"
                )
            ]

            # Determine optimization method
            method_str = self.config.get('optimization_method', 'bayesian')
            if method_str == 'bayesian':
                opt_method = OptimizationMethod.BAYESIAN_OPTIMIZATION
            elif method_str == 'optuna':
                opt_method = OptimizationMethod.OPTUNA_TPE
            else:
                opt_method = OptimizationMethod.BAYESIAN_OPTIMIZATION

            async def optimize_async():
                # Start optimization session (runs in background)
                session_id = await self.ai_optimizer.start_optimization(
                    strategy_name="selection_strategy",
                    parameters=parameters,
                    optimization_method=opt_method,
                    objective_type=ObjectiveType.MAXIMIZE_SHARPE,
                    n_trials=20  # Quick optimization
                )

                # Wait a bit for initial results
                await asyncio.sleep(2)

                # Get status
                status = await self.ai_optimizer.get_optimization_status(session_id)
                return status

            # Run async optimization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            optimization_status = loop.run_until_complete(optimize_async())
            loop.close()

            self.last_optimization_time = datetime.now()
            self.ai_stats['total_optimization_runs'] += 1

            best_result = optimization_status.get('best_result', {})
            improvement = 'N/A'
            if best_result and best_result.get('objective_value'):
                improvement = f"{best_result['objective_value']:.4f}"

            logger.info(f"[AI_OPTIMIZATION] Optimization started - Best value so far: {improvement}")

            return {
                'success': True,
                'session_status': optimization_status,
                'timestamp': self.last_optimization_time.isoformat()
            }

        except Exception as e:
            logger.error(f"[AI_OPTIMIZATION] Optimization failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_ai_stock_recommendations(self, universe: List[str],
                                     top_n: int = 20) -> Dict[str, Any]:
        """
        Get AI-enhanced stock recommendations.

        Args:
            universe: List of stock symbols to analyze
            top_n: Number of top stocks to recommend

        Returns:
            AI recommendations dictionary
        """
        if not AI_STRATEGY_AVAILABLE or not self.ai_strategy:
            return {'success': False, 'error': 'AI Enhanced Strategy not available'}

        try:
            logger.info(f"[AI_SELECTION] Getting AI recommendations from {len(universe)} stocks...")

            # Get AI-enhanced selections
            selections = self.ai_strategy.select_stocks(
                universe=universe,
                max_stocks=top_n
            )

            self.ai_stats['total_ai_selections'] += 1
            if selections:
                self.ai_stats['last_ai_score'] = selections[0].get('score') if selections else None

            logger.info(f"[AI_SELECTION] Selected {len(selections)} stocks with AI enhancement")

            return {
                'success': True,
                'selections': selections,
                'count': len(selections),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[AI_SELECTION] Selection failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_ai_status(self) -> Dict[str, Any]:
        """Get current AI integration status and statistics."""
        return {
            'enabled': self.enabled,
            'components': {
                'learning_engine': AI_LEARNING_AVAILABLE,
                'strategy_optimizer': AI_OPTIMIZER_AVAILABLE,
                'enhanced_strategy': AI_STRATEGY_AVAILABLE
            },
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'last_optimization': self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            'statistics': self.ai_stats,
            'intervals': {
                'training_interval': self.training_interval,
                'optimization_interval': self.optimization_interval
            }
        }

    def update_from_trading_results(self, trading_results: Dict[str, Any]):
        """
        Update AI models based on trading results (reinforcement learning).

        Args:
            trading_results: Dictionary with trading execution results
        """
        if not self.enabled:
            return

        try:
            # Extract performance metrics
            if 'execution_results' in trading_results:
                for result in trading_results['execution_results']:
                    if result.get('success'):
                        # Update AI learning from successful trades
                        if AI_LEARNING_AVAILABLE and self.ai_learning_engine:
                            self.ai_learning_engine.update_from_trade(result)

            logger.info("[AI_INTEGRATION] Updated AI models from trading results")

        except Exception as e:
            logger.error(f"[AI_INTEGRATION] Failed to update from trading results: {e}")


# Global AI manager instance
ai_manager = None


def get_ai_manager(config: Optional[Dict[str, Any]] = None) -> AIIntegrationManager:
    """Get or create global AI manager instance."""
    global ai_manager
    if ai_manager is None:
        ai_manager = AIIntegrationManager(config)
    return ai_manager


def initialize_ai(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize AI integration for the trading system.

    Args:
        config: AI configuration dictionary

    Returns:
        True if AI successfully initialized, False otherwise
    """
    manager = get_ai_manager(config)
    return manager.is_enabled()