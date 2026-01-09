#!/usr/bin/env python3
"""
Market State Machine for Dynamic Trading Parameter Adjustment

This module implements a market regime detection system that adapts trading
parameters based on current market conditions including volatility, momentum,
and sentiment indicators.

Market States:
- BULL_MARKET: High momentum, low volatility, positive sentiment
- BEAR_MARKET: Negative momentum, high volatility, negative sentiment
- SIDEWAYS_MARKET: Low momentum, medium volatility, neutral sentiment
- HIGH_VOLATILITY: Extreme volatility regardless of direction
- CRISIS_MODE: Emergency state with extreme risk conditions

The state machine dynamically adjusts:
- Position sizing multipliers
- Risk thresholds
- Factor weights in scoring
- Stop-loss parameters
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Import existing factor modules
try:
    from bot.factors.market_factors import (
        compute_market_heat_index,
        market_sentiment_features
    )
    from bot.factors.momentum_factors import momentum_features
    from bot.factors.technical_factors import technical_features
    HAS_FACTORS = True
except ImportError as e:
    logging.warning(f"Factor modules not available: {e}")
    HAS_FACTORS = False

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market regime states"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS_MODE = "crisis_mode"


@dataclass
class MarketStateParams:
    """Parameters for each market state"""
    position_size_multiplier: float
    risk_threshold_multiplier: float
    stop_loss_multiplier: float
    factor_weights: Dict[str, float]
    max_positions: int
    volatility_target: float

    def __post_init__(self):
        """Validate parameters"""
        if self.position_size_multiplier <= 0:
            raise ValueError("Position size multiplier must be positive")
        if self.risk_threshold_multiplier <= 0:
            raise ValueError("Risk threshold multiplier must be positive")


@dataclass
class MarketStateSignals:
    """Market state detection signals"""
    volatility_percentile: float
    momentum_score: float
    sentiment_score: float
    trend_strength: float
    market_stress: float
    timestamp: datetime


class MarketStateMachine:
    """
    Market state detection and parameter adjustment system
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize market state machine

        Args:
            config: Configuration dictionary for state parameters
        """
        self.config = config or {}
        self.current_state = MarketState.SIDEWAYS_MARKET
        self.state_history = []
        self.state_params = self._initialize_state_params()
        self.signals_history = []

        # State transition thresholds
        self.volatility_high_threshold = self.config.get('volatility_high_threshold', 75)
        self.volatility_crisis_threshold = self.config.get('volatility_crisis_threshold', 95)
        self.momentum_bull_threshold = self.config.get('momentum_bull_threshold', 60)
        self.momentum_bear_threshold = self.config.get('momentum_bear_threshold', -60)
        self.stress_crisis_threshold = self.config.get('stress_crisis_threshold', 80)

        logger.info("Market state machine initialized")

    def _initialize_state_params(self) -> Dict[MarketState, MarketStateParams]:
        """Initialize parameters for each market state"""

        # Default factor weights (can be overridden by config)
        default_weights = {
            'valuation': 0.25,
            'momentum': 0.25,
            'technical': 0.25,
            'volume': 0.15,
            'market_sentiment': 0.10
        }

        return {
            MarketState.BULL_MARKET: MarketStateParams(
                position_size_multiplier=1.2,
                risk_threshold_multiplier=0.8,
                stop_loss_multiplier=1.1,
                factor_weights={
                    'valuation': 0.15,  # Less focus on valuation in bull market
                    'momentum': 0.35,   # More focus on momentum
                    'technical': 0.30,
                    'volume': 0.15,
                    'market_sentiment': 0.05
                },
                max_positions=25,
                volatility_target=0.15
            ),

            MarketState.BEAR_MARKET: MarketStateParams(
                position_size_multiplier=0.6,
                risk_threshold_multiplier=1.5,
                stop_loss_multiplier=0.8,
                factor_weights={
                    'valuation': 0.40,  # More focus on valuation in bear market
                    'momentum': 0.10,   # Less momentum focus
                    'technical': 0.20,
                    'volume': 0.15,
                    'market_sentiment': 0.15
                },
                max_positions=15,
                volatility_target=0.25
            ),

            MarketState.SIDEWAYS_MARKET: MarketStateParams(
                position_size_multiplier=1.0,
                risk_threshold_multiplier=1.0,
                stop_loss_multiplier=1.0,
                factor_weights=default_weights,
                max_positions=20,
                volatility_target=0.20
            ),

            MarketState.HIGH_VOLATILITY: MarketStateParams(
                position_size_multiplier=0.7,
                risk_threshold_multiplier=1.3,
                stop_loss_multiplier=0.9,
                factor_weights={
                    'valuation': 0.30,
                    'momentum': 0.15,
                    'technical': 0.35,  # More technical focus in volatile markets
                    'volume': 0.15,
                    'market_sentiment': 0.05
                },
                max_positions=15,
                volatility_target=0.35
            ),

            MarketState.CRISIS_MODE: MarketStateParams(
                position_size_multiplier=0.3,
                risk_threshold_multiplier=2.0,
                stop_loss_multiplier=0.7,
                factor_weights={
                    'valuation': 0.45,  # Maximum focus on fundamentals
                    'momentum': 0.05,
                    'technical': 0.25,
                    'volume': 0.20,
                    'market_sentiment': 0.05
                },
                max_positions=10,
                volatility_target=0.50
            )
        }

    def calculate_market_signals(self, market_data: Dict[str, pd.DataFrame]) -> MarketStateSignals:
        """
        Calculate market state detection signals

        Args:
            market_data: Dictionary of symbol -> price data

        Returns:
            MarketStateSignals object with current market conditions
        """
        if not market_data or not HAS_FACTORS:
            # Return default signals if no data or factors unavailable
            return MarketStateSignals(
                volatility_percentile=50.0,
                momentum_score=0.0,
                sentiment_score=0.0,
                trend_strength=0.0,
                market_stress=0.0,
                timestamp=datetime.now()
            )

        try:
            # Calculate market-wide volatility
            all_returns = []
            for symbol, df in market_data.items():
                if df is not None and not df.empty and 'close' in df.columns:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        all_returns.extend(returns.tolist())

            if all_returns:
                market_volatility = np.std(all_returns) * np.sqrt(252)  # Annualized
                # Calculate volatility percentile (simplified - in production use rolling window)
                volatility_percentile = min(100, max(0, market_volatility * 100 / 0.6))  # 60% vol = 100th percentile
            else:
                volatility_percentile = 50.0

            # Calculate momentum score using existing momentum factors
            momentum_scores = []
            for symbol, df in market_data.items():
                if df is not None and not df.empty:
                    try:
                        momentum_data = momentum_features(df)
                        if momentum_data is not None and len(momentum_data) > 0:
                            # Take most recent momentum score
                            recent_momentum = momentum_data.iloc[-1] if isinstance(momentum_data, pd.DataFrame) else momentum_data
                            if isinstance(recent_momentum, (pd.Series, dict)):
                                momentum_scores.append(float(list(recent_momentum.values())[0] if isinstance(recent_momentum, dict) else recent_momentum.iloc[0]))
                    except Exception:
                        continue

            momentum_score = np.mean(momentum_scores) if momentum_scores else 0.0

            # Calculate sentiment score using market factors
            sentiment_scores = []
            for symbol, df in market_data.items():
                if df is not None and not df.empty:
                    try:
                        sentiment_data = market_sentiment_features(df, market_data)
                        if sentiment_data is not None and len(sentiment_data) > 0:
                            recent_sentiment = sentiment_data.iloc[-1] if isinstance(sentiment_data, pd.DataFrame) else sentiment_data
                            if isinstance(recent_sentiment, (pd.Series, dict)):
                                sentiment_scores.append(float(list(recent_sentiment.values())[0] if isinstance(recent_sentiment, dict) else recent_sentiment.iloc[0]))
                    except Exception:
                        continue

            sentiment_score = np.mean(sentiment_scores) if sentiment_scores else 0.0

            # Calculate trend strength (simplified)
            trend_strength = abs(momentum_score) if momentum_score else 0.0

            # Calculate market stress index
            market_stress = volatility_percentile * 0.6 + abs(sentiment_score) * 0.4

            return MarketStateSignals(
                volatility_percentile=volatility_percentile,
                momentum_score=momentum_score,
                sentiment_score=sentiment_score,
                trend_strength=trend_strength,
                market_stress=market_stress,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error calculating market signals: {e}")
            return MarketStateSignals(
                volatility_percentile=50.0,
                momentum_score=0.0,
                sentiment_score=0.0,
                trend_strength=0.0,
                market_stress=50.0,
                timestamp=datetime.now()
            )

    def determine_market_state(self, signals: MarketStateSignals) -> MarketState:
        """
        Determine market state based on signals

        Args:
            signals: Market state signals

        Returns:
            Detected market state
        """
        # Crisis mode detection (highest priority)
        if (signals.volatility_percentile > self.volatility_crisis_threshold or
            signals.market_stress > self.stress_crisis_threshold):
            return MarketState.CRISIS_MODE

        # High volatility detection
        if signals.volatility_percentile > self.volatility_high_threshold:
            return MarketState.HIGH_VOLATILITY

        # Bull/bear market detection based on momentum and sentiment
        if (signals.momentum_score > self.momentum_bull_threshold and
            signals.sentiment_score > 0):
            return MarketState.BULL_MARKET

        if (signals.momentum_score < self.momentum_bear_threshold or
            signals.sentiment_score < -50):
            return MarketState.BEAR_MARKET

        # Default to sideways market
        return MarketState.SIDEWAYS_MARKET

    def update_state(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[MarketState, MarketStateParams]:
        """
        Update market state based on current market data

        Args:
            market_data: Dictionary of symbol -> price data

        Returns:
            Tuple of (new_state, state_parameters)
        """
        # Calculate current market signals
        signals = self.calculate_market_signals(market_data)
        self.signals_history.append(signals)

        # Keep only recent signals (last 100 updates)
        if len(self.signals_history) > 100:
            self.signals_history = self.signals_history[-100:]

        # Determine new state
        new_state = self.determine_market_state(signals)

        # Update state if changed
        if new_state != self.current_state:
            logger.info(f"Market state transition: {self.current_state.value} -> {new_state.value}")
            self.state_history.append({
                'timestamp': datetime.now(),
                'old_state': self.current_state,
                'new_state': new_state,
                'signals': signals
            })
            self.current_state = new_state

        # Get parameters for current state
        current_params = self.state_params[self.current_state]

        return self.current_state, current_params

    def get_adjusted_factor_weights(self) -> Dict[str, float]:
        """
        Get factor weights adjusted for current market state

        Returns:
            Dictionary of factor weights
        """
        return self.state_params[self.current_state].factor_weights.copy()

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier for current market state

        Returns:
            Position size multiplier
        """
        return self.state_params[self.current_state].position_size_multiplier

    def get_risk_multiplier(self) -> float:
        """
        Get risk threshold multiplier for current market state

        Returns:
            Risk multiplier
        """
        return self.state_params[self.current_state].risk_threshold_multiplier

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about current market state

        Returns:
            Dictionary with state information
        """
        current_params = self.state_params[self.current_state]
        latest_signals = self.signals_history[-1] if self.signals_history else None

        return {
            'current_state': self.current_state.value,
            'state_duration': len([h for h in self.state_history if h['new_state'] == self.current_state]),
            'parameters': {
                'position_size_multiplier': current_params.position_size_multiplier,
                'risk_threshold_multiplier': current_params.risk_threshold_multiplier,
                'stop_loss_multiplier': current_params.stop_loss_multiplier,
                'max_positions': current_params.max_positions,
                'volatility_target': current_params.volatility_target
            },
            'factor_weights': current_params.factor_weights,
            'latest_signals': {
                'volatility_percentile': latest_signals.volatility_percentile if latest_signals else None,
                'momentum_score': latest_signals.momentum_score if latest_signals else None,
                'sentiment_score': latest_signals.sentiment_score if latest_signals else None,
                'market_stress': latest_signals.market_stress if latest_signals else None,
                'timestamp': latest_signals.timestamp.isoformat() if latest_signals else None
            },
            'state_transitions': len(self.state_history),
            'last_transition': self.state_history[-1]['timestamp'].isoformat() if self.state_history else None
        }


# Factory function for easy integration
def create_market_state_machine(config: Optional[Dict[str, Any]] = None) -> MarketStateMachine:
    """
    Factory function to create market state machine

    Args:
        config: Optional configuration dictionary

    Returns:
        MarketStateMachine instance
    """
    return MarketStateMachine(config)


if __name__ == "__main__":
    # Test the market state machine
    import json

    # Create test market data
    test_data = {}
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        dates = pd.date_range(start='2024-01-01', end='2024-09-20', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
        test_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)

    # Initialize and test state machine
    state_machine = create_market_state_machine()

    # Update state with test data
    state, params = state_machine.update_state(test_data)

    # Print results
    print(f"Detected market state: {state.value}")
    print(f"Position size multiplier: {params.position_size_multiplier}")
    print(f"Risk multiplier: {params.risk_threshold_multiplier}")

    # Print full state info
    state_info = state_machine.get_state_info()
    print("\nFull state information:")
    print(json.dumps(state_info, indent=2, default=str))