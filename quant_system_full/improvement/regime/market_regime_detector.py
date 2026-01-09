"""
Market Regime Detection Engine

This module implements a comprehensive market regime detection system that identifies
bull/bear/sideways market conditions using multiple technical and fundamental indicators.

Key Features:
- VIX level and slope analysis
- Advance/Decline line analysis
- 52-week high/low ratio analysis
- Moving average relationship analysis
- Multi-factor scoring with consensus voting
- State transition debouncing (3-day confirmation)
- Historical regime persistence tracking
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


@dataclass
class RegimeIndicator:
    """Individual regime indicator result."""
    name: str
    value: float
    normalized_score: float
    regime_vote: MarketRegime
    confidence: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    current_regime: MarketRegime
    confidence: float
    indicators: List[RegimeIndicator]
    consensus_score: Dict[str, float]
    days_in_regime: int
    last_regime_change: Optional[str]
    regime_strength: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # VIX thresholds
    vix_low_threshold: float = 15.0
    vix_high_threshold: float = 25.0

    # Moving average periods
    short_ma_period: int = 20
    long_ma_period: int = 50

    # Advance/Decline parameters
    ad_lookback_days: int = 20

    # 52-week high/low parameters
    hl_lookback_days: int = 252  # ~1 year

    # Consensus requirements
    min_consensus_votes: int = 3
    confidence_threshold: float = 0.6

    # State transition debouncing
    debounce_days: int = 3

    # Data requirements
    min_history_days: int = 60


class MarketRegimeDetector:
    """
    Market regime detection engine using multiple technical indicators.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize regime detector.

        Args:
            config: Configuration parameters
        """
        self.config = config or RegimeConfig()

        # State tracking
        self.regime_history: List[Dict] = []
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_start_date: Optional[datetime] = None
        self.days_in_current_regime = 0

        # Indicator cache
        self.indicator_cache: Dict[str, Any] = {}
        self.last_analysis: Optional[RegimeAnalysis] = None

        # State file path
        self.state_file = Path("dashboard/state/regime.json")

        # Load existing state
        self._load_state()

        logger.info("[regime] Market regime detector initialized")

    def detect_regime(self, market_data: pd.DataFrame,
                     vix_data: Optional[pd.DataFrame] = None,
                     advance_decline_data: Optional[pd.DataFrame] = None) -> RegimeAnalysis:
        """
        Detect current market regime using multiple indicators.

        Args:
            market_data: Market price data (OHLCV format)
            vix_data: VIX data (optional)
            advance_decline_data: Advance/decline line data (optional)

        Returns:
            Complete regime analysis
        """
        try:
            if len(market_data) < self.config.min_history_days:
                logger.warning(f"[regime] Insufficient data: {len(market_data)} < {self.config.min_history_days}")
                return self._create_unknown_analysis("Insufficient historical data")

            # Calculate individual indicators
            indicators = []

            # 1. VIX-based indicator
            if vix_data is not None and len(vix_data) > 10:
                vix_indicator = self._analyze_vix(vix_data)
                indicators.append(vix_indicator)

            # 2. Moving average relationship
            ma_indicator = self._analyze_moving_averages(market_data)
            indicators.append(ma_indicator)

            # 3. Price momentum indicator
            momentum_indicator = self._analyze_momentum(market_data)
            indicators.append(momentum_indicator)

            # 4. Volatility indicator
            volatility_indicator = self._analyze_volatility(market_data)
            indicators.append(volatility_indicator)

            # 5. 52-week high/low ratio
            hl_indicator = self._analyze_high_low_ratio(market_data)
            indicators.append(hl_indicator)

            # 6. Advance/Decline analysis (if available)
            if advance_decline_data is not None and len(advance_decline_data) > self.config.ad_lookback_days:
                ad_indicator = self._analyze_advance_decline(advance_decline_data)
                indicators.append(ad_indicator)

            # Calculate consensus
            regime_votes = [ind.regime_vote for ind in indicators if ind.regime_vote != MarketRegime.UNKNOWN]

            if len(regime_votes) < self.config.min_consensus_votes:
                logger.warning(f"[regime] Insufficient valid indicators: {len(regime_votes)}")
                return self._create_unknown_analysis("Insufficient valid indicators")

            # Count votes for each regime
            vote_counts = {regime: 0 for regime in MarketRegime if regime != MarketRegime.UNKNOWN}
            confidence_scores = {regime: [] for regime in MarketRegime if regime != MarketRegime.UNKNOWN}

            for indicator in indicators:
                if indicator.regime_vote != MarketRegime.UNKNOWN:
                    vote_counts[indicator.regime_vote] += 1
                    confidence_scores[indicator.regime_vote].append(indicator.confidence)

            # Determine winning regime
            max_votes = max(vote_counts.values())
            winning_regimes = [regime for regime, votes in vote_counts.items() if votes == max_votes]

            if len(winning_regimes) > 1:
                # Tie-breaking by average confidence
                regime_confidences = {
                    regime: np.mean(confidence_scores[regime]) if confidence_scores[regime] else 0
                    for regime in winning_regimes
                }
                detected_regime = max(regime_confidences, key=regime_confidences.get)
                confidence = regime_confidences[detected_regime]
            else:
                detected_regime = winning_regimes[0]
                confidence = np.mean(confidence_scores[detected_regime]) if confidence_scores[detected_regime] else 0

            # Apply debouncing logic
            final_regime, days_in_regime = self._apply_debouncing(detected_regime)

            # Calculate regime strength
            regime_strength = self._calculate_regime_strength(indicators, final_regime)

            # Create analysis result
            consensus_score = {
                regime.value: (vote_counts[regime] / len(regime_votes)) for regime in vote_counts
            }

            analysis = RegimeAnalysis(
                current_regime=final_regime,
                confidence=confidence,
                indicators=indicators,
                consensus_score=consensus_score,
                days_in_regime=days_in_regime,
                last_regime_change=self.regime_start_date.isoformat() if self.regime_start_date else None,
                regime_strength=regime_strength,
                timestamp=datetime.now().isoformat()
            )

            # Update state
            self._update_state(analysis)
            self.last_analysis = analysis

            logger.info(f"[regime] Detected regime: {final_regime.value} (confidence: {confidence:.2f})")

            return analysis

        except Exception as e:
            logger.error(f"[regime] Regime detection failed: {e}")
            return self._create_unknown_analysis(f"Detection error: {str(e)}")

    def _analyze_vix(self, vix_data: pd.DataFrame) -> RegimeIndicator:
        """Analyze VIX level and slope for regime indication."""
        try:
            current_vix = vix_data['close'].iloc[-1]
            vix_10d_avg = vix_data['close'].rolling(10).mean().iloc[-1]
            vix_slope = (current_vix - vix_10d_avg) / vix_10d_avg

            # Normalize VIX level
            vix_normalized = (current_vix - self.config.vix_low_threshold) / (self.config.vix_high_threshold - self.config.vix_low_threshold)
            vix_normalized = np.clip(vix_normalized, 0, 1)

            # Determine regime vote
            if current_vix < self.config.vix_low_threshold and vix_slope < 0:
                regime_vote = MarketRegime.BULL
                confidence = min(0.9, (self.config.vix_low_threshold - current_vix) / self.config.vix_low_threshold + 0.3)
            elif current_vix > self.config.vix_high_threshold or vix_slope > 0.1:
                regime_vote = MarketRegime.BEAR
                confidence = min(0.9, (current_vix - self.config.vix_high_threshold) / self.config.vix_high_threshold + 0.3)
            else:
                regime_vote = MarketRegime.SIDEWAYS
                confidence = 0.6

            return RegimeIndicator(
                name="VIX_Analysis",
                value=current_vix,
                normalized_score=1 - vix_normalized,  # Lower VIX = higher score
                regime_vote=regime_vote,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"[regime] VIX analysis failed: {e}")
            return RegimeIndicator("VIX_Analysis", 0, 0, MarketRegime.UNKNOWN, 0)

    def _analyze_moving_averages(self, market_data: pd.DataFrame) -> RegimeIndicator:
        """Analyze moving average relationships."""
        try:
            prices = market_data['close']
            short_ma = prices.rolling(self.config.short_ma_period).mean()
            long_ma = prices.rolling(self.config.long_ma_period).mean()

            current_price = prices.iloc[-1]
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]

            # Calculate MA slope
            short_ma_slope = (current_short_ma - short_ma.iloc[-5]) / short_ma.iloc[-5]
            long_ma_slope = (current_long_ma - long_ma.iloc[-10]) / long_ma.iloc[-10]

            # Price position relative to MAs
            price_vs_short = (current_price - current_short_ma) / current_short_ma
            price_vs_long = (current_price - current_long_ma) / current_long_ma

            # MA cross relationship
            ma_cross = (current_short_ma - current_long_ma) / current_long_ma

            # Scoring
            normalized_score = (price_vs_short + price_vs_long + ma_cross) / 3
            normalized_score = (normalized_score + 0.1) / 0.2  # Normalize to 0-1
            normalized_score = np.clip(normalized_score, 0, 1)

            # Regime determination
            if (current_price > current_short_ma > current_long_ma and
                short_ma_slope > 0.01 and long_ma_slope > 0.005):
                regime_vote = MarketRegime.BULL
                confidence = min(0.9, normalized_score + 0.2)
            elif (current_price < current_short_ma < current_long_ma and
                  short_ma_slope < -0.01 and long_ma_slope < -0.005):
                regime_vote = MarketRegime.BEAR
                confidence = min(0.9, (1 - normalized_score) + 0.2)
            else:
                regime_vote = MarketRegime.SIDEWAYS
                confidence = 0.6

            return RegimeIndicator(
                name="Moving_Average_Analysis",
                value=ma_cross,
                normalized_score=normalized_score,
                regime_vote=regime_vote,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"[regime] Moving average analysis failed: {e}")
            return RegimeIndicator("Moving_Average_Analysis", 0, 0, MarketRegime.UNKNOWN, 0)

    def _analyze_momentum(self, market_data: pd.DataFrame) -> RegimeIndicator:
        """Analyze price momentum."""
        try:
            prices = market_data['close']
            returns = prices.pct_change()

            # Various momentum measures
            momentum_5d = (prices.iloc[-1] / prices.iloc[-6] - 1)
            momentum_20d = (prices.iloc[-1] / prices.iloc[-21] - 1)
            momentum_60d = (prices.iloc[-1] / prices.iloc[-61] - 1)

            # Average momentum
            avg_momentum = (momentum_5d + momentum_20d + momentum_60d) / 3

            # Momentum acceleration
            recent_returns = returns.iloc[-10:].mean()
            older_returns = returns.iloc[-30:-10].mean()
            momentum_acceleration = recent_returns - older_returns

            # Scoring
            normalized_score = (avg_momentum + 0.2) / 0.4  # Normalize around +/-20%
            normalized_score = np.clip(normalized_score, 0, 1)

            # Regime determination
            if avg_momentum > 0.05 and momentum_acceleration > 0:
                regime_vote = MarketRegime.BULL
                confidence = min(0.9, normalized_score)
            elif avg_momentum < -0.05 and momentum_acceleration < 0:
                regime_vote = MarketRegime.BEAR
                confidence = min(0.9, 1 - normalized_score)
            else:
                regime_vote = MarketRegime.SIDEWAYS
                confidence = 0.6

            return RegimeIndicator(
                name="Momentum_Analysis",
                value=avg_momentum,
                normalized_score=normalized_score,
                regime_vote=regime_vote,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"[regime] Momentum analysis failed: {e}")
            return RegimeIndicator("Momentum_Analysis", 0, 0, MarketRegime.UNKNOWN, 0)

    def _analyze_volatility(self, market_data: pd.DataFrame) -> RegimeIndicator:
        """Analyze market volatility patterns."""
        try:
            prices = market_data['close']
            returns = prices.pct_change()

            # Calculate volatilities
            vol_10d = returns.rolling(10).std() * np.sqrt(252)
            vol_30d = returns.rolling(30).std() * np.sqrt(252)
            vol_60d = returns.rolling(60).std() * np.sqrt(252)

            current_vol = vol_10d.iloc[-1]
            avg_vol = vol_60d.iloc[-1]

            # Volatility regime
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            vol_trend = (vol_10d.iloc[-1] - vol_30d.iloc[-1]) / vol_30d.iloc[-1]

            # Scoring (lower volatility generally favors bull markets)
            normalized_score = 1 / (1 + vol_ratio)  # Inverse relationship
            normalized_score = np.clip(normalized_score, 0, 1)

            # Regime determination
            if vol_ratio < 0.8 and vol_trend < 0:
                regime_vote = MarketRegime.BULL
                confidence = min(0.8, normalized_score)
            elif vol_ratio > 1.5 or vol_trend > 0.2:
                regime_vote = MarketRegime.BEAR
                confidence = min(0.8, 1 - normalized_score)
            else:
                regime_vote = MarketRegime.SIDEWAYS
                confidence = 0.5

            return RegimeIndicator(
                name="Volatility_Analysis",
                value=current_vol,
                normalized_score=normalized_score,
                regime_vote=regime_vote,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"[regime] Volatility analysis failed: {e}")
            return RegimeIndicator("Volatility_Analysis", 0, 0, MarketRegime.UNKNOWN, 0)

    def _analyze_high_low_ratio(self, market_data: pd.DataFrame) -> RegimeIndicator:
        """Analyze 52-week high/low ratio."""
        try:
            prices = market_data['close']
            highs = market_data['high']
            lows = market_data['low']

            # Calculate 52-week high/low
            lookback = min(self.config.hl_lookback_days, len(prices))
            week_52_high = highs.rolling(lookback).max()
            week_52_low = lows.rolling(lookback).min()

            current_price = prices.iloc[-1]
            current_52w_high = week_52_high.iloc[-1]
            current_52w_low = week_52_low.iloc[-1]

            # Position in 52-week range
            if current_52w_high > current_52w_low:
                hl_position = (current_price - current_52w_low) / (current_52w_high - current_52w_low)
            else:
                hl_position = 0.5

            # Trend in high/low ratio
            hl_trend = (hl_position -
                       ((prices.iloc[-20] - current_52w_low) / (current_52w_high - current_52w_low)
                        if current_52w_high > current_52w_low else 0.5))

            # Scoring
            normalized_score = hl_position

            # Regime determination
            if hl_position > 0.8 and hl_trend > 0:
                regime_vote = MarketRegime.BULL
                confidence = min(0.9, hl_position)
            elif hl_position < 0.2 and hl_trend < 0:
                regime_vote = MarketRegime.BEAR
                confidence = min(0.9, 1 - hl_position)
            else:
                regime_vote = MarketRegime.SIDEWAYS
                confidence = 0.6

            return RegimeIndicator(
                name="High_Low_Ratio",
                value=hl_position,
                normalized_score=normalized_score,
                regime_vote=regime_vote,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"[regime] High/Low ratio analysis failed: {e}")
            return RegimeIndicator("High_Low_Ratio", 0, 0, MarketRegime.UNKNOWN, 0)

    def _analyze_advance_decline(self, ad_data: pd.DataFrame) -> RegimeIndicator:
        """Analyze advance/decline line."""
        try:
            ad_line = ad_data['ad_line'] if 'ad_line' in ad_data.columns else ad_data.iloc[:, 0]

            # AD line momentum
            ad_momentum = (ad_line.iloc[-1] - ad_line.iloc[-self.config.ad_lookback_days]) / ad_line.iloc[-self.config.ad_lookback_days]
            ad_slope = (ad_line.iloc[-1] - ad_line.iloc[-5]) / ad_line.iloc[-5]

            # Normalize score
            normalized_score = (ad_momentum + 0.1) / 0.2
            normalized_score = np.clip(normalized_score, 0, 1)

            # Regime determination
            if ad_momentum > 0.02 and ad_slope > 0:
                regime_vote = MarketRegime.BULL
                confidence = min(0.8, normalized_score)
            elif ad_momentum < -0.02 and ad_slope < 0:
                regime_vote = MarketRegime.BEAR
                confidence = min(0.8, 1 - normalized_score)
            else:
                regime_vote = MarketRegime.SIDEWAYS
                confidence = 0.6

            return RegimeIndicator(
                name="Advance_Decline_Analysis",
                value=ad_momentum,
                normalized_score=normalized_score,
                regime_vote=regime_vote,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"[regime] Advance/Decline analysis failed: {e}")
            return RegimeIndicator("Advance_Decline_Analysis", 0, 0, MarketRegime.UNKNOWN, 0)

    def _apply_debouncing(self, detected_regime: MarketRegime) -> Tuple[MarketRegime, int]:
        """Apply state transition debouncing logic."""
        try:
            # If no current regime, accept the new one immediately
            if self.current_regime == MarketRegime.UNKNOWN:
                self.current_regime = detected_regime
                self.regime_start_date = datetime.now()
                self.days_in_current_regime = 1
                return detected_regime, 1

            # If same regime, increment days
            if detected_regime == self.current_regime:
                self.days_in_current_regime += 1
                return self.current_regime, self.days_in_current_regime

            # Different regime detected - check debouncing
            if not hasattr(self, '_regime_change_buffer'):
                self._regime_change_buffer = []

            # Add to buffer
            self._regime_change_buffer.append({
                'regime': detected_regime,
                'date': datetime.now()
            })

            # Keep only recent days
            cutoff_date = datetime.now() - timedelta(days=self.config.debounce_days)
            self._regime_change_buffer = [
                entry for entry in self._regime_change_buffer
                if entry['date'] > cutoff_date
            ]

            # Check if we have consistent signals
            if len(self._regime_change_buffer) >= self.config.debounce_days:
                recent_regimes = [entry['regime'] for entry in self._regime_change_buffer]
                if all(regime == detected_regime for regime in recent_regimes):
                    # Regime change confirmed
                    logger.info(f"[regime] Regime change confirmed: {self.current_regime.value} -> {detected_regime.value}")
                    self.current_regime = detected_regime
                    self.regime_start_date = datetime.now()
                    self.days_in_current_regime = 1
                    self._regime_change_buffer = []
                    return detected_regime, 1

            # Not enough confirmation, keep current regime
            return self.current_regime, self.days_in_current_regime

        except Exception as e:
            logger.error(f"[regime] Debouncing failed: {e}")
            return self.current_regime, self.days_in_current_regime

    def _calculate_regime_strength(self, indicators: List[RegimeIndicator], regime: MarketRegime) -> float:
        """Calculate the strength of the current regime."""
        try:
            # Find indicators that voted for this regime
            supporting_indicators = [ind for ind in indicators if ind.regime_vote == regime]

            if not supporting_indicators:
                return 0.0

            # Calculate average confidence of supporting indicators
            avg_confidence = np.mean([ind.confidence for ind in supporting_indicators])

            # Weight by number of supporting indicators
            indicator_weight = len(supporting_indicators) / len(indicators)

            # Combine confidence and consensus
            regime_strength = (avg_confidence * 0.7) + (indicator_weight * 0.3)

            return min(1.0, regime_strength)

        except Exception as e:
            logger.error(f"[regime] Regime strength calculation failed: {e}")
            return 0.0

    def _create_unknown_analysis(self, reason: str) -> RegimeAnalysis:
        """Create analysis result for unknown regime."""
        return RegimeAnalysis(
            current_regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            indicators=[],
            consensus_score={regime.value: 0.0 for regime in MarketRegime if regime != MarketRegime.UNKNOWN},
            days_in_regime=0,
            last_regime_change=None,
            regime_strength=0.0,
            timestamp=datetime.now().isoformat()
        )

    def _update_state(self, analysis: RegimeAnalysis):
        """Update regime state and save to file."""
        try:
            # Add to history
            self.regime_history.append({
                'regime': analysis.current_regime.value,
                'confidence': analysis.confidence,
                'strength': analysis.regime_strength,
                'timestamp': analysis.timestamp
            })

            # Keep only recent history (last 100 days)
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]

            # Save state to file
            self._save_state(analysis)

        except Exception as e:
            logger.error(f"[regime] State update failed: {e}")

    def _save_state(self, analysis: RegimeAnalysis):
        """Save current state to JSON file."""
        try:
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state_data = {
                'current_analysis': asdict(analysis),
                'regime_history': self.regime_history,
                'detector_state': {
                    'current_regime': self.current_regime.value,
                    'regime_start_date': self.regime_start_date.isoformat() if self.regime_start_date else None,
                    'days_in_current_regime': self.days_in_current_regime
                },
                'last_update': datetime.now().isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            logger.debug(f"[regime] State saved to {self.state_file}")

        except Exception as e:
            logger.error(f"[regime] Failed to save state: {e}")

    def _load_state(self):
        """Load state from JSON file."""
        try:
            if not self.state_file.exists():
                logger.info("[regime] No existing state file found")
                return

            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            # Load detector state
            detector_state = state_data.get('detector_state', {})
            self.current_regime = MarketRegime(detector_state.get('current_regime', 'unknown'))

            if detector_state.get('regime_start_date'):
                self.regime_start_date = datetime.fromisoformat(detector_state['regime_start_date'])

            self.days_in_current_regime = detector_state.get('days_in_current_regime', 0)

            # Load history
            self.regime_history = state_data.get('regime_history', [])

            logger.info(f"[regime] State loaded: current regime = {self.current_regime.value}")

        except Exception as e:
            logger.error(f"[regime] Failed to load state: {e}")

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get comprehensive regime status summary."""
        try:
            return {
                'current_regime': self.current_regime.value,
                'days_in_regime': self.days_in_current_regime,
                'regime_start_date': self.regime_start_date.isoformat() if self.regime_start_date else None,
                'last_analysis': asdict(self.last_analysis) if self.last_analysis else None,
                'regime_history_length': len(self.regime_history),
                'recent_regime_changes': self.regime_history[-5:] if len(self.regime_history) >= 5 else self.regime_history,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[regime] Failed to get regime summary: {e}")
            return {'error': str(e)}


def create_market_regime_detector(custom_config: Optional[Dict] = None) -> MarketRegimeDetector:
    """
    Create and configure a market regime detector.

    Args:
        custom_config: Custom configuration parameters

    Returns:
        Configured MarketRegimeDetector instance
    """
    config = RegimeConfig()

    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return MarketRegimeDetector(config)


if __name__ == "__main__":
    # Test regime detection with sample data
    print("=== Market Regime Detector Test ===")

    # Create sample market data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    sample_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01),
        'high': 0,
        'low': 0,
        'volume': 1000000
    }, index=dates)

    sample_data['high'] = sample_data['close'] * 1.02
    sample_data['low'] = sample_data['close'] * 0.98

    # Create detector and analyze
    detector = create_market_regime_detector()
    analysis = detector.detect_regime(sample_data)

    print(f"Detected Regime: {analysis.current_regime.value}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"Regime Strength: {analysis.regime_strength:.2f}")
    print(f"Active Indicators: {len(analysis.indicators)}")

    for indicator in analysis.indicators:
        print(f"  - {indicator.name}: {indicator.regime_vote.value} (confidence: {indicator.confidence:.2f})")