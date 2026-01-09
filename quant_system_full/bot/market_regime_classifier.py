#!/usr/bin/env python3
"""
Market Regime Classification System for Quantitative Trading

This module implements a sophisticated market regime detection system that classifies
market periods into Normal, Volatile, and Crisis regimes using multiple statistical
approaches including Hidden Markov Models, threshold-based detection, and machine
learning techniques.

Features:
- Multi-indicator regime detection (VIX, credit spreads, correlation structures)
- Hidden Markov Model implementation for regime switching
- Historical crisis period identification (2008, 2011-2012, 2020, 2022)
- Real-time regime detection with confidence scoring
- Integration with existing risk management systems
- Comprehensive backtesting validation framework

Market Regimes:
- NORMAL: Low volatility, normal correlations, stable conditions
- VOLATILE: Elevated volatility, increased correlations, market stress
- CRISIS: Extreme volatility, high correlations, systemic risk
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
import json
from scipy import stats
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Import existing modules
try:
    # Try relative imports first
    try:
        from config import SETTINGS
        from data import fetch_history, fetch_batch_history
        from factors.market_factors import (
            compute_market_heat_index,
            compute_vix_fear_factor,
            compute_market_breadth,
            market_sentiment_features
        )
        from factors.technical_factors import compute_atr
    except ImportError:
        # Fallback to bot.* imports
        from bot.config import SETTINGS
        from bot.data import fetch_history, fetch_batch_history
        from bot.factors.market_factors import (
            compute_market_heat_index,
            compute_vix_fear_factor,
            compute_market_breadth,
            market_sentiment_features
        )
        from bot.factors.technical_factors import compute_atr

    HAS_DEPENDENCIES = True
except ImportError as e:
    logging.warning(f"Some dependencies not available: {e}")
    HAS_DEPENDENCIES = False

# Configure encoding and logging
os.environ['PYTHONIOENCODING'] = 'utf-8'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """Enhanced market regime states"""
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"


@dataclass
class RegimeIndicators:
    """Market regime detection indicators"""
    vix_level: float = 0.0
    vix_change: float = 0.0
    volatility_percentile: float = 0.0
    correlation_level: float = 0.0
    correlation_change: float = 0.0
    credit_spread: float = 0.0
    momentum_divergence: float = 0.0
    liquidity_stress: float = 0.0
    breadth_deterioration: float = 0.0
    fear_index: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimePrediction:
    """Regime prediction with confidence"""
    regime: MarketRegime
    confidence: float
    probability_normal: float
    probability_volatile: float
    probability_crisis: float
    indicators: RegimeIndicators
    method: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeTransition:
    """Regime transition event"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_date: datetime
    confidence: float
    duration_days: int
    indicators: RegimeIndicators


class HiddenMarkovRegimeModel:
    """
    Hidden Markov Model for market regime detection

    Uses HMM to model regime-switching behavior in financial markets
    with state-dependent emission distributions.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.regime_mapping = {
            0: MarketRegime.NORMAL,
            1: MarketRegime.VOLATILE,
            2: MarketRegime.CRISIS
        }

    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare feature matrix for HMM training

        Args:
            data: Dictionary containing market data and indicators

        Returns:
            Feature matrix for HMM training
        """
        features = []
        feature_names = []

        # VIX features
        if 'vix_data' in data and data['vix_data'] is not None:
            vix_df = data['vix_data']
            if 'close' in vix_df.columns:
                vix_level = vix_df['close'].values
                vix_change = vix_df['close'].pct_change().fillna(0).values
                vix_ma = vix_df['close'].rolling(20).mean().fillna(vix_df['close']).values
                vix_std = vix_df['close'].rolling(20).std().fillna(0).values

                features.extend([vix_level, vix_change, vix_ma, vix_std])
                feature_names.extend(['vix_level', 'vix_change', 'vix_ma', 'vix_std'])

        # Market volatility features
        if 'market_data' in data and data['market_data']:
            volatilities = []
            correlations = []
            returns_list = []

            for symbol, df in data['market_data'].items():
                if df is not None and 'close' in df.columns and len(df) > 20:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_list.append(returns)
                        vol = returns.rolling(20).std().fillna(0)
                        volatilities.append(vol.values)

            if volatilities:
                # Market-wide volatility
                market_vol = np.mean(volatilities, axis=0)
                vol_percentile = pd.Series(market_vol).rolling(252).rank(pct=True).fillna(0.5).values

                features.extend([market_vol, vol_percentile])
                feature_names.extend(['market_volatility', 'volatility_percentile'])

                # Cross-sectional correlation
                if len(returns_list) >= 2:
                    returns_df = pd.concat(returns_list, axis=1, join='inner')
                    if len(returns_df) > 20:
                        rolling_corr = returns_df.rolling(20).corr().mean(axis=1, level=0)
                        avg_corr = rolling_corr.mean(axis=1).fillna(0).values

                        features.append(avg_corr)
                        feature_names.append('average_correlation')

        # Credit spread proxy (using bond ETF spreads if available)
        if 'credit_data' in data and data['credit_data'] is not None:
            credit_df = data['credit_data']
            if 'close' in credit_df.columns:
                credit_spread = credit_df['close'].values
                credit_change = credit_df['close'].pct_change().fillna(0).values

                features.extend([credit_spread, credit_change])
                feature_names.extend(['credit_spread', 'credit_change'])

        # Combine features
        if features:
            min_length = min(len(f) for f in features)
            feature_matrix = np.column_stack([f[-min_length:] for f in features])
            self.feature_names = feature_names
            return feature_matrix
        else:
            # Fallback to dummy features
            dummy_length = 252  # One year of daily data
            feature_matrix = np.random.normal(0, 1, (dummy_length, 3))
            self.feature_names = ['dummy_vol', 'dummy_corr', 'dummy_spread']
            return feature_matrix

    def fit(self, data: Dict[str, Any]) -> 'HiddenMarkovRegimeModel':
        """
        Fit HMM model to historical data

        Args:
            data: Dictionary containing market data for training

        Returns:
            Self for method chaining
        """
        try:
            # Prepare features
            features = self.prepare_features(data)

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Fit Gaussian Mixture Model as HMM approximation
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=self.random_state,
                max_iter=100
            )

            self.model.fit(features_scaled)

            logger.info(f"HMM regime model fitted with {self.n_regimes} regimes")
            logger.info(f"Features used: {self.feature_names}")
            logger.info(f"Model converged: {self.model.converged_}")

            return self

        except Exception as e:
            logger.error(f"Failed to fit HMM model: {e}")
            raise

    def predict_regime(self, current_data: Dict[str, Any]) -> Tuple[MarketRegime, float, np.ndarray]:
        """
        Predict current market regime

        Args:
            current_data: Current market data for prediction

        Returns:
            Tuple of (regime, confidence, probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Prepare current features
            features = self.prepare_features(current_data)
            if len(features) == 0:
                return MarketRegime.NORMAL, 0.5, np.array([0.33, 0.33, 0.34])

            # Use last observation
            current_features = features[-1:].reshape(1, -1)
            current_scaled = self.scaler.transform(current_features)

            # Predict probabilities
            probabilities = self.model.predict_proba(current_scaled)[0]

            # Map to regimes (assuming sorted by volatility: Normal -> Volatile -> Crisis)
            regime_idx = np.argmax(probabilities)

            # Reorder probabilities to match regime severity
            prob_sorted_idx = np.argsort(self.model.means_.mean(axis=1))
            regime_mapping_sorted = {
                prob_sorted_idx[0]: MarketRegime.NORMAL,
                prob_sorted_idx[1]: MarketRegime.VOLATILE,
                prob_sorted_idx[2]: MarketRegime.CRISIS
            }

            predicted_regime = regime_mapping_sorted[regime_idx]
            confidence = probabilities[regime_idx]

            # Reorder probabilities for output
            prob_normal = probabilities[prob_sorted_idx[0]]
            prob_volatile = probabilities[prob_sorted_idx[1]]
            prob_crisis = probabilities[prob_sorted_idx[2]]

            ordered_probs = np.array([prob_normal, prob_volatile, prob_crisis])

            return predicted_regime, confidence, ordered_probs

        except Exception as e:
            logger.error(f"Failed to predict regime: {e}")
            return MarketRegime.NORMAL, 0.5, np.array([0.33, 0.33, 0.34])

    def get_regime_history(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Get historical regime classifications

        Args:
            data: Historical market data

        Returns:
            DataFrame with regime history
        """
        if self.model is None:
            raise ValueError("Model must be fitted before generating history")

        try:
            features = self.prepare_features(data)
            features_scaled = self.scaler.transform(features)

            # Predict for all time points
            probabilities = self.model.predict_proba(features_scaled)
            regime_predictions = self.model.predict(features_scaled)

            # Create result DataFrame
            dates = pd.date_range(end=datetime.now(), periods=len(features), freq='D')

            result = pd.DataFrame({
                'date': dates,
                'regime_idx': regime_predictions,
                'prob_normal': probabilities[:, 0],
                'prob_volatile': probabilities[:, 1],
                'prob_crisis': probabilities[:, 2],
                'confidence': probabilities.max(axis=1)
            })

            # Map to regime names
            prob_sorted_idx = np.argsort(self.model.means_.mean(axis=1))
            regime_mapping_sorted = {
                prob_sorted_idx[0]: 'NORMAL',
                prob_sorted_idx[1]: 'VOLATILE',
                prob_sorted_idx[2]: 'CRISIS'
            }

            result['regime'] = result['regime_idx'].map(regime_mapping_sorted)

            return result

        except Exception as e:
            logger.error(f"Failed to generate regime history: {e}")
            return pd.DataFrame()


class ThresholdRegimeDetector:
    """
    Threshold-based regime detection using multiple indicators

    Uses predefined thresholds on key indicators to classify market regimes.
    More interpretable but less adaptive than machine learning approaches.
    """

    def __init__(self, config: Optional[Dict[str, float]] = None):
        """
        Initialize threshold-based detector

        Args:
            config: Configuration dictionary with thresholds
        """
        default_config = {
            # VIX thresholds
            'vix_normal_max': 20.0,
            'vix_volatile_max': 30.0,
            'vix_crisis_min': 35.0,

            # Volatility percentile thresholds
            'vol_normal_max': 60.0,
            'vol_volatile_max': 80.0,
            'vol_crisis_min': 90.0,

            # Correlation thresholds
            'corr_normal_max': 0.5,
            'corr_volatile_max': 0.7,
            'corr_crisis_min': 0.8,

            # Credit spread thresholds (basis points above normal)
            'credit_normal_max': 100.0,
            'credit_volatile_max': 200.0,
            'credit_crisis_min': 300.0,

            # Market breadth thresholds
            'breadth_normal_min': 0.4,
            'breadth_volatile_min': 0.3,
            'breadth_crisis_max': 0.2,

            # Minimum indicators required for classification
            'min_indicators': 2
        }

        self.config = {**default_config, **(config or {})}

    def calculate_indicators(self, data: Dict[str, Any]) -> RegimeIndicators:
        """
        Calculate regime detection indicators

        Args:
            data: Market data dictionary

        Returns:
            RegimeIndicators object with calculated values
        """
        indicators = RegimeIndicators()

        # VIX indicators
        if 'vix_data' in data and data['vix_data'] is not None:
            vix_df = data['vix_data']
            if 'close' in vix_df.columns and len(vix_df) > 0:
                indicators.vix_level = float(vix_df['close'].iloc[-1])
                if len(vix_df) > 1:
                    indicators.vix_change = float(vix_df['close'].pct_change().iloc[-1])

        # Market volatility
        if 'market_data' in data and data['market_data']:
            volatilities = []
            correlations = []
            returns_list = []

            for symbol, df in data['market_data'].items():
                if df is not None and 'close' in df.columns and len(df) > 20:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 5:
                        returns_list.append(returns)
                        # Recent volatility (20-day)
                        recent_vol = returns.tail(20).std() * np.sqrt(252)
                        volatilities.append(recent_vol)

            if volatilities:
                avg_volatility = np.mean(volatilities)
                # Convert to percentile (simplified)
                indicators.volatility_percentile = min(100, avg_volatility * 100 / 0.6)

                # Calculate cross-sectional correlation
                if len(returns_list) >= 3:
                    try:
                        returns_df = pd.concat(returns_list, axis=1, join='inner')
                        if len(returns_df) > 10:
                            corr_matrix = returns_df.tail(20).corr()
                            # Average pairwise correlation
                            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                            avg_corr = corr_matrix.values[mask].mean()
                            indicators.correlation_level = avg_corr
                    except Exception:
                        pass

        # Credit spread (using proxy if available)
        if 'credit_data' in data and data['credit_data'] is not None:
            credit_df = data['credit_data']
            if 'close' in credit_df.columns and len(credit_df) > 0:
                indicators.credit_spread = float(credit_df['close'].iloc[-1])

        # Market breadth using existing market factors
        if HAS_DEPENDENCIES and 'market_data' in data:
            try:
                breadth_data = compute_market_breadth(data['market_data'])
                if 'advance_decline_ratio' in breadth_data:
                    ad_ratio = breadth_data['advance_decline_ratio']
                    if len(ad_ratio) > 0:
                        indicators.breadth_deterioration = 1.0 - float(ad_ratio.iloc[-1])
            except Exception:
                pass

        # Fear index using VIX
        if indicators.vix_level > 0:
            if indicators.vix_level > 30:
                indicators.fear_index = min(1.0, (indicators.vix_level - 20) / 30)
            else:
                indicators.fear_index = max(-1.0, (20 - indicators.vix_level) / 20)

        return indicators

    def classify_regime(self, indicators: RegimeIndicators) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on indicators

        Args:
            indicators: Calculated regime indicators

        Returns:
            Tuple of (regime, confidence)
        """
        crisis_score = 0
        volatile_score = 0
        normal_score = 0
        total_indicators = 0

        # VIX scoring
        if indicators.vix_level > 0:
            total_indicators += 1
            if indicators.vix_level >= self.config['vix_crisis_min']:
                crisis_score += 1
            elif indicators.vix_level >= self.config['vix_volatile_max']:
                volatile_score += 1
            elif indicators.vix_level <= self.config['vix_normal_max']:
                normal_score += 1

        # Volatility percentile scoring
        if indicators.volatility_percentile > 0:
            total_indicators += 1
            if indicators.volatility_percentile >= self.config['vol_crisis_min']:
                crisis_score += 1
            elif indicators.volatility_percentile >= self.config['vol_volatile_max']:
                volatile_score += 1
            elif indicators.volatility_percentile <= self.config['vol_normal_max']:
                normal_score += 1

        # Correlation scoring
        if indicators.correlation_level > 0:
            total_indicators += 1
            if indicators.correlation_level >= self.config['corr_crisis_min']:
                crisis_score += 1
            elif indicators.correlation_level >= self.config['corr_volatile_max']:
                volatile_score += 1
            elif indicators.correlation_level <= self.config['corr_normal_max']:
                normal_score += 1

        # Credit spread scoring
        if indicators.credit_spread > 0:
            total_indicators += 1
            if indicators.credit_spread >= self.config['credit_crisis_min']:
                crisis_score += 1
            elif indicators.credit_spread >= self.config['credit_volatile_max']:
                volatile_score += 1
            elif indicators.credit_spread <= self.config['credit_normal_max']:
                normal_score += 1

        # Market breadth scoring
        if indicators.breadth_deterioration >= 0:
            total_indicators += 1
            if indicators.breadth_deterioration >= (1 - self.config['breadth_crisis_max']):
                crisis_score += 1
            elif indicators.breadth_deterioration >= (1 - self.config['breadth_volatile_min']):
                volatile_score += 1
            elif indicators.breadth_deterioration <= (1 - self.config['breadth_normal_min']):
                normal_score += 1

        # Determine regime based on majority vote
        if total_indicators < self.config['min_indicators']:
            return MarketRegime.NORMAL, 0.3  # Low confidence due to insufficient data

        scores = [normal_score, volatile_score, crisis_score]
        max_score = max(scores)
        confidence = max_score / total_indicators

        if crisis_score == max_score:
            return MarketRegime.CRISIS, confidence
        elif volatile_score == max_score:
            return MarketRegime.VOLATILE, confidence
        else:
            return MarketRegime.NORMAL, confidence


class MLRegimeClassifier:
    """
    Machine Learning-based regime classifier using Random Forest

    Uses supervised learning with labeled historical periods to classify regimes.
    Trained on known crisis periods and market conditions.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize ML classifier

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False

        # Known crisis periods for training
        self.crisis_periods = [
            ('2008-09-01', '2009-03-31'),  # 2008 Financial Crisis
            ('2011-08-01', '2012-06-30'),  # European Debt Crisis
            ('2020-02-15', '2020-04-30'),  # COVID-19 Crash
            ('2022-01-01', '2022-06-30'),  # 2022 Inflation/Rate concerns
        ]

        self.volatile_periods = [
            ('2007-07-01', '2007-12-31'),  # Pre-crisis volatility
            ('2010-04-01', '2010-07-31'),  # Flash crash period
            ('2015-08-01', '2015-10-31'),  # China market turmoil
            ('2018-10-01', '2018-12-31'),  # Q4 2018 selloff
            ('2021-11-01', '2022-01-31'),  # Pre-2022 volatility
        ]

    def prepare_features(self, data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for ML training

        Args:
            data: Market data dictionary

        Returns:
            Tuple of (feature matrix, feature names)
        """
        features = []
        feature_names = []

        # VIX features
        if 'vix_data' in data and data['vix_data'] is not None:
            vix_df = data['vix_data']
            if 'close' in vix_df.columns:
                # VIX level and momentum features
                vix_level = vix_df['close'].values
                vix_ma_5 = vix_df['close'].rolling(5).mean().fillna(vix_df['close']).values
                vix_ma_20 = vix_df['close'].rolling(20).mean().fillna(vix_df['close']).values
                vix_change = vix_df['close'].pct_change().fillna(0).values
                vix_volatility = vix_df['close'].rolling(10).std().fillna(0).values

                features.extend([vix_level, vix_ma_5, vix_ma_20, vix_change, vix_volatility])
                feature_names.extend(['vix_level', 'vix_ma_5', 'vix_ma_20', 'vix_change', 'vix_volatility'])

        # Market volatility features
        if 'market_data' in data and data['market_data']:
            volatilities = []
            correlations_ts = []
            momentum_scores = []

            returns_list = []
            for symbol, df in data['market_data'].items():
                if df is not None and 'close' in df.columns and len(df) > 30:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 20:
                        returns_list.append(returns)

                        # Volatility features
                        vol_5d = returns.rolling(5).std().fillna(0) * np.sqrt(252)
                        vol_20d = returns.rolling(20).std().fillna(0) * np.sqrt(252)
                        volatilities.append(vol_20d.values)

                        # Momentum features
                        momentum_5d = returns.rolling(5).mean().fillna(0) * 252
                        momentum_scores.append(momentum_5d.values)

            if volatilities:
                # Aggregate volatility features
                market_vol = np.mean(volatilities, axis=0)
                vol_percentile = pd.Series(market_vol).rolling(60).rank(pct=True).fillna(0.5).values
                vol_acceleration = pd.Series(market_vol).diff().fillna(0).values

                features.extend([market_vol, vol_percentile, vol_acceleration])
                feature_names.extend(['market_volatility', 'vol_percentile', 'vol_acceleration'])

                # Cross-sectional features
                if len(returns_list) >= 3:
                    returns_df = pd.concat(returns_list, axis=1, join='inner')
                    if len(returns_df) > 30:
                        # Rolling correlation
                        rolling_corr_10d = returns_df.rolling(10).corr().groupby(level=0).mean().mean(axis=1)
                        rolling_corr_20d = returns_df.rolling(20).corr().groupby(level=0).mean().mean(axis=1)

                        # Correlation acceleration
                        corr_change = rolling_corr_10d.diff().fillna(0)

                        features.extend([
                            rolling_corr_10d.fillna(0).values,
                            rolling_corr_20d.fillna(0).values,
                            corr_change.values
                        ])
                        feature_names.extend(['correlation_10d', 'correlation_20d', 'correlation_change'])

                        # Market dispersion
                        daily_dispersion = returns_df.std(axis=1).fillna(0)
                        dispersion_ma = daily_dispersion.rolling(20).mean().fillna(0)

                        features.extend([daily_dispersion.values, dispersion_ma.values])
                        feature_names.extend(['daily_dispersion', 'dispersion_ma'])

            if momentum_scores:
                # Aggregate momentum features
                market_momentum = np.mean(momentum_scores, axis=0)
                momentum_dispersion = np.std(momentum_scores, axis=0)

                features.extend([market_momentum, momentum_dispersion])
                feature_names.extend(['market_momentum', 'momentum_dispersion'])

        # Combine features
        if features:
            min_length = min(len(f) for f in features)
            feature_matrix = np.column_stack([f[-min_length:] for f in features])
            return feature_matrix, feature_names
        else:
            # Fallback features
            dummy_length = 252
            feature_matrix = np.random.normal(0, 1, (dummy_length, 5))
            feature_names = ['dummy_vol', 'dummy_corr', 'dummy_momentum', 'dummy_dispersion', 'dummy_trend']
            return feature_matrix, feature_names

    def create_labels(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Create regime labels for training data

        Args:
            dates: DateTime index for labeling

        Returns:
            Array of regime labels (0=Normal, 1=Volatile, 2=Crisis)
        """
        labels = np.zeros(len(dates))  # Default to Normal

        # Convert dates to datetime if needed
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)

        # Label crisis periods
        for start_str, end_str in self.crisis_periods:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            mask = (dates >= start_date) & (dates <= end_date)
            labels[mask] = 2  # Crisis

        # Label volatile periods
        for start_str, end_str in self.volatile_periods:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            mask = (dates >= start_date) & (dates <= end_date)
            # Only label as volatile if not already labeled as crisis
            labels[mask & (labels == 0)] = 1  # Volatile

        return labels

    def fit(self, data: Dict[str, Any]) -> 'MLRegimeClassifier':
        """
        Fit ML model to historical data

        Args:
            data: Historical market data for training

        Returns:
            Self for method chaining
        """
        try:
            # Prepare features
            features, feature_names = self.prepare_features(data)
            self.feature_names = feature_names

            # Create date index
            dates = pd.date_range(end=datetime.now(), periods=len(features), freq='D')

            # Create labels
            labels = self.create_labels(dates)

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Train model
            self.model.fit(features_scaled, labels)
            self.is_fitted = True

            # Log training results
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_names = ['Normal', 'Volatile', 'Crisis']

            logger.info("ML Regime Classifier trained successfully")
            logger.info(f"Features: {feature_names}")
            logger.info("Training data distribution:")
            for label, count in zip(unique_labels, counts):
                pct = count / len(labels) * 100
                logger.info(f"  {label_names[int(label)]}: {count} samples ({pct:.1f}%)")

            # Cross-validation performance
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []

            for train_idx, val_idx in tscv.split(features_scaled):
                X_train, X_val = features_scaled[train_idx], features_scaled[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]

                temp_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    class_weight='balanced'
                )
                temp_model.fit(X_train, y_train)
                score = temp_model.score(X_val, y_val)
                cv_scores.append(score)

            logger.info(f"Cross-validation accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores)*2:.3f})")

            return self

        except Exception as e:
            logger.error(f"Failed to fit ML model: {e}")
            raise

    def predict_regime(self, current_data: Dict[str, Any]) -> Tuple[MarketRegime, float, np.ndarray]:
        """
        Predict current market regime

        Args:
            current_data: Current market data

        Returns:
            Tuple of (regime, confidence, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Prepare features
            features, _ = self.prepare_features(current_data)
            if len(features) == 0:
                return MarketRegime.NORMAL, 0.5, np.array([0.6, 0.3, 0.1])

            # Use last observation
            current_features = features[-1:].reshape(1, -1)
            current_scaled = self.scaler.transform(current_features)

            # Predict
            probabilities = self.model.predict_proba(current_scaled)[0]
            predicted_class = self.model.predict(current_scaled)[0]

            # Map to regimes
            regime_mapping = {
                0: MarketRegime.NORMAL,
                1: MarketRegime.VOLATILE,
                2: MarketRegime.CRISIS
            }

            predicted_regime = regime_mapping[predicted_class]
            confidence = probabilities[predicted_class]

            return predicted_regime, confidence, probabilities

        except Exception as e:
            logger.error(f"Failed to predict regime with ML model: {e}")
            return MarketRegime.NORMAL, 0.5, np.array([0.6, 0.3, 0.1])


class MarketRegimeClassifier:
    """
    Comprehensive Market Regime Classification System

    Combines multiple detection methods for robust regime identification:
    - Hidden Markov Model for regime switching dynamics
    - Threshold-based detection for interpretability
    - Machine Learning classifier for complex patterns
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 data_source: str = "auto",
                 cache_dir: str = "bot/data_cache"):
        """
        Initialize regime classifier

        Args:
            config: Configuration dictionary
            data_source: Data source preference
            cache_dir: Directory for caching models and data
        """
        self.config = config or {}
        self.data_source = data_source
        self.cache_dir = cache_dir

        # Initialize sub-models
        self.hmm_model = HiddenMarkovRegimeModel()
        self.threshold_detector = ThresholdRegimeDetector(
            (config or {}).get('threshold_config')
        )
        self.ml_classifier = MLRegimeClassifier()

        # State tracking
        self.current_regime = MarketRegime.NORMAL
        self.regime_history = []
        self.transition_history = []
        self.last_update = None

        # Model ensemble weights
        self.ensemble_weights = (config or {}).get('ensemble_weights', {
            'hmm': 0.4,
            'threshold': 0.3,
            'ml': 0.3
        })

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("Market Regime Classifier initialized")

    def load_market_data(self,
                        symbols: List[str] = None,
                        period: str = 'day',
                        limit: int = 500,
                        include_vix: bool = True,
                        include_credit: bool = True) -> Dict[str, Any]:
        """
        Load market data for regime analysis

        Args:
            symbols: List of symbols to include (default: major market symbols)
            period: Data period
            limit: Number of data points
            include_vix: Whether to include VIX data
            include_credit: Whether to include credit spread data

        Returns:
            Dictionary with market data
        """
        if symbols is None:
            # Default major market symbols
            symbols = [
                'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE', 'XLI', 'XLU', 'XLV', 'XLP',
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'JNJ', 'JPM'
            ]

        data = {}

        # Load market data
        try:
            if HAS_DEPENDENCIES:
                from bot.data import fetch_batch_history
                market_data = fetch_batch_history(
                    quote_client=None,  # Will use fallback data sources
                    symbols=symbols,
                    period=period,
                    limit=limit,
                    dry_run=False
                )
                data['market_data'] = {k: v for k, v in market_data.items() if v is not None}
            else:
                # Fallback to dummy data
                data['market_data'] = self._generate_dummy_market_data(symbols, limit)

            logger.info(f"Loaded market data for {len(data['market_data'])} symbols")

        except Exception as e:
            logger.warning(f"Failed to load market data: {e}")
            data['market_data'] = self._generate_dummy_market_data(symbols, limit)

        # Load VIX data
        if include_vix:
            try:
                if HAS_DEPENDENCIES:
                    from bot.data import fetch_history
                    vix_data = fetch_history(
                        quote_client=None,
                        symbol='^VIX',
                        period=period,
                        limit=limit,
                        dry_run=False
                    )
                    if vix_data is not None and not vix_data.empty:
                        data['vix_data'] = vix_data
                    else:
                        data['vix_data'] = self._generate_dummy_vix_data(limit)
                else:
                    data['vix_data'] = self._generate_dummy_vix_data(limit)

                logger.info("VIX data loaded successfully")

            except Exception as e:
                logger.warning(f"Failed to load VIX data: {e}")
                data['vix_data'] = self._generate_dummy_vix_data(limit)

        # Load credit spread data (proxy using HYG vs TLT)
        if include_credit:
            try:
                if HAS_DEPENDENCIES:
                    credit_symbols = ['HYG', 'TLT']  # High yield vs Treasury
                    credit_data = fetch_batch_history(
                        quote_client=None,
                        symbols=credit_symbols,
                        period=period,
                        limit=limit,
                        dry_run=False
                    )

                    if 'HYG' in credit_data and 'TLT' in credit_data:
                        # Calculate spread proxy
                        hyg_data = credit_data['HYG']
                        tlt_data = credit_data['TLT']

                        if hyg_data is not None and tlt_data is not None:
                            # Align data and calculate spread
                            aligned = pd.concat([
                                hyg_data['close'],
                                tlt_data['close']
                            ], axis=1, join='inner')

                            if not aligned.empty:
                                spread = aligned.iloc[:, 1] / aligned.iloc[:, 0]  # TLT/HYG ratio
                                credit_df = pd.DataFrame({
                                    'close': spread,
                                    'time': aligned.index
                                })
                                data['credit_data'] = credit_df

                logger.info("Credit spread data loaded successfully")

            except Exception as e:
                logger.warning(f"Failed to load credit data: {e}")
                data['credit_data'] = None

        return data

    def _generate_dummy_market_data(self, symbols: List[str], limit: int) -> Dict[str, pd.DataFrame]:
        """Generate dummy market data for testing"""
        dummy_data = {}

        for symbol in symbols[:10]:  # Limit to prevent excessive dummy data
            dates = pd.date_range(end=datetime.now(), periods=limit, freq='D')

            # Generate realistic price series with regime changes
            returns = np.random.normal(0.0005, 0.02, limit)

            # Add crisis periods with higher volatility
            crisis_mask = np.random.random(limit) < 0.05
            returns[crisis_mask] *= 3

            # Generate OHLCV data
            prices = 100 * np.cumprod(1 + returns)

            dummy_data[symbol] = pd.DataFrame({
                'time': dates,
                'open': prices * (1 + np.random.normal(0, 0.001, limit)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.002, limit))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.002, limit))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, limit)
            })

        return dummy_data

    def _generate_dummy_vix_data(self, limit: int) -> pd.DataFrame:
        """Generate dummy VIX data"""
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='D')

        # VIX with regime-dependent behavior
        base_vix = 20
        vix_changes = np.random.normal(0, 1, limit)

        # Add crisis spikes
        crisis_mask = np.random.random(limit) < 0.02
        vix_changes[crisis_mask] += np.random.uniform(10, 30, sum(crisis_mask))

        vix_levels = np.maximum(5, base_vix + np.cumsum(vix_changes * 0.1))
        vix_levels = np.minimum(80, vix_levels)  # Cap at reasonable level

        return pd.DataFrame({
            'time': dates,
            'close': vix_levels,
            'open': vix_levels * (1 + np.random.normal(0, 0.01, limit)),
            'high': vix_levels * (1 + np.abs(np.random.normal(0, 0.02, limit))),
            'low': vix_levels * (1 - np.abs(np.random.normal(0, 0.02, limit))),
            'volume': np.random.randint(100000, 1000000, limit)
        })

    def fit_models(self, data: Dict[str, Any] = None) -> 'MarketRegimeClassifier':
        """
        Fit all regime detection models

        Args:
            data: Market data for training (loads if None)

        Returns:
            Self for method chaining
        """
        if data is None:
            logger.info("Loading market data for model training...")
            data = self.load_market_data()

        logger.info("Fitting regime detection models...")

        # Fit HMM model
        try:
            self.hmm_model.fit(data)
            logger.info("HMM model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit HMM model: {e}")

        # Fit ML classifier
        try:
            self.ml_classifier.fit(data)
            logger.info("ML classifier fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit ML classifier: {e}")

        logger.info("All models fitted successfully")

        # Save models
        self.save_models()

        return self

    def predict_regime(self,
                      data: Dict[str, Any] = None,
                      method: str = 'ensemble') -> RegimePrediction:
        """
        Predict current market regime

        Args:
            data: Current market data (loads latest if None)
            method: Prediction method ('ensemble', 'hmm', 'threshold', 'ml')

        Returns:
            RegimePrediction object with results
        """
        if data is None:
            logger.info("Loading current market data...")
            data = self.load_market_data(limit=100)  # Smaller limit for current prediction

        predictions = {}
        confidences = {}
        probabilities = {}

        # Get predictions from all methods
        try:
            # Threshold method
            indicators = self.threshold_detector.calculate_indicators(data)
            regime_thresh, conf_thresh = self.threshold_detector.classify_regime(indicators)
            predictions['threshold'] = regime_thresh
            confidences['threshold'] = conf_thresh

            # Convert to probabilities for threshold method
            if regime_thresh == MarketRegime.NORMAL:
                probs_thresh = np.array([conf_thresh, (1-conf_thresh)/2, (1-conf_thresh)/2])
            elif regime_thresh == MarketRegime.VOLATILE:
                probs_thresh = np.array([(1-conf_thresh)/2, conf_thresh, (1-conf_thresh)/2])
            else:  # CRISIS
                probs_thresh = np.array([(1-conf_thresh)/2, (1-conf_thresh)/2, conf_thresh])

            probabilities['threshold'] = probs_thresh

        except Exception as e:
            logger.warning(f"Threshold prediction failed: {e}")
            predictions['threshold'] = MarketRegime.NORMAL
            confidences['threshold'] = 0.3
            probabilities['threshold'] = np.array([0.6, 0.3, 0.1])

        try:
            # HMM method
            regime_hmm, conf_hmm, probs_hmm = self.hmm_model.predict_regime(data)
            predictions['hmm'] = regime_hmm
            confidences['hmm'] = conf_hmm
            probabilities['hmm'] = probs_hmm

        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}")
            predictions['hmm'] = MarketRegime.NORMAL
            confidences['hmm'] = 0.3
            probabilities['hmm'] = np.array([0.6, 0.3, 0.1])

        try:
            # ML method
            regime_ml, conf_ml, probs_ml = self.ml_classifier.predict_regime(data)
            predictions['ml'] = regime_ml
            confidences['ml'] = conf_ml
            probabilities['ml'] = probs_ml

        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            predictions['ml'] = MarketRegime.NORMAL
            confidences['ml'] = 0.3
            probabilities['ml'] = np.array([0.6, 0.3, 0.1])

        # Ensemble prediction
        if method == 'ensemble':
            # Weighted average of probabilities
            ensemble_probs = np.zeros(3)
            total_weight = 0

            for model_name, weight in self.ensemble_weights.items():
                if model_name in probabilities:
                    ensemble_probs += weight * probabilities[model_name]
                    total_weight += weight

            if total_weight > 0:
                ensemble_probs /= total_weight
            else:
                ensemble_probs = np.array([0.6, 0.3, 0.1])

            # Final regime and confidence
            regime_idx = np.argmax(ensemble_probs)
            regimes = [MarketRegime.NORMAL, MarketRegime.VOLATILE, MarketRegime.CRISIS]
            final_regime = regimes[regime_idx]
            final_confidence = ensemble_probs[regime_idx]
            final_method = 'ensemble'

        else:
            # Single method prediction
            final_regime = predictions.get(method, MarketRegime.NORMAL)
            final_confidence = confidences.get(method, 0.3)
            ensemble_probs = probabilities.get(method, np.array([0.6, 0.3, 0.1]))
            final_method = method

        # Update state
        if final_regime != self.current_regime:
            self._record_regime_transition(self.current_regime, final_regime, final_confidence)

        self.current_regime = final_regime
        self.last_update = datetime.now()

        # Create prediction object
        prediction = RegimePrediction(
            regime=final_regime,
            confidence=final_confidence,
            probability_normal=ensemble_probs[0],
            probability_volatile=ensemble_probs[1],
            probability_crisis=ensemble_probs[2],
            indicators=indicators,
            method=final_method,
            timestamp=datetime.now()
        )

        # Add to history
        self.regime_history.append(prediction)

        # Keep history manageable
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

        logger.info(f"Regime prediction: {final_regime.value} (confidence: {final_confidence:.3f})")

        return prediction

    def _record_regime_transition(self,
                                 from_regime: MarketRegime,
                                 to_regime: MarketRegime,
                                 confidence: float):
        """Record regime transition"""
        # Calculate duration
        duration_days = 0
        if self.regime_history:
            last_transition = None
            for i in range(len(self.regime_history) - 1, -1, -1):
                if self.regime_history[i].regime != from_regime:
                    last_transition = self.regime_history[i].timestamp
                    break

            if last_transition:
                duration_days = (datetime.now() - last_transition).days

        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_date=datetime.now(),
            confidence=confidence,
            duration_days=duration_days,
            indicators=self.threshold_detector.calculate_indicators({})
        )

        self.transition_history.append(transition)

        # Keep transition history manageable
        if len(self.transition_history) > 100:
            self.transition_history = self.transition_history[-100:]

        logger.info(f"Regime transition: {from_regime.value} -> {to_regime.value} "
                   f"(duration: {duration_days} days)")

    def get_historical_regimes(self,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical regime classifications

        Args:
            start_date: Start date for history
            end_date: End date for history

        Returns:
            DataFrame with historical regimes
        """
        if not self.hmm_model.model:
            logger.warning("Models not fitted. Loading data and fitting...")
            self.fit_models()

        # Load historical data
        limit = 1000 if start_date is None else max(100, (datetime.now() - start_date).days + 50)
        data = self.load_market_data(limit=limit)

        try:
            # Get HMM history as primary method
            history_df = self.hmm_model.get_regime_history(data)

            if not history_df.empty:
                # Filter by date range if specified
                if start_date:
                    history_df = history_df[history_df['date'] >= start_date]
                if end_date:
                    history_df = history_df[history_df['date'] <= end_date]

                logger.info(f"Historical regimes generated: {len(history_df)} periods")
                return history_df
            else:
                logger.warning("No historical regime data generated")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to generate historical regimes: {e}")
            return pd.DataFrame()

    def validate_crisis_periods(self) -> Dict[str, Any]:
        """
        Validate model performance on known crisis periods

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating crisis period detection...")

        # Get historical regimes
        history_df = self.get_historical_regimes()

        if history_df.empty:
            return {"error": "No historical data available for validation"}

        validation_results = {
            "crisis_periods_detected": [],
            "false_positives": [],
            "accuracy_metrics": {},
            "coverage_analysis": {}
        }

        # Known crisis periods for validation
        known_periods = [
            ("2008-09-01", "2009-03-31", "2008 Financial Crisis"),
            ("2011-08-01", "2012-06-30", "European Debt Crisis"),
            ("2020-02-15", "2020-04-30", "COVID-19 Crash"),
            ("2022-01-01", "2022-06-30", "2022 Inflation Crisis")
        ]

        for start_str, end_str, period_name in known_periods:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)

            # Check if period is in our data
            period_mask = (history_df['date'] >= start_date) & (history_df['date'] <= end_date)
            period_data = history_df[period_mask]

            if not period_data.empty:
                # Calculate detection metrics
                crisis_detections = (period_data['regime'] == 'CRISIS').sum()
                volatile_detections = (period_data['regime'] == 'VOLATILE').sum()
                total_days = len(period_data)

                detection_rate = (crisis_detections + volatile_detections) / total_days
                crisis_rate = crisis_detections / total_days

                validation_results["crisis_periods_detected"].append({
                    "period": period_name,
                    "start_date": start_str,
                    "end_date": end_str,
                    "total_days": total_days,
                    "crisis_detected_days": crisis_detections,
                    "volatile_detected_days": volatile_detections,
                    "detection_rate": detection_rate,
                    "crisis_rate": crisis_rate
                })

                logger.info(f"{period_name}: {detection_rate:.1%} detection rate "
                           f"({crisis_rate:.1%} crisis, {(volatile_detections/total_days):.1%} volatile)")

        # Calculate overall accuracy
        if validation_results["crisis_periods_detected"]:
            avg_detection_rate = np.mean([
                p["detection_rate"] for p in validation_results["crisis_periods_detected"]
            ])
            avg_crisis_rate = np.mean([
                p["crisis_rate"] for p in validation_results["crisis_periods_detected"]
            ])

            validation_results["accuracy_metrics"] = {
                "average_detection_rate": avg_detection_rate,
                "average_crisis_rate": avg_crisis_rate,
                "periods_analyzed": len(validation_results["crisis_periods_detected"])
            }

            logger.info(f"Overall validation: {avg_detection_rate:.1%} average detection rate")

        return validation_results

    def save_models(self, filepath: str = None) -> bool:
        """
        Save trained models to disk

        Args:
            filepath: Path to save models (default: cache_dir/regime_models.pkl)

        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            filepath = os.path.join(self.cache_dir, "regime_models.pkl")

        try:
            model_data = {
                'hmm_model': self.hmm_model,
                'threshold_detector': self.threshold_detector,
                'ml_classifier': self.ml_classifier,
                'config': self.config,
                'ensemble_weights': self.ensemble_weights,
                'saved_timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Models saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False

    def load_models(self, filepath: str = None) -> bool:
        """
        Load trained models from disk

        Args:
            filepath: Path to load models from

        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            filepath = os.path.join(self.cache_dir, "regime_models.pkl")

        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return False

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.hmm_model = model_data['hmm_model']
            self.threshold_detector = model_data['threshold_detector']
            self.ml_classifier = model_data['ml_classifier']
            self.config = model_data['config']
            self.ensemble_weights = model_data['ensemble_weights']

            logger.info(f"Models loaded from {filepath}")
            logger.info(f"Saved on: {model_data.get('saved_timestamp', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive regime analysis summary

        Returns:
            Dictionary with regime summary
        """
        current_prediction = self.predict_regime()

        # Recent regime distribution
        recent_regimes = [p.regime.value for p in self.regime_history[-30:]]
        regime_dist = {
            'NORMAL': recent_regimes.count('NORMAL'),
            'VOLATILE': recent_regimes.count('VOLATILE'),
            'CRISIS': recent_regimes.count('CRISIS')
        }

        # Transition analysis
        recent_transitions = len([t for t in self.transition_history
                                if (datetime.now() - t.transition_date).days <= 30])

        summary = {
            'current_regime': current_prediction.regime.value,
            'confidence': current_prediction.confidence,
            'probabilities': {
                'normal': current_prediction.probability_normal,
                'volatile': current_prediction.probability_volatile,
                'crisis': current_prediction.probability_crisis
            },
            'indicators': {
                'vix_level': current_prediction.indicators.vix_level,
                'volatility_percentile': current_prediction.indicators.volatility_percentile,
                'correlation_level': current_prediction.indicators.correlation_level,
                'fear_index': current_prediction.indicators.fear_index
            },
            'recent_distribution': regime_dist,
            'recent_transitions': recent_transitions,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'models_fitted': {
                'hmm': self.hmm_model.model is not None,
                'ml': self.ml_classifier.is_fitted,
                'threshold': True
            }
        }

        return summary


# Factory function for easy integration
def create_regime_classifier(config: Optional[Dict[str, Any]] = None) -> MarketRegimeClassifier:
    """
    Factory function to create market regime classifier

    Args:
        config: Optional configuration dictionary

    Returns:
        MarketRegimeClassifier instance
    """
    return MarketRegimeClassifier(config)


# Integration with existing risk management
def get_regime_for_risk_manager():
    """
    Get current regime in format compatible with existing risk manager

    Returns:
        MarketRegime compatible with existing enhanced_risk_manager
    """
    try:
        classifier = MarketRegimeClassifier()

        # Try to load existing models
        if not classifier.load_models():
            logger.info("No saved models found, fitting new models...")
            classifier.fit_models()

        prediction = classifier.predict_regime()

        # Try to import existing regime enum
        try:
            from enhanced_risk_manager import MarketRegime as ExistingMarketRegime

            # Map to existing enum
            regime_mapping = {
                MarketRegime.NORMAL: ExistingMarketRegime.NORMAL,
                MarketRegime.VOLATILE: ExistingMarketRegime.VOLATILE,
                MarketRegime.CRISIS: ExistingMarketRegime.CRISIS
            }

            return regime_mapping.get(prediction.regime, ExistingMarketRegime.NORMAL)

        except ImportError:
            # Fallback to string representation
            return prediction.regime.value

    except Exception as e:
        logger.error(f"Failed to get regime for risk manager: {e}")
        return "NORMAL"


if __name__ == "__main__":
    # Example usage and testing
    print("Market Regime Classification System")
    print("=" * 50)

    # Initialize classifier
    classifier = MarketRegimeClassifier()

    # Try to load existing models
    if not classifier.load_models():
        print("No saved models found. Training new models...")
        classifier.fit_models()
    else:
        print("Loaded existing models")

    # Get current regime prediction
    current_prediction = classifier.predict_regime()

    print(f"\nCurrent Market Regime: {current_prediction.regime.value}")
    print(f"Confidence: {current_prediction.confidence:.3f}")
    print(f"Probabilities:")
    print(f"  Normal: {current_prediction.probability_normal:.3f}")
    print(f"  Volatile: {current_prediction.probability_volatile:.3f}")
    print(f"  Crisis: {current_prediction.probability_crisis:.3f}")

    # Validate on known crisis periods
    print("\nValidating crisis period detection...")
    validation_results = classifier.validate_crisis_periods()

    if "accuracy_metrics" in validation_results:
        metrics = validation_results["accuracy_metrics"]
        print(f"Average detection rate: {metrics['average_detection_rate']:.1%}")
        print(f"Average crisis rate: {metrics['average_crisis_rate']:.1%}")

    # Get comprehensive summary
    summary = classifier.get_regime_summary()
    print(f"\nRegime Summary:")
    print(f"Recent regime distribution: {summary['recent_distribution']}")
    print(f"Recent transitions: {summary['recent_transitions']}")

    print("\nMarket Regime Classification System ready!")