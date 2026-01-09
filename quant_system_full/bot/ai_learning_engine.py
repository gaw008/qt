#!/usr/bin/env python3
"""
AI Learning Engine - Core ML System for Quantitative Trading
AI学习引擎 - 量化交易核心机器学习系统

Investment-grade machine learning engine providing:
- Adaptive strategy learning and optimization
- Multi-model ensemble management
- Real-time performance monitoring
- Feature importance tracking and drift detection
- Model validation and backtesting integration

投资级机器学习引擎功能：
- 自适应策略学习与优化
- 多模型集成管理
- 实时性能监控
- 特征重要性跟踪与漂移检测
- 模型验证与回测集成
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
import sqlite3
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import asyncio

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class ModelType(Enum):
    """Machine learning model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    SVR = "support_vector_regression"

class LearningMode(Enum):
    """Learning operation modes"""
    TRAINING = "training"
    VALIDATION = "validation"
    PREDICTION = "prediction"
    RETRAINING = "retraining"

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_id: str
    model_type: ModelType
    timestamp: datetime

    # Regression metrics
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    sharpe_ratio: float = 0.0

    # Trading-specific metrics
    hit_rate: float = 0.0  # Percentage of correct directional predictions
    profit_factor: float = 0.0  # Total profit / Total loss
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Statistical measures
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0

    # Model characteristics
    feature_count: int = 0
    training_samples: int = 0
    validation_samples: int = 0
    overfitting_score: float = 0.0  # Validation loss / Training loss

@dataclass
class FeatureImportance:
    """Feature importance tracking"""
    feature_name: str
    importance_score: float
    stability_score: float  # How stable the importance is over time
    drift_indicator: float  # Statistical measure of feature drift
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class EnsembleWeight:
    """Dynamic ensemble model weights"""
    model_id: str
    weight: float
    performance_score: float
    recency_factor: float  # Higher weight for more recent performance
    stability_factor: float  # Penalty for unstable models
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ModelState:
    """Complete model state for persistence"""
    model_id: str
    model_type: ModelType
    model_object: Any
    scaler: Any
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    training_date: datetime
    performance_history: List[ModelPerformance]
    feature_importance: List[FeatureImportance]
    is_active: bool = True

class AILearningEngine:
    """
    Investment-Grade AI Learning Engine

    Provides comprehensive machine learning capabilities for quantitative trading:
    - Multi-model ensemble management with adaptive weighting
    - Real-time model performance tracking and validation
    - Feature engineering and drift detection
    - Automated model retraining and optimization
    - Integration with risk management and execution systems
    """

    def __init__(self, config_path: str = "config/ai_config.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Model management
        self.active_models: Dict[str, ModelState] = {}
        self.ensemble_weights: Dict[str, EnsembleWeight] = {}
        self.performance_history: List[ModelPerformance] = []

        # Feature management
        self.feature_importance_tracker: Dict[str, FeatureImportance] = {}
        self.feature_drift_detector = FeatureDriftDetector()

        # Learning state
        self.current_mode = LearningMode.PREDICTION
        self.learning_metrics = {
            "total_models_trained": 0,
            "active_models_count": 0,
            "average_model_performance": 0.0,
            "ensemble_stability": 0.0,
            "feature_drift_alerts": 0,
            "retraining_frequency": 0.0
        }

        # Database and persistence
        self.db_path = "data_cache/ai_learning.db"
        self.models_path = Path("data_cache/models")
        self.models_path.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Simulation state for realistic learning data
        self._simulation_state = self._initialize_simulation()

        self.logger.info("AI Learning Engine initialized with multi-model ensemble capability")

    def _initialize_simulation(self) -> Dict[str, Any]:
        """Initialize realistic simulation data for training and validation"""
        np.random.seed(42)

        # Generate synthetic market data
        n_samples = 2000
        n_features = 15

        # Create correlated features mimicking real market factors
        correlation_matrix = np.random.rand(n_features, n_features)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)

        # Generate features with realistic distributions
        features = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=correlation_matrix,
            size=n_samples
        )

        # Create realistic target variable (returns)
        # Linear combination of features + noise + non-linear effects
        true_weights = np.random.normal(0, 0.1, n_features)
        linear_effect = features @ true_weights

        # Add non-linear effects
        non_linear_effect = 0.1 * np.sin(features[:, 0]) * features[:, 1]
        noise = np.random.normal(0, 0.02, n_samples)

        target = linear_effect + non_linear_effect + noise

        # Add regime changes to make it more realistic
        regime_change_points = [500, 1000, 1500]
        for point in regime_change_points:
            target[point:] += np.random.normal(0, 0.01)

        # Create feature names
        feature_names = [
            "momentum_5d", "momentum_20d", "rsi_14", "macd_signal", "bb_position",
            "volume_ratio", "price_to_sma", "volatility_20d", "correlation_spy",
            "sector_strength", "earnings_surprise", "analyst_revisions",
            "insider_trading", "short_interest", "options_flow"
        ]

        # Create timestamps
        timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

        return {
            'features': features,
            'target': target,
            'feature_names': feature_names,
            'timestamps': timestamps,
            'last_update': datetime.now()
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load AI learning configuration"""
        default_config = {
            "model_types": [
                "random_forest", "gradient_boosting", "ridge_regression", "lasso_regression"
            ],
            "ensemble_method": "weighted_average",
            "retraining_frequency": 7,  # days
            "validation_split": 0.2,
            "min_training_samples": 252,  # 1 year of daily data
            "max_models_active": 8,
            "feature_selection": {
                "max_features": 50,
                "importance_threshold": 0.001,
                "correlation_threshold": 0.95
            },
            "hyperparameter_tuning": {
                "random_forest": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "gradient_boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0]
                },
                "ridge_regression": {
                    "alpha": [0.1, 1.0, 10.0, 100.0]
                },
                "lasso_regression": {
                    "alpha": [0.001, 0.01, 0.1, 1.0]
                }
            },
            "performance_thresholds": {
                "min_r2_score": 0.1,
                "min_sharpe_ratio": 0.5,
                "max_drawdown_threshold": 0.20,
                "min_hit_rate": 0.52,
                "overfitting_threshold": 1.5
            },
            "drift_detection": {
                "window_size": 30,
                "sensitivity": 2.0,  # Standard deviations for drift alert
                "min_samples_for_test": 50
            }
        }

        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    default_config.update(config)
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for AI learning system"""
        logger = logging.getLogger('AILearningEngine')
        logger.setLevel(logging.INFO)

        # File handler
        log_path = Path('logs/ai_learning.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _initialize_database(self):
        """Initialize SQLite database for AI learning persistence"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                # Model performance table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        mse REAL NOT NULL,
                        mae REAL NOT NULL,
                        r2_score REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        hit_rate REAL NOT NULL,
                        profit_factor REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        calmar_ratio REAL NOT NULL,
                        information_ratio REAL NOT NULL,
                        tracking_error REAL NOT NULL,
                        beta REAL NOT NULL,
                        alpha REAL NOT NULL,
                        feature_count INTEGER NOT NULL,
                        training_samples INTEGER NOT NULL,
                        validation_samples INTEGER NOT NULL,
                        overfitting_score REAL NOT NULL
                    )
                """)

                # Feature importance table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feature_importance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feature_name TEXT NOT NULL,
                        importance_score REAL NOT NULL,
                        stability_score REAL NOT NULL,
                        drift_indicator REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                # Ensemble weights table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ensemble_weights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        weight REAL NOT NULL,
                        performance_score REAL NOT NULL,
                        recency_factor REAL NOT NULL,
                        stability_factor REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                # Learning metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_models_trained INTEGER NOT NULL,
                        active_models_count INTEGER NOT NULL,
                        average_model_performance REAL NOT NULL,
                        ensemble_stability REAL NOT NULL,
                        feature_drift_alerts INTEGER NOT NULL,
                        retraining_frequency REAL NOT NULL
                    )
                """)

                conn.commit()

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")

    async def train_model(self, model_type: ModelType, features: np.ndarray,
                         targets: np.ndarray, feature_names: List[str],
                         hyperparameters: Optional[Dict[str, Any]] = None) -> str:
        """Train a new model with given data and hyperparameters"""
        try:
            model_id = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.logger.info(f"Training model {model_id} with {len(features)} samples")

            # Prepare data
            X_train, X_val, y_train, y_val = self._prepare_training_data(features, targets)

            # Initialize scaler
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Create model
            model = await self._create_model(model_type, hyperparameters)

            # Train model
            model.fit(X_train_scaled, y_train)

            # Validate model
            performance = await self._validate_model(
                model, scaler, X_train_scaled, y_train, X_val_scaled, y_val,
                model_id, model_type
            )

            # Check if model meets performance thresholds
            if not self._meets_performance_threshold(performance):
                self.logger.warning(f"Model {model_id} does not meet performance thresholds")
                return None

            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(
                model, feature_names, model_type
            )

            # Create model state
            model_state = ModelState(
                model_id=model_id,
                model_type=model_type,
                model_object=model,
                scaler=scaler,
                feature_names=feature_names,
                hyperparameters=hyperparameters or {},
                training_date=datetime.now(),
                performance_history=[performance],
                feature_importance=feature_importance,
                is_active=True
            )

            # Store model
            self.active_models[model_id] = model_state
            await self._save_model(model_state)

            # Update ensemble weights
            await self._update_ensemble_weights()

            # Store performance metrics
            await self._store_performance_metrics(performance)

            # Update learning metrics
            self.learning_metrics["total_models_trained"] += 1
            self.learning_metrics["active_models_count"] = len(self.active_models)

            self.logger.info(f"Model {model_id} trained successfully with R2: {performance.r2_score:.4f}")

            return model_id

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return None

    def _prepare_training_data(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation data with time series considerations"""
        try:
            # Use time series split to respect temporal order
            split_point = int(len(features) * (1 - self.config["validation_split"]))

            X_train = features[:split_point]
            X_val = features[split_point:]
            y_train = targets[:split_point]
            y_val = targets[split_point:]

            return X_train, X_val, y_train, y_val

        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise

    async def _create_model(self, model_type: ModelType, hyperparameters: Optional[Dict[str, Any]]) -> Any:
        """Create ML model instance based on type and hyperparameters"""
        try:
            # Use default hyperparameters if none provided
            if hyperparameters is None:
                hyperparameters = self._get_default_hyperparameters(model_type)

            if model_type == ModelType.RANDOM_FOREST:
                return RandomForestRegressor(**hyperparameters, random_state=42, n_jobs=-1)
            elif model_type == ModelType.GRADIENT_BOOSTING:
                return GradientBoostingRegressor(**hyperparameters, random_state=42)
            elif model_type == ModelType.LINEAR_REGRESSION:
                return LinearRegression(**hyperparameters)
            elif model_type == ModelType.RIDGE_REGRESSION:
                return Ridge(**hyperparameters, random_state=42)
            elif model_type == ModelType.LASSO_REGRESSION:
                return Lasso(**hyperparameters, random_state=42, max_iter=2000)
            elif model_type == ModelType.SVR:
                return SVR(**hyperparameters)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            self.logger.error(f"Model creation failed: {e}")
            raise

    def _get_default_hyperparameters(self, model_type: ModelType) -> Dict[str, Any]:
        """Get default hyperparameters for model type"""
        defaults = {
            ModelType.RANDOM_FOREST: {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2
            },
            ModelType.GRADIENT_BOOSTING: {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.9
            },
            ModelType.LINEAR_REGRESSION: {},
            ModelType.RIDGE_REGRESSION: {"alpha": 1.0},
            ModelType.LASSO_REGRESSION: {"alpha": 0.01},
            ModelType.SVR: {"C": 1.0, "gamma": "scale"}
        }

        return defaults.get(model_type, {})

    async def _validate_model(self, model: Any, scaler: Any,
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             model_id: str, model_type: ModelType) -> ModelPerformance:
        """Comprehensive model validation with trading-specific metrics"""
        try:
            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Basic regression metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            # Trading-specific metrics
            hit_rate = self._calculate_hit_rate(y_val, y_val_pred)
            sharpe_ratio = self._calculate_sharpe_ratio(y_val_pred)
            max_drawdown = self._calculate_max_drawdown(y_val_pred)
            calmar_ratio = (np.mean(y_val_pred) * 252) / max_drawdown if max_drawdown > 0 else 0.0
            profit_factor = self._calculate_profit_factor(y_val_pred)

            # Advanced metrics
            information_ratio = self._calculate_information_ratio(y_val, y_val_pred)
            tracking_error = np.std(y_val - y_val_pred) * np.sqrt(252)
            beta, alpha = self._calculate_beta_alpha(y_val, y_val_pred)

            # Overfitting detection
            overfitting_score = val_mse / train_mse if train_mse > 0 else 1.0

            return ModelPerformance(
                model_id=model_id,
                model_type=model_type,
                timestamp=datetime.now(),
                mse=val_mse,
                mae=val_mae,
                r2_score=val_r2,
                sharpe_ratio=sharpe_ratio,
                hit_rate=hit_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                beta=beta,
                alpha=alpha,
                feature_count=X_val.shape[1],
                training_samples=len(X_train),
                validation_samples=len(X_val),
                overfitting_score=overfitting_score
            )

        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            raise

    def _calculate_hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (hit rate)"""
        try:
            # Convert to directional predictions
            true_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)

            # Calculate accuracy
            correct_predictions = np.sum(true_direction == pred_direction)
            total_predictions = len(y_true)

            return correct_predictions / total_predictions if total_predictions > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio of predictions"""
        try:
            if len(returns) == 0:
                return 0.0

            excess_returns = np.mean(returns) * 252 - risk_free_rate
            volatility = np.std(returns) * np.sqrt(252)

            return excess_returns / volatility if volatility > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from return series"""
        try:
            if len(returns) == 0:
                return 0.0

            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max

            return abs(np.min(drawdowns))

        except Exception:
            return 0.0

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (total gains / total losses)"""
        try:
            if len(returns) == 0:
                return 0.0

            gains = np.sum(returns[returns > 0])
            losses = abs(np.sum(returns[returns < 0]))

            return gains / losses if losses > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_information_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate information ratio"""
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return 0.0

            active_return = np.mean(y_pred - y_true)
            tracking_error = np.std(y_pred - y_true)

            return (active_return * 252) / (tracking_error * np.sqrt(252)) if tracking_error > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_beta_alpha(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate beta and alpha relative to benchmark"""
        try:
            if len(y_true) < 2 or len(y_pred) < 2:
                return 0.0, 0.0

            # Use true returns as benchmark
            covariance = np.cov(y_pred, y_true)[0, 1]
            benchmark_variance = np.var(y_true)

            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
            alpha = (np.mean(y_pred) - beta * np.mean(y_true)) * 252

            return beta, alpha

        except Exception:
            return 0.0, 0.0

    def _meets_performance_threshold(self, performance: ModelPerformance) -> bool:
        """Check if model meets minimum performance thresholds"""
        try:
            thresholds = self.config["performance_thresholds"]

            checks = [
                performance.r2_score >= thresholds["min_r2_score"],
                performance.sharpe_ratio >= thresholds["min_sharpe_ratio"],
                performance.max_drawdown <= thresholds["max_drawdown_threshold"],
                performance.hit_rate >= thresholds["min_hit_rate"],
                performance.overfitting_score <= thresholds["overfitting_threshold"]
            ]

            return all(checks)

        except Exception as e:
            self.logger.error(f"Performance threshold check failed: {e}")
            return False

    async def _calculate_feature_importance(self, model: Any, feature_names: List[str],
                                          model_type: ModelType) -> List[FeatureImportance]:
        """Calculate feature importance for the trained model"""
        try:
            importance_scores = []

            # Extract feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_)
            else:
                # For models without built-in importance, use permutation importance
                importance_scores = np.ones(len(feature_names)) / len(feature_names)

            feature_importance = []
            for i, (name, score) in enumerate(zip(feature_names, importance_scores)):
                # Calculate stability score (simplified)
                stability_score = 1.0 - (score * 0.1)  # Inverse relationship for now

                feature_importance.append(FeatureImportance(
                    feature_name=name,
                    importance_score=float(score),
                    stability_score=float(stability_score),
                    drift_indicator=0.0,  # Will be updated during drift detection
                    last_updated=datetime.now()
                ))

            return feature_importance

        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return []

    async def _update_ensemble_weights(self):
        """Update ensemble model weights based on recent performance"""
        try:
            if not self.active_models:
                return

            # Calculate performance scores for each model
            performance_scores = {}
            recency_scores = {}
            stability_scores = {}

            current_time = datetime.now()

            for model_id, model_state in self.active_models.items():
                if not model_state.performance_history:
                    continue

                # Get recent performance
                recent_performance = model_state.performance_history[-1]

                # Combined performance score
                performance_score = (
                    recent_performance.r2_score * 0.3 +
                    recent_performance.sharpe_ratio * 0.3 +
                    recent_performance.hit_rate * 0.2 +
                    (1 - recent_performance.max_drawdown) * 0.2
                )

                performance_scores[model_id] = max(0.0, performance_score)

                # Recency factor (favor recent models)
                days_old = (current_time - model_state.training_date).days
                recency_factor = np.exp(-days_old / 30.0)  # Decay over 30 days
                recency_scores[model_id] = recency_factor

                # Stability factor (penalize overfitting)
                stability_factor = 1.0 / (1.0 + recent_performance.overfitting_score)
                stability_scores[model_id] = stability_factor

            # Calculate ensemble weights
            total_score = 0.0
            for model_id in performance_scores:
                combined_score = (
                    performance_scores[model_id] * 0.6 +
                    recency_scores[model_id] * 0.3 +
                    stability_scores[model_id] * 0.1
                )
                total_score += combined_score

            # Normalize weights
            self.ensemble_weights.clear()
            for model_id in performance_scores:
                if total_score > 0:
                    combined_score = (
                        performance_scores[model_id] * 0.6 +
                        recency_scores[model_id] * 0.3 +
                        stability_scores[model_id] * 0.1
                    )
                    weight = combined_score / total_score
                else:
                    weight = 1.0 / len(performance_scores)

                self.ensemble_weights[model_id] = EnsembleWeight(
                    model_id=model_id,
                    weight=weight,
                    performance_score=performance_scores[model_id],
                    recency_factor=recency_scores[model_id],
                    stability_factor=stability_scores[model_id],
                    last_updated=current_time
                )

            # Store ensemble weights
            await self._store_ensemble_weights()

            self.logger.info(f"Updated ensemble weights for {len(self.ensemble_weights)} models")

        except Exception as e:
            self.logger.error(f"Ensemble weight update failed: {e}")

    async def predict_ensemble(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate ensemble predictions from all active models"""
        try:
            if not self.active_models or not self.ensemble_weights:
                return {"error": "No active models for prediction"}

            predictions = []
            weights = []
            model_contributions = {}

            for model_id, model_state in self.active_models.items():
                if not model_state.is_active or model_id not in self.ensemble_weights:
                    continue

                try:
                    # Scale features
                    features_scaled = model_state.scaler.transform(features.reshape(1, -1))

                    # Get prediction
                    prediction = model_state.model_object.predict(features_scaled)[0]

                    # Get model weight
                    weight = self.ensemble_weights[model_id].weight

                    predictions.append(prediction)
                    weights.append(weight)
                    model_contributions[model_id] = {
                        "prediction": float(prediction),
                        "weight": float(weight),
                        "model_type": model_state.model_type.value
                    }

                except Exception as e:
                    self.logger.warning(f"Prediction failed for model {model_id}: {e}")
                    continue

            if not predictions:
                return {"error": "No successful predictions"}

            # Calculate weighted ensemble prediction
            predictions_array = np.array(predictions)
            weights_array = np.array(weights)

            # Normalize weights
            weights_normalized = weights_array / np.sum(weights_array)

            ensemble_prediction = np.sum(predictions_array * weights_normalized)

            # Calculate prediction confidence (inverse of prediction variance)
            prediction_variance = np.sum(weights_normalized * (predictions_array - ensemble_prediction) ** 2)
            confidence = 1.0 / (1.0 + prediction_variance)

            return {
                "ensemble_prediction": float(ensemble_prediction),
                "confidence": float(confidence),
                "individual_predictions": model_contributions,
                "active_models_count": len(predictions),
                "prediction_variance": float(prediction_variance),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return {"error": str(e)}

    async def retrain_models(self) -> Dict[str, str]:
        """Retrain models that need updating based on performance degradation"""
        try:
            retrain_results = {}
            current_time = datetime.now()

            # Get simulation data for retraining
            sim_data = self._simulation_state
            features = sim_data['features']
            targets = sim_data['target']
            feature_names = sim_data['feature_names']

            for model_id, model_state in list(self.active_models.items()):
                # Check if model needs retraining
                days_since_training = (current_time - model_state.training_date).days

                should_retrain = False

                # Time-based retraining
                if days_since_training >= self.config["retraining_frequency"]:
                    should_retrain = True
                    reason = "scheduled_retraining"

                # Performance-based retraining
                if model_state.performance_history:
                    recent_performance = model_state.performance_history[-1]
                    if (recent_performance.r2_score < self.config["performance_thresholds"]["min_r2_score"] or
                        recent_performance.overfitting_score > self.config["performance_thresholds"]["overfitting_threshold"]):
                        should_retrain = True
                        reason = "performance_degradation"

                if should_retrain:
                    self.logger.info(f"Retraining model {model_id} due to {reason}")

                    try:
                        # Train new model
                        new_model_id = await self.train_model(
                            model_state.model_type,
                            features,
                            targets,
                            feature_names,
                            model_state.hyperparameters
                        )

                        if new_model_id:
                            # Deactivate old model
                            model_state.is_active = False
                            retrain_results[model_id] = f"replaced_with_{new_model_id}"
                        else:
                            retrain_results[model_id] = "failed"

                    except Exception as e:
                        self.logger.error(f"Retraining failed for model {model_id}: {e}")
                        retrain_results[model_id] = f"error_{str(e)[:50]}"

            # Clean up inactive models
            await self._cleanup_inactive_models()

            return retrain_results

        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            return {"error": str(e)}

    async def detect_feature_drift(self, new_features: np.ndarray,
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Detect feature drift in new data compared to training distribution"""
        try:
            drift_results = {}

            # Get historical feature statistics
            historical_features = self._simulation_state['features']

            for i, feature_name in enumerate(feature_names):
                if i >= new_features.shape[1] or i >= historical_features.shape[1]:
                    continue

                # Current feature values
                current_values = new_features[:, i]
                historical_values = historical_features[:, i]

                # Statistical tests for drift
                drift_score = self._calculate_drift_score(current_values, historical_values)

                # Update feature importance tracker
                if feature_name in self.feature_importance_tracker:
                    self.feature_importance_tracker[feature_name].drift_indicator = drift_score
                    self.feature_importance_tracker[feature_name].last_updated = datetime.now()

                drift_results[feature_name] = {
                    "drift_score": float(drift_score),
                    "is_drifted": drift_score > self.config["drift_detection"]["sensitivity"],
                    "current_mean": float(np.mean(current_values)),
                    "current_std": float(np.std(current_values)),
                    "historical_mean": float(np.mean(historical_values)),
                    "historical_std": float(np.std(historical_values))
                }

            # Count drifted features
            drifted_features = [name for name, result in drift_results.items()
                              if result["is_drifted"]]

            # Update learning metrics
            self.learning_metrics["feature_drift_alerts"] = len(drifted_features)

            return {
                "drift_results": drift_results,
                "total_features_tested": len(drift_results),
                "drifted_features_count": len(drifted_features),
                "drifted_features": drifted_features,
                "overall_drift_score": np.mean([r["drift_score"] for r in drift_results.values()]),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Feature drift detection failed: {e}")
            return {"error": str(e)}

    def _calculate_drift_score(self, current_values: np.ndarray,
                              historical_values: np.ndarray) -> float:
        """Calculate statistical drift score between current and historical values"""
        try:
            from scipy import stats

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(current_values, historical_values)

            # Population stability index (PSI) approximation
            def calculate_psi(expected, actual, buckets=10):
                expected_percents = np.histogram(expected, buckets)[0] / len(expected)
                actual_percents = np.histogram(actual, buckets)[0] / len(actual)

                # Avoid division by zero
                expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
                actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

                psi = np.sum((actual_percents - expected_percents) *
                           np.log(actual_percents / expected_percents))

                return psi

            psi_score = calculate_psi(historical_values, current_values)

            # Combined drift score
            drift_score = ks_stat * 2 + psi_score * 0.5

            return drift_score

        except Exception as e:
            self.logger.error(f"Drift score calculation failed: {e}")
            return 0.0

    async def _save_model(self, model_state: ModelState):
        """Save model to disk for persistence"""
        try:
            model_file = self.models_path / f"{model_state.model_id}.pkl"

            # Save model components
            model_data = {
                'model_id': model_state.model_id,
                'model_type': model_state.model_type.value,
                'model_object': model_state.model_object,
                'scaler': model_state.scaler,
                'feature_names': model_state.feature_names,
                'hyperparameters': model_state.hyperparameters,
                'training_date': model_state.training_date.isoformat(),
                'is_active': model_state.is_active
            }

            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"Model saved: {model_file}")

        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")

    async def load_model(self, model_id: str) -> bool:
        """Load model from disk"""
        try:
            model_file = self.models_path / f"{model_id}.pkl"

            if not model_file.exists():
                self.logger.error(f"Model file not found: {model_file}")
                return False

            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            # Reconstruct model state
            model_state = ModelState(
                model_id=model_data['model_id'],
                model_type=ModelType(model_data['model_type']),
                model_object=model_data['model_object'],
                scaler=model_data['scaler'],
                feature_names=model_data['feature_names'],
                hyperparameters=model_data['hyperparameters'],
                training_date=datetime.fromisoformat(model_data['training_date']),
                performance_history=[],
                feature_importance=[],
                is_active=model_data['is_active']
            )

            self.active_models[model_id] = model_state

            self.logger.info(f"Model loaded: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False

    async def _cleanup_inactive_models(self):
        """Remove inactive models from memory and optionally from disk"""
        try:
            inactive_models = [model_id for model_id, model_state in self.active_models.items()
                             if not model_state.is_active]

            for model_id in inactive_models:
                del self.active_models[model_id]

                # Remove from ensemble weights
                if model_id in self.ensemble_weights:
                    del self.ensemble_weights[model_id]

            if inactive_models:
                self.logger.info(f"Cleaned up {len(inactive_models)} inactive models")

        except Exception as e:
            self.logger.error(f"Model cleanup failed: {e}")

    async def _store_performance_metrics(self, performance: ModelPerformance):
        """Store performance metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO model_performance (
                        model_id, model_type, timestamp, mse, mae, r2_score,
                        sharpe_ratio, hit_rate, profit_factor, max_drawdown,
                        calmar_ratio, information_ratio, tracking_error, beta,
                        alpha, feature_count, training_samples, validation_samples,
                        overfitting_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance.model_id, performance.model_type.value,
                    performance.timestamp.isoformat(), performance.mse, performance.mae,
                    performance.r2_score, performance.sharpe_ratio, performance.hit_rate,
                    performance.profit_factor, performance.max_drawdown, performance.calmar_ratio,
                    performance.information_ratio, performance.tracking_error, performance.beta,
                    performance.alpha, performance.feature_count, performance.training_samples,
                    performance.validation_samples, performance.overfitting_score
                ))
                conn.commit()

        except Exception as e:
            self.logger.error(f"Performance metrics storage failed: {e}")

    async def _store_ensemble_weights(self):
        """Store ensemble weights in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for weight_data in self.ensemble_weights.values():
                    conn.execute("""
                        INSERT INTO ensemble_weights (
                            model_id, weight, performance_score, recency_factor,
                            stability_factor, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        weight_data.model_id, weight_data.weight,
                        weight_data.performance_score, weight_data.recency_factor,
                        weight_data.stability_factor, weight_data.last_updated.isoformat()
                    ))
                conn.commit()

        except Exception as e:
            self.logger.error(f"Ensemble weights storage failed: {e}")

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current AI learning system status"""
        try:
            active_models_info = {}
            for model_id, model_state in self.active_models.items():
                active_models_info[model_id] = {
                    "type": model_state.model_type.value,
                    "training_date": model_state.training_date.isoformat(),
                    "feature_count": len(model_state.feature_names),
                    "is_active": model_state.is_active,
                    "performance_records": len(model_state.performance_history)
                }

            ensemble_weights_info = {
                model_id: {
                    "weight": weight_data.weight,
                    "performance_score": weight_data.performance_score
                }
                for model_id, weight_data in self.ensemble_weights.items()
            }

            return {
                "learning_metrics": self.learning_metrics.copy(),
                "active_models": active_models_info,
                "ensemble_weights": ensemble_weights_info,
                "current_mode": self.current_mode.value,
                "total_performance_records": len(self.performance_history),
                "feature_drift_tracker_size": len(self.feature_importance_tracker),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}

    async def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning system report"""
        try:
            # Model performance summary
            model_performances = []
            for model_state in self.active_models.values():
                if model_state.performance_history:
                    latest_perf = model_state.performance_history[-1]
                    model_performances.append({
                        "model_id": model_state.model_id,
                        "model_type": model_state.model_type.value,
                        "r2_score": latest_perf.r2_score,
                        "sharpe_ratio": latest_perf.sharpe_ratio,
                        "hit_rate": latest_perf.hit_rate,
                        "max_drawdown": latest_perf.max_drawdown
                    })

            # Feature importance summary
            feature_summary = {}
            for model_state in self.active_models.values():
                for feature_imp in model_state.feature_importance:
                    if feature_imp.feature_name not in feature_summary:
                        feature_summary[feature_imp.feature_name] = {
                            "importance_scores": [],
                            "stability_scores": []
                        }
                    feature_summary[feature_imp.feature_name]["importance_scores"].append(
                        feature_imp.importance_score
                    )
                    feature_summary[feature_imp.feature_name]["stability_scores"].append(
                        feature_imp.stability_score
                    )

            # Calculate average importance and stability
            feature_rankings = []
            for feature_name, data in feature_summary.items():
                feature_rankings.append({
                    "feature_name": feature_name,
                    "avg_importance": np.mean(data["importance_scores"]),
                    "avg_stability": np.mean(data["stability_scores"]),
                    "importance_std": np.std(data["importance_scores"])
                })

            # Sort by average importance
            feature_rankings.sort(key=lambda x: x["avg_importance"], reverse=True)

            report = {
                "report_timestamp": datetime.now().isoformat(),
                "system_summary": {
                    "total_models": len(self.active_models),
                    "active_models": len([m for m in self.active_models.values() if m.is_active]),
                    "average_r2_score": np.mean([p["r2_score"] for p in model_performances]) if model_performances else 0.0,
                    "average_sharpe_ratio": np.mean([p["sharpe_ratio"] for p in model_performances]) if model_performances else 0.0,
                    "average_hit_rate": np.mean([p["hit_rate"] for p in model_performances]) if model_performances else 0.0
                },
                "model_performance": model_performances,
                "feature_rankings": feature_rankings[:20],  # Top 20 features
                "ensemble_status": {
                    "total_weight": sum(w.weight for w in self.ensemble_weights.values()),
                    "model_count": len(self.ensemble_weights),
                    "weight_distribution": {
                        model_id: weight.weight
                        for model_id, weight in self.ensemble_weights.items()
                    }
                },
                "learning_metrics": self.learning_metrics.copy()
            }

            # Save report
            report_path = Path("reports") / f"ai_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"Learning report generated: {report_path}")

            return report

        except Exception as e:
            self.logger.error(f"Learning report generation failed: {e}")
            return {"error": str(e)}


class FeatureDriftDetector:
    """Helper class for feature drift detection"""

    def __init__(self):
        self.baseline_statistics = {}
        self.drift_history = {}

    def update_baseline(self, feature_name: str, values: np.ndarray):
        """Update baseline statistics for a feature"""
        self.baseline_statistics[feature_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'percentiles': np.percentile(values, [5, 25, 50, 75, 95])
        }

    def detect_drift(self, feature_name: str, new_values: np.ndarray) -> float:
        """Detect drift for a specific feature"""
        if feature_name not in self.baseline_statistics:
            return 0.0

        baseline = self.baseline_statistics[feature_name]

        # Calculate various drift metrics
        mean_shift = abs(np.mean(new_values) - baseline['mean']) / baseline['std']
        std_ratio = np.std(new_values) / baseline['std']

        # Percentile-based drift
        new_percentiles = np.percentile(new_values, [5, 25, 50, 75, 95])
        percentile_drift = np.mean(np.abs(new_percentiles - baseline['percentiles']) / baseline['std'])

        # Combined drift score
        drift_score = mean_shift + abs(1 - std_ratio) + percentile_drift

        return drift_score


# Example usage and testing
async def main():
    """Main function for testing the AI learning engine"""
    print("AI Learning Engine - Investment Grade Machine Learning")
    print("=" * 60)

    # Initialize AI learning engine
    ai_engine = AILearningEngine()

    # Get simulation data
    sim_data = ai_engine._simulation_state
    features = sim_data['features']
    targets = sim_data['target']
    feature_names = sim_data['feature_names']

    print(f"Training data: {features.shape[0]} samples, {features.shape[1]} features")

    # Train multiple models
    model_types = [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, ModelType.RIDGE_REGRESSION]

    for model_type in model_types:
        print(f"\nTraining {model_type.value} model...")
        model_id = await ai_engine.train_model(model_type, features, targets, feature_names)
        if model_id:
            print(f"Model {model_id} trained successfully")

    # Test ensemble prediction
    print("\nTesting ensemble prediction...")
    test_features = features[0]  # Use first sample as test
    prediction_result = await ai_engine.predict_ensemble(test_features)

    if "error" not in prediction_result:
        print(f"Ensemble prediction: {prediction_result['ensemble_prediction']:.4f}")
        print(f"Confidence: {prediction_result['confidence']:.4f}")
        print(f"Active models: {prediction_result['active_models_count']}")

    # Test feature drift detection
    print("\nTesting feature drift detection...")
    drift_results = await ai_engine.detect_feature_drift(features[:100], feature_names)
    print(f"Features tested: {drift_results['total_features_tested']}")
    print(f"Drifted features: {drift_results['drifted_features_count']}")

    # Generate learning report
    print("\nGenerating learning report...")
    report = await ai_engine.generate_learning_report()
    print(f"Report generated with {len(report.get('model_performance', []))} model records")

    # Get system status
    status = ai_engine.get_learning_status()
    print(f"\nSystem Status:")
    print(f"Active models: {status['learning_metrics']['active_models_count']}")
    print(f"Total trained: {status['learning_metrics']['total_models_trained']}")

    print("\nAI Learning Engine test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())