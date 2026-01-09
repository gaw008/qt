#!/usr/bin/env python3
"""
Validation Integration Adapter

Provides seamless integration between the Walk-Forward Validation Framework
and existing system components including Purged K-Fold CV, Risk Management,
and Historical Data Management.

Key Features:
- Unified validation interface for all system strategies
- Automatic data preparation and preprocessing
- Integration with existing risk management systems
- Performance benchmark generation and comparison
- Automated reporting and result visualization
- Quality assurance pipeline integration
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import warnings
from pathlib import Path
import json

# Import validation framework components
from bot.walk_forward_validator import (
    WalkForwardValidator, WalkForwardConfig, WalkForwardResults,
    ValidationPhase, WindowType, StatisticalTest
)
from bot.purged_kfold_validator import PurgedKFoldCV, ValidationConfig, ValidationResults
from bot.enhanced_risk_manager import EnhancedRiskManager, TailRiskMetrics, MarketRegime
from bot.historical_data_manager import HistoricalDataManager

# Import existing system components
try:
    from bot.scoring_engine import MultiFactorScoringEngine
    from bot.selection_strategies.base_strategy import SelectionStrategy
    from bot.portfolio import MultiStockPortfolio
    from bot.data import fetch_history, fetch_batch_history
    from bot.config import SETTINGS
    SYSTEM_IMPORTS = True
except ImportError as e:
    logging.warning(f"Could not import all system components: {e}")
    SYSTEM_IMPORTS = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class ValidationPipeline:
    """Configuration for integrated validation pipeline"""
    # Strategy configuration
    strategy_name: str
    strategy_function: Optional[Callable] = None
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_optimizer: Optional[Callable] = None

    # Data configuration
    symbols: List[str] = field(default_factory=list)
    start_date: str = "2006-01-01"
    end_date: str = "2024-12-31"
    benchmark_symbol: str = "SPY"

    # Validation methods
    enable_walk_forward: bool = True
    enable_purged_kfold: bool = True
    enable_risk_assessment: bool = True

    # Walk-forward configuration
    walk_forward_config: Optional[WalkForwardConfig] = None

    # Purged K-fold configuration
    purged_kfold_config: Optional[ValidationConfig] = None

    # Output configuration
    results_directory: str = "reports/integrated_validation"
    generate_comprehensive_report: bool = True
    export_data: bool = True


@dataclass
class IntegratedValidationResults:
    """Comprehensive results from integrated validation"""
    pipeline_config: ValidationPipeline
    validation_timestamp: str
    total_validation_time: float

    # Walk-forward results
    walk_forward_results: Optional[WalkForwardResults] = None

    # Purged K-fold results
    purged_kfold_results: Optional[ValidationResults] = None

    # Risk assessment results
    risk_assessment: Optional[Dict[str, Any]] = None

    # Cross-validation comparison
    validation_consistency: Dict[str, Any] = field(default_factory=dict)

    # Final assessment
    overall_validation_passed: bool = False
    validation_confidence: float = 0.0
    recommendation: str = "REJECT"
    quality_warnings: List[str] = field(default_factory=list)

    # Performance summary
    performance_summary: Dict[str, Any] = field(default_factory=dict)

    # Benchmark comparison
    benchmark_analysis: Dict[str, Any] = field(default_factory=dict)


class ValidationIntegrationAdapter:
    """
    Integration adapter for comprehensive strategy validation using
    multiple validation methodologies
    """

    def __init__(self):
        # Initialize core components
        self.data_manager = HistoricalDataManager() if SYSTEM_IMPORTS else None
        self.risk_manager = EnhancedRiskManager()

        # Validation component cache
        self._validator_cache = {}
        self._data_cache = {}

        logger.info("Validation integration adapter initialized")

    def validate_strategy(self, pipeline: ValidationPipeline) -> IntegratedValidationResults:
        """
        Perform comprehensive strategy validation using multiple methodologies

        Args:
            pipeline: Validation pipeline configuration

        Returns:
            Comprehensive validation results
        """
        validation_start = datetime.now()
        logger.info(f"Starting integrated validation for strategy: {pipeline.strategy_name}")

        # Prepare data
        data, benchmark_data = self._prepare_validation_data(pipeline)

        # Initialize results
        results = IntegratedValidationResults(
            pipeline_config=pipeline,
            validation_timestamp=validation_start.isoformat(),
            total_validation_time=0.0
        )

        # Perform walk-forward validation
        if pipeline.enable_walk_forward:
            try:
                results.walk_forward_results = self._perform_walk_forward_validation(
                    pipeline, data, benchmark_data
                )
                logger.info("Walk-forward validation completed successfully")
            except Exception as e:
                logger.error(f"Walk-forward validation failed: {e}")
                results.quality_warnings.append(f"Walk-forward validation error: {str(e)}")

        # Perform purged K-fold validation
        if pipeline.enable_purged_kfold:
            try:
                results.purged_kfold_results = self._perform_purged_kfold_validation(
                    pipeline, data
                )
                logger.info("Purged K-fold validation completed successfully")
            except Exception as e:
                logger.error(f"Purged K-fold validation failed: {e}")
                results.quality_warnings.append(f"Purged K-fold validation error: {str(e)}")

        # Perform risk assessment
        if pipeline.enable_risk_assessment:
            try:
                results.risk_assessment = self._perform_risk_assessment(
                    pipeline, data, results
                )
                logger.info("Risk assessment completed successfully")
            except Exception as e:
                logger.error(f"Risk assessment failed: {e}")
                results.quality_warnings.append(f"Risk assessment error: {str(e)}")

        # Cross-validation consistency analysis
        results.validation_consistency = self._analyze_validation_consistency(results)

        # Performance summary
        results.performance_summary = self._generate_performance_summary(results)

        # Benchmark analysis
        results.benchmark_analysis = self._analyze_benchmark_performance(
            results, benchmark_data
        )

        # Final assessment
        results.overall_validation_passed, results.validation_confidence = \
            self._assess_overall_validation(results)

        results.recommendation = self._generate_recommendation(results)

        # Calculate total time
        results.total_validation_time = (datetime.now() - validation_start).total_seconds()

        # Generate comprehensive report
        if pipeline.generate_comprehensive_report:
            self._generate_comprehensive_report(results)

        # Export data if requested
        if pipeline.export_data:
            self._export_validation_data(results, data, benchmark_data)

        logger.info(f"Integrated validation completed in {results.total_validation_time:.1f}s")
        logger.info(f"Final recommendation: {results.recommendation}")

        return results

    def _prepare_validation_data(
        self,
        pipeline: ValidationPipeline
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for validation"""

        # Check cache first
        cache_key = f"{hash(tuple(pipeline.symbols))}_{pipeline.start_date}_{pipeline.end_date}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        logger.info(f"Preparing validation data for {len(pipeline.symbols)} symbols")

        # Load historical data
        if self.data_manager and len(pipeline.symbols) > 0:
            # Use integrated data manager
            main_symbol = pipeline.symbols[0]  # Use first symbol as primary
            data = self.data_manager.get_historical_data(
                symbol=main_symbol,
                start_date=pipeline.start_date,
                end_date=pipeline.end_date,
                adjusted=True
            )
        else:
            # Generate synthetic data for testing
            data = self._generate_synthetic_data(pipeline)

        # Load benchmark data
        try:
            if self.data_manager:
                benchmark_data = self.data_manager.get_historical_data(
                    symbol=pipeline.benchmark_symbol,
                    start_date=pipeline.start_date,
                    end_date=pipeline.end_date,
                    adjusted=True
                )
            else:
                benchmark_data = self._generate_synthetic_benchmark(pipeline, data)
        except Exception as e:
            logger.warning(f"Could not load benchmark data: {e}")
            benchmark_data = self._generate_synthetic_benchmark(pipeline, data)

        # Data quality checks
        data = self._ensure_data_quality(data)
        benchmark_data = self._ensure_data_quality(benchmark_data)

        # Cache results
        self._data_cache[cache_key] = (data, benchmark_data)

        logger.info(f"Data preparation completed: {len(data)} main observations, "
                   f"{len(benchmark_data)} benchmark observations")

        return data, benchmark_data

    def _generate_synthetic_data(self, pipeline: ValidationPipeline) -> pd.DataFrame:
        """Generate synthetic market data for testing"""
        logger.info("Generating synthetic market data for validation")

        # Date range
        start_dt = pd.to_datetime(pipeline.start_date)
        end_dt = pd.to_datetime(pipeline.end_date)
        dates = pd.date_range(start_dt, end_dt, freq='D')

        # Generate realistic returns with regime changes
        np.random.seed(42)  # For reproducibility
        n_days = len(dates)

        # Base returns with volatility clustering
        returns = np.random.randn(n_days) * 0.01

        # Add regime-specific patterns
        # Bull market periods
        bull_periods = [
            (0, int(0.3 * n_days)),      # Early period
            (int(0.6 * n_days), int(0.8 * n_days))  # Recovery period
        ]

        # Crisis periods
        crisis_periods = [
            (int(0.25 * n_days), int(0.35 * n_days)),  # Financial crisis
            (int(0.82 * n_days), int(0.85 * n_days))   # COVID-like event
        ]

        # Apply regime patterns
        for start, end in bull_periods:
            returns[start:end] += 0.0003  # Positive drift

        for start, end in crisis_periods:
            returns[start:end] -= 0.001   # Negative drift
            returns[start:end] *= 2.0     # Higher volatility

        # Generate price levels
        initial_price = 100.0
        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLC data
        data = pd.DataFrame(index=dates)
        data['close'] = prices

        # Simple OHLC generation
        daily_range = np.abs(np.random.randn(n_days)) * 0.02 * prices
        data['high'] = prices + daily_range * np.random.uniform(0.3, 0.7, n_days)
        data['low'] = prices - daily_range * np.random.uniform(0.3, 0.7, n_days)
        data['open'] = prices + (data['high'] - data['low']) * np.random.uniform(-0.5, 0.5, n_days)

        # Volume
        data['volume'] = np.random.randint(1000000, 10000000, n_days)

        # Ensure OHLC consistency
        data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
        data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])

        return data

    def _generate_synthetic_benchmark(
        self,
        pipeline: ValidationPipeline,
        main_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate synthetic benchmark data"""

        # Use main data dates
        dates = main_data.index

        # Generate benchmark returns (slightly lower volatility)
        main_returns = main_data['close'].pct_change().fillna(0)
        benchmark_returns = main_returns * 0.8 + np.random.randn(len(dates)) * 0.005

        # Generate benchmark prices
        initial_price = 100.0
        benchmark_prices = initial_price * np.exp(np.cumsum(benchmark_returns))

        benchmark_data = pd.DataFrame({'close': benchmark_prices}, index=dates)
        return benchmark_data

    def _ensure_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data quality for validation"""

        if data is None or len(data) == 0:
            raise ValueError("Empty data provided for validation")

        # Remove NaN values
        data = data.dropna()

        # Ensure required columns
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Sort by date
        data = data.sort_index()

        # Remove duplicates
        data = data[~data.index.duplicated(keep='last')]

        return data

    def _perform_walk_forward_validation(
        self,
        pipeline: ValidationPipeline,
        data: pd.DataFrame,
        benchmark_data: pd.DataFrame
    ) -> WalkForwardResults:
        """Perform walk-forward validation"""

        logger.info("Performing walk-forward validation")

        # Configure walk-forward validation
        if pipeline.walk_forward_config:
            config = pipeline.walk_forward_config
        else:
            config = WalkForwardConfig(
                min_train_months=24,
                test_window_months=6,
                step_months=3,
                window_type=WindowType.EXPANDING,
                confidence_level=0.95,
                bootstrap_samples=5000,
                multiple_testing_method="fdr_bh",
                save_detailed_results=True,
                results_directory=pipeline.results_directory
            )

        # Initialize validator
        validator = WalkForwardValidator(config)

        # Prepare strategy function
        strategy_func = self._prepare_strategy_function(pipeline)

        # Run validation
        results = validator.validate_strategy(
            strategy_func=strategy_func,
            data=data,
            benchmark_data=benchmark_data,
            strategy_params=pipeline.strategy_parameters,
            param_optimizer=pipeline.parameter_optimizer
        )

        return results

    def _perform_purged_kfold_validation(
        self,
        pipeline: ValidationPipeline,
        data: pd.DataFrame
    ) -> ValidationResults:
        """Perform purged K-fold cross-validation"""

        logger.info("Performing purged K-fold validation")

        # Configure purged K-fold validation
        if pipeline.purged_kfold_config:
            config = pipeline.purged_kfold_config
        else:
            config = ValidationConfig(
                n_splits=5,
                embargo_days=5,
                purge_days=3,
                min_train_samples=252,
                test_size_ratio=0.2,
                confidence_level=0.95
            )

        # Initialize validator
        validator = PurgedKFoldCV(config)

        # Prepare features and target
        features, target = self._prepare_features_and_target(data, pipeline)

        # Prepare model function
        model_func = self._prepare_model_function(pipeline)

        # Run validation
        results = validator.cross_validate_model(
            features=features,
            target=target,
            timestamps=data.index,
            model_func=model_func
        )

        return results

    def _prepare_strategy_function(self, pipeline: ValidationPipeline) -> Callable:
        """Prepare strategy function for validation"""

        if pipeline.strategy_function:
            return pipeline.strategy_function

        # Default momentum strategy
        def default_momentum_strategy(data: pd.DataFrame, **params) -> pd.Series:
            lookback = params.get('lookback', 20)
            threshold = params.get('threshold', 0.02)

            if 'close' not in data.columns or len(data) < lookback + 1:
                return pd.Series(index=data.index, data=0.0)

            returns = data['close'].pct_change()
            momentum = returns.rolling(lookback).mean()

            signals = np.where(momentum > threshold, 1,
                             np.where(momentum < -threshold, -1, 0))

            strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns
            return strategy_returns.fillna(0)

        return default_momentum_strategy

    def _prepare_features_and_target(
        self,
        data: pd.DataFrame,
        pipeline: ValidationPipeline
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model validation"""

        # Calculate basic technical indicators as features
        features = pd.DataFrame(index=data.index)

        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['returns_lag1'] = features['returns'].shift(1)
        features['volatility'] = features['returns'].rolling(20).std()

        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'ma_{period}'] = data['close'].rolling(period).mean()
            features[f'ma_ratio_{period}'] = data['close'] / features[f'ma_{period}']

        # Technical indicators
        if len(data) > 50:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))

        # Volume features (if available)
        if 'volume' in data.columns:
            features['volume_ma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_ma']

        # Remove NaN values
        features = features.dropna()

        # Target: forward returns
        target = features['returns'].shift(-1).dropna()

        # Align features and target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]

        return features, target

    def _prepare_model_function(self, pipeline: ValidationPipeline) -> Callable:
        """Prepare model function for cross-validation"""

        def linear_model(X_train, y_train):
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            # Fit model
            model = LinearRegression()
            model.fit(X_scaled, y_train)

            # Create prediction function
            def predict(X_test):
                X_test_scaled = scaler.transform(X_test)
                return model.predict(X_test_scaled)

            # Add predict method to model
            model.predict = predict
            return model

        return linear_model

    def _perform_risk_assessment(
        self,
        pipeline: ValidationPipeline,
        data: pd.DataFrame,
        results: IntegratedValidationResults
    ) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""

        logger.info("Performing risk assessment")

        # Generate strategy returns for risk analysis
        strategy_func = self._prepare_strategy_function(pipeline)
        strategy_returns = strategy_func(data, **pipeline.strategy_parameters)

        if len(strategy_returns) == 0:
            return {'error': 'No strategy returns for risk assessment'}

        # Calculate tail risk metrics
        tail_metrics = self.risk_manager.calculate_tail_risk_metrics(strategy_returns.values)

        # Market regime analysis
        market_data = {
            'vix': 20.0,  # Default VIX level
            'market_correlation': 0.5,
            'momentum_strength': 0.0
        }

        # Simulate portfolio for risk assessment
        mock_portfolio = {
            'total_value': 1000000,
            'positions': [
                {'symbol': symbol, 'market_value': 100000, 'sector': 'Technology'}
                for symbol in pipeline.symbols[:10]  # Limit to 10 positions
            ]
        }

        # Perform risk assessment
        risk_assessment = self.risk_manager.assess_portfolio_risk(
            portfolio=mock_portfolio,
            market_data=market_data,
            returns_history=strategy_returns.values
        )

        # Add strategy-specific risk metrics
        risk_assessment['strategy_tail_metrics'] = {
            'es_97_5': tail_metrics.es_97_5,
            'es_99': tail_metrics.es_99,
            'tail_ratio': tail_metrics.tail_ratio,
            'max_drawdown': tail_metrics.max_drawdown,
            'calmar_ratio': tail_metrics.calmar_ratio,
            'skewness': tail_metrics.skewness,
            'kurtosis': tail_metrics.kurtosis
        }

        return risk_assessment

    def _analyze_validation_consistency(
        self,
        results: IntegratedValidationResults
    ) -> Dict[str, Any]:
        """Analyze consistency across validation methods"""

        consistency = {
            'methods_used': [],
            'performance_agreement': {},
            'risk_agreement': {},
            'overall_consistency_score': 0.0
        }

        # Track which methods were used
        if results.walk_forward_results:
            consistency['methods_used'].append('walk_forward')

        if results.purged_kfold_results:
            consistency['methods_used'].append('purged_kfold')

        if results.risk_assessment:
            consistency['methods_used'].append('risk_assessment')

        # Performance agreement analysis
        if len(consistency['methods_used']) >= 2:
            performance_metrics = {}

            # Extract metrics from walk-forward
            if results.walk_forward_results:
                wf_metrics = {}
                for phase, phase_result in results.walk_forward_results.phase_results.items():
                    wf_metrics[f'wf_{phase.value}_sharpe'] = phase_result.sharpe_ratio
                    wf_metrics[f'wf_{phase.value}_drawdown'] = phase_result.max_drawdown

                performance_metrics.update(wf_metrics)

            # Extract metrics from purged K-fold
            if results.purged_kfold_results:
                performance_metrics['pkf_mean_score'] = results.purged_kfold_results.mean_score
                performance_metrics['pkf_significance'] = results.purged_kfold_results.is_significant

            # Calculate agreement scores
            if len(performance_metrics) > 0:
                consistency['performance_agreement'] = performance_metrics

        # Overall consistency score
        consistency['overall_consistency_score'] = self._calculate_consistency_score(results)

        return consistency

    def _calculate_consistency_score(self, results: IntegratedValidationResults) -> float:
        """Calculate overall consistency score across validation methods"""

        scores = []

        # Walk-forward consistency
        if results.walk_forward_results:
            wf_score = 0.0
            passed_phases = sum(1 for phase_result in results.walk_forward_results.phase_results.values()
                              if phase_result.sharpe_ratio > 0)
            total_phases = len(results.walk_forward_results.phase_results)

            if total_phases > 0:
                wf_score = passed_phases / total_phases

            scores.append(wf_score)

        # Purged K-fold consistency
        if results.purged_kfold_results:
            pkf_score = 1.0 if results.purged_kfold_results.is_significant else 0.0
            scores.append(pkf_score)

        # Risk assessment consistency
        if results.risk_assessment:
            risk_violations = results.risk_assessment.get('risk_violations', [])
            risk_score = 1.0 if len(risk_violations) == 0 else 0.5
            scores.append(risk_score)

        return np.mean(scores) if scores else 0.0

    def _generate_performance_summary(
        self,
        results: IntegratedValidationResults
    ) -> Dict[str, Any]:
        """Generate comprehensive performance summary"""

        summary = {
            'strategy_name': results.pipeline_config.strategy_name,
            'validation_methods': len(results.validation_consistency.get('methods_used', [])),
            'overall_performance': {},
            'risk_metrics': {},
            'statistical_significance': {},
            'recommendations': []
        }

        # Walk-forward performance
        if results.walk_forward_results:
            wf_performance = {}
            for phase, phase_result in results.walk_forward_results.phase_results.items():
                wf_performance[f'{phase.value}_sharpe'] = phase_result.sharpe_ratio
                wf_performance[f'{phase.value}_return'] = phase_result.mean_returns
                wf_performance[f'{phase.value}_drawdown'] = phase_result.max_drawdown

            summary['overall_performance']['walk_forward'] = wf_performance

            # Overall statistical significance
            summary['statistical_significance']['walk_forward'] = {
                'overall_significant': results.walk_forward_results.overall_significance.get('overall_significant', False),
                'combined_p_value': results.walk_forward_results.overall_significance.get('combined_p_value', 1.0)
            }

        # Purged K-fold performance
        if results.purged_kfold_results:
            summary['overall_performance']['purged_kfold'] = {
                'mean_score': results.purged_kfold_results.mean_score,
                'confidence_interval': results.purged_kfold_results.confidence_interval,
                'overfitting_ratio': results.purged_kfold_results.overfitting_ratio
            }

            summary['statistical_significance']['purged_kfold'] = {
                'significant': results.purged_kfold_results.is_significant,
                'p_value': results.purged_kfold_results.p_value
            }

        # Risk metrics
        if results.risk_assessment:
            tail_metrics = results.risk_assessment.get('strategy_tail_metrics', {})
            summary['risk_metrics'] = {
                'expected_shortfall_97_5': tail_metrics.get('es_97_5', 0),
                'max_drawdown': tail_metrics.get('max_drawdown', 0),
                'calmar_ratio': tail_metrics.get('calmar_ratio', 0),
                'tail_ratio': tail_metrics.get('tail_ratio', 0)
            }

        return summary

    def _analyze_benchmark_performance(
        self,
        results: IntegratedValidationResults,
        benchmark_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze performance relative to benchmark"""

        analysis = {
            'benchmark_symbol': results.pipeline_config.benchmark_symbol,
            'comparison_available': False
        }

        if benchmark_data is None or len(benchmark_data) == 0:
            return analysis

        # Calculate benchmark metrics
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        benchmark_performance = self._calculate_basic_metrics(benchmark_returns)

        analysis['benchmark_performance'] = benchmark_performance
        analysis['comparison_available'] = True

        # Compare with strategy performance
        if results.walk_forward_results:
            strategy_metrics = {}
            for phase, phase_result in results.walk_forward_results.phase_results.items():
                strategy_metrics[f'{phase.value}_sharpe'] = phase_result.sharpe_ratio
                strategy_metrics[f'{phase.value}_return'] = phase_result.mean_returns

            analysis['strategy_vs_benchmark'] = {
                'strategy_metrics': strategy_metrics,
                'outperformance': self._calculate_outperformance(strategy_metrics, benchmark_performance)
            }

        return analysis

    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics"""

        if len(returns) == 0:
            return {}

        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown)
        }

    def _calculate_outperformance(
        self,
        strategy_metrics: Dict[str, float],
        benchmark_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate strategy outperformance vs benchmark"""

        outperformance = {}

        benchmark_sharpe = benchmark_metrics.get('sharpe_ratio', 0)
        benchmark_return = benchmark_metrics.get('annualized_return', 0)

        # Calculate average strategy performance
        strategy_sharpe_values = [v for k, v in strategy_metrics.items() if 'sharpe' in k]
        strategy_return_values = [v for k, v in strategy_metrics.items() if 'return' in k]

        if strategy_sharpe_values:
            avg_strategy_sharpe = np.mean(strategy_sharpe_values)
            outperformance['sharpe_ratio_outperformance'] = avg_strategy_sharpe - benchmark_sharpe

        if strategy_return_values:
            avg_strategy_return = np.mean(strategy_return_values)
            outperformance['return_outperformance'] = avg_strategy_return - benchmark_return

        return outperformance

    def _assess_overall_validation(
        self,
        results: IntegratedValidationResults
    ) -> Tuple[bool, float]:
        """Assess overall validation status and confidence"""

        validation_scores = []
        confidence_factors = []

        # Walk-forward validation assessment
        if results.walk_forward_results:
            wf_passed = results.walk_forward_results.validation_passed
            validation_scores.append(1.0 if wf_passed else 0.0)

            # Confidence based on number of successful phases
            successful_phases = sum(1 for phase_result in results.walk_forward_results.phase_results.values()
                                  if phase_result.sharpe_ratio > 0)
            total_phases = len(results.walk_forward_results.phase_results)
            confidence_factors.append(successful_phases / total_phases if total_phases > 0 else 0)

        # Purged K-fold validation assessment
        if results.purged_kfold_results:
            pkf_passed = results.purged_kfold_results.is_significant and results.purged_kfold_results.mean_score > 0
            validation_scores.append(1.0 if pkf_passed else 0.0)
            confidence_factors.append(1.0 - results.purged_kfold_results.p_value)

        # Risk assessment
        if results.risk_assessment:
            risk_violations = results.risk_assessment.get('risk_violations', [])
            risk_passed = len(risk_violations) == 0
            validation_scores.append(1.0 if risk_passed else 0.5)
            confidence_factors.append(0.8 if risk_passed else 0.3)

        # Overall assessment
        if not validation_scores:
            return False, 0.0

        overall_passed = np.mean(validation_scores) > 0.6  # 60% threshold
        confidence = np.mean(confidence_factors) if confidence_factors else 0.0

        return overall_passed, confidence

    def _generate_recommendation(self, results: IntegratedValidationResults) -> str:
        """Generate final recommendation based on validation results"""

        if not results.overall_validation_passed:
            return "REJECT"

        confidence = results.validation_confidence

        if confidence > 0.8:
            return "ACCEPT_HIGH_CONFIDENCE"
        elif confidence > 0.6:
            return "ACCEPT_MEDIUM_CONFIDENCE"
        elif confidence > 0.4:
            return "ACCEPT_LOW_CONFIDENCE"
        else:
            return "CONDITIONAL_ACCEPT"

    def _generate_comprehensive_report(self, results: IntegratedValidationResults):
        """Generate comprehensive validation report"""

        # Create results directory
        results_dir = Path(results.pipeline_config.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert results to dictionary
        results_dict = self._convert_results_to_dict(results)

        # Save comprehensive results
        results_file = results_dir / f"integrated_validation_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)

        # Generate summary report
        self._generate_summary_report(results, results_dir, timestamp)

        logger.info(f"Comprehensive validation report saved to {results_dir}")

    def _convert_results_to_dict(self, results: IntegratedValidationResults) -> Dict[str, Any]:
        """Convert results object to dictionary for JSON serialization"""

        results_dict = {
            'pipeline_config': {
                'strategy_name': results.pipeline_config.strategy_name,
                'symbols': results.pipeline_config.symbols,
                'start_date': results.pipeline_config.start_date,
                'end_date': results.pipeline_config.end_date,
                'benchmark_symbol': results.pipeline_config.benchmark_symbol,
                'validation_methods': {
                    'walk_forward': results.pipeline_config.enable_walk_forward,
                    'purged_kfold': results.pipeline_config.enable_purged_kfold,
                    'risk_assessment': results.pipeline_config.enable_risk_assessment
                }
            },
            'validation_timestamp': results.validation_timestamp,
            'total_validation_time': results.total_validation_time,
            'overall_validation_passed': results.overall_validation_passed,
            'validation_confidence': results.validation_confidence,
            'recommendation': results.recommendation,
            'quality_warnings': results.quality_warnings,
            'validation_consistency': results.validation_consistency,
            'performance_summary': results.performance_summary,
            'benchmark_analysis': results.benchmark_analysis
        }

        # Add method-specific results (simplified)
        if results.walk_forward_results:
            results_dict['walk_forward_summary'] = {
                'validation_passed': results.walk_forward_results.validation_passed,
                'phase_count': len(results.walk_forward_results.phase_results),
                'overall_significant': results.walk_forward_results.overall_significance.get('overall_significant', False)
            }

        if results.purged_kfold_results:
            results_dict['purged_kfold_summary'] = {
                'mean_score': results.purged_kfold_results.mean_score,
                'is_significant': results.purged_kfold_results.is_significant,
                'p_value': results.purged_kfold_results.p_value
            }

        if results.risk_assessment:
            results_dict['risk_assessment_summary'] = {
                'risk_violations': results.risk_assessment.get('risk_violations', []),
                'active_alerts': results.risk_assessment.get('active_alerts', 0)
            }

        return results_dict

    def _generate_summary_report(
        self,
        results: IntegratedValidationResults,
        results_dir: Path,
        timestamp: str
    ):
        """Generate human-readable summary report"""

        report_file = results_dir / f"validation_summary_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Validation Summary Report\n\n")
            f.write(f"**Strategy:** {results.pipeline_config.strategy_name}\n")
            f.write(f"**Timestamp:** {results.validation_timestamp}\n")
            f.write(f"**Validation Time:** {results.total_validation_time:.1f} seconds\n\n")

            # Overall assessment
            f.write(f"## Overall Assessment\n\n")
            f.write(f"**Status:** {'PASSED' if results.overall_validation_passed else 'FAILED'}\n")
            f.write(f"**Confidence:** {results.validation_confidence:.3f}\n")
            f.write(f"**Recommendation:** {results.recommendation}\n\n")

            # Validation methods
            f.write(f"## Validation Methods\n\n")
            methods = results.validation_consistency.get('methods_used', [])
            for method in methods:
                f.write(f"- {method.replace('_', ' ').title()}\n")

            # Performance summary
            if results.performance_summary:
                f.write(f"\n## Performance Summary\n\n")
                for method, metrics in results.performance_summary.get('overall_performance', {}).items():
                    f.write(f"### {method.replace('_', ' ').title()}\n\n")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")

            # Risk metrics
            risk_metrics = results.performance_summary.get('risk_metrics', {})
            if risk_metrics:
                f.write(f"\n## Risk Metrics\n\n")
                for metric, value in risk_metrics.items():
                    f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")

            # Quality warnings
            if results.quality_warnings:
                f.write(f"\n## Quality Warnings\n\n")
                for warning in results.quality_warnings:
                    f.write(f"- {warning}\n")

        logger.info(f"Summary report saved to {report_file}")

    def _export_validation_data(
        self,
        results: IntegratedValidationResults,
        data: pd.DataFrame,
        benchmark_data: pd.DataFrame
    ):
        """Export validation data for further analysis"""

        export_dir = Path(results.pipeline_config.results_directory) / "data_exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export main data
        data_file = export_dir / f"strategy_data_{timestamp}.csv"
        data.to_csv(data_file)

        # Export benchmark data
        benchmark_file = export_dir / f"benchmark_data_{timestamp}.csv"
        benchmark_data.to_csv(benchmark_file)

        logger.info(f"Validation data exported to {export_dir}")


# Convenience functions for easy integration

def validate_strategy_comprehensive(
    strategy_name: str,
    strategy_function: Callable,
    symbols: List[str] = None,
    strategy_params: Dict[str, Any] = None,
    start_date: str = "2006-01-01",
    end_date: str = "2024-12-31"
) -> IntegratedValidationResults:
    """
    Convenience function for comprehensive strategy validation

    Args:
        strategy_name: Name of the strategy
        strategy_function: Strategy function to validate
        symbols: List of symbols (optional)
        strategy_params: Strategy parameters
        start_date: Start date for validation
        end_date: End date for validation

    Returns:
        Comprehensive validation results
    """
    # Create validation pipeline
    pipeline = ValidationPipeline(
        strategy_name=strategy_name,
        strategy_function=strategy_function,
        symbols=symbols or ["AAPL"],  # Default symbol
        strategy_parameters=strategy_params or {},
        start_date=start_date,
        end_date=end_date,
        enable_walk_forward=True,
        enable_purged_kfold=True,
        enable_risk_assessment=True,
        generate_comprehensive_report=True,
        export_data=True
    )

    # Initialize adapter and run validation
    adapter = ValidationIntegrationAdapter()
    return adapter.validate_strategy(pipeline)


if __name__ == "__main__":
    # Example usage of the Validation Integration Adapter
    print("Validation Integration Adapter - Comprehensive Strategy Testing")
    print("=" * 80)

    # Example momentum strategy
    def example_momentum_strategy(data: pd.DataFrame, lookback: int = 20, threshold: float = 0.02) -> pd.Series:
        """Example momentum strategy"""
        if 'close' not in data.columns or len(data) < lookback + 1:
            return pd.Series(index=data.index, data=0.0)

        returns = data['close'].pct_change()
        momentum = returns.rolling(lookback).mean()

        signals = np.where(momentum > threshold, 1,
                          np.where(momentum < -threshold, -1, 0))

        strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    # Run comprehensive validation
    results = validate_strategy_comprehensive(
        strategy_name="Example_Momentum_Strategy",
        strategy_function=example_momentum_strategy,
        symbols=["AAPL", "MSFT"],
        strategy_params={"lookback": 20, "threshold": 0.02},
        start_date="2015-01-01",
        end_date="2023-12-31"
    )

    # Print summary
    print(f"\nValidation Results:")
    print(f"Strategy: {results.pipeline_config.strategy_name}")
    print(f"Overall Validation: {'PASSED' if results.overall_validation_passed else 'FAILED'}")
    print(f"Confidence: {results.validation_confidence:.3f}")
    print(f"Recommendation: {results.recommendation}")
    print(f"Validation Time: {results.total_validation_time:.1f} seconds")

    if results.quality_warnings:
        print(f"\nQuality Warnings:")
        for warning in results.quality_warnings:
            print(f"  - {warning}")

    print(f"\nValidation methods used: {results.validation_consistency.get('methods_used', [])}")
    print(f"\nIntegration adapter demonstration completed successfully!")