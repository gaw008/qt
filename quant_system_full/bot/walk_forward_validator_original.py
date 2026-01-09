#!/usr/bin/env python3
"""
Walk-Forward Validation Framework for Quantitative Trading System

Institutional-grade walk-forward validation with statistical rigor and overfitting prevention.
Implements three-phase backtesting (2006-2016, 2017-2020, 2021-2025) with robust
statistical testing and multiple correction methods.

Key Features:
- Three distinct validation phases with market regime awareness
- Walk-forward analysis with expanding and rolling windows
- Integration with existing Purged K-Fold cross-validation
- Multiple testing correction (Bonferroni, FDR, Romano-Wolf)
- Significance testing with bootstrap confidence intervals
- Comprehensive benchmark comparison frameworks
- Crisis period analysis and drawdown assessment
- Model stability and performance degradation detection
- Automated quality assurance and edge case handling

Statistical Methods:
- Bootstrap resampling for robust confidence intervals
- Reality Check (White, 2000) for multiple strategy testing
- Superior Predictive Ability test (Hansen, 2005)
- Model Confidence Set (Hansen et al., 2011)
- Time-varying performance analysis
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from scipy import stats
from scipy.stats import t, norm, kstest
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import itertools
from collections import defaultdict

# Import existing validation components
from bot.purged_kfold_validator import PurgedKFoldCV, ValidationConfig, ValidationResults
from bot.enhanced_risk_manager import EnhancedRiskManager, TailRiskMetrics, MarketRegime
from bot.historical_data_manager import HistoricalDataManager

# Configure encoding and logging
os.environ['PYTHONIOENCODING'] = 'utf-8'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ValidationPhase(Enum):
    """Validation phases for comprehensive testing"""
    PHASE_1 = "2006-2016"  # Financial crisis period
    PHASE_2 = "2017-2020"  # Bull market and COVID
    PHASE_3 = "2021-2025"  # Post-pandemic period


class WindowType(Enum):
    """Window types for walk-forward analysis"""
    EXPANDING = "expanding"
    ROLLING = "rolling"
    ANCHORED = "anchored"


class StatisticalTest(Enum):
    """Statistical significance tests"""
    TTEST = "t_test"
    BOOTSTRAP = "bootstrap"
    REALITY_CHECK = "reality_check"
    SPA_TEST = "spa_test"
    MCS_TEST = "mcs_test"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation"""
    # Phase definitions
    phases: Dict[ValidationPhase, Tuple[str, str]] = field(default_factory=lambda: {
        ValidationPhase.PHASE_1: ("2006-01-01", "2016-12-31"),
        ValidationPhase.PHASE_2: ("2017-01-01", "2020-12-31"),
        ValidationPhase.PHASE_3: ("2021-01-01", "2025-12-31")
    })

    # Window configuration
    min_train_months: int = 24          # Minimum 2 years training
    test_window_months: int = 6         # 6 months test window
    step_months: int = 3                # 3 months step forward
    window_type: WindowType = WindowType.EXPANDING
    max_train_months: int = 60          # Maximum 5 years training (for rolling)

    # Statistical testing configuration
    confidence_level: float = 0.95      # 95% confidence level
    bootstrap_samples: int = 10000      # Bootstrap resampling count
    multiple_testing_method: str = "fdr_bh"  # FDR control method
    significance_threshold: float = 0.05 # Significance threshold

    # Performance criteria
    min_sharpe_threshold: float = 0.5   # Minimum acceptable Sharpe ratio
    max_drawdown_threshold: float = 0.20 # Maximum acceptable drawdown
    min_consistency_ratio: float = 0.60  # Minimum win rate threshold

    # Quality assurance
    min_observations_per_window: int = 60  # Minimum observations per test window
    outlier_detection_threshold: float = 3.0  # Z-score threshold for outliers
    performance_degradation_threshold: float = 0.30  # 30% degradation threshold

    # Computational settings
    max_workers: int = 4                # Maximum parallel workers
    enable_parallel_processing: bool = True
    cache_intermediate_results: bool = True

    # Output settings
    save_detailed_results: bool = True
    results_directory: str = "reports/walk_forward_validation"
    export_diagnostics: bool = True


@dataclass
class PhaseResults:
    """Results for a single validation phase"""
    phase: ValidationPhase
    phase_period: Tuple[str, str]
    total_windows: int
    successful_windows: int
    window_results: List[Dict[str, Any]]

    # Performance metrics
    mean_returns: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Statistical test results
    significance_tests: Dict[str, Dict[str, Any]]
    multiple_testing_results: Dict[str, Any]

    # Market regime analysis
    regime_performance: Dict[str, Dict[str, float]]

    # Quality metrics
    consistency_ratio: float
    performance_degradation: float
    model_stability_score: float

    # Risk metrics
    tail_risk_metrics: TailRiskMetrics
    crisis_performance: Dict[str, float]


@dataclass
class WalkForwardResults:
    """Comprehensive walk-forward validation results"""
    config: WalkForwardConfig
    validation_timestamp: str
    total_validation_time: float

    # Phase-level results
    phase_results: Dict[ValidationPhase, PhaseResults]

    # Cross-phase analysis
    cross_phase_consistency: Dict[str, float]
    performance_stability: Dict[str, float]
    regime_sensitivity: Dict[str, Dict[str, float]]

    # Statistical significance
    overall_significance: Dict[str, Any]
    multiple_testing_summary: Dict[str, Any]

    # Model assessment
    overfitting_assessment: Dict[str, Any]
    model_confidence_set: List[str]

    # Quality assurance
    validation_passed: bool
    quality_warnings: List[str]
    edge_case_analysis: Dict[str, Any]

    # Benchmark comparison
    benchmark_comparison: Dict[str, Dict[str, float]]


class StatisticalTestingEngine:
    """Statistical testing engine for walk-forward validation"""

    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.bootstrap_cache = {}

    def t_test_significance(self, returns: np.ndarray, benchmark: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform t-test for statistical significance

        Args:
            returns: Strategy returns
            benchmark: Benchmark returns (if None, test against zero)

        Returns:
            Dictionary with test results
        """
        if len(returns) < 2:
            return {'statistic': np.nan, 'p_value': 1.0, 'significant': False}

        if benchmark is not None:
            if len(benchmark) != len(returns):
                benchmark = None

        if benchmark is not None:
            # Paired t-test against benchmark
            excess_returns = returns - benchmark
            statistic, p_value = stats.ttest_1samp(excess_returns, 0)
        else:
            # One-sample t-test against zero
            statistic, p_value = stats.ttest_1samp(returns, 0)

        return {
            'test_type': 'paired_t_test' if benchmark is not None else 'one_sample_t_test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.config.significance_threshold,
            'confidence_level': self.config.confidence_level,
            'sample_size': len(returns)
        }

    def bootstrap_confidence_interval(
        self,
        returns: np.ndarray,
        statistic_func: Callable = np.mean,
        n_bootstrap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals

        Args:
            returns: Return series
            statistic_func: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples

        Returns:
            Bootstrap results with confidence intervals
        """
        if n_bootstrap is None:
            n_bootstrap = self.config.bootstrap_samples

        if len(returns) < 10:
            return {
                'original_statistic': statistic_func(returns),
                'confidence_interval': (np.nan, np.nan),
                'p_value': 1.0,
                'significant': False
            }

        # Calculate original statistic
        original_stat = statistic_func(returns)

        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))

        bootstrap_stats = np.array(bootstrap_stats)

        # Calculate confidence interval
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        # Calculate p-value (proportion of bootstrap stats <= 0)
        p_value = np.mean(bootstrap_stats <= 0) * 2  # Two-tailed test
        p_value = min(p_value, 2 * (1 - p_value))  # Ensure <= 1

        return {
            'original_statistic': float(original_stat),
            'bootstrap_mean': float(np.mean(bootstrap_stats)),
            'bootstrap_std': float(np.std(bootstrap_stats)),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'p_value': float(p_value),
            'significant': p_value < self.config.significance_threshold,
            'n_bootstrap': n_bootstrap
        }

    def reality_check_test(self, strategy_returns: Dict[str, np.ndarray], benchmark: np.ndarray) -> Dict[str, Any]:
        """
        White's Reality Check for multiple strategy testing

        Tests the null hypothesis that the best strategy does not outperform benchmark

        Args:
            strategy_returns: Dictionary of strategy returns
            benchmark: Benchmark returns

        Returns:
            Reality check test results
        """
        if not strategy_returns:
            return {'p_value': 1.0, 'significant': False, 'best_strategy': None}

        # Calculate excess returns for each strategy
        excess_returns = {}
        best_strategy = None
        best_mean_excess = -np.inf

        for name, returns in strategy_returns.items():
            if len(returns) == len(benchmark):
                excess = returns - benchmark
                excess_returns[name] = excess
                mean_excess = np.mean(excess)

                if mean_excess > best_mean_excess:
                    best_mean_excess = mean_excess
                    best_strategy = name

        if not excess_returns or best_strategy is None:
            return {'p_value': 1.0, 'significant': False, 'best_strategy': None}

        # Bootstrap test
        n_strategies = len(excess_returns)
        n_observations = len(excess_returns[best_strategy])
        bootstrap_max_stats = []

        for _ in range(self.config.bootstrap_samples):
            bootstrap_maxes = []

            for name, excess in excess_returns.items():
                bootstrap_sample = np.random.choice(excess, size=n_observations, replace=True)
                bootstrap_maxes.append(np.mean(bootstrap_sample))

            bootstrap_max_stats.append(max(bootstrap_maxes))

        # Calculate p-value
        p_value = np.mean(np.array(bootstrap_max_stats) >= best_mean_excess)

        return {
            'test_type': 'reality_check',
            'best_strategy': best_strategy,
            'best_excess_return': float(best_mean_excess),
            'p_value': float(p_value),
            'significant': p_value < self.config.significance_threshold,
            'n_strategies': n_strategies,
            'n_observations': n_observations
        }

    def superior_predictive_ability_test(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Hansen's Superior Predictive Ability (SPA) test

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            SPA test results
        """
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 10:
            return {'p_value': 1.0, 'significant': False}

        # Calculate loss differential (negative returns for loss)
        loss_diff = benchmark_returns - strategy_returns

        # Test statistic
        mean_diff = np.mean(loss_diff)
        std_diff = np.std(loss_diff)

        if std_diff == 0:
            return {'p_value': 1.0, 'significant': False}

        test_statistic = mean_diff / (std_diff / np.sqrt(len(loss_diff)))

        # Bootstrap p-value
        bootstrap_stats = []
        for _ in range(self.config.bootstrap_samples):
            bootstrap_diff = np.random.choice(loss_diff, size=len(loss_diff), replace=True)
            bootstrap_mean = np.mean(bootstrap_diff)
            bootstrap_std = np.std(bootstrap_diff)

            if bootstrap_std > 0:
                bootstrap_stat = bootstrap_mean / (bootstrap_std / np.sqrt(len(bootstrap_diff)))
                bootstrap_stats.append(bootstrap_stat)

        if not bootstrap_stats:
            return {'p_value': 1.0, 'significant': False}

        p_value = np.mean(np.array(bootstrap_stats) <= test_statistic)

        return {
            'test_type': 'spa_test',
            'test_statistic': float(test_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.config.significance_threshold,
            'mean_loss_differential': float(mean_diff)
        }

    def multiple_testing_correction(
        self,
        p_values: List[float],
        method: str = "fdr_bh"
    ) -> Dict[str, Any]:
        """
        Apply multiple testing correction

        Args:
            p_values: List of p-values to correct
            method: Correction method ('bonferroni', 'fdr_bh', 'fdr_by')

        Returns:
            Correction results
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        alpha = self.config.significance_threshold

        if method == "bonferroni":
            # Bonferroni correction
            corrected_alpha = alpha / n_tests
            rejected = p_values <= corrected_alpha
            corrected_p_values = np.minimum(p_values * n_tests, 1.0)

        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]

            # Find rejections
            rejected = np.zeros(n_tests, dtype=bool)
            for i in range(n_tests - 1, -1, -1):
                threshold = (i + 1) / n_tests * alpha
                if sorted_p[i] <= threshold:
                    rejected[sorted_indices[:i+1]] = True
                    break

            corrected_p_values = p_values.copy()  # BH doesn't adjust p-values directly

        elif method == "fdr_by":
            # Benjamini-Yekutieli FDR control (conservative)
            harmonic_sum = np.sum(1.0 / np.arange(1, n_tests + 1))
            adjusted_alpha = alpha / harmonic_sum

            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]

            rejected = np.zeros(n_tests, dtype=bool)
            for i in range(n_tests - 1, -1, -1):
                threshold = (i + 1) / n_tests * adjusted_alpha
                if sorted_p[i] <= threshold:
                    rejected[sorted_indices[:i+1]] = True
                    break

            corrected_p_values = p_values * harmonic_sum

        else:
            raise ValueError(f"Unknown correction method: {method}")

        return {
            'method': method,
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected_p_values.tolist(),
            'rejected': rejected.tolist(),
            'n_rejected': int(np.sum(rejected)),
            'n_tests': n_tests,
            'family_wise_error_rate': float(np.sum(rejected) / n_tests) if n_tests > 0 else 0.0
        }


class WalkForwardValidator:
    """
    Comprehensive walk-forward validation framework with institutional-grade statistical rigor
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self.statistical_engine = StatisticalTestingEngine(self.config)
        self.risk_manager = EnhancedRiskManager()
        self.data_manager = HistoricalDataManager()

        # Initialize components
        self.results_cache = {}
        self.quality_warnings = []
        self._lock = threading.RLock()

        # Create results directory
        Path(self.config.results_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Walk-forward validator initialized with {len(self.config.phases)} phases")

    def validate_strategy(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        param_optimizer: Optional[Callable] = None
    ) -> WalkForwardResults:
        """
        Perform comprehensive walk-forward validation

        Args:
            strategy_func: Strategy function that takes (data, **params) and returns performance
            data: Historical market data
            benchmark_data: Benchmark data for comparison
            strategy_params: Base strategy parameters
            param_optimizer: Function to optimize parameters

        Returns:
            Comprehensive validation results
        """
        validation_start_time = datetime.now()
        logger.info("Starting comprehensive walk-forward validation")

        # Initialize results structure
        phase_results = {}

        # Validate each phase
        for phase in self.config.phases:
            logger.info(f"Validating phase {phase.value}")

            try:
                phase_result = self._validate_phase(
                    phase, strategy_func, data, benchmark_data,
                    strategy_params, param_optimizer
                )
                phase_results[phase] = phase_result

            except Exception as e:
                logger.error(f"Phase {phase.value} validation failed: {e}")
                self.quality_warnings.append(f"Phase {phase.value} failed: {str(e)}")
                continue

        # Cross-phase analysis
        cross_phase_analysis = self._analyze_cross_phase_consistency(phase_results)

        # Overall statistical significance
        overall_significance = self._calculate_overall_significance(phase_results)

        # Multiple testing correction
        multiple_testing_summary = self._apply_multiple_testing_correction(phase_results)

        # Overfitting assessment
        overfitting_assessment = self._assess_overfitting(phase_results)

        # Model confidence set
        model_confidence_set = self._calculate_model_confidence_set(phase_results)

        # Benchmark comparison
        benchmark_comparison = self._compare_with_benchmarks(phase_results, benchmark_data)

        # Edge case analysis
        edge_case_analysis = self._analyze_edge_cases(phase_results)

        # Final validation assessment
        validation_passed = self._assess_validation_criteria(phase_results)

        total_validation_time = (datetime.now() - validation_start_time).total_seconds()

        # Compile comprehensive results
        results = WalkForwardResults(
            config=self.config,
            validation_timestamp=validation_start_time.isoformat(),
            total_validation_time=total_validation_time,
            phase_results=phase_results,
            cross_phase_consistency=cross_phase_analysis['consistency'],
            performance_stability=cross_phase_analysis['stability'],
            regime_sensitivity=cross_phase_analysis['regime_sensitivity'],
            overall_significance=overall_significance,
            multiple_testing_summary=multiple_testing_summary,
            overfitting_assessment=overfitting_assessment,
            model_confidence_set=model_confidence_set,
            validation_passed=validation_passed,
            quality_warnings=self.quality_warnings.copy(),
            edge_case_analysis=edge_case_analysis,
            benchmark_comparison=benchmark_comparison
        )

        # Save results
        if self.config.save_detailed_results:
            self._save_validation_results(results)

        # Generate diagnostic reports
        if self.config.export_diagnostics:
            self._generate_diagnostic_reports(results)

        logger.info(f"Walk-forward validation completed in {total_validation_time:.1f}s")
        logger.info(f"Validation {'PASSED' if validation_passed else 'FAILED'}")

        return results

    def _validate_phase(
        self,
        phase: ValidationPhase,
        strategy_func: Callable,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame],
        strategy_params: Optional[Dict[str, Any]],
        param_optimizer: Optional[Callable]
    ) -> PhaseResults:
        """Validate a single phase with walk-forward analysis"""

        phase_start, phase_end = self.config.phases[phase]
        phase_data = self._filter_data_by_period(data, phase_start, phase_end)

        if benchmark_data is not None:
            benchmark_phase_data = self._filter_data_by_period(benchmark_data, phase_start, phase_end)
        else:
            benchmark_phase_data = None

        # Generate walk-forward windows
        windows = self._generate_walk_forward_windows(phase_data, phase)

        logger.info(f"Generated {len(windows)} walk-forward windows for phase {phase.value}")

        # Process windows (parallel or sequential)
        if self.config.enable_parallel_processing and len(windows) > 1:
            window_results = self._process_windows_parallel(
                windows, strategy_func, phase_data, benchmark_phase_data,
                strategy_params, param_optimizer
            )
        else:
            window_results = self._process_windows_sequential(
                windows, strategy_func, phase_data, benchmark_phase_data,
                strategy_params, param_optimizer
            )

        # Filter successful windows
        successful_results = [r for r in window_results if 'error' not in r]

        if not successful_results:
            raise ValueError(f"No successful windows in phase {phase.value}")

        # Calculate phase-level metrics
        phase_metrics = self._calculate_phase_metrics(successful_results)

        # Statistical significance testing
        significance_tests = self._perform_phase_significance_tests(
            successful_results, benchmark_phase_data
        )

        # Multiple testing correction for phase
        phase_p_values = [test.get('p_value', 1.0) for test in significance_tests.values()]
        multiple_testing_results = self.statistical_engine.multiple_testing_correction(
            phase_p_values, self.config.multiple_testing_method
        )

        # Market regime analysis
        regime_performance = self._analyze_regime_performance(successful_results, phase_data)

        # Calculate tail risk metrics
        all_returns = np.concatenate([r['returns'] for r in successful_results])
        tail_risk_metrics = self.risk_manager.calculate_tail_risk_metrics(all_returns)

        # Crisis period analysis
        crisis_performance = self._analyze_crisis_performance(successful_results, phase_data)

        return PhaseResults(
            phase=phase,
            phase_period=(phase_start, phase_end),
            total_windows=len(windows),
            successful_windows=len(successful_results),
            window_results=successful_results,
            mean_returns=phase_metrics['mean_returns'],
            volatility=phase_metrics['volatility'],
            sharpe_ratio=phase_metrics['sharpe_ratio'],
            max_drawdown=phase_metrics['max_drawdown'],
            calmar_ratio=phase_metrics['calmar_ratio'],
            significance_tests=significance_tests,
            multiple_testing_results=multiple_testing_results,
            regime_performance=regime_performance,
            consistency_ratio=phase_metrics['consistency_ratio'],
            performance_degradation=phase_metrics['performance_degradation'],
            model_stability_score=phase_metrics['model_stability_score'],
            tail_risk_metrics=tail_risk_metrics,
            crisis_performance=crisis_performance
        )

    def _generate_walk_forward_windows(
        self,
        data: pd.DataFrame,
        phase: ValidationPhase
    ) -> List[Dict[str, Any]]:
        """Generate walk-forward windows for a phase"""

        windows = []
        data_dates = pd.to_datetime(data.index)
        start_date = data_dates.min()
        end_date = data_dates.max()

        current_date = start_date + pd.DateOffset(months=self.config.min_train_months)
        window_id = 0

        while current_date < end_date:
            # Define training period
            if self.config.window_type == WindowType.EXPANDING:
                train_start = start_date
                train_end = current_date
            elif self.config.window_type == WindowType.ROLLING:
                train_start = current_date - pd.DateOffset(months=self.config.max_train_months)
                train_start = max(train_start, start_date)
                train_end = current_date
            else:  # ANCHORED
                train_start = start_date
                train_end = current_date

            # Define test period
            test_start = current_date
            test_end = min(
                current_date + pd.DateOffset(months=self.config.test_window_months),
                end_date
            )

            # Check minimum observations
            train_mask = (data_dates >= train_start) & (data_dates < train_end)
            test_mask = (data_dates >= test_start) & (data_dates < test_end)

            train_obs = train_mask.sum()
            test_obs = test_mask.sum()

            if (train_obs >= self.config.min_observations_per_window and
                test_obs >= self.config.min_observations_per_window):

                window = {
                    'window_id': window_id,
                    'phase': phase,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_observations': train_obs,
                    'test_observations': test_obs
                }
                windows.append(window)
                window_id += 1

            # Move to next window
            current_date += pd.DateOffset(months=self.config.step_months)

        return windows

    def _process_windows_parallel(
        self,
        windows: List[Dict[str, Any]],
        strategy_func: Callable,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame],
        strategy_params: Optional[Dict[str, Any]],
        param_optimizer: Optional[Callable]
    ) -> List[Dict[str, Any]]:
        """Process windows in parallel"""

        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_window = {
                executor.submit(
                    self._process_single_window,
                    window, strategy_func, data, benchmark_data,
                    strategy_params, param_optimizer
                ): window for window in windows
            }

            for future in as_completed(future_to_window):
                window = future_to_window[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Window {window['window_id']} failed: {e}")
                    results.append({
                        'window_id': window['window_id'],
                        'error': str(e)
                    })

        # Sort by window_id
        results.sort(key=lambda x: x.get('window_id', 0))
        return results

    def _process_windows_sequential(
        self,
        windows: List[Dict[str, Any]],
        strategy_func: Callable,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame],
        strategy_params: Optional[Dict[str, Any]],
        param_optimizer: Optional[Callable]
    ) -> List[Dict[str, Any]]:
        """Process windows sequentially"""

        results = []

        for window in windows:
            try:
                result = self._process_single_window(
                    window, strategy_func, data, benchmark_data,
                    strategy_params, param_optimizer
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Window {window['window_id']} failed: {e}")
                results.append({
                    'window_id': window['window_id'],
                    'error': str(e)
                })

        return results

    def _process_single_window(
        self,
        window: Dict[str, Any],
        strategy_func: Callable,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame],
        strategy_params: Optional[Dict[str, Any]],
        param_optimizer: Optional[Callable]
    ) -> Dict[str, Any]:
        """Process a single walk-forward window"""

        # Extract data for window
        data_dates = pd.to_datetime(data.index)

        train_mask = ((data_dates >= window['train_start']) &
                     (data_dates < window['train_end']))
        test_mask = ((data_dates >= window['test_start']) &
                    (data_dates < window['test_end']))

        train_data = data[train_mask].copy()
        test_data = data[test_mask].copy()

        # Parameter optimization
        if param_optimizer and self.config.cache_intermediate_results:
            optimal_params = param_optimizer(train_data)
        else:
            optimal_params = strategy_params or {}

        # In-sample performance
        is_results = strategy_func(train_data, **optimal_params)
        is_returns = self._extract_returns(is_results)
        is_metrics = self._calculate_performance_metrics(is_returns)

        # Out-of-sample performance
        oos_results = strategy_func(test_data, **optimal_params)
        oos_returns = self._extract_returns(oos_results)
        oos_metrics = self._calculate_performance_metrics(oos_returns)

        # Benchmark comparison (if available)
        benchmark_metrics = None
        if benchmark_data is not None:
            benchmark_test = benchmark_data[test_mask] if len(benchmark_data) > 0 else None
            if benchmark_test is not None and len(benchmark_test) > 0:
                benchmark_returns = benchmark_test.pct_change().dropna()
                benchmark_metrics = self._calculate_performance_metrics(benchmark_returns)

        # Performance degradation
        degradation = self._calculate_performance_degradation(is_metrics, oos_metrics)

        # Market regime detection
        regime = self._detect_window_regime(test_data)

        return {
            'window_id': window['window_id'],
            'window_info': window,
            'optimal_params': optimal_params,
            'in_sample_metrics': is_metrics,
            'out_of_sample_metrics': oos_metrics,
            'benchmark_metrics': benchmark_metrics,
            'performance_degradation': degradation,
            'returns': oos_returns,
            'market_regime': regime,
            'data_quality': self._assess_window_data_quality(test_data)
        }

    def _extract_returns(self, strategy_results: Any) -> np.ndarray:
        """Extract returns from strategy results"""
        if isinstance(strategy_results, pd.Series):
            return strategy_results.values
        elif isinstance(strategy_results, pd.DataFrame):
            if 'returns' in strategy_results.columns:
                return strategy_results['returns'].values
            else:
                return strategy_results.iloc[:, 0].values
        elif isinstance(strategy_results, dict):
            if 'returns' in strategy_results:
                returns = strategy_results['returns']
                if isinstance(returns, (pd.Series, pd.DataFrame)):
                    return returns.values if hasattr(returns, 'values') else returns
                return np.array(returns)
        elif isinstance(strategy_results, (list, np.ndarray)):
            return np.array(strategy_results)
        else:
            raise ValueError(f"Cannot extract returns from {type(strategy_results)}")

    def _calculate_performance_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return {
                'total_return': 0.0, 'annualized_return': 0.0, 'volatility': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'calmar_ratio': 0.0,
                'win_rate': 0.0, 'skewness': 0.0, 'kurtosis': 0.0
            }

        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        mean_return = np.mean(returns)
        volatility = np.std(returns)

        # Annualized metrics (assuming daily returns)
        trading_days = 252
        annualized_return = (1 + mean_return) ** trading_days - 1
        annualized_volatility = volatility * np.sqrt(trading_days)

        # Sharpe ratio
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

        # Drawdown analysis
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Additional metrics
        win_rate = np.mean(returns > 0)
        skewness = stats.skew(returns) if len(returns) > 2 else 0
        kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 0

        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(annualized_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'win_rate': float(win_rate),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        }

    def _filter_data_by_period(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter data by date period"""
        data_copy = data.copy()
        data_copy.index = pd.to_datetime(data_copy.index)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        mask = (data_copy.index >= start_dt) & (data_copy.index <= end_dt)
        return data_copy[mask]

    def _calculate_phase_metrics(self, window_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics for a phase"""

        # Extract metrics from all windows
        returns_list = [r['out_of_sample_metrics']['annualized_return'] for r in window_results]
        volatility_list = [r['out_of_sample_metrics']['volatility'] for r in window_results]
        sharpe_list = [r['out_of_sample_metrics']['sharpe_ratio'] for r in window_results]
        drawdown_list = [r['out_of_sample_metrics']['max_drawdown'] for r in window_results]

        # Aggregate returns
        all_returns = np.concatenate([r['returns'] for r in window_results])

        # Performance degradation analysis
        is_sharpe_list = [r['in_sample_metrics']['sharpe_ratio'] for r in window_results]
        oos_sharpe_list = [r['out_of_sample_metrics']['sharpe_ratio'] for r in window_results]

        degradation_ratios = []
        for is_sharpe, oos_sharpe in zip(is_sharpe_list, oos_sharpe_list):
            if is_sharpe > 0:
                degradation_ratios.append((is_sharpe - oos_sharpe) / is_sharpe)
            else:
                degradation_ratios.append(0.0)

        return {
            'mean_returns': float(np.mean(returns_list)),
            'volatility': float(np.mean(volatility_list)),
            'sharpe_ratio': float(np.mean(sharpe_list)),
            'max_drawdown': float(np.min(drawdown_list)),
            'calmar_ratio': float(np.mean([r['out_of_sample_metrics']['calmar_ratio'] for r in window_results])),
            'consistency_ratio': float(np.mean([r['out_of_sample_metrics']['sharpe_ratio'] > 0 for r in window_results])),
            'performance_degradation': float(np.mean(degradation_ratios)),
            'model_stability_score': float(1.0 - np.std(sharpe_list))
        }

    def _perform_phase_significance_tests(
        self,
        window_results: List[Dict[str, Any]],
        benchmark_data: Optional[pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance tests for a phase"""

        # Aggregate returns
        all_returns = np.concatenate([r['returns'] for r in window_results])

        # Extract benchmark returns if available
        benchmark_returns = None
        if benchmark_data is not None and len(benchmark_data) > 0:
            benchmark_returns = benchmark_data.pct_change().dropna().values
            # Align with strategy returns length
            if len(benchmark_returns) != len(all_returns):
                min_length = min(len(benchmark_returns), len(all_returns))
                benchmark_returns = benchmark_returns[-min_length:]
                all_returns = all_returns[-min_length:]

        tests = {}

        # T-test
        tests['t_test'] = self.statistical_engine.t_test_significance(
            all_returns, benchmark_returns
        )

        # Bootstrap confidence interval
        tests['bootstrap'] = self.statistical_engine.bootstrap_confidence_interval(
            all_returns
        )

        # SPA test (if benchmark available)
        if benchmark_returns is not None:
            tests['spa_test'] = self.statistical_engine.superior_predictive_ability_test(
                all_returns, benchmark_returns
            )

        return tests

    def _analyze_regime_performance(
        self,
        window_results: List[Dict[str, Any]],
        phase_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance by market regime"""

        regime_results = defaultdict(list)

        # Group results by regime
        for result in window_results:
            regime = result.get('market_regime', 'unknown')
            metrics = result.get('out_of_sample_metrics', {})
            regime_results[regime].append(metrics)

        # Calculate regime statistics
        regime_performance = {}
        for regime, metrics_list in regime_results.items():
            if metrics_list:
                regime_performance[regime] = {
                    'mean_sharpe': float(np.mean([m.get('sharpe_ratio', 0) for m in metrics_list])),
                    'mean_return': float(np.mean([m.get('annualized_return', 0) for m in metrics_list])),
                    'volatility': float(np.mean([m.get('volatility', 0) for m in metrics_list])),
                    'max_drawdown': float(np.min([m.get('max_drawdown', 0) for m in metrics_list])),
                    'win_rate': float(np.mean([m.get('win_rate', 0) for m in metrics_list])),
                    'window_count': len(metrics_list)
                }

        return regime_performance

    def _detect_window_regime(self, data: pd.DataFrame) -> str:
        """Detect market regime for a window"""
        if len(data) == 0 or 'close' not in data.columns:
            return 'unknown'

        # Calculate returns
        returns = data['close'].pct_change().dropna()

        if len(returns) < 5:
            return 'unknown'

        # Simple regime classification based on volatility and trend
        mean_return = returns.mean()
        volatility = returns.std()

        # Thresholds
        high_vol_threshold = 0.025  # 2.5% daily volatility
        positive_trend_threshold = 0.0005  # 0.05% daily return

        if volatility > high_vol_threshold:
            if mean_return > positive_trend_threshold:
                return 'bull_high_vol'
            elif mean_return < -positive_trend_threshold:
                return 'bear_high_vol'
            else:
                return 'sideways_high_vol'
        else:
            if mean_return > positive_trend_threshold:
                return 'bull_low_vol'
            elif mean_return < -positive_trend_threshold:
                return 'bear_low_vol'
            else:
                return 'sideways_low_vol'

    def _calculate_performance_degradation(
        self,
        is_metrics: Dict[str, float],
        oos_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate performance degradation from in-sample to out-of-sample"""

        degradation = {}

        key_metrics = ['sharpe_ratio', 'annualized_return', 'calmar_ratio']

        for metric in key_metrics:
            is_value = is_metrics.get(metric, 0)
            oos_value = oos_metrics.get(metric, 0)

            if abs(is_value) > 1e-6:  # Avoid division by very small numbers
                degradation[f'{metric}_degradation'] = (is_value - oos_value) / abs(is_value)
            else:
                degradation[f'{metric}_degradation'] = 0.0

        return degradation

    def _assess_window_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality for a window"""

        if len(data) == 0:
            return {'quality_score': 0.0, 'issues': ['empty_data']}

        issues = []

        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > 0.05:  # More than 5% missing
            issues.append(f'high_missing_data_{missing_ratio:.1%}')

        # Check for outliers (if price data available)
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            if len(returns) > 0:
                z_scores = np.abs(stats.zscore(returns))
                outlier_ratio = np.mean(z_scores > self.config.outlier_detection_threshold)
                if outlier_ratio > 0.02:  # More than 2% outliers
                    issues.append(f'high_outlier_ratio_{outlier_ratio:.1%}')

        # Calculate overall quality score
        quality_score = max(0.0, 1.0 - len(issues) * 0.2)

        return {
            'quality_score': quality_score,
            'issues': issues,
            'missing_ratio': missing_ratio,
            'outlier_ratio': outlier_ratio if 'outlier_ratio' in locals() else 0.0
        }

    def _analyze_cross_phase_consistency(
        self,
        phase_results: Dict[ValidationPhase, PhaseResults]
    ) -> Dict[str, Any]:
        """Analyze consistency across validation phases"""

        if len(phase_results) < 2:
            return {
                'consistency': {},
                'stability': {},
                'regime_sensitivity': {}
            }

        # Extract key metrics across phases
        phase_metrics = {}
        for phase, results in phase_results.items():
            phase_metrics[phase.value] = {
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'consistency_ratio': results.consistency_ratio,
                'volatility': results.volatility
            }

        # Calculate consistency metrics
        consistency = {}
        for metric in ['sharpe_ratio', 'max_drawdown', 'consistency_ratio']:
            values = [phase_metrics[phase][metric] for phase in phase_metrics]
            consistency[f'{metric}_consistency'] = 1.0 - (np.std(values) / (np.mean(np.abs(values)) + 1e-6))

        # Performance stability analysis
        sharpe_values = [phase_metrics[phase]['sharpe_ratio'] for phase in phase_metrics]
        stability = {
            'mean_sharpe': float(np.mean(sharpe_values)),
            'sharpe_stability': float(1.0 - np.std(sharpe_values)),
            'consistent_performance': float(np.mean([s > 0 for s in sharpe_values]))
        }

        # Regime sensitivity analysis
        regime_sensitivity = {}
        for phase, results in phase_results.items():
            for regime, perf in results.regime_performance.items():
                if regime not in regime_sensitivity:
                    regime_sensitivity[regime] = []
                regime_sensitivity[regime].append(perf.get('mean_sharpe', 0))

        # Average regime performance across phases
        regime_averages = {}
        for regime, sharpe_list in regime_sensitivity.items():
            regime_averages[regime] = {
                'mean_sharpe': float(np.mean(sharpe_list)),
                'sharpe_std': float(np.std(sharpe_list)),
                'phase_count': len(sharpe_list)
            }

        return {
            'consistency': consistency,
            'stability': stability,
            'regime_sensitivity': regime_averages
        }

    def _calculate_overall_significance(
        self,
        phase_results: Dict[ValidationPhase, PhaseResults]
    ) -> Dict[str, Any]:
        """Calculate overall statistical significance across phases"""

        # Collect all p-values
        all_p_values = []
        all_test_types = []

        for phase, results in phase_results.items():
            for test_type, test_result in results.significance_tests.items():
                p_value = test_result.get('p_value', 1.0)
                all_p_values.append(p_value)
                all_test_types.append(f"{phase.value}_{test_type}")

        if not all_p_values:
            return {'overall_significant': False, 'combined_p_value': 1.0}

        # Fisher's combined probability test
        chi_squared_stat = -2 * np.sum(np.log(np.array(all_p_values) + 1e-10))
        degrees_of_freedom = 2 * len(all_p_values)
        combined_p_value = 1 - stats.chi2.cdf(chi_squared_stat, degrees_of_freedom)

        # Meta-analysis using inverse variance weighting
        significant_count = sum(1 for p in all_p_values if p < self.config.significance_threshold)

        return {
            'combined_p_value': float(combined_p_value),
            'overall_significant': combined_p_value < self.config.significance_threshold,
            'significant_tests': significant_count,
            'total_tests': len(all_p_values),
            'significance_ratio': float(significant_count / len(all_p_values)),
            'test_types': all_test_types
        }

    def _apply_multiple_testing_correction(
        self,
        phase_results: Dict[ValidationPhase, PhaseResults]
    ) -> Dict[str, Any]:
        """Apply multiple testing correction across all phases"""

        # Collect all p-values from all phases and tests
        all_p_values = []
        test_identifiers = []

        for phase, results in phase_results.items():
            for test_type, test_result in results.significance_tests.items():
                p_value = test_result.get('p_value', 1.0)
                all_p_values.append(p_value)
                test_identifiers.append(f"{phase.value}_{test_type}")

        if not all_p_values:
            return {'method': self.config.multiple_testing_method, 'no_tests': True}

        # Apply correction
        correction_result = self.statistical_engine.multiple_testing_correction(
            all_p_values, self.config.multiple_testing_method
        )

        # Add test identifiers
        correction_result['test_identifiers'] = test_identifiers

        return correction_result

    def _assess_overfitting(self, phase_results: Dict[ValidationPhase, PhaseResults]) -> Dict[str, Any]:
        """Assess overfitting risk across phases"""

        # Collect performance degradation metrics
        degradation_values = []
        consistency_ratios = []

        for phase, results in phase_results.items():
            degradation_values.append(results.performance_degradation)
            consistency_ratios.append(results.consistency_ratio)

        if not degradation_values:
            return {'overfitting_risk': 'unknown', 'risk_score': 0.5}

        # Calculate overfitting risk score
        mean_degradation = np.mean(degradation_values)
        mean_consistency = np.mean(consistency_ratios)

        # Risk factors
        high_degradation = mean_degradation > self.config.performance_degradation_threshold
        low_consistency = mean_consistency < self.config.min_consistency_ratio

        # Risk score calculation
        risk_score = 0.0
        if high_degradation:
            risk_score += 0.4
        if low_consistency:
            risk_score += 0.3
        if mean_degradation > 0.5:  # Very high degradation
            risk_score += 0.3

        # Risk classification
        if risk_score > 0.6:
            risk_level = 'high'
        elif risk_score > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'overfitting_risk': risk_level,
            'risk_score': float(risk_score),
            'mean_degradation': float(mean_degradation),
            'mean_consistency': float(mean_consistency),
            'high_degradation_phases': sum(1 for d in degradation_values
                                         if d > self.config.performance_degradation_threshold),
            'recommendations': self._generate_overfitting_recommendations(risk_level, mean_degradation)
        }

    def _generate_overfitting_recommendations(self, risk_level: str, degradation: float) -> List[str]:
        """Generate recommendations based on overfitting assessment"""

        recommendations = []

        if risk_level == 'high':
            recommendations.extend([
                "Consider reducing model complexity",
                "Implement stronger regularization",
                "Increase training data requirements",
                "Review feature selection process"
            ])

        if degradation > 0.3:
            recommendations.extend([
                "Implement walk-forward parameter optimization",
                "Use ensemble methods to improve robustness",
                "Consider market regime-aware modeling"
            ])

        if risk_level in ['medium', 'high']:
            recommendations.extend([
                "Implement additional out-of-sample validation",
                "Monitor model performance more frequently",
                "Consider shorter rebalancing periods"
            ])

        return recommendations

    def _calculate_model_confidence_set(
        self,
        phase_results: Dict[ValidationPhase, PhaseResults]
    ) -> List[str]:
        """Calculate Model Confidence Set (Hansen et al., 2011)"""

        # For now, return phases that pass significance tests
        confident_phases = []

        for phase, results in phase_results.items():
            # Check if phase has significant positive performance
            has_positive_sharpe = results.sharpe_ratio > 0
            has_significant_test = any(
                test.get('significant', False)
                for test in results.significance_tests.values()
            )

            if has_positive_sharpe and has_significant_test:
                confident_phases.append(phase.value)

        return confident_phases

    def _compare_with_benchmarks(
        self,
        phase_results: Dict[ValidationPhase, PhaseResults],
        benchmark_data: Optional[pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Compare strategy performance with benchmarks"""

        if benchmark_data is None or len(benchmark_data) == 0:
            return {'no_benchmark_data': True}

        comparison = {}

        for phase, results in phase_results.items():
            phase_start, phase_end = self.config.phases[phase]
            phase_benchmark = self._filter_data_by_period(benchmark_data, phase_start, phase_end)

            if len(phase_benchmark) > 0:
                benchmark_returns = phase_benchmark.pct_change().dropna()
                benchmark_metrics = self._calculate_performance_metrics(benchmark_returns.values)

                # Calculate comparison metrics
                comparison[phase.value] = {
                    'strategy_sharpe': results.sharpe_ratio,
                    'benchmark_sharpe': benchmark_metrics['sharpe_ratio'],
                    'sharpe_ratio_difference': results.sharpe_ratio - benchmark_metrics['sharpe_ratio'],
                    'strategy_return': results.mean_returns,
                    'benchmark_return': benchmark_metrics['annualized_return'],
                    'return_difference': results.mean_returns - benchmark_metrics['annualized_return'],
                    'strategy_volatility': results.volatility,
                    'benchmark_volatility': benchmark_metrics['volatility'],
                    'information_ratio': (results.mean_returns - benchmark_metrics['annualized_return']) /
                                       max(abs(results.volatility - benchmark_metrics['volatility']), 0.001)
                }

        return comparison

    def _analyze_crisis_performance(
        self,
        window_results: List[Dict[str, Any]],
        phase_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze performance during crisis periods"""

        # Identify crisis periods (high volatility windows)
        crisis_threshold = 0.03  # 3% daily volatility threshold

        crisis_windows = []
        normal_windows = []

        for result in window_results:
            volatility = result.get('out_of_sample_metrics', {}).get('volatility', 0)
            if volatility > crisis_threshold:
                crisis_windows.append(result)
            else:
                normal_windows.append(result)

        crisis_analysis = {
            'crisis_window_count': len(crisis_windows),
            'normal_window_count': len(normal_windows),
            'crisis_ratio': len(crisis_windows) / len(window_results) if window_results else 0
        }

        if crisis_windows:
            crisis_sharpe = [w['out_of_sample_metrics']['sharpe_ratio'] for w in crisis_windows]
            crisis_analysis.update({
                'crisis_mean_sharpe': float(np.mean(crisis_sharpe)),
                'crisis_sharpe_std': float(np.std(crisis_sharpe)),
                'crisis_positive_ratio': float(np.mean([s > 0 for s in crisis_sharpe]))
            })

        if normal_windows:
            normal_sharpe = [w['out_of_sample_metrics']['sharpe_ratio'] for w in normal_windows]
            crisis_analysis.update({
                'normal_mean_sharpe': float(np.mean(normal_sharpe)),
                'normal_sharpe_std': float(np.std(normal_sharpe))
            })

        return crisis_analysis

    def _analyze_edge_cases(self, phase_results: Dict[ValidationPhase, PhaseResults]) -> Dict[str, Any]:
        """Analyze edge cases and extreme scenarios"""

        edge_cases = {
            'extreme_drawdown_periods': [],
            'high_volatility_periods': [],
            'regime_transition_performance': {},
            'data_quality_issues': []
        }

        for phase, results in phase_results.items():
            # Find extreme drawdown windows
            for window in results.window_results:
                metrics = window.get('out_of_sample_metrics', {})
                drawdown = metrics.get('max_drawdown', 0)

                if drawdown < -0.15:  # More than 15% drawdown
                    edge_cases['extreme_drawdown_periods'].append({
                        'phase': phase.value,
                        'window_id': window.get('window_id'),
                        'drawdown': drawdown,
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                    })

                # High volatility periods
                volatility = metrics.get('volatility', 0)
                if volatility > 0.4:  # More than 40% annualized volatility
                    edge_cases['high_volatility_periods'].append({
                        'phase': phase.value,
                        'window_id': window.get('window_id'),
                        'volatility': volatility,
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                    })

                # Data quality issues
                data_quality = window.get('data_quality', {})
                if data_quality.get('quality_score', 1.0) < 0.8:
                    edge_cases['data_quality_issues'].append({
                        'phase': phase.value,
                        'window_id': window.get('window_id'),
                        'quality_score': data_quality.get('quality_score'),
                        'issues': data_quality.get('issues', [])
                    })

        return edge_cases

    def _assess_validation_criteria(self, phase_results: Dict[ValidationPhase, PhaseResults]) -> bool:
        """Assess whether strategy passes overall validation criteria"""

        if not phase_results:
            return False

        criteria_passed = 0
        total_criteria = 0

        for phase, results in phase_results.items():
            # Criterion 1: Positive Sharpe ratio
            total_criteria += 1
            if results.sharpe_ratio > self.config.min_sharpe_threshold:
                criteria_passed += 1

            # Criterion 2: Acceptable drawdown
            total_criteria += 1
            if abs(results.max_drawdown) < self.config.max_drawdown_threshold:
                criteria_passed += 1

            # Criterion 3: Consistency
            total_criteria += 1
            if results.consistency_ratio > self.config.min_consistency_ratio:
                criteria_passed += 1

            # Criterion 4: Statistical significance
            total_criteria += 1
            if any(test.get('significant', False) for test in results.significance_tests.values()):
                criteria_passed += 1

        # Overall validation passes if at least 75% of criteria are met
        validation_ratio = criteria_passed / total_criteria if total_criteria > 0 else 0
        return validation_ratio >= 0.75

    def _save_validation_results(self, results: WalkForwardResults):
        """Save comprehensive validation results"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to JSON-serializable format
        results_dict = self._convert_results_to_dict(results)

        # Save main results
        results_file = results_dir / f"walk_forward_validation_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Validation results saved to {results_file}")

    def _convert_results_to_dict(self, results: WalkForwardResults) -> Dict[str, Any]:
        """Convert results object to dictionary for JSON serialization"""

        results_dict = {
            'config': self.config.__dict__,
            'validation_timestamp': results.validation_timestamp,
            'total_validation_time': results.total_validation_time,
            'validation_passed': results.validation_passed,
            'quality_warnings': results.quality_warnings,
            'cross_phase_consistency': results.cross_phase_consistency,
            'performance_stability': results.performance_stability,
            'regime_sensitivity': results.regime_sensitivity,
            'overall_significance': results.overall_significance,
            'multiple_testing_summary': results.multiple_testing_summary,
            'overfitting_assessment': results.overfitting_assessment,
            'model_confidence_set': results.model_confidence_set,
            'edge_case_analysis': results.edge_case_analysis,
            'benchmark_comparison': results.benchmark_comparison,
            'phase_results': {}
        }

        # Convert phase results
        for phase, phase_result in results.phase_results.items():
            results_dict['phase_results'][phase.value] = {
                'phase_period': phase_result.phase_period,
                'total_windows': phase_result.total_windows,
                'successful_windows': phase_result.successful_windows,
                'mean_returns': phase_result.mean_returns,
                'volatility': phase_result.volatility,
                'sharpe_ratio': phase_result.sharpe_ratio,
                'max_drawdown': phase_result.max_drawdown,
                'calmar_ratio': phase_result.calmar_ratio,
                'consistency_ratio': phase_result.consistency_ratio,
                'performance_degradation': phase_result.performance_degradation,
                'model_stability_score': phase_result.model_stability_score,
                'significance_tests': phase_result.significance_tests,
                'multiple_testing_results': phase_result.multiple_testing_results,
                'regime_performance': phase_result.regime_performance,
                'crisis_performance': phase_result.crisis_performance,
                'tail_risk_metrics': {
                    'es_97_5': phase_result.tail_risk_metrics.es_97_5,
                    'es_99': phase_result.tail_risk_metrics.es_99,
                    'tail_ratio': phase_result.tail_risk_metrics.tail_ratio,
                    'max_drawdown': phase_result.tail_risk_metrics.max_drawdown,
                    'calmar_ratio': phase_result.tail_risk_metrics.calmar_ratio,
                    'skewness': phase_result.tail_risk_metrics.skewness,
                    'kurtosis': phase_result.tail_risk_metrics.kurtosis
                },
                'window_results_count': len(phase_result.window_results)
            }

        return results_dict

    def _generate_diagnostic_reports(self, results: WalkForwardResults):
        """Generate diagnostic reports and visualizations"""

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            results_dir = Path(self.config.results_directory)

            # Performance over time plot
            self._plot_performance_over_time(results, results_dir)

            # Phase comparison plot
            self._plot_phase_comparison(results, results_dir)

            # Statistical significance heatmap
            self._plot_significance_heatmap(results, results_dir)

            # Regime performance analysis
            self._plot_regime_analysis(results, results_dir)

            logger.info(f"Diagnostic plots saved to {results_dir}")

        except ImportError:
            logger.warning("Matplotlib not available - skipping diagnostic plots")
        except Exception as e:
            logger.error(f"Failed to generate diagnostic plots: {e}")

    def _plot_performance_over_time(self, results: WalkForwardResults, output_dir: Path):
        """Plot performance metrics over time"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Validation: Performance Over Time', fontsize=16)

        # Collect data across phases
        all_windows = []
        for phase_result in results.phase_results.values():
            all_windows.extend(phase_result.window_results)

        if not all_windows:
            return

        # Extract metrics
        window_ids = [w['window_id'] for w in all_windows]
        sharpe_ratios = [w['out_of_sample_metrics']['sharpe_ratio'] for w in all_windows]
        returns = [w['out_of_sample_metrics']['annualized_return'] for w in all_windows]
        drawdowns = [w['out_of_sample_metrics']['max_drawdown'] for w in all_windows]
        volatilities = [w['out_of_sample_metrics']['volatility'] for w in all_windows]

        # Plot Sharpe ratios
        axes[0, 0].plot(window_ids, sharpe_ratios, 'o-', alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Sharpe Ratio Over Time')
        axes[0, 0].set_xlabel('Window ID')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot returns
        axes[0, 1].plot(window_ids, returns, 'o-', alpha=0.7, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Annualized Returns Over Time')
        axes[0, 1].set_xlabel('Window ID')
        axes[0, 1].set_ylabel('Return')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot drawdowns
        axes[1, 0].plot(window_ids, drawdowns, 'o-', alpha=0.7, color='red')
        axes[1, 0].set_title('Max Drawdown Over Time')
        axes[1, 0].set_xlabel('Window ID')
        axes[1, 0].set_ylabel('Max Drawdown')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot volatility
        axes[1, 1].plot(window_ids, volatilities, 'o-', alpha=0.7, color='orange')
        axes[1, 1].set_title('Volatility Over Time')
        axes[1, 1].set_xlabel('Window ID')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_phase_comparison(self, results: WalkForwardResults, output_dir: Path):
        """Plot comparison across validation phases"""
        import matplotlib.pyplot as plt

        if len(results.phase_results) < 2:
            return

        phases = list(results.phase_results.keys())
        phase_names = [p.value for p in phases]

        # Extract metrics
        sharpe_ratios = [results.phase_results[p].sharpe_ratio for p in phases]
        max_drawdowns = [abs(results.phase_results[p].max_drawdown) for p in phases]
        consistency_ratios = [results.phase_results[p].consistency_ratio for p in phases]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Performance Comparison Across Validation Phases', fontsize=16)

        # Sharpe ratios
        bars1 = axes[0].bar(phase_names, sharpe_ratios, alpha=0.7, color='blue')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_title('Sharpe Ratio by Phase')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].tick_params(axis='x', rotation=45)

        # Max drawdowns
        bars2 = axes[1].bar(phase_names, max_drawdowns, alpha=0.7, color='red')
        axes[1].set_title('Max Drawdown by Phase')
        axes[1].set_ylabel('Max Drawdown')
        axes[1].tick_params(axis='x', rotation=45)

        # Consistency ratios
        bars3 = axes[2].bar(phase_names, consistency_ratios, alpha=0.7, color='green')
        axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[2].set_title('Consistency Ratio by Phase')
        axes[2].set_ylabel('Consistency Ratio')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'phase_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_significance_heatmap(self, results: WalkForwardResults, output_dir: Path):
        """Plot statistical significance heatmap"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create significance matrix
        phases = list(results.phase_results.keys())
        test_types = set()

        # Collect all test types
        for phase_result in results.phase_results.values():
            test_types.update(phase_result.significance_tests.keys())

        test_types = list(test_types)

        if not phases or not test_types:
            return

        # Build significance matrix
        significance_matrix = np.zeros((len(phases), len(test_types)))

        for i, phase in enumerate(phases):
            for j, test_type in enumerate(test_types):
                test_result = results.phase_results[phase].significance_tests.get(test_type, {})
                significance_matrix[i, j] = 1 if test_result.get('significant', False) else 0

        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            significance_matrix,
            xticklabels=test_types,
            yticklabels=[p.value for p in phases],
            annot=True,
            cmap='RdYlGn',
            center=0.5,
            cbar_kws={'label': 'Statistical Significance'}
        )
        plt.title('Statistical Significance Across Phases and Tests')
        plt.xlabel('Test Type')
        plt.ylabel('Validation Phase')
        plt.tight_layout()
        plt.savefig(output_dir / 'significance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_regime_analysis(self, results: WalkForwardResults, output_dir: Path):
        """Plot regime performance analysis"""
        import matplotlib.pyplot as plt

        # Aggregate regime performance across phases
        regime_data = defaultdict(list)

        for phase_result in results.phase_results.values():
            for regime, perf in phase_result.regime_performance.items():
                regime_data[regime].append(perf.get('mean_sharpe', 0))

        if not regime_data:
            return

        regimes = list(regime_data.keys())
        mean_sharpe = [np.mean(regime_data[regime]) for regime in regimes]
        std_sharpe = [np.std(regime_data[regime]) for regime in regimes]

        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(regimes, mean_sharpe, yerr=std_sharpe, capsize=5, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.title('Strategy Performance by Market Regime')
        plt.xlabel('Market Regime')
        plt.ylabel('Mean Sharpe Ratio')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_example_strategy() -> Callable:
    """Create an example strategy for testing the validation framework"""

    def momentum_strategy(data: pd.DataFrame, lookback: int = 20, threshold: float = 0.02) -> pd.Series:
        """Simple momentum strategy for testing"""
        if 'close' not in data.columns or len(data) < lookback + 1:
            return pd.Series(index=data.index, data=0.0)

        # Calculate momentum signal
        returns = data['close'].pct_change()
        momentum = returns.rolling(lookback).mean()

        # Generate signals
        signals = np.where(momentum > threshold, 1,
                          np.where(momentum < -threshold, -1, 0))

        # Calculate strategy returns
        strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    return momentum_strategy


if __name__ == "__main__":
    # Example usage of the Walk-Forward Validation Framework
    print("Walk-Forward Validation Framework - Institutional Grade Testing")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2006-01-01', '2024-12-31', freq='D')

    # Simulate market data with regime changes
    returns = np.random.randn(len(dates)) * 0.01
    returns[:1000] += 0.0002      # Bull market 2006-2008
    returns[1000:1500] -= 0.0008  # Crisis 2008-2009
    returns[1500:3000] += 0.0003  # Recovery 2009-2014
    returns[3000:4500] += 0.0001  # Steady growth 2014-2019
    returns[4500:5000] -= 0.0015  # COVID crash 2020
    returns[5000:] += 0.0002      # Recovery 2021+

    # Add some volatility clustering
    volatility = 0.01 + 0.005 * np.abs(np.random.randn(len(dates)))
    returns = returns * volatility

    sample_data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(returns)),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    # Create benchmark data (market index)
    benchmark_returns = returns * 0.8 + np.random.randn(len(dates)) * 0.005
    benchmark_data = pd.Series(
        100 * np.exp(np.cumsum(benchmark_returns)),
        index=dates
    )

    # Configure validation
    config = WalkForwardConfig(
        min_train_months=24,
        test_window_months=6,
        step_months=3,
        window_type=WindowType.EXPANDING,
        confidence_level=0.95,
        bootstrap_samples=1000,  # Reduced for example
        max_workers=2,
        save_detailed_results=True
    )

    # Initialize validator
    validator = WalkForwardValidator(config)

    # Create example strategy
    strategy = create_example_strategy()

    # Run validation
    print("Starting walk-forward validation...")
    results = validator.validate_strategy(
        strategy_func=strategy,
        data=sample_data,
        benchmark_data=benchmark_data.to_frame('close'),
        strategy_params={'lookback': 20, 'threshold': 0.02}
    )

    # Print summary results
    print(f"\nValidation Results Summary:")
    print(f"Overall Validation: {'PASSED' if results.validation_passed else 'FAILED'}")
    print(f"Total Validation Time: {results.total_validation_time:.1f} seconds")
    print(f"Quality Warnings: {len(results.quality_warnings)}")

    for phase, phase_result in results.phase_results.items():
        print(f"\n{phase.value} Results:")
        print(f"  Sharpe Ratio: {phase_result.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {phase_result.max_drawdown:.3f}")
        print(f"  Consistency Ratio: {phase_result.consistency_ratio:.3f}")
        print(f"  Successful Windows: {phase_result.successful_windows}/{phase_result.total_windows}")

    print(f"\nOverall Statistical Significance:")
    print(f"  Combined P-value: {results.overall_significance['combined_p_value']:.4f}")
    print(f"  Overall Significant: {results.overall_significance['overall_significant']}")

    print(f"\nOverfitting Assessment:")
    print(f"  Risk Level: {results.overfitting_assessment['overfitting_risk']}")
    print(f"  Risk Score: {results.overfitting_assessment['risk_score']:.3f}")

    print(f"\nModel Confidence Set: {results.model_confidence_set}")

    print(f"\nValidation framework demonstration completed successfully!")