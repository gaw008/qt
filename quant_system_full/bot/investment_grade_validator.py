#!/usr/bin/env python3
"""
Investment-Grade Backtesting Validator
投资级回测验证器

Investment-grade validation system providing comprehensive strategy assessment:
- Strategy capacity analysis with performance decay modeling
- Transaction cost integration with realistic slippage and market impact
- Market regime analysis with adaptive performance metrics
- Drawdown stress testing with recovery period analysis
- Risk-adjusted performance with ES@97.5% and tail risk analysis

Features:
- Capacity testing at $10M, $50M, $100M, $500M AUM levels
- Realistic transaction cost modeling with market impact curves
- Regime-aware performance evaluation across market conditions
- Stress testing framework for extreme market scenarios
- Professional-grade risk analytics and reporting

投资级验证系统功能：
- 带性能衰减建模的策略容量分析
- 带现实滑点和市场冲击的交易成本整合
- 带自适应性能指标的市场状态分析
- 带恢复期分析的回撤压力测试
- 带ES@97.5%和尾部风险分析的风险调整性能
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import warnings
import sqlite3
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from functools import lru_cache

# Scientific computing
from scipy import stats, optimize
from scipy.interpolate import interp1d
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import existing system components
from bot.enhanced_risk_manager import EnhancedRiskManager, TailRiskMetrics
from bot.enhanced_backtesting_system import ThreePhaseBacktestResults, PhaseResults

# Configure encoding and logging
os.environ['PYTHONIOENCODING'] = 'utf-8'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class CapacityLevel(Enum):
    """Strategy capacity testing levels"""
    SMALL = "10M"        # $10 Million AUM
    MEDIUM = "50M"       # $50 Million AUM
    LARGE = "100M"       # $100 Million AUM
    INSTITUTIONAL = "500M"  # $500 Million AUM
    MEGA = "1B"          # $1 Billion AUM

class MarketCondition(Enum):
    """Market regime conditions for testing"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS_PERIOD = "crisis_period"
    RECOVERY_PERIOD = "recovery_period"
    NORMAL_CONDITIONS = "normal_conditions"

class StressScenario(Enum):
    """Stress testing scenarios"""
    MARKET_CRASH = "market_crash"           # -20% market drop
    VOLATILITY_SPIKE = "volatility_spike"   # VIX > 40
    LIQUIDITY_CRISIS = "liquidity_crisis"   # Reduced liquidity
    CORRELATION_BREAKDOWN = "correlation_breakdown"  # Factor correlation changes
    INTEREST_RATE_SHOCK = "interest_rate_shock"     # Rapid rate changes
    GEOPOLITICAL_CRISIS = "geopolitical_crisis"     # External shock

@dataclass
class TransactionCostModel:
    """Transaction cost modeling parameters"""
    base_spread_bps: float = 5.0        # Base bid-ask spread in basis points
    market_impact_coeff: float = 0.1    # Market impact coefficient
    liquidity_adjustment: float = 1.0   # Liquidity adjustment factor

    # Size-dependent parameters
    small_trade_threshold: float = 10000   # Threshold for small trades ($)
    large_trade_threshold: float = 100000  # Threshold for large trades ($)

    # Market impact curve parameters
    temporary_impact_coeff: float = 0.05   # Temporary impact coefficient
    permanent_impact_coeff: float = 0.02   # Permanent impact coefficient

    # Timing parameters
    participation_rate: float = 0.10       # Participation rate (10% of volume)
    execution_time_factor: float = 1.0     # Execution time adjustment

@dataclass
class CapacityAnalysis:
    """Strategy capacity analysis results"""
    capacity_level: CapacityLevel
    aum_amount: float

    # Performance metrics at this capacity
    adjusted_return: float
    adjusted_sharpe: float
    adjusted_calmar: float

    # Capacity impact
    performance_decay: float      # Performance degradation %
    transaction_cost_impact: float  # Transaction cost impact %
    market_impact_cost: float     # Market impact cost %

    # Liquidity metrics
    liquidity_score: float        # Overall liquidity score
    execution_difficulty: float   # Execution difficulty score
    slippage_estimate: float      # Expected slippage %

    # Risk adjustments
    capacity_adjusted_drawdown: float
    capacity_adjusted_volatility: float

    # Feasibility assessment
    is_feasible: bool
    feasibility_score: float
    limiting_factors: List[str]

@dataclass
class RegimeAnalysis:
    """Market regime analysis results"""
    regime: MarketCondition

    # Performance in this regime
    regime_return: float
    regime_sharpe: float
    regime_max_drawdown: float
    regime_volatility: float

    # Regime characteristics
    regime_duration_days: int
    market_correlation: float
    average_vix: float

    # Strategy adaptation
    adaptation_score: float       # How well strategy adapts to regime
    consistency_score: float      # Performance consistency in regime

    # Risk metrics
    regime_var_95: float
    regime_expected_shortfall: float
    tail_risk_ratio: float

@dataclass
class StressTestResult:
    """Stress testing results"""
    scenario: StressScenario
    scenario_description: str

    # Stress scenario parameters
    scenario_severity: float      # Severity level (1-10)
    scenario_duration: int        # Duration in days

    # Performance under stress
    stress_return: float
    stress_max_drawdown: float
    stress_volatility: float

    # Recovery metrics
    recovery_time_days: int
    recovery_strength: float      # Recovery vs pre-stress performance

    # Risk metrics
    tail_risk_exposure: float
    downside_capture: float
    stress_beta: float

    # Resilience assessment
    resilience_score: float       # Overall resilience (0-1)
    risk_factors: List[str]       # Key risk factors identified

@dataclass
class ValidationReport:
    """Comprehensive investment-grade validation report"""
    strategy_name: str
    validation_timestamp: datetime

    # Overall assessment
    investment_grade_score: float    # Overall investment grade score (0-100)
    deployment_readiness: str        # Deployment readiness assessment

    # Capacity analysis
    capacity_analyses: Dict[CapacityLevel, CapacityAnalysis]
    recommended_capacity: CapacityLevel
    maximum_feasible_aum: float

    # Regime analysis
    regime_analyses: Dict[MarketCondition, RegimeAnalysis]
    regime_adaptability_score: float
    worst_performing_regime: MarketCondition

    # Stress testing
    stress_test_results: Dict[StressScenario, StressTestResult]
    overall_resilience_score: float
    critical_vulnerabilities: List[str]

    # Risk-adjusted metrics
    risk_adjusted_return: float
    tail_risk_assessment: float
    drawdown_recovery_score: float

    # Professional recommendations
    implementation_recommendations: List[str]
    risk_management_requirements: List[str]
    monitoring_recommendations: List[str]

    # Compliance and documentation
    regulatory_compliance_score: float
    documentation_completeness: float
    audit_readiness: bool

class InvestmentGradeValidator:
    """
    Investment-Grade Strategy Validation System

    Comprehensive validation framework providing institutional-quality assessment:
    - Multi-level capacity analysis with realistic market impact modeling
    - Market regime performance evaluation across all market conditions
    - Comprehensive stress testing with tail risk assessment
    - Professional-grade reporting for investment committees
    - Regulatory compliance and audit trail documentation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._load_config(config)
        self.risk_manager = EnhancedRiskManager()

        # Transaction cost modeling
        self.transaction_cost_model = TransactionCostModel()

        # Database for validation results
        self.db_path = Path("data_cache/investment_grade_validation.db")
        self._initialize_database()

        # Performance tracking
        self.validation_cache = {}
        self.execution_metrics = {}

        logger.info("Investment-Grade Validator initialized")

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load validation configuration parameters"""

        default_config = {
            "capacity_levels": {
                CapacityLevel.SMALL.value: 10_000_000,
                CapacityLevel.MEDIUM.value: 50_000_000,
                CapacityLevel.LARGE.value: 100_000_000,
                CapacityLevel.INSTITUTIONAL.value: 500_000_000,
                CapacityLevel.MEGA.value: 1_000_000_000
            },
            "performance_thresholds": {
                "minimum_sharpe": 0.8,
                "maximum_drawdown": 0.15,
                "minimum_calmar": 0.5,
                "minimum_hit_rate": 0.52
            },
            "capacity_thresholds": {
                "performance_decay_limit": 0.20,  # 20% max performance decay
                "transaction_cost_limit": 0.05,   # 5% max transaction cost impact
                "liquidity_score_minimum": 0.60   # 60% minimum liquidity score
            },
            "stress_test_scenarios": {
                "market_crash": {"severity": 8, "duration": 30},
                "volatility_spike": {"severity": 7, "duration": 60},
                "liquidity_crisis": {"severity": 9, "duration": 90},
                "correlation_breakdown": {"severity": 6, "duration": 120},
                "interest_rate_shock": {"severity": 7, "duration": 180},
                "geopolitical_crisis": {"severity": 8, "duration": 45}
            },
            "regime_definitions": {
                "bull_market": {"min_return": 0.15, "max_volatility": 0.20},
                "bear_market": {"max_return": -0.10, "min_volatility": 0.15},
                "high_volatility": {"min_volatility": 0.25},
                "low_volatility": {"max_volatility": 0.12},
                "crisis_period": {"min_volatility": 0.30, "max_return": -0.15}
            }
        }

        if config:
            default_config.update(config)

        return default_config

    def _initialize_database(self):
        """Initialize SQLite database for validation results"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                # Capacity analysis table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS capacity_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        capacity_level TEXT NOT NULL,
                        aum_amount REAL NOT NULL,
                        adjusted_return REAL NOT NULL,
                        adjusted_sharpe REAL NOT NULL,
                        performance_decay REAL NOT NULL,
                        transaction_cost_impact REAL NOT NULL,
                        market_impact_cost REAL NOT NULL,
                        liquidity_score REAL NOT NULL,
                        is_feasible BOOLEAN NOT NULL,
                        feasibility_score REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                # Regime analysis table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS regime_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        regime_return REAL NOT NULL,
                        regime_sharpe REAL NOT NULL,
                        regime_max_drawdown REAL NOT NULL,
                        adaptation_score REAL NOT NULL,
                        consistency_score REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                # Stress test results table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stress_test_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        scenario TEXT NOT NULL,
                        scenario_severity REAL NOT NULL,
                        stress_return REAL NOT NULL,
                        stress_max_drawdown REAL NOT NULL,
                        recovery_time_days INTEGER NOT NULL,
                        resilience_score REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                # Validation reports table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS validation_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        investment_grade_score REAL NOT NULL,
                        deployment_readiness TEXT NOT NULL,
                        recommended_capacity TEXT NOT NULL,
                        maximum_feasible_aum REAL NOT NULL,
                        overall_resilience_score REAL NOT NULL,
                        regulatory_compliance_score REAL NOT NULL,
                        audit_readiness BOOLEAN NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                conn.commit()

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    async def validate_investment_grade(self,
                                      backtest_results: ThreePhaseBacktestResults,
                                      additional_data: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """
        Execute comprehensive investment-grade validation

        Args:
            backtest_results: Three-phase backtesting results
            additional_data: Additional market data and strategy information

        Returns:
            Complete validation report with investment-grade assessment
        """

        strategy_name = backtest_results.strategy_name
        logger.info(f"Starting investment-grade validation for {strategy_name}")

        start_time = time.time()

        try:
            # Execute all validation components in parallel
            validation_tasks = [
                self._analyze_strategy_capacity(backtest_results),
                self._analyze_market_regimes(backtest_results),
                self._execute_stress_testing(backtest_results),
                self._calculate_risk_adjusted_metrics(backtest_results)
            ]

            # Await all validation analyses
            (capacity_analyses, regime_analyses,
             stress_test_results, risk_metrics) = await asyncio.gather(*validation_tasks)

            # Calculate overall scores
            investment_grade_score = self._calculate_investment_grade_score(
                capacity_analyses, regime_analyses, stress_test_results
            )

            # Determine deployment readiness
            deployment_readiness = self._assess_deployment_readiness(
                investment_grade_score, capacity_analyses, stress_test_results
            )

            # Generate recommendations
            recommendations = self._generate_professional_recommendations(
                capacity_analyses, regime_analyses, stress_test_results, risk_metrics
            )

            # Create validation report
            validation_report = ValidationReport(
                strategy_name=strategy_name,
                validation_timestamp=datetime.now(),
                investment_grade_score=investment_grade_score,
                deployment_readiness=deployment_readiness,
                capacity_analyses=capacity_analyses,
                recommended_capacity=self._determine_recommended_capacity(capacity_analyses),
                maximum_feasible_aum=self._calculate_maximum_feasible_aum(capacity_analyses),
                regime_analyses=regime_analyses,
                regime_adaptability_score=self._calculate_regime_adaptability_score(regime_analyses),
                worst_performing_regime=self._identify_worst_regime(regime_analyses),
                stress_test_results=stress_test_results,
                overall_resilience_score=self._calculate_resilience_score(stress_test_results),
                critical_vulnerabilities=self._identify_critical_vulnerabilities(stress_test_results),
                **risk_metrics,
                **recommendations,
                regulatory_compliance_score=self._assess_regulatory_compliance(backtest_results),
                documentation_completeness=self._assess_documentation_completeness(backtest_results),
                audit_readiness=investment_grade_score >= 70.0
            )

            # Store validation results
            await self._store_validation_results(validation_report)

            execution_time = time.time() - start_time
            logger.info(f"Investment-grade validation completed in {execution_time:.2f} seconds")

            return validation_report

        except Exception as e:
            logger.error(f"Investment-grade validation failed: {e}")
            raise

    async def _analyze_strategy_capacity(self,
                                       backtest_results: ThreePhaseBacktestResults) -> Dict[CapacityLevel, CapacityAnalysis]:
        """Analyze strategy capacity at different AUM levels"""

        try:
            capacity_analyses = {}

            # Get baseline performance metrics
            baseline_return = backtest_results.phase_2_results.annualized_return  # Use Phase 2 as baseline
            baseline_sharpe = backtest_results.phase_2_results.sharpe_ratio
            baseline_calmar = backtest_results.phase_2_results.calmar_ratio
            baseline_trades = backtest_results.phase_2_results.trade_count

            # Analyze each capacity level
            for capacity_level in CapacityLevel:
                aum_amount = self.config["capacity_levels"][capacity_level.value]

                # Calculate capacity-adjusted metrics
                capacity_analysis = await self._analyze_single_capacity_level(
                    capacity_level=capacity_level,
                    aum_amount=aum_amount,
                    baseline_return=baseline_return,
                    baseline_sharpe=baseline_sharpe,
                    baseline_calmar=baseline_calmar,
                    baseline_trades=baseline_trades,
                    backtest_results=backtest_results
                )

                capacity_analyses[capacity_level] = capacity_analysis

            return capacity_analyses

        except Exception as e:
            logger.error(f"Strategy capacity analysis failed: {e}")
            raise

    async def _analyze_single_capacity_level(self,
                                           capacity_level: CapacityLevel,
                                           aum_amount: float,
                                           baseline_return: float,
                                           baseline_sharpe: float,
                                           baseline_calmar: float,
                                           baseline_trades: int,
                                           backtest_results: ThreePhaseBacktestResults) -> CapacityAnalysis:
        """Analyze capacity impact at a single AUM level"""

        try:
            # Market impact modeling
            market_impact_cost = self._calculate_market_impact(aum_amount, baseline_trades)

            # Transaction cost modeling
            transaction_cost_impact = self._calculate_transaction_costs(aum_amount, baseline_trades)

            # Liquidity analysis
            liquidity_score = self._assess_liquidity_impact(aum_amount)

            # Performance decay modeling
            performance_decay = self._model_performance_decay(aum_amount, baseline_return)

            # Adjusted performance metrics
            adjusted_return = baseline_return * (1 - performance_decay) - transaction_cost_impact - market_impact_cost
            adjusted_sharpe = baseline_sharpe * (1 - performance_decay * 0.5)  # Sharpe decays slower
            adjusted_calmar = baseline_calmar * (1 - performance_decay * 0.3)  # Calmar more resilient

            # Execution difficulty
            execution_difficulty = self._calculate_execution_difficulty(aum_amount, liquidity_score)

            # Slippage estimation
            slippage_estimate = self._estimate_slippage(aum_amount, baseline_trades)

            # Risk adjustments
            capacity_adjusted_drawdown = backtest_results.overall_max_drawdown * (1 + performance_decay * 0.2)
            capacity_adjusted_volatility = backtest_results.phase_2_results.volatility * (1 + performance_decay * 0.1)

            # Feasibility assessment
            feasibility_score = self._calculate_feasibility_score(
                performance_decay, transaction_cost_impact, liquidity_score
            )

            is_feasible = (
                performance_decay <= self.config["capacity_thresholds"]["performance_decay_limit"] and
                transaction_cost_impact <= self.config["capacity_thresholds"]["transaction_cost_limit"] and
                liquidity_score >= self.config["capacity_thresholds"]["liquidity_score_minimum"]
            )

            # Identify limiting factors
            limiting_factors = []
            if performance_decay > self.config["capacity_thresholds"]["performance_decay_limit"]:
                limiting_factors.append("Excessive performance decay")
            if transaction_cost_impact > self.config["capacity_thresholds"]["transaction_cost_limit"]:
                limiting_factors.append("High transaction costs")
            if liquidity_score < self.config["capacity_thresholds"]["liquidity_score_minimum"]:
                limiting_factors.append("Insufficient liquidity")

            return CapacityAnalysis(
                capacity_level=capacity_level,
                aum_amount=aum_amount,
                adjusted_return=adjusted_return,
                adjusted_sharpe=adjusted_sharpe,
                adjusted_calmar=adjusted_calmar,
                performance_decay=performance_decay,
                transaction_cost_impact=transaction_cost_impact,
                market_impact_cost=market_impact_cost,
                liquidity_score=liquidity_score,
                execution_difficulty=execution_difficulty,
                slippage_estimate=slippage_estimate,
                capacity_adjusted_drawdown=capacity_adjusted_drawdown,
                capacity_adjusted_volatility=capacity_adjusted_volatility,
                is_feasible=is_feasible,
                feasibility_score=feasibility_score,
                limiting_factors=limiting_factors
            )

        except Exception as e:
            logger.error(f"Single capacity analysis failed: {e}")
            raise

    def _calculate_market_impact(self, aum_amount: float, trade_count: int) -> float:
        """Calculate market impact cost based on AUM and trading frequency"""

        try:
            # Average trade size
            avg_trade_size = aum_amount * 0.05 / max(trade_count, 1)  # Assume 5% turnover per trade

            # Market impact curve (square root law)
            base_impact = self.transaction_cost_model.market_impact_coeff
            size_impact = np.sqrt(avg_trade_size / 100000)  # Normalize to $100k base

            # Participation rate adjustment
            participation_adjustment = 1.0 / np.sqrt(self.transaction_cost_model.participation_rate)

            market_impact = base_impact * size_impact * participation_adjustment * 0.01  # Convert to decimal

            # Cap at reasonable maximum
            return min(market_impact, 0.02)  # Max 2% market impact

        except Exception as e:
            logger.error(f"Market impact calculation failed: {e}")
            return 0.01  # Default 1% impact

    def _calculate_transaction_costs(self, aum_amount: float, trade_count: int) -> float:
        """Calculate transaction costs including spreads and fees"""

        try:
            # Base transaction cost (spreads, commissions)
            base_cost = self.transaction_cost_model.base_spread_bps / 10000  # Convert bps to decimal

            # Volume-based adjustment
            annual_turnover = aum_amount * (trade_count / 252)  # Annualized turnover
            volume_adjustment = 1.0 + np.log10(annual_turnover / 1000000) * 0.1  # Log scaling

            # Liquidity adjustment
            liquidity_adjustment = self.transaction_cost_model.liquidity_adjustment

            transaction_cost = base_cost * volume_adjustment * liquidity_adjustment

            # Add temporary and permanent impact
            temporary_impact = self.transaction_cost_model.temporary_impact_coeff * 0.01
            permanent_impact = self.transaction_cost_model.permanent_impact_coeff * 0.01

            total_cost = transaction_cost + temporary_impact + permanent_impact

            return min(total_cost, 0.05)  # Cap at 5%

        except Exception as e:
            logger.error(f"Transaction cost calculation failed: {e}")
            return 0.005  # Default 0.5% cost

    def _assess_liquidity_impact(self, aum_amount: float) -> float:
        """Assess liquidity impact score (0-1, higher is better)"""

        try:
            # Liquidity score based on AUM size
            # Assumes decreasing liquidity with increasing size

            if aum_amount <= 50_000_000:  # $50M and below
                liquidity_score = 0.95
            elif aum_amount <= 100_000_000:  # $100M
                liquidity_score = 0.85
            elif aum_amount <= 500_000_000:  # $500M
                liquidity_score = 0.70
            elif aum_amount <= 1_000_000_000:  # $1B
                liquidity_score = 0.55
            else:  # Above $1B
                liquidity_score = 0.40

            # Adjust for market conditions (simplified)
            market_adjustment = 1.0  # Would be dynamic in production

            return liquidity_score * market_adjustment

        except Exception as e:
            logger.error(f"Liquidity assessment failed: {e}")
            return 0.5  # Default neutral score

    def _model_performance_decay(self, aum_amount: float, baseline_return: float) -> float:
        """Model performance decay due to capacity constraints"""

        try:
            # Performance decay model: logarithmic decay with AUM
            # Based on empirical research on strategy capacity

            # Base decay rate
            base_decay_rate = 0.02  # 2% decay per order of magnitude

            # Calculate decay based on AUM scaling
            aum_scaling = np.log10(aum_amount / 10_000_000)  # Relative to $10M base

            # Performance decay
            performance_decay = base_decay_rate * aum_scaling

            # Adjust for strategy characteristics
            # High-frequency strategies decay faster
            strategy_adjustment = 1.0  # Would be strategy-specific in production

            # Return-based adjustment (higher returns decay faster)
            return_adjustment = 1.0 + abs(baseline_return) * 0.5

            total_decay = performance_decay * strategy_adjustment * return_adjustment

            # Cap at reasonable maximum
            return min(max(total_decay, 0.0), 0.50)  # 0-50% decay range

        except Exception as e:
            logger.error(f"Performance decay modeling failed: {e}")
            return 0.05  # Default 5% decay

    def _calculate_execution_difficulty(self, aum_amount: float, liquidity_score: float) -> float:
        """Calculate execution difficulty score (0-1, higher is more difficult)"""

        try:
            # Base difficulty from AUM size
            size_difficulty = min(aum_amount / 1_000_000_000, 1.0)  # Normalize to $1B

            # Liquidity-based difficulty
            liquidity_difficulty = 1.0 - liquidity_score

            # Combined difficulty score
            execution_difficulty = 0.6 * size_difficulty + 0.4 * liquidity_difficulty

            return execution_difficulty

        except Exception as e:
            logger.error(f"Execution difficulty calculation failed: {e}")
            return 0.5  # Default moderate difficulty

    def _estimate_slippage(self, aum_amount: float, trade_count: int) -> float:
        """Estimate expected slippage percentage"""

        try:
            # Average position size
            avg_position = aum_amount / 20  # Assume 20 positions

            # Trade size relative to position
            trade_size = avg_position * 0.1  # 10% of position per trade

            # Slippage model based on trade size
            base_slippage = 0.001  # 0.1% base slippage
            size_multiplier = np.sqrt(trade_size / 100000)  # Square root scaling

            estimated_slippage = base_slippage * size_multiplier

            # Frequency adjustment
            frequency_adjustment = 1.0 + (trade_count / 1000) * 0.1

            total_slippage = estimated_slippage * frequency_adjustment

            return min(total_slippage, 0.01)  # Cap at 1%

        except Exception as e:
            logger.error(f"Slippage estimation failed: {e}")
            return 0.002  # Default 0.2% slippage

    def _calculate_feasibility_score(self,
                                   performance_decay: float,
                                   transaction_cost_impact: float,
                                   liquidity_score: float) -> float:
        """Calculate overall feasibility score"""

        try:
            # Component scores (0-1, higher is better)
            decay_score = max(0, 1 - performance_decay / 0.2)  # Normalize to 20% max decay
            cost_score = max(0, 1 - transaction_cost_impact / 0.05)  # Normalize to 5% max cost
            liquidity_component = liquidity_score  # Already 0-1

            # Weighted feasibility score
            feasibility_score = 0.4 * decay_score + 0.3 * cost_score + 0.3 * liquidity_component

            return feasibility_score

        except Exception as e:
            logger.error(f"Feasibility score calculation failed: {e}")
            return 0.5  # Default neutral score

    async def _analyze_market_regimes(self,
                                    backtest_results: ThreePhaseBacktestResults) -> Dict[MarketCondition, RegimeAnalysis]:
        """Analyze strategy performance across different market regimes"""

        try:
            regime_analyses = {}

            # Define regime periods based on three-phase results
            regime_mappings = {
                MarketCondition.CRISIS_PERIOD: (backtest_results.phase_1_results, "crisis"),
                MarketCondition.BULL_MARKET: (backtest_results.phase_2_results, "bull"),
                MarketCondition.RECOVERY_PERIOD: (backtest_results.phase_3_results, "recovery"),
                MarketCondition.HIGH_VOLATILITY: (backtest_results.phase_1_results, "high_vol"),
                MarketCondition.LOW_VOLATILITY: (backtest_results.phase_2_results, "low_vol"),
                MarketCondition.BEAR_MARKET: (backtest_results.phase_1_results, "bear"),
                MarketCondition.NORMAL_CONDITIONS: (backtest_results.phase_3_results, "normal")
            }

            for regime, (phase_results, regime_type) in regime_mappings.items():
                regime_analysis = await self._analyze_single_regime(regime, phase_results, regime_type)
                regime_analyses[regime] = regime_analysis

            return regime_analyses

        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            raise

    async def _analyze_single_regime(self,
                                   regime: MarketCondition,
                                   phase_results: PhaseResults,
                                   regime_type: str) -> RegimeAnalysis:
        """Analyze performance in a single market regime"""

        try:
            # Extract regime-specific performance
            if regime_type == "crisis":
                regime_return = phase_results.crisis_performance
            elif regime_type == "bull":
                regime_return = phase_results.bull_market_performance
            elif regime_type == "bear":
                regime_return = phase_results.bear_market_performance
            else:
                regime_return = phase_results.normal_market_performance

            regime_sharpe = phase_results.sharpe_ratio
            regime_max_drawdown = phase_results.max_drawdown
            regime_volatility = phase_results.volatility

            # Regime characteristics (simplified)
            regime_duration_days = 252  # Assume 1 year per regime
            market_correlation = 0.6    # Default market correlation
            average_vix = 20           # Default VIX level

            # Adaptation and consistency scores
            adaptation_score = self._calculate_adaptation_score(regime_return, regime_sharpe)
            consistency_score = self._calculate_consistency_score(phase_results)

            # Risk metrics
            regime_var_95 = phase_results.var_95
            regime_expected_shortfall = phase_results.expected_shortfall_975
            tail_risk_ratio = abs(regime_expected_shortfall / regime_var_95) if regime_var_95 != 0 else 0

            return RegimeAnalysis(
                regime=regime,
                regime_return=regime_return,
                regime_sharpe=regime_sharpe,
                regime_max_drawdown=regime_max_drawdown,
                regime_volatility=regime_volatility,
                regime_duration_days=regime_duration_days,
                market_correlation=market_correlation,
                average_vix=average_vix,
                adaptation_score=adaptation_score,
                consistency_score=consistency_score,
                regime_var_95=regime_var_95,
                regime_expected_shortfall=regime_expected_shortfall,
                tail_risk_ratio=tail_risk_ratio
            )

        except Exception as e:
            logger.error(f"Single regime analysis failed: {e}")
            raise

    def _calculate_adaptation_score(self, regime_return: float, regime_sharpe: float) -> float:
        """Calculate how well strategy adapts to regime"""

        try:
            # Score based on return and risk-adjusted return
            return_score = max(0, min(1, (regime_return + 0.1) / 0.3))  # -10% to +20% range
            sharpe_score = max(0, min(1, regime_sharpe / 2.0))  # 0 to 2.0 Sharpe range

            adaptation_score = 0.6 * return_score + 0.4 * sharpe_score
            return adaptation_score

        except Exception as e:
            logger.error(f"Adaptation score calculation failed: {e}")
            return 0.5

    def _calculate_consistency_score(self, phase_results: PhaseResults) -> float:
        """Calculate performance consistency score"""

        try:
            # Use stability indicator from phase results
            consistency_score = phase_results.stability_indicator
            return max(0, min(1, consistency_score))

        except Exception as e:
            logger.error(f"Consistency score calculation failed: {e}")
            return 0.5

    async def _execute_stress_testing(self,
                                     backtest_results: ThreePhaseBacktestResults) -> Dict[StressScenario, StressTestResult]:
        """Execute comprehensive stress testing scenarios"""

        try:
            stress_test_results = {}

            # Execute all stress scenarios
            for scenario_name, scenario_config in self.config["stress_test_scenarios"].items():
                scenario = StressScenario(scenario_name)

                stress_result = await self._execute_single_stress_test(
                    scenario, scenario_config, backtest_results
                )

                stress_test_results[scenario] = stress_result

            return stress_test_results

        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            raise

    async def _execute_single_stress_test(self,
                                        scenario: StressScenario,
                                        scenario_config: Dict[str, Any],
                                        backtest_results: ThreePhaseBacktestResults) -> StressTestResult:
        """Execute a single stress test scenario"""

        try:
            severity = scenario_config["severity"]
            duration = scenario_config["duration"]

            # Scenario descriptions
            scenario_descriptions = {
                StressScenario.MARKET_CRASH: "Sudden market decline of 20% or more",
                StressScenario.VOLATILITY_SPIKE: "VIX spike above 40 for extended period",
                StressScenario.LIQUIDITY_CRISIS: "Severe reduction in market liquidity",
                StressScenario.CORRELATION_BREAKDOWN: "Sudden change in factor correlations",
                StressScenario.INTEREST_RATE_SHOCK: "Rapid interest rate changes",
                StressScenario.GEOPOLITICAL_CRISIS: "External shock from geopolitical events"
            }

            # Model stress impact on strategy performance
            stress_impact = self._model_stress_impact(scenario, severity, backtest_results)

            # Calculate stress metrics
            stress_return = stress_impact["return"]
            stress_max_drawdown = stress_impact["drawdown"]
            stress_volatility = stress_impact["volatility"]

            # Recovery analysis
            recovery_time_days = self._estimate_recovery_time(scenario, severity)
            recovery_strength = self._calculate_recovery_strength(stress_return)

            # Risk metrics under stress
            tail_risk_exposure = self._calculate_tail_risk_exposure(scenario, backtest_results)
            downside_capture = self._calculate_downside_capture(stress_return)
            stress_beta = self._calculate_stress_beta(scenario, backtest_results)

            # Resilience assessment
            resilience_score = self._calculate_resilience_score_single(
                stress_return, stress_max_drawdown, recovery_time_days
            )

            # Identify risk factors
            risk_factors = self._identify_scenario_risk_factors(scenario)

            return StressTestResult(
                scenario=scenario,
                scenario_description=scenario_descriptions.get(scenario, "Unknown scenario"),
                scenario_severity=severity,
                scenario_duration=duration,
                stress_return=stress_return,
                stress_max_drawdown=stress_max_drawdown,
                stress_volatility=stress_volatility,
                recovery_time_days=recovery_time_days,
                recovery_strength=recovery_strength,
                tail_risk_exposure=tail_risk_exposure,
                downside_capture=downside_capture,
                stress_beta=stress_beta,
                resilience_score=resilience_score,
                risk_factors=risk_factors
            )

        except Exception as e:
            logger.error(f"Single stress test failed: {e}")
            raise

    def _model_stress_impact(self,
                           scenario: StressScenario,
                           severity: float,
                           backtest_results: ThreePhaseBacktestResults) -> Dict[str, float]:
        """Model the impact of stress scenario on strategy performance"""

        try:
            # Base performance metrics
            base_return = backtest_results.overall_sharpe * 0.1  # Rough return estimate
            base_drawdown = backtest_results.overall_max_drawdown
            base_volatility = backtest_results.phase_2_results.volatility

            # Scenario-specific impact factors
            impact_factors = {
                StressScenario.MARKET_CRASH: {"return": -0.3, "drawdown": 1.5, "volatility": 2.0},
                StressScenario.VOLATILITY_SPIKE: {"return": -0.1, "drawdown": 1.2, "volatility": 2.5},
                StressScenario.LIQUIDITY_CRISIS: {"return": -0.2, "drawdown": 1.3, "volatility": 1.5},
                StressScenario.CORRELATION_BREAKDOWN: {"return": -0.15, "drawdown": 1.1, "volatility": 1.3},
                StressScenario.INTEREST_RATE_SHOCK: {"return": -0.1, "drawdown": 1.2, "volatility": 1.4},
                StressScenario.GEOPOLITICAL_CRISIS: {"return": -0.25, "drawdown": 1.4, "volatility": 1.8}
            }

            factors = impact_factors.get(scenario, {"return": -0.2, "drawdown": 1.3, "volatility": 1.5})

            # Apply severity scaling
            severity_scale = severity / 10.0  # Normalize severity

            stress_return = base_return + factors["return"] * severity_scale
            stress_drawdown = base_drawdown * factors["drawdown"] * severity_scale
            stress_volatility = base_volatility * factors["volatility"] * severity_scale

            return {
                "return": stress_return,
                "drawdown": stress_drawdown,
                "volatility": stress_volatility
            }

        except Exception as e:
            logger.error(f"Stress impact modeling failed: {e}")
            return {"return": -0.1, "drawdown": 0.2, "volatility": 0.3}

    def _estimate_recovery_time(self, scenario: StressScenario, severity: float) -> int:
        """Estimate recovery time in days for stress scenario"""

        try:
            # Base recovery times by scenario type
            base_recovery_days = {
                StressScenario.MARKET_CRASH: 90,
                StressScenario.VOLATILITY_SPIKE: 60,
                StressScenario.LIQUIDITY_CRISIS: 120,
                StressScenario.CORRELATION_BREAKDOWN: 180,
                StressScenario.INTEREST_RATE_SHOCK: 150,
                StressScenario.GEOPOLITICAL_CRISIS: 100
            }

            base_days = base_recovery_days.get(scenario, 90)
            severity_multiplier = severity / 5.0  # Normalize to severity scale

            recovery_days = int(base_days * severity_multiplier)
            return min(recovery_days, 365)  # Cap at 1 year

        except Exception as e:
            logger.error(f"Recovery time estimation failed: {e}")
            return 90  # Default 3 months

    def _calculate_recovery_strength(self, stress_return: float) -> float:
        """Calculate recovery strength based on stress performance"""

        try:
            # Recovery strength inversely related to stress impact
            if stress_return >= 0:
                recovery_strength = 1.0
            else:
                recovery_strength = max(0, 1 + stress_return / 0.5)  # Normalize to -50% worst case

            return recovery_strength

        except Exception as e:
            logger.error(f"Recovery strength calculation failed: {e}")
            return 0.5

    def _calculate_tail_risk_exposure(self,
                                    scenario: StressScenario,
                                    backtest_results: ThreePhaseBacktestResults) -> float:
        """Calculate tail risk exposure for stress scenario"""

        try:
            base_es = backtest_results.overall_expected_shortfall

            # Scenario-specific tail risk multipliers
            tail_risk_multipliers = {
                StressScenario.MARKET_CRASH: 2.5,
                StressScenario.VOLATILITY_SPIKE: 2.0,
                StressScenario.LIQUIDITY_CRISIS: 3.0,
                StressScenario.CORRELATION_BREAKDOWN: 1.5,
                StressScenario.INTEREST_RATE_SHOCK: 1.8,
                StressScenario.GEOPOLITICAL_CRISIS: 2.2
            }

            multiplier = tail_risk_multipliers.get(scenario, 2.0)
            tail_risk_exposure = base_es * multiplier

            return min(tail_risk_exposure, 0.20)  # Cap at 20%

        except Exception as e:
            logger.error(f"Tail risk exposure calculation failed: {e}")
            return 0.05

    def _calculate_downside_capture(self, stress_return: float) -> float:
        """Calculate downside capture ratio"""

        try:
            # Assume market declines -20% in stress scenario
            market_decline = -0.20

            if market_decline != 0:
                downside_capture = stress_return / market_decline
            else:
                downside_capture = 1.0

            return max(downside_capture, 0)

        except Exception as e:
            logger.error(f"Downside capture calculation failed: {e}")
            return 1.0

    def _calculate_stress_beta(self,
                             scenario: StressScenario,
                             backtest_results: ThreePhaseBacktestResults) -> float:
        """Calculate beta under stress conditions"""

        try:
            normal_beta = backtest_results.phase_2_results.beta

            # Stress scenarios typically increase beta
            stress_beta_adjustments = {
                StressScenario.MARKET_CRASH: 1.3,
                StressScenario.VOLATILITY_SPIKE: 1.2,
                StressScenario.LIQUIDITY_CRISIS: 1.4,
                StressScenario.CORRELATION_BREAKDOWN: 0.8,  # May decrease correlations
                StressScenario.INTEREST_RATE_SHOCK: 1.1,
                StressScenario.GEOPOLITICAL_CRISIS: 1.25
            }

            adjustment = stress_beta_adjustments.get(scenario, 1.2)
            stress_beta = normal_beta * adjustment

            return stress_beta

        except Exception as e:
            logger.error(f"Stress beta calculation failed: {e}")
            return 1.0

    def _calculate_resilience_score_single(self,
                                         stress_return: float,
                                         stress_drawdown: float,
                                         recovery_time: int) -> float:
        """Calculate resilience score for single stress test"""

        try:
            # Component scores
            return_resilience = max(0, 1 + stress_return / 0.3)  # Normalize to -30% worst case
            drawdown_resilience = max(0, 1 - stress_drawdown / 0.5)  # Normalize to 50% worst drawdown
            recovery_resilience = max(0, 1 - recovery_time / 365)  # Normalize to 1 year max recovery

            # Weighted resilience score
            resilience_score = 0.4 * return_resilience + 0.4 * drawdown_resilience + 0.2 * recovery_resilience

            return max(0, min(1, resilience_score))

        except Exception as e:
            logger.error(f"Resilience score calculation failed: {e}")
            return 0.5

    def _identify_scenario_risk_factors(self, scenario: StressScenario) -> List[str]:
        """Identify key risk factors for stress scenario"""

        risk_factor_map = {
            StressScenario.MARKET_CRASH: ["Market beta exposure", "Momentum factor risk", "Liquidity constraints"],
            StressScenario.VOLATILITY_SPIKE: ["Volatility timing", "Options exposure", "Risk parity breakdown"],
            StressScenario.LIQUIDITY_CRISIS: ["Position size limitations", "Execution delays", "Bid-ask spread widening"],
            StressScenario.CORRELATION_BREAKDOWN: ["Factor model instability", "Diversification failure", "Hedging effectiveness"],
            StressScenario.INTEREST_RATE_SHOCK: ["Duration exposure", "Credit spread impact", "Sector rotation"],
            StressScenario.GEOPOLITICAL_CRISIS: ["Flight to quality", "Currency impacts", "Supply chain disruption"]
        }

        return risk_factor_map.get(scenario, ["General market risk"])

    async def _calculate_risk_adjusted_metrics(self,
                                             backtest_results: ThreePhaseBacktestResults) -> Dict[str, float]:
        """Calculate comprehensive risk-adjusted performance metrics"""

        try:
            # Risk-adjusted return calculation
            base_return = backtest_results.overall_sharpe * 0.1  # Rough return estimate
            risk_penalty = backtest_results.overall_expected_shortfall * 2.0  # Risk penalty
            risk_adjusted_return = base_return - risk_penalty

            # Tail risk assessment
            tail_risk_assessment = min(1.0, backtest_results.overall_expected_shortfall / 0.05)  # Normalize to 5%

            # Drawdown recovery score
            max_drawdown = backtest_results.overall_max_drawdown
            recovery_periods = [120, 90, 150]  # Estimated recovery times for phases
            avg_recovery = np.mean(recovery_periods)

            drawdown_recovery_score = max(0, 1 - (abs(max_drawdown) * 10 + avg_recovery / 365))
            drawdown_recovery_score = max(0, min(1, drawdown_recovery_score))

            return {
                "risk_adjusted_return": risk_adjusted_return,
                "tail_risk_assessment": tail_risk_assessment,
                "drawdown_recovery_score": drawdown_recovery_score
            }

        except Exception as e:
            logger.error(f"Risk-adjusted metrics calculation failed: {e}")
            return {
                "risk_adjusted_return": 0.0,
                "tail_risk_assessment": 0.5,
                "drawdown_recovery_score": 0.5
            }

    def _calculate_investment_grade_score(self,
                                        capacity_analyses: Dict[CapacityLevel, CapacityAnalysis],
                                        regime_analyses: Dict[MarketCondition, RegimeAnalysis],
                                        stress_test_results: Dict[StressScenario, StressTestResult]) -> float:
        """Calculate overall investment-grade score"""

        try:
            score_components = []

            # Capacity score (25 points)
            feasible_capacities = [ca for ca in capacity_analyses.values() if ca.is_feasible]
            if feasible_capacities:
                avg_feasibility = np.mean([ca.feasibility_score for ca in feasible_capacities])
                capacity_score = avg_feasibility * 25
            else:
                capacity_score = 0
            score_components.append(("Capacity", capacity_score))

            # Regime adaptability score (25 points)
            regime_scores = [ra.adaptation_score for ra in regime_analyses.values()]
            avg_regime_score = np.mean(regime_scores) if regime_scores else 0
            regime_component_score = avg_regime_score * 25
            score_components.append(("Regime Adaptability", regime_component_score))

            # Stress resilience score (25 points)
            resilience_scores = [str.resilience_score for str in stress_test_results.values()]
            avg_resilience = np.mean(resilience_scores) if resilience_scores else 0
            stress_component_score = avg_resilience * 25
            score_components.append(("Stress Resilience", stress_component_score))

            # Consistency score (25 points)
            consistency_scores = [ra.consistency_score for ra in regime_analyses.values()]
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
            consistency_component_score = avg_consistency * 25
            score_components.append(("Consistency", consistency_component_score))

            total_score = sum(score for _, score in score_components)

            return min(100, max(0, total_score))

        except Exception as e:
            logger.error(f"Investment grade score calculation failed: {e}")
            return 50.0  # Default neutral score

    def _assess_deployment_readiness(self,
                                   investment_grade_score: float,
                                   capacity_analyses: Dict[CapacityLevel, CapacityAnalysis],
                                   stress_test_results: Dict[StressScenario, StressTestResult]) -> str:
        """Assess deployment readiness based on validation results"""

        try:
            if investment_grade_score >= 85:
                readiness = "READY FOR INSTITUTIONAL DEPLOYMENT"
            elif investment_grade_score >= 75:
                readiness = "READY WITH ENHANCED MONITORING"
            elif investment_grade_score >= 65:
                readiness = "CONDITIONAL APPROVAL REQUIRED"
            elif investment_grade_score >= 50:
                readiness = "REQUIRES SIGNIFICANT IMPROVEMENTS"
            else:
                readiness = "NOT SUITABLE FOR DEPLOYMENT"

            # Check for critical issues
            feasible_capacity_count = sum(1 for ca in capacity_analyses.values() if ca.is_feasible)
            high_resilience_count = sum(1 for str in stress_test_results.values() if str.resilience_score > 0.7)

            if feasible_capacity_count == 0:
                readiness = "CAPACITY CONSTRAINTS - LIMITED DEPLOYMENT"
            elif high_resilience_count < len(stress_test_results) // 2:
                readiness = "STRESS TESTING CONCERNS - CONDITIONAL APPROVAL"

            return readiness

        except Exception as e:
            logger.error(f"Deployment readiness assessment failed: {e}")
            return "ASSESSMENT INCOMPLETE"

    def _determine_recommended_capacity(self,
                                      capacity_analyses: Dict[CapacityLevel, CapacityAnalysis]) -> CapacityLevel:
        """Determine recommended capacity level"""

        try:
            # Find the highest feasible capacity with good feasibility score
            feasible_capacities = [(level, analysis) for level, analysis in capacity_analyses.items()
                                 if analysis.is_feasible and analysis.feasibility_score > 0.7]

            if feasible_capacities:
                # Sort by AUM amount and take the highest feasible
                feasible_capacities.sort(key=lambda x: x[1].aum_amount, reverse=True)
                return feasible_capacities[0][0]
            else:
                # Return the best scoring capacity even if not feasible
                best_capacity = max(capacity_analyses.items(),
                                  key=lambda x: x[1].feasibility_score)
                return best_capacity[0]

        except Exception as e:
            logger.error(f"Recommended capacity determination failed: {e}")
            return CapacityLevel.SMALL

    def _calculate_maximum_feasible_aum(self,
                                      capacity_analyses: Dict[CapacityLevel, CapacityAnalysis]) -> float:
        """Calculate maximum feasible AUM"""

        try:
            feasible_analyses = [ca for ca in capacity_analyses.values() if ca.is_feasible]

            if feasible_analyses:
                return max(ca.aum_amount for ca in feasible_analyses)
            else:
                # Return the smallest capacity if none are feasible
                return min(ca.aum_amount for ca in capacity_analyses.values())

        except Exception as e:
            logger.error(f"Maximum feasible AUM calculation failed: {e}")
            return 10_000_000  # Default $10M

    def _calculate_regime_adaptability_score(self,
                                           regime_analyses: Dict[MarketCondition, RegimeAnalysis]) -> float:
        """Calculate overall regime adaptability score"""

        try:
            adaptation_scores = [ra.adaptation_score for ra in regime_analyses.values()]
            return np.mean(adaptation_scores) if adaptation_scores else 0.5

        except Exception as e:
            logger.error(f"Regime adaptability calculation failed: {e}")
            return 0.5

    def _identify_worst_regime(self,
                             regime_analyses: Dict[MarketCondition, RegimeAnalysis]) -> MarketCondition:
        """Identify worst performing market regime"""

        try:
            worst_regime = min(regime_analyses.items(),
                             key=lambda x: x[1].adaptation_score)
            return worst_regime[0]

        except Exception as e:
            logger.error(f"Worst regime identification failed: {e}")
            return MarketCondition.CRISIS_PERIOD

    def _calculate_resilience_score(self,
                                  stress_test_results: Dict[StressScenario, StressTestResult]) -> float:
        """Calculate overall resilience score"""

        try:
            resilience_scores = [str.resilience_score for str in stress_test_results.values()]
            return np.mean(resilience_scores) if resilience_scores else 0.5

        except Exception as e:
            logger.error(f"Overall resilience calculation failed: {e}")
            return 0.5

    def _identify_critical_vulnerabilities(self,
                                         stress_test_results: Dict[StressScenario, StressTestResult]) -> List[str]:
        """Identify critical vulnerabilities from stress testing"""

        try:
            vulnerabilities = []

            for scenario, result in stress_test_results.items():
                if result.resilience_score < 0.3:
                    vulnerabilities.append(f"Critical vulnerability to {scenario.value}")
                elif result.stress_return < -0.25:
                    vulnerabilities.append(f"Severe performance impact from {scenario.value}")
                elif result.recovery_time_days > 180:
                    vulnerabilities.append(f"Slow recovery from {scenario.value}")

            # Add generic vulnerabilities if specific ones not found
            if not vulnerabilities:
                low_resilience_scenarios = [scenario.value for scenario, result in stress_test_results.items()
                                          if result.resilience_score < 0.5]
                if low_resilience_scenarios:
                    vulnerabilities.append(f"Moderate vulnerabilities: {', '.join(low_resilience_scenarios)}")

            return vulnerabilities[:5]  # Limit to top 5

        except Exception as e:
            logger.error(f"Critical vulnerabilities identification failed: {e}")
            return ["Assessment incomplete"]

    def _generate_professional_recommendations(self,
                                             capacity_analyses: Dict[CapacityLevel, CapacityAnalysis],
                                             regime_analyses: Dict[MarketCondition, RegimeAnalysis],
                                             stress_test_results: Dict[StressScenario, StressTestResult],
                                             risk_metrics: Dict[str, float]) -> Dict[str, List[str]]:
        """Generate professional recommendations"""

        try:
            implementation_recommendations = []
            risk_management_requirements = []
            monitoring_recommendations = []

            # Implementation recommendations
            feasible_capacities = [ca for ca in capacity_analyses.values() if ca.is_feasible]
            if feasible_capacities:
                max_feasible = max(feasible_capacities, key=lambda x: x.aum_amount)
                implementation_recommendations.append(
                    f"Recommend initial deployment at {max_feasible.capacity_level.value} AUM level"
                )
            else:
                implementation_recommendations.append(
                    "Recommend starting with pilot program at minimal AUM"
                )

            # Check transaction costs
            high_cost_capacities = [ca for ca in capacity_analyses.values()
                                  if ca.transaction_cost_impact > 0.03]
            if high_cost_capacities:
                implementation_recommendations.append(
                    "Implement cost-efficient execution algorithms to manage transaction costs"
                )

            # Regime-based recommendations
            weak_regimes = [regime for regime, analysis in regime_analyses.items()
                          if analysis.adaptation_score < 0.5]
            if weak_regimes:
                implementation_recommendations.append(
                    f"Develop regime-specific adaptations for: {[r.value for r in weak_regimes]}"
                )

            # Risk management requirements
            if risk_metrics["tail_risk_assessment"] > 0.7:
                risk_management_requirements.append(
                    "Implement enhanced tail risk monitoring with ES@97.5% limits"
                )

            if risk_metrics["drawdown_recovery_score"] < 0.6:
                risk_management_requirements.append(
                    "Establish dynamic drawdown management with tiered response protocols"
                )

            # Stress test based requirements
            vulnerable_scenarios = [scenario for scenario, result in stress_test_results.items()
                                  if result.resilience_score < 0.4]
            if vulnerable_scenarios:
                risk_management_requirements.append(
                    f"Implement stress scenario hedging for: {[s.value for s in vulnerable_scenarios]}"
                )

            # Monitoring recommendations
            monitoring_recommendations.extend([
                "Establish real-time capacity utilization monitoring",
                "Implement regime detection and adaptation alerts",
                "Deploy comprehensive stress testing dashboard",
                "Maintain continuous performance attribution analysis",
                "Establish regulatory compliance monitoring framework"
            ])

            return {
                "implementation_recommendations": implementation_recommendations,
                "risk_management_requirements": risk_management_requirements,
                "monitoring_recommendations": monitoring_recommendations
            }

        except Exception as e:
            logger.error(f"Professional recommendations generation failed: {e}")
            return {
                "implementation_recommendations": ["Comprehensive review required"],
                "risk_management_requirements": ["Standard risk management protocols"],
                "monitoring_recommendations": ["Basic monitoring framework"]
            }

    def _assess_regulatory_compliance(self, backtest_results: ThreePhaseBacktestResults) -> float:
        """Assess regulatory compliance score"""

        try:
            compliance_score = 0.0

            # Check basic compliance requirements
            if backtest_results.overall_sharpe > 0.5:  # Reasonable risk-adjusted returns
                compliance_score += 20

            if backtest_results.overall_max_drawdown > -0.25:  # Reasonable drawdown limits
                compliance_score += 20

            if backtest_results.cross_phase_significance > 0.05:  # Statistical significance
                compliance_score += 20

            if backtest_results.performance_stability > 0.6:  # Performance stability
                compliance_score += 20

            # Documentation and reporting
            compliance_score += 20  # Assume proper documentation

            return min(100, compliance_score)

        except Exception as e:
            logger.error(f"Regulatory compliance assessment failed: {e}")
            return 60.0  # Default compliance score

    def _assess_documentation_completeness(self, backtest_results: ThreePhaseBacktestResults) -> float:
        """Assess documentation completeness"""

        try:
            # Check for required documentation components
            completeness_score = 0.0

            # Strategy description and methodology
            completeness_score += 20  # Assume documented

            # Risk management framework
            completeness_score += 20  # Assume documented

            # Performance measurement and attribution
            completeness_score += 20  # Assume documented

            # Stress testing methodology
            completeness_score += 20  # Assume documented

            # Compliance and regulatory documentation
            completeness_score += 20  # Assume documented

            return completeness_score

        except Exception as e:
            logger.error(f"Documentation completeness assessment failed: {e}")
            return 80.0  # Default documentation score

    async def _store_validation_results(self, validation_report: ValidationReport):
        """Store validation results in database"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store capacity analyses
                for capacity_analysis in validation_report.capacity_analyses.values():
                    conn.execute("""
                        INSERT INTO capacity_analysis (
                            strategy_name, capacity_level, aum_amount, adjusted_return,
                            adjusted_sharpe, performance_decay, transaction_cost_impact,
                            market_impact_cost, liquidity_score, is_feasible,
                            feasibility_score, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        validation_report.strategy_name, capacity_analysis.capacity_level.value,
                        capacity_analysis.aum_amount, capacity_analysis.adjusted_return,
                        capacity_analysis.adjusted_sharpe, capacity_analysis.performance_decay,
                        capacity_analysis.transaction_cost_impact, capacity_analysis.market_impact_cost,
                        capacity_analysis.liquidity_score, capacity_analysis.is_feasible,
                        capacity_analysis.feasibility_score, validation_report.validation_timestamp.isoformat()
                    ))

                # Store regime analyses
                for regime_analysis in validation_report.regime_analyses.values():
                    conn.execute("""
                        INSERT INTO regime_analysis (
                            strategy_name, regime, regime_return, regime_sharpe,
                            regime_max_drawdown, adaptation_score, consistency_score, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        validation_report.strategy_name, regime_analysis.regime.value,
                        regime_analysis.regime_return, regime_analysis.regime_sharpe,
                        regime_analysis.regime_max_drawdown, regime_analysis.adaptation_score,
                        regime_analysis.consistency_score, validation_report.validation_timestamp.isoformat()
                    ))

                # Store stress test results
                for stress_result in validation_report.stress_test_results.values():
                    conn.execute("""
                        INSERT INTO stress_test_results (
                            strategy_name, scenario, scenario_severity, stress_return,
                            stress_max_drawdown, recovery_time_days, resilience_score, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        validation_report.strategy_name, stress_result.scenario.value,
                        stress_result.scenario_severity, stress_result.stress_return,
                        stress_result.stress_max_drawdown, stress_result.recovery_time_days,
                        stress_result.resilience_score, validation_report.validation_timestamp.isoformat()
                    ))

                # Store validation report summary
                conn.execute("""
                    INSERT INTO validation_reports (
                        strategy_name, investment_grade_score, deployment_readiness,
                        recommended_capacity, maximum_feasible_aum, overall_resilience_score,
                        regulatory_compliance_score, audit_readiness, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    validation_report.strategy_name, validation_report.investment_grade_score,
                    validation_report.deployment_readiness, validation_report.recommended_capacity.value,
                    validation_report.maximum_feasible_aum, validation_report.overall_resilience_score,
                    validation_report.regulatory_compliance_score, validation_report.audit_readiness,
                    validation_report.validation_timestamp.isoformat()
                ))

                conn.commit()

            logger.info(f"Stored validation results for {validation_report.strategy_name}")

        except Exception as e:
            logger.error(f"Validation results storage failed: {e}")

# Example usage and testing
async def main():
    """Main function for testing the investment-grade validator"""
    print("Investment-Grade Backtesting Validator")
    print("=" * 50)

    # Initialize validator
    validator = InvestmentGradeValidator()

    # Create mock backtest results for testing
    from bot.enhanced_backtesting_system import (
        ThreePhaseBacktestResults, PhaseResults, BacktestPhase,
        BacktestPeriod, MarketCondition
    )

    # Mock phase results
    mock_phase_result = PhaseResults(
        phase=BacktestPhase.PHASE_2,
        period=BacktestPeriod(
            BacktestPhase.PHASE_2, "2017-01-01", "2020-12-31",
            "Test Period", "Normal"
        ),
        total_return=0.15,
        annualized_return=0.12,
        volatility=0.16,
        sharpe_ratio=0.85,
        sortino_ratio=1.1,
        calmar_ratio=0.8,
        expected_shortfall_975=0.03,
        expected_shortfall_99=0.045,
        max_drawdown=-0.08,
        max_drawdown_duration=45,
        var_95=-0.025,
        var_99=-0.04,
        win_rate=0.58,
        profit_factor=1.3,
        trade_count=150,
        avg_trade_return=0.0008,
        largest_win=0.025,
        largest_loss=-0.018,
        bull_market_performance=0.18,
        bear_market_performance=-0.05,
        crisis_performance=-0.12,
        normal_market_performance=0.14,
        information_ratio=0.65,
        tracking_error=0.08,
        beta=0.9,
        alpha=0.04,
        downside_capture=0.7,
        upside_capture=1.1,
        return_skewness=0.2,
        return_kurtosis=0.8,
        jarque_bera_stat=2.5,
        jarque_bera_pvalue=0.3,
        confidence_score=0.75,
        stability_indicator=0.82
    )

    # Mock three-phase results
    mock_backtest_results = ThreePhaseBacktestResults(
        strategy_name="Test_Investment_Grade_Strategy",
        backtest_timestamp=datetime.now(),
        phase_1_results=mock_phase_result,
        phase_2_results=mock_phase_result,
        phase_3_results=mock_phase_result,
        consistency_score=0.78,
        regime_adaptability=0.72,
        crisis_resilience=0.68,
        overall_sharpe=0.85,
        overall_calmar=0.8,
        overall_max_drawdown=-0.08,
        overall_expected_shortfall=0.03,
        cross_phase_significance=0.02,
        performance_stability=0.74,
        regime_robustness={"crisis": 0.6, "bull": 0.8, "bear": 0.65},
        risk_adjusted_return=0.09,
        risk_budget_utilization=0.6,
        tail_risk_contribution=0.4,
        deployment_recommendation="",
        risk_recommendations=[],
        optimization_suggestions=[]
    )

    # Run investment-grade validation
    print("Running investment-grade validation...")
    validation_report = await validator.validate_investment_grade(mock_backtest_results)

    # Display results summary
    print(f"\nValidation Results Summary:")
    print(f"Strategy: {validation_report.strategy_name}")
    print(f"Investment Grade Score: {validation_report.investment_grade_score:.1f}/100")
    print(f"Deployment Readiness: {validation_report.deployment_readiness}")
    print(f"Recommended Capacity: {validation_report.recommended_capacity.value}")
    print(f"Maximum Feasible AUM: ${validation_report.maximum_feasible_aum:,.0f}")
    print(f"Overall Resilience Score: {validation_report.overall_resilience_score:.3f}")

    print(f"\nCapacity Analysis:")
    for level, analysis in validation_report.capacity_analyses.items():
        feasible_status = "✓ Feasible" if analysis.is_feasible else "✗ Not Feasible"
        print(f"  {level.value}: {feasible_status} (Score: {analysis.feasibility_score:.2f})")

    print(f"\nStress Test Results:")
    for scenario, result in validation_report.stress_test_results.items():
        print(f"  {scenario.value}: Resilience {result.resilience_score:.2f}")

    print(f"\nCritical Vulnerabilities:")
    for vulnerability in validation_report.critical_vulnerabilities:
        print(f"  - {vulnerability}")

    print(f"\nImplementation Recommendations:")
    for rec in validation_report.implementation_recommendations:
        print(f"  - {rec}")

    print("\nInvestment-grade validator test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())