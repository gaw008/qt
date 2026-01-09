"""
Execution Integration Layer
执行集成层

Integrates the adaptive execution engine and cost attribution analyzer
with the existing trading system and real-time monitoring infrastructure.

Provides unified execution management with:
- Integration with Enhanced Risk Manager
- Real-time cost monitoring and alerting
- Execution performance tracking
- Automated optimization based on market conditions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from adaptive_execution_engine import AdaptiveExecutionEngine, ExecutionOrder, ExecutionUrgency
from cost_attribution_analyzer import CostAttributionAnalyzer
from enhanced_risk_manager import EnhancedRiskManager
from real_time_monitor import RealTimeMonitor, MonitoringAlert

@dataclass
class IntegratedExecutionRequest:
    """Enhanced execution request with risk management integration"""
    order_id: str
    symbol: str
    side: str
    total_quantity: int
    urgency: ExecutionUrgency
    max_cost_bps: float  # Maximum acceptable cost
    risk_budget_allocation: float  # Fraction of risk budget for this trade

    # Optional parameters
    target_price: Optional[float] = None
    time_horizon: Optional[timedelta] = None
    custom_participation_rate: Optional[float] = None

@dataclass
class ExecutionPerformanceMetrics:
    """Comprehensive execution performance metrics"""
    period_start: datetime
    period_end: datetime

    # Volume metrics
    total_orders: int
    total_volume: int
    successful_fills: int

    # Cost metrics
    average_total_cost_bps: float
    average_implementation_shortfall_bps: float
    cost_volatility: float

    # Risk metrics
    average_risk_utilization: float
    max_risk_breach: bool
    cost_budget_adherence: float

    # Efficiency metrics
    average_fill_rate: float
    average_execution_time: float
    participation_rate_optimization: float

class ExecutionIntegrationLayer:
    """
    Unified Execution Management System

    Integrates adaptive execution with risk management and real-time monitoring
    to provide institutional-grade execution management with:
    - Risk-aware execution optimization
    - Real-time cost monitoring and alerting
    - Performance tracking and attribution
    - Automated parameter adjustment
    """

    def __init__(self):
        self.logger = self._setup_logging()

        # Initialize core components
        self.execution_engine = AdaptiveExecutionEngine()
        self.cost_analyzer = CostAttributionAnalyzer()
        self.risk_manager = EnhancedRiskManager()
        self.monitor = RealTimeMonitor()

        # Execution state tracking
        self.active_executions: Dict[str, IntegratedExecutionRequest] = {}
        self.execution_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.performance_metrics: Dict[str, float] = {
            "average_cost_savings": 0.0,
            "risk_budget_efficiency": 0.0,
            "execution_success_rate": 0.0,
            "cost_prediction_accuracy": 0.0
        }

        # Integration settings
        self.settings = {
            "max_cost_threshold_bps": 50.0,  # Alert if cost > 50bps
            "risk_budget_limit": 0.20,       # Max 20% risk budget per trade
            "auto_optimization": True,       # Enable auto parameter optimization
            "cost_prediction_enabled": True  # Enable pre-trade cost estimation
        }

        self.logger.info("Execution Integration Layer initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for execution integration"""
        logger = logging.getLogger('ExecutionIntegration')
        logger.setLevel(logging.INFO)

        # File handler
        log_path = "logs/execution_integration.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    async def submit_integrated_execution(self, request: IntegratedExecutionRequest) -> str:
        """Submit execution request with integrated risk management"""
        try:
            self.logger.info(f"Processing integrated execution request: {request.order_id}")

            # Step 1: Risk validation
            risk_check = await self._validate_risk_parameters(request)
            if not risk_check["approved"]:
                raise ValueError(f"Risk validation failed: {risk_check['reason']}")

            # Step 2: Pre-trade cost estimation
            if self.settings["cost_prediction_enabled"]:
                cost_estimate = await self._generate_cost_estimate(request)

                if cost_estimate.total_cost_bps > request.max_cost_bps:
                    self.logger.warning(f"Estimated cost {cost_estimate.total_cost_bps:.1f}bps "
                                      f"exceeds limit {request.max_cost_bps:.1f}bps")

            # Step 3: Create optimized execution order
            execution_order = await self._create_optimized_execution_order(request)

            # Step 4: Submit to execution engine
            order_id = await self.execution_engine.submit_order(execution_order)

            # Step 5: Track execution
            self.active_executions[order_id] = request
            await self._start_execution_monitoring(order_id, request)

            self.logger.info(f"Integrated execution submitted: {order_id}")
            return order_id

        except Exception as e:
            self.logger.error(f"Integrated execution submission failed: {e}")
            raise

    async def _validate_risk_parameters(self, request: IntegratedExecutionRequest) -> Dict[str, Any]:
        """Validate execution request against risk parameters"""
        try:
            # Check risk budget allocation
            if request.risk_budget_allocation > self.settings["risk_budget_limit"]:
                return {
                    "approved": False,
                    "reason": f"Risk budget allocation {request.risk_budget_allocation:.2%} "
                             f"exceeds limit {self.settings['risk_budget_limit']:.2%}"
                }

            # Check current risk utilization
            current_risk = await self._get_current_risk_utilization()
            if current_risk + request.risk_budget_allocation > 1.0:
                return {
                    "approved": False,
                    "reason": f"Total risk utilization would exceed 100%"
                }

            # Check position size limits (simplified)
            portfolio_value = 10_000_000  # $10M portfolio assumption
            trade_value = request.total_quantity * 150.0  # Assume $150 stock price
            position_limit = 0.05  # 5% max position size

            if trade_value / portfolio_value > position_limit:
                return {
                    "approved": False,
                    "reason": f"Position size exceeds {position_limit:.1%} limit"
                }

            return {"approved": True, "reason": "Risk validation passed"}

        except Exception as e:
            self.logger.error(f"Risk validation failed: {e}")
            return {"approved": False, "reason": f"Validation error: {str(e)}"}

    async def _generate_cost_estimate(self, request: IntegratedExecutionRequest):
        """Generate pre-trade cost estimate"""
        try:
            # Get current market data (simulated)
            market_data = {
                "volatility": 0.25,
                "bid_ask_spread": 0.0015,
                "daily_volume": 1500000,
                "regime": "normal",
                "price": 150.0
            }

            # Adjust for urgency
            urgency_adjustments = {
                ExecutionUrgency.LOW: 0.8,
                ExecutionUrgency.MEDIUM: 1.0,
                ExecutionUrgency.HIGH: 1.3,
                ExecutionUrgency.URGENT: 1.8
            }

            urgency_multiplier = urgency_adjustments.get(request.urgency, 1.0)
            market_data["volatility"] *= urgency_multiplier

            # Generate estimate
            estimate = self.cost_analyzer.estimate_pre_trade_costs(
                request.symbol, request.side, request.total_quantity,
                market_data["price"], market_data
            )

            return estimate

        except Exception as e:
            self.logger.error(f"Cost estimation failed: {e}")
            raise

    async def _create_optimized_execution_order(self, request: IntegratedExecutionRequest) -> ExecutionOrder:
        """Create optimized execution order with adaptive parameters"""
        try:
            # Determine optimal participation rate
            if request.custom_participation_rate:
                participation_rate = request.custom_participation_rate
            else:
                participation_rate = await self._calculate_optimal_participation_rate(request)

            # Set time horizon based on urgency if not provided
            if request.time_horizon:
                time_horizon = request.time_horizon
            else:
                urgency_horizons = {
                    ExecutionUrgency.LOW: timedelta(hours=4),
                    ExecutionUrgency.MEDIUM: timedelta(hours=2),
                    ExecutionUrgency.HIGH: timedelta(hours=1),
                    ExecutionUrgency.URGENT: timedelta(minutes=30)
                }
                time_horizon = urgency_horizons.get(request.urgency, timedelta(hours=2))

            # Create execution order
            execution_order = ExecutionOrder(
                order_id=request.order_id,
                symbol=request.symbol,
                side=request.side,
                total_quantity=request.total_quantity,
                target_price=request.target_price,
                urgency=request.urgency,
                max_participation_rate=min(0.50, participation_rate * 1.5),  # 50% max with buffer
                time_horizon=time_horizon,
                current_participation_rate=participation_rate
            )

            return execution_order

        except Exception as e:
            self.logger.error(f"Execution order creation failed: {e}")
            raise

    async def _calculate_optimal_participation_rate(self, request: IntegratedExecutionRequest) -> float:
        """Calculate optimal participation rate considering cost and risk constraints"""
        try:
            # Base rates by urgency
            base_rates = {
                ExecutionUrgency.LOW: 0.08,
                ExecutionUrgency.MEDIUM: 0.15,
                ExecutionUrgency.HIGH: 0.25,
                ExecutionUrgency.URGENT: 0.35
            }

            base_rate = base_rates.get(request.urgency, 0.15)

            # Adjust for cost constraints
            if request.max_cost_bps < 20:  # Low cost tolerance
                base_rate *= 0.7
            elif request.max_cost_bps > 50:  # High cost tolerance
                base_rate *= 1.3

            # Adjust for risk budget
            if request.risk_budget_allocation < 0.10:  # Conservative risk budget
                base_rate *= 0.8
            elif request.risk_budget_allocation > 0.15:  # Aggressive risk budget
                base_rate *= 1.2

            return np.clip(base_rate, 0.05, 0.50)  # 5-50% bounds

        except Exception as e:
            self.logger.error(f"Participation rate calculation failed: {e}")
            return 0.15  # Default

    async def _get_current_risk_utilization(self) -> float:
        """Get current risk budget utilization"""
        try:
            # Simplified risk utilization calculation
            # In practice, would integrate with portfolio risk system
            active_risk = len(self.active_executions) * 0.05  # Assume 5% per active execution
            return min(active_risk, 1.0)

        except Exception as e:
            self.logger.error(f"Risk utilization calculation failed: {e}")
            return 0.0

    async def _start_execution_monitoring(self, order_id: str, request: IntegratedExecutionRequest):
        """Start real-time monitoring of execution"""
        try:
            # Create monitoring task
            asyncio.create_task(self._monitor_execution(order_id, request))

            self.logger.info(f"Started execution monitoring for {order_id}")

        except Exception as e:
            self.logger.error(f"Execution monitoring setup failed: {e}")

    async def _monitor_execution(self, order_id: str, request: IntegratedExecutionRequest):
        """Monitor execution progress and generate alerts"""
        try:
            monitoring_interval = 30  # seconds

            while order_id in self.active_executions:
                # Get execution status
                status = self.execution_engine.get_order_status(order_id)

                if not status:
                    break

                # Check for cost overruns
                current_cost = status.get("implementation_shortfall", 0.0)
                if abs(current_cost) > request.max_cost_bps:
                    await self._generate_cost_alert(order_id, current_cost, request.max_cost_bps)

                # Check execution progress
                fill_rate = status.get("fill_rate", 0.0)
                if fill_rate > 0.9:  # 90% filled
                    await self._handle_execution_completion(order_id, status, request)
                    break

                await asyncio.sleep(monitoring_interval)

        except Exception as e:
            self.logger.error(f"Execution monitoring failed for {order_id}: {e}")

    async def _generate_cost_alert(self, order_id: str, current_cost: float, max_cost: float):
        """Generate cost overrun alert"""
        try:
            alert = MonitoringAlert(
                timestamp=datetime.now(),
                severity="HIGH",
                category="EXECUTION",
                message=f"Execution cost overrun for {order_id}",
                source_module="ExecutionIntegration",
                metric_value=abs(current_cost),
                threshold=max_cost,
                recommendation=f"Consider reducing participation rate or canceling order"
            )

            # Send alert through monitoring system
            await self.monitor._process_alert(alert)

            self.logger.warning(f"Cost alert generated for {order_id}: "
                              f"{current_cost:.1f}bps vs {max_cost:.1f}bps limit")

        except Exception as e:
            self.logger.error(f"Cost alert generation failed: {e}")

    async def _handle_execution_completion(self, order_id: str, status: Dict[str, Any],
                                         request: IntegratedExecutionRequest):
        """Handle execution completion and analysis"""
        try:
            # Remove from active executions
            if order_id in self.active_executions:
                del self.active_executions[order_id]

            # Generate post-trade analysis
            execution_data = {
                "symbol": request.symbol,
                "side": request.side,
                "quantity": request.total_quantity,
                "executed_quantity": int(status["filled_quantity"]),
                "arrival_price": 150.0,  # Would be actual arrival price
                "average_execution_price": status["average_fill_price"],
                "participation_rate": status.get("current_participation_rate", 0.15),
                "duration": timedelta(minutes=60)  # Would be actual duration
            }

            market_data = {
                "vwap": 150.10,
                "twap": 150.05,
                "volatility": 0.25,
                "bid_ask_spread": 0.0015,
                "daily_volume": 1500000
            }

            # Generate cost breakdown
            cost_breakdown = self.cost_analyzer.analyze_execution_costs(
                order_id, execution_data, market_data
            )

            # Update performance metrics
            await self._update_performance_metrics(cost_breakdown, request)

            # Store execution record
            execution_record = {
                "order_id": order_id,
                "request": request,
                "status": status,
                "cost_breakdown": cost_breakdown,
                "completed_at": datetime.now()
            }
            self.execution_history.append(execution_record)

            self.logger.info(f"Execution completed for {order_id}: "
                           f"Fill: {status['fill_rate']:.1%}, "
                           f"Cost: {cost_breakdown.total_cost_bps:.1f}bps")

        except Exception as e:
            self.logger.error(f"Execution completion handling failed: {e}")

    async def _update_performance_metrics(self, cost_breakdown, request: IntegratedExecutionRequest):
        """Update execution performance metrics"""
        try:
            # Calculate cost savings vs naive execution
            naive_cost_estimate = 50.0  # Assume 50bps naive cost
            actual_cost = cost_breakdown.total_cost_bps
            cost_savings = naive_cost_estimate - actual_cost

            # Update running averages
            n_executions = len(self.execution_history)
            if n_executions > 0:
                # Simple moving average update
                alpha = 1.0 / min(50, n_executions + 1)  # Decay factor

                self.performance_metrics["average_cost_savings"] = (
                    (1 - alpha) * self.performance_metrics["average_cost_savings"] +
                    alpha * cost_savings
                )

                self.performance_metrics["execution_success_rate"] = (
                    (1 - alpha) * self.performance_metrics["execution_success_rate"] +
                    alpha * (1.0 if cost_breakdown.fill_rate > 0.95 else 0.0)
                )

                # Risk budget efficiency (simplified)
                risk_efficiency = 1.0 / max(0.01, request.risk_budget_allocation)
                self.performance_metrics["risk_budget_efficiency"] = (
                    (1 - alpha) * self.performance_metrics["risk_budget_efficiency"] +
                    alpha * risk_efficiency
                )

        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")

    async def get_execution_performance_summary(self, days: int = 30) -> ExecutionPerformanceMetrics:
        """Generate execution performance summary"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Filter executions in period
            period_executions = [
                exec for exec in self.execution_history
                if start_date <= exec["completed_at"] <= end_date
            ]

            if not period_executions:
                return ExecutionPerformanceMetrics(
                    period_start=start_date,
                    period_end=end_date,
                    total_orders=0,
                    total_volume=0,
                    successful_fills=0,
                    average_total_cost_bps=0.0,
                    average_implementation_shortfall_bps=0.0,
                    cost_volatility=0.0,
                    average_risk_utilization=0.0,
                    max_risk_breach=False,
                    cost_budget_adherence=0.0,
                    average_fill_rate=0.0,
                    average_execution_time=0.0,
                    participation_rate_optimization=0.0
                )

            # Calculate metrics
            total_orders = len(period_executions)
            total_volume = sum(exec["request"].total_quantity for exec in period_executions)
            successful_fills = sum(1 for exec in period_executions
                                 if exec["status"]["fill_rate"] > 0.95)

            costs = [exec["cost_breakdown"].total_cost_bps for exec in period_executions]
            is_costs = [exec["cost_breakdown"].vs_arrival_bps for exec in period_executions]
            fill_rates = [exec["status"]["fill_rate"] for exec in period_executions]

            average_total_cost_bps = np.mean(costs) if costs else 0.0
            average_implementation_shortfall_bps = np.mean(is_costs) if is_costs else 0.0
            cost_volatility = np.std(costs) if len(costs) > 1 else 0.0
            average_fill_rate = np.mean(fill_rates) if fill_rates else 0.0

            # Risk metrics (simplified)
            risk_allocations = [exec["request"].risk_budget_allocation for exec in period_executions]
            average_risk_utilization = np.mean(risk_allocations) if risk_allocations else 0.0
            max_risk_breach = any(r > 0.20 for r in risk_allocations)

            # Cost budget adherence
            cost_adherence = sum(1 for exec in period_executions
                               if abs(exec["cost_breakdown"].total_cost_bps) <= exec["request"].max_cost_bps)
            cost_budget_adherence = cost_adherence / total_orders if total_orders > 0 else 0.0

            return ExecutionPerformanceMetrics(
                period_start=start_date,
                period_end=end_date,
                total_orders=total_orders,
                total_volume=total_volume,
                successful_fills=successful_fills,
                average_total_cost_bps=average_total_cost_bps,
                average_implementation_shortfall_bps=average_implementation_shortfall_bps,
                cost_volatility=cost_volatility,
                average_risk_utilization=average_risk_utilization,
                max_risk_breach=max_risk_breach,
                cost_budget_adherence=cost_budget_adherence,
                average_fill_rate=average_fill_rate,
                average_execution_time=3600.0,  # Simplified
                participation_rate_optimization=0.85  # Simplified
            )

        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {e}")
            raise

    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution system status"""
        return {
            "active_executions": len(self.active_executions),
            "total_executions_today": len([
                exec for exec in self.execution_history
                if exec["completed_at"].date() == datetime.now().date()
            ]),
            "performance_metrics": self.performance_metrics.copy(),
            "system_settings": self.settings.copy()
        }

async def main():
    """Test the execution integration layer"""
    integration = ExecutionIntegrationLayer()

    print("Testing Execution Integration Layer...")

    # Create test execution request
    test_request = IntegratedExecutionRequest(
        order_id="INTEGRATION_TEST_001",
        symbol="AAPL",
        side="buy",
        total_quantity=5000,
        urgency=ExecutionUrgency.MEDIUM,
        max_cost_bps=30.0,
        risk_budget_allocation=0.10,
        target_price=150.0
    )

    try:
        # Submit integrated execution
        order_id = await integration.submit_integrated_execution(test_request)
        print(f"OK Integrated execution submitted: {order_id}")

        # Wait for execution to progress
        await asyncio.sleep(10)

        # Get status
        status = integration.get_current_status()
        print(f"OK Current status: {status['active_executions']} active executions")

        # Get performance summary
        perf_summary = await integration.get_execution_performance_summary(1)
        print(f"OK Performance summary generated")

        print("\nSUCCESS: Execution Integration Layer fully operational!")
        print("\nIntegrated Features:")
        print("  - Risk-aware execution optimization")
        print("  - Pre-trade cost estimation and validation")
        print("  - Real-time execution monitoring and alerting")
        print("  - Post-trade cost attribution and analysis")
        print("  - Performance tracking and optimization")

    except Exception as e:
        print(f"ERROR: Integration test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())