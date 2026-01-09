"""
Enhanced Trading System Runner with Phase 3 Integration

This module extends the basic runner with:
- Market regime detection and strategy adaptation
- Real-time risk monitoring and exposure tracking
- Intelligent portfolio management
"""

import os, time, random
from dotenv import load_dotenv
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import threading
import logging
import json

# Add paths for imports
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
BOT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bot"))
IMPROVEMENT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "improvement"))
sys.path.append(BASE)
sys.path.append(BOT_BASE)
sys.path.append(IMPROVEMENT_BASE)

from state_manager import is_killed, write_status, append_log, read_status
from market_time import get_market_manager, MarketPhase, MarketType
from config import SETTINGS

# Import Phase 3 modules
try:
    from regime.market_regime_detector import MarketRegimeDetector, MarketRegime
    from regime.regime_strategy_adapter import RegimeStrategyAdapter
    from monitoring.risk_exposure_monitor import RiskExposureMonitor
    from monitoring.performance_attribution import PerformanceAttributor
    PHASE3_AVAILABLE = True
    print("[PHASE3] Phase 3 modules loaded successfully")
except ImportError as e:
    PHASE3_AVAILABLE = False
    print(f"[PHASE3] Phase 3 modules not available: {e}")

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedMarketScheduler:
    """
    Enhanced market-aware scheduler with Phase 3 capabilities:
    - Market regime detection and adaptation
    - Real-time risk monitoring
    - Intelligent portfolio management
    """

    def __init__(self, market_type: str = "US"):
        """Initialize enhanced scheduler."""
        self.market_manager = get_market_manager(market_type)
        self.market_type = market_type
        self.running = False

        # Phase 3 components
        if PHASE3_AVAILABLE:
            self.regime_detector = MarketRegimeDetector()
            self.strategy_adapter = RegimeStrategyAdapter()
            self.risk_monitor = RiskExposureMonitor()
            self.performance_attributor = PerformanceAttributor()
            append_log("[PHASE3] Enhanced components initialized")
        else:
            self.regime_detector = None
            self.strategy_adapter = None
            self.risk_monitor = None
            self.performance_attributor = None
            append_log("[PHASE3] Running in basic mode (Phase 3 not available)")

        # Task tracking
        self.current_regime = None
        self.adapted_params = None
        self.risk_metrics = None
        self.last_regime_check = None

        # Import basic scheduler for delegation
        try:
            from runner import MarketAwareScheduler
            self.basic_scheduler = MarketAwareScheduler(market_type)
            append_log("[ENHANCED] Basic scheduler loaded for delegation")
        except ImportError:
            self.basic_scheduler = None
            append_log("[ENHANCED] Warning: Basic scheduler not available")

    def detect_market_regime(self, market_data, vix_data=None):
        """Detect current market regime and adapt strategy."""
        if not PHASE3_AVAILABLE or not self.regime_detector:
            return None

        try:
            append_log("[REGIME] Starting market regime detection")

            # Detect current regime
            regime_analysis = self.regime_detector.detect_regime(market_data, vix_data)
            self.current_regime = regime_analysis

            # Adapt strategy parameters
            adapted_params = self.strategy_adapter.adapt_strategy(regime_analysis)
            self.adapted_params = adapted_params

            append_log(f"[REGIME] Detected: {regime_analysis.current_regime.value} "
                      f"(confidence: {regime_analysis.confidence:.1%})")
            append_log(f"[REGIME] Adapted params: pos_size={adapted_params.position_size:.1%}, "
                      f"stop_loss={adapted_params.stop_loss_level:.1%}, "
                      f"max_positions={adapted_params.max_positions}")

            # Update status with regime information
            regime_status = {
                "regime": regime_analysis.current_regime.value,
                "confidence": regime_analysis.confidence,
                "indicators_count": len(regime_analysis.indicators),
                "adapted_position_size": adapted_params.position_size,
                "adapted_stop_loss": adapted_params.stop_loss_level,
                "adapted_max_positions": adapted_params.max_positions,
                "last_regime_check": datetime.now().isoformat()
            }

            return regime_status

        except Exception as e:
            append_log(f"[REGIME ERROR] {e}")
            return None

    def monitor_portfolio_risk(self, portfolio_data):
        """Monitor portfolio risk exposure in real-time."""
        if not PHASE3_AVAILABLE or not self.risk_monitor:
            return None

        try:
            append_log("[RISK] Starting portfolio risk analysis")

            # Calculate risk metrics
            risk_metrics = self.risk_monitor.calculate_exposure_metrics(portfolio_data)
            self.risk_metrics = risk_metrics

            # Check risk violations
            violations = self.risk_monitor.check_risk_limits(risk_metrics)

            append_log(f"[RISK] Portfolio analysis complete: "
                      f"concentration={risk_metrics.concentration_ratio:.3f}, "
                      f"effective_positions={risk_metrics.effective_positions}")

            if violations:
                append_log(f"[RISK WARNING] {len(violations)} violations detected:")
                for violation in violations:
                    append_log(f"  - {violation}")

            # Generate risk report
            risk_report = self.risk_monitor.generate_risk_report(risk_metrics)

            return {
                "risk_metrics": {
                    "concentration_ratio": risk_metrics.concentration_ratio,
                    "max_position_weight": risk_metrics.max_position_weight,
                    "effective_positions": risk_metrics.effective_positions,
                    "total_exposure": risk_metrics.total_exposure
                },
                "sector_exposure": risk_metrics.sector_exposure,
                "violations": violations,
                "risk_status": risk_report.get("risk_status", "UNKNOWN"),
                "last_risk_check": datetime.now().isoformat()
            }

        except Exception as e:
            append_log(f"[RISK ERROR] {e}")
            return None

    def enhanced_stock_selection_task(self):
        """Enhanced stock selection with regime awareness."""
        try:
            append_log("[ENHANCED SELECTION] Starting regime-aware stock selection")

            # Step 1: Get market data for regime detection
            market_data, vix_data = self._get_market_data_for_regime()

            # Step 2: Detect market regime and adapt parameters
            regime_status = None
            if market_data is not None:
                regime_status = self.detect_market_regime(market_data, vix_data)

            # Step 3: Run basic selection with adapted parameters
            if self.basic_scheduler:
                # Temporarily override selection criteria if regime detected
                if self.adapted_params:
                    append_log(f"[ENHANCED SELECTION] Using regime-adapted parameters")
                    # Could modify selection criteria here based on adapted_params

                # Force selection to run regardless of market timing (for enhanced system)
                append_log("[ENHANCED SELECTION] Running selection regardless of market phase")

                # Directly run the selection logic from basic scheduler
                try:
                    # Import selection strategies
                    from selection_strategies.value_momentum import ValueMomentumStrategy
                    from selection_strategies.technical_breakout import TechnicalBreakoutStrategy
                    from selection_strategies.earnings_momentum import EarningsMomentumStrategy
                    from selection_strategies.base_strategy import SelectionCriteria

                    # Get stock universe from CSV file (5000+ stocks)
                    universe = self.basic_scheduler._get_stock_universe()

                    # Configure selection criteria (potentially adapted based on regime)
                    criteria = SelectionCriteria(
                        max_stocks=self.adapted_params.max_positions if self.adapted_params else 10,
                        min_market_cap=1e9,  # $1B minimum
                        max_market_cap=1e12,  # $1T maximum
                        min_volume=100000,
                        min_price=5.0,
                        max_price=500.0,
                        min_score_threshold=50.0
                    )

                    # Run multiple selection strategies
                    all_results = {}
                    strategies = [
                        ValueMomentumStrategy(),
                        TechnicalBreakoutStrategy(),
                        EarningsMomentumStrategy()
                    ]

                    for strategy in strategies:
                        try:
                            append_log(f"[ENHANCED SELECTION] Running {strategy.name} strategy")
                            results = strategy.select_stocks(universe, criteria)
                            all_results[strategy.name] = results

                            # Log strategy results
                            summary = results.to_summary()
                            append_log(f"[ENHANCED SELECTION] {strategy.name}: {summary['total_selected']} stocks selected, "
                                     f"avg_score: {summary.get('avg_score', 0):.1f}")

                        except Exception as e:
                            append_log(f"[ENHANCED SELECTION] Error running {strategy.name}: {e}")
                            continue

                    # Combine and rank results
                    combined_selections = self.basic_scheduler._combine_strategy_results(all_results)

                    # Update status with selection results
                    self.basic_scheduler._update_selection_status(combined_selections)

                    append_log(f"[ENHANCED SELECTION] Selection completed: {len(combined_selections)} stocks")

                except Exception as e:
                    append_log(f"[ENHANCED SELECTION] Selection logic error: {e}")
                    # Fallback to basic scheduler anyway
                    self.basic_scheduler.stock_selection_task()

            # Step 4: Update status with regime information
            if regime_status:
                status = read_status()
                status.update({"market_regime": regime_status})
                write_status(status)

            append_log("[ENHANCED SELECTION] Regime-aware selection completed")

        except Exception as e:
            append_log(f"[ENHANCED SELECTION ERROR] {e}")
            # Fallback to basic selection
            if self.basic_scheduler:
                self.basic_scheduler.stock_selection_task()

    def enhanced_portfolio_monitoring_task(self):
        """Enhanced portfolio monitoring with risk analysis."""
        try:
            append_log("[ENHANCED MONITORING] Starting enhanced portfolio monitoring")

            # Step 1: Get current portfolio data
            portfolio_data = self._get_current_portfolio_data()

            # Step 2: Perform risk analysis
            risk_status = None
            if portfolio_data:
                risk_status = self.monitor_portfolio_risk(portfolio_data)

            # Step 3: Run basic monitoring
            if self.basic_scheduler:
                self.basic_scheduler.market_monitoring_task()

            # Step 4: Update status with risk information
            if risk_status:
                status = read_status()
                status.update({"portfolio_risk": risk_status})
                write_status(status)

            append_log("[ENHANCED MONITORING] Enhanced monitoring completed")

        except Exception as e:
            append_log(f"[ENHANCED MONITORING ERROR] {e}")
            # Fallback to basic monitoring
            if self.basic_scheduler:
                self.basic_scheduler.market_monitoring_task()

    def _get_market_data_for_regime(self):
        """Get market data for regime detection."""
        try:
            # Try to get SPY data and VIX data
            # This is a simplified implementation - in production you'd use real data
            from datetime import datetime, timedelta
            import pandas as pd
            import numpy as np

            # Generate sample data for testing
            # In production, replace with real data fetching
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=100)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            # Simple market data simulation
            np.random.seed(42)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 400 * np.exp(np.cumsum(returns))

            market_data = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(50000000, 200000000, len(dates)),
                'high': prices * 1.01,
                'low': prices * 0.99,
            }, index=dates)

            # VIX data simulation
            vix_values = 20 + np.random.normal(0, 5, len(dates))
            vix_data = pd.DataFrame({
                'close': np.maximum(10, vix_values)
            }, index=dates)

            append_log(f"[REGIME DATA] Generated market data: {len(dates)} days")
            return market_data, vix_data

        except Exception as e:
            append_log(f"[REGIME DATA ERROR] {e}")
            return None, None

    def _get_current_portfolio_data(self):
        """Get current portfolio data for risk analysis."""
        try:
            # Try to get portfolio from status
            status = read_status()
            positions = status.get("positions", [])

            if not positions:
                append_log("[RISK DATA] No positions found in status")
                return None

            # Convert to format expected by risk monitor
            portfolio_data = {}
            for pos in positions:
                symbol = pos.get("symbol", "")
                if symbol:
                    # Calculate weight (simplified)
                    value = pos.get("value", 0)
                    total_value = sum(p.get("value", 0) for p in positions)
                    weight = value / total_value if total_value > 0 else 0

                    portfolio_data[symbol] = {
                        "weight": weight,
                        "sector": "Technology",  # Simplified - would map from real data
                        "market_cap": 1e11  # Simplified - would get from real data
                    }

            append_log(f"[RISK DATA] Portfolio data: {len(portfolio_data)} positions")
            return portfolio_data

        except Exception as e:
            append_log(f"[RISK DATA ERROR] {e}")
            return None

    def run_enhanced_cycle(self):
        """Run one cycle of enhanced trading system."""
        try:
            append_log("[ENHANCED CYCLE] Starting enhanced trading cycle")

            # Check market phase
            current_phase = self.market_manager.get_current_phase()
            append_log(f"[ENHANCED CYCLE] Market phase: {current_phase.value}")

            # Determine what tasks to run based on market phase
            if current_phase in [MarketPhase.PRE_MARKET, MarketPhase.AFTER_HOURS, MarketPhase.CLOSED]:
                # During closed hours: run selection with regime awareness
                self.enhanced_stock_selection_task()

            # Always run enhanced monitoring
            self.enhanced_portfolio_monitoring_task()

            # Run trading tasks if market is open (delegate to basic scheduler)
            if current_phase in [MarketPhase.OPEN, MarketPhase.ACTIVE] and self.basic_scheduler:
                append_log("[ENHANCED CYCLE] Delegating trading tasks to basic scheduler")
                # Run trading tasks
                try:
                    self.basic_scheduler.real_trading_task()
                except Exception as e:
                    append_log(f"[ENHANCED CYCLE] Trading task error: {e}")

            append_log("[ENHANCED CYCLE] Enhanced cycle completed")

        except Exception as e:
            append_log(f"[ENHANCED CYCLE ERROR] {e}")

    def start_enhanced_monitoring(self):
        """Start the enhanced monitoring system."""
        append_log("[ENHANCED] Starting enhanced trading system monitoring")

        if PHASE3_AVAILABLE:
            append_log("[ENHANCED] Phase 3 capabilities enabled")
        else:
            append_log("[ENHANCED] Running in basic mode")

        self.running = True

        while self.running and not is_killed():
            try:
                self.run_enhanced_cycle()
                time.sleep(30)  # Run every 30 seconds

            except KeyboardInterrupt:
                append_log("[ENHANCED] Shutdown requested")
                break
            except Exception as e:
                append_log(f"[ENHANCED ERROR] {e}")
                time.sleep(10)  # Wait before retrying

        append_log("[ENHANCED] Enhanced monitoring stopped")

    def stop(self):
        """Stop the enhanced scheduler."""
        self.running = False
        if self.basic_scheduler:
            self.basic_scheduler.running = False


def main():
    """Main entry point for enhanced runner."""
    enhanced_scheduler = EnhancedMarketScheduler()

    try:
        enhanced_scheduler.start_enhanced_monitoring()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        enhanced_scheduler.stop()
        print("Enhanced scheduler stopped.")


if __name__ == "__main__":
    main()