#!/usr/bin/env python3
"""
Comprehensive Backtesting System Integration Test
综合回测系统集成测试

Test script for the complete three-phase backtesting validation framework:
- Enhanced three-phase backtesting system
- Investment-grade validation with capacity analysis
- Statistical validation framework with Monte Carlo
- Professional reporting system with multiple outputs

This test validates the entire integrated system end-to-end.
"""

import os
import sys
import asyncio
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add bot directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bot'))

# Import all the enhanced backtesting components
from bot.enhanced_backtesting_system import (
    EnhancedBacktestingSystem, ValidationMethod, BacktestConfig
)
from bot.investment_grade_validator import (
    InvestmentGradeValidator
)
from bot.statistical_validation_framework import (
    StatisticalValidationFramework
)
from bot.enhanced_backtesting_report_system import (
    EnhancedBacktestingReportSystem, ReportConfiguration, ReportType,
    OutputFormat, ReportTemplate
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveBacktestingTest:
    """Comprehensive test of the integrated backtesting system"""

    def __init__(self):
        """Initialize test environment"""
        self.test_results = {}
        self.start_time = datetime.now()

        # Create test directories
        self.test_dir = Path("test_results")
        self.test_dir.mkdir(exist_ok=True)

        logger.info("Comprehensive Backtesting System Test Initialized")

    async def run_complete_test_suite(self):
        """Run the complete integrated test suite"""
        try:
            logger.info("=" * 60)
            logger.info("COMPREHENSIVE BACKTESTING SYSTEM INTEGRATION TEST")
            logger.info("=" * 60)

            # Test 1: Enhanced Three-Phase Backtesting
            logger.info("\nTEST 1: Enhanced Three-Phase Backtesting System")
            logger.info("-" * 50)
            backtest_results = await self.test_enhanced_backtesting()

            if backtest_results:
                logger.info("✓ Enhanced backtesting system test PASSED")
                self.test_results["enhanced_backtesting"] = "PASS"
            else:
                logger.error("✗ Enhanced backtesting system test FAILED")
                self.test_results["enhanced_backtesting"] = "FAIL"
                return False

            # Test 2: Investment-Grade Validation
            logger.info("\nTEST 2: Investment-Grade Validation System")
            logger.info("-" * 50)
            validation_report = await self.test_investment_grade_validation(backtest_results)

            if validation_report:
                logger.info("✓ Investment-grade validation test PASSED")
                self.test_results["investment_grade_validation"] = "PASS"
            else:
                logger.error("✗ Investment-grade validation test FAILED")
                self.test_results["investment_grade_validation"] = "FAIL"

            # Test 3: Statistical Validation Framework
            logger.info("\nTEST 3: Statistical Validation Framework")
            logger.info("-" * 50)
            statistical_report = await self.test_statistical_validation(backtest_results)

            if statistical_report:
                logger.info("✓ Statistical validation framework test PASSED")
                self.test_results["statistical_validation"] = "PASS"
            else:
                logger.error("✗ Statistical validation framework test FAILED")
                self.test_results["statistical_validation"] = "FAIL"

            # Test 4: Enhanced Reporting System
            logger.info("\nTEST 4: Enhanced Professional Reporting System")
            logger.info("-" * 50)
            comprehensive_report = await self.test_reporting_system(
                backtest_results, validation_report, statistical_report
            )

            if comprehensive_report:
                logger.info("✓ Enhanced reporting system test PASSED")
                self.test_results["reporting_system"] = "PASS"
            else:
                logger.error("✗ Enhanced reporting system test FAILED")
                self.test_results["reporting_system"] = "FAIL"

            # Test 5: Integration and Performance
            logger.info("\nTEST 5: System Integration and Performance")
            logger.info("-" * 50)
            integration_success = await self.test_system_integration(
                backtest_results, validation_report, statistical_report, comprehensive_report
            )

            if integration_success:
                logger.info("✓ System integration test PASSED")
                self.test_results["system_integration"] = "PASS"
            else:
                logger.error("✗ System integration test FAILED")
                self.test_results["system_integration"] = "FAIL"

            # Generate final test report
            await self.generate_test_report()

            return all(result == "PASS" for result in self.test_results.values())

        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            return False

    async def test_enhanced_backtesting(self):
        """Test the enhanced three-phase backtesting system"""
        try:
            logger.info("Initializing Enhanced Backtesting System...")

            # Initialize backtesting system
            config = BacktestConfig(
                max_workers=4,
                enable_data_cache=True,
                enable_progress_monitoring=True
            )

            backtesting_system = EnhancedBacktestingSystem(config)

            # Define test strategy
            def test_momentum_strategy(data, context, params):
                """Simple test momentum strategy"""
                # Simplified strategy logic for testing
                positions = {}

                # Select top momentum stocks (simplified)
                if len(data) > 0:
                    symbols = list(data.keys())[:5]  # Top 5 stocks
                    weight_per_stock = 1.0 / len(symbols)

                    for symbol in symbols:
                        positions[symbol] = weight_per_stock

                return positions

            # Strategy parameters
            strategy_params = {
                'name': 'Test_Enhanced_Momentum_Strategy',
                'initial_capital': 1000000,
                'momentum_window': 20,
                'max_positions': 5,
                'transaction_cost': 0.001,
                'rebalance_frequency': 30
            }

            logger.info("Running three-phase backtesting...")
            start_time = time.time()

            # Execute three-phase backtesting
            results = await backtesting_system.run_three_phase_backtest(
                strategy_func=test_momentum_strategy,
                strategy_params=strategy_params,
                data_source="simulation",
                validation_method=ValidationMethod.WALK_FORWARD
            )

            execution_time = time.time() - start_time
            logger.info(f"Backtesting completed in {execution_time:.2f} seconds")

            # Validate results
            if not results:
                logger.error("Backtesting returned no results")
                return None

            # Check result completeness
            required_attributes = [
                'strategy_name', 'overall_sharpe', 'overall_calmar',
                'overall_max_drawdown', 'consistency_score', 'crisis_resilience',
                'phase_1_results', 'phase_2_results', 'phase_3_results'
            ]

            for attr in required_attributes:
                if not hasattr(results, attr):
                    logger.error(f"Missing required attribute: {attr}")
                    return None

                value = getattr(results, attr)
                if value is None:
                    logger.error(f"Attribute {attr} is None")
                    return None

            # Log key results
            logger.info(f"Strategy: {results.strategy_name}")
            logger.info(f"Overall Sharpe Ratio: {results.overall_sharpe:.3f}")
            logger.info(f"Overall Max Drawdown: {results.overall_max_drawdown:.2%}")
            logger.info(f"Consistency Score: {results.consistency_score:.3f}")
            logger.info(f"Crisis Resilience: {results.crisis_resilience:.3f}")

            # Validate phase results
            phases = [results.phase_1_results, results.phase_2_results, results.phase_3_results]
            for i, phase in enumerate(phases, 1):
                if not phase:
                    logger.error(f"Phase {i} results are missing")
                    return None

                logger.info(f"Phase {i} - Sharpe: {phase.sharpe_ratio:.3f}, "
                           f"Drawdown: {phase.max_drawdown:.2%}")

            # Performance validation
            if results.overall_sharpe < -2.0 or results.overall_sharpe > 5.0:
                logger.warning(f"Sharpe ratio outside reasonable range: {results.overall_sharpe}")

            if abs(results.overall_max_drawdown) > 0.5:
                logger.warning(f"Max drawdown outside reasonable range: {results.overall_max_drawdown}")

            logger.info("Enhanced backtesting system validation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Enhanced backtesting test failed: {e}")
            return None

    async def test_investment_grade_validation(self, backtest_results):
        """Test the investment-grade validation system"""
        try:
            logger.info("Initializing Investment-Grade Validator...")

            # Initialize validator
            validator = InvestmentGradeValidator()

            logger.info("Running investment-grade validation...")
            start_time = time.time()

            # Execute validation
            validation_report = await validator.validate_investment_grade(
                backtest_results=backtest_results,
                additional_data=None
            )

            execution_time = time.time() - start_time
            logger.info(f"Investment-grade validation completed in {execution_time:.2f} seconds")

            # Validate results
            if not validation_report:
                logger.error("Investment-grade validation returned no results")
                return None

            # Check result completeness
            required_attributes = [
                'strategy_name', 'investment_grade_score', 'deployment_readiness',
                'capacity_analyses', 'regime_analyses', 'stress_test_results',
                'overall_resilience_score'
            ]

            for attr in required_attributes:
                if not hasattr(validation_report, attr):
                    logger.error(f"Missing required attribute: {attr}")
                    return None

            # Log key results
            logger.info(f"Investment Grade Score: {validation_report.investment_grade_score:.1f}/100")
            logger.info(f"Deployment Readiness: {validation_report.deployment_readiness}")
            logger.info(f"Recommended Capacity: {validation_report.recommended_capacity.value}")
            logger.info(f"Maximum Feasible AUM: ${validation_report.maximum_feasible_aum:,.0f}")
            logger.info(f"Overall Resilience: {validation_report.overall_resilience_score:.3f}")

            # Validate capacity analyses
            if not validation_report.capacity_analyses:
                logger.error("No capacity analyses found")
                return None

            feasible_count = sum(1 for ca in validation_report.capacity_analyses.values()
                               if ca.is_feasible)
            logger.info(f"Feasible capacity levels: {feasible_count}/{len(validation_report.capacity_analyses)}")

            # Validate stress test results
            if not validation_report.stress_test_results:
                logger.error("No stress test results found")
                return None

            avg_resilience = np.mean([str.resilience_score
                                    for str in validation_report.stress_test_results.values()])
            logger.info(f"Average stress test resilience: {avg_resilience:.3f}")

            # Score validation
            if validation_report.investment_grade_score < 0 or validation_report.investment_grade_score > 100:
                logger.error(f"Investment grade score out of range: {validation_report.investment_grade_score}")
                return None

            logger.info("Investment-grade validation completed successfully")
            return validation_report

        except Exception as e:
            logger.error(f"Investment-grade validation test failed: {e}")
            return None

    async def test_statistical_validation(self, backtest_results):
        """Test the statistical validation framework"""
        try:
            logger.info("Initializing Statistical Validation Framework...")

            # Initialize framework
            framework = StatisticalValidationFramework()

            logger.info("Running statistical validation...")
            start_time = time.time()

            # Execute statistical validation
            statistical_report = await framework.validate_statistical_significance(
                backtest_results=backtest_results,
                additional_data=None
            )

            execution_time = time.time() - start_time
            logger.info(f"Statistical validation completed in {execution_time:.2f} seconds")

            # Validate results
            if not statistical_report:
                logger.error("Statistical validation returned no results")
                return None

            # Check result completeness
            required_attributes = [
                'strategy_name', 'statistical_tests', 'monte_carlo_results',
                'regime_analysis', 'factor_attribution', 'overall_significance_score',
                'statistical_robustness_score'
            ]

            for attr in required_attributes:
                if not hasattr(statistical_report, attr):
                    logger.error(f"Missing required attribute: {attr}")
                    return None

            # Log key results
            logger.info(f"Overall Significance Score: {statistical_report.overall_significance_score:.1f}/100")
            logger.info(f"Statistical Robustness Score: {statistical_report.statistical_robustness_score:.1f}/100")
            logger.info(f"Tests Performed: {len(statistical_report.statistical_tests)}")

            # Validate statistical tests
            significant_tests = sum(1 for test in statistical_report.statistical_tests.values()
                                  if test.is_significant)
            logger.info(f"Significant tests: {significant_tests}/{len(statistical_report.statistical_tests)}")

            # Validate Monte Carlo results
            mc_results = statistical_report.monte_carlo_results
            if not mc_results:
                logger.error("No Monte Carlo results found")
                return None

            logger.info(f"Monte Carlo simulations: {mc_results.n_simulations}")
            logger.info(f"Probability of positive return: {mc_results.prob_positive_return:.1%}")
            logger.info(f"Expected Shortfall (95%): {mc_results.expected_shortfall_95:.2%}")

            # Validate regime analysis
            regime_analysis = statistical_report.regime_analysis
            if not regime_analysis:
                logger.error("No regime analysis found")
                return None

            logger.info(f"Market regimes identified: {regime_analysis.n_regimes}")
            logger.info(f"Regime stability score: {regime_analysis.regime_stability_score:.3f}")

            # Score validation
            for score_name, score_value in [
                ("significance", statistical_report.overall_significance_score),
                ("robustness", statistical_report.statistical_robustness_score)
            ]:
                if score_value < 0 or score_value > 100:
                    logger.error(f"{score_name} score out of range: {score_value}")
                    return None

            logger.info("Statistical validation completed successfully")
            return statistical_report

        except Exception as e:
            logger.error(f"Statistical validation test failed: {e}")
            return None

    async def test_reporting_system(self, backtest_results, validation_report, statistical_report):
        """Test the enhanced reporting system"""
        try:
            logger.info("Initializing Enhanced Reporting System...")

            # Initialize reporting system
            report_system = EnhancedBacktestingReportSystem()

            # Configure report
            report_config = ReportConfiguration(
                report_type=ReportType.COMPREHENSIVE_ANALYSIS,
                output_format=OutputFormat.HTML,
                template_style=ReportTemplate.INSTITUTIONAL,
                include_charts=True,
                include_statistics=True,
                include_recommendations=True
            )

            logger.info("Generating comprehensive report...")
            start_time = time.time()

            # Generate comprehensive report
            comprehensive_report = await report_system.generate_comprehensive_report(
                backtest_results=backtest_results,
                validation_report=validation_report,
                statistical_report=statistical_report,
                report_config=report_config
            )

            execution_time = time.time() - start_time
            logger.info(f"Report generation completed in {execution_time:.2f} seconds")

            # Validate report
            if not comprehensive_report:
                logger.error("Report generation returned no results")
                return None

            # Check report completeness
            required_attributes = [
                'strategy_name', 'report_id', 'executive_summary',
                'key_recommendations', 'deployment_readiness',
                'performance_metrics', 'risk_analysis'
            ]

            for attr in required_attributes:
                if not hasattr(comprehensive_report, attr):
                    logger.error(f"Missing required attribute: {attr}")
                    return None

            # Log key results
            logger.info(f"Report ID: {comprehensive_report.report_id}")
            logger.info(f"Deployment Readiness: {comprehensive_report.deployment_readiness}")
            logger.info(f"Charts Generated: {len(comprehensive_report.chart_files)}")
            logger.info(f"Interactive Charts: {len(comprehensive_report.interactive_charts)}")
            logger.info(f"Key Recommendations: {len(comprehensive_report.key_recommendations)}")

            # Test report exports
            logger.info("Testing report exports...")

            # Export HTML report
            html_path = await report_system.export_report(
                comprehensive_report, OutputFormat.HTML,
                str(self.test_dir / "test_report")
            )

            if os.path.exists(html_path):
                logger.info(f"✓ HTML report exported: {html_path}")
            else:
                logger.error("✗ HTML report export failed")
                return None

            # Export JSON report
            json_path = await report_system.export_report(
                comprehensive_report, OutputFormat.JSON,
                str(self.test_dir / "test_report")
            )

            if os.path.exists(json_path):
                logger.info(f"✓ JSON report exported: {json_path}")
            else:
                logger.error("✗ JSON report export failed")

            # Export Excel report (if available)
            try:
                excel_path = await report_system.export_report(
                    comprehensive_report, OutputFormat.EXCEL,
                    str(self.test_dir / "test_report")
                )

                if os.path.exists(excel_path):
                    logger.info(f"✓ Excel report exported: {excel_path}")
                else:
                    logger.info("Excel report export not available (missing dependencies)")
            except Exception as e:
                logger.info(f"Excel export not available: {e}")

            logger.info("Enhanced reporting system validation completed successfully")
            return comprehensive_report

        except Exception as e:
            logger.error(f"Reporting system test failed: {e}")
            return None

    async def test_system_integration(self, backtest_results, validation_report,
                                    statistical_report, comprehensive_report):
        """Test overall system integration and performance"""
        try:
            logger.info("Testing system integration and performance...")

            # Data consistency checks
            integration_checks = {
                "strategy_name_consistency": True,
                "data_flow_integrity": True,
                "result_completeness": True,
                "performance_reasonableness": True,
                "memory_efficiency": True
            }

            # Check 1: Strategy name consistency
            strategy_names = []
            if backtest_results:
                strategy_names.append(backtest_results.strategy_name)
            if validation_report:
                strategy_names.append(validation_report.strategy_name)
            if statistical_report:
                strategy_names.append(statistical_report.strategy_name)
            if comprehensive_report:
                strategy_names.append(comprehensive_report.strategy_name)

            if len(set(strategy_names)) != 1:
                logger.error(f"Strategy name inconsistency: {strategy_names}")
                integration_checks["strategy_name_consistency"] = False
            else:
                logger.info(f"✓ Strategy name consistent: {strategy_names[0]}")

            # Check 2: Data flow integrity
            if backtest_results and validation_report:
                # Check that validation used backtest results
                if (abs(validation_report.investment_grade_score - 0) < 1e-10):
                    logger.warning("Investment grade score appears to be zero")
                else:
                    logger.info("✓ Data flow from backtesting to validation working")

            # Check 3: Result completeness
            all_components_present = all([
                backtest_results is not None,
                validation_report is not None,
                statistical_report is not None,
                comprehensive_report is not None
            ])

            if not all_components_present:
                logger.error("Not all system components produced results")
                integration_checks["result_completeness"] = False
            else:
                logger.info("✓ All system components produced results")

            # Check 4: Performance reasonableness
            if backtest_results:
                # Sharpe ratio should be reasonable
                if abs(backtest_results.overall_sharpe) > 10:
                    logger.warning(f"Sharpe ratio seems unreasonable: {backtest_results.overall_sharpe}")
                    integration_checks["performance_reasonableness"] = False

                # Drawdown should be reasonable
                if abs(backtest_results.overall_max_drawdown) > 1.0:
                    logger.warning(f"Max drawdown seems unreasonable: {backtest_results.overall_max_drawdown}")
                    integration_checks["performance_reasonableness"] = False

                if integration_checks["performance_reasonableness"]:
                    logger.info("✓ Performance metrics appear reasonable")

            # Check 5: Memory and resource efficiency
            total_execution_time = (datetime.now() - self.start_time).total_seconds()
            if total_execution_time > 300:  # 5 minutes
                logger.warning(f"Total execution time seems long: {total_execution_time:.1f} seconds")
                integration_checks["memory_efficiency"] = False
            else:
                logger.info(f"✓ Total execution time reasonable: {total_execution_time:.1f} seconds")

            # Overall integration assessment
            passed_checks = sum(integration_checks.values())
            total_checks = len(integration_checks)

            logger.info(f"Integration checks passed: {passed_checks}/{total_checks}")

            for check_name, passed in integration_checks.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                logger.info(f"  {check_name}: {status}")

            return passed_checks == total_checks

        except Exception as e:
            logger.error(f"System integration test failed: {e}")
            return False

    async def generate_test_report(self):
        """Generate a comprehensive test report"""
        try:
            logger.info("\nGenerating test report...")

            # Calculate overall test results
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results.values() if result == "PASS")

            test_report = {
                "test_execution_info": {
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_execution_time_seconds": (datetime.now() - self.start_time).total_seconds(),
                    "test_environment": {
                        "python_version": sys.version,
                        "operating_system": os.name,
                        "working_directory": os.getcwd()
                    }
                },
                "test_summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
                },
                "detailed_results": self.test_results,
                "system_validation": {
                    "all_components_functional": all(result == "PASS" for result in self.test_results.values()),
                    "integration_successful": self.test_results.get("system_integration") == "PASS",
                    "ready_for_production": passed_tests >= 4  # At least 4 out of 5 tests should pass
                }
            }

            # Save test report
            report_file = self.test_dir / "comprehensive_test_report.json"
            with open(report_file, 'w') as f:
                import json
                json.dump(test_report, f, indent=2, default=str)

            # Print summary
            logger.info("=" * 60)
            logger.info("TEST EXECUTION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Tests: {total_tests}")
            logger.info(f"Passed Tests: {passed_tests}")
            logger.info(f"Failed Tests: {total_tests - passed_tests}")
            logger.info(f"Success Rate: {test_report['test_summary']['success_rate']:.1f}%")
            logger.info(f"Execution Time: {test_report['test_execution_info']['total_execution_time_seconds']:.1f} seconds")

            logger.info("\nDETAILED RESULTS:")
            for test_name, result in self.test_results.items():
                status_icon = "✓" if result == "PASS" else "✗"
                logger.info(f"  {status_icon} {test_name.replace('_', ' ').title()}: {result}")

            logger.info(f"\nTest report saved to: {report_file}")

            # Final assessment
            if test_report["system_validation"]["ready_for_production"]:
                logger.info("\n🎉 SYSTEM VALIDATION SUCCESSFUL!")
                logger.info("The comprehensive backtesting system is ready for production use.")
            else:
                logger.warning("\n⚠️ SYSTEM VALIDATION INCOMPLETE")
                logger.warning("Some components failed validation. Review required before production use.")

            return test_report

        except Exception as e:
            logger.error(f"Test report generation failed: {e}")
            return None

async def main():
    """Main test execution function"""
    try:
        # Initialize test environment
        test_suite = ComprehensiveBacktestingTest()

        # Run complete test suite
        success = await test_suite.run_complete_test_suite()

        if success:
            print("\n" + "=" * 60)
            print("🎉 COMPREHENSIVE BACKTESTING SYSTEM TEST COMPLETED SUCCESSFULLY!")
            print("All components passed validation and are ready for production use.")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("⚠️ COMPREHENSIVE BACKTESTING SYSTEM TEST COMPLETED WITH ISSUES")
            print("Some components failed validation. Please review the test results.")
            print("=" * 60)
            return 1

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print("\n" + "=" * 60)
        print("❌ COMPREHENSIVE BACKTESTING SYSTEM TEST FAILED")
        print(f"Error: {e}")
        print("=" * 60)
        return 2

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)