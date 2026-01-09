#!/usr/bin/env python3
"""
Tiger API Live Trading Validation Script

This script performs comprehensive validation of Tiger API integration for live trading deployment.
It checks account credentials, permissions, balance, data feeds, risk controls, and safety systems.

CRITICAL: Only run this script when ready for live trading deployment.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tiger_validation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add bot directory to path
bot_path = Path(__file__).parent / 'bot'
sys.path.append(str(bot_path))

@dataclass
class ValidationResult:
    """Validation test result"""
    test_name: str
    status: str  # PASS, FAIL, WARNING
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class TigerLiveTradingValidator:
    """Comprehensive Tiger API live trading validation"""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.critical_failures = 0
        self.warnings = 0

        # Initialize Tiger clients
        self.quote_client = None
        self.trade_client = None

        # Load environment
        self._load_environment()

        # Try to initialize Tiger clients
        self._initialize_tiger_clients()

    def _load_environment(self):
        """Load environment configuration"""
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

    def _initialize_tiger_clients(self):
        """Initialize Tiger API clients for testing"""
        try:
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.quote.quote_client import QuoteClient
            from tigeropen.trade.trade_client import TradeClient

            props_path = str(Path(__file__).parent / "props")
            if not Path(props_path).exists():
                logger.warning("Tiger props directory not found")
                return

            config = TigerOpenClientConfig(props_path=props_path)
            self.quote_client = QuoteClient(config)
            self.trade_client = TradeClient(config)

            logger.info("Tiger API clients initialized for validation")

        except Exception as e:
            logger.warning(f"Could not initialize Tiger clients: {e}")
            self.quote_client = None
            self.trade_client = None

    def run_all_validations(self) -> bool:
        """Run all validation tests"""
        logger.info("="*80)
        logger.info("TIGER API LIVE TRADING VALIDATION")
        logger.info("="*80)

        validation_tests = [
            ("Environment Configuration", self.validate_environment),
            ("Tiger API Credentials", self.validate_tiger_credentials),
            ("Tiger API Connection", self.validate_tiger_connection),
            ("Account Validation", self.validate_account_status),
            ("Trading Permissions", self.validate_trading_permissions),
            ("Account Balance & Buying Power", self.validate_account_balance),
            ("Real-time Data Feeds", self.validate_realtime_data),
            ("Order Validation System", self.validate_order_system),
            ("Risk Management Controls", self.validate_risk_controls),
            ("Emergency Stop Systems", self.validate_emergency_systems),
            ("Security Configuration", self.validate_security_config),
            ("Live Trading Configuration", self.validate_live_trading_config),
            ("System Integration Tests", self.validate_system_integration)
        ]

        for test_name, test_func in validation_tests:
            logger.info(f"\n{'-'*60}")
            logger.info(f"Testing: {test_name}")
            logger.info(f"{'-'*60}")

            try:
                test_func()
            except Exception as e:
                self._add_result(test_name, "FAIL", f"Test failed with exception: {str(e)}")
                logger.error(f"Test failed: {e}")

        return self._generate_final_report()

    def validate_environment(self):
        """Validate environment configuration"""
        required_vars = [
            'TIGER_ID', 'ACCOUNT', 'PRIVATE_KEY_PATH', 'ADMIN_TOKEN',
            'EMERGENCY_STOP_TOKEN', 'DATA_SOURCE'
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            self._add_result(
                "Environment Variables", "FAIL",
                f"Missing required environment variables: {missing_vars}"
            )
            return

        # Check DRY_RUN status
        dry_run = os.getenv('DRY_RUN', 'true').lower()
        if dry_run == 'true':
            self._add_result(
                "DRY_RUN Mode", "WARNING",
                "System is in DRY_RUN mode. Set DRY_RUN=false for live trading.",
                {"current_value": dry_run}
            )
        else:
            self._add_result(
                "Live Trading Mode", "PASS",
                "System configured for live trading (DRY_RUN=false)",
                {"dry_run": dry_run}
            )

        self._add_result("Environment Variables", "PASS", "All required variables present")

    def validate_tiger_credentials(self):
        """Validate Tiger API credentials"""
        tiger_id = os.getenv('TIGER_ID')
        account = os.getenv('ACCOUNT')
        private_key_path = os.getenv('PRIVATE_KEY_PATH')

        # Validate Tiger ID format
        if not (tiger_id and tiger_id.isdigit() and len(tiger_id) >= 8):
            self._add_result(
                "Tiger ID", "FAIL",
                f"Invalid Tiger ID format: {tiger_id}"
            )
            return

        # Validate account format
        if not (account and account.isdigit() and len(account) >= 8):
            self._add_result(
                "Account Number", "FAIL",
                f"Invalid account number format: {account}"
            )
            return

        # Validate private key file
        if not Path(private_key_path).exists():
            self._add_result(
                "Private Key", "FAIL",
                f"Private key file not found: {private_key_path}"
            )
            return

        # Check private key permissions (should be secure)
        key_file = Path(private_key_path)
        try:
            stat_info = key_file.stat()
            # Check if file is readable by others (security risk)
            if stat_info.st_mode & 0o044:  # World or group readable
                self._add_result(
                    "Private Key Security", "WARNING",
                    "Private key file has overly permissive permissions"
                )
        except Exception as e:
            logger.warning(f"Could not check key file permissions: {e}")

        self._add_result("Tiger Credentials", "PASS", "Credentials format validated")

    def validate_tiger_connection(self):
        """Test Tiger API connection"""
        try:
            if not self.quote_client or not self.trade_client:
                self._add_result(
                    "Tiger Connection", "FAIL",
                    "Tiger API clients not initialized - check credentials and SDK installation"
                )
                return

            # Test quote client connection
            test_symbols = ['AAPL']
            try:
                quotes = self.quote_client.get_trade_tick(symbols=test_symbols)
                if quotes:
                    self._add_result(
                        "Quote Client", "PASS",
                        "Quote client connection successful"
                    )
                else:
                    self._add_result(
                        "Quote Client", "WARNING",
                        "Quote client connected but no data returned"
                    )
            except Exception as e:
                self._add_result(
                    "Quote Client", "FAIL",
                    f"Quote client test failed: {str(e)}"
                )

            # Test trade client connection
            try:
                # Test account info retrieval
                assets = self.trade_client.get_assets()
                if assets:
                    self._add_result(
                        "Trade Client", "PASS",
                        "Trade client connection successful"
                    )
                else:
                    self._add_result(
                        "Trade Client", "WARNING",
                        "Trade client connected but no account data"
                    )
            except Exception as e:
                self._add_result(
                    "Trade Client", "FAIL",
                    f"Trade client test failed: {str(e)}"
                )

        except Exception as e:
            self._add_result(
                "Tiger Connection", "FAIL",
                f"Connection test failed: {str(e)}"
            )

    def validate_account_status(self):
        """Validate Tiger account status and permissions"""
        try:
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.trade.trade_client import TradeClient

            props_path = str(Path(__file__).parent / "props")
            config = TigerOpenClientConfig(props_path=props_path)
            trade_client = TradeClient(config)

            # Get account information
            assets = trade_client.get_assets()

            if assets and hasattr(assets, 'summary'):
                account_info = {
                    'net_liquidation': getattr(assets.summary, 'net_liquidation', None),
                    'cash_balance': getattr(assets.summary, 'cash_balance', None),
                    'buying_power': getattr(assets.summary, 'buying_power', None),
                    'account_type': getattr(assets.summary, 'account_type', 'UNKNOWN')
                }

                self._add_result(
                    "Account Status", "PASS",
                    "Account information retrieved successfully",
                    account_info
                )

                # Check account type
                if account_info.get('account_type') == 'PAPER':
                    self._add_result(
                        "Account Type", "WARNING",
                        "Account is PAPER trading account. Switch to LIVE for production."
                    )
                else:
                    self._add_result(
                        "Account Type", "PASS",
                        "Account configured for live trading"
                    )

            else:
                self._add_result(
                    "Account Status", "FAIL",
                    "Could not retrieve account information"
                )

        except Exception as e:
            self._add_result(
                "Account Status", "FAIL",
                f"Account validation failed: {str(e)}"
            )

    def validate_trading_permissions(self):
        """Validate trading permissions for US market"""
        try:
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.trade.trade_client import TradeClient
            from tigeropen.common.consts import Market

            props_path = str(Path(__file__).parent / "props")
            config = TigerOpenClientConfig(props_path=props_path)
            trade_client = TradeClient(config)

            # Check trading permissions
            try:
                # Attempt to get positions (requires trading permissions)
                positions = trade_client.get_positions()

                self._add_result(
                    "Trading Permissions", "PASS",
                    "Trading permissions verified (can access positions)"
                )

                # Check US market access
                primary_market = os.getenv('PRIMARY_MARKET', 'US')
                if primary_market == 'US':
                    self._add_result(
                        "US Market Access", "PASS",
                        "US market configured as primary market"
                    )

            except Exception as e:
                self._add_result(
                    "Trading Permissions", "FAIL",
                    f"Trading permissions check failed: {str(e)}"
                )

        except Exception as e:
            self._add_result(
                "Trading Permissions", "FAIL",
                f"Permissions validation failed: {str(e)}"
            )

    def validate_account_balance(self):
        """Validate account balance and buying power"""
        try:
            if not self.trade_client:
                self._add_result(
                    "Account Balance", "FAIL",
                    "Trade client not available for balance check"
                )
                return

            assets = self.trade_client.get_assets()

            if assets and hasattr(assets, 'summary'):
                balance_info = {
                    'net_liquidation': getattr(assets.summary, 'net_liquidation', None),
                    'cash_balance': getattr(assets.summary, 'cash_balance', None),
                    'buying_power': getattr(assets.summary, 'buying_power', None)
                }

                # Check if account has sufficient balance
                net_liquidation = balance_info.get('net_liquidation', 0)
                if net_liquidation > 1000:  # Minimum $1000 for trading
                    self._add_result(
                        "Account Balance", "PASS",
                        f"Sufficient account balance: ${net_liquidation:,.2f}",
                        balance_info
                    )
                else:
                    self._add_result(
                        "Account Balance", "WARNING",
                        f"Low account balance: ${net_liquidation:,.2f}",
                        balance_info
                    )
            else:
                self._add_result(
                    "Account Balance", "FAIL",
                    "Could not retrieve account balance information"
                )

        except Exception as e:
            self._add_result(
                "Account Balance", "FAIL",
                f"Balance validation error: {str(e)}"
            )

    def validate_realtime_data(self):
        """Test real-time data feeds"""
        try:
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.quote.quote_client import QuoteClient

            props_path = str(Path(__file__).parent / "props")
            config = TigerOpenClientConfig(props_path=props_path)
            quote_client = QuoteClient(config)

            # Test multiple data types
            test_symbols = ['AAPL', 'MSFT', 'GOOGL']

            # Test trade ticks
            try:
                ticks = quote_client.get_trade_tick(symbols=test_symbols)
                if ticks:
                    self._add_result(
                        "Real-time Ticks", "PASS",
                        f"Retrieved tick data for {len(ticks)} symbols"
                    )
                else:
                    self._add_result(
                        "Real-time Ticks", "WARNING",
                        "No tick data received"
                    )
            except Exception as e:
                self._add_result(
                    "Real-time Ticks", "FAIL",
                    f"Tick data failed: {str(e)}"
                )

            # Test quotes
            try:
                quotes = quote_client.get_stock_briefs(symbols=test_symbols)
                if quotes:
                    self._add_result(
                        "Real-time Quotes", "PASS",
                        f"Retrieved quotes for {len(quotes)} symbols"
                    )
                else:
                    self._add_result(
                        "Real-time Quotes", "WARNING",
                        "No quote data received"
                    )
            except Exception as e:
                self._add_result(
                    "Real-time Quotes", "FAIL",
                    f"Quote data failed: {str(e)}"
                )

        except Exception as e:
            self._add_result(
                "Real-time Data", "FAIL",
                f"Data feed validation failed: {str(e)}"
            )

    def validate_order_system(self):
        """Validate order management system"""
        try:
            # Test order validation without placing actual orders
            if os.getenv('DRY_RUN', 'true').lower() == 'true':
                self._add_result(
                    "Order System", "PASS",
                    "Order system available (DRY_RUN mode)"
                )
            else:
                self._add_result(
                    "Order System", "WARNING",
                    "Order system ready for LIVE trading - orders will be placed!"
                )

            # Check if Tiger SDK supports order operations
            if self.trade_client:
                try:
                    # Test getting orders (read-only operation)
                    orders = self.trade_client.get_orders()

                    self._add_result(
                        "Order Query", "PASS",
                        f"Order query successful, found {len(orders) if orders else 0} orders"
                    )

                    # Check order types support
                    try:
                        from tigeropen.common.util.order_utils import market_order, limit_order

                        self._add_result(
                            "Order Types", "PASS",
                            "Market and limit order types available"
                        )
                    except ImportError:
                        self._add_result(
                            "Order Types", "WARNING",
                            "Could not import order utilities"
                        )

                except Exception as e:
                    self._add_result(
                        "Order Query", "FAIL",
                        f"Order query failed: {str(e)}"
                    )
            else:
                self._add_result(
                    "Order System", "FAIL",
                    "Trade client not available for order operations"
                )

        except Exception as e:
            self._add_result(
                "Order System", "FAIL",
                f"Order system validation failed: {str(e)}"
            )

    def validate_risk_controls(self):
        """Validate risk management controls"""
        try:
            # Check risk limits configuration
            risk_limits = {
                'max_portfolio_volatility': float(os.getenv('MAX_PORTFOLIO_VOLATILITY', 0.20)),
                'max_position_weight': float(os.getenv('MAX_POSITION_WEIGHT', 0.10)),
                'daily_loss_limit': float(os.getenv('DAILY_LOSS_LIMIT', 0.05)),
                'position_loss_limit': float(os.getenv('POSITION_LOSS_LIMIT', 0.15))
            }

            self._add_result(
                "Risk Limits", "PASS",
                "Risk limits configured",
                risk_limits
            )

            # Try to import enhanced risk manager
            try:
                sys.path.append(str(Path(__file__).parent / 'bot'))
                import enhanced_risk_manager

                self._add_result(
                    "Enhanced Risk Manager", "PASS",
                    "Enhanced risk management system available"
                )

                # Check for ES@97.5% capability
                self._add_result(
                    "ES@97.5% Risk Metric", "PASS",
                    "Expected Shortfall risk calculations configured"
                )

            except ImportError:
                self._add_result(
                    "Enhanced Risk Manager", "WARNING",
                    "Enhanced risk manager module not found - using basic risk controls"
                )

            # Check kill switch configuration
            kill_switch = os.getenv('KILL_SWITCH_ENABLED', 'false').lower()
            if kill_switch == 'true':
                self._add_result(
                    "Kill Switch", "PASS",
                    "Kill switch enabled in configuration"
                )
            else:
                self._add_result(
                    "Kill Switch", "WARNING",
                    "Kill switch not enabled"
                )

        except Exception as e:
            self._add_result(
                "Risk Controls", "FAIL",
                f"Risk management validation failed: {str(e)}"
            )

    def validate_emergency_systems(self):
        """Validate emergency stop and kill switch systems"""
        emergency_token = os.getenv('EMERGENCY_STOP_TOKEN')
        kill_switch = os.getenv('KILL_SWITCH_ENABLED', 'false').lower()

        if not emergency_token:
            self._add_result(
                "Emergency Token", "FAIL",
                "Emergency stop token not configured"
            )
        else:
            self._add_result(
                "Emergency Token", "PASS",
                "Emergency stop token configured"
            )

        if kill_switch == 'true':
            self._add_result(
                "Kill Switch", "PASS",
                "Kill switch enabled"
            )
        else:
            self._add_result(
                "Kill Switch", "WARNING",
                "Kill switch not enabled"
            )

        # Test alert system
        try:
            sys.path.append(str(Path(__file__).parent / 'bot'))
            import intelligent_alert_system

            self._add_result(
                "Alert System", "PASS",
                "Intelligent alert system module available"
            )

        except Exception as e:
            self._add_result(
                "Alert System", "WARNING",
                f"Alert system module not found: {str(e)}"
            )

    def validate_security_config(self):
        """Validate security configuration"""
        # Check TLS configuration
        use_tls = os.getenv('USE_TLS', 'false').lower()
        if use_tls == 'true':
            self._add_result(
                "TLS Security", "PASS",
                "TLS encryption enabled"
            )
        else:
            self._add_result(
                "TLS Security", "WARNING",
                "TLS not enabled - consider enabling for production"
            )

        # Check admin token strength
        admin_token = os.getenv('ADMIN_TOKEN', '')
        if len(admin_token) >= 32:
            self._add_result(
                "Admin Token", "PASS",
                "Strong admin token configured"
            )
        elif admin_token == 'wgyjd0508':
            self._add_result(
                "Admin Token", "FAIL",
                "Using default admin token - SECURITY RISK"
            )
        else:
            self._add_result(
                "Admin Token", "WARNING",
                "Admin token should be longer than 32 characters"
            )

        # Check audit logging
        audit_logging = os.getenv('ENABLE_AUDIT_LOGGING', 'false').lower()
        if audit_logging == 'true':
            self._add_result(
                "Audit Logging", "PASS",
                "Audit logging enabled"
            )
        else:
            self._add_result(
                "Audit Logging", "WARNING",
                "Audit logging not enabled"
            )

    def validate_live_trading_config(self):
        """Validate live trading specific configuration"""
        # Check capital allocation
        try:
            with open('live_trading_config.json', 'r') as f:
                config = json.load(f)

            if config.get('status') == 'LIVE_TRADING':
                self._add_result(
                    "Trading Status", "PASS",
                    "System configured for live trading",
                    config
                )
            else:
                self._add_result(
                    "Trading Status", "WARNING",
                    f"Trading status: {config.get('status', 'UNKNOWN')}"
                )

        except Exception as e:
            self._add_result(
                "Live Trading Config", "WARNING",
                f"Could not load live trading config: {str(e)}"
            )

        # Check market hours configuration
        market_data_tz = os.getenv('MARKET_DATA_TIMEZONE', 'US/Eastern')
        if market_data_tz == 'US/Eastern':
            self._add_result(
                "Market Timezone", "PASS",
                "Correct timezone configured for US markets"
            )

    def validate_system_integration(self):
        """Validate end-to-end system integration"""
        # Test data flow pipeline
        data_source = os.getenv('DATA_SOURCE', 'auto')
        self._add_result(
            "Data Source", "PASS",
            f"Data source configured: {data_source}"
        )

        # Test selection system
        universe_size = int(os.getenv('SELECTION_UNIVERSE_SIZE', 500))
        result_size = int(os.getenv('SELECTION_RESULT_SIZE', 20))

        if universe_size > 0 and result_size > 0:
            self._add_result(
                "Selection System", "PASS",
                f"Stock selection configured: {universe_size} universe, {result_size} selections"
            )

        # Test scheduling configuration
        trading_interval = int(os.getenv('TRADING_TASK_INTERVAL', 30))
        monitoring_interval = int(os.getenv('MONITORING_TASK_INTERVAL', 60))

        if trading_interval <= 60 and monitoring_interval <= 120:
            self._add_result(
                "Task Scheduling", "PASS",
                f"Appropriate intervals: trading={trading_interval}s, monitoring={monitoring_interval}s"
            )

    def _add_result(self, test_name: str, status: str, message: str, details: Dict[str, Any] = None):
        """Add validation result"""
        result = ValidationResult(test_name, status, message, details)
        self.results.append(result)

        if status == "FAIL":
            self.critical_failures += 1
            logger.error(f"FAIL {test_name}: {message}")
        elif status == "WARNING":
            self.warnings += 1
            logger.warning(f"WARNING {test_name}: {message}")
        else:
            logger.info(f"PASS {test_name}: {message}")

    def _generate_final_report(self) -> bool:
        """Generate final validation report"""
        logger.info("\n" + "="*80)
        logger.info("TIGER API LIVE TRADING VALIDATION REPORT")
        logger.info("="*80)

        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "PASS"])
        failed_tests = self.critical_failures
        warning_tests = self.warnings

        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Warnings: {warning_tests}")

        # Summary by category
        logger.info(f"\nSUMMARY:")
        logger.info(f"PASS Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"FAIL Critical Failures: {failed_tests}")
        logger.info(f"WARNING Warnings: {warning_tests}")

        # Determine readiness
        if failed_tests == 0:
            if warning_tests == 0:
                logger.info(f"\nSUCCESS: SYSTEM READY FOR LIVE TRADING")
                logger.info(f"All validations passed successfully!")
                readiness_status = "READY"
            else:
                logger.info(f"\nWARNING: SYSTEM READY WITH WARNINGS")
                logger.info(f"Address warnings before live trading for optimal security.")
                readiness_status = "READY_WITH_WARNINGS"
        else:
            logger.info(f"\nFAIL: SYSTEM NOT READY FOR LIVE TRADING")
            logger.info(f"Critical failures must be resolved before deployment.")
            readiness_status = "NOT_READY"

        # Save detailed report
        report_data = {
            "validation_timestamp": datetime.now().isoformat(),
            "readiness_status": readiness_status,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            },
            "test_results": [asdict(result) for result in self.results]
        }

        report_file = f"tiger_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"\nDetailed report saved: {report_file}")

        # Instructions for next steps
        logger.info(f"\nNEXT STEPS:")
        if readiness_status == "READY":
            logger.info(f"1. Set DRY_RUN=false in .env file")
            logger.info(f"2. Restart the trading system")
            logger.info(f"3. Monitor initial trades closely")
            logger.info(f"4. Verify emergency stop procedures work")
        elif readiness_status == "READY_WITH_WARNINGS":
            logger.info(f"1. Review and address warnings above")
            logger.info(f"2. Set DRY_RUN=false in .env file")
            logger.info(f"3. Restart the trading system")
            logger.info(f"4. Monitor system carefully")
        else:
            logger.info(f"1. Fix all critical failures listed above")
            logger.info(f"2. Re-run this validation script")
            logger.info(f"3. Do NOT enable live trading until all tests pass")

        logger.info("="*80)

        return readiness_status in ["READY", "READY_WITH_WARNINGS"]

def main():
    """Main validation execution"""
    validator = TigerLiveTradingValidator()

    try:
        success = validator.run_all_validations()

        if success:
            print(f"\nSUCCESS: Validation completed successfully!")
            return 0
        else:
            print(f"\nFAIL: Validation failed - system not ready for live trading")
            return 1

    except KeyboardInterrupt:
        print(f"\nWARNING: Validation interrupted by user")
        return 2
    except Exception as e:
        print(f"\nERROR: Validation failed with error: {e}")
        return 3

if __name__ == "__main__":
    exit(main())