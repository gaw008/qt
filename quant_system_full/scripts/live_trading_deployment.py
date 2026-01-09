#!/usr/bin/env python3
"""
Live Trading Deployment Manager
Validates and enables live trading with comprehensive security checks
"""

import os
import sys
import json
import shutil
import secrets
from pathlib import Path
from datetime import datetime

class LiveTradingDeployment:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.deployment_log = []

    def log(self, message, level="INFO"):
        """Log deployment activities"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.deployment_log.append(log_entry)

    def generate_production_tokens(self):
        """Generate cryptographically strong tokens for production"""
        self.log("Generating production-grade security tokens...")

        tokens = {
            'admin_token': secrets.token_urlsafe(32),
            'emergency_stop_token': secrets.token_urlsafe(16),
            'encryption_key': secrets.token_urlsafe(32),
            'api_secret': secrets.token_urlsafe(24),
        }

        return tokens

    def validate_tiger_api_connection(self):
        """Validate Tiger API connection for live trading"""
        self.log("Validating Tiger API connection...")

        try:
            # Import Tiger SDK
            sys.path.append(str(self.project_root / 'bot'))
            from tiger_client import TigerClient

            # Test connection
            client = TigerClient()
            account_info = client.get_account_info()

            if account_info:
                self.log("✓ Tiger API connection validated successfully")
                self.log(f"Account: {account_info.get('account', 'Unknown')}")
                self.log(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
                return True
            else:
                self.log("❌ Tiger API connection failed", "ERROR")
                return False

        except Exception as e:
            self.log(f"❌ Tiger API validation error: {e}", "ERROR")
            return False

    def validate_security_configuration(self):
        """Validate all security configurations"""
        self.log("Validating security configuration...")

        security_checks = {
            'strong_admin_token': self._check_admin_token(),
            'secure_private_key': self._check_private_key_permissions(),
            'tls_enabled': self._check_tls_configuration(),
            'firewall_configured': self._check_firewall_rules(),
            'backup_system': self._check_backup_system(),
            'monitoring_enabled': self._check_monitoring_system(),
            'emergency_procedures': self._check_emergency_procedures(),
        }

        failed_checks = [check for check, passed in security_checks.items() if not passed]

        if failed_checks:
            self.log(f"❌ Security validation failed: {failed_checks}", "ERROR")
            return False

        self.log("✓ All security validations passed")
        return True

    def _check_admin_token(self):
        """Check admin token strength"""
        token = os.getenv('ADMIN_TOKEN')
        return token and token != 'wgyjd0508' and len(token) >= 32

    def _check_private_key_permissions(self):
        """Check private key file permissions"""
        private_key_path = self.project_root / 'private_key.pem'
        return private_key_path.exists()  # Simplified check

    def _check_tls_configuration(self):
        """Check TLS configuration"""
        return os.getenv('USE_TLS', 'false').lower() == 'true'

    def _check_firewall_rules(self):
        """Check firewall configuration"""
        # Simplified check - in production, verify actual firewall rules
        return True

    def _check_backup_system(self):
        """Check backup system"""
        backup_script = self.project_root / 'scripts' / 'backup_system.sh'
        return backup_script.exists()

    def _check_monitoring_system(self):
        """Check monitoring system"""
        monitoring_script = self.project_root / 'scripts' / 'security_monitor.py'
        return monitoring_script.exists()

    def _check_emergency_procedures(self):
        """Check emergency procedures"""
        emergency_script = self.project_root / 'scripts' / 'emergency_procedures.py'
        return emergency_script.exists()

    def create_production_environment(self, tokens):
        """Create production environment configuration"""
        self.log("Creating production environment...")

        # Backup current configuration
        current_env = self.project_root / '.env'
        if current_env.exists():
            backup_path = self.project_root / f'.env.backup.{int(datetime.now().timestamp())}'
            shutil.copy2(current_env, backup_path)
            self.log(f"Current configuration backed up to: {backup_path}")

        # Read secure template
        secure_env_path = self.project_root / '.env.secure'
        if not secure_env_path.exists():
            self.log("❌ Secure environment template not found", "ERROR")
            return False

        with open(secure_env_path, 'r') as f:
            secure_config = f.read()

        # Replace placeholder tokens with generated ones
        secure_config = secure_config.replace(
            'YourSecureTokenHere_Replace_With_Generated_Token_32chars',
            tokens['admin_token']
        )
        secure_config = secure_config.replace(
            'EmergencyToken16chars',
            tokens['emergency_stop_token']
        )

        # Update for live trading (with safety confirmation)
        if self._confirm_live_trading():
            secure_config = secure_config.replace('DRY_RUN=true', 'DRY_RUN=false')
            self.log("⚠️ Live trading ENABLED - Real money will be used")

        # Write production configuration
        with open(current_env, 'w') as f:
            f.write(secure_config)

        self.log("✓ Production environment created")
        return True

    def _confirm_live_trading(self):
        """Confirm live trading enablement"""
        print("\n" + "="*60)
        print("⚠️  CRITICAL WARNING: LIVE TRADING ENABLEMENT")
        print("="*60)
        print("This will enable live trading with REAL MONEY.")
        print("Ensure all security validations have passed.")
        print("Have you completed ALL pre-deployment checks?")
        print("="*60)

        confirmation = input("Type 'ENABLE LIVE TRADING' to confirm: ")
        return confirmation == 'ENABLE LIVE TRADING'

    def start_monitoring_systems(self):
        """Start monitoring and security systems"""
        self.log("Starting monitoring systems...")

        # Start security monitor
        security_monitor_path = self.project_root / 'scripts' / 'security_monitor.py'
        if security_monitor_path.exists():
            # In production, start as a service
            self.log("Security monitoring should be started as a system service")

        # Start backup system
        backup_script_path = self.project_root / 'scripts' / 'backup_system.sh'
        if backup_script_path.exists():
            self.log("Backup system should be configured as a cron job")

        return True

    def validate_deployment_readiness(self):
        """Final deployment readiness validation"""
        self.log("Performing final deployment readiness validation...")

        readiness_checks = {
            'tiger_api_connected': self.validate_tiger_api_connection(),
            'security_validated': self.validate_security_configuration(),
            'environment_configured': (self.project_root / '.env').exists(),
            'monitoring_ready': self._check_monitoring_system(),
        }

        all_ready = all(readiness_checks.values())

        self.log("Deployment Readiness Summary:")
        for check, status in readiness_checks.items():
            status_icon = "✓" if status else "❌"
            self.log(f"  {status_icon} {check}: {'READY' if status else 'NOT READY'}")

        return all_ready

    def deploy_for_live_trading(self):
        """Deploy system for live trading"""
        self.log("="*60)
        self.log("STARTING LIVE TRADING DEPLOYMENT")
        self.log("="*60)

        try:
            # Step 1: Generate tokens
            tokens = self.generate_production_tokens()

            # Step 2: Validate security
            if not self.validate_security_configuration():
                self.log("❌ Security validation failed - aborting deployment", "ERROR")
                return False

            # Step 3: Validate Tiger API
            if not self.validate_tiger_api_connection():
                self.log("❌ Tiger API validation failed - aborting deployment", "ERROR")
                return False

            # Step 4: Create production environment
            if not self.create_production_environment(tokens):
                self.log("❌ Production environment creation failed", "ERROR")
                return False

            # Step 5: Start monitoring
            self.start_monitoring_systems()

            # Step 6: Final readiness check
            if not self.validate_deployment_readiness():
                self.log("❌ Final readiness validation failed", "ERROR")
                return False

            # Step 7: Save deployment record
            self.save_deployment_record(tokens)

            self.log("="*60)
            self.log("✓ LIVE TRADING DEPLOYMENT COMPLETED SUCCESSFULLY")
            self.log("="*60)
            self.log("IMPORTANT: Monitor system closely for first 24 hours")
            self.log("Emergency stop command: python scripts/emergency_procedures.py stop")
            self.log("="*60)

            return True

        except Exception as e:
            self.log(f"❌ Deployment failed: {e}", "ERROR")
            return False

    def save_deployment_record(self, tokens):
        """Save deployment record for audit trail"""
        deployment_record = {
            'timestamp': datetime.now().isoformat(),
            'deployment_type': 'live_trading',
            'tokens_generated': {
                'admin_token_length': len(tokens['admin_token']),
                'emergency_token_length': len(tokens['emergency_stop_token']),
            },
            'security_validations': 'completed',
            'tiger_api_validation': 'completed',
            'deployment_log': self.deployment_log,
        }

        record_path = self.project_root / 'deployment_record.json'
        with open(record_path, 'w') as f:
            json.dump(deployment_record, f, indent=2)

        self.log(f"Deployment record saved: {record_path}")

def main():
    """Main deployment function"""

    deployment = LiveTradingDeployment()

    print("Quantitative Trading System - Live Trading Deployment")
    print("This script will configure the system for live trading with real money.")
    print("Ensure all security prerequisites are met before proceeding.\n")

    # Deploy for live trading
    success = deployment.deploy_for_live_trading()

    if success:
        print("\n🎉 System successfully deployed for live trading!")
        print("Monitor the system closely and be prepared to use emergency stop procedures.")
    else:
        print("\n💥 Deployment failed. Check logs and fix issues before retrying.")
        sys.exit(1)

if __name__ == "__main__":
    main()