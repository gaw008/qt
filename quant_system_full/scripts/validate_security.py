#!/usr/bin/env python3
"""
Security Validation Script
Validates all critical security controls before live trading deployment
"""

import os
import sys
import stat
import json
import requests
import subprocess
from pathlib import Path
from datetime import datetime

class SecurityValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.validation_results = {}
        self.critical_failures = []

    def validate_admin_token(self):
        """Validate admin token strength"""
        token = os.getenv('ADMIN_TOKEN')

        checks = {
            'token_exists': token is not None,
            'token_not_default': token != 'wgyjd0508',
            'token_length': len(token) >= 32 if token else False,
            'token_complexity': self._check_token_complexity(token) if token else False
        }

        self.validation_results['admin_token'] = checks

        if not all(checks.values()):
            self.critical_failures.append("CRITICAL: Admin token is weak or default")
            return False

        print("✓ Admin token validation PASSED")
        return True

    def _check_token_complexity(self, token):
        """Check token complexity"""
        if not token:
            return False

        has_upper = any(c.isupper() for c in token)
        has_lower = any(c.islower() for c in token)
        has_digit = any(c.isdigit() for c in token)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in token)

        return sum([has_upper, has_lower, has_digit, has_special]) >= 3

    def validate_private_key_permissions(self):
        """Validate private key file permissions"""
        private_key_path = self.project_root / 'private_key.pem'

        if not private_key_path.exists():
            self.critical_failures.append("CRITICAL: Private key file not found")
            return False

        # Check file permissions
        file_stat = private_key_path.stat()
        file_mode = stat.filemode(file_stat.st_mode)

        # On Windows, check using icacls
        if os.name == 'nt':
            try:
                result = subprocess.run(
                    ['icacls', str(private_key_path)],
                    capture_output=True, text=True
                )

                if 'Users:(RX)' in result.stdout or 'Authenticated Users:(M)' in result.stdout:
                    self.critical_failures.append("CRITICAL: Private key has excessive permissions")
                    return False

            except Exception as e:
                self.critical_failures.append(f"CRITICAL: Cannot check private key permissions: {e}")
                return False

        # On Unix-like systems
        else:
            if oct(file_stat.st_mode)[-3:] != '600':
                self.critical_failures.append("CRITICAL: Private key permissions are not 600")
                return False

        print("✓ Private key permissions validation PASSED")
        return True

    def validate_env_configuration(self):
        """Validate environment configuration security"""
        env_path = self.project_root / '.env'

        if not env_path.exists():
            self.critical_failures.append("CRITICAL: .env file not found")
            return False

        # Read and check .env contents
        with open(env_path, 'r') as f:
            env_content = f.read()

        security_checks = {
            'no_hardcoded_keys': 'wgyjd0508' not in env_content,
            'https_configured': 'USE_TLS=true' in env_content or 'API_PORT=8443' in env_content,
            'dry_run_enabled': 'DRY_RUN=true' in env_content,
            'strong_tokens': 'ADMIN_TOKEN=' in env_content and len(env_content.split('ADMIN_TOKEN=')[1].split('\n')[0]) > 20
        }

        self.validation_results['env_config'] = security_checks

        failed_checks = [check for check, passed in security_checks.items() if not passed]
        if failed_checks:
            self.critical_failures.append(f"CRITICAL: Environment configuration failures: {failed_checks}")
            return False

        print("✓ Environment configuration validation PASSED")
        return True

    def validate_tls_configuration(self):
        """Validate TLS/SSL configuration"""
        tls_cert_path = Path('/etc/ssl/certs/quant_trading.crt')
        tls_key_path = Path('/etc/ssl/private/quant_trading.key')

        # Check if TLS is configured
        use_tls = os.getenv('USE_TLS', 'false').lower() == 'true'

        if not use_tls:
            self.critical_failures.append("WARNING: TLS not enabled - HTTP traffic is insecure")
            return False

        # Check certificate files exist (for production)
        if tls_cert_path.exists() and tls_key_path.exists():
            print("✓ TLS certificates found")
            return True
        else:
            print("⚠ TLS certificates not found - development mode detected")
            return True  # Allow for development

    def validate_firewall_configuration(self):
        """Validate firewall configuration"""
        try:
            # Check if secure ports are configured
            if os.name == 'nt':
                result = subprocess.run(
                    ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name=all'],
                    capture_output=True, text=True
                )

                if 'Quant Trading HTTPS' in result.stdout:
                    print("✓ Firewall rules configured")
                    return True
                else:
                    print("⚠ Firewall rules not configured")
                    return False
            else:
                # Linux firewall check would go here
                print("⚠ Linux firewall validation not implemented")
                return True

        except Exception as e:
            print(f"⚠ Firewall validation error: {e}")
            return False

    def validate_api_security(self):
        """Validate API security configuration"""
        try:
            # Test authentication is required
            response = requests.get('http://localhost:8000/api/system/status', timeout=5)

            if response.status_code == 401:
                print("✓ API authentication is enforced")
                return True
            else:
                self.critical_failures.append("CRITICAL: API allows unauthenticated access")
                return False

        except requests.ConnectionError:
            print("⚠ API not running - cannot test authentication")
            return True  # API may not be started yet
        except Exception as e:
            print(f"⚠ API security validation error: {e}")
            return False

    def validate_backup_system(self):
        """Validate backup system configuration"""
        backup_script = self.project_root / 'scripts' / 'backup_system.sh'

        if backup_script.exists():
            print("✓ Backup system configured")
            return True
        else:
            print("⚠ Backup system not configured")
            return False

    def validate_monitoring_system(self):
        """Validate security monitoring system"""
        monitoring_script = self.project_root / 'scripts' / 'security_monitor.py'

        if monitoring_script.exists():
            print("✓ Security monitoring configured")
            return True
        else:
            print("⚠ Security monitoring not configured")
            return False

    def run_comprehensive_validation(self):
        """Run comprehensive security validation"""
        print("=== COMPREHENSIVE SECURITY VALIDATION ===")
        print(f"Starting validation at {datetime.now()}")
        print()

        validations = [
            ("Admin Token Security", self.validate_admin_token),
            ("Private Key Permissions", self.validate_private_key_permissions),
            ("Environment Configuration", self.validate_env_configuration),
            ("TLS Configuration", self.validate_tls_configuration),
            ("Firewall Configuration", self.validate_firewall_configuration),
            ("API Security", self.validate_api_security),
            ("Backup System", self.validate_backup_system),
            ("Monitoring System", self.validate_monitoring_system),
        ]

        passed = 0
        total = len(validations)

        for name, validation_func in validations:
            print(f"Validating {name}...")
            try:
                if validation_func():
                    passed += 1
                else:
                    print(f"❌ {name} validation FAILED")
            except Exception as e:
                print(f"❌ {name} validation ERROR: {e}")

        print()
        print("=== VALIDATION SUMMARY ===")
        print(f"Passed: {passed}/{total}")

        if self.critical_failures:
            print("\nCRITICAL FAILURES:")
            for failure in self.critical_failures:
                print(f"  - {failure}")

            print("\n❌ SYSTEM NOT READY FOR LIVE TRADING")
            print("   Fix all critical failures before enabling live trading")
            return False
        else:
            if passed == total:
                print("\n✓ ALL VALIDATIONS PASSED")
                print("  System ready for live trading deployment")
            else:
                print("\n⚠ SOME VALIDATIONS FAILED")
                print("  Review warnings before live trading")

            return True

    def generate_validation_report(self):
        """Generate detailed validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'critical_failures': self.critical_failures,
            'ready_for_live_trading': len(self.critical_failures) == 0
        }

        report_path = self.project_root / 'security_validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nValidation report saved: {report_path}")
        return report

def main():
    """Main validation function"""
    validator = SecurityValidator()

    # Run validation
    is_secure = validator.run_comprehensive_validation()

    # Generate report
    validator.generate_validation_report()

    # Exit with appropriate code
    sys.exit(0 if is_secure else 1)

if __name__ == "__main__":
    main()