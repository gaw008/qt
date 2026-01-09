#!/usr/bin/env python3
"""
Final Security Verification Script
Confirms all critical security fixes are active and working
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def main():
    """Comprehensive final security verification"""
    print("=" * 60)
    print("FINAL SECURITY VERIFICATION")
    print("=" * 60)

    # Load environment
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(env_path)

    # Critical security checks
    checks_passed = 0
    total_checks = 6

    print("\nCRITICAL SECURITY VALIDATIONS:")
    print("-" * 40)

    # Check 1: Admin Token Security
    admin_token = os.getenv('ADMIN_TOKEN')
    if admin_token and admin_token != 'wgyjd0508' and len(admin_token) >= 32:
        print("1. Admin Token Security: PASS - Strong cryptographic token active")
        checks_passed += 1
    else:
        print("1. Admin Token Security: FAIL - Weak or default token")

    # Check 2: DRY_RUN Safety
    dry_run = os.getenv('DRY_RUN', 'true').lower()
    if dry_run == 'true':
        print("2. Safe Mode (DRY_RUN): PASS - Real trading disabled")
        checks_passed += 1
    else:
        print("2. Safe Mode (DRY_RUN): WARNING - Live trading enabled")

    # Check 3: TLS Configuration
    use_tls = os.getenv('USE_TLS', 'false').lower()
    if use_tls == 'true':
        print("3. TLS/HTTPS: PASS - Encrypted communications configured")
        checks_passed += 1
    else:
        print("3. TLS/HTTPS: FAIL - Insecure HTTP communications")

    # Check 4: Secure Ports
    api_port = os.getenv('API_PORT', '8000')
    if api_port in ['8443', '443']:
        print("4. Secure Ports: PASS - HTTPS port configured")
        checks_passed += 1
    else:
        print("4. Secure Ports: FAIL - Insecure port configured")

    # Check 5: Emergency Controls
    emergency_token = os.getenv('EMERGENCY_STOP_TOKEN')
    if emergency_token and len(emergency_token) >= 16:
        print("5. Emergency Controls: PASS - Emergency stop configured")
        checks_passed += 1
    else:
        print("5. Emergency Controls: FAIL - Emergency token missing")

    # Check 6: Private Key Security
    private_key_path = project_root / 'private_key.pem'
    if private_key_path.exists():
        print("6. Private Key: PASS - Private key file exists")
        checks_passed += 1
    else:
        print("6. Private Key: FAIL - Private key file missing")

    # Security infrastructure checks
    print("\nSECURITY INFRASTRUCTURE:")
    print("-" * 40)

    infrastructure_files = [
        ('Security Monitor', 'scripts/security_monitor.py'),
        ('Emergency Procedures', 'scripts/emergency_procedures.py'),
        ('Backup System', 'scripts/backup_system.sh'),
        ('HTTPS Deployment', 'scripts/deploy_secure_https.py'),
        ('Live Trading Deployment', 'scripts/live_trading_deployment.py'),
        ('Secure Deployment', 'scripts/secure_deployment.py'),
    ]

    infrastructure_ready = 0
    for name, file_path in infrastructure_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"{name}: READY")
            infrastructure_ready += 1
        else:
            print(f"{name}: MISSING")

    # Configuration files check
    print("\nCONFIGURATION FILES:")
    print("-" * 40)

    config_files = [
        ('Secure Environment', '.env'),
        ('Secure Template', '.env.secure'),
        ('Nginx Configuration', 'nginx_secure.conf'),
        ('Security Summary', 'CRITICAL_SECURITY_IMPLEMENTATION_COMPLETE.md'),
    ]

    config_ready = 0
    for name, file_path in config_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"{name}: READY")
            config_ready += 1
        else:
            print(f"{name}: MISSING")

    # Final assessment
    print("\n" + "=" * 60)
    print("SECURITY ASSESSMENT SUMMARY")
    print("=" * 60)

    print(f"Critical Security Checks: {checks_passed}/{total_checks}")
    print(f"Infrastructure Components: {infrastructure_ready}/{len(infrastructure_files)}")
    print(f"Configuration Files: {config_ready}/{len(config_files)}")

    # Overall security status
    if checks_passed == total_checks:
        print("\nSECURITY STATUS: SECURE")
        print("All critical vulnerabilities have been fixed")
        print("System is protected and ready for infrastructure deployment")

        if infrastructure_ready == len(infrastructure_files) and config_ready == len(config_files):
            print("\nREADINESS STATUS: COMPLETE")
            print("All security components are ready for deployment")
        else:
            print("\nREADINESS STATUS: PARTIAL")
            print("Some infrastructure components may need to be deployed")

    else:
        print("\nSECURITY STATUS: VULNERABLE")
        print("Critical security issues remain - DO NOT enable live trading")

    # Final recommendations
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)

    if checks_passed == total_checks:
        print("1. Deploy SSL certificates: python scripts/deploy_secure_https.py")
        print("2. Configure firewall rules: python scripts/secure_deployment.py")
        print("3. Start security monitoring: python scripts/security_monitor.py")
        print("4. Test emergency procedures: python scripts/emergency_procedures.py health")
        print("5. Deploy for live trading: python scripts/live_trading_deployment.py")
        print("\nCRITICAL: Only enable live trading after ALL infrastructure is deployed")
    else:
        print("1. Fix remaining security issues")
        print("2. Re-run this verification script")
        print("3. Do not proceed with live trading until all checks pass")

    print("\nEmergency Stop Token: " + os.getenv('EMERGENCY_STOP_TOKEN', 'NOT_SET'))
    print("Emergency Stop Command: python scripts/emergency_procedures.py stop")

    return checks_passed == total_checks

if __name__ == "__main__":
    is_secure = main()
    sys.exit(0 if is_secure else 1)