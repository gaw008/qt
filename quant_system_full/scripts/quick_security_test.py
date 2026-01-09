#!/usr/bin/env python3
"""
Quick Security Test Script
Validates critical security fixes are active
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_security_configuration():
    """Test current security configuration"""
    print("=== CRITICAL SECURITY VALIDATION ===")

    # Load environment
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(env_path)

    # Test results
    results = {}

    # Test 1: Admin Token Security
    admin_token = os.getenv('ADMIN_TOKEN')
    if admin_token and admin_token != 'wgyjd0508' and len(admin_token) >= 32:
        results['admin_token'] = "PASS - Strong token active"
    else:
        results['admin_token'] = "FAIL - Weak or default token"

    # Test 2: DRY_RUN Status
    dry_run = os.getenv('DRY_RUN', 'true').lower()
    if dry_run == 'true':
        results['dry_run'] = "PASS - Safe mode enabled"
    else:
        results['dry_run'] = "WARNING - Live trading enabled"

    # Test 3: TLS Configuration
    use_tls = os.getenv('USE_TLS', 'false').lower()
    if use_tls == 'true':
        results['tls'] = "PASS - TLS enabled"
    else:
        results['tls'] = "WARNING - TLS not enabled"

    # Test 4: Secure Ports
    api_port = os.getenv('API_PORT', '8000')
    if api_port in ['8443', '443']:
        results['secure_ports'] = "PASS - Secure port configured"
    else:
        results['secure_ports'] = "WARNING - Insecure port configured"

    # Test 5: Private Key Security
    private_key_path = project_root / 'private_key.pem'
    if private_key_path.exists():
        results['private_key'] = "PASS - Private key exists"
    else:
        results['private_key'] = "FAIL - Private key missing"

    # Test 6: Emergency Token
    emergency_token = os.getenv('EMERGENCY_STOP_TOKEN')
    if emergency_token and len(emergency_token) >= 16:
        results['emergency_token'] = "PASS - Emergency token configured"
    else:
        results['emergency_token'] = "FAIL - Emergency token missing"

    # Display results
    print("\nSECURITY TEST RESULTS:")
    print("-" * 50)

    passed = 0
    total = len(results)

    for test, result in results.items():
        status = "PASS" if result.startswith("PASS") else "FAIL" if result.startswith("FAIL") else "WARN"
        icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
        print(f"{icon} {test.replace('_', ' ').title()}: {result}")

        if status == "PASS":
            passed += 1

    print("-" * 50)
    print(f"PASSED: {passed}/{total} tests")

    # Overall assessment
    critical_failures = [k for k, v in results.items() if v.startswith("FAIL")]

    if critical_failures:
        print("\n❌ CRITICAL FAILURES DETECTED")
        print("System NOT ready for live trading")
        print("Fix the following issues:")
        for failure in critical_failures:
            print(f"  - {failure.replace('_', ' ').title()}")
        return False
    else:
        warnings = [k for k, v in results.items() if v.startswith("WARNING")]
        if warnings:
            print("\n⚠️ WARNINGS DETECTED")
            print("Review the following before live trading:")
            for warning in warnings:
                print(f"  - {warning.replace('_', ' ').title()}")
        else:
            print("\n✅ ALL SECURITY TESTS PASSED")
            print("Critical security fixes are active")

        return True

def main():
    """Main test function"""
    try:
        is_secure = test_security_configuration()

        print("\n" + "="*50)
        if is_secure:
            print("SECURITY STATUS: PROTECTED")
            print("Critical vulnerabilities have been fixed")
        else:
            print("SECURITY STATUS: VULNERABLE")
            print("Apply security fixes before proceeding")
        print("="*50)

        sys.exit(0 if is_secure else 1)

    except Exception as e:
        print(f"Security test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()