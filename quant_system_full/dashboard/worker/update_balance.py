#!/usr/bin/env python3
"""
Account Balance Update Tool

Use this script to manually update your account balance.
"""

import sys
import argparse
from account_balance_manager import get_balance_manager

def main():
    parser = argparse.ArgumentParser(description='Update account balance')
    parser.add_argument('balance', type=float, help='New account balance amount')
    parser.add_argument('--reason', default='Manual update', help='Reason for balance update')
    
    args = parser.parse_args()
    
    # Get balance manager
    manager = get_balance_manager()
    
    # Show current balance
    current_balance = manager.get_available_balance()
    print(f"Current Balance: ${current_balance:.2f}")
    
    # Update balance
    manager.update_balance(args.balance, args.reason)
    
    print(f"Balance Updated: ${current_balance:.2f} -> ${args.balance:.2f}")
    print(f"Reason: {args.reason}")
    
    # Show account summary
    print("\nAccount Summary:")
    summary = manager.get_account_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()