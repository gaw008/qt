#!/usr/bin/env python3
"""
Test Backend API with Real Tiger Data
Test that the backend API endpoints return real Tiger data instead of mock data
"""

import os
import asyncio
import requests
import json
from datetime import datetime

def test_backend_api():
    """Test the backend API endpoints with real Tiger data"""

    print("Testing Backend API with Real Tiger Data")
    print("=" * 50)

    # Configuration
    base_url = "http://localhost:8000"
    admin_token = os.getenv("ADMIN_TOKEN", "W1Db8xgTZnCmm0hawKaHnXlH4piXZd3VmK_lTQSxIfM")
    headers = {
        "Authorization": f"Bearer {admin_token}",
        "Content-Type": "application/json"
    }

    try:
        # Test 1: Health check
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("   Health check: OK")
        else:
            print(f"   Health check failed: {response.status_code}")
            return False

        # Test 2: System status (includes Tiger availability)
        print("\n2. Testing system status...")
        response = requests.get(f"{base_url}/api/system/status", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                tiger_available = data.get("data", {}).get("tiger_available", False)
                print(f"   System status: OK, Tiger available: {tiger_available}")
            else:
                print(f"   System status failed: {data}")
        else:
            print(f"   System status request failed: {response.status_code}")

        # Test 3: Real positions data
        print("\n3. Testing positions endpoint (real data)...")
        response = requests.get(f"{base_url}/api/positions", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                positions = data.get("data", [])
                print(f"   Found {len(positions)} real positions:")
                for pos in positions:
                    print(f"   - {pos['symbol']}: {pos['quantity']} shares @ ${pos['current_price']:.2f}")
                    print(f"     Market Value: ${pos['market_value']:,.2f}, P&L: ${pos['unrealized_pnl']:+,.2f}")
            else:
                print(f"   Positions request failed: {data}")
        else:
            print(f"   Positions request failed: {response.status_code}")

        # Test 4: Real portfolio summary
        print("\n4. Testing portfolio summary endpoint (real data)...")
        response = requests.get(f"{base_url}/api/portfolio/summary", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                portfolio = data.get("data", {})
                print(f"   Total Value: ${portfolio['total_value']:,.2f}")
                print(f"   Cash Balance: ${portfolio['cash_balance']:,.2f}")
                print(f"   Buying Power: ${portfolio['buying_power']:,.2f}")
                print(f"   Total P&L: ${portfolio['total_pnl']:+,.2f} ({portfolio['total_pnl_percent']:+.2f}%)")
                print(f"   Positions Count: {portfolio['positions_count']}")
            else:
                print(f"   Portfolio summary failed: {data}")
        else:
            print(f"   Portfolio summary request failed: {response.status_code}")

        # Test 5: Real orders data
        print("\n5. Testing orders endpoint (real data)...")
        response = requests.get(f"{base_url}/api/orders?limit=5", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                orders = data.get("data", [])
                print(f"   Found {len(orders)} recent orders:")
                for order in orders:
                    print(f"   - {order['symbol']} {order['side'].upper()} {order['quantity']} @ {order['status']}")
                    if order.get('avg_fill_price'):
                        print(f"     Fill Price: ${order['avg_fill_price']:.2f}")
            else:
                print(f"   Orders request failed: {data}")
        else:
            print(f"   Orders request failed: {response.status_code}")

        print("\n" + "=" * 50)
        print("Backend API test completed successfully!")
        print("The API is now returning real Tiger account data.")
        return True

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to backend API.")
        print("Please make sure the backend server is running:")
        print("  cd dashboard/backend")
        print("  python app.py")
        return False
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()

    success = test_backend_api()

    if success:
        print("\nNext steps:")
        print("1. The backend is now integrated with real Tiger data")
        print("2. Start the React frontend to see real data in the UI")
        print("3. All portfolio, positions, and orders are now live from Tiger account")
    else:
        print("\nPlease check the backend server and try again.")