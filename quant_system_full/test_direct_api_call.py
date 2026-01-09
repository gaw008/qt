#!/usr/bin/env python3
"""
Test Direct API Call with Real Data
Test the API endpoints to verify they're returning the actual Tiger data we expect
"""

import requests
import json
import os

def test_api_endpoints():
    """Test API endpoints and compare with expected real data"""

    print("Testing API Endpoints for Real Data")
    print("=" * 45)

    # Expected real data from our Tiger account
    expected_symbols = ["C", "CAT"]  # These are the actual positions we have
    expected_total_value = 12132.94  # Actual account value

    # Configuration
    base_url = "http://localhost:8000"
    admin_token = os.getenv("ADMIN_TOKEN", "W1Db8xgTZnCmm0hawKaHnXlH4piXZd3VmK_lTQSxIfM")
    headers = {
        "Authorization": f"Bearer {admin_token}",
        "Content-Type": "application/json"
    }

    try:
        # Test positions
        print("\n1. Testing positions endpoint...")
        response = requests.get(f"{base_url}/api/positions", headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                positions = data.get("data", [])
                symbols = [pos["symbol"] for pos in positions]

                print(f"   Returned positions: {symbols}")
                print(f"   Expected positions: {expected_symbols}")

                # Check if we're getting real data
                is_real_data = any(symbol in expected_symbols for symbol in symbols)
                print(f"   Contains real Tiger symbols: {is_real_data}")

                if is_real_data:
                    print("   SUCCESS: Getting real Tiger positions!")
                    for pos in positions:
                        if pos["symbol"] in expected_symbols:
                            print(f"   Real position: {pos['symbol']} - {pos['quantity']} shares @ ${pos['current_price']:.2f}")
                else:
                    print("   WARNING: Still getting mock data (AAPL, GOOGL)")
            else:
                print(f"   API error: {data}")
        else:
            print(f"   HTTP error: {response.status_code}")

        # Test portfolio summary
        print("\n2. Testing portfolio summary...")
        response = requests.get(f"{base_url}/api/portfolio/summary", headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                portfolio = data.get("data", {})
                total_value = portfolio.get("total_value", 0)

                print(f"   Returned total value: ${total_value:,.2f}")
                print(f"   Expected total value: ${expected_total_value:,.2f}")

                # Check if values are close (within reasonable range)
                is_real_data = abs(total_value - expected_total_value) < 1000
                print(f"   Appears to be real data: {is_real_data}")

                if is_real_data:
                    print("   SUCCESS: Getting real Tiger portfolio data!")
                    print(f"   Cash: ${portfolio.get('cash_balance', 0):,.2f}")
                    print(f"   Buying Power: ${portfolio.get('buying_power', 0):,.2f}")
                    print(f"   P&L: ${portfolio.get('total_pnl', 0):+,.2f}")
                else:
                    print("   WARNING: Still getting mock data (250k portfolio)")
            else:
                print(f"   API error: {data}")
        else:
            print(f"   HTTP error: {response.status_code}")

        # Test orders
        print("\n3. Testing orders endpoint...")
        response = requests.get(f"{base_url}/api/orders?limit=3", headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                orders = data.get("data", [])
                symbols = [order["symbol"] for order in orders]

                print(f"   Order symbols: {symbols}")

                # Check if we're getting real orders
                is_real_data = any(symbol in expected_symbols for symbol in symbols)
                print(f"   Contains real Tiger orders: {is_real_data}")

                if is_real_data:
                    print("   SUCCESS: Getting real Tiger orders!")
                    for order in orders:
                        if order["symbol"] in expected_symbols:
                            print(f"   Real order: {order['symbol']} {order['side'].upper()} {order['quantity']} @ {order['status']}")
                else:
                    print("   WARNING: Still getting mock orders")
            else:
                print(f"   API error: {data}")
        else:
            print(f"   HTTP error: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to backend API.")
        print("Please make sure the backend server is running on port 8000")
        return False
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        return False

    print("\n" + "=" * 45)
    return True

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_api_endpoints()