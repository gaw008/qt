#!/usr/bin/env python3
"""
Direct Yahoo Finance API Test
"""

import yfinance as yf
import time

def test_yahoo_api():
    print('Testing Yahoo Finance API directly...')
    print('Current time:', time.strftime('%Y-%m-%d %H:%M:%S'))

    # Test a simple stock request
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']

    for symbol in test_symbols:
        try:
            print(f'\nTesting {symbol}...')
            ticker = yf.Ticker(symbol)

            # Try to get basic info
            info = ticker.info
            if info:
                print(f'  Company: {info.get("longName", "Unknown")}')
                print(f'  Price: ${info.get("currentPrice", "N/A")}')

            # Try to get history
            hist = ticker.history(period='5d')
            if not hist.empty:
                latest_price = hist['Close'].iloc[-1]
                print(f'  Latest Close: ${latest_price:.2f}')
                print(f'  Data points: {len(hist)}')
            else:
                print(f'  No historical data')

        except Exception as e:
            print(f'  ERROR: {e}')

        # Add delay between requests
        time.sleep(2)

    print('\nDirect Yahoo Finance API test completed.')

if __name__ == "__main__":
    test_yahoo_api()