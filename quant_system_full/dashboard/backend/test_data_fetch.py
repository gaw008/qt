"""
Test script to verify data fetching and saving functionality
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Create data directories
data_dir = Path(__file__).parent.parent.parent / "data_cache"
data_dir.mkdir(exist_ok=True)
market_data_dir = data_dir / "market_data"
market_data_dir.mkdir(exist_ok=True)

print(f"📁 Data will be saved to: {market_data_dir}")

# Test symbols
test_symbols = ['AAPL', 'GOOGL', 'MSFT']
saved_data = {}

for symbol in test_symbols:
    try:
        print(f"🔍 Fetching {symbol} data...")
        
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1y', interval='1d')  # 1 year for testing
        
        if not hist.empty:
            # Save raw market data
            raw_data_file = market_data_dir / f"{symbol}_1y_test.parquet"
            hist.to_parquet(raw_data_file)
            
            saved_data[symbol] = {
                "start_date": str(hist.index[0].date()),
                "end_date": str(hist.index[-1].date()),
                "total_records": len(hist),
                "latest_price": float(hist['Close'].iloc[-1]),
                "data_file": raw_data_file.name,
                "file_size_bytes": raw_data_file.stat().st_size
            }
            
            print(f"✅ Saved {symbol}: {len(hist)} records, latest price: ${hist['Close'].iloc[-1]:.2f}")
        else:
            print(f"❌ No data for {symbol}")
            
    except Exception as e:
        print(f"⚠️ Error with {symbol}: {e}")

# Save summary
if saved_data:
    summary_file = market_data_dir / f"test_data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(saved_data, f, indent=2)
    
    print(f"\n📊 Summary saved to: {summary_file}")
    print(f"💾 Total symbols saved: {len(saved_data)}")
    
    # List all files in directory
    print(f"\n📂 Files in market data directory:")
    for file_path in sorted(market_data_dir.glob("*")):
        size_kb = file_path.stat().st_size / 1024
        print(f"   {file_path.name} ({size_kb:.1f} KB)")

else:
    print("❌ No data was saved")

print("\n✅ Test completed!")