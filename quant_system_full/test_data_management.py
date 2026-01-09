"""
Test script for the data management system components.

This script tests the integration between:
- sector_manager.py
- data.py (batch functionality)
- stock_screener.py

Usage:
python test_data_management.py
"""

import sys
import os
sys.path.append('bot')

from bot.config import SETTINGS
from bot.sector_manager import sector_manager, get_sector_stocks, get_all_stocks, list_sectors
from bot.data import fetch_batch_history, fetch_sector_history
from bot.stock_screener import StockScreener, ScreeningCriteria, screen_top_stocks


def test_sector_manager():
    """Test sector management functionality."""
    print("\n=== Testing Sector Manager ===")
    
    # Test list sectors
    sectors = list_sectors()
    print(f"Available sectors: {sectors}")
    
    # Test get sector stocks
    if sectors:
        sector_name = sectors[0]
        stocks = get_sector_stocks(sector_name)
        print(f"Stocks in {sector_name}: {len(stocks)} stocks")
        print(f"First 5 stocks: {stocks[:5]}")
        
        # Test sector summary
        summary = sector_manager.get_sector_summary(sector_name)
        if summary:
            print(f"Sector summary for {sector_name}:")
            print(f"  - Total stocks: {summary['total_stocks']}")
            print(f"  - Active: {summary['active']}")
            print(f"  - Sample stocks: {summary['sample_stocks'][:3]}")
    
    # Test get all stocks
    all_stocks = get_all_stocks()
    print(f"Total unique stocks across all sectors: {len(all_stocks)}")
    
    print("[OK] Sector Manager tests completed")


def test_batch_data_acquisition():
    """Test batch data acquisition functionality."""
    print("\n=== Testing Batch Data Acquisition ===")
    
    # Test with a small sample of symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"Testing batch fetch for symbols: {test_symbols}")
    
    # Test with dry run (placeholder data)
    data = fetch_batch_history(
        quote_client=None,
        symbols=test_symbols,
        period='day',
        limit=10,
        dry_run=True,  # Use placeholder data for testing
        max_concurrent=2,
        delay_between_requests=0.1
    )
    
    print(f"Fetched data for {len(data)} symbols")
    for symbol, df in data.items():
        if df is not None:
            print(f"  {symbol}: {len(df)} rows")
        else:
            print(f"  {symbol}: No data")
    
    # Test sector data acquisition
    sectors = list_sectors()
    if sectors:
        sector_name = sectors[0]
        print(f"\nTesting sector data fetch for: {sector_name}")
        
        sector_data = fetch_sector_history(
            quote_client=None,
            sector_name=sector_name,
            period='day',
            limit=5,
            dry_run=True,
            validate_symbols=False  # Skip validation for speed
        )
        
        print(f"Fetched sector data: {len(sector_data)} stocks")
        successful = sum(1 for df in sector_data.values() if df is not None)
        print(f"Successful fetches: {successful}/{len(sector_data)}")
    
    print("[OK] Batch Data Acquisition tests completed")


def test_stock_screener():
    """Test stock screening functionality."""
    print("\n=== Testing Stock Screener ===")
    
    # Create screening criteria (relaxed for testing with placeholder data)
    criteria = ScreeningCriteria(
        top_n=5,
        min_market_cap=1e6,    # Lower minimum for testing
        max_market_cap=1e15,   # Higher maximum for testing
        min_price=1.0,         # Lower minimum price
        max_price=10000.0,     # Higher maximum price
        min_volume=1000,       # Lower volume requirement
        valuation_weight=0.3,
        volume_weight=0.2,
        momentum_weight=0.25,
        quality_weight=0.15,
        technical_weight=0.1
    )
    
    print(f"Created screening criteria: top_n={criteria.top_n}")
    
    # Initialize screener
    screener = StockScreener(criteria)
    
    # Test with a small set of symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    print(f"Testing screening with symbols: {test_symbols}")
    
    # Screen stocks (with dry run data)
    results = screener.screen_stocks(
        quote_client=None,
        symbols=test_symbols,
        dry_run=True
    )
    
    print(f"Screening results: {len(results)} stocks")
    
    if results:
        print("Top stocks:")
        for i, stock in enumerate(results[:3], 1):
            print(f"  {i}. {stock.symbol}: score={stock.final_score:.3f}")
            print(f"     Price: ${stock.current_price:.2f}, Volume: {stock.volume:,}")
            print(f"     Valuation: {stock.valuation_score:.3f}, Volume: {stock.volume_score:.3f}")
    
    # Test screening summary
    summary = screener.get_screening_summary()
    if "error" not in summary:
        print(f"\nScreening summary:")
        print(f"  Total stocks: {summary['total_stocks']}")
        print(f"  Score stats: mean={summary['score_stats']['mean']:.3f}, "
              f"std={summary['score_stats']['std']:.3f}")
    
    # Test convenience function
    print(f"\nTesting convenience function...")
    quick_results = screen_top_stocks(
        quote_client=None,
        top_n=3,
        dry_run=True
    )
    print(f"Quick screen results: {len(quick_results)} stocks")
    
    print("[OK] Stock Screener tests completed")


def test_integration():
    """Test integration between all components."""
    print("\n=== Testing Integration ===")
    
    # Get stocks from sector manager
    sectors = list_sectors()
    if sectors:
        sector_name = sectors[0]
        sector_stocks = get_sector_stocks(sector_name)[:5]  # Limit to 5 for testing
        
        print(f"Testing integration with {len(sector_stocks)} stocks from {sector_name}")
        
        # Fetch data for these stocks
        data = fetch_batch_history(
            quote_client=None,
            symbols=sector_stocks,
            dry_run=True,
            limit=10
        )
        
        # Screen the stocks
        screener = StockScreener(ScreeningCriteria(top_n=3))
        results = screener.screen_stocks(
            quote_client=None,
            symbols=sector_stocks,
            dry_run=True
        )
        
        print(f"Integration test results:")
        print(f"  - Fetched data for {len(data)} stocks")
        print(f"  - Screened {len(results)} stocks")
        
        if results:
            print(f"  - Top stock: {results[0].symbol} (score: {results[0].final_score:.3f})")
    
    print("[OK] Integration tests completed")


def test_configuration():
    """Test configuration and settings."""
    print("\n=== Testing Configuration ===")
    
    print(f"Current configuration:")
    print(f"  DRY_RUN: {SETTINGS.dry_run}")
    print(f"  DATA_SOURCE: {SETTINGS.data_source}")
    print(f"  YAHOO_API_TIMEOUT: {SETTINGS.yahoo_api_timeout}")
    print(f"  USE_MCP_TOOLS: {SETTINGS.use_mcp_tools}")
    
    # Test sector manager configuration
    print(f"\nSector Manager:")
    print(f"  Total sectors: {len(sector_manager.sectors)}")
    print(f"  Cache size: {len(sector_manager._symbol_cache)}")
    
    # Test if we can save/load sector config
    config_test_path = "test_sector_config.json"
    if sector_manager.save_config(config_test_path):
        print(f"  [OK] Successfully saved config to {config_test_path}")
        
        # Clean up
        import os
        if os.path.exists(config_test_path):
            os.remove(config_test_path)
            print(f"  [OK] Cleaned up test config file")
    
    print("[OK] Configuration tests completed")


def main():
    """Run all tests."""
    print("Starting Data Management System Tests...")
    print(f"Python path: {sys.path}")
    
    try:
        test_configuration()
        test_sector_manager()
        test_batch_data_acquisition()
        test_stock_screener()
        test_integration()
        
        print("\n" + "="*50)
        print("[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)