#!/usr/bin/env python3
"""
Simple test script to verify 5700+ asset system (no Unicode)
"""
import sys
import json
from pathlib import Path
from multi_asset_sync import get_full_asset_universe, get_asset_universe_stats

def test_asset_universe():
    """Test the comprehensive asset universe."""
    print("=== Testing 5700+ Asset System ===")
    print()
    
    # Get asset statistics
    stats = get_asset_universe_stats()
    print("Asset Universe Statistics:")
    for asset_type, count in sorted(stats.items()):
        print(f"  {asset_type.upper()}: {count:,}")
    print()
    
    # Get full asset universe
    all_assets = get_full_asset_universe()
    print(f"Total Assets: {len(all_assets):,}")
    
    # Verify target achievement
    target = 5700
    achievement_rate = (len(all_assets) / target) * 100
    status = "TARGET ACHIEVED" if len(all_assets) >= target else "TARGET NOT MET"
    print(f"Target: {target:,}")
    print(f"Achievement Rate: {achievement_rate:.1f}%")
    print(f"Status: {status}")
    print()
    
    # Sample assets by type
    asset_types = {}
    for asset in all_assets:
        asset_type = asset['type']
        if asset_type not in asset_types:
            asset_types[asset_type] = []
        if len(asset_types[asset_type]) < 5:  # Sample first 5 of each type
            asset_types[asset_type].append(asset)
    
    print("Sample Assets by Type:")
    for asset_type, samples in asset_types.items():
        print(f"\n{asset_type.upper()} samples:")
        for asset in samples:
            symbol = asset['symbol']
            name = asset['name'][:30] + "..." if len(asset['name']) > 30 else asset['name']
            sector = asset.get('sector', 'N/A')
            market_cap = asset.get('market_cap', 0)
            
            if market_cap > 1000000000:
                cap_str = f"{market_cap/1000000000:.1f}B"
            elif market_cap > 1000000:
                cap_str = f"{market_cap/1000000:.1f}M"
            else:
                cap_str = f"{market_cap:,.0f}"
            
            print(f"  {symbol:6} | {name:30} | {sector:15} | {cap_str:>8}")
    
    print(f"\nSystem successfully integrated {len(all_assets):,} assets!")
    
    if len(all_assets) >= 5700:
        print("*** 5,700+ ASSET TARGET SUCCESSFULLY ACHIEVED! ***")
        return True
    else:
        print(f"*** NEED {5700 - len(all_assets):,} MORE ASSETS TO REACH TARGET ***")
        return False

if __name__ == "__main__":
    try:
        success = test_asset_universe()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Error testing asset universe: {e}")
        import traceback
        traceback.print_exc()
        exit(1)