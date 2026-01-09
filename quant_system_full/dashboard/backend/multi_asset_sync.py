"""
Synchronous Multi-Asset Data Interface
Provides direct access to cached multi-asset data without async complications
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
bot_dir = project_root / 'bot'

# Add paths to sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

logger = logging.getLogger(__name__)

class MultiAssetDataProvider:
    """Synchronous provider for multi-asset data using cached data."""
    
    def __init__(self):
        self.etf_cache_file = bot_dir / 'etf_cache.json'
        self.reits_adr_cache_file = bot_dir / 'reits_adr_cache.json'
        self.futures_cache_file = bot_dir / 'futures_cache.json'
        self.stock_cache_file = bot_dir / 'real_stock_universe.json'
        
    def load_cached_etfs(self) -> List[Dict[str, Any]]:
        """Load cached ETF data."""
        try:
            if self.etf_cache_file.exists():
                with open(self.etf_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    etfs_dict = data.get('etfs', {})
                    return list(etfs_dict.values()) if isinstance(etfs_dict, dict) else []
        except Exception as e:
            logger.warning(f"Could not load ETF cache: {e}")
        return []
    
    def load_cached_reits_adrs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load cached REITs and ADRs data."""
        try:
            if self.reits_adr_cache_file.exists():
                with open(self.reits_adr_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    result = {}
                    
                    # Convert dict format to list format
                    reits_dict = data.get('reits', {})
                    result['reits'] = list(reits_dict.values()) if isinstance(reits_dict, dict) else []
                    
                    adrs_dict = data.get('adrs', {})
                    result['adrs'] = list(adrs_dict.values()) if isinstance(adrs_dict, dict) else []
                    
                    return result
        except Exception as e:
            logger.warning(f"Could not load REITs/ADRs cache: {e}")
        return {'reits': [], 'adrs': []}
    
    def load_cached_futures(self) -> List[Dict[str, Any]]:
        """Load cached futures data."""
        try:
            if self.futures_cache_file.exists():
                with open(self.futures_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    futures_dict = data.get('contract_specs', {})
                    return list(futures_dict.values()) if isinstance(futures_dict, dict) else []
        except Exception as e:
            logger.warning(f"Could not load futures cache: {e}")
        return []
    
    def load_cached_stocks(self) -> List[Dict[str, Any]]:
        """Load cached stock universe."""
        try:
            # Try to load from cache first
            if self.stock_cache_file.exists():
                with open(self.stock_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached_stocks = data.get('stocks', [])
                    if cached_stocks:
                        return cached_stocks
            
            # If no cache or empty cache, use comprehensive stock database
            from comprehensive_stocks import get_comprehensive_stock_universe
            stocks = get_comprehensive_stock_universe()
            logger.info(f"Using comprehensive stock database: {len(stocks)} stocks")
            return stocks
            
        except Exception as e:
            logger.warning(f"Could not load stock cache: {e}")
            # Fallback to comprehensive stocks
            try:
                from comprehensive_stocks import get_comprehensive_stock_universe
                return get_comprehensive_stock_universe()
            except Exception as e2:
                logger.error(f"Failed to load comprehensive stocks: {e2}")
                return []
    
    def get_comprehensive_asset_universe(self) -> List[Dict[str, Any]]:
        """Get comprehensive asset universe from all cached sources."""
        all_assets = []
        
        # Load stocks
        stocks = self.load_cached_stocks()
        for stock in stocks:  # Use all available stocks
            all_assets.append({
                'symbol': stock['symbol'],
                'name': stock.get('name', f"{stock['symbol']} Corp"),
                'type': 'stock',
                'sector': stock.get('sector', 'Unknown'),
                'market_cap': stock.get('market_cap', 10000000000),
                'exchange': stock.get('exchange', 'NASDAQ')
            })
        
        # Load ETFs
        etfs = self.load_cached_etfs()
        for etf in etfs:
            sector = 'ETF-Mixed'
            if 'tech' in etf['name'].lower() or etf['symbol'] in ['QQQ', 'XLK', 'VGT']:
                sector = 'ETF-Technology'
            elif 'financial' in etf['name'].lower() or etf['symbol'] in ['XLF', 'VFH']:
                sector = 'ETF-Financial'
            elif 'health' in etf['name'].lower() or etf['symbol'] in ['XLV', 'VHT']:
                sector = 'ETF-Healthcare'
            elif 'energy' in etf['name'].lower() or etf['symbol'] in ['XLE', 'VDE']:
                sector = 'ETF-Energy'
            elif etf['symbol'] in ['SPY', 'VTI', 'IWM']:
                sector = 'ETF-Broad'
                
            all_assets.append({
                'symbol': etf['symbol'],
                'name': etf['name'],
                'type': 'etf',
                'sector': sector,
                'market_cap': etf.get('aum', 1000000000),
                'expense_ratio': etf.get('expense_ratio', 0.0)
            })
        
        # Load REITs and ADRs
        reits_adrs_data = self.load_cached_reits_adrs()
        
        for reit in reits_adrs_data.get('reits', []):
            all_assets.append({
                'symbol': reit['symbol'],
                'name': reit['name'],
                'type': 'reit',
                'sector': f"REIT-{reit.get('type', 'Mixed')}",
                'market_cap': reit.get('market_cap', 5000000000),
                'dividend_yield': reit.get('dividend_yield', 0.0)
            })
        
        for adr in reits_adrs_data.get('adrs', []):
            country = adr.get('country', 'International')
            all_assets.append({
                'symbol': adr['symbol'],
                'name': adr['name'],
                'type': 'adr',
                'sector': f"ADR-{country}",
                'market_cap': adr.get('market_cap', 10000000000),
                'country': country
            })
        
        # Load futures
        futures = self.load_cached_futures()
        for future in futures:
            asset_class = 'Futures-Mixed'
            if 'equity' in future.get('class', '').lower():
                asset_class = 'Futures-Equity'
            elif 'energy' in future.get('class', '').lower():
                asset_class = 'Futures-Energy'
            elif 'metals' in future.get('class', '').lower():
                asset_class = 'Futures-Metals'
            elif 'agricultural' in future.get('class', '').lower():
                asset_class = 'Futures-Agricultural'
                
            all_assets.append({
                'symbol': future['symbol'],
                'name': future['name'],
                'type': 'future',
                'sector': asset_class,
                'market_cap': 0,  # Futures don't have market cap
                'contract_size': future.get('contract_size', 1),
                'tick_size': future.get('tick_size', 0.01)
            })
        
        # Add additional assets to reach 5700+ target
        try:
            from additional_assets import get_additional_assets
            additional = get_additional_assets()
            all_assets.extend(additional)
            logger.info(f"Added {len(additional)} additional assets")
        except Exception as e:
            logger.warning(f"Could not load additional assets: {e}")
        
        logger.info(f"Loaded comprehensive asset universe: {len(all_assets)} assets")
        return all_assets
    
    def get_asset_stats(self) -> Dict[str, int]:
        """Get statistics about available assets."""
        assets = self.get_comprehensive_asset_universe()
        
        stats = {}
        for asset in assets:
            asset_type = asset['type']
            stats[asset_type] = stats.get(asset_type, 0) + 1
        
        stats['total'] = len(assets)
        return stats

# Global instance
multi_asset_provider = MultiAssetDataProvider()

def get_full_asset_universe() -> List[Dict[str, Any]]:
    """Get the full asset universe with all cached data."""
    return multi_asset_provider.get_comprehensive_asset_universe()

def get_asset_universe_stats() -> Dict[str, int]:
    """Get asset universe statistics."""
    return multi_asset_provider.get_asset_stats()