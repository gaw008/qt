"""
Sector Stock Management System

This module provides comprehensive sector-based stock management functionality
for the quantitative trading system. It handles sector definitions, stock lists,
and industry classification with dynamic configuration support.

Features:
- Industry classification and sector grouping
- Dynamic sector configuration management
- Stock filtering and validation
- Batch stock list retrieval for screening
- Integration with existing data sources
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
import pandas as pd

from bot.config import SETTINGS
from bot.yahoo_data import validate_symbol, fetch_yahoo_ticker_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SectorConfig:
    """Configuration for a specific sector."""
    name: str
    display_name: str
    stocks: List[str]
    description: str = ""
    active: bool = True
    min_market_cap: float = 0  # Minimum market cap filter
    max_stocks: int = 100      # Maximum stocks per sector
    exclude_symbols: List[str] = None  # Symbols to exclude
    
    def __post_init__(self):
        if self.exclude_symbols is None:
            self.exclude_symbols = []


class SectorManager:
    """
    Manages sector definitions, stock lists, and industry classification.
    
    This class provides methods to:
    - Define and manage stock sectors
    - Get all stocks in a sector
    - Validate stock symbols
    - Filter stocks by various criteria
    - Dynamically update sector configurations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the sector manager.
        
        Args:
            config_path: Path to sector configuration file. If None, uses default sectors.
        """
        self.config_path = config_path or "bot/sector_config.json"
        self.sectors: Dict[str, SectorConfig] = {}
        self._symbol_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load default sectors or from config file
        self._load_sectors()
        logger.info(f"[sector_manager] Initialized with {len(self.sectors)} sectors")
    
    def _load_sectors(self):
        """Load sector configurations from file or use default definitions."""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    
                for sector_name, sector_data in data.items():
                    self.sectors[sector_name] = SectorConfig(
                        name=sector_name,
                        **sector_data
                    )
                logger.info(f"[sector_manager] Loaded sectors from {config_file}")
                return
            except Exception as e:
                logger.error(f"[sector_manager] Failed to load config from {config_file}: {e}")
        
        # Use default sector definitions
        self._load_default_sectors()
    
    def _load_default_sectors(self):
        """Load default sector definitions."""
        default_sectors = {
            "technology": SectorConfig(
                name="technology",
                display_name="Technology",
                description="Technology and software companies",
                stocks=[
                    # Mega-cap tech
                    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE",
                    # Large-cap tech
                    "CRM", "ORCL", "IBM", "INTC", "CSCO", "AMD", "QCOM", "TXN", "AVGO", "MU",
                    # Mid-cap tech
                    "PYPL", "SHOP", "SNAP", "TWTR", "UBER", "LYFT", "ZM", "DOCU", "WORK", "OKTA",
                    # Semiconductor
                    "TSM", "ASML", "LRCX", "KLAC", "AMAT", "MRVL", "MCHP", "ADI", "XLNX", "SWKS"
                ],
                min_market_cap=1e9,  # 1B minimum
                max_stocks=50
            ),
            
            "healthcare": SectorConfig(
                name="healthcare",
                display_name="Healthcare & Biotechnology", 
                description="Healthcare, pharmaceuticals, and biotechnology",
                stocks=[
                    # Big Pharma
                    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "MDT", "AMGN", "GILD",
                    # Biotech
                    "REGN", "VRTX", "BIIB", "ILMN", "MRNA", "BNTX", "IQV", "A", "SYK", "BSX",
                    # Healthcare services
                    "CVS", "ANTM", "CI", "HUM", "CNC", "MOH", "ELV", "VEEV", "DXCM", "ISRG"
                ],
                min_market_cap=1e9,
                max_stocks=40
            ),
            
            "financial": SectorConfig(
                name="financial",
                display_name="Financial Services",
                description="Banks, insurance, and financial services",
                stocks=[
                    # Major banks
                    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
                    # Insurance & diversified
                    "BRK.B", "BLK", "AXP", "V", "MA", "SPGI", "ICE", "CME", "MCO", "MSCI",
                    # Regional banks
                    "RF", "KEY", "FITB", "HBAN", "CFG", "ZION", "MTB", "STI", "CMA", "NTRS"
                ],
                min_market_cap=5e9,  # 5B minimum for financials
                max_stocks=35
            ),
            
            "consumer_discretionary": SectorConfig(
                name="consumer_discretionary", 
                display_name="Consumer Discretionary",
                description="Retail, automotive, and consumer goods",
                stocks=[
                    # E-commerce & retail
                    "AMZN", "HD", "LOW", "TGT", "WMT", "COST", "TJX", "SBUX", "NKE", "LULU",
                    # Automotive
                    "TSLA", "GM", "F", "RIVN", "LCID", "NIO", "XPEV", "LI", "RACE", "MBLY",
                    # Entertainment & media
                    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR", "DISH", "PARA", "WBD"
                ],
                min_market_cap=1e9,
                max_stocks=40
            ),
            
            "energy": SectorConfig(
                name="energy",
                display_name="Energy",
                description="Oil, gas, and renewable energy companies", 
                stocks=[
                    # Oil majors
                    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "HES", "DVN",
                    # Pipeline/midstream
                    "KMI", "EPD", "ET", "OKE", "WMB", "MPLX", "PAGP", "TRGP", "EQM", "AM",
                    # Renewable/clean energy
                    "NEE", "DUK", "SO", "EXC", "AEP", "XEL", "PCG", "ED", "PEG", "SRE"
                ],
                min_market_cap=2e9,  # 2B minimum 
                max_stocks=30
            )
        }
        
        self.sectors.update(default_sectors)
        logger.info(f"[sector_manager] Loaded {len(default_sectors)} default sectors")
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """
        Save current sector configuration to file.
        
        Args:
            path: File path to save configuration. Uses self.config_path if None.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        save_path = path or self.config_path
        
        try:
            config_data = {}
            for name, sector in self.sectors.items():
                # Convert SectorConfig to dict, excluding the name field
                sector_dict = asdict(sector)
                sector_dict.pop('name', None)
                config_data[name] = sector_dict
            
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"[sector_manager] Saved configuration to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"[sector_manager] Failed to save config to {save_path}: {e}")
            return False
    
    def get_sector(self, sector_name: str) -> Optional[SectorConfig]:
        """Get sector configuration by name."""
        return self.sectors.get(sector_name.lower())
    
    def list_sectors(self, active_only: bool = True) -> List[str]:
        """
        Get list of available sector names.
        
        Args:
            active_only: If True, return only active sectors.
            
        Returns:
            List of sector names.
        """
        if active_only:
            return [name for name, sector in self.sectors.items() if sector.active]
        return list(self.sectors.keys())
    
    def get_sector_stocks(self, sector_name: str, validate: bool = False) -> List[str]:
        """
        Get all stocks in a sector.
        
        Args:
            sector_name: Name of the sector.
            validate: If True, validate symbols using Yahoo Finance.
            
        Returns:
            List of stock symbols in the sector.
        """
        sector = self.get_sector(sector_name)
        if not sector:
            logger.warning(f"[sector_manager] Sector '{sector_name}' not found")
            return []
        
        if not sector.active:
            logger.warning(f"[sector_manager] Sector '{sector_name}' is inactive")
            return []
        
        stocks = [s for s in sector.stocks if s not in sector.exclude_symbols]
        
        if validate:
            validated_stocks = []
            for symbol in stocks:
                if self._validate_and_cache_symbol(symbol):
                    validated_stocks.append(symbol)
                else:
                    logger.warning(f"[sector_manager] Invalid symbol: {symbol}")
            stocks = validated_stocks
        
        # Apply max_stocks limit
        if len(stocks) > sector.max_stocks:
            logger.info(f"[sector_manager] Limiting {sector_name} to {sector.max_stocks} stocks")
            stocks = stocks[:sector.max_stocks]
        
        return stocks
    
    def get_all_stocks(self, validate: bool = False) -> List[str]:
        """
        Get all unique stocks across all active sectors.
        
        Args:
            validate: If True, validate symbols using Yahoo Finance.
            
        Returns:
            List of unique stock symbols.
        """
        all_stocks = set()
        
        for sector_name in self.list_sectors(active_only=True):
            sector_stocks = self.get_sector_stocks(sector_name, validate=validate)
            all_stocks.update(sector_stocks)
        
        return sorted(list(all_stocks))
    
    def filter_stocks_by_market_cap(
        self, 
        symbols: List[str], 
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None
    ) -> List[str]:
        """
        Filter stocks by market capitalization.
        
        Args:
            symbols: List of stock symbols to filter.
            min_market_cap: Minimum market cap in USD.
            max_market_cap: Maximum market cap in USD.
            
        Returns:
            Filtered list of stock symbols.
        """
        if not min_market_cap and not max_market_cap:
            return symbols
        
        filtered_stocks = []
        
        for symbol in symbols:
            info = self._get_symbol_info(symbol)
            if not info:
                continue
                
            market_cap = info.get('market_cap', 0)
            if market_cap <= 0:
                continue
            
            # Apply filters
            if min_market_cap and market_cap < min_market_cap:
                continue
            if max_market_cap and market_cap > max_market_cap:
                continue
                
            filtered_stocks.append(symbol)
        
        logger.info(f"[sector_manager] Filtered {len(symbols)} -> {len(filtered_stocks)} stocks by market cap")
        return filtered_stocks
    
    def add_sector(self, sector_config: SectorConfig) -> bool:
        """
        Add or update a sector configuration.
        
        Args:
            sector_config: Sector configuration to add.
            
        Returns:
            True if added successfully.
        """
        try:
            self.sectors[sector_config.name.lower()] = sector_config
            logger.info(f"[sector_manager] Added/updated sector: {sector_config.name}")
            return True
        except Exception as e:
            logger.error(f"[sector_manager] Failed to add sector: {e}")
            return False
    
    def remove_sector(self, sector_name: str) -> bool:
        """
        Remove a sector configuration.
        
        Args:
            sector_name: Name of sector to remove.
            
        Returns:
            True if removed successfully.
        """
        sector_key = sector_name.lower()
        if sector_key in self.sectors:
            del self.sectors[sector_key]
            logger.info(f"[sector_manager] Removed sector: {sector_name}")
            return True
        
        logger.warning(f"[sector_manager] Sector not found: {sector_name}")
        return False
    
    def get_stocks_by_industry(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Group stocks by their industry classification.
        
        Args:
            symbols: List of stock symbols to classify.
            
        Returns:
            Dictionary mapping industry names to lists of symbols.
        """
        industry_groups = {}
        
        for symbol in symbols:
            info = self._get_symbol_info(symbol)
            if not info:
                continue
                
            industry = info.get('industry', 'Unknown')
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(symbol)
        
        return industry_groups
    
    def validate_stock_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and has data available.
        
        Args:
            symbol: Stock symbol to validate.
            
        Returns:
            True if symbol is valid, False otherwise.
        """
        return self._validate_and_cache_symbol(symbol)
    
    def validate_sector_stocks(self, sector_name: str) -> Dict[str, bool]:
        """
        Validate all stocks in a sector.
        
        Args:
            sector_name: Name of the sector to validate.
            
        Returns:
            Dictionary mapping symbols to their validity status.
        """
        stocks = self.get_sector_stocks(sector_name, validate=False)
        results = {}
        
        for symbol in stocks:
            results[symbol] = self.validate_stock_symbol(symbol)
        
        valid_count = sum(results.values())
        logger.info(f"[sector_manager] Sector '{sector_name}': {valid_count}/{len(stocks)} valid symbols")
        
        return results
    
    def get_sector_summary(self, sector_name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive summary of a sector.
        
        Args:
            sector_name: Name of the sector.
            
        Returns:
            Dictionary containing sector summary information.
        """
        sector = self.get_sector(sector_name)
        if not sector:
            return None
        
        stocks = self.get_sector_stocks(sector_name, validate=False)
        validation_results = self.validate_sector_stocks(sector_name)
        
        # Get industry breakdown
        valid_stocks = [s for s, valid in validation_results.items() if valid]
        industry_breakdown = self.get_stocks_by_industry(valid_stocks)
        
        return {
            'sector_name': sector.name,
            'display_name': sector.display_name,
            'description': sector.description,
            'active': sector.active,
            'total_stocks': len(stocks),
            'valid_stocks': len(valid_stocks),
            'invalid_stocks': len(stocks) - len(valid_stocks),
            'min_market_cap': sector.min_market_cap,
            'max_stocks': sector.max_stocks,
            'excluded_symbols': sector.exclude_symbols,
            'industry_breakdown': {k: len(v) for k, v in industry_breakdown.items()},
            'sample_stocks': valid_stocks[:10]  # Show first 10 valid stocks
        }
    
    def _validate_and_cache_symbol(self, symbol: str) -> bool:
        """Validate symbol and cache the result."""
        if symbol in self._symbol_cache:
            return bool(self._symbol_cache[symbol])
        
        # Use Yahoo Finance validation
        try:
            is_valid = validate_symbol(symbol)
            if is_valid:
                # Cache basic info
                info = fetch_yahoo_ticker_info(symbol)
                self._symbol_cache[symbol] = info or {'symbol': symbol, 'valid': True}
            else:
                self._symbol_cache[symbol] = None
            
            return is_valid
            
        except Exception as e:
            logger.error(f"[sector_manager] Validation failed for {symbol}: {e}")
            self._symbol_cache[symbol] = None
            return False
    
    def _get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached symbol information or fetch if not available."""
        if symbol not in self._symbol_cache:
            self._validate_and_cache_symbol(symbol)
        
        return self._symbol_cache.get(symbol)
    
    def clear_cache(self):
        """Clear the symbol information cache."""
        self._symbol_cache.clear()
        logger.info("[sector_manager] Cleared symbol cache")


# Default sector manager instance
sector_manager = SectorManager()


def get_sector_stocks(sector_name: str, validate: bool = False) -> List[str]:
    """
    Convenience function to get stocks in a sector.
    
    Args:
        sector_name: Name of the sector.
        validate: If True, validate symbols.
        
    Returns:
        List of stock symbols in the sector.
    """
    return sector_manager.get_sector_stocks(sector_name, validate)


def get_all_stocks(validate: bool = False) -> List[str]:
    """
    Convenience function to get all stocks across sectors.
    
    Args:
        validate: If True, validate symbols.
        
    Returns:
        List of all unique stock symbols.
    """
    return sector_manager.get_all_stocks(validate)


def list_sectors(active_only: bool = True) -> List[str]:
    """
    Convenience function to list available sectors.
    
    Args:
        active_only: If True, return only active sectors.
        
    Returns:
        List of sector names.
    """
    return sector_manager.list_sectors(active_only)