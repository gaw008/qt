"""
Massive Stock Universe Builder - Target 30,000+ stocks
Fetches stocks from all major US exchanges and international markets
"""
import pandas as pd
import requests
import json
import yfinance as yf
from typing import List, Set, Dict
import time
from pathlib import Path

class MassiveStockUniverseBuilder:
    """Build a massive stock universe from multiple data sources targeting 30K+ stocks"""
    
    def __init__(self):
        self.all_stocks = set()
        self.stock_details = {}
        self.debug_info = []
        
    def log_debug(self, message: str):
        """Add debug message"""
        print(f"[UNIVERSE] {message}")
        self.debug_info.append(message)
    
    def get_all_stocks(self) -> List[str]:
        """Main method to fetch all stocks from multiple sources"""
        self.log_debug("Starting massive stock universe building...")
        
        # Core US Market Indices
        self.fetch_sp500_stocks()
        self.fetch_sp400_midcap_stocks() 
        self.fetch_sp600_smallcap_stocks()
        self.fetch_russell3000_stocks()
        self.fetch_nasdaq_composite_stocks()
        self.fetch_dow_jones_stocks()
        
        # Exchange-based fetching
        self.fetch_all_nasdaq_stocks()
        self.fetch_all_nyse_stocks() 
        self.fetch_all_amex_stocks()
        
        # Sector and Style ETFs
        self.fetch_sector_etf_holdings()
        self.fetch_style_etf_holdings()
        self.fetch_international_etf_holdings()
        self.fetch_commodity_etf_holdings()
        
        # Additional Sources
        self.fetch_crypto_related_stocks()
        self.fetch_penny_stocks()
        self.fetch_reit_stocks()
        self.fetch_utility_stocks()
        
        # Clean and filter
        filtered_stocks = self.clean_and_filter_stocks()
        
        self.log_debug(f"Final universe: {len(filtered_stocks)} unique stocks")
        return filtered_stocks
    
    def fetch_sp500_stocks(self):
        """Fetch S&P 500 constituents"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            self.all_stocks.update(symbols)
            self.log_debug(f"Added {len(symbols)} S&P 500 stocks")
        except Exception as e:
            self.log_debug(f"S&P 500 fetch failed: {e}")
    
    def fetch_sp400_midcap_stocks(self):
        """Fetch S&P 400 MidCap constituents"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
            tables = pd.read_html(url)
            sp400_table = tables[0]
            symbols = sp400_table['Symbol'].tolist()
            self.all_stocks.update(symbols)
            self.log_debug(f"Added {len(symbols)} S&P 400 MidCap stocks")
        except Exception as e:
            self.log_debug(f"S&P 400 fetch failed: {e}")
    
    def fetch_sp600_smallcap_stocks(self):
        """Fetch S&P 600 SmallCap constituents"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
            tables = pd.read_html(url)
            sp600_table = tables[0]
            symbols = sp600_table['Symbol'].tolist()
            self.all_stocks.update(symbols)
            self.log_debug(f"Added {len(symbols)} S&P 600 SmallCap stocks")
        except Exception as e:
            self.log_debug(f"S&P 600 fetch failed: {e}")
    
    def fetch_russell3000_stocks(self):
        """Fetch Russell 3000 constituents using multiple methods"""
        try:
            # Method 1: Try to get from Russell website or financial data providers
            # Note: Russell 3000 data is often paid, so we'll use approximation methods
            
            # Method 2: Get Russell 1000 + Russell 2000 (which roughly equals Russell 3000)
            russell_symbols = set()
            
            # Try to get Russell 1000 from ETF holdings
            try:
                iwb_holdings = self.fetch_etf_holdings('IWB')  # iShares Russell 1000 ETF
                russell_symbols.update(iwb_holdings)
                self.log_debug(f"Added {len(iwb_holdings)} Russell 1000 stocks from IWB")
            except:
                pass
            
            # Try to get Russell 2000 from ETF holdings  
            try:
                iwm_holdings = self.fetch_etf_holdings('IWM')  # iShares Russell 2000 ETF
                russell_symbols.update(iwm_holdings)
                self.log_debug(f"Added {len(iwm_holdings)} Russell 2000 stocks from IWM")
            except:
                pass
            
            self.all_stocks.update(russell_symbols)
            self.log_debug(f"Total Russell stocks added: {len(russell_symbols)}")
            
        except Exception as e:
            self.log_debug(f"Russell 3000 fetch failed: {e}")
    
    def fetch_nasdaq_composite_stocks(self):
        """Fetch all NASDAQ Composite stocks"""
        try:
            # Use FTP method to get NASDAQ listed companies
            nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
            nasdaq_df = pd.read_csv(nasdaq_url, sep='|')
            nasdaq_symbols = nasdaq_df['Symbol'].str.replace('$', '', regex=False).tolist()
            nasdaq_symbols = [s for s in nasdaq_symbols if s and s != 'File Creation Time']
            
            self.all_stocks.update(nasdaq_symbols)
            self.log_debug(f"Added {len(nasdaq_symbols)} NASDAQ Composite stocks")
            
        except Exception as e:
            self.log_debug(f"NASDAQ Composite fetch failed: {e}")
            # Fallback method using NASDAQ 100 ETF
            try:
                qqq_holdings = self.fetch_etf_holdings('QQQ')
                self.all_stocks.update(qqq_holdings)
                self.log_debug(f"Fallback: Added {len(qqq_holdings)} from QQQ ETF")
            except:
                pass
    
    def fetch_dow_jones_stocks(self):
        """Fetch Dow Jones Industrial Average constituents"""
        try:
            url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
            tables = pd.read_html(url)
            dow_table = tables[1]  # Usually the second table
            symbols = dow_table['Symbol'].tolist()
            self.all_stocks.update(symbols)
            self.log_debug(f"Added {len(symbols)} Dow Jones stocks")
        except Exception as e:
            self.log_debug(f"Dow Jones fetch failed: {e}")
    
    def fetch_all_nasdaq_stocks(self):
        """Fetch all NASDAQ exchange stocks"""
        try:
            nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
            nasdaq_df = pd.read_csv(nasdaq_url, sep='|')
            symbols = nasdaq_df['Symbol'].str.replace('$', '', regex=False).tolist()
            symbols = [s for s in symbols if s and s != 'File Creation Time' and len(s) <= 5]
            
            self.all_stocks.update(symbols)
            self.log_debug(f"Added {len(symbols)} NASDAQ exchange stocks")
            
        except Exception as e:
            self.log_debug(f"NASDAQ exchange fetch failed: {e}")
    
    def fetch_all_nyse_stocks(self):
        """Fetch all NYSE exchange stocks"""
        try:
            # NYSE data from NASDAQ FTP (they provide all exchange data)
            nyse_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
            nyse_df = pd.read_csv(nyse_url, sep='|')
            # Filter for NYSE symbols
            nyse_symbols = nyse_df[nyse_df['Exchange'] == 'N']['NASDAQ Symbol'].tolist()
            nyse_symbols = [s for s in nyse_symbols if s and len(s) <= 5]
            
            self.all_stocks.update(nyse_symbols)
            self.log_debug(f"Added {len(nyse_symbols)} NYSE stocks")
            
        except Exception as e:
            self.log_debug(f"NYSE fetch failed: {e}")
    
    def fetch_all_amex_stocks(self):
        """Fetch all AMEX exchange stocks"""
        try:
            # AMEX data from NASDAQ FTP
            amex_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
            amex_df = pd.read_csv(amex_url, sep='|')
            # Filter for AMEX symbols
            amex_symbols = amex_df[amex_df['Exchange'] == 'A']['NASDAQ Symbol'].tolist()
            amex_symbols = [s for s in amex_symbols if s and len(s) <= 5]
            
            self.all_stocks.update(amex_symbols)
            self.log_debug(f"Added {len(amex_symbols)} AMEX stocks")
            
        except Exception as e:
            self.log_debug(f"AMEX fetch failed: {e}")
    
    def fetch_sector_etf_holdings(self):
        """Fetch holdings from major sector ETFs"""
        sector_etfs = {
            # SPDR Sector ETFs
            'XLK': 'Technology', 'XLV': 'Healthcare', 'XLF': 'Financial',
            'XLE': 'Energy', 'XLI': 'Industrial', 'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary', 'XLU': 'Utilities', 'XLB': 'Materials',
            'XLRE': 'Real Estate', 'XLC': 'Communication',
            
            # Vanguard Sector ETFs  
            'VGT': 'Technology', 'VHT': 'Healthcare', 'VFH': 'Financials',
            'VDE': 'Energy', 'VIS': 'Industrials', 'VDC': 'Consumer Staples',
            'VCR': 'Consumer Discretionary', 'VPU': 'Utilities', 'VAW': 'Materials',
            'VNQ': 'Real Estate',
            
            # iShares Sector ETFs
            'IYW': 'Technology', 'IYH': 'Healthcare', 'IYF': 'Financial',
            'IYE': 'Energy', 'IYJ': 'Industrial', 'IYK': 'Consumer Goods',
            'IYC': 'Consumer Discretionary', 'IDU': 'Utilities', 'IYM': 'Materials',
        }
        
        all_holdings = set()
        for etf_symbol, sector in sector_etfs.items():
            try:
                holdings = self.fetch_etf_holdings(etf_symbol)
                all_holdings.update(holdings)
                self.log_debug(f"Added {len(holdings)} holdings from {etf_symbol} ({sector})")
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                self.log_debug(f"Failed to fetch {etf_symbol}: {e}")
        
        self.all_stocks.update(all_holdings)
        self.log_debug(f"Total sector ETF holdings: {len(all_holdings)}")
    
    def fetch_style_etf_holdings(self):
        """Fetch holdings from style-based ETFs"""
        style_etfs = {
            'IVV': 'S&P 500', 'IWM': 'Russell 2000', 'IWB': 'Russell 1000',
            'VTI': 'Total Stock Market', 'VXF': 'Extended Market',
            'IJR': 'S&P Small Cap', 'IJH': 'S&P Mid Cap',
            'VB': 'Small Cap Value', 'VBK': 'Small Cap Growth',
            'VO': 'Mid Cap', 'VV': 'Large Cap', 'VUG': 'Growth',
            'VTV': 'Value', 'MTUM': 'Momentum', 'QUAL': 'Quality',
            'SIZE': 'Size Factor', 'USMV': 'Min Volatility'
        }
        
        all_holdings = set()
        for etf_symbol, style in style_etfs.items():
            try:
                holdings = self.fetch_etf_holdings(etf_symbol)
                all_holdings.update(holdings)
                self.log_debug(f"Added {len(holdings)} holdings from {etf_symbol} ({style})")
                time.sleep(0.1)
            except Exception as e:
                self.log_debug(f"Failed to fetch {etf_symbol}: {e}")
        
        self.all_stocks.update(all_holdings)
        self.log_debug(f"Total style ETF holdings: {len(all_holdings)}")
    
    def fetch_international_etf_holdings(self):
        """Fetch holdings from international ETFs (US-listed foreign stocks)"""
        international_etfs = {
            'VEA': 'Developed Markets', 'VWO': 'Emerging Markets',
            'IEFA': 'Europe', 'IEMG': 'Emerging Markets Core',
            'EFA': 'Europe/Pacific', 'EEM': 'Emerging Markets',
            'VGK': 'Europe', 'VPL': 'Pacific', 'VSS': 'International Small Cap',
            'ACWX': 'World ex-US', 'VXUS': 'International Stock',
            'FEZ': 'Eurozone', 'ASHR': 'China A-Shares'
        }
        
        all_holdings = set()
        for etf_symbol, region in international_etfs.items():
            try:
                holdings = self.fetch_etf_holdings(etf_symbol)
                # Filter for US-listed symbols only
                us_holdings = [h for h in holdings if len(h) <= 5 and '.' not in h]
                all_holdings.update(us_holdings)
                self.log_debug(f"Added {len(us_holdings)} US-listed holdings from {etf_symbol} ({region})")
                time.sleep(0.1)
            except Exception as e:
                self.log_debug(f"Failed to fetch {etf_symbol}: {e}")
        
        self.all_stocks.update(all_holdings)
        self.log_debug(f"Total international ETF US holdings: {len(all_holdings)}")
    
    def fetch_commodity_etf_holdings(self):
        """Fetch holdings from commodity and resource ETFs"""
        commodity_etfs = {
            'GLD': 'Gold', 'SLV': 'Silver', 'DBO': 'Oil',
            'UNG': 'Natural Gas', 'DBA': 'Agriculture',
            'PDBC': 'Commodities', 'GSG': 'Commodities Broad',
            'XME': 'Metals & Mining', 'MOO': 'Agriculture',
            'WOOD': 'Timber', 'PICK': 'Mining'
        }
        
        all_holdings = set()
        for etf_symbol, commodity in commodity_etfs.items():
            try:
                holdings = self.fetch_etf_holdings(etf_symbol)
                all_holdings.update(holdings)
                self.log_debug(f"Added {len(holdings)} holdings from {etf_symbol} ({commodity})")
                time.sleep(0.1)
            except Exception as e:
                self.log_debug(f"Failed to fetch {etf_symbol}: {e}")
        
        self.all_stocks.update(all_holdings)
        self.log_debug(f"Total commodity ETF holdings: {len(all_holdings)}")
    
    def fetch_crypto_related_stocks(self):
        """Fetch cryptocurrency and blockchain-related stocks"""
        crypto_stocks = [
            'COIN', 'MSTR', 'RIOT', 'MARA', 'CAN', 'BITF', 'HUT', 
            'BTBT', 'SOS', 'EBON', 'ANY', 'FRMO', 'OSTK', 'PYPL',
            'SQ', 'NVDA', 'AMD', 'TSM', 'INTC'  # Crypto mining hardware
        ]
        
        # Add crypto ETFs holdings
        crypto_etfs = ['BITO', 'BITI', 'BLOK', 'LEGR', 'KOIN']
        for etf in crypto_etfs:
            try:
                holdings = self.fetch_etf_holdings(etf)
                crypto_stocks.extend(holdings)
                time.sleep(0.1)
            except:
                pass
        
        self.all_stocks.update(crypto_stocks)
        self.log_debug(f"Added {len(set(crypto_stocks))} crypto-related stocks")
    
    def fetch_penny_stocks(self):
        """Fetch penny stocks and micro-cap stocks"""
        try:
            # Use OTC markets data or micro-cap ETF holdings
            microcap_etfs = ['IWC', 'PZI', 'FDM']  # Micro-cap ETFs
            penny_stocks = set()
            
            for etf in microcap_etfs:
                try:
                    holdings = self.fetch_etf_holdings(etf)
                    penny_stocks.update(holdings)
                    time.sleep(0.1)
                except:
                    pass
            
            self.all_stocks.update(penny_stocks)
            self.log_debug(f"Added {len(penny_stocks)} micro-cap/penny stocks")
            
        except Exception as e:
            self.log_debug(f"Penny stocks fetch failed: {e}")
    
    def fetch_reit_stocks(self):
        """Fetch Real Estate Investment Trust stocks"""
        reit_etfs = ['VNQ', 'SCHH', 'IYR', 'FREL', 'REM', 'MORT', 'RWR']
        all_reits = set()
        
        for etf in reit_etfs:
            try:
                holdings = self.fetch_etf_holdings(etf)
                all_reits.update(holdings)
                time.sleep(0.1)
            except:
                pass
        
        self.all_stocks.update(all_reits)
        self.log_debug(f"Added {len(all_reits)} REIT stocks")
    
    def fetch_utility_stocks(self):
        """Fetch utility sector stocks"""
        utility_etfs = ['XLU', 'VPU', 'IDU', 'FUTY']
        all_utilities = set()
        
        for etf in utility_etfs:
            try:
                holdings = self.fetch_etf_holdings(etf)
                all_utilities.update(holdings)
                time.sleep(0.1)
            except:
                pass
        
        self.all_stocks.update(all_utilities)
        self.log_debug(f"Added {len(all_utilities)} utility stocks")
    
    def fetch_etf_holdings(self, etf_symbol: str) -> List[str]:
        """Fetch ETF holdings using yfinance"""
        try:
            ticker = yf.Ticker(etf_symbol)
            # Get basic info first
            info = ticker.info
            
            # Try to get holdings data (not always available in yfinance)
            # This is a limitation - yfinance doesn't provide detailed holdings
            # For production, you'd need a data provider like Alpha Vantage, IEX, etc.
            
            # For now, return empty list as yfinance doesn't provide holdings
            return []
            
        except Exception as e:
            self.log_debug(f"ETF {etf_symbol} fetch failed: {e}")
            return []
    
    def clean_and_filter_stocks(self) -> List[str]:
        """Clean and filter the stock universe"""
        # Convert to list and remove empty/invalid symbols
        stocks = list(self.all_stocks)
        
        # Basic filtering
        filtered = []
        for stock in stocks:
            if (stock and 
                isinstance(stock, str) and 
                len(stock.strip()) > 0 and 
                len(stock.strip()) <= 5 and  # Most US stocks are 1-5 characters
                stock.strip().replace('.', '').replace('-', '').isalpha() and  # Only letters (with dots/dashes)
                not stock.startswith('.')):
                filtered.append(stock.strip().upper())
        
        # Remove duplicates and sort
        unique_stocks = sorted(list(set(filtered)))
        
        self.log_debug(f"Cleaned {len(stocks)} raw symbols to {len(unique_stocks)} valid stocks")
        return unique_stocks

def get_massive_stock_universe() -> List[str]:
    """Main function to get massive stock universe"""
    builder = MassiveStockUniverseBuilder()
    return builder.get_all_stocks()

if __name__ == "__main__":
    # Test the massive universe builder
    stocks = get_massive_stock_universe()
    print(f"Total stocks fetched: {len(stocks)}")
    print("Sample stocks:", stocks[:20])