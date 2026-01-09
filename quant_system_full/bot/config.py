import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Settings:
    tiger_id: str = os.getenv("TIGER_ID","")
    account: str = os.getenv("ACCOUNT","")
    private_key_path: str = os.getenv("PRIVATE_KEY_PATH","")
    secret_key: str = os.getenv("SECRET_KEY","")
    timezone: str = os.getenv("TIMEZONE","US/Eastern")
    lang: str = os.getenv("LANG","en_US")
    dry_run: bool = os.getenv("DRY_RUN","true").lower()=="true"
    
    # Data source preferences: "tiger", "yahoo_api", "yahoo_mcp", "auto" 
    # "auto" tries yahoo_api -> yahoo_mcp -> tiger -> placeholder
    data_source: str = os.getenv("DATA_SOURCE","auto")
    
    # Yahoo Finance API settings (extended for 3-hour selection cycle)
    yahoo_api_timeout: float = float(os.getenv("YAHOO_API_TIMEOUT", "20.0"))
    yahoo_api_retries: int = int(os.getenv("YAHOO_API_RETRIES", "3"))
    yahoo_api_retry_delay: float = float(os.getenv("YAHOO_API_RETRY_DELAY", "3.0"))
    
    # Yahoo Finance MCP proxy settings (optional)
    yahoo_mcp_proxy_url: str = os.getenv("YAHOO_MCP_PROXY_URL","")
    
    # Whether to use MCP tools for enhanced data access
    use_mcp_tools: bool = os.getenv("USE_MCP_TOOLS","true").lower()=="true"
    
    # Market configuration
    primary_market: str = os.getenv("PRIMARY_MARKET", "US")  # US or CN
    secondary_markets: str = os.getenv("SECONDARY_MARKETS", "")  # Comma-separated list
    market_data_timezone: str = os.getenv("MARKET_DATA_TIMEZONE", "US/Eastern")
    
    # Selection strategy configuration
    default_selection_strategy: str = os.getenv("DEFAULT_SELECTION_STRATEGY", "value_momentum")
    selection_universe_size: int = int(os.getenv("SELECTION_UNIVERSE_SIZE", "500"))
    selection_result_size: int = int(os.getenv("SELECTION_RESULT_SIZE", "20"))
    selection_min_score: float = float(os.getenv("SELECTION_MIN_SCORE", "0.0"))
    
    # Task scheduling configuration
    selection_task_interval: int = int(os.getenv("SELECTION_TASK_INTERVAL", "10800"))  # 3 hours (to avoid API rate limits)
    trading_task_interval: int = int(os.getenv("TRADING_TASK_INTERVAL", "30"))      # 30 seconds
    monitoring_task_interval: int = int(os.getenv("MONITORING_TASK_INTERVAL", "120")) # 2 minutes
    recovery_task_interval: int = int(os.getenv("RECOVERY_TASK_INTERVAL", "300"))   # 5 minutes
    
    # Stock universe configuration
    stock_universe_file: str = os.getenv("STOCK_UNIVERSE_FILE", "")  # Path to CSV file
    exclude_sectors: str = os.getenv("EXCLUDE_SECTORS", "")  # Comma-separated
    include_sectors: str = os.getenv("INCLUDE_SECTORS", "")  # Comma-separated
    min_market_cap: float = float(os.getenv("MIN_MARKET_CAP", "1000000000"))  # $1B
    max_market_cap: float = float(os.getenv("MAX_MARKET_CAP", "1000000000000"))  # $1T
    min_daily_volume: int = int(os.getenv("MIN_DAILY_VOLUME", "100000"))
    min_stock_price: float = float(os.getenv("MIN_STOCK_PRICE", "5.0"))
    max_stock_price: float = float(os.getenv("MAX_STOCK_PRICE", "1000.0"))
    
    def get_secondary_markets_list(self) -> list[str]:
        """Get secondary markets as a list."""
        if not self.secondary_markets:
            return []
        return [m.strip().upper() for m in self.secondary_markets.split(",") if m.strip()]
    
    def get_exclude_sectors_list(self) -> list[str]:
        """Get excluded sectors as a list."""
        if not self.exclude_sectors:
            return []
        return [s.strip() for s in self.exclude_sectors.split(",") if s.strip()]
    
    def get_include_sectors_list(self) -> list[str]:
        """Get included sectors as a list."""
        if not self.include_sectors:
            return []
        return [s.strip() for s in self.include_sectors.split(",") if s.strip()]

SETTINGS = Settings()
