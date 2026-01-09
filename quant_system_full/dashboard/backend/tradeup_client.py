try:
    from .config import SETTINGS
except ImportError:
    from config import SETTINGS
from pathlib import Path
import logging

try:
    from tigeropen.tiger_open_config import TigerOpenClientConfig
    from tigeropen.common.util.signature_utils import read_private_key
    from tigeropen.quote.quote_client import QuoteClient
    from tigeropen.trade.trade_client import TradeClient
    SDK_AVAILABLE = True
except Exception as e:
    SDK_AVAILABLE = False
    logging.warning(f"Tiger SDK not available: {e}")

def build_clients():
    """Build Tiger API quote and trade clients"""
    # Allow running without SDK/credentials when DRY_RUN is enabled
    if SETTINGS.dry_run:
        logging.info("DRY_RUN enabled, returning None clients")
        return None, None

    if not SDK_AVAILABLE:
        raise RuntimeError("Tiger SDK not available. Install it: pip install tigeropen")

    try:
        props_dir = str((Path(__file__).resolve().parents[2] / 'props').resolve())
        cfg = TigerOpenClientConfig(props_path=props_dir)

        # Override with .env values when non-empty (but NOT private key - use props file key)
        if SETTINGS.tiger_id and SETTINGS.tiger_id.strip():
            cfg.tiger_id = SETTINGS.tiger_id
        if SETTINGS.account and SETTINGS.account.strip():
            cfg.account = SETTINGS.account
        if SETTINGS.secret_key and SETTINGS.secret_key.strip():
            cfg.secret_key = SETTINGS.secret_key
        # Note: Private key is loaded from props file, not from .env path
        
        cfg.timezone = SETTINGS.timezone
        cfg.language = SETTINGS.lang

        logging.info(f"Initializing Tiger clients for account: {cfg.account}")
        quote_client = QuoteClient(cfg)
        trade_client = TradeClient(cfg)
        
        logging.info("Tiger API clients initialized successfully")
        return quote_client, trade_client
        
    except Exception as e:
        logging.error(f"Failed to initialize Tiger clients: {e}")
        raise RuntimeError(f"Tiger client initialization failed: {e}")

def get_client_config():
    """Get Tiger client configuration for direct API usage"""
    if not SDK_AVAILABLE:
        raise RuntimeError("Tiger SDK not available")
        
    props_dir = str((Path(__file__).resolve().parents[2] / 'props').resolve())
    cfg = TigerOpenClientConfig(props_path=props_dir)
    
    # Override with .env values (but NOT private key - use props file key)
    if SETTINGS.tiger_id:
        cfg.tiger_id = SETTINGS.tiger_id
    if SETTINGS.account:
        cfg.account = SETTINGS.account  
    if SETTINGS.secret_key:
        cfg.secret_key = SETTINGS.secret_key
    # Note: Private key is loaded from props file, not from .env path
        
    return cfg
