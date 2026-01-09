from .config import SETTINGS
from pathlib import Path
try:
    from tigeropen.tiger_open_config import TigerOpenClientConfig
    from tigeropen.common.util.signature_utils import read_private_key
    from tigeropen.quote.quote_client import QuoteClient
    from tigeropen.trade.trade_client import TradeClient
    SDK_AVAILABLE = True
except Exception:
    SDK_AVAILABLE = False

def build_clients():
    # Allow running without SDK/credentials when DRY_RUN is enabled
    if SETTINGS.dry_run:
        return None, None

    if not SDK_AVAILABLE:
        raise RuntimeError("TradeUP SDK not available. Install it from GitHub.")

    props_dir = str((Path(__file__).resolve().parents[1] / 'props').resolve())
    cfg = TigerOpenClientConfig(props_path=props_dir)

    # 优先使用 .env 覆盖（仅当 .env 中有非空值时才覆盖 props 文件中的配置）
    if SETTINGS.tiger_id and SETTINGS.tiger_id.strip():
        cfg.tiger_id = SETTINGS.tiger_id
    if SETTINGS.account and SETTINGS.account.strip():
        cfg.account = SETTINGS.account
    if SETTINGS.secret_key and SETTINGS.secret_key.strip():
        cfg.secret_key = SETTINGS.secret_key
    if getattr(SETTINGS, 'private_key_path', '') and SETTINGS.private_key_path.strip():
        cfg.private_key = read_private_key(SETTINGS.private_key_path)
    cfg.timezone = SETTINGS.timezone
    cfg.language = SETTINGS.lang
    return QuoteClient(cfg), TradeClient(cfg)
