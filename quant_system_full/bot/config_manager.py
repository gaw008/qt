#!/usr/bin/env python3
"""
Centralized Configuration Manager
Resolves hardcoded values and port conflicts
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ServerConfig:
    """Server configuration with environment variable support"""
    api_port: int = int(os.getenv("API_PORT", "8000"))
    frontend_port: int = int(os.getenv("FRONTEND_PORT", "3000"))
    websocket_port: int = int(os.getenv("WEBSOCKET_PORT", "8001"))
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))

    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    cors_origins: list = None

    def __post_init__(self):
        if self.cors_origins is None:
            origins_str = os.getenv("CORS_ORIGINS",
                f"http://localhost:{self.frontend_port},http://localhost:5173,http://localhost:8080")
            self.cors_origins = [origin.strip() for origin in origins_str.split(",")]


@dataclass
class TradingConfig:
    """Trading system configuration"""
    # AI Configuration
    enable_daily_ai_training: bool = os.getenv("ENABLE_DAILY_AI_TRAINING", "true").lower() == "true"
    ai_training_frequency: str = os.getenv("AI_TRAINING_FREQUENCY", "daily")
    ai_model_type: str = os.getenv("AI_MODEL_TYPE", "lightgbm")
    ai_target_metric: str = os.getenv("AI_TARGET_METRIC", "sharpe_ratio")
    ai_data_source: str = os.getenv("AI_DATA_SOURCE", "yahoo_api")
    ai_selection_weight: float = float(os.getenv("AI_SELECTION_WEIGHT", "0.4"))
    ai_trading_weight: float = float(os.getenv("AI_TRADING_WEIGHT", "0.6"))
    ai_min_training_days: int = int(os.getenv("AI_MIN_TRAINING_DAYS", "30"))
    ai_auto_retrain_threshold: float = float(os.getenv("AI_AUTO_RETRAIN_THRESHOLD", "0.3"))

    # Trading Configuration
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    data_source: str = os.getenv("DATA_SOURCE", "auto")
    tiger_id: str = os.getenv("TIGER_ID", "")
    account: str = os.getenv("ACCOUNT", "")

    # Risk Management
    daily_loss_limit: float = float(os.getenv("DAILY_LOSS_LIMIT", "0.05"))
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))

    # Performance Settings
    selection_universe_size: int = int(os.getenv("SELECTION_UNIVERSE_SIZE", "4000"))
    selection_result_size: int = int(os.getenv("SELECTION_RESULT_SIZE", "20"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "1000"))
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))


@dataclass
class PathConfig:
    """Path configuration with dynamic resolution"""
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @property
    def private_key_path(self) -> str:
        env_path = os.getenv("PRIVATE_KEY_PATH")
        if env_path and os.path.isabs(env_path):
            return env_path
        return os.path.join(self.base_dir, "private_key.pem")

    @property
    def state_dir(self) -> str:
        return os.path.join(self.base_dir, "dashboard", "state")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.base_dir, "data_cache")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.base_dir, "reports")


class ConfigurationManager:
    """Centralized configuration manager"""

    def __init__(self):
        self.server = ServerConfig()
        self.trading = TradingConfig()
        self.paths = PathConfig()

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            "server": self.server.__dict__,
            "trading": self.trading.__dict__,
            "paths": self.paths.__dict__
        }

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        warnings = []

        # Check port conflicts
        ports = [self.server.api_port, self.server.frontend_port,
                self.server.websocket_port, self.server.streamlit_port]
        if len(ports) != len(set(ports)):
            issues.append("Port conflicts detected in server configuration")

        # Check required paths
        if not os.path.exists(self.paths.private_key_path):
            warnings.append(f"Private key not found at: {self.paths.private_key_path}")

        # Check required trading credentials
        if not self.trading.tiger_id:
            warnings.append("Tiger ID not configured")

        # Check AI configuration consistency
        if self.trading.ai_selection_weight + self.trading.ai_trading_weight > 2.0:
            warnings.append("AI weights configuration may be inconsistent")

        return {
            "status": "valid" if not issues else "invalid",
            "issues": issues,
            "warnings": warnings
        }


# Global configuration instance
_config_manager: Optional[ConfigurationManager] = None


def get_config() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def reload_config() -> ConfigurationManager:
    """Reload configuration from environment"""
    global _config_manager
    load_dotenv(override=True)  # Reload .env file
    _config_manager = ConfigurationManager()
    return _config_manager


# Export commonly used configurations
config = get_config()
server_config = config.server
trading_config = config.trading
path_config = config.paths