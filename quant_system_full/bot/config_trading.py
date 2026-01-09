"""
Enhanced Trading Configuration System
Replaces hardcoded values with dynamic, configurable parameters
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    UNKNOWN = "unknown"

class TradingMode(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"

@dataclass
class TradingThresholds:
    """Trading signal thresholds with market regime adaptation"""
    
    # Z-Score based thresholds (方案A) - NEW PRIMARY SYSTEM
    z_strong_buy_threshold: float = float(os.getenv("Z_STRONG_BUY_THRESHOLD", "0.8"))  # z >= +0.8
    z_buy_threshold: float = float(os.getenv("Z_BUY_THRESHOLD", "0.4"))              # z >= +0.4
    z_reduce_threshold: float = float(os.getenv("Z_REDUCE_THRESHOLD", "0.0"))        # z < +0.4 but >= 0
    z_sell_threshold: float = float(os.getenv("Z_SELL_THRESHOLD", "0.0"))           # z < 0 (clearance)
    
    # Legacy raw score thresholds (FALLBACK ONLY)
    strong_buy_threshold: float = float(os.getenv("STRONG_BUY_THRESHOLD", "80.0"))
    buy_threshold: float = float(os.getenv("BUY_THRESHOLD", "70.0"))
    hold_threshold: float = float(os.getenv("HOLD_THRESHOLD", "50.0"))
    sell_threshold: float = float(os.getenv("SELL_THRESHOLD", "30.0"))
    strong_sell_threshold: float = float(os.getenv("STRONG_SELL_THRESHOLD", "20.0"))
    
    # Z-Score system configuration
    use_z_score_system: bool = os.getenv("USE_Z_SCORE_SYSTEM", "true").lower() == "true"
    z_clip_extreme_values: bool = os.getenv("Z_CLIP_EXTREME_VALUES", "true").lower() == "true"
    z_clip_min: float = float(os.getenv("Z_CLIP_MIN", "-5.0"))
    z_clip_max: float = float(os.getenv("Z_CLIP_MAX", "5.0"))
    
    # Technical signal thresholds
    bullish_technical_threshold: float = float(os.getenv("BULLISH_TECHNICAL_THRESHOLD", "0.6"))
    bearish_technical_threshold: float = float(os.getenv("BEARISH_TECHNICAL_THRESHOLD", "0.4"))
    
    # Confidence thresholds
    min_confidence_threshold: float = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.6"))
    high_confidence_threshold: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.8"))
    
    # Regime-based adjustments
    regime_adjustments: Dict[MarketRegime, Dict[str, float]] = field(default_factory=lambda: {
        MarketRegime.BULL: {
            "strong_buy_threshold": 75.0,  # Lower threshold in bull market
            "buy_threshold": 65.0,
            "multiplier": 1.1
        },
        MarketRegime.BEAR: {
            "strong_buy_threshold": 85.0,  # Higher threshold in bear market
            "buy_threshold": 75.0,
            "multiplier": 0.9
        },
        MarketRegime.SIDEWAYS: {
            "strong_buy_threshold": 80.0,  # Standard thresholds
            "buy_threshold": 70.0,
            "multiplier": 1.0
        },
        MarketRegime.CRISIS: {
            "strong_buy_threshold": 90.0,  # Very high threshold in crisis
            "buy_threshold": 80.0,
            "multiplier": 0.8
        }
    })

@dataclass
class TechnicalFactors:
    """Technical analysis parameters"""
    
    # RSI parameters
    rsi_period: int = int(os.getenv("RSI_PERIOD", "14"))
    rsi_overbought: float = float(os.getenv("RSI_OVERBOUGHT", "70.0"))
    rsi_oversold: float = float(os.getenv("RSI_OVERSOLD", "30.0"))
    
    # MACD parameters
    macd_fast_period: int = int(os.getenv("MACD_FAST_PERIOD", "12"))
    macd_slow_period: int = int(os.getenv("MACD_SLOW_PERIOD", "26"))
    macd_signal_period: int = int(os.getenv("MACD_SIGNAL_PERIOD", "9"))
    
    # Bollinger Bands
    bollinger_period: int = int(os.getenv("BOLLINGER_PERIOD", "20"))
    bollinger_std_multiplier: float = float(os.getenv("BOLLINGER_STD_MULTIPLIER", "2.0"))
    
    # Moving Averages
    sma_short_period: int = int(os.getenv("SMA_SHORT_PERIOD", "10"))
    sma_medium_period: int = int(os.getenv("SMA_MEDIUM_PERIOD", "20"))
    sma_long_period: int = int(os.getenv("SMA_LONG_PERIOD", "50"))
    
    # Volume indicators
    volume_sma_period: int = int(os.getenv("VOLUME_SMA_PERIOD", "20"))
    obv_period: int = int(os.getenv("OBV_PERIOD", "20"))
    
    # Volatility
    atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
    volatility_period: int = int(os.getenv("VOLATILITY_PERIOD", "20"))
    
    # Support/Resistance
    support_resistance_period: int = int(os.getenv("SUPPORT_RESISTANCE_PERIOD", "20"))
    breakout_threshold: float = float(os.getenv("BREAKOUT_THRESHOLD", "0.02"))

@dataclass
class RiskParameters:
    """Risk management parameters with dynamic sizing"""
    
    # Position sizing
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.05"))  # 5% max
    min_position_size: float = float(os.getenv("MIN_POSITION_SIZE", "0.005"))  # 0.5% min
    default_position_size: float = float(os.getenv("DEFAULT_POSITION_SIZE", "0.02"))  # 2% default
    
    # Risk limits
    portfolio_risk_limit: float = float(os.getenv("PORTFOLIO_RISK_LIMIT", "0.15"))  # 15% portfolio risk
    daily_loss_limit: float = float(os.getenv("DAILY_LOSS_LIMIT", "0.03"))  # 3% daily loss limit
    max_drawdown_limit: float = float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.10"))  # 10% max drawdown
    
    # Stop loss parameters
    stop_loss_threshold: float = float(os.getenv("STOP_LOSS_THRESHOLD", "0.05"))  # 5% stop loss
    trailing_stop_distance: float = float(os.getenv("TRAILING_STOP_DISTANCE", "0.03"))  # 3% trailing
    
    # Kelly criterion parameters
    use_kelly_criterion: bool = os.getenv("USE_KELLY_CRITERION", "true").lower() == "true"
    kelly_multiplier: float = float(os.getenv("KELLY_MULTIPLIER", "0.25"))  # Conservative Kelly
    max_kelly_size: float = float(os.getenv("MAX_KELLY_SIZE", "0.08"))  # Cap Kelly at 8%
    
    # Volatility targeting
    target_volatility: float = float(os.getenv("TARGET_VOLATILITY", "0.15"))  # 15% annualized
    volatility_lookback: int = int(os.getenv("VOLATILITY_LOOKBACK", "252"))  # 1 year
    
    # Correlation limits
    max_correlation: float = float(os.getenv("MAX_CORRELATION", "0.7"))  # Max 70% correlation
    correlation_period: int = int(os.getenv("CORRELATION_PERIOD", "60"))  # 60-day correlation

@dataclass
class StockFilters:
    """Stock universe filtering criteria"""
    
    # Price filters
    min_stock_price: float = float(os.getenv("MIN_STOCK_PRICE", "5.0"))
    max_stock_price: float = float(os.getenv("MAX_STOCK_PRICE", "1000.0"))
    
    # Volume filters
    min_daily_volume: int = int(os.getenv("MIN_DAILY_VOLUME", "100000"))
    min_dollar_volume: float = float(os.getenv("MIN_DOLLAR_VOLUME", "5000000"))  # $5M daily
    
    # Market cap filters
    min_market_cap: float = float(os.getenv("MIN_MARKET_CAP", "1000000000"))  # $1B
    max_market_cap: float = float(os.getenv("MAX_MARKET_CAP", "5000000000000"))  # $5T
    
    # Fundamental filters
    min_pe_ratio: float = float(os.getenv("MIN_PE_RATIO", "1.0"))
    max_pe_ratio: float = float(os.getenv("MAX_PE_RATIO", "100.0"))
    min_revenue_growth: float = float(os.getenv("MIN_REVENUE_GROWTH", "-0.5"))  # -50% min growth
    
    # Quality filters
    min_analyst_coverage: int = int(os.getenv("MIN_ANALYST_COVERAGE", "3"))
    exclude_penny_stocks: bool = os.getenv("EXCLUDE_PENNY_STOCKS", "true").lower() == "true"
    exclude_new_ipos: bool = os.getenv("EXCLUDE_NEW_IPOS", "true").lower() == "true"
    ipo_exclusion_days: int = int(os.getenv("IPO_EXCLUSION_DAYS", "180"))  # 6 months
    
    # Sector and exchange filters
    allowed_exchanges: List[str] = field(default_factory=lambda: 
        os.getenv("ALLOWED_EXCHANGES", "NASDAQ,NYSE").split(","))
    excluded_sectors: List[str] = field(default_factory=lambda: 
        os.getenv("EXCLUDED_SECTORS", "").split(",") if os.getenv("EXCLUDED_SECTORS") else [])

@dataclass
class MarketAdjustmentConfig:
    """Dynamic market adjustment parameters"""
    
    # VIX-based adjustment
    vix_low_threshold: float = float(os.getenv("VIX_LOW_THRESHOLD", "15.0"))
    vix_normal_threshold: float = float(os.getenv("VIX_NORMAL_THRESHOLD", "20.0"))
    vix_elevated_threshold: float = float(os.getenv("VIX_ELEVATED_THRESHOLD", "25.0"))
    vix_high_threshold: float = float(os.getenv("VIX_HIGH_THRESHOLD", "30.0"))
    
    # VIX factor mapping
    vix_factors: Dict[str, float] = field(default_factory=lambda: {
        "low": float(os.getenv("VIX_LOW_FACTOR", "1.15")),      # Bullish
        "normal": float(os.getenv("VIX_NORMAL_FACTOR", "1.05")), # Slightly bullish
        "elevated": float(os.getenv("VIX_ELEVATED_FACTOR", "0.95")), # Slightly bearish
        "high": float(os.getenv("VIX_HIGH_FACTOR", "0.85")),    # Bearish
        "extreme": float(os.getenv("VIX_EXTREME_FACTOR", "0.75")) # Very bearish
    })
    
    # Factor weights
    market_trend_weight: float = float(os.getenv("MARKET_TREND_WEIGHT", "0.20"))
    vix_weight: float = float(os.getenv("VIX_WEIGHT", "0.30"))
    breadth_weight: float = float(os.getenv("BREADTH_WEIGHT", "0.20"))
    dollar_weight: float = float(os.getenv("DOLLAR_WEIGHT", "0.15"))
    yield_weight: float = float(os.getenv("YIELD_WEIGHT", "0.15"))
    
    # Adjustment bounds
    min_adjustment_factor: float = float(os.getenv("MIN_ADJUSTMENT_FACTOR", "0.7"))
    max_adjustment_factor: float = float(os.getenv("MAX_ADJUSTMENT_FACTOR", "1.3"))
    
    # Update frequency
    adjustment_update_minutes: int = int(os.getenv("ADJUSTMENT_UPDATE_MINUTES", "30"))

@dataclass
class FactorWeights:
    """Multi-factor model weights with regime awareness"""
    
    # Base factor weights (sum should = 1.0)
    valuation_weight: float = float(os.getenv("VALUATION_WEIGHT", "0.25"))
    momentum_weight: float = float(os.getenv("MOMENTUM_WEIGHT", "0.25"))
    technical_weight: float = float(os.getenv("TECHNICAL_WEIGHT", "0.20"))
    volume_weight: float = float(os.getenv("VOLUME_WEIGHT", "0.15"))
    sentiment_weight: float = float(os.getenv("SENTIMENT_WEIGHT", "0.15"))
    
    # Regime-specific weight adjustments
    regime_weight_adjustments: Dict[MarketRegime, Dict[str, float]] = field(default_factory=lambda: {
        MarketRegime.BULL: {
            "momentum_weight": 0.35,    # Increase momentum in bull market
            "valuation_weight": 0.15,   # Decrease valuation focus
            "sentiment_weight": 0.20
        },
        MarketRegime.BEAR: {
            "valuation_weight": 0.35,   # Focus on value in bear market
            "momentum_weight": 0.15,    # Reduce momentum focus
            "technical_weight": 0.25
        },
        MarketRegime.CRISIS: {
            "technical_weight": 0.40,   # Technical analysis critical in crisis
            "sentiment_weight": 0.25,   # Monitor sentiment closely
            "valuation_weight": 0.20
        }
    })

@dataclass
class ExecutionConfig:
    """Order execution and timing parameters"""
    
    # Execution algorithms
    default_execution_algo: str = os.getenv("DEFAULT_EXECUTION_ALGO", "VWAP")  # VWAP, TWAP, POV
    max_participation_rate: float = float(os.getenv("MAX_PARTICIPATION_RATE", "0.10"))  # 10% of volume
    
    # Order timing
    max_order_duration_minutes: int = int(os.getenv("MAX_ORDER_DURATION_MINUTES", "60"))
    order_slice_size_shares: int = int(os.getenv("ORDER_SLICE_SIZE_SHARES", "100"))
    order_slice_interval_seconds: int = int(os.getenv("ORDER_SLICE_INTERVAL_SECONDS", "30"))
    
    # Smart routing
    use_smart_routing: bool = os.getenv("USE_SMART_ROUTING", "true").lower() == "true"
    preferred_venues: List[str] = field(default_factory=lambda: 
        os.getenv("PREFERRED_VENUES", "ARCA,NASDAQ,NYSE").split(","))
    
    # Market impact control
    max_market_impact_bps: float = float(os.getenv("MAX_MARKET_IMPACT_BPS", "5.0"))  # 5 basis points
    liquidity_threshold: float = float(os.getenv("LIQUIDITY_THRESHOLD", "0.01"))  # 1% of daily volume
    
    # Slippage control
    max_slippage_bps: float = float(os.getenv("MAX_SLIPPAGE_BPS", "10.0"))  # 10 basis points
    slippage_monitoring_period: int = int(os.getenv("SLIPPAGE_MONITORING_PERIOD", "30"))  # 30 days

@dataclass
class PerformanceConfig:
    """Performance and optimization parameters"""
    
    # Data processing
    batch_size: int = int(os.getenv("SCREENING_BATCH_SIZE", "200"))
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
    cache_duration_minutes: int = int(os.getenv("CACHE_DURATION_MINUTES", "5"))
    
    # Memory management
    memory_cache_ratio: float = float(os.getenv("MEMORY_CACHE_RATIO", "0.50"))
    max_memory_usage_gb: float = float(os.getenv("MAX_MEMORY_USAGE_GB", "8.0"))
    
    # GPU settings (if available)
    use_gpu: bool = os.getenv("USE_GPU", "false").lower() == "true"
    gpu_memory_fraction: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.80"))
    
    # Threading
    max_worker_threads: int = int(os.getenv("MAX_WORKER_THREADS", "8"))
    async_processing: bool = os.getenv("ASYNC_PROCESSING", "true").lower() == "true"

@dataclass
class ModelConfig:
    """Machine learning model parameters"""
    
    # Model types
    use_ml_models: bool = os.getenv("USE_ML_MODELS", "true").lower() == "true"
    ml_model_weight: float = float(os.getenv("ML_MODEL_WEIGHT", "0.7"))  # 70% ML, 30% traditional
    
    # Model update frequency
    model_retrain_days: int = int(os.getenv("MODEL_RETRAIN_DAYS", "7"))
    performance_check_days: int = int(os.getenv("PERFORMANCE_CHECK_DAYS", "1"))
    
    # Model thresholds
    model_confidence_threshold: float = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.6"))
    strategy_elimination_threshold: float = float(os.getenv("STRATEGY_ELIMINATION_THRESHOLD", "0.3"))
    
    # Ensemble settings
    max_models_in_ensemble: int = int(os.getenv("MAX_MODELS_IN_ENSEMBLE", "5"))
    model_diversity_threshold: float = float(os.getenv("MODEL_DIVERSITY_THRESHOLD", "0.3"))

@dataclass
class TradingModeProfiles:
    """Pre-configured trading mode profiles"""
    
    profiles: Dict[TradingMode, Dict] = field(default_factory=lambda: {
        TradingMode.CONSERVATIVE: {
            "strong_buy_threshold": 85.0,
            "buy_threshold": 75.0,
            "max_position_size": 0.03,
            "stop_loss_threshold": 0.03,
            "min_confidence_threshold": 0.75,
            "market_adjustment_cap": 1.1
        },
        TradingMode.MODERATE: {
            "strong_buy_threshold": 80.0,
            "buy_threshold": 70.0,
            "max_position_size": 0.05,
            "stop_loss_threshold": 0.05,
            "min_confidence_threshold": 0.6,
            "market_adjustment_cap": 1.2
        },
        TradingMode.AGGRESSIVE: {
            "strong_buy_threshold": 75.0,
            "buy_threshold": 65.0,
            "max_position_size": 0.08,
            "stop_loss_threshold": 0.07,
            "min_confidence_threshold": 0.5,
            "market_adjustment_cap": 1.3
        },
        TradingMode.ULTRA_AGGRESSIVE: {
            "strong_buy_threshold": 70.0,
            "buy_threshold": 60.0,
            "max_position_size": 0.10,
            "stop_loss_threshold": 0.08,
            "min_confidence_threshold": 0.4,
            "market_adjustment_cap": 1.5
        }
    })

class EnhancedTradingConfig:
    """Main configuration class that combines all config sections"""
    
    def __init__(self, trading_mode: TradingMode = None):
        # Load base configurations
        self.thresholds = TradingThresholds()
        self.technical = TechnicalFactors()
        self.risk = RiskParameters()
        self.filters = StockFilters()
        self.market_adjustment = MarketAdjustmentConfig()
        self.factor_weights = FactorWeights()
        self.execution = ExecutionConfig()
        self.performance = PerformanceConfig()
        self.model = ModelConfig()
        
        # Load trading mode profile
        self.trading_mode = trading_mode or TradingMode(os.getenv("TRADING_MODE", "moderate"))
        self.profiles = TradingModeProfiles()
        
        # Apply trading mode overrides
        self.apply_trading_mode_profile()
        
        # Current market regime (updated dynamically)
        self.current_regime = MarketRegime.UNKNOWN
        
    def apply_trading_mode_profile(self):
        """Apply trading mode specific parameter overrides"""
        profile = self.profiles.profiles[self.trading_mode]
        
        # Override thresholds
        self.thresholds.strong_buy_threshold = profile["strong_buy_threshold"]
        self.thresholds.buy_threshold = profile["buy_threshold"]
        
        # Override risk parameters
        self.risk.max_position_size = profile["max_position_size"]
        self.risk.stop_loss_threshold = profile["stop_loss_threshold"]
        
        # Override confidence
        self.thresholds.min_confidence_threshold = profile["min_confidence_threshold"]
        
    def update_market_regime(self, regime: MarketRegime):
        """Update market regime and adjust parameters accordingly"""
        self.current_regime = regime
        
        # Apply regime-specific threshold adjustments
        if regime in self.thresholds.regime_adjustments:
            adjustments = self.thresholds.regime_adjustments[regime]
            self.thresholds.strong_buy_threshold = adjustments.get("strong_buy_threshold", 
                                                                  self.thresholds.strong_buy_threshold)
            self.thresholds.buy_threshold = adjustments.get("buy_threshold", 
                                                           self.thresholds.buy_threshold)
        
        # Apply regime-specific factor weight adjustments
        if regime in self.factor_weights.regime_weight_adjustments:
            weight_adjustments = self.factor_weights.regime_weight_adjustments[regime]
            for factor, weight in weight_adjustments.items():
                setattr(self.factor_weights, factor, weight)
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current threshold values for logging/monitoring"""
        return {
            "strong_buy": self.thresholds.strong_buy_threshold,
            "buy": self.thresholds.buy_threshold,
            "hold": self.thresholds.hold_threshold,
            "sell": self.thresholds.sell_threshold,
            "min_confidence": self.thresholds.min_confidence_threshold
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration for logical consistency"""
        errors = []
        
        # Check threshold ordering
        if not (self.thresholds.strong_sell_threshold < 
                self.thresholds.sell_threshold < 
                self.thresholds.hold_threshold < 
                self.thresholds.buy_threshold < 
                self.thresholds.strong_buy_threshold):
            errors.append("Trading thresholds are not in correct order")
        
        # Check position size bounds
        if self.risk.max_position_size <= self.risk.min_position_size:
            errors.append("Max position size must be greater than min position size")
        
        # Check factor weights sum to ~1.0
        total_weight = (self.factor_weights.valuation_weight + 
                       self.factor_weights.momentum_weight + 
                       self.factor_weights.technical_weight + 
                       self.factor_weights.volume_weight + 
                       self.factor_weights.sentiment_weight)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Factor weights sum to {total_weight:.3f}, should be ~1.0")
        
        return errors

# Global configuration instance
ENHANCED_CONFIG = EnhancedTradingConfig()

def get_trading_config() -> EnhancedTradingConfig:
    """Get the global enhanced trading configuration"""
    return ENHANCED_CONFIG

def set_trading_mode(mode: TradingMode):
    """Change trading mode and update configuration"""
    global ENHANCED_CONFIG
    ENHANCED_CONFIG.trading_mode = mode
    ENHANCED_CONFIG.apply_trading_mode_profile()
    return ENHANCED_CONFIG