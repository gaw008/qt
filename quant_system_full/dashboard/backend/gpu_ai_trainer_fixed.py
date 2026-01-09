"""
Fixed GPU AI Training Manager
Direct integration of GPU training pipeline with fixed imports.
"""

import json
import time
import logging
import threading
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

# Machine learning imports
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
    import optuna
    ML_AVAILABLE = True
    print("[GPU_AI] LightGBM, sklearn, and Optuna available for GPU training")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"[GPU_AI] ML libraries not available: {e}")

# GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[GPU_AI] GPU acceleration available (CUDA)")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("[GPU_AI] GPU acceleration not available, using CPU")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training status enumeration."""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TrainingProgress:
    """Container for training progress data."""
    status: str
    current_epoch: int
    total_epochs: int
    current_loss: float
    validation_accuracy: float
    win_rate: float
    return_rate: float
    sharpe_ratio: float
    training_time_elapsed: int  # seconds
    estimated_time_remaining: int  # seconds
    last_updated: str
    error_message: Optional[str] = None


class GPUAITrainerFixed:
    """
    Fixed GPU AI Training Manager with proper imports and GPU acceleration.
    """
    
    def __init__(self, state_dir: Path = None):
        """Initialize the GPU AI Trainer."""
        self.state_dir = state_dir or Path(__file__).parent.parent / "state"
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "gpu_ai_trainer_state.json"
        self.models_dir = self.state_dir / "gpu_models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Create data storage directories
        self.data_dir = Path(__file__).parent.parent.parent / "data_cache"
        self.data_dir.mkdir(exist_ok=True)
        self.market_data_dir = self.data_dir / "market_data"
        self.market_data_dir.mkdir(exist_ok=True)
        self.features_data_dir = self.data_dir / "features_data"
        self.features_data_dir.mkdir(exist_ok=True)
        
        # GPU configuration
        self.use_gpu = GPU_AVAILABLE
        if self.use_gpu:
            self._configure_gpu()
        
        # Training state
        self.current_status = TrainingStatus.IDLE
        self.training_thread = None
        self.stop_training_flag = False
        self.model = None
        
        # Initialize default values
        self.training_progress = TrainingProgress(
            status=TrainingStatus.IDLE.value,
            current_epoch=0,
            total_epochs=0,
            current_loss=0.0,
            validation_accuracy=0.0,
            win_rate=0.0,
            return_rate=0.0,
            sharpe_ratio=0.0,
            training_time_elapsed=0,
            estimated_time_remaining=0,
            last_updated=datetime.now().isoformat()
        )
        
        self.training_logs = []
        
        # Load saved state
        self._load_state()
        
        logger.info(f"GPU AI Trainer initialized. GPU: {'Available' if self.use_gpu else 'Not Available'}")
        logger.info(f"Data storage directories:")
        logger.info(f"  - Raw market data: {self.market_data_dir}")
        logger.info(f"  - Processed features: {self.features_data_dir}")
        logger.info(f"  - Trained models: {self.models_dir}")
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance."""
        try:
            if GPU_AVAILABLE:
                # Check GPU memory
                meminfo = cp.cuda.MemoryInfo()
                total_memory = meminfo.total / 1024**3  # GB
                
                logger.info(f"GPU detected: {total_memory:.1f} GB memory available")
                
                # Set optimal GPU parameters for LightGBM
                self.gpu_params = {
                    'device_type': 'gpu',
                    'gpu_use_dp': True,
                    'max_bin': 255,
                    'num_leaves': min(255, 2**8)  # Optimize for GPU memory
                }
                
        except Exception as e:
            logger.warning(f"GPU configuration failed: {e}. Falling back to CPU.")
            self.use_gpu = False
    
    def _load_state(self):
        """Load training state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                if 'training_progress' in data:
                    prog_data = data['training_progress']
                    self.training_progress = TrainingProgress(**prog_data)
                    self.current_status = TrainingStatus(self.training_progress.status)
                
                logger.info("GPU training state loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load GPU training state: {e}")
    
    def _save_state(self):
        """Save training state to file."""
        try:
            state_data = {
                "training_progress": asdict(self.training_progress),
                "use_gpu": self.use_gpu,
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save GPU training state: {e}")
    
    def start_training(self, 
                      data_source: str = "yahoo_api",
                      model_type: str = "lightgbm", 
                      target_metric: str = "sharpe_ratio") -> bool:
        """
        Start GPU-accelerated AI training using real market data.
        """
        if self.current_status in [TrainingStatus.TRAINING, TrainingStatus.PREPARING]:
            logger.warning("GPU training already in progress")
            return False
        
        if not ML_AVAILABLE:
            logger.error("Machine learning libraries not available")
            self._add_log("ERROR: LightGBM and required ML libraries not installed")
            return False
        
        # Reset flags and status
        self.stop_training_flag = False
        self.current_status = TrainingStatus.PREPARING
        self.training_progress.status = TrainingStatus.PREPARING.value
        self.training_progress.error_message = None
        
        # Start training in background thread
        self.training_thread = threading.Thread(
            target=self._gpu_training_loop,
            args=(data_source, model_type, target_metric),
            daemon=True
        )
        self.training_thread.start()
        
        device_info = "GPU-accelerated" if self.use_gpu else "CPU-based"
        self._add_log(f"Started {device_info} AI training with {model_type}")
        logger.info(f"GPU AI training started with {data_source} data source")
        
        return True
    
    def _gpu_training_loop(self, data_source: str, model_type: str, target_metric: str):
        """
        GPU-accelerated training loop using LightGBM and real market data.
        """
        start_time = time.time()
        
        try:
            # Step 1: Data Collection
            self.current_status = TrainingStatus.PREPARING
            self.training_progress.status = TrainingStatus.PREPARING.value
            self._add_log("Collecting real market data for GPU training...")
            
            # Extended stock universe for comprehensive training
            training_symbols = [
                'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC',
                'CRM', 'ADBE', 'ORCL', 'IBM', 'CSCO', 'QCOM', 'BABA', 'TSM', 'ASML', 'SHOP'
            ]
            
            # Fetch comprehensive market data with persistent storage
            all_training_data = []
            raw_data_cache = {}
            
            for i, symbol in enumerate(training_symbols):
                try:
                    self._add_log(f"Fetching {symbol} data... ({i+1}/{len(training_symbols)})")
                    
                    # Check if we have cached data first
                    raw_data_file = self.market_data_dir / f"{symbol}_3y_raw.parquet"
                    
                    if raw_data_file.exists():
                        self._add_log(f"Loading cached data for {symbol}")
                        hist = pd.read_parquet(raw_data_file)
                        hist.index = pd.to_datetime(hist.index)
                    else:
                        # Fetch fresh data from Yahoo Finance
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period='3y', interval='1d')
                        
                        if not hist.empty:
                            # Save raw market data to cache
                            hist.to_parquet(raw_data_file)
                            self._add_log(f"Saved raw data for {symbol} ({len(hist)} records)")
                    
                    if not hist.empty and len(hist) > 100:
                        # Store raw data for reference
                        raw_data_cache[symbol] = hist.copy()
                        
                        # Create comprehensive features
                        features_df = self._create_advanced_features(hist, symbol)
                        if not features_df.empty:
                            # Save processed features data
                            features_file = self.features_data_dir / f"{symbol}_features.parquet"
                            features_df.to_parquet(features_file)
                            
                            all_training_data.append(features_df)
                            self._add_log(f"Processed & saved {len(features_df)} feature samples for {symbol}")
                    
                    if self.stop_training_flag:
                        break
                        
                except Exception as e:
                    self._add_log(f"Warning: Could not fetch {symbol}: {str(e)}")
                    
                # Rate limiting
                time.sleep(0.3)
            
            # Save consolidated raw data summary
            if raw_data_cache:
                data_summary = {}
                for symbol, data in raw_data_cache.items():
                    data_summary[symbol] = {
                        "start_date": str(data.index[0].date()),
                        "end_date": str(data.index[-1].date()), 
                        "total_records": len(data),
                        "latest_price": float(data['Close'].iloc[-1]),
                        "data_file": f"{symbol}_3y_raw.parquet"
                    }
                
                summary_file = self.market_data_dir / "data_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(data_summary, f, indent=2)
                self._add_log(f"Saved market data summary for {len(data_summary)} symbols")
            
            if not all_training_data:
                raise Exception("No training data could be collected from any symbols")
            
            # Combine all training data
            combined_data = pd.concat(all_training_data, ignore_index=True)
            self._add_log(f"Combined training dataset: {len(combined_data):,} samples from {len(all_training_data)} stocks")
            
            # Save combined training dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_data_file = self.features_data_dir / f"combined_training_data_{timestamp}.parquet"
            combined_data.to_parquet(combined_data_file)
            self._add_log(f"Saved combined training dataset: {combined_data_file.name}")
            
            # Step 2: Feature Engineering and Preparation
            self._add_log("Preparing features for GPU training...")
            
            # Select features (exclude target and metadata columns)
            feature_cols = [col for col in combined_data.columns 
                          if col not in ['target', 'symbol', 'date'] and not col.startswith('_')]
            
            X = combined_data[feature_cols].fillna(0)  # Fill any remaining NaN values
            y = combined_data['target'].fillna(0)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            self._add_log(f"Features prepared: {len(feature_cols)} features, {len(X):,} samples")
            
            # Step 3: Train-Test Split (Time series aware)
            # Use time-series split to respect temporal ordering
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            self._add_log(f"Training: {len(X_train):,} samples, Testing: {len(X_test):,} samples")
            
            # Step 4: GPU-Accelerated Model Training
            self.current_status = TrainingStatus.TRAINING
            self.training_progress.status = TrainingStatus.TRAINING.value
            self.training_progress.total_epochs = 100
            
            device_type = "GPU" if self.use_gpu else "CPU"
            self._add_log(f"Starting LightGBM {device_type} training...")
            
            # Configure LightGBM parameters for GPU or CPU
            base_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 127,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            # Add GPU-specific parameters if available
            if self.use_gpu and GPU_AVAILABLE:
                base_params.update(self.gpu_params)
                self._add_log("GPU acceleration enabled for training")
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            # Training with callbacks
            callbacks = [
                lgb.early_stopping(stopping_rounds=20),
                self._training_callback
            ]
            
            # Train the model
            self._add_log("🔥 Training LightGBM model with advanced features...")
            
            self.model = lgb.train(
                base_params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=callbacks
            )
            
            if self.stop_training_flag:
                self.current_status = TrainingStatus.STOPPED
                self._add_log("🛑 Training stopped by user")
                return
            
            # Step 5: Model Evaluation
            self.current_status = TrainingStatus.EVALUATING
            self._add_log("📊 Evaluating trained model performance...")
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate comprehensive metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            # Calculate financial metrics
            test_returns = y_pred_test
            if len(test_returns) > 0:
                # Sharpe ratio (annualized)
                sharpe = np.mean(test_returns) / np.std(test_returns) * np.sqrt(252) if np.std(test_returns) > 0 else 0
                # Win rate
                win_rate = np.mean(test_returns > 0)
                # Total return (annualized)
                total_return = np.mean(test_returns) * 252
                # Validation accuracy (based on direction prediction)
                direction_accuracy = np.mean((test_returns > 0) == (y_test > 0))
            else:
                sharpe = 0
                win_rate = 0.5
                total_return = 0
                direction_accuracy = 0.5
            
            # Update final metrics
            self.training_progress.current_loss = float(test_mse)
            self.training_progress.validation_accuracy = float(direction_accuracy)
            self.training_progress.win_rate = float(win_rate)
            self.training_progress.return_rate = float(total_return)
            self.training_progress.sharpe_ratio = float(sharpe)
            
            # Save model
            model_path = self.models_dir / f"gpu_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            self.model.save_model(str(model_path))
            
            # Training completed successfully
            self.current_status = TrainingStatus.COMPLETED
            self.training_progress.status = TrainingStatus.COMPLETED.value
            self.training_progress.current_epoch = 100
            
            # Feature importance analysis
            feature_importance = self.model.feature_importance(importance_type='gain')
            top_features = sorted(zip(feature_cols, feature_importance), key=lambda x: x[1], reverse=True)[:10]
            
            self._add_log("GPU-accelerated AI training completed successfully!")
            self._add_log(f"Final Performance Metrics:")
            self._add_log(f"   • Sharpe Ratio: {sharpe:.3f}")
            self._add_log(f"   • Win Rate: {win_rate:.1%}")
            self._add_log(f"   • Annual Return: {total_return:.1%}")
            self._add_log(f"   • Direction Accuracy: {direction_accuracy:.1%}")
            self._add_log(f"   • Test MSE: {test_mse:.6f}")
            self._add_log(f"   • Train MSE: {train_mse:.6f}")
            self._add_log(f"Model trained on {len(combined_data):,} samples from {len(training_symbols)} stocks")
            self._add_log(f"Model saved to: {model_path.name}")
            
            # Log top features
            self._add_log("Top 5 Important Features:")
            for i, (feature, importance) in enumerate(top_features[:5]):
                self._add_log(f"   {i+1}. {feature}: {importance:.2f}")
            
        except Exception as e:
            self.current_status = TrainingStatus.FAILED
            self.training_progress.status = TrainingStatus.FAILED.value
            self.training_progress.error_message = str(e)
            self._add_log(f"GPU AI training failed: {e}")
            logger.error(f"GPU AI training failed: {e}")
        
        finally:
            self._save_state()
    
    def _create_advanced_features(self, price_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive technical and fundamental features."""
        try:
            df = price_data.copy()
            
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
            df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open']
            
            # Moving averages (multiple timeframes)
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['Close'].rolling(period).mean()
                df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
                df[f'sma_ratio_{period}'] = df['Close'] / df[f'sma_{period}']
                
            # Technical indicators
            df['rsi'] = self._calculate_rsi(df['Close'], 14)
            df['rsi_30'] = self._calculate_rsi(df['Close'], 30)
            
            # Bollinger Bands
            bb_period = 20
            bb_std = df['Close'].rolling(bb_period).std()
            bb_mean = df['Close'].rolling(bb_period).mean()
            df['bb_upper'] = bb_mean + (bb_std * 2)
            df['bb_lower'] = bb_mean - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mean
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Volatility features
            df['volatility_20'] = df['returns'].rolling(20).std()
            for period in [10, 30]:
                df[f'volatility_{period}'] = df['returns'].rolling(period).std()
                df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df['volatility_20']
            
            # Volume features
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            df['volume_price_trend'] = df['Volume'] * np.sign(df['returns'])
            
            # Price momentum features
            for period in [1, 3, 5, 10, 20]:
                df[f'momentum_{period}'] = df['Close'].pct_change(period)
                
            # Statistical features
            df['price_std_20'] = df['Close'].rolling(20).std()
            df['price_skew_20'] = df['returns'].rolling(20).skew()
            df['price_kurt_20'] = df['returns'].rolling(20).kurt()
            
            # Target variable - next day return
            df['target'] = df['returns'].shift(-1)
            
            # Select feature columns and clean data
            feature_cols = [col for col in df.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
            
            ml_df = df[feature_cols].copy()
            
            # Remove rows with NaN values
            ml_df = ml_df.dropna()
            
            if len(ml_df) > 50:  # Ensure sufficient data
                ml_df['symbol'] = symbol
                return ml_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self._add_log(f"Error creating features for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _training_callback(self, env):
        """Callback function for LightGBM training progress."""
        if hasattr(env, 'iteration'):
            epoch = env.iteration + 1
            self.training_progress.current_epoch = epoch
            
            # Update progress metrics during training
            if len(env.evaluation_result_list) > 0:
                # Get validation score (RMSE)
                val_score = env.evaluation_result_list[0][2]
                self.training_progress.current_loss = float(val_score)
            
            elapsed = int(time.time() - getattr(self, '_training_start_time', time.time()))
            self.training_progress.training_time_elapsed = elapsed
            
            if epoch > 0:
                time_per_epoch = elapsed / epoch
                remaining = int(time_per_epoch * (100 - epoch))
                self.training_progress.estimated_time_remaining = remaining
            
            self.training_progress.last_updated = datetime.now().isoformat()
            
            if epoch % 10 == 0:
                device = "GPU" if self.use_gpu else "CPU"
                self._add_log(f"🔄 {device} Training: Epoch {epoch}/100, Loss: {self.training_progress.current_loss:.4f}")
            
            self._save_state()
        
        return self.stop_training_flag
    
    def stop_training(self) -> bool:
        """Stop the current training process."""
        if self.current_status in [TrainingStatus.TRAINING, TrainingStatus.PREPARING]:
            self.stop_training_flag = True
            self._add_log("🛑 Stop signal sent to GPU training process")
            return True
        return False
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        return asdict(self.training_progress)
    
    def get_training_logs(self, limit: int = 100) -> List[str]:
        """Get recent training logs."""
        return self.training_logs[-limit:] if self.training_logs else []
    
    def _add_log(self, message: str):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.training_logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.training_logs) > 1000:
            self.training_logs = self.training_logs[-500:]
        
        logger.info(f"[GPU_AI] {message}")


# Create global instance
gpu_ai_trainer = GPUAITrainerFixed()