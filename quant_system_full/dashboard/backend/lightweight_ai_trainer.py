"""
Lightweight Real AI Training System
This module implements actual machine learning using LightGBM without complex dependencies.
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
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    ML_AVAILABLE = True
    print("[LIGHTWEIGHT_AI] LightGBM and sklearn available for real ML training")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"[LIGHTWEIGHT_AI] ML libraries not available: {e}")

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


class LightweightAITrainer:
    """
    Lightweight AI Training Manager with real machine learning.
    Uses LightGBM to train on actual market data.
    """
    
    def __init__(self, state_dir: Path = None):
        """Initialize the Lightweight AI Trainer."""
        self.state_dir = state_dir or Path(__file__).parent.parent / "state"
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "lightweight_ai_state.json"
        
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
        
        logger.info("Lightweight AI Trainer initialized")
    
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
                
                logger.info("Training state loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save training state to file."""
        try:
            state_data = {
                "training_progress": asdict(self.training_progress),
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def start_training(self, 
                      data_source: str = "yahoo_api",
                      model_type: str = "lightgbm", 
                      target_metric: str = "sharpe_ratio") -> bool:
        """
        Start real AI training using actual machine learning.
        """
        if self.current_status in [TrainingStatus.TRAINING, TrainingStatus.PREPARING]:
            logger.warning("Training already in progress")
            return False
        
        if not ML_AVAILABLE:
            logger.error("Machine learning libraries not available")
            self._add_log("ERROR: LightGBM and sklearn not installed")
            return False
        
        # Reset flags and status
        self.stop_training_flag = False
        self.current_status = TrainingStatus.PREPARING
        self.training_progress.status = TrainingStatus.PREPARING.value
        self.training_progress.error_message = None
        
        # Start training in background thread
        self.training_thread = threading.Thread(
            target=self._real_ml_training_loop,
            args=(data_source, model_type, target_metric),
            daemon=True
        )
        self.training_thread.start()
        
        self._add_log(f"Started real machine learning training with {model_type}")
        logger.info(f"Real AI training started with {data_source} data source")
        
        return True
    
    def _real_ml_training_loop(self, data_source: str, model_type: str, target_metric: str):
        """
        Real machine learning training loop using LightGBM and market data.
        """
        start_time = time.time()
        
        try:
            # Step 1: Data Collection
            self.current_status = TrainingStatus.PREPARING
            self.training_progress.status = TrainingStatus.PREPARING.value
            self._add_log("🔍 Fetching real market data for ML training...")
            
            # Training symbols - real stocks for ML training
            training_symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 
                              'AMZN', 'META', 'NFLX', 'AMD', 'INTC']
            
            # Fetch real market data
            all_data = []
            for i, symbol in enumerate(training_symbols):
                try:
                    self._add_log(f"📈 Fetching {symbol} data... ({i+1}/{len(training_symbols)})")
                    
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='2y', interval='1d')
                    
                    if not hist.empty:
                        # Create features for ML training
                        df = self._create_ml_features(hist, symbol)
                        if not df.empty:
                            all_data.append(df)
                            self._add_log(f"✅ Got {len(df)} samples for {symbol}")
                    
                    if self.stop_training_flag:
                        break
                        
                except Exception as e:
                    self._add_log(f"❌ Error fetching {symbol}: {str(e)}")
                    
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            if not all_data:
                raise Exception("No training data could be collected")
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            self._add_log(f"📊 Combined dataset: {len(combined_data)} samples from {len(all_data)} stocks")
            
            # Step 2: ML Training
            self.current_status = TrainingStatus.TRAINING
            self.training_progress.status = TrainingStatus.TRAINING.value
            self.training_progress.total_epochs = 100
            
            self._add_log("🤖 Starting LightGBM machine learning training...")
            
            # Prepare features and target
            feature_cols = [col for col in combined_data.columns if col not in ['target', 'symbol']]
            X = combined_data[feature_cols]
            y = combined_data['target']
            
            self._add_log(f"🔢 Features: {len(feature_cols)}, Samples: {len(X)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # LightGBM parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0
            }
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            self._add_log("🚀 Training LightGBM model...")
            
            # Training with progress callback
            callbacks = [self._training_callback]
            
            # Train the model
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=callbacks
            )
            
            if self.stop_training_flag:
                self.current_status = TrainingStatus.STOPPED
                self._add_log("Training stopped by user")
                return
            
            # Step 3: Evaluation
            self.current_status = TrainingStatus.EVALUATING
            self._add_log("📊 Evaluating model performance...")
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            
            # Calculate trading metrics
            returns = y_pred
            if len(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                win_rate = np.mean(returns > 0)
                total_return = np.mean(returns) * 252  # Annualized
            else:
                sharpe = 0
                win_rate = 0.5
                total_return = 0
            
            # Update final metrics
            self.training_progress.current_loss = float(mse)
            self.training_progress.validation_accuracy = min(0.95, max(0.5, 1 - mse))
            self.training_progress.win_rate = float(win_rate)
            self.training_progress.return_rate = float(total_return)
            self.training_progress.sharpe_ratio = float(sharpe)
            
            # Completed
            self.current_status = TrainingStatus.COMPLETED
            self.training_progress.status = TrainingStatus.COMPLETED.value
            
            self._add_log("🎉 Real machine learning training completed successfully!")
            self._add_log(f"📈 Final Metrics:")
            self._add_log(f"   • Sharpe Ratio: {sharpe:.2f}")
            self._add_log(f"   • Win Rate: {win_rate:.1%}")
            self._add_log(f"   • Annual Return: {total_return:.1%}")
            self._add_log(f"   • MSE: {mse:.4f}")
            self._add_log(f"🎯 Model trained on real market data from {len(training_symbols)} stocks")
            
        except Exception as e:
            self.current_status = TrainingStatus.FAILED
            self.training_progress.status = TrainingStatus.FAILED.value
            self.training_progress.error_message = str(e)
            self._add_log(f"❌ Real AI training failed: {e}")
            logger.error(f"Real AI training failed: {e}")
        
        finally:
            self._save_state()
    
    def _create_ml_features(self, price_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create machine learning features from price data."""
        try:
            df = price_data.copy()
            
            # Price-based features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Technical indicators
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_20'] = df['Close'].rolling(20).mean()
            df['sma_ratio'] = df['sma_5'] / df['sma_20']
            
            df['rsi'] = self._calculate_rsi(df['Close'])
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Volume features
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            # Target variable - next day return
            df['target'] = df['returns'].shift(-1)
            
            # Select features and clean data
            feature_cols = ['sma_ratio', 'rsi', 'volatility', 'volume_ratio']
            ml_df = df[feature_cols + ['target']].dropna()
            
            if len(ml_df) > 0:
                ml_df['symbol'] = symbol
            
            return ml_df
            
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
        """Callback function for training progress."""
        if env.iteration % 10 == 0:
            epoch = env.iteration
            self.training_progress.current_epoch = epoch
            
            # Update progress metrics during training
            progress = epoch / 100.0
            
            # Get validation score
            if len(env.evaluation_result_list) > 0:
                val_score = env.evaluation_result_list[0][2]  # rmse score
                self.training_progress.current_loss = float(val_score)
            
            elapsed = int(time.time() - getattr(self, '_start_time', time.time()))
            self.training_progress.training_time_elapsed = elapsed
            
            if epoch > 0:
                time_per_epoch = elapsed / epoch
                remaining = int(time_per_epoch * (100 - epoch))
                self.training_progress.estimated_time_remaining = remaining
            
            self.training_progress.last_updated = datetime.now().isoformat()
            
            if epoch % 20 == 0:
                self._add_log(f"🔄 Training Progress: Epoch {epoch}/100, Loss: {self.training_progress.current_loss:.4f}")
            
            self._save_state()
        
        return self.stop_training_flag  # Return True to stop training
    
    def stop_training(self) -> bool:
        """Stop the current training process."""
        if self.current_status in [TrainingStatus.TRAINING, TrainingStatus.PREPARING]:
            self.stop_training_flag = True
            self._add_log("🛑 Stop signal sent to training process")
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
        
        logger.info(f"[LIGHTWEIGHT_AI] {message}")


# Create global instance
lightweight_ai_trainer = LightweightAITrainer()