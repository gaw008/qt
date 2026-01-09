"""
Streaming Selection Strategy

This module implements a streaming approach to stock selection where stocks are
analyzed and filtered incrementally in batches, providing real-time results and
significant memory efficiency improvements.
"""

import pandas as pd
import numpy as np
import gc
import heapq
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging
import json
import os
import gzip
import shutil
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import performance monitoring
try:
    from ..performance_monitor import get_performance_monitor, start_performance_monitoring
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False

try:
    from .base_strategy import (
        BaseSelectionStrategy, SelectionResult, SelectionCriteria, 
        StrategyResults, SelectionAction
    )
except ImportError:
    from base_strategy import (
        BaseSelectionStrategy, SelectionResult, SelectionCriteria, 
        StrategyResults, SelectionAction
    )

logger = logging.getLogger(__name__)


@dataclass
class CandidateStock:
    """Represents a candidate stock in the streaming pipeline."""
    symbol: str
    score: float
    action: SelectionAction
    reasoning: str
    metrics: Dict[str, Any]
    timestamp: datetime
    confidence: float = 1.0
    
    def to_selection_result(self) -> SelectionResult:
        """Convert to SelectionResult format."""
        return SelectionResult(
            symbol=self.symbol,
            score=self.score,
            action=self.action,
            reasoning=self.reasoning,
            metrics=self.metrics,
            timestamp=self.timestamp,
            confidence=self.confidence
        )


@dataclass
class StreamingStats:
    """Statistics for the streaming selection process."""
    total_processed: int = 0
    batches_completed: int = 0
    total_batches: int = 0
    candidates_in_pool: int = 0
    current_min_threshold: float = 0.0
    avg_processing_time_per_batch: float = 0.0
    early_stopped: bool = False
    early_stop_reason: str = ""
    
    def get_progress_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_batches == 0:
            return 0.0
        return (self.batches_completed / self.total_batches) * 100


@dataclass
class StreamingCheckpoint:
    """Checkpoint data for streaming selection recovery."""
    strategy_name: str
    timestamp: datetime
    batch_index: int
    total_batches: int
    processed_symbols: List[str]
    candidate_pool: List[Dict[str, Any]]
    stats: Dict[str, Any]
    current_min_threshold: float
    best_score_seen: float
    batches_without_improvement: int
    universe_size: int
    criteria: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for JSON serialization."""
        return {
            'strategy_name': self.strategy_name,
            'timestamp': self.timestamp.isoformat(),
            'batch_index': self.batch_index,
            'total_batches': self.total_batches,
            'processed_symbols': self.processed_symbols,
            'candidate_pool': self.candidate_pool,
            'stats': self.stats,
            'current_min_threshold': self.current_min_threshold,
            'best_score_seen': self.best_score_seen,
            'batches_without_improvement': self.batches_without_improvement,
            'universe_size': self.universe_size,
            'criteria': self.criteria
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamingCheckpoint':
        """Create checkpoint from dictionary."""
        return cls(
            strategy_name=data['strategy_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            batch_index=data['batch_index'],
            total_batches=data['total_batches'],
            processed_symbols=data['processed_symbols'],
            candidate_pool=data['candidate_pool'],
            stats=data['stats'],
            current_min_threshold=data['current_min_threshold'],
            best_score_seen=data['best_score_seen'],
            batches_without_improvement=data['batches_without_improvement'],
            universe_size=data['universe_size'],
            criteria=data['criteria']
        )


class StreamingSelectionStrategy(BaseSelectionStrategy):
    """
    Streaming selection strategy that processes stocks incrementally,
    providing real-time results and memory efficiency.
    """
    
    def __init__(self, 
                 name: str = "StreamingSelection",
                 description: str = "Streaming incremental selection strategy",
                 candidate_pool_size: int = 100,
                 intermediate_save_interval: int = 5,
                 early_stop_patience: int = 10,
                 min_score_threshold: float = 60.0,
                 progress_callback = None):
        """
        Initialize streaming selection strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            candidate_pool_size: Maximum number of candidates to keep in memory
            intermediate_save_interval: Save results every N batches
            early_stop_patience: Stop if no improvements for N batches
            min_score_threshold: Minimum score to consider a stock
        """
        super().__init__(name, description)
        
        self.candidate_pool_size = candidate_pool_size
        self.intermediate_save_interval = intermediate_save_interval
        self.early_stop_patience = early_stop_patience
        self.min_score_threshold = min_score_threshold
        self.progress_callback = progress_callback
        
        # Streaming state
        self.candidate_pool: List[CandidateStock] = []
        self.stats = StreamingStats()
        self.batch_times: List[float] = []
        self.batches_without_improvement = 0
        self.best_score_seen = 0.0
        
        # Results storage path
        self.results_dir = "dashboard/state"
        self.intermediate_results_file = os.path.join(self.results_dir, "streaming_selection_progress.json")
        
        # Enhanced checkpoint functionality with compression and versioning
        self.enable_checkpoints = True
        self.checkpoint_interval = 3
        self.checkpoint_compression = True  # Enable compression for large-scale operations
        self.max_checkpoint_versions = 5   # Keep 5 versions for recovery
        
        # Checkpoint file paths
        self.checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{name}_checkpoint.json")
        self.compressed_checkpoint_file = os.path.join(self.checkpoint_dir, f"{name}_checkpoint.json.gz")
        
        # Error recovery tracking
        self.failed_symbols: List[str] = []
        self.error_counts: Dict[str, int] = {}
        self.max_symbol_retries = 3
        
        # Performance monitoring integration
        self.performance_monitor = None
        self.enable_performance_monitoring = True
        if PERFORMANCE_MONITOR_AVAILABLE and self.enable_performance_monitoring:
            self.performance_monitor = get_performance_monitor()
            # Start monitoring if processing large stock universe (>=1000 stocks)
            if candidate_pool_size >= 100:  # Enable for significant workloads
                self.performance_monitor.start_monitoring()
                logger.info("Performance monitoring enabled for large-scale processing")
        
        logger.info(f"Initialized streaming strategy: pool_size={candidate_pool_size}, "
                   f"save_interval={intermediate_save_interval}, early_stop={early_stop_patience}, "
                   f"performance_monitoring={'enabled' if self.performance_monitor else 'disabled'}")
    
    def select_stocks(self, 
                     universe: List[str], 
                     criteria: Optional[SelectionCriteria] = None) -> StrategyResults:
        """
        Select stocks using streaming approach with checkpoint recovery.
        
        Args:
            universe: List of stock symbols to evaluate
            criteria: Selection criteria and parameters
            
        Returns:
            StrategyResults containing selected stocks and metadata
        """
        start_time = time.time()
        criteria = self.validate_criteria(criteria)
        
        # Try to load checkpoint first
        checkpoint = self.load_checkpoint()
        resume_from_batch = 0
        
        if checkpoint is not None:
            logger.info(f"Found checkpoint, attempting to resume from batch {checkpoint.batch_index}")
            if self.restore_from_checkpoint(checkpoint):
                # Filter universe to skip already processed symbols
                processed_count = checkpoint.batch_index * 100
                universe = universe[processed_count:]  # Skip processed symbols
                resume_from_batch = checkpoint.batch_index
                logger.info(f"Resumed from checkpoint, processing remaining {len(universe)} symbols")
            else:
                logger.warning("Failed to restore from checkpoint, starting fresh")
                self._reset_streaming_state(universe, criteria)
        else:
            # Reset streaming state for fresh start
            self._reset_streaming_state(universe, criteria)
        
        logger.info(f"Starting streaming selection on {len(universe)} stocks")
        logger.info(f"Configuration: pool_size={self.candidate_pool_size}, "
                   f"threshold={self.min_score_threshold}, checkpoints={self.enable_checkpoints}")
        
        # Process stocks in batches
        errors = []
        try:
            # Use batch processing from base class but with streaming logic
            filtered_universe = self.filter_universe_batch(universe, criteria)
            errors.extend(self._process_stocks_streaming(filtered_universe, criteria, resume_from_batch))
            
        except Exception as e:
            error_msg = f"Error in streaming selection: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Finalize results
        execution_time = time.time() - start_time
        selected_stocks = self._finalize_selection(criteria.max_stocks)
        
        # Save final results and clear checkpoint
        self._save_final_results(selected_stocks, execution_time)
        self.clear_checkpoint()  # Remove checkpoint after successful completion
        
        # Include recovery statistics in results
        recovery_stats = self.get_recovery_stats()
        
        results = StrategyResults(
            strategy_name=self.name,
            selected_stocks=selected_stocks,
            total_candidates=self.stats.total_processed,
            execution_time=execution_time,
            criteria_used=criteria,
            errors=errors,
            metadata={
                'streaming_stats': {
                    'batches_completed': self.stats.batches_completed,
                    'total_batches': self.stats.total_batches,
                    'candidates_in_pool': self.stats.candidates_in_pool,
                    'current_min_threshold': self.stats.current_min_threshold,
                    'early_stopped': self.stats.early_stopped,
                    'early_stop_reason': self.stats.early_stop_reason,
                    'avg_batch_time': self.stats.avg_processing_time_per_batch,
                    'resumed_from_checkpoint': checkpoint is not None,
                    'resume_batch': resume_from_batch
                },
                'recovery_stats': recovery_stats
            }
        )
        
        logger.info(f"Streaming selection completed: {len(selected_stocks)} stocks selected "
                   f"in {execution_time:.2f}s ({self.stats.batches_completed} batches)")
        if recovery_stats['failed_symbols_count'] > 0:
            logger.info(f"Recovery stats: {recovery_stats['failed_symbols_count']} failed, "
                       f"{recovery_stats['skipped_symbols_count']} skipped")
        return results
    
    def _reset_streaming_state(self, universe: List[str], criteria: SelectionCriteria):
        """Reset streaming state for new selection."""
        self.candidate_pool = []
        self.stats = StreamingStats()
        self.batch_times = []
        self.batches_without_improvement = 0
        self.best_score_seen = 0.0
        
        # Calculate dynamic batch configuration
        batch_size = int(os.getenv('BATCH_SIZE', '100'))  # Use env configuration
        self.current_batch_size = batch_size
        self.stats.total_batches = (len(universe) + batch_size - 1) // batch_size
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _process_stocks_streaming(self, universe: List[str], criteria: SelectionCriteria, resume_from_batch: int = 0) -> List[str]:
        """Process stocks using streaming approach with memory management."""
        errors = []
        batch_size = self.current_batch_size  # Use dynamic batch size
        
        for batch_idx in range(0, len(universe), batch_size):
            batch_start_time = time.time()
            batch_symbols = universe[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1 + resume_from_batch
            
            logger.info(f"Processing streaming batch {batch_num}/{self.stats.total_batches}: "
                       f"{len(batch_symbols)} symbols")
            
            # Send progress update via callback
            if self.progress_callback:
                progress_percentage = min(100.0, (self.stats.total_processed / len(universe)) * 100)
                estimated_time_remaining = self._calculate_eta(batch_start_time, batch_num, self.stats.total_batches)
                
                top_candidates = [
                    {"symbol": c.symbol, "score": c.score} 
                    for c in self._get_sorted_candidates()[:5]
                ]
                
                try:
                    self.progress_callback({
                        "status": "processing",
                        "total_stocks": len(universe),
                        "processed_stocks": self.stats.total_processed,
                        "current_batch": batch_num,
                        "total_batches": self.stats.total_batches,
                        "strategy": self.name,
                        "candidate_pool_size": len(self.candidate_pool),
                        "top_candidates": top_candidates,
                        "eta_seconds": estimated_time_remaining,
                        "progress_percentage": progress_percentage
                    })
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
            
            # Filter out symbols that should be skipped due to repeated failures
            filtered_symbols = [symbol for symbol in batch_symbols if not self.should_skip_symbol(symbol)]
            if len(filtered_symbols) < len(batch_symbols):
                skipped_count = len(batch_symbols) - len(filtered_symbols)
                logger.info(f"Skipping {skipped_count} symbols due to repeated failures")
            
            try:
                # Process current batch
                batch_candidates = self._process_batch_streaming(filtered_symbols)
                
                # Update candidate pool
                self._update_candidate_pool(batch_candidates)
                
                # Update statistics
                batch_time = time.time() - batch_start_time
                self.batch_times.append(batch_time)
                self.stats.batches_completed = batch_num
                self.stats.total_processed += len(batch_symbols)  # Count all symbols, even skipped ones
                self.stats.avg_processing_time_per_batch = sum(self.batch_times) / len(self.batch_times)
                self.stats.candidates_in_pool = len(self.candidate_pool)
                
                # Record performance metrics
                if self.performance_monitor:
                    processing_time_ms = batch_time * 1000
                    avg_time_per_stock = processing_time_ms / len(batch_symbols)
                    for _ in batch_symbols:
                        self.performance_monitor.record_stock_processed(avg_time_per_stock)
                
                # Save checkpoint if enabled
                if self.enable_checkpoints and batch_num % self.checkpoint_interval == 0:
                    full_universe = universe  # This may need to be passed differently for proper checkpoint
                    if self.save_checkpoint(batch_num, full_universe, criteria):
                        logger.debug(f"Checkpoint saved at batch {batch_num}")
                
                # Save intermediate results if needed
                if batch_num % self.intermediate_save_interval == 0:
                    self._save_intermediate_results(batch_num)
                
                # Check for early stopping
                if self._should_early_stop(batch_candidates):
                    self.stats.early_stopped = True
                    self.stats.early_stop_reason = f"No improvement for {self.early_stop_patience} batches"
                    logger.info(f"Early stopping triggered after batch {batch_num}")
                    # Save final checkpoint before stopping
                    if self.enable_checkpoints:
                        self.save_checkpoint(batch_num, universe, criteria)
                    break
                
                # Memory management and cleanup
                self._perform_memory_management(batch_num)
                
                logger.info(f"Batch {batch_num} completed in {batch_time:.2f}s, "
                           f"pool size: {len(self.candidate_pool)}, "
                           f"min threshold: {self.stats.current_min_threshold:.1f}")
                
            except Exception as e:
                error_msg = f"Error processing batch {batch_num}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
                
                # Record error for problematic symbols
                for symbol in batch_symbols:
                    self.record_symbol_error(symbol, str(e))
        
        return errors
    
    def _process_batch_streaming(self, symbols: List[str]) -> List[CandidateStock]:
        """Process a single batch of symbols for streaming."""
        candidates = []
        
        for symbol in symbols:
            try:
                # Get stock data
                data = self.get_stock_data(symbol)
                if data is None:
                    continue
                
                # Calculate score using the strategy's scoring logic
                score = self.calculate_score(symbol, data)
                
                # Only consider stocks above minimum threshold
                if score >= self.min_score_threshold:
                    # Note: Action determination now happens after cross-sectional z-score normalization
                    # This is just a placeholder - actual trading decisions use z-score thresholds
                    if score >= self.min_score_threshold:
                        action = SelectionAction.BUY  # Placeholder - actual action determined by z-score
                        reasoning = f"Candidate stock for z-score evaluation (raw_score: {score:.1f})"
                    else:
                        action = SelectionAction.HOLD
                        reasoning = f"Below minimum threshold (raw_score: {score:.1f})"
                    
                    candidate = CandidateStock(
                        symbol=symbol,
                        score=score,
                        action=action,
                        reasoning=reasoning,
                        metrics={"score": score, "data_points": len(data.get('price_history', []))},
                        timestamp=datetime.now()
                    )
                    candidates.append(candidate)
                
            except Exception as e:
                logger.warning(f"Error processing {symbol} in streaming batch: {e}")
        
        return candidates
    
    def _update_candidate_pool(self, new_candidates: List[CandidateStock]):
        """Update the candidate pool with new candidates using heapq for efficiency."""
        if not new_candidates:
            self.batches_without_improvement += 1
            return
        
        # Convert candidate pool to min-heap format if not already done
        if not hasattr(self, '_heap_initialized'):
            self._init_heap_pool()
            self._heap_initialized = True
        
        # Process each new candidate
        for candidate in new_candidates:
            # If pool is not full, add candidate
            if len(self.candidate_pool) < self.candidate_pool_size:
                # Use negative score for max-heap behavior with heapq (min-heap)
                heapq.heappush(self.candidate_pool, (-candidate.score, candidate))
            else:
                # Pool is full, check if new candidate is better than worst
                if self.candidate_pool and -self.candidate_pool[0][0] < candidate.score:
                    # Remove worst candidate and add new one
                    heapq.heappop(self.candidate_pool)
                    heapq.heappush(self.candidate_pool, (-candidate.score, candidate))
        
        # Update threshold and improvement tracking
        if self.candidate_pool:
            # Min score is the top of min-heap (worst in our top-k)
            new_threshold = -self.candidate_pool[0][0]
            self.stats.current_min_threshold = new_threshold
            
            # Find best score for improvement tracking
            best_current_score = max(-score for score, _ in self.candidate_pool)
            if best_current_score > self.best_score_seen:
                self.best_score_seen = best_current_score
                self.batches_without_improvement = 0
            else:
                self.batches_without_improvement += 1
            
            # Update minimum threshold for future batches
            self.min_score_threshold = max(self.min_score_threshold, new_threshold * 0.8)
    
    def _init_heap_pool(self):
        """Initialize candidate pool as a min-heap for efficient TopK operations."""
        if not self.candidate_pool:
            return
            
        # Convert existing list to heap format: (negative_score, candidate)
        heap_items = [(-candidate.score, candidate) for candidate in self.candidate_pool]
        heapq.heapify(heap_items)
        self.candidate_pool = heap_items
    
    def _get_sorted_candidates(self) -> List[CandidateStock]:
        """Get candidates sorted by score (highest first) from heap format."""
        if not self.candidate_pool:
            return []
            
        # Extract candidates and sort by score (descending)
        candidates = [candidate for _, candidate in self.candidate_pool]
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates
    
    def _should_early_stop(self, batch_candidates: List[CandidateStock]) -> bool:
        """Determine if early stopping should be triggered."""
        if self.early_stop_patience <= 0:
            return False
        
        # Check if we have enough batches without improvement
        return self.batches_without_improvement >= self.early_stop_patience
    
    def _save_intermediate_results(self, batch_num: int):
        """Save intermediate results to file."""
        try:
            current_top_20 = self._get_sorted_candidates()[:20]
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "batch_completed": batch_num,
                "total_batches": self.stats.total_batches,
                "progress_percentage": self.stats.get_progress_percentage(),
                "stocks_processed": self.stats.total_processed,
                "candidates_in_pool": len(self.candidate_pool),
                "min_threshold": self.stats.current_min_threshold,
                "avg_batch_time": self.stats.avg_processing_time_per_batch,
                "top_20_current": [
                    {
                        "symbol": c.symbol,
                        "score": c.score,
                        "action": c.action.value,
                        "reasoning": c.reasoning
                    } for c in current_top_20
                ],
                "stats": {
                    "batches_without_improvement": self.batches_without_improvement,
                    "best_score_seen": self.best_score_seen,
                    "early_stop_risk": self.batches_without_improvement / max(1, self.early_stop_patience)
                }
            }
            
            with open(self.intermediate_results_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved intermediate results: {len(current_top_20)} top stocks, "
                       f"{self.stats.get_progress_percentage():.1f}% complete")
                       
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")
    
    def _finalize_selection(self, max_stocks: int) -> List[SelectionResult]:
        """Finalize the selection from candidate pool."""
        if not self.candidate_pool:
            return []
        
        # Get top stocks up to max_stocks limit
        top_candidates = self._get_sorted_candidates()[:min(max_stocks, len(self.candidate_pool))]
        
        # Convert to SelectionResult objects
        return [candidate.to_selection_result() for candidate in top_candidates]
    
    def _save_final_results(self, selected_stocks: List[SelectionResult], execution_time: float):
        """Save final results to the standard location."""
        try:
            final_results_file = os.path.join(self.results_dir, "streaming_selection_final.json")
            
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "strategy_name": self.name,
                "execution_time": execution_time,
                "total_processed": self.stats.total_processed,
                "batches_completed": self.stats.batches_completed,
                "early_stopped": self.stats.early_stopped,
                "early_stop_reason": self.stats.early_stop_reason,
                "selected_stocks": [stock.to_dict() for stock in selected_stocks],
                "performance_metrics": {
                    "avg_batch_time": self.stats.avg_processing_time_per_batch,
                    "total_candidates_evaluated": self.stats.candidates_in_pool,
                    "memory_efficiency_ratio": self.candidate_pool_size / max(1, self.stats.total_processed),
                    "selection_efficiency": len(selected_stocks) / max(1, self.stats.total_processed)
                }
            }
            
            with open(final_results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved final streaming results to {final_results_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save final results: {e}")
    
    def calculate_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Default scoring implementation for streaming strategy.
        This can be overridden by specific strategy implementations.
        
        Args:
            symbol: Stock symbol
            data: Stock data dictionary
            
        Returns:
            Score (0-100, higher is better)
        """
        try:
            fundamentals = data.get('fundamentals', {})
            price_history = data.get('price_history')
            
            if price_history is None or len(price_history) < 20:
                return 0.0
            
            # Simple scoring based on basic metrics
            score = 50.0  # Base score
            
            # Price momentum (20-day return)
            current_price = price_history['close'].iloc[-1]
            if len(price_history) >= 20:
                price_20d = price_history['close'].iloc[-20]
                momentum = (current_price - price_20d) / price_20d
                score += momentum * 50  # Scale momentum impact
            
            # Volume trend (recent vs average)
            recent_volume = price_history['volume'].iloc[-5:].mean()
            avg_volume = price_history['volume'].mean()
            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                score += (volume_ratio - 1) * 20  # Reward higher recent volume
            
            # Market cap consideration
            market_cap = fundamentals.get('market_cap', 0)
            if market_cap > 1e9:  # Prefer larger companies
                score += 10
            
            # Ensure score is within bounds
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating score for {symbol}: {e}")
            return 0.0
    
    # =================== CHECKPOINT FUNCTIONALITY ===================
    
    def _rotate_checkpoint_versions(self):
        """
        Rotate existing checkpoint versions to maintain version history.
        Keeps up to max_checkpoint_versions versions for recovery.
        """
        try:
            # Check for compressed checkpoint first
            current_checkpoint = self.compressed_checkpoint_file if self.checkpoint_compression else self.checkpoint_file
            
            if os.path.exists(current_checkpoint):
                # Rotate existing versions
                for i in range(self.max_checkpoint_versions - 1, 0, -1):
                    old_file = f"{current_checkpoint}.v{i}"
                    new_file = f"{current_checkpoint}.v{i+1}"
                    
                    if os.path.exists(old_file):
                        if i == self.max_checkpoint_versions - 1:
                            # Remove oldest version
                            os.remove(old_file)
                        else:
                            shutil.move(old_file, new_file)
                
                # Move current checkpoint to v1
                version_file = f"{current_checkpoint}.v1"
                shutil.move(current_checkpoint, version_file)
                
                logger.debug(f"Rotated checkpoint versions, keeping {self.max_checkpoint_versions} versions")
                
        except Exception as e:
            logger.warning(f"Failed to rotate checkpoint versions: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Clean up checkpoint files older than 7 days."""
        try:
            if not os.path.exists(self.checkpoint_dir):
                return
                
            cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
            
            for filename in os.listdir(self.checkpoint_dir):
                filepath = os.path.join(self.checkpoint_dir, filename)
                if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up old checkpoint: {filename}")
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints."""
        try:
            checkpoint_info = {
                'compression_enabled': self.checkpoint_compression,
                'max_versions': self.max_checkpoint_versions,
                'checkpoint_dir': self.checkpoint_dir,
                'available_versions': []
            }
            
            if not os.path.exists(self.checkpoint_dir):
                return checkpoint_info
            
            # Check for main checkpoint
            main_file = self.compressed_checkpoint_file if self.checkpoint_compression else self.checkpoint_file
            if os.path.exists(main_file):
                stat = os.stat(main_file)
                checkpoint_info['available_versions'].append({
                    'version': 'current',
                    'file': main_file,
                    'size_bytes': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'compressed': self.checkpoint_compression
                })
            
            # Check for versioned checkpoints
            for i in range(1, self.max_checkpoint_versions + 1):
                version_file = f"{main_file}.v{i}"
                if os.path.exists(version_file):
                    stat = os.stat(version_file)
                    checkpoint_info['available_versions'].append({
                        'version': f'v{i}',
                        'file': version_file,
                        'size_bytes': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'compressed': self.checkpoint_compression
                    })
            
            return checkpoint_info
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            return {'error': str(e)}
    
    def save_checkpoint(self, 
                       batch_index: int, 
                       universe: List[str], 
                       criteria: SelectionCriteria) -> bool:
        """
        Save current progress to checkpoint file.
        
        Args:
            batch_index: Current batch index
            universe: Full stock universe
            criteria: Selection criteria
            
        Returns:
            bool: True if checkpoint saved successfully
        """
        if not self.enable_checkpoints:
            return False
            
        try:
            # Create checkpoint directory if it doesn't exist
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # Rotate existing checkpoint versions
            self._rotate_checkpoint_versions()
            
            # Convert candidate pool to serializable format
            candidate_pool_data = []
            sorted_candidates = self._get_sorted_candidates()
            for candidate in sorted_candidates:
                candidate_pool_data.append({
                    'symbol': candidate.symbol,
                    'score': candidate.score,
                    'action': candidate.action.value,
                    'reasoning': candidate.reasoning,
                    'metrics': candidate.metrics,
                    'timestamp': candidate.timestamp.isoformat(),
                    'confidence': candidate.confidence
                })
            
            # Create checkpoint data
            checkpoint = StreamingCheckpoint(
                strategy_name=self.name,
                timestamp=datetime.now(),
                batch_index=batch_index,
                total_batches=self.stats.total_batches,
                processed_symbols=universe[:batch_index * 100],  # Symbols processed so far
                candidate_pool=candidate_pool_data,
                stats={
                    'total_processed': self.stats.total_processed,
                    'batches_completed': self.stats.batches_completed,
                    'candidates_in_pool': self.stats.candidates_in_pool,
                    'current_min_threshold': self.stats.current_min_threshold,
                    'avg_processing_time_per_batch': self.stats.avg_processing_time_per_batch,
                    'early_stopped': self.stats.early_stopped,
                    'early_stop_reason': self.stats.early_stop_reason
                },
                current_min_threshold=self.stats.current_min_threshold,
                best_score_seen=self.best_score_seen,
                batches_without_improvement=self.batches_without_improvement,
                universe_size=len(universe),
                criteria={
                    'max_stocks': criteria.max_stocks,
                    'min_market_cap': criteria.min_market_cap,
                    'max_market_cap': criteria.max_market_cap,
                    'sectors': criteria.sectors,
                    'exclude_sectors': criteria.exclude_sectors
                }
            )
            
            checkpoint_data = checkpoint.to_dict()
            
            # Save checkpoint with optional compression
            if self.checkpoint_compression:
                # Save compressed version for 5000+ stock operations
                with gzip.open(self.compressed_checkpoint_file, 'wt', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, separators=(',', ':'))  # Compact format
                file_size = os.path.getsize(self.compressed_checkpoint_file)
                logger.info(f"Compressed checkpoint saved at batch {batch_index}/{self.stats.total_batches} "
                           f"({len(self.candidate_pool)} candidates, {file_size/1024:.1f}KB compressed)")
            else:
                # Save uncompressed version for smaller operations
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                file_size = os.path.getsize(self.checkpoint_file)
                logger.info(f"Checkpoint saved at batch {batch_index}/{self.stats.total_batches} "
                           f"({len(self.candidate_pool)} candidates, {file_size/1024:.1f}KB)")
            
            # Cleanup old checkpoints periodically
            if batch_index % 10 == 0:  # Every 10 batches
                self._cleanup_old_checkpoints()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self) -> Optional[StreamingCheckpoint]:
        """
        Load checkpoint from file if it exists and is valid.
        
        Returns:
            StreamingCheckpoint if found and valid, None otherwise
        """
        if not self.enable_checkpoints:
            return None
        
        # Check for compressed checkpoint first, then uncompressed
        checkpoint_file_to_load = None
        if self.checkpoint_compression and os.path.exists(self.compressed_checkpoint_file):
            checkpoint_file_to_load = self.compressed_checkpoint_file
        elif os.path.exists(self.checkpoint_file):
            checkpoint_file_to_load = self.checkpoint_file
        
        if checkpoint_file_to_load is None:
            return None
            
        try:
            # Load checkpoint data based on compression
            if checkpoint_file_to_load.endswith('.gz'):
                with gzip.open(checkpoint_file_to_load, 'rt', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                logger.debug(f"Loaded compressed checkpoint from {checkpoint_file_to_load}")
            else:
                with open(checkpoint_file_to_load, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                logger.debug(f"Loaded uncompressed checkpoint from {checkpoint_file_to_load}")
                
            checkpoint = StreamingCheckpoint.from_dict(checkpoint_data)
            
            # Validate checkpoint is recent (< 24 hours old)
            if (datetime.now() - checkpoint.timestamp).total_seconds() > 86400:
                logger.warning("Checkpoint is too old (>24h), ignoring")
                return None
                
            # Validate checkpoint is for the same strategy
            if checkpoint.strategy_name != self.name:
                logger.warning(f"Checkpoint is for different strategy ({checkpoint.strategy_name}), ignoring")
                return None
                
            logger.info(f"Loaded checkpoint from batch {checkpoint.batch_index}/{checkpoint.total_batches}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def restore_from_checkpoint(self, checkpoint: StreamingCheckpoint) -> bool:
        """
        Restore streaming state from checkpoint.
        
        Args:
            checkpoint: Checkpoint data to restore from
            
        Returns:
            bool: True if restored successfully
        """
        try:
            # Restore statistics
            self.stats.total_processed = checkpoint.stats['total_processed']
            self.stats.batches_completed = checkpoint.stats['batches_completed']
            self.stats.candidates_in_pool = checkpoint.stats['candidates_in_pool']
            self.stats.current_min_threshold = checkpoint.stats['current_min_threshold']
            self.stats.avg_processing_time_per_batch = checkpoint.stats['avg_processing_time_per_batch']
            self.stats.early_stopped = checkpoint.stats['early_stopped']
            self.stats.early_stop_reason = checkpoint.stats['early_stop_reason']
            self.stats.total_batches = checkpoint.total_batches
            
            # Restore streaming state
            self.best_score_seen = checkpoint.best_score_seen
            self.batches_without_improvement = checkpoint.batches_without_improvement
            
            # Restore candidate pool
            self.candidate_pool = []
            for candidate_data in checkpoint.candidate_pool:
                candidate = CandidateStock(
                    symbol=candidate_data['symbol'],
                    score=candidate_data['score'],
                    action=SelectionAction(candidate_data['action']),
                    reasoning=candidate_data['reasoning'],
                    metrics=candidate_data['metrics'],
                    timestamp=datetime.fromisoformat(candidate_data['timestamp']),
                    confidence=candidate_data['confidence']
                )
                self.candidate_pool.append(candidate)
            
            # Convert to heap format after restoring
            if self.candidate_pool:
                self._init_heap_pool()
                self._heap_initialized = True
                
            logger.info(f"Restored state from checkpoint: {len(self.candidate_pool)} candidates, "
                       f"batch {checkpoint.batch_index}/{checkpoint.total_batches}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return False
    
    def clear_checkpoint(self):
        """Remove checkpoint files and versions after successful completion."""
        try:
            files_cleared = 0
            
            # Clear main checkpoint files
            for checkpoint_file in [self.checkpoint_file, self.compressed_checkpoint_file]:
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                    files_cleared += 1
                
                # Clear versioned files
                for i in range(1, self.max_checkpoint_versions + 1):
                    version_file = f"{checkpoint_file}.v{i}"
                    if os.path.exists(version_file):
                        os.remove(version_file)
                        files_cleared += 1
            
            if files_cleared > 0:
                logger.info(f"Cleared {files_cleared} checkpoint files")
                
        except Exception as e:
            logger.warning(f"Failed to clear checkpoint files: {e}")
    
    def should_skip_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol should be skipped due to repeated failures.
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            bool: True if symbol should be skipped
        """
        return self.error_counts.get(symbol, 0) >= self.max_symbol_retries
    
    def record_symbol_error(self, symbol: str, error: str):
        """
        Record an error for a symbol and track failure count.
        
        Args:
            symbol: Stock symbol that failed
            error: Error message
        """
        self.error_counts[symbol] = self.error_counts.get(symbol, 0) + 1
        if symbol not in self.failed_symbols:
            self.failed_symbols.append(symbol)
            
        if self.error_counts[symbol] >= self.max_symbol_retries:
            logger.warning(f"Symbol {symbol} failed {self.max_symbol_retries} times, will skip")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get error recovery statistics."""
        return {
            'failed_symbols_count': len(self.failed_symbols),
            'skipped_symbols_count': len([s for s, count in self.error_counts.items() 
                                        if count >= self.max_symbol_retries]),
            'total_errors': sum(self.error_counts.values()),
            'error_rate': len(self.failed_symbols) / max(1, self.stats.total_processed)
        }
    
    def _perform_memory_management(self, batch_num: int) -> None:
        """
        Perform memory management and garbage collection.
        
        Args:
            batch_num: Current batch number
        """
        try:
            # Check if it's time for memory management (every 5 batches)
            if batch_num % 5 == 0:
                # Get memory usage if psutil is available
                if PSUTIL_AVAILABLE:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    memory_percent = process.memory_percent()
                    
                    logger.info(f"Memory usage: {memory_mb:.1f}MB ({memory_percent:.1f}%)")
                    
                    # If memory usage is high, perform aggressive cleanup
                    if memory_percent > 80.0:
                        logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                        
                        # Adjust batch size dynamically
                        if self.current_batch_size > 20:
                            self.current_batch_size = max(20, int(self.current_batch_size * 0.8))
                            logger.info(f"Reduced batch size to {self.current_batch_size}")
                        
                        # Force garbage collection
                        gc.collect()
                        
                        # Check memory after cleanup
                        memory_after = process.memory_percent()
                        logger.info(f"Memory after cleanup: {memory_after:.1f}%")
                else:
                    # Fallback: periodic garbage collection without monitoring
                    logger.debug(f"Performing periodic garbage collection (batch {batch_num})")
                
                # Always run garbage collection every 5 batches
                gc.collect()
                
                # Trim candidate pool if it's getting too large
                if len(self.candidate_pool) > self.candidate_pool_size * 1.5:
                    self._trim_candidate_pool()
                    logger.info(f"Trimmed candidate pool to {len(self.candidate_pool)} candidates")
                    
        except Exception as e:
            logger.warning(f"Error during memory management: {e}")
    
    def _trim_candidate_pool(self) -> None:
        """Trim candidate pool to target size, keeping highest scores."""
        if len(self.candidate_pool) <= self.candidate_pool_size:
            return
            
        try:
            # Get sorted candidates and keep only the top ones
            sorted_candidates = self._get_sorted_candidates()
            top_candidates = sorted_candidates[:self.candidate_pool_size]
            
            # Rebuild heap with top candidates
            self.candidate_pool = [(-c.score, c) for c in top_candidates]
            heapq.heapify(self.candidate_pool)
            
            # Update minimum threshold
            if self.candidate_pool:
                self.stats.current_min_threshold = -self.candidate_pool[0][0]
                
        except Exception as e:
            logger.warning(f"Error trimming candidate pool: {e}")

    def _calculate_eta(self, batch_start_time: float, current_batch: int, total_batches: int) -> Optional[int]:
        """Calculate estimated time remaining in seconds."""
        try:
            if current_batch <= 1 or len(self.batch_times) == 0:
                return None
            
            # Calculate average batch time
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            remaining_batches = total_batches - current_batch
            eta_seconds = int(avg_batch_time * remaining_batches)
            
            return eta_seconds
        except Exception:
            return None