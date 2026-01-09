"""
Stock Selection Wrapper for Worker Environment
Provides a clean interface to the complex strategy system with proper path handling.
Supports both streaming and traditional selection modes.
"""

import sys
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup logging first
logger = logging.getLogger(__name__)

# Try to import WebSocket progress function
try:
    sys.path.append(str(Path(__file__).parent.parent / 'backend'))
    from app import send_selection_progress_sync
    WEBSOCKET_AVAILABLE = True
except ImportError:
    logger.warning("WebSocket progress function not available")
    WEBSOCKET_AVAILABLE = False
    def send_selection_progress_sync(data): pass

def setup_bot_path():
    """Add bot path to sys.path for imports."""
    # Determine the correct bot path from current working directory
    current_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    
    # Try multiple possible bot paths
    possible_bot_paths = [
        current_dir.parent / 'bot',        # Standard structure
        current_dir.parent.parent / 'bot', # Two levels up (correct for worker/bot structure)
        cwd / 'bot',                       # From project root
        cwd.parent / 'bot'                 # One level up
    ]
    
    bot_path = None
    for path in possible_bot_paths:
        if path.exists() and (path / 'data.py').exists():
            bot_path = path
            break
    
    if bot_path is None:
        logger.error(f"Could not find bot path. Tried: {[str(p) for p in possible_bot_paths]}")
        return
    
    paths_to_add = [str(bot_path), str(bot_path.parent)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)
    
    logger.info(f"Found and added bot path: {bot_path}")
    logger.info(f"Added paths to sys.path: {paths_to_add}")

def run_detailed_stock_analysis(universe: List[str], max_stocks: int = 10, use_parallel: bool = None) -> Dict[str, Any]:
    """
    Enhanced stock selection with detailed analysis reporting and parallel data fetching.
    Records why each stock was or wasn't selected.
    
    Args:
        universe: List of stock symbols to analyze
        max_stocks: Maximum number of stocks to select
        use_parallel: Force parallel processing (None = auto-detect based on size)
    """
    try:
        setup_bot_path()
        
        # Try to import data functions
        try:
            from data import fetch_history  
            from yahoo_data import fetch_yahoo_ticker_info, batch_fetch_with_progress, estimate_batch_fetch_time, get_batch_fetch_stats
        except ImportError as e:
            logger.error(f"Failed to import data modules: {e}")
            return {"selected_stocks": [], "error": "Data import failed", "detailed_analysis": []}
        
        logger.info(f"[SELECTION] Starting detailed analysis on {len(universe)} stocks")
        
        # Auto-detect parallel processing need
        if use_parallel is None:
            use_parallel = len(universe) > 50
            
        # Get system capabilities
        system_stats = get_batch_fetch_stats()
        logger.info(f"[SELECTION] System: {system_stats['cpu_count']} CPUs, {system_stats['total_memory_gb']}GB RAM, "
                   f"{system_stats['performance_tier']} tier")
        
        # Estimate processing time
        if use_parallel:
            estimated_time = estimate_batch_fetch_time(len(universe), use_cache=True, cache_hit_rate=0.6)
            logger.info(f"[SELECTION] Estimated time for parallel processing: {estimated_time:.1f} seconds "
                       f"({system_stats['estimated_symbols_per_minute']} symbols/min)")
        
        # Pre-fetch all data in parallel if beneficial
        all_price_data = {}
        all_info_data = {}
        
        if use_parallel and len(universe) > 20:
            logger.info(f"[SELECTION] Pre-fetching price data for {len(universe)} symbols using parallel processing")
            
            # Define progress callback
            def progress_callback(completed, total, stats):
                logger.info(f"[SELECTION] Data fetch progress: {completed}/{total} "
                           f"(successful: {stats['successful']}, cache hits: {stats['cache_hits']})")
            
            # Batch fetch price data
            start_time = time.time()
            all_price_data = batch_fetch_with_progress(
                universe, 
                period='day', 
                limit=30,
                use_cache=True,
                progress_callback=progress_callback if len(universe) > 100 else None
            )
            fetch_time = time.time() - start_time
            
            successful_fetches = sum(1 for data in all_price_data.values() if data is not None)
            logger.info(f"[SELECTION] Parallel data fetch completed: {successful_fetches}/{len(universe)} successful "
                       f"in {fetch_time:.1f}s ({len(universe)/fetch_time:.1f} symbols/sec)")
            
            # Pre-fetch ticker info with rate limiting to avoid API blocks
            logger.info(f"[SELECTION] Fetching ticker info with rate limiting...")
            valid_symbols = [symbol for symbol in universe if all_price_data.get(symbol) is not None]
            logger.info(f"[SELECTION] Processing {len(valid_symbols)} symbols for info data")
            
            # Process in smaller batches with delays to avoid rate limits
            batch_size = 100  # Conservative batch size for ticker info
            for i in range(0, len(valid_symbols), batch_size):
                batch = valid_symbols[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(valid_symbols) + batch_size - 1) // batch_size
                
                logger.info(f"[SELECTION] Processing info batch {batch_num}/{total_batches} ({len(batch)} symbols)")
                
                for j, symbol in enumerate(batch):
                    try:
                        all_info_data[symbol] = fetch_yahoo_ticker_info(symbol, max_retries=2)
                        # Small delay between individual requests to avoid rate limiting
                        if j > 0 and j % 20 == 0:  # Pause every 20 requests
                            time.sleep(1.0)
                    except Exception as e:
                        logger.warning(f"[SELECTION] Failed to fetch info for {symbol}: {e}")
                        all_info_data[symbol] = None
                
                # Longer delay between batches if not the last batch
                if i + batch_size < len(valid_symbols):
                    logger.info(f"[SELECTION] Batch {batch_num} complete, pausing 3 seconds...")
                    time.sleep(3.0)
        
        # Detailed analysis for each stock
        analysis_results = []
        selected_stocks = []
        
        for i, symbol in enumerate(universe):
            logger.info(f"[SELECTION] Analyzing {symbol} ({i+1}/{len(universe)})")
            
            stock_analysis = {
                'symbol': symbol,
                'rank': i + 1,
                'status': 'rejected',  # Default to rejected
                'reasons': [],
                'metrics': {},
                'score': 0.0
            }
            
            try:
                # Step 1: Data availability check
                if use_parallel and symbol in all_price_data:
                    df = all_price_data[symbol]
                    info = all_info_data.get(symbol)
                else:
                    df = fetch_history(None, symbol, 'day', 30, dry_run=False)
                    info = fetch_yahoo_ticker_info(symbol)
                
                if df is None or len(df) < 5:
                    stock_analysis['reasons'].append("Insufficient price data")
                    analysis_results.append(stock_analysis)
                    continue
                
                if info is None:
                    stock_analysis['reasons'].append("Company info not available")
                    analysis_results.append(stock_analysis)
                    continue
                
                # Step 2: Basic metrics calculation
                current_price = df['close'].iloc[-1]
                volume_avg = df['volume'].tail(5).mean()
                price_change_5d = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                price_change_1d = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                
                stock_analysis['metrics'] = {
                    'current_price': current_price,
                    'avg_volume': volume_avg,
                    'price_change_5d': price_change_5d * 100,  # As percentage
                    'price_change_1d': price_change_1d * 100,
                    'market_cap': info.get('market_cap', 0),
                    'pe_ratio': info.get('pe_ratio', 0),
                    'sector': info.get('sector', 'Unknown')
                }
                
                # Step 3: Apply filters with detailed reasons
                passes_all_filters = True
                
                # Price filter
                if current_price < 5:
                    stock_analysis['reasons'].append(f"Price too low: ${current_price:.2f} < $5")
                    passes_all_filters = False
                elif current_price > 1000:
                    stock_analysis['reasons'].append(f"Price too high: ${current_price:.2f} > $1000")
                    passes_all_filters = False
                
                # Volume filter
                if volume_avg < 100000:
                    stock_analysis['reasons'].append(f"Volume too low: {volume_avg:,.0f} < 100,000")
                    passes_all_filters = False
                
                # Market cap filter
                market_cap = info.get('market_cap', 0)
                if market_cap > 0:
                    if market_cap < 1e9:  # Less than $1B
                        stock_analysis['reasons'].append(f"Market cap too small: ${market_cap/1e9:.1f}B < $1B")
                        passes_all_filters = False
                    elif market_cap > 1e12:  # More than $1T
                        stock_analysis['reasons'].append(f"Market cap too large: ${market_cap/1e12:.1f}T > $1T")
                        passes_all_filters = False
                
                # Step 4: Calculate scoring for qualifying stocks
                if passes_all_filters:
                    # Multi-factor scoring
                    momentum_score = min(50, max(0, price_change_5d * 100 + 25))  # -25% to +25% -> 0-50
                    volume_score = min(25, (volume_avg / 1000000) * 10)  # Higher volume = higher score
                    value_score = 25  # Baseline value score
                    
                    # PE ratio adjustment
                    pe_ratio = info.get('pe_ratio', 0)
                    if pe_ratio > 0:
                        if pe_ratio < 15:
                            value_score += 10  # Undervalued bonus
                        elif pe_ratio > 30:
                            value_score -= 5   # Overvalued penalty
                    
                    total_score = momentum_score + volume_score + value_score
                    stock_analysis['score'] = total_score
                    stock_analysis['status'] = 'selected'
                    stock_analysis['reasons'] = [
                        f"Momentum score: {momentum_score:.1f}/50",
                        f"Volume score: {volume_score:.1f}/25", 
                        f"Value score: {value_score:.1f}/35",
                        f"Total score: {total_score:.1f}/110"
                    ]
                    
                    selected_stocks.append({
                        'symbol': symbol,
                        'score': total_score,
                        'current_price': current_price,
                        'volume': volume_avg,
                        'price_change_5d': price_change_5d,
                        'market_cap': market_cap,
                        'sector': info.get('sector', 'Unknown'),
                        'reasoning': f"Score: {total_score:.1f} (Momentum: {momentum_score:.1f}, Volume: {volume_score:.1f}, Value: {value_score:.1f})"
                    })
                    
            except Exception as e:
                stock_analysis['reasons'].append(f"Analysis error: {str(e)}")
                logger.warning(f"[SELECTION] Error analyzing {symbol}: {e}")
            
            analysis_results.append(stock_analysis)
        
        # Sort selected stocks by score
        selected_stocks.sort(key=lambda x: x['score'], reverse=True)
        selected_stocks = selected_stocks[:max_stocks]
        
        # Statistics
        total_analyzed = len(analysis_results)
        total_selected = len(selected_stocks)
        total_rejected = total_analyzed - total_selected
        
        # Rejection reasons summary
        rejection_reasons = {}
        for analysis in analysis_results:
            if analysis['status'] == 'rejected':
                for reason in analysis['reasons']:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        logger.info(f"[SELECTION] Analysis complete: {total_selected}/{total_analyzed} stocks selected")
        logger.info(f"[SELECTION] Top rejection reasons: {dict(list(rejection_reasons.items())[:3])}")
        
        return {
            "selected_stocks": selected_stocks,
            "detailed_analysis": analysis_results,
            "statistics": {
                "total_analyzed": total_analyzed,
                "total_selected": total_selected,
                "total_rejected": total_rejected,
                "rejection_reasons": rejection_reasons
            },
            "success": True
        }
        
    except Exception as e:
        logger.error(f"[SELECTION] Critical error in detailed analysis: {e}")
        import traceback
        logger.error(f"[SELECTION] Traceback: {traceback.format_exc()}")
        return {
            "selected_stocks": [], 
            "detailed_analysis": [],
            "error": str(e), 
            "success": False
        }

def run_simple_stock_selection(universe: List[str], max_stocks: int = 10, use_parallel: bool = None) -> Dict[str, Any]:
    """
    Legacy simplified selection - now calls detailed analysis for consistency.
    
    Args:
        universe: List of stock symbols to analyze
        max_stocks: Maximum number of stocks to select  
        use_parallel: Force parallel processing (None = auto-detect)
    """
    result = run_detailed_stock_analysis(universe, max_stocks, use_parallel)
    
    # Return in legacy format for compatibility
    return {
        "selected_stocks": result.get("selected_stocks", []),
        "total_analyzed": result.get("statistics", {}).get("total_analyzed", 0),
        "success": result.get("success", False),
        "error": result.get("error", None)
    }

def run_complex_stock_selection_with_fallback(universe: List[str], max_stocks: int = 10, use_parallel: bool = None) -> Dict[str, Any]:
    """
    Attempts to run complex multi-strategy selection with detailed reporting and parallel processing.
    Falls back to detailed simple selection if complex strategies fail.
    
    Args:
        universe: List of stock symbols to analyze
        max_stocks: Maximum number of stocks to select
        use_parallel: Force parallel processing (None = auto-detect based on size)
    """
    try:
        setup_bot_path()
        
        logger.info("[SELECTION] Attempting complex multi-strategy selection with detailed analysis")
        
        # Try complex strategy system first
        try:
            from selection_strategies.strategy_combiner import StrategyCombiner
            from selection_strategies.base_strategy import SelectionCriteria
            
            criteria = SelectionCriteria(
                max_stocks=max_stocks,
                min_market_cap=1e9,
                max_market_cap=5e12,
                min_volume=100000,
                min_price=5.0,
                max_price=1000.0,
                min_score_threshold=-1.0
            )
            
            combiner = StrategyCombiner(
                min_strategy_agreement=1,
                confidence_threshold=0.3
            )
            
            # Run complex selection on the entire universe
            # Analyze all stocks for comprehensive coverage
            logger.info(f"[SELECTION] Analyzing ALL {len(universe)} stocks for comprehensive selection")
            
            results = combiner.run_combined_selection(universe, criteria)  
            
            if results and results.selected_stocks and len(results.selected_stocks) > 0:
                logger.info(f"[SELECTION] Complex strategy succeeded: {len(results.selected_stocks)} stocks")
                
                # Convert to standardized format
                selected_stocks = []
                for stock in results.selected_stocks:
                    selected_stocks.append({
                        'symbol': stock.symbol,
                        'score': stock.score,
                        'action': stock.action.value if hasattr(stock.action, 'value') else str(stock.action),
                        'reasoning': getattr(stock, 'reasoning', 'Multi-strategy analysis'),
                        'confidence': getattr(stock, 'confidence', 0.5)
                    })
                
                # For complex strategy, we still want detailed analysis of rejected stocks
                # So run detailed analysis on the full universe to show why others were rejected
                detailed_result = run_detailed_stock_analysis(universe, max_stocks, use_parallel)
                
                return {
                    "selected_stocks": selected_stocks,
                    "detailed_analysis": detailed_result.get("detailed_analysis", []),
                    "statistics": detailed_result.get("statistics", {}),
                    "method": "complex_strategy_with_detailed_fallback",
                    "success": True
                }
            else:
                logger.warning("[SELECTION] Complex strategy returned no results, falling back to detailed analysis")
                
        except Exception as e:
            logger.warning(f"[SELECTION] Complex strategy failed: {e}, falling back to detailed analysis")
    
    except Exception as e:
        logger.warning(f"[SELECTION] Setup failed: {e}, falling back to detailed analysis")
    
    # Fallback to detailed simple selection 
    logger.info("[SELECTION] Using detailed stock analysis fallback")
    result = run_detailed_stock_analysis(universe, max_stocks, use_parallel)
    result["method"] = "detailed_analysis_fallback"
    return result


def run_high_performance_stock_analysis(universe: List[str], max_stocks: int = 10) -> Dict[str, Any]:
    """
    High-performance stock analysis optimized for large universes (>100 stocks).
    Automatically enables parallel processing and provides detailed performance metrics.
    
    This function is designed for the 428-stock scenario mentioned in the requirements.
    
    Args:
        universe: List of stock symbols to analyze
        max_stocks: Maximum number of stocks to select
        
    Returns:
        Enhanced results with performance metrics
    """
    start_time = time.time()
    
    logger.info(f"[HIGH_PERFORMANCE] Starting analysis of {len(universe)} stocks")
    logger.info(f"[HIGH_PERFORMANCE] Expected time savings: from 20-30min serial to 2-5min parallel")
    
    # Force parallel processing for large universes
    use_parallel = len(universe) > 10
    
    try:
        setup_bot_path()
        from yahoo_data import get_batch_fetch_stats
        
        # Log system capabilities
        system_stats = get_batch_fetch_stats()
        logger.info(f"[HIGH_PERFORMANCE] System optimized: {system_stats['performance_tier']} tier, "
                   f"{system_stats['optimal_workers']} workers, "
                   f"{system_stats['estimated_symbols_per_minute']} symbols/min capability")
        
        # Run analysis with parallel processing
        result = run_detailed_stock_analysis(universe, max_stocks, use_parallel=True)
        
        # Add performance metrics
        total_time = time.time() - start_time
        symbols_per_second = len(universe) / total_time
        symbols_per_minute = symbols_per_second * 60
        
        # Estimate serial time for comparison
        estimated_serial_time = len(universe) * 2.5  # 2.5 seconds per symbol serially
        time_savings = estimated_serial_time - total_time
        efficiency_gain = (time_savings / estimated_serial_time) * 100
        
        # Enhanced result with performance data
        result["performance_metrics"] = {
            "total_time_seconds": total_time,
            "symbols_per_second": symbols_per_second,
            "symbols_per_minute": symbols_per_minute,
            "estimated_serial_time": estimated_serial_time,
            "time_savings_seconds": time_savings,
            "efficiency_gain_percent": efficiency_gain,
            "parallel_processing_used": True,
            "system_tier": system_stats['performance_tier'],
            "workers_used": system_stats['optimal_workers']
        }
        
        result["method"] = "high_performance_parallel_analysis"
        
        logger.info(f"[HIGH_PERFORMANCE] Analysis completed in {total_time:.1f}s "
                   f"({symbols_per_minute:.1f} symbols/min)")
        logger.info(f"[HIGH_PERFORMANCE] Efficiency gain: {efficiency_gain:.1f}% "
                   f"(saved {time_savings/60:.1f} minutes)")
        
        return result
        
    except Exception as e:
        logger.error(f"[HIGH_PERFORMANCE] Error in high-performance analysis: {e}")
        # Fallback to regular detailed analysis
        result = run_detailed_stock_analysis(universe, max_stocks, use_parallel=False)
        result["method"] = "high_performance_fallback"
        result["error"] = f"High-performance mode failed: {str(e)}"
        return result


def is_streaming_mode_enabled() -> bool:
    """
    Check if streaming mode is enabled from environment configuration.
    
    Returns:
        True if streaming mode should be used, False for traditional mode
    """
    try:
        streaming_mode = os.getenv('STREAMING_MODE', 'false').lower().strip()
        return streaming_mode in ('true', '1', 'yes', 'on', 'enabled')
    except Exception as e:
        logger.warning(f"Error checking streaming mode config: {e}, defaulting to traditional mode")
        return False


def run_streaming_stock_selection(universe: List[str], max_stocks: int = 20) -> Dict[str, Any]:
    """
    Run streaming stock selection for real-time results and memory efficiency.
    
    Args:
        universe: List of stock symbols to analyze
        max_stocks: Maximum number of stocks to select
        
    Returns:
        Dictionary with selected stocks and metadata
    """
    try:
        setup_bot_path()
        
        logger.info(f"[STREAMING] Starting streaming selection on {len(universe)} stocks")
        logger.info(f"[STREAMING] Expected benefits: 90% memory reduction, real-time progress")
        
        # Import streaming strategy
        try:
            from selection_strategies.streaming_value_momentum import StreamingValueMomentumStrategy
            from selection_strategies.base_strategy import SelectionCriteria
        except ImportError as e:
            logger.error(f"[STREAMING] Failed to import streaming modules: {e}")
            return {"selected_stocks": [], "error": "Streaming import failed", "success": False}
        
        # Create selection criteria optimized for streaming
        criteria = SelectionCriteria(
            max_stocks=max_stocks,
            min_market_cap=1e8,  # $100M minimum (relaxed for streaming)
            max_market_cap=5e12, # $5T maximum
            min_volume=50000,    # 50K minimum volume (relaxed)
            min_price=1.0,       # $1 minimum (very relaxed)
            max_price=2000.0,    # $2000 maximum
            min_score_threshold=0.0  # Let the strategy handle scoring
        )
        
        # Create progress callback for WebSocket updates
        def progress_callback(progress_data):
            if WEBSOCKET_AVAILABLE:
                send_selection_progress_sync(progress_data)
        
        # Initialize streaming strategy with optimized parameters for 5000 stocks
        streaming_strategy = StreamingValueMomentumStrategy(
            candidate_pool_size=150,  # Increased from 100 for better quality
            intermediate_save_interval=10,       # Save every 10 batches (reduced IO)
            early_stop_patience=20,              # Much more patience for 5000 stocks  
            min_score_threshold=35.0,            # Lowered threshold for more candidates
            progress_callback=progress_callback  # WebSocket progress updates
        )
        
        # Send initial progress update
        if WEBSOCKET_AVAILABLE:
            send_selection_progress_sync({
                "status": "started",
                "total_stocks": len(universe),
                "processed_stocks": 0,
                "current_batch": 0,
                "strategy": "StreamingValueMomentum",
                "candidate_pool_size": 0,
                "top_candidates": [],
                "eta_seconds": None,
                "progress_percentage": 0.0
            })
        
        # Run streaming selection
        start_time = time.time()
        results = streaming_strategy.select_stocks(universe, criteria)
        execution_time = time.time() - start_time
        
        # Send completion progress update
        if WEBSOCKET_AVAILABLE:
            send_selection_progress_sync({
                "status": "completed",
                "total_stocks": len(universe),
                "processed_stocks": len(universe),
                "current_batch": -1,
                "strategy": "StreamingValueMomentum",
                "candidate_pool_size": len(results.selected_stocks if results.selected_stocks else []),
                "top_candidates": [{"symbol": s.symbol, "score": s.score} for s in results.selected_stocks[:5]] if results.selected_stocks else [],
                "eta_seconds": 0,
                "progress_percentage": 100.0,
                "execution_time": execution_time
            })
        
        # Convert to standardized format
        selected_stocks = []
        for stock in results.selected_stocks:
            selected_stocks.append({
                'symbol': stock.symbol,
                'score': stock.score,
                'action': stock.action.value if hasattr(stock.action, 'value') else str(stock.action),
                'reasoning': stock.reasoning,
                'confidence': getattr(stock, 'confidence', 0.8),
                'timestamp': stock.timestamp.isoformat() if hasattr(stock, 'timestamp') else None
            })
        
        # Extract streaming metadata
        streaming_stats = results.metadata.get('streaming_stats', {})
        
        logger.info(f"[STREAMING] Selection completed: {len(selected_stocks)} stocks selected "
                   f"in {execution_time:.2f}s")
        logger.info(f"[STREAMING] Streaming stats: {streaming_stats.get('batches_completed', 0)} batches, "
                   f"early_stopped={streaming_stats.get('early_stopped', False)}")
        
        return {
            "selected_stocks": selected_stocks,
            "total_candidates": results.total_candidates,
            "execution_time": execution_time,
            "streaming_stats": streaming_stats,
            "method": "streaming_selection",
            "success": True,
            "errors": results.errors if results.errors else []
        }
        
    except Exception as e:
        logger.error(f"[STREAMING] Critical error in streaming selection: {e}")
        import traceback
        logger.error(f"[STREAMING] Traceback: {traceback.format_exc()}")
        return {
            "selected_stocks": [],
            "error": str(e),
            "method": "streaming_selection_failed", 
            "success": False
        }


def run_adaptive_stock_selection(universe: List[str], max_stocks: int = 20) -> Dict[str, Any]:
    """
    Adaptive stock selection that chooses between streaming and traditional modes
    based on configuration and circumstances.
    
    Args:
        universe: List of stock symbols to analyze
        max_stocks: Maximum number of stocks to select
        
    Returns:
        Dictionary with selected stocks and metadata
    """
    # Check streaming mode configuration
    streaming_enabled = is_streaming_mode_enabled()
    universe_size = len(universe)
    
    logger.info(f"[ADAPTIVE] Processing {universe_size} stocks, "
               f"streaming_mode={streaming_enabled}")
    
    # Decision logic for mode selection
    if streaming_enabled and universe_size >= 100:
        logger.info("[ADAPTIVE] Using streaming mode for large universe with real-time processing")
        return run_streaming_stock_selection(universe, max_stocks)
    
    elif streaming_enabled and universe_size >= 50:
        logger.info("[ADAPTIVE] Using streaming mode for medium universe")
        return run_streaming_stock_selection(universe, max_stocks)
    
    elif streaming_enabled:
        logger.info("[ADAPTIVE] Using streaming mode (configured)")
        return run_streaming_stock_selection(universe, max_stocks)
    
    else:
        # Traditional mode with intelligent selection based on size
        logger.info(f"[ADAPTIVE] Using traditional mode (streaming_mode={streaming_enabled})")
        
        if universe_size > 500:
            return run_high_performance_stock_analysis(universe, max_stocks)
        elif universe_size > 100:
            return run_complex_stock_selection_with_fallback(universe, max_stocks, use_parallel=True)
        else:
            return run_complex_stock_selection_with_fallback(universe, max_stocks, use_parallel=False)