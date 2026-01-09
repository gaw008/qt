"""
Risk-Integrated Stock Selection System

This module provides integrated stock selection that combines:
- Multi-factor scoring engine
- Stock screening capabilities  
- Comprehensive risk filtering
- Portfolio-level risk management
- Dynamic position sizing

Acts as the main interface for risk-aware stock selection in the trading system.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import json
from pathlib import Path

# Import core modules
from .config import SETTINGS
from .data import get_batch_latest_data, fetch_batch_history

# Import selection and scoring modules
try:
    from .scoring_engine import MultiFactorScoringEngine, FactorWeights, ScoringResult
    from .stock_screener import StockScreener, ScreeningCriteria, StockScore
    from .risk_filters import RiskFilterEngine, RiskLimits, RiskMetrics
    HAS_ALL_MODULES = True
except ImportError as e:
    HAS_ALL_MODULES = False
    warnings.warn(f"Some modules not available: {e}")

# Import sector management if available
try:
    from .sector_manager import get_all_stocks, get_sector_stocks, get_stock_sector
    HAS_SECTOR_MANAGER = True
except ImportError:
    HAS_SECTOR_MANAGER = False
    warnings.warn("Sector manager not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    """Configuration for integrated selection system."""
    
    # Selection parameters
    max_positions: int = 20
    min_positions: int = 5
    selection_universe_size: int = 500
    
    # Scoring weights
    factor_weights: Optional[FactorWeights] = None
    
    # Screening criteria
    screening_criteria: Optional[ScreeningCriteria] = None
    
    # Risk limits
    risk_limits: Optional[RiskLimits] = None
    
    # Portfolio parameters
    target_volatility: float = 0.15  # 15% target volatility
    max_sector_allocation: float = 0.30  # 30% max per sector
    rebalance_threshold: float = 0.05  # 5% rebalance threshold
    
    # Market conditions
    market_regime_sensitivity: float = 1.0  # Sensitivity to market conditions
    defensive_mode_threshold: float = 25.0  # VIX threshold for defensive mode
    
    def __post_init__(self):
        if self.factor_weights is None:
            self.factor_weights = FactorWeights()
        if self.screening_criteria is None:
            self.screening_criteria = ScreeningCriteria(top_n=self.max_positions)
        if self.risk_limits is None:
            self.risk_limits = RiskLimits()


@dataclass
class SelectionResult:
    """Results from integrated selection process."""
    
    # Selected positions
    selected_symbols: List[str]
    position_weights: Dict[str, float]
    
    # Detailed scoring
    scoring_result: Optional[ScoringResult] = None
    stock_scores: List[StockScore] = field(default_factory=list)
    risk_metrics: Dict[str, RiskMetrics] = field(default_factory=dict)
    
    # Portfolio metrics
    portfolio_score: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_risk_score: float = 0.0
    
    # Selection statistics
    selection_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Risk validation
    risk_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_used: Optional[SelectionConfig] = None


class RiskIntegratedSelector:
    """
    Main class for risk-integrated stock selection.
    
    This class orchestrates the entire selection process:
    1. Universe definition and data loading
    2. Multi-factor scoring
    3. Risk filtering and validation  
    4. Portfolio construction with risk controls
    5. Performance validation and reporting
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        """
        Initialize the integrated selector.
        
        Args:
            config: Selection configuration
        """
        self.config = config or SelectionConfig()
        
        # Initialize component engines if available
        if HAS_ALL_MODULES:
            self.scoring_engine = MultiFactorScoringEngine(self.config.factor_weights)
            self.screener = StockScreener(self.config.screening_criteria)
            self.risk_engine = RiskFilterEngine(self.config.risk_limits)
        else:
            self.scoring_engine = None
            self.screener = None
            self.risk_engine = None
            logger.warning("[risk_selection] Enhanced modules not available")
        
        # Selection history and state
        self.selection_history: List[SelectionResult] = []
        self.current_portfolio: Dict[str, float] = {}
        self.last_rebalance_date: Optional[datetime] = None
        
        logger.info("[risk_selection] Risk-integrated selector initialized")
    
    def run_integrated_selection(self,
                                quote_client,
                                universe: Optional[List[str]] = None,
                                current_positions: Optional[Dict[str, float]] = None,
                                market_data: Optional[Dict[str, pd.DataFrame]] = None,
                                dry_run: bool = False) -> SelectionResult:
        """
        Run complete integrated selection process.
        
        Args:
            quote_client: Tiger SDK quote client
            universe: Optional stock universe (if None, uses sector manager)
            current_positions: Current portfolio positions
            market_data: Optional pre-loaded market data
            dry_run: Use placeholder data
            
        Returns:
            SelectionResult with complete selection analysis
        """
        logger.info("[risk_selection] Starting integrated selection process")
        
        if not HAS_ALL_MODULES:
            return self._fallback_selection(universe)
        
        try:
            # Step 1: Define universe and load data
            selection_universe = self._define_selection_universe(universe)
            
            if not selection_universe:
                logger.warning("[risk_selection] Empty selection universe")
                return SelectionResult(selected_symbols=[], position_weights={})
            
            logger.info(f"[risk_selection] Universe size: {len(selection_universe)} symbols")
            
            # Step 2: Load market data
            if market_data is None:
                market_data = self._load_market_data(quote_client, selection_universe, dry_run)
            
            if not market_data:
                logger.warning("[risk_selection] No market data available")
                return SelectionResult(selected_symbols=[], position_weights={})
            
            logger.info(f"[risk_selection] Loaded data for {len(market_data)} symbols")
            
            # Step 3: Run multi-factor scoring
            scoring_result = self._run_scoring(market_data)
            
            if scoring_result.scores.empty:
                logger.warning("[risk_selection] No scoring results")
                return SelectionResult(selected_symbols=[], position_weights={})
            
            logger.info(f"[risk_selection] Scored {len(scoring_result.scores)} symbols")
            
            # Step 4: Apply risk filters
            candidate_symbols = scoring_result.scores['symbol'].tolist()
            sector_mapping = self._get_sector_mapping(candidate_symbols)
            
            filtered_symbols, risk_metrics = self.risk_engine.apply_risk_filters(
                candidate_symbols, market_data, sector_mapping, None, current_positions
            )
            
            if not filtered_symbols:
                logger.warning("[risk_selection] No symbols passed risk filters")
                return SelectionResult(selected_symbols=[], position_weights={})
            
            logger.info(f"[risk_selection] {len(filtered_symbols)} symbols passed risk filters")
            
            # Step 5: Select top stocks
            final_selection = self._select_top_stocks(scoring_result, filtered_symbols)
            
            if not final_selection:
                logger.warning("[risk_selection] No stocks in final selection")
                return SelectionResult(selected_symbols=[], position_weights={})
            
            # Step 6: Calculate risk-adjusted position weights
            position_weights = self._calculate_position_weights(
                final_selection, scoring_result, risk_metrics, market_data, sector_mapping
            )
            
            # Step 7: Validate portfolio risk
            risk_validation = self._validate_portfolio_risk(
                position_weights, market_data, risk_metrics, sector_mapping
            )
            
            # Step 8: Create detailed result
            result = self._create_selection_result(
                list(position_weights.keys()), position_weights, scoring_result,
                risk_metrics, risk_validation, market_data
            )
            
            # Store result
            self.selection_history.append(result)
            if len(self.selection_history) > 100:  # Keep last 100 selections
                self.selection_history = self.selection_history[-100:]
            
            logger.info(f"[risk_selection] Selection completed: {len(result.selected_symbols)} positions")
            
            return result
            
        except Exception as e:
            logger.error(f"[risk_selection] Selection process failed: {e}")
            return SelectionResult(selected_symbols=[], position_weights={})
    
    def check_rebalance_needed(self,
                             current_positions: Dict[str, float],
                             market_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if portfolio rebalancing is needed.
        
        Args:
            current_positions: Current portfolio positions
            market_data: Optional market data for analysis
            
        Returns:
            Tuple of (rebalance_needed, rebalance_analysis)
        """
        if not self.risk_engine or not current_positions:
            return False, {"reason": "No risk engine or positions"}
        
        rebalance_analysis = {
            "rebalance_needed": False,
            "reasons": [],
            "drift_analysis": {},
            "risk_analysis": {}
        }
        
        try:
            # Check position drift
            if self.last_rebalance_date:
                days_since_rebalance = (datetime.now() - self.last_rebalance_date).days
                if days_since_rebalance >= 30:  # Monthly rebalance
                    rebalance_analysis["rebalance_needed"] = True
                    rebalance_analysis["reasons"].append("Scheduled rebalancing due")
            
            # Check risk limit breaches if market data available
            if market_data:
                symbols = list(current_positions.keys())
                risk_metrics = self.risk_engine._calculate_risk_metrics(
                    symbols, market_data, None, None
                )
                
                # Validate current portfolio
                risk_validation = self.risk_engine.validate_portfolio_risk(
                    current_positions, market_data, risk_metrics
                )
                
                rebalance_analysis["risk_analysis"] = risk_validation
                
                if not risk_validation.get("passes_validation", True):
                    rebalance_analysis["rebalance_needed"] = True
                    rebalance_analysis["reasons"].append("Risk limit violations")
            
            return rebalance_analysis["rebalance_needed"], rebalance_analysis
            
        except Exception as e:
            logger.warning(f"[risk_selection] Rebalance check failed: {e}")
            return False, {"error": str(e)}\n    \n    def get_selection_summary(self) -> Dict[str, Any]:\n        """Get summary of recent selections."""\n        if not self.selection_history:\n            return {"error": "No selection history available"}\n        \n        recent_results = self.selection_history[-10:]  # Last 10 selections\n        \n        summary = {\n            "total_selections": len(self.selection_history),\n            "recent_selections": len(recent_results),\n            "avg_positions": np.mean([len(r.selected_symbols) for r in recent_results]),\n            "avg_portfolio_score": np.mean([r.portfolio_score for r in recent_results]),\n            "avg_risk_score": np.mean([r.portfolio_risk_score for r in recent_results]),\n            "last_selection_date": recent_results[-1].timestamp,\n            "position_stability": self._calculate_position_stability()\n        }\n        \n        return summary\n    \n    def _define_selection_universe(self, universe: Optional[List[str]]) -> List[str]:\n        """Define the stock selection universe."""\n        if universe:\n            return universe\n        \n        if HAS_SECTOR_MANAGER:\n            try:\n                # Get universe from sector manager\n                all_stocks = get_all_stocks(validate=True)\n                \n                # Apply basic filters from config\n                filtered_universe = []\n                exclude_sectors = SETTINGS.get_exclude_sectors_list()\n                include_sectors = SETTINGS.get_include_sectors_list()\n                \n                for symbol in all_stocks:\n                    sector = get_stock_sector(symbol)\n                    \n                    # Skip excluded sectors\n                    if exclude_sectors and sector in exclude_sectors:\n                        continue\n                    \n                    # Include only specified sectors if configured\n                    if include_sectors and sector not in include_sectors:\n                        continue\n                    \n                    filtered_universe.append(symbol)\n                \n                # Limit universe size\n                if len(filtered_universe) > self.config.selection_universe_size:\n                    filtered_universe = filtered_universe[:self.config.selection_universe_size]\n                \n                return filtered_universe\n                \n            except Exception as e:\n                logger.warning(f"[risk_selection] Error getting universe from sector manager: {e}")\n        \n        # Fallback universe\n        return [\n            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V",\n            "PG", "UNH", "HD", "MA", "BAC", "DIS", "ADBE", "NFLX", "CRM", "CMCSA",\n            "XOM", "ABT", "VZ", "PFE", "KO", "NKE", "INTC", "MRK", "TMO", "DHR"\n        ]\n    \n    def _load_market_data(self,\n                         quote_client,\n                         symbols: List[str],\n                         dry_run: bool) -> Dict[str, pd.DataFrame]:\n        """Load market data for symbol universe."""\n        try:\n            return get_batch_latest_data(quote_client, symbols, dry_run)\n        except Exception as e:\n            logger.error(f"[risk_selection] Failed to load market data: {e}")\n            return {}\n    \n    def _run_scoring(self, market_data: Dict[str, pd.DataFrame]) -> ScoringResult:\n        """Run multi-factor scoring on market data."""\n        try:\n            return self.scoring_engine.calculate_composite_scores(market_data)\n        except Exception as e:\n            logger.error(f"[risk_selection] Scoring failed: {e}")\n            return ScoringResult(\n                scores=pd.DataFrame(),\n                factor_contributions=pd.DataFrame(),\n                factor_correlations=pd.DataFrame(),\n                weights_used={}\n            )\n    \n    def _get_sector_mapping(self, symbols: List[str]) -> Dict[str, str]:\n        """Get sector mapping for symbols."""\n        sector_mapping = {}\n        \n        if HAS_SECTOR_MANAGER:\n            for symbol in symbols:\n                try:\n                    sector = get_stock_sector(symbol)\n                    sector_mapping[symbol] = sector\n                except Exception:\n                    sector_mapping[symbol] = "Unknown"\n        else:\n            # Fallback sector assignment\n            for symbol in symbols:\n                sector_mapping[symbol] = "Technology"  # Default sector\n        \n        return sector_mapping\n    \n    def _select_top_stocks(self,\n                          scoring_result: ScoringResult,\n                          filtered_symbols: List[str]) -> List[str]:\n        """Select top stocks from filtered candidates."""\n        # Filter scoring results to only include risk-filtered symbols\n        filtered_scores = scoring_result.scores[\n            scoring_result.scores['symbol'].isin(filtered_symbols)\n        ].copy()\n        \n        if filtered_scores.empty:\n            return []\n        \n        # Select top N stocks\n        top_stocks = filtered_scores.nlargest(self.config.max_positions, 'composite_score')\n        \n        # Ensure minimum positions if available\n        n_selected = len(top_stocks)\n        if n_selected < self.config.min_positions and len(filtered_scores) >= self.config.min_positions:\n            top_stocks = filtered_scores.nlargest(self.config.min_positions, 'composite_score')\n        \n        return top_stocks['symbol'].tolist()\n    \n    def _calculate_position_weights(self,\n                                  selected_symbols: List[str],\n                                  scoring_result: ScoringResult,\n                                  risk_metrics: Dict[str, RiskMetrics],\n                                  market_data: Dict[str, pd.DataFrame],\n                                  sector_mapping: Dict[str, str]) -> Dict[str, float]:\n        """Calculate risk-adjusted position weights."""\n        if not selected_symbols:\n            return {}\n        \n        # Extract scores for selected symbols\n        symbol_scores = {}\n        for symbol in selected_symbols:\n            score_row = scoring_result.scores[scoring_result.scores['symbol'] == symbol]\n            if not score_row.empty:\n                symbol_scores[symbol] = score_row['composite_score'].iloc[0]\n            else:\n                symbol_scores[symbol] = 0.0\n        \n        # Use risk engine to calculate weights\n        position_weights = self.risk_engine.calculate_position_sizes(\n            selected_symbols, symbol_scores, risk_metrics, sector_mapping\n        )\n        \n        return position_weights\n    \n    def _validate_portfolio_risk(self,\n                               position_weights: Dict[str, float],\n                               market_data: Dict[str, pd.DataFrame],\n                               risk_metrics: Dict[str, RiskMetrics],\n                               sector_mapping: Dict[str, str]) -> Dict[str, Any]:\n        """Validate portfolio-level risk."""\n        return self.risk_engine.validate_portfolio_risk(\n            position_weights, market_data, risk_metrics, sector_mapping\n        )\n    \n    def _create_selection_result(self,\n                               selected_symbols: List[str],\n                               position_weights: Dict[str, float],\n                               scoring_result: ScoringResult,\n                               risk_metrics: Dict[str, RiskMetrics],\n                               risk_validation: Dict[str, Any],\n                               market_data: Dict[str, pd.DataFrame]) -> SelectionResult:\n        """Create comprehensive selection result."""\n        \n        # Calculate portfolio metrics\n        portfolio_score = sum(\n            weight * scoring_result.scores[scoring_result.scores['symbol'] == symbol]['composite_score'].iloc[0]\n            for symbol, weight in position_weights.items()\n            if not scoring_result.scores[scoring_result.scores['symbol'] == symbol].empty\n        )\n        \n        portfolio_risk_score = sum(\n            weight * risk_metrics.get(symbol, RiskMetrics(symbol)).overall_risk_score\n            for symbol, weight in position_weights.items()\n        )\n        \n        # Calculate portfolio volatility\n        portfolio_volatility = risk_validation.get('risk_metrics', {}).get('portfolio_volatility', 0.0)\n        \n        # Selection statistics\n        selection_stats = {\n            "total_universe": len(market_data),\n            "scored_symbols": len(scoring_result.scores),\n            "risk_filtered": len([s for s in risk_metrics.values() if s.passes_filters]),\n            "final_selection": len(selected_symbols),\n            "avg_score": np.mean([scoring_result.scores[scoring_result.scores['symbol'] == s]['composite_score'].iloc[0] \n                               for s in selected_symbols \n                               if not scoring_result.scores[scoring_result.scores['symbol'] == s].empty]),\n            "score_range": {\n                "min": min([scoring_result.scores[scoring_result.scores['symbol'] == s]['composite_score'].iloc[0] \n                          for s in selected_symbols \n                          if not scoring_result.scores[scoring_result.scores['symbol'] == s].empty], default=0),\n                "max": max([scoring_result.scores[scoring_result.scores['symbol'] == s]['composite_score'].iloc[0] \n                          for s in selected_symbols \n                          if not scoring_result.scores[scoring_result.scores['symbol'] == s].empty], default=0)\n            }\n        }\n        \n        result = SelectionResult(\n            selected_symbols=selected_symbols,\n            position_weights=position_weights,\n            scoring_result=scoring_result,\n            risk_metrics=risk_metrics,\n            portfolio_score=portfolio_score,\n            portfolio_volatility=portfolio_volatility,\n            portfolio_risk_score=portfolio_risk_score,\n            selection_stats=selection_stats,\n            risk_validation=risk_validation,\n            config_used=self.config\n        )\n        \n        return result\n    \n    def _calculate_position_stability(self) -> float:\n        """Calculate stability of position selection over time."""\n        if len(self.selection_history) < 2:\n            return 0.0\n        \n        stability_scores = []\n        \n        for i in range(1, len(self.selection_history)):\n            prev_symbols = set(self.selection_history[i-1].selected_symbols)\n            curr_symbols = set(self.selection_history[i].selected_symbols)\n            \n            if len(prev_symbols) == 0:\n                continue\n            \n            # Calculate overlap\n            overlap = len(prev_symbols.intersection(curr_symbols))\n            stability = overlap / len(prev_symbols)\n            stability_scores.append(stability)\n        \n        return np.mean(stability_scores) if stability_scores else 0.0\n    \n    def _fallback_selection(self, universe: Optional[List[str]]) -> SelectionResult:\n        """Fallback selection when enhanced modules not available."""\n        logger.warning("[risk_selection] Using fallback selection")\n        \n        fallback_universe = universe or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]\n        selected = fallback_universe[:5]\n        equal_weight = 1.0 / len(selected) if selected else 0.0\n        weights = {symbol: equal_weight for symbol in selected}\n        \n        return SelectionResult(\n            selected_symbols=selected,\n            position_weights=weights,\n            selection_stats={"fallback_mode": True}\n        )\n    \n    def save_selection_history(self, filepath: str):\n        """Save selection history to file."""\n        history_data = {\n            "timestamp": datetime.now().isoformat(),\n            "config": {\n                "max_positions": self.config.max_positions,\n                "target_volatility": self.config.target_volatility,\n                "max_sector_allocation": self.config.max_sector_allocation\n            },\n            "selections": []\n        }\n        \n        for result in self.selection_history:\n            selection_data = {\n                "timestamp": result.timestamp,\n                "selected_symbols": result.selected_symbols,\n                "position_weights": result.position_weights,\n                "portfolio_score": result.portfolio_score,\n                "portfolio_volatility": result.portfolio_volatility,\n                "selection_stats": result.selection_stats\n            }\n            history_data["selections"].append(selection_data)\n        \n        Path(filepath).parent.mkdir(parents=True, exist_ok=True)\n        \n        with open(filepath, 'w') as f:\n            json.dump(history_data, f, indent=2)\n        \n        logger.info(f"[risk_selection] Selection history saved to {filepath}")\n    \n    def load_selection_history(self, filepath: str):\n        """Load selection history from file."""\n        try:\n            with open(filepath, 'r') as f:\n                history_data = json.load(f)\n            \n            # Reconstruct selection results (simplified)\n            self.selection_history = []\n            \n            for selection_data in history_data.get("selections", []):\n                result = SelectionResult(\n                    selected_symbols=selection_data.get("selected_symbols", []),\n                    position_weights=selection_data.get("position_weights", {}),\n                    portfolio_score=selection_data.get("portfolio_score", 0.0),\n                    portfolio_volatility=selection_data.get("portfolio_volatility", 0.0),\n                    selection_stats=selection_data.get("selection_stats", {}),\n                    timestamp=selection_data.get("timestamp", "")\n                )\n                self.selection_history.append(result)\n            \n            logger.info(f"[risk_selection] Loaded {len(self.selection_history)} selections from {filepath}")\n            \n        except Exception as e:\n            logger.error(f"[risk_selection] Failed to load selection history: {e}")\n\n\n# Utility functions for easy integration\n\ndef run_risk_aware_selection(quote_client,\n                            max_positions: int = 20,\n                            target_volatility: float = 0.15,\n                            universe: Optional[List[str]] = None,\n                            dry_run: bool = False) -> SelectionResult:\n    \"\"\"\n    Convenience function for risk-aware stock selection.\n    \n    Args:\n        quote_client: Tiger SDK quote client\n        max_positions: Maximum number of positions\n        target_volatility: Target portfolio volatility\n        universe: Optional stock universe\n        dry_run: Use placeholder data\n        \n    Returns:\n        SelectionResult with risk-aware selection\n    \"\"\"\n    config = SelectionConfig(\n        max_positions=max_positions,\n        target_volatility=target_volatility\n    )\n    \n    selector = RiskIntegratedSelector(config)\n    return selector.run_integrated_selection(quote_client, universe=universe, dry_run=dry_run)\n\n\ndef get_rebalance_recommendation(current_positions: Dict[str, float],\n                               quote_client,\n                               dry_run: bool = False) -> Tuple[bool, SelectionResult]:\n    \"\"\"\n    Get rebalancing recommendation for current portfolio.\n    \n    Args:\n        current_positions: Current portfolio positions\n        quote_client: Tiger SDK quote client\n        dry_run: Use placeholder data\n        \n    Returns:\n        Tuple of (should_rebalance, new_selection_result)\n    \"\"\"\n    selector = RiskIntegratedSelector()\n    \n    # Check if rebalance needed\n    should_rebalance, analysis = selector.check_rebalance_needed(current_positions)\n    \n    # Generate new selection if rebalancing recommended\n    if should_rebalance:\n        new_selection = selector.run_integrated_selection(\n            quote_client, \n            current_positions=current_positions, \n            dry_run=dry_run\n        )\n        return True, new_selection\n    else:\n        return False, SelectionResult(selected_symbols=[], position_weights={})\n\n\ndef validate_portfolio_risk_compliance(positions: Dict[str, float],\n                                     quote_client,\n                                     dry_run: bool = False) -> Dict[str, Any]:\n    \"\"\"\n    Validate if current portfolio meets risk compliance requirements.\n    \n    Args:\n        positions: Current portfolio positions\n        quote_client: Tiger SDK quote client\n        dry_run: Use placeholder data\n        \n    Returns:\n        Risk compliance validation results\n    \"\"\"\n    if not HAS_ALL_MODULES:\n        return {"error": "Enhanced modules not available"}\n    \n    selector = RiskIntegratedSelector()\n    \n    # Load market data for positions\n    symbols = list(positions.keys())\n    market_data = selector._load_market_data(quote_client, symbols, dry_run)\n    \n    if not market_data:\n        return {"error": "Could not load market data"}\n    \n    # Calculate risk metrics\n    risk_metrics = selector.risk_engine._calculate_risk_metrics(\n        symbols, market_data, None, None\n    )\n    \n    # Validate portfolio\n    validation_result = selector.risk_engine.validate_portfolio_risk(\n        positions, market_data, risk_metrics\n    )\n    \n    return validation_result