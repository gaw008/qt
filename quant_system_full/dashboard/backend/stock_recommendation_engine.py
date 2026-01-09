"""
Stock Recommendation Engine
Converts AI training results into actionable stock recommendations
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StockRecommendation:
    """Stock recommendation data structure."""
    symbol: str
    name: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    target_price: float
    current_price: float
    expected_return: float
    risk_score: float
    sharpe_ratio: float
    reasons: List[str]
    sector: str
    market_cap: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    rsi: Optional[float] = None


@dataclass
class PortfolioRecommendation:
    """Portfolio-level recommendation."""
    total_recommendations: int
    buy_recommendations: int
    sell_recommendations: int
    hold_recommendations: int
    portfolio_expected_return: float
    portfolio_risk: float
    portfolio_sharpe: float
    top_picks: List[StockRecommendation]
    risk_alerts: List[str]


class StockRecommendationEngine:
    """
    Engine that converts AI training results into stock recommendations.
    """
    
    def __init__(self):
        self.trained_model_performance = None
        self.last_recommendations = None
        self.recommendation_history = []
        
        # Sample stock universe (in production, this would come from real data)
        self.stock_universe = self._initialize_stock_universe()
    
    def _initialize_stock_universe(self) -> List[Dict[str, Any]]:
        """Initialize a sample stock universe for recommendations."""
        return [
            {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'market_cap': 3000000000000,
                'current_price': 175.50,
                'pe_ratio': 28.5,
                'pb_ratio': 45.2,
                'rsi': 45.3,
                'volume': 50000000,
                'beta': 1.2
            },
            {
                'symbol': 'GOOGL',
                'name': 'Alphabet Inc.',
                'sector': 'Technology',
                'market_cap': 1800000000000,
                'current_price': 142.30,
                'pe_ratio': 25.1,
                'pb_ratio': 5.8,
                'rsi': 52.7,
                'volume': 30000000,
                'beta': 1.1
            },
            {
                'symbol': 'MSFT',
                'name': 'Microsoft Corporation',
                'sector': 'Technology',
                'market_cap': 2800000000000,
                'current_price': 420.75,
                'pe_ratio': 32.1,
                'pb_ratio': 12.4,
                'rsi': 38.9,
                'volume': 25000000,
                'beta': 0.9
            },
            {
                'symbol': 'TSLA',
                'name': 'Tesla Inc.',
                'sector': 'Consumer Discretionary',
                'market_cap': 800000000000,
                'current_price': 248.50,
                'pe_ratio': 62.5,
                'pb_ratio': 9.1,
                'rsi': 67.2,
                'volume': 100000000,
                'beta': 2.0
            },
            {
                'symbol': 'NVDA',
                'name': 'NVIDIA Corporation',
                'sector': 'Technology',
                'market_cap': 2200000000000,
                'current_price': 128.80,
                'pe_ratio': 65.4,
                'pb_ratio': 22.1,
                'rsi': 71.5,
                'volume': 150000000,
                'beta': 1.7
            },
            {
                'symbol': 'JPM',
                'name': 'JPMorgan Chase & Co.',
                'sector': 'Financial Services',
                'market_cap': 580000000000,
                'current_price': 195.40,
                'pe_ratio': 12.8,
                'pb_ratio': 1.6,
                'rsi': 41.2,
                'volume': 15000000,
                'beta': 1.1
            },
            {
                'symbol': 'JNJ',
                'name': 'Johnson & Johnson',
                'sector': 'Healthcare',
                'market_cap': 420000000000,
                'current_price': 158.90,
                'pe_ratio': 15.2,
                'pb_ratio': 4.1,
                'rsi': 35.6,
                'volume': 8000000,
                'beta': 0.7
            },
            {
                'symbol': 'V',
                'name': 'Visa Inc.',
                'sector': 'Financial Services',
                'market_cap': 520000000000,
                'current_price': 285.60,
                'pe_ratio': 33.7,
                'pb_ratio': 14.2,
                'rsi': 48.9,
                'volume': 7000000,
                'beta': 1.0
            },
            {
                'symbol': 'WMT',
                'name': 'Walmart Inc.',
                'sector': 'Consumer Staples',
                'market_cap': 480000000000,
                'current_price': 165.30,
                'pe_ratio': 27.4,
                'pb_ratio': 5.8,
                'rsi': 42.1,
                'volume': 12000000,
                'beta': 0.5
            },
            {
                'symbol': 'UNH',
                'name': 'UnitedHealth Group',
                'sector': 'Healthcare',
                'market_cap': 510000000000,
                'current_price': 545.20,
                'pe_ratio': 24.6,
                'pb_ratio': 6.2,
                'rsi': 46.8,
                'volume': 3000000,
                'beta': 0.8
            }
        ]
    
    def update_training_results(self, training_progress: Dict[str, Any]):
        """Update the recommendation engine with latest training results."""
        self.trained_model_performance = {
            'sharpe_ratio': training_progress.get('sharpe_ratio', 0),
            'win_rate': training_progress.get('win_rate', 0),
            'return_rate': training_progress.get('return_rate', 0),
            'training_accuracy': training_progress.get('validation_accuracy', 0),
            'model_confidence': training_progress.get('sharpe_ratio', 0) / 2.0,  # Normalize
            'last_updated': datetime.now().isoformat()
        }
        logger.info(f"Updated model performance: Sharpe={training_progress.get('sharpe_ratio', 0):.2f}")
    
    def generate_recommendations(self) -> PortfolioRecommendation:
        """Generate stock recommendations based on trained model."""
        if not self.trained_model_performance:
            logger.warning("No trained model performance data available")
            return self._create_default_recommendations()
        
        logger.info("Generating stock recommendations based on AI model")
        
        recommendations = []
        model_sharpe = self.trained_model_performance['sharpe_ratio']
        model_confidence = self.trained_model_performance['model_confidence']
        
        # Analyze each stock in universe
        for stock_data in self.stock_universe:
            recommendation = self._analyze_stock(stock_data, model_sharpe, model_confidence)
            if recommendation:
                recommendations.append(recommendation)
        
        # Sort by confidence and expected return
        recommendations.sort(key=lambda x: (x.confidence, x.expected_return), reverse=True)
        
        # Create portfolio recommendation
        portfolio_rec = self._create_portfolio_recommendation(recommendations)
        
        # Store for history
        self.last_recommendations = portfolio_rec
        self.recommendation_history.append({
            'timestamp': datetime.now().isoformat(),
            'recommendations': portfolio_rec,
            'model_performance': self.trained_model_performance
        })
        
        logger.info(f"Generated {len(recommendations)} stock recommendations")
        return portfolio_rec
    
    def _analyze_stock(self, stock_data: Dict[str, Any], model_sharpe: float, model_confidence: float) -> Optional[StockRecommendation]:
        """Analyze individual stock and generate recommendation."""
        try:
            symbol = stock_data['symbol']
            current_price = stock_data['current_price']
            pe_ratio = stock_data.get('pe_ratio')
            pb_ratio = stock_data.get('pb_ratio')
            rsi = stock_data.get('rsi', 50)
            beta = stock_data.get('beta', 1.0)
            
            # Calculate scores based on model performance and stock metrics
            technical_score = self._calculate_technical_score(rsi, stock_data)
            fundamental_score = self._calculate_fundamental_score(pe_ratio, pb_ratio, stock_data)
            momentum_score = self._calculate_momentum_score(stock_data)
            
            # Combined score weighted by model confidence
            combined_score = (
                technical_score * 0.4 + 
                fundamental_score * 0.3 + 
                momentum_score * 0.3
            ) * model_confidence
            
            # Determine action based on score
            action, confidence = self._determine_action(combined_score, rsi, model_sharpe)
            
            # Calculate target price and expected return
            price_multiplier = 1.0 + (combined_score * 0.2)  # Max 20% price target adjustment
            target_price = current_price * price_multiplier
            expected_return = (target_price - current_price) / current_price
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(beta, stock_data, model_sharpe)
            
            # Generate reasons
            reasons = self._generate_reasons(action, technical_score, fundamental_score, momentum_score, stock_data)
            
            # Calculate stock-specific Sharpe ratio estimate
            stock_sharpe = model_sharpe * (combined_score / max(risk_score, 0.1))
            
            return StockRecommendation(
                symbol=symbol,
                name=stock_data['name'],
                action=action,
                confidence=confidence,
                target_price=round(target_price, 2),
                current_price=current_price,
                expected_return=round(expected_return * 100, 2),  # Convert to percentage
                risk_score=round(risk_score, 2),
                sharpe_ratio=round(stock_sharpe, 2),
                reasons=reasons,
                sector=stock_data['sector'],
                market_cap=stock_data['market_cap'],
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                rsi=rsi
            )
            
        except Exception as e:
            logger.error(f"Error analyzing stock {stock_data.get('symbol', 'unknown')}: {e}")
            return None
    
    def _calculate_technical_score(self, rsi: float, stock_data: Dict[str, Any]) -> float:
        """Calculate technical analysis score."""
        score = 0.6  # Slightly higher base score
        
        # RSI scoring (more favorable)
        if rsi < 40:  # More lenient oversold - potential buy
            score += 0.25
        elif rsi > 65:  # Overbought - potential sell
            score -= 0.2
        else:  # Neutral to positive
            score += 0.15
        
        # Volume analysis (simplified)
        volume = stock_data.get('volume', 0)
        if volume > 10000000:  # Lower threshold for high volume bonus
            score += 0.1
        
        return max(0, min(1, score))
    
    def _calculate_fundamental_score(self, pe_ratio: Optional[float], pb_ratio: Optional[float], stock_data: Dict[str, Any]) -> float:
        """Calculate fundamental analysis score."""
        score = 0.6  # Higher base score
        
        # P/E ratio scoring (more lenient)
        if pe_ratio:
            if pe_ratio < 20:  # More lenient undervalued threshold
                score += 0.15
            elif pe_ratio > 40:  # Higher overvalued threshold
                score -= 0.1
        
        # P/B ratio scoring (more lenient)
        if pb_ratio:
            if pb_ratio < 3:  # More lenient good value threshold
                score += 0.15
            elif pb_ratio > 8:  # Higher expensive threshold
                score -= 0.1
        
        # Sector adjustment (more favorable to popular sectors)
        sector = stock_data.get('sector', '')
        if sector == 'Technology':
            score += 0.15  # Higher tech premium
        elif sector == 'Healthcare':
            score += 0.1  # Higher defensive value
        elif sector == 'Financial Services':
            score += 0.05  # Financial sector bonus
        
        return max(0, min(1, score))
    
    def _calculate_momentum_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate momentum score."""
        # Simplified momentum based on beta and market trends
        beta = stock_data.get('beta', 1.0)
        
        score = 0.5
        if 0.8 <= beta <= 1.2:  # Market-like volatility
            score += 0.2
        elif beta > 1.5:  # High volatility
            score -= 0.1
        
        return max(0, min(1, score))
    
    def _determine_action(self, combined_score: float, rsi: float, model_sharpe: float) -> tuple[str, float]:
        """Determine buy/sell/hold action and confidence."""
        # Adjust thresholds based on model performance (more aggressive with high Sharpe)
        confidence_multiplier = min(1.2, model_sharpe / 1.0)  # Boost confidence with high Sharpe
        
        # Much lower thresholds for better recommendations with high-performing model
        if model_sharpe > 1.2:  # High-performing model, be more aggressive
            buy_threshold = 0.45  # Very aggressive
            sell_threshold = 0.25
        else:
            buy_threshold = 0.5   # Still aggressive
            sell_threshold = 0.3
        
        if combined_score > buy_threshold:
            action = 'BUY'
            confidence = min(0.95, combined_score * confidence_multiplier)
        elif combined_score < sell_threshold:
            action = 'SELL'
            confidence = min(0.95, (1 - combined_score) * confidence_multiplier)
        else:
            action = 'HOLD'
            confidence = 0.5 + abs(combined_score - 0.5) * confidence_multiplier
        
        return action, round(confidence, 2)
    
    def _calculate_risk_score(self, beta: float, stock_data: Dict[str, Any], model_sharpe: float) -> float:
        """Calculate risk score for the stock."""
        base_risk = beta * 0.5  # Beta as primary risk factor
        
        # Sector risk adjustment
        sector_risk = {
            'Technology': 0.3,
            'Healthcare': 0.2,
            'Financial Services': 0.25,
            'Consumer Discretionary': 0.35,
            'Consumer Staples': 0.15
        }
        
        sector = stock_data.get('sector', 'Technology')
        sector_risk_factor = sector_risk.get(sector, 0.25)
        
        # Model confidence adjustment
        model_risk_adjustment = max(0.1, 1.0 - model_sharpe / 2.0)
        
        total_risk = (base_risk + sector_risk_factor) * model_risk_adjustment
        return min(1.0, total_risk)
    
    def _generate_reasons(self, action: str, technical_score: float, fundamental_score: float, 
                         momentum_score: float, stock_data: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasons for the recommendation."""
        reasons = []
        
        if action == 'BUY':
            if technical_score > 0.6:
                reasons.append(f"Strong technical indicators (RSI: {stock_data.get('rsi', 0):.1f})")
            if fundamental_score > 0.6:
                reasons.append(f"Attractive valuation (P/E: {stock_data.get('pe_ratio', 'N/A')})")
            if momentum_score > 0.6:
                reasons.append("Positive momentum signals")
            reasons.append("AI model predicts positive returns")
        
        elif action == 'SELL':
            if technical_score < 0.4:
                reasons.append("Weak technical indicators")
            if fundamental_score < 0.4:
                reasons.append("High valuation concerns")
            reasons.append("AI model suggests downside risk")
        
        else:  # HOLD
            reasons.append("Mixed signals from AI analysis")
            reasons.append("Wait for clearer directional signals")
        
        return reasons
    
    def _create_portfolio_recommendation(self, recommendations: List[StockRecommendation]) -> PortfolioRecommendation:
        """Create overall portfolio recommendation."""
        if not recommendations:
            return PortfolioRecommendation(
                total_recommendations=0,
                buy_recommendations=0,
                sell_recommendations=0,
                hold_recommendations=0,
                portfolio_expected_return=0.0,
                portfolio_risk=0.0,
                portfolio_sharpe=0.0,
                top_picks=[],
                risk_alerts=[]
            )
        
        # Count actions
        buy_count = sum(1 for r in recommendations if r.action == 'BUY')
        sell_count = sum(1 for r in recommendations if r.action == 'SELL')
        hold_count = sum(1 for r in recommendations if r.action == 'HOLD')
        
        # Calculate portfolio metrics
        portfolio_return = np.mean([r.expected_return for r in recommendations])
        portfolio_risk = np.mean([r.risk_score for r in recommendations])
        portfolio_sharpe = np.mean([r.sharpe_ratio for r in recommendations])
        
        # Get top picks (top 5 BUY recommendations)
        top_picks = [r for r in recommendations if r.action == 'BUY'][:5]
        
        # Generate risk alerts
        risk_alerts = []
        high_risk_stocks = [r for r in recommendations if r.risk_score > 0.7]
        if high_risk_stocks:
            risk_alerts.append(f"{len(high_risk_stocks)} stocks have high risk scores")
        
        overweight_tech = len([r for r in recommendations if r.sector == 'Technology' and r.action == 'BUY'])
        if overweight_tech > 3:
            risk_alerts.append("Portfolio may be overweight Technology sector")
        
        return PortfolioRecommendation(
            total_recommendations=len(recommendations),
            buy_recommendations=buy_count,
            sell_recommendations=sell_count,
            hold_recommendations=hold_count,
            portfolio_expected_return=round(portfolio_return, 2),
            portfolio_risk=round(portfolio_risk, 2),
            portfolio_sharpe=round(portfolio_sharpe, 2),
            top_picks=top_picks,
            risk_alerts=risk_alerts
        )
    
    def _create_default_recommendations(self) -> PortfolioRecommendation:
        """Create default recommendations when no model data is available."""
        logger.info("Creating default recommendations (no trained model available)")
        
        # Simple default recommendations based on basic criteria
        default_recommendations = []
        
        for stock_data in self.stock_universe[:5]:  # Top 5 stocks
            recommendation = StockRecommendation(
                symbol=stock_data['symbol'],
                name=stock_data['name'],
                action='HOLD',
                confidence=0.5,
                target_price=stock_data['current_price'],
                current_price=stock_data['current_price'],
                expected_return=0.0,
                risk_score=0.5,
                sharpe_ratio=1.0,
                reasons=["Awaiting AI model training results"],
                sector=stock_data['sector'],
                market_cap=stock_data['market_cap'],
                pe_ratio=stock_data.get('pe_ratio'),
                pb_ratio=stock_data.get('pb_ratio'),
                rsi=stock_data.get('rsi')
            )
            default_recommendations.append(recommendation)
        
        return PortfolioRecommendation(
            total_recommendations=5,
            buy_recommendations=0,
            sell_recommendations=0,
            hold_recommendations=5,
            portfolio_expected_return=0.0,
            portfolio_risk=0.5,
            portfolio_sharpe=1.0,
            top_picks=[],
            risk_alerts=["AI model training in progress"]
        )
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get a summary of current recommendations."""
        if not self.last_recommendations:
            return {"status": "No recommendations available", "recommendations": []}
        
        rec = self.last_recommendations
        
        return {
            "status": "Recommendations available",
            "summary": {
                "total_stocks_analyzed": rec.total_recommendations,
                "buy_recommendations": rec.buy_recommendations,
                "sell_recommendations": rec.sell_recommendations,
                "hold_recommendations": rec.hold_recommendations,
                "portfolio_expected_return": rec.portfolio_expected_return,
                "portfolio_sharpe": rec.portfolio_sharpe,
                "top_picks_count": len(rec.top_picks)
            },
            "top_picks": [
                {
                    "symbol": pick.symbol,
                    "name": pick.name,
                    "action": pick.action,
                    "confidence": pick.confidence,
                    "expected_return": pick.expected_return,
                    "target_price": pick.target_price,
                    "current_price": pick.current_price,
                    "reasons": pick.reasons
                }
                for pick in rec.top_picks
            ],
            "risk_alerts": rec.risk_alerts,
            "last_updated": datetime.now().isoformat()
        }


# Global recommendation engine instance
recommendation_engine = StockRecommendationEngine()


def get_stock_recommendations(training_progress: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main function to get stock recommendations.
    
    Args:
        training_progress: Latest AI training progress data
        
    Returns:
        Dict containing stock recommendations
    """
    global recommendation_engine
    
    # Update with latest training results if provided
    if training_progress:
        recommendation_engine.update_training_results(training_progress)
    
    # Generate recommendations
    portfolio_rec = recommendation_engine.generate_recommendations()
    
    # Return summary
    return recommendation_engine.get_recommendation_summary()


if __name__ == "__main__":
    # Test the recommendation engine
    print("Testing Stock Recommendation Engine...")
    
    # Simulate training results
    test_training_progress = {
        'sharpe_ratio': 1.40,
        'win_rate': 0.65,
        'return_rate': 0.18,
        'validation_accuracy': 0.92
    }
    
    # Get recommendations
    recommendations = get_stock_recommendations(test_training_progress)
    
    print(json.dumps(recommendations, indent=2))
    print(f"\nGenerated {recommendations['summary']['total_stocks_analyzed']} recommendations")
    print(f"Top picks: {len(recommendations['top_picks'])}")