#!/usr/bin/env python3
"""
Comprehensive demonstration of the multi-factor scoring system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bot.scoring_engine import MultiFactorScoringEngine, FactorWeights

def generate_realistic_market_data(symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA'], days=90):
    """Generate realistic market data with different characteristics."""
    data = {}
    
    # Define stock characteristics
    stock_profiles = {
        'AAPL': {'volatility': 0.025, 'trend': 0.0008, 'pe_base': 25},
        'GOOGL': {'volatility': 0.030, 'trend': 0.0006, 'pe_base': 22},
        'MSFT': {'volatility': 0.022, 'trend': 0.0010, 'pe_base': 28},
        'TSLA': {'volatility': 0.045, 'trend': 0.0005, 'pe_base': 50},
        'AMZN': {'volatility': 0.035, 'trend': 0.0007, 'pe_base': 45},
        'NVDA': {'volatility': 0.040, 'trend': 0.0012, 'pe_base': 35}
    }
    
    for symbol in symbols:
        profile = stock_profiles.get(symbol, {'volatility': 0.025, 'trend': 0.0005, 'pe_base': 30})
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate price data with specific characteristics
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(profile['trend'], profile['volatility'], days)
        
        prices = [150 + np.random.uniform(-20, 20)]  # Starting price
        for ret in returns[1:]:
            prices.append(max(1, prices[-1] * (1 + ret)))  # Prevent negative prices
        
        closes = np.array(prices)
        
        # Create realistic OHLC
        daily_volatility = np.random.uniform(0.005, 0.02, days)
        highs = closes * (1 + daily_volatility)
        lows = closes * (1 - daily_volatility)
        opens = np.concatenate([[closes[0]], closes[:-1]]) * (1 + np.random.normal(0, 0.005, days))
        
        # Ensure OHLC relationships
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        # Volume with patterns (higher on big moves)
        price_changes = np.abs(np.diff(np.concatenate([[closes[0]], closes])))
        base_volume = np.random.lognormal(16, 0.5, days)
        volume_multiplier = 1 + 2 * price_changes / closes
        volumes = (base_volume * volume_multiplier).astype(int)
        
        df = pd.DataFrame({
            'time': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        df.set_index('time', inplace=True)
        
        # Add fundamental data based on stock characteristics
        market_cap_base = np.random.uniform(500e9, 3000e9)  # 500B to 3T
        df['market_cap'] = market_cap_base * (closes / closes[0])
        df['total_debt'] = df['market_cap'] * np.random.uniform(0.05, 0.25)
        df['cash_equiv'] = df['market_cap'] * np.random.uniform(0.08, 0.20)
        
        # Revenue and profitability
        revenue_multiple = np.random.uniform(0.8, 1.5)
        df['revenue_ttm'] = df['market_cap'] * revenue_multiple
        df['ebitda_ttm'] = df['revenue_ttm'] * np.random.uniform(0.15, 0.35)
        df['fcf_ttm'] = df['ebitda_ttm'] * np.random.uniform(0.6, 1.1)
        df['book_equity'] = df['market_cap'] / profile['pe_base'] * np.random.uniform(0.8, 1.2)
        
        data[symbol] = df
    
    return data

def analyze_factor_performance(engine, result):
    """Analyze the performance and characteristics of each factor."""
    print("\nFactor Analysis:")
    print("-" * 50)
    
    # Factor weights
    print("Factor Weights:")
    for factor, weight in result.weights_used.items():
        print(f"  {factor:20}: {weight:8.1%}")
    
    # Factor correlations
    if not result.factor_correlations.empty:
        print(f"\nFactor Correlations:")
        corr_matrix = result.factor_correlations
        
        # Show only significant correlations
        for i, factor1 in enumerate(corr_matrix.columns):
            for j, factor2 in enumerate(corr_matrix.columns):
                if i < j:  # Upper triangle only
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.3:  # Show correlations > 30%
                        status = "HIGH" if abs(corr) > 0.7 else "MEDIUM"
                        print(f"  {factor1:12} <-> {factor2:12}: {corr:6.3f} ({status})")
    
    # Factor contributions for top stocks
    if not result.factor_contributions.empty:
        print(f"\nTop 3 Stock Factor Breakdown:")
        top_3 = result.scores.nsmallest(3, 'rank')
        
        for _, stock_row in top_3.iterrows():
            symbol = stock_row['symbol']
            score = stock_row['composite_score']
            
            contrib_row = result.factor_contributions[
                result.factor_contributions['symbol'] == symbol
            ]
            
            print(f"\n  {symbol} (Score: {score:.3f}):")
            if not contrib_row.empty:
                for factor in result.weights_used.keys():
                    if factor in contrib_row.columns:
                        contrib = contrib_row[factor].iloc[0]
                        print(f"    {factor:20}: {contrib:8.3f}")

def generate_trading_recommendation(engine, result):
    """Generate detailed trading recommendations."""
    print("\nTrading Recommendations:")
    print("-" * 50)
    
    # Different signal thresholds
    conservative_signals = engine.get_trading_signals(
        result, buy_threshold=0.8, sell_threshold=0.2, max_positions=2
    )
    moderate_signals = engine.get_trading_signals(
        result, buy_threshold=0.7, sell_threshold=0.3, max_positions=3
    )
    aggressive_signals = engine.get_trading_signals(
        result, buy_threshold=0.6, sell_threshold=0.4, max_positions=5
    )
    
    strategies = [
        ("Conservative", conservative_signals),
        ("Moderate", moderate_signals),
        ("Aggressive", aggressive_signals)
    ]
    
    for strategy_name, signals in strategies:
        buys = signals[signals['signal'] == 1]
        sells = signals[signals['signal'] == -1]
        
        print(f"\n{strategy_name} Strategy:")
        print(f"  Buy Recommendations ({len(buys)}):")
        for _, row in buys.iterrows():
            print(f"    {row['symbol']:6} - Score: {row['composite_score']:6.3f}, "
                  f"Rank: {int(row['rank']):2}, Percentile: {row['percentile']:5.1%}")
        
        print(f"  Sell Recommendations ({len(sells)}):")
        for _, row in sells.iterrows():
            print(f"    {row['symbol']:6} - Score: {row['composite_score']:6.3f}, "
                  f"Rank: {int(row['rank']):2}, Percentile: {row['percentile']:5.1%}")

def demonstrate_custom_weights():
    """Demonstrate different weight configurations."""
    print("\nCustom Weight Configurations:")
    print("-" * 50)
    
    # Generate sample data
    data = generate_realistic_market_data(symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA'], days=60)
    
    # Different weight configurations
    weight_configs = {
        "Value-Focused": FactorWeights(
            valuation_weight=0.5,
            momentum_weight=0.15,
            technical_weight=0.15,
            volume_weight=0.1,
            market_sentiment_weight=0.1
        ),
        "Momentum-Focused": FactorWeights(
            valuation_weight=0.1,
            momentum_weight=0.4,
            technical_weight=0.3,
            volume_weight=0.1,
            market_sentiment_weight=0.1
        ),
        "Technical-Focused": FactorWeights(
            valuation_weight=0.15,
            momentum_weight=0.2,
            technical_weight=0.45,
            volume_weight=0.1,
            market_sentiment_weight=0.1
        ),
        "Balanced": FactorWeights(
            valuation_weight=0.25,
            momentum_weight=0.2,
            technical_weight=0.25,
            volume_weight=0.15,
            market_sentiment_weight=0.15
        )
    }
    
    results = {}
    for config_name, weights in weight_configs.items():
        engine = MultiFactorScoringEngine(weights)
        result = engine.calculate_composite_scores(data)
        results[config_name] = result
        
        print(f"\n{config_name} Configuration:")
        top_stock = result.scores.nsmallest(1, 'rank').iloc[0]
        print(f"  Top Pick: {top_stock['symbol']} (Score: {top_stock['composite_score']:.3f})")
        
        # Show ranking differences
        ranking = result.scores.sort_values('rank')[['symbol', 'rank']].reset_index(drop=True)
        ranking_str = " > ".join([f"{row['symbol']}({int(row['rank'])})" for _, row in ranking.iterrows()])
        print(f"  Ranking: {ranking_str}")

def main():
    """Main demonstration function."""
    print("Multi-Factor Scoring System Comprehensive Demonstration")
    print("=" * 60)
    
    # Generate realistic market data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
    print(f"Generating realistic market data for {len(symbols)} stocks...")
    data = generate_realistic_market_data(symbols, days=120)
    
    # Initialize scoring engine with balanced weights
    weights = FactorWeights(
        valuation_weight=0.25,
        volume_weight=0.15,
        momentum_weight=0.25,
        technical_weight=0.25,
        market_sentiment_weight=0.10,
        enable_dynamic_weights=False,  # Disable for demonstration
        high_correlation_threshold=0.75
    )
    
    engine = MultiFactorScoringEngine(weights)
    print("Multi-factor scoring engine initialized")
    
    # Calculate composite scores
    print("Calculating composite scores...")
    result = engine.calculate_composite_scores(data)
    
    # Display main results
    print(f"\nComposite Scoring Results:")
    print("-" * 50)
    print(f"Total stocks analyzed: {len(result.scores)}")
    
    # Show all stocks ranked
    print(f"\nComplete Stock Rankings:")
    full_ranking = result.scores.sort_values('rank')[['symbol', 'composite_score', 'rank', 'percentile']]
    for i, (_, row) in enumerate(full_ranking.iterrows(), 1):
        print(f"  {i}. {row['symbol']:6} - Score: {row['composite_score']:7.3f}, "
              f"Percentile: {row['percentile']:5.1%}")
    
    # Detailed analysis
    analyze_factor_performance(engine, result)
    
    # Trading recommendations
    generate_trading_recommendation(engine, result)
    
    # Score explanations
    explanation = engine.explain_scores(result, top_n=3)
    
    print(f"\nScore Distribution Statistics:")
    print("-" * 50)
    stats = explanation['score_statistics']
    print(f"  Mean:     {stats['mean']:8.3f}")
    print(f"  Std Dev:  {stats['std']:8.3f}")
    print(f"  Range:    [{stats['min']:6.3f}, {stats['max']:6.3f}]")
    print(f"  Median:   {stats['median']:8.3f}")
    print(f"  Skewness: {stats['skewness']:8.3f}")
    
    # Demonstrate different weight configurations
    demonstrate_custom_weights()
    
    print(f"\n{'=' * 60}")
    print("Demonstration completed successfully!")
    print("The multi-factor scoring system provides comprehensive")
    print("stock analysis combining valuation, momentum, technical,")
    print("volume, and market sentiment factors.")

if __name__ == "__main__":
    main()