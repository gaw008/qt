#!/usr/bin/env python3
"""
Test script for the multi-factor scoring system.

This script validates that all factor modules work together and demonstrates
the scoring engine functionality with sample data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Import the scoring system
from bot.scoring_engine import MultiFactorScoringEngine, FactorWeights

def generate_sample_data(symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], 
                        days=100, start_price=100):
    """Generate sample OHLCV data for testing."""
    data = {}
    
    for symbol in symbols:
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price data with random walk
        np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
        returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
        
        prices = [start_price * (1 + np.random.uniform(-0.1, 0.1))]  # Starting price with noise
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.01, days)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.01, days)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        
        # Ensure OHLC relationships are correct
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        volumes = np.random.lognormal(15, 1, days).astype(int)  # Realistic volume distribution
        
        df = pd.DataFrame({
            'time': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        df.set_index('time', inplace=True)
        
        # Add some fundamental data for valuation factors
        df['market_cap'] = closes * 1e9  # Simplified market cap
        df['total_debt'] = df['market_cap'] * np.random.uniform(0.1, 0.3, days)
        df['cash_equiv'] = df['market_cap'] * np.random.uniform(0.05, 0.15, days)
        df['ebitda_ttm'] = df['market_cap'] * np.random.uniform(0.05, 0.20, days)
        df['revenue_ttm'] = df['ebitda_ttm'] * np.random.uniform(3, 8, days)
        df['book_equity'] = df['market_cap'] * np.random.uniform(0.3, 0.8, days)
        df['fcf_ttm'] = df['ebitda_ttm'] * np.random.uniform(0.6, 1.2, days)
        
        data[symbol] = df
    
    return data

def test_factor_modules():
    """Test individual factor modules."""
    print("Testing individual factor modules...")
    
    # Generate sample data
    data = generate_sample_data(symbols=['TEST'], days=50)
    test_df = data['TEST']
    
    try:
        # Test momentum factors
        print("- Testing momentum factors...")
        from bot.factors.momentum_factors import momentum_features
        mom_result = momentum_features(test_df)
        assert 'momentum_score' in mom_result.columns
        print("  ✓ Momentum factors working")
        
        # Test technical factors
        print("- Testing technical factors...")
        from bot.factors.technical_factors import technical_features
        tech_result = technical_features(test_df)
        assert 'technical_score' in tech_result.columns
        print("  ✓ Technical factors working")
        
        # Test volume factors  
        print("- Testing volume factors...")
        from bot.factors.volume_factors import volume_features
        vol_result = volume_features(test_df)
        assert 'vol_score' in vol_result.columns
        print("  ✓ Volume factors working")
        
        # Test valuation factors
        print("- Testing valuation factors...")
        from bot.factors.valuation import valuation_score
        val_result = valuation_score(test_df.iloc[[-1]])
        assert 'ValuationScore' in val_result.columns
        print("  ✓ Valuation factors working")
        
        # Test market factors
        print("- Testing market sentiment factors...")
        from bot.factors.market_factors import market_sentiment_features
        market_result = market_sentiment_features({'TEST': test_df}, symbol='TEST')
        assert 'market_sentiment_score' in market_result.columns
        print("  ✓ Market sentiment factors working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing factor modules: {e}")
        return False

def test_scoring_engine():
    """Test the comprehensive scoring engine."""
    print("\nTesting scoring engine...")
    
    try:
        # Generate sample data for multiple symbols
        data = generate_sample_data(days=60)
        print(f"- Generated sample data for {len(data)} symbols")
        
        # Initialize scoring engine with custom weights
        weights = FactorWeights(
            valuation_weight=0.3,
            momentum_weight=0.25, 
            technical_weight=0.25,
            volume_weight=0.1,
            market_sentiment_weight=0.1,
            enable_dynamic_weights=False  # Disable for testing
        )
        
        engine = MultiFactorScoringEngine(weights)
        print("- Scoring engine initialized")
        
        # Calculate composite scores
        print("- Calculating composite scores...")
        result = engine.calculate_composite_scores(data)
        
        if result.scores.empty:
            print("  ❌ No scores generated")
            return False
            
        print(f"  ✓ Generated scores for {len(result.scores)} symbols")
        print(f"  ✓ Weights used: {result.weights_used}")
        
        # Test score explanations
        print("- Generating score explanations...")
        explanation = engine.explain_scores(result, top_n=3)
        
        if explanation:
            print(f"  ✓ Top stocks: {[s['symbol'] for s in explanation['top_stocks']]}")
            print(f"  ✓ Score statistics: mean={explanation['score_statistics']['mean']:.3f}")
        
        # Test trading signals
        print("- Generating trading signals...")
        signals = engine.get_trading_signals(result, buy_threshold=0.6, sell_threshold=0.4)
        
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        print(f"  ✓ Buy signals: {len(buy_signals)} stocks")
        print(f"  ✓ Sell signals: {len(sell_signals)} stocks")
        
        if len(buy_signals) > 0:
            print(f"    Top buy: {buy_signals.iloc[0]['symbol']} (score: {buy_signals.iloc[0]['composite_score']:.3f})")
        
        # Test correlation analysis
        if not result.factor_correlations.empty:
            print("- Factor correlation analysis:")
            correlations = result.factor_correlations
            print(f"  ✓ Correlation matrix shape: {correlations.shape}")
            
            # Find highest correlation
            mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
            correlations_masked = correlations.where(mask)
            max_corr_idx = np.unravel_index(np.nanargmax(np.abs(correlations_masked.values)), 
                                          correlations_masked.shape)
            if max_corr_idx[0] < len(correlations.index) and max_corr_idx[1] < len(correlations.columns):
                factor1 = correlations.index[max_corr_idx[0]]
                factor2 = correlations.columns[max_corr_idx[1]]
                max_corr = correlations_masked.iloc[max_corr_idx[0], max_corr_idx[1]]
                print(f"    Highest correlation: {factor1} vs {factor2} = {max_corr:.3f}")
        
        print("  ✓ Scoring engine test completed successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing scoring engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_persistence():
    """Test saving and loading configuration."""
    print("\nTesting configuration persistence...")
    
    try:
        # Create engine with custom weights
        weights = FactorWeights(
            valuation_weight=0.4,
            momentum_weight=0.2,
            technical_weight=0.2,
            volume_weight=0.1,
            market_sentiment_weight=0.1
        )
        
        engine = MultiFactorScoringEngine(weights)
        
        # Save configuration
        config_path = "test_scoring_config.json"
        engine.save_configuration(config_path)
        print("- Configuration saved")
        
        # Load configuration
        loaded_engine = MultiFactorScoringEngine.load_configuration(config_path)
        print("- Configuration loaded")
        
        # Verify weights match
        original_weights = engine.weights
        loaded_weights = loaded_engine.weights
        
        assert abs(original_weights.valuation_weight - loaded_weights.valuation_weight) < 1e-6
        assert abs(original_weights.momentum_weight - loaded_weights.momentum_weight) < 1e-6
        
        print("  ✓ Configuration persistence test passed")
        
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing configuration persistence: {e}")
        return False

def run_comprehensive_example():
    """Run a comprehensive example of the scoring system."""
    print("\n" + "="*60)
    print("COMPREHENSIVE SCORING SYSTEM EXAMPLE")
    print("="*60)
    
    try:
        # Generate realistic market data
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        data = generate_sample_data(symbols, days=120, start_price=150)
        print(f"Generated market data for {len(symbols)} symbols over 120 days")
        
        # Create scoring engine with balanced weights
        weights = FactorWeights(
            valuation_weight=0.25,
            volume_weight=0.15,
            momentum_weight=0.25,
            technical_weight=0.25,
            market_sentiment_weight=0.10,
            enable_dynamic_weights=True,
            high_correlation_threshold=0.75
        )
        
        engine = MultiFactorScoringEngine(weights)
        print("Initialized multi-factor scoring engine")
        
        # Calculate scores
        result = engine.calculate_composite_scores(data)
        print(f"Calculated composite scores for {len(result.scores)} symbols")
        
        # Display results
        print("\nTop 5 Ranked Stocks:")
        top_5 = result.scores.nsmallest(5, 'rank')[['symbol', 'composite_score', 'rank', 'percentile']]
        for _, row in top_5.iterrows():
            print(f"  {row['symbol']}: Score={row['composite_score']:.3f}, "
                  f"Rank={int(row['rank'])}, Percentile={row['percentile']:.1%}")
        
        print(f"\nFactor Weights Used:")
        for factor, weight in result.weights_used.items():
            print(f"  {factor}: {weight:.1%}")
        
        # Generate detailed explanation
        explanation = engine.explain_scores(result, top_n=3)
        
        print(f"\nScore Statistics:")
        stats = explanation['score_statistics']
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Trading signals
        signals = engine.get_trading_signals(result, buy_threshold=0.7, sell_threshold=0.3, max_positions=3)
        
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        print(f"\nTrading Signals Generated:")
        print(f"  Buy candidates ({len(buy_signals)}): {', '.join(buy_signals['symbol'].tolist())}")
        print(f"  Sell candidates ({len(sell_signals)}): {', '.join(sell_signals['symbol'].tolist())}")
        
        # Factor correlations
        if not result.factor_correlations.empty:
            print(f"\nFactor Correlation Summary:")
            correlations = result.factor_correlations
            for i, factor1 in enumerate(correlations.columns):
                for j, factor2 in enumerate(correlations.columns):
                    if i < j:  # Upper triangle only
                        corr = correlations.iloc[i, j]
                        if abs(corr) > 0.5:  # Only show significant correlations
                            print(f"  {factor1} ↔ {factor2}: {corr:.3f}")
        
        print("\n✓ Comprehensive example completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in comprehensive example: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("Multi-Factor Scoring System Test Suite")
    print("="*50)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    all_tests_passed = True
    
    # Run individual tests
    all_tests_passed &= test_factor_modules()
    all_tests_passed &= test_scoring_engine() 
    all_tests_passed &= test_configuration_persistence()
    
    # Run comprehensive example
    if all_tests_passed:
        run_comprehensive_example()
    
    print(f"\n{'='*50}")
    if all_tests_passed:
        print("🎉 All tests passed! The multi-factor scoring system is working correctly.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)