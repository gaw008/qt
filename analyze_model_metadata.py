#!/usr/bin/env python3
"""
Model Metadata Analysis Script
Analyzes model performance metrics and creates tiered classifications
"""

import json
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

def load_model_metadata(file_path):
    """Load model metadata from JSON file"""
    print(f"Loading model metadata from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['models']

def analyze_accuracy_distribution(models):
    """Analyze the distribution of model accuracies"""
    accuracies = []
    for symbol, model_data in models.items():
        accuracy = model_data['metrics']['direction_accuracy']
        accuracies.append((symbol, accuracy))
    
    # Define accuracy ranges
    ranges = {
        '>70%': 0,
        '60-70%': 0,
        '55-60%': 0,
        '50-55%': 0,
        '<50%': 0
    }
    
    # Count models in each range
    for symbol, accuracy in accuracies:
        if accuracy > 0.70:
            ranges['>70%'] += 1
        elif 0.60 <= accuracy <= 0.70:
            ranges['60-70%'] += 1
        elif 0.55 <= accuracy < 0.60:
            ranges['55-60%'] += 1
        elif 0.50 <= accuracy < 0.55:
            ranges['50-55%'] += 1
        else:
            ranges['<50%'] += 1
    
    return ranges, accuracies

def find_sweet_spot_models(models, min_acc=0.60, max_acc=0.70, top_n=100):
    """Find top N models in the sweet spot accuracy range (60-70%)"""
    sweet_spot_models = []
    
    for symbol, model_data in models.items():
        accuracy = model_data['metrics']['direction_accuracy']
        correlation = model_data['metrics']['correlation']
        
        if min_acc <= accuracy <= max_acc:
            sweet_spot_models.append({
                'symbol': symbol,
                'accuracy': accuracy,
                'correlation': correlation,
                'samples': model_data['samples'],
                'features': model_data['features']
            })
    
    # Sort by accuracy descending
    sweet_spot_models.sort(key=lambda x: x['accuracy'], reverse=True)
    
    return sweet_spot_models[:top_n]

def identify_low_correlation_models(models, correlation_threshold=0.2):
    """Identify models with good accuracy AND low correlation"""
    low_corr_models = []
    
    for symbol, model_data in models.items():
        accuracy = model_data['metrics']['direction_accuracy']
        correlation = abs(model_data['metrics']['correlation'])  # Use absolute value
        
        if accuracy >= 0.60 and correlation <= correlation_threshold:
            low_corr_models.append({
                'symbol': symbol,
                'accuracy': accuracy,
                'correlation': model_data['metrics']['correlation'],  # Keep original sign
                'abs_correlation': correlation
            })
    
    # Sort by accuracy descending
    low_corr_models.sort(key=lambda x: x['accuracy'], reverse=True)
    
    return low_corr_models

def create_tiered_classification(models):
    """Create tiered classification of models"""
    tiers = {
        'Tier 1': [],  # 65-70% accuracy with low correlation
        'Tier 2': [],  # 60-65% accuracy
        'Tier 3': []   # 55-60% accuracy
    }
    
    for symbol, model_data in models.items():
        accuracy = model_data['metrics']['direction_accuracy']
        correlation = abs(model_data['metrics']['correlation'])
        
        model_info = {
            'symbol': symbol,
            'accuracy': accuracy,
            'correlation': model_data['metrics']['correlation'],
            'abs_correlation': correlation,
            'samples': model_data['samples']
        }
        
        if 0.65 <= accuracy <= 0.70 and correlation <= 0.2:
            tiers['Tier 1'].append(model_info)
        elif 0.60 <= accuracy < 0.65:
            tiers['Tier 2'].append(model_info)
        elif 0.55 <= accuracy < 0.60:
            tiers['Tier 3'].append(model_info)
    
    # Sort each tier by accuracy descending
    for tier in tiers.values():
        tier.sort(key=lambda x: x['accuracy'], reverse=True)
    
    return tiers

def print_analysis_results(ranges, sweet_spot_models, low_corr_models, tiers, models):
    """Print comprehensive analysis results"""
    
    print("=" * 80)
    print("MODEL METADATA ANALYSIS RESULTS")
    print("=" * 80)
    
    # 1. Accuracy Distribution
    print("\n1. ACCURACY DISTRIBUTION:")
    print("-" * 40)
    total_models = sum(ranges.values())
    for range_name, count in ranges.items():
        percentage = (count / total_models) * 100
        print(f"{range_name:>8}: {count:>6} models ({percentage:>5.1f}%)")
    print(f"{'Total':>8}: {total_models:>6} models")
    
    # 2. Sweet Spot Models (60-70%)
    print(f"\n2. TOP 100 SWEET SPOT MODELS (60-70% accuracy):")
    print("-" * 40)
    print(f"Found {len(sweet_spot_models)} models in 60-70% range")
    if sweet_spot_models:
        print(f"Top 10 examples:")
        for i, model in enumerate(sweet_spot_models[:10], 1):
            print(f"{i:>2}. {model['symbol']:>6}: {model['accuracy']:.3f} acc, {model['correlation']:>+.3f} corr")
    
    # 3. Low Correlation Models
    print(f"\n3. MODELS WITH GOOD ACCURACY & LOW CORRELATION (|corr| <= 0.2):")
    print("-" * 40)
    print(f"Found {len(low_corr_models)} models with >=60% accuracy and low correlation")
    if low_corr_models:
        print(f"Top 10 examples:")
        for i, model in enumerate(low_corr_models[:10], 1):
            print(f"{i:>2}. {model['symbol']:>6}: {model['accuracy']:.3f} acc, {model['correlation']:>+.3f} corr")
    
    # 4. Tiered Classification
    print(f"\n4. TIERED CLASSIFICATION:")
    print("-" * 40)
    
    for tier_name, tier_models in tiers.items():
        print(f"\n{tier_name}: {len(tier_models)} models")
        
        if tier_name == "Tier 1":
            print("  (65-70% accuracy with |correlation| <= 0.2)")
        elif tier_name == "Tier 2":
            print("  (60-65% accuracy)")
        elif tier_name == "Tier 3":
            print("  (55-60% accuracy)")
        
        if tier_models:
            # Show top 5 examples and stats
            print(f"  Top 5 examples:")
            for i, model in enumerate(tier_models[:5], 1):
                print(f"    {i}. {model['symbol']:>6}: {model['accuracy']:.3f} acc, {model['correlation']:>+.3f} corr")
            
            # Show accuracy range within tier
            accuracies = [m['accuracy'] for m in tier_models]
            print(f"  Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")
            print(f"  Average accuracy: {np.mean(accuracies):.3f}")
    
    # 5. Summary Statistics
    print(f"\n5. SUMMARY STATISTICS:")
    print("-" * 40)
    all_accuracies = []
    all_correlations = []
    
    # Calculate overall stats from all models
    for symbol, model_data in models.items():
        all_accuracies.append(model_data['metrics']['direction_accuracy'])
        corr = model_data['metrics']['correlation']
        if not pd.isna(corr) and corr is not None:  # Filter out NaN/None values
            all_correlations.append(corr)
    
    print(f"Total models analyzed: {len(all_accuracies)}")
    print(f"Average accuracy: {np.mean(all_accuracies):.3f}")
    print(f"Median accuracy: {np.median(all_accuracies):.3f}")
    print(f"Std dev accuracy: {np.std(all_accuracies):.3f}")
    if all_correlations:
        print(f"Average correlation: {np.mean(all_correlations):.3f}")
        print(f"Average |correlation|: {np.mean(np.abs(all_correlations)):.3f}")
    else:
        print(f"Average correlation: N/A (no valid correlations)")
        print(f"Average |correlation|: N/A (no valid correlations)")
    
    # 6. Key Insights
    print(f"\n6. KEY INSIGHTS:")
    print("-" * 40)
    high_performers = sum(1 for acc in all_accuracies if acc > 0.70)
    sweet_spot = ranges['60-70%']
    low_performers = ranges['<50%']
    
    print(f"• {high_performers} models ({high_performers/total_models*100:.1f}%) achieve >70% accuracy")
    print(f"• {sweet_spot} models ({sweet_spot/total_models*100:.1f}%) in the sweet spot (60-70%)")
    print(f"• {len(low_corr_models)} models have both good accuracy (>=60%) and low correlation")
    print(f"• Tier 1 contains {len(tiers['Tier 1'])} truly predictive models")
    print(f"• {low_performers} models ({low_performers/total_models*100:.1f}%) perform worse than random")

def main():
    """Main analysis function"""
    try:
        # Load data
        models = load_model_metadata(r"C:\quant_system_v2\quant_system_full\gpu_models\model_metadata.json")
        
        # Perform analysis
        ranges, accuracies = analyze_accuracy_distribution(models)
        sweet_spot_models = find_sweet_spot_models(models)
        low_corr_models = identify_low_correlation_models(models)
        tiers = create_tiered_classification(models)
        
        # Print results
        print_analysis_results(ranges, sweet_spot_models, low_corr_models, tiers, models)
        
        return {
            'ranges': ranges,
            'sweet_spot_models': sweet_spot_models,
            'low_corr_models': low_corr_models,
            'tiers': tiers
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

if __name__ == "__main__":
    results = main()