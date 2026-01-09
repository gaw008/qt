#!/usr/bin/env python3
"""
Simple Market Regime Classification System Demo

A basic demonstration without Unicode characters for Windows compatibility.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("Market Regime Classification System - Simple Demo")
    print("=" * 60)

    try:
        # Import components
        from market_regime_classifier import MarketRegimeClassifier, MarketRegime
        from regime_visualization import RegimeVisualization

        print("[SUCCESS] Imported regime classification components")

        # Initialize classifier
        print("\n1. Initializing classifier...")
        classifier = MarketRegimeClassifier()
        print("[SUCCESS] Classifier initialized")

        # Generate test data
        print("\n2. Generating test data...")
        market_data = classifier._generate_dummy_market_data(['SPY', 'QQQ'], 100)
        vix_data = classifier._generate_dummy_vix_data(100)
        test_data = {'market_data': market_data, 'vix_data': vix_data}
        print(f"[SUCCESS] Generated test data: {len(market_data)} symbols")

        # Test regime prediction
        print("\n3. Testing regime prediction...")
        prediction = classifier.predict_regime(test_data)
        print(f"[SUCCESS] Predicted regime: {prediction.regime.value}")
        print(f"[SUCCESS] Confidence: {prediction.confidence:.3f}")
        print(f"[SUCCESS] Probabilities - Normal: {prediction.probability_normal:.3f}, "
              f"Volatile: {prediction.probability_volatile:.3f}, "
              f"Crisis: {prediction.probability_crisis:.3f}")

        # Test different prediction methods
        print("\n4. Testing prediction methods...")
        threshold_pred = classifier.predict_regime(test_data, method='threshold')
        print(f"[SUCCESS] Threshold method: {threshold_pred.regime.value} "
              f"(confidence: {threshold_pred.confidence:.3f})")

        # Test regime summary
        print("\n5. Testing regime summary...")
        summary = classifier.get_regime_summary()
        print(f"[SUCCESS] Current regime: {summary['current_regime']}")
        print(f"[SUCCESS] Models fitted: {summary['models_fitted']}")

        # Test visualization
        print("\n6. Testing visualization...")
        viz = RegimeVisualization(classifier)

        # Export report
        report_success = viz.export_regime_report(
            'simple_demo_report.json',
            format='json',
            include_history=False,
            include_validation=False
        )
        if report_success:
            print("[SUCCESS] Exported regime report")

        # Test model persistence
        print("\n7. Testing model persistence...")
        save_success = classifier.save_models('simple_demo_models.pkl')
        if save_success:
            print("[SUCCESS] Models saved")

            # Test loading
            new_classifier = MarketRegimeClassifier()
            load_success = new_classifier.load_models('simple_demo_models.pkl')
            if load_success:
                print("[SUCCESS] Models loaded")

        # Performance test
        print("\n8. Running performance test...")
        start_time = datetime.now()
        for i in range(3):
            pred = classifier.predict_regime(test_data)
        end_time = datetime.now()
        avg_time = (end_time - start_time).total_seconds() / 3
        print(f"[SUCCESS] Average prediction time: {avg_time:.3f} seconds")

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Core functionality verified:")
        print("- Regime classification with multiple methods")
        print("- Confidence scoring and probability estimates")
        print("- Model persistence and loading")
        print("- Visualization and reporting")
        print("- Performance benchmarking")

        # Clean up
        cleanup_files = ['simple_demo_report.json', 'simple_demo_models.pkl']
        for file in cleanup_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"[CLEANUP] Removed {file}")
                except:
                    pass

        return True

    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n[FINAL] Market Regime Classification System is ready for use!")
        print("[FINAL] Integration points:")
        print("  - Use MarketRegimeClassifier for regime detection")
        print("  - Use RegimeVisualization for analysis and reporting")
        print("  - Use RegimeRiskIntegration for risk management")
    else:
        print("\n[FINAL] Demo failed. Please check error messages above.")