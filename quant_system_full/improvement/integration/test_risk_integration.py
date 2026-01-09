#!/usr/bin/env python3
"""
Test Risk Management Integration

Test the risk-aware trading adapter with actual market data
"""

import sys
import os
from pathlib import Path
import logging

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "bot"))

def test_risk_integration():
    """Test risk integration with current portfolio data"""
    try:
        # Import risk adapter
        from risk_aware_trading_adapter import (
            create_risk_aware_trading_engine,
            RiskConstraints
        )

        print("Testing Risk Management Integration")
        print("=" * 50)

        # Create adapter with custom constraints
        risk_constraints = RiskConstraints(
            max_portfolio_var_95=0.03,  # 3% daily VaR limit
            max_position_concentration=0.20,  # 20% max position
            max_correlation_threshold=0.75,  # 75% max correlation
            min_diversification_count=3,  # Minimum 3 positions
            max_sector_concentration=0.30  # 30% max sector weight
        )

        adapter = create_risk_aware_trading_engine(
            dry_run=True,
            enable_risk_monitoring=True,
            risk_constraints=risk_constraints
        )

        # Test with current status data
        import json
        status_file = project_root / "dashboard" / "state" / "status.json"

        if status_file.exists():
            with open(status_file, 'r') as f:
                status_data = json.load(f)

            # Extract current positions
            current_positions = []
            if 'real_positions' in status_data:
                for pos in status_data['real_positions']:
                    current_positions.append({
                        'symbol': pos['symbol'],
                        'qty': pos['quantity'],
                        'price': pos['market_price'],
                        'value': pos['market_value'],
                        'action': 'hold'
                    })

            # Extract recommended positions
            recommended_positions = []
            if 'selection_results' in status_data and 'top_picks' in status_data['selection_results']:
                for i, pick in enumerate(status_data['selection_results']['top_picks'][:5]):
                    # Simulate position sizing (10k per position)
                    symbol = pick['symbol']
                    price = 100.0  # Default price for testing
                    value = 10000.0
                    qty = int(value / price)

                    recommended_positions.append({
                        'symbol': symbol,
                        'qty': qty,
                        'price': price,
                        'value': value,
                        'action': pick['dominant_action'],
                        'score': pick['avg_score']
                    })

            print(f"Current Positions: {len(current_positions)}")
            for pos in current_positions:
                print(f"  {pos['symbol']}: {pos['qty']} @ ${pos['price']:.2f} = ${pos['value']:,.0f}")

            print(f"\nRecommended Positions: {len(recommended_positions)}")
            for pos in recommended_positions:
                print(f"  {pos['symbol']}: {pos['qty']} @ ${pos['price']:.2f} = ${pos['value']:,.0f} ({pos['action']})")

            # Update market data
            market_data = {}
            for pos in current_positions + recommended_positions:
                market_data[pos['symbol']] = {'price': pos['price']}

            if market_data:
                adapter.update_market_data(market_data)
                print(f"\nUpdated market data for {len(market_data)} symbols")

                # Assess portfolio risk
                current_values = {pos['symbol']: pos['value'] for pos in current_positions}
                recommended_values = {pos['symbol']: pos['value'] for pos in recommended_positions}

                risk_assessment = adapter.assess_portfolio_risk(current_values, recommended_values)

                print("\nRisk Assessment Results:")
                print("=" * 30)
                print(f"Risk Approved: {risk_assessment.approved}")
                print(f"Risk Score: {risk_assessment.risk_score:.1f}/100")
                print(f"Portfolio VaR (95%): {risk_assessment.var_95:.4f}")
                print(f"Concentration Risk (HHI): {risk_assessment.concentration_risk:.4f}")

                if risk_assessment.warnings:
                    print(f"\nWarnings:")
                    for warning in risk_assessment.warnings:
                        print(f"  - {warning}")

                if risk_assessment.recommendations:
                    print(f"\nRecommendations:")
                    for rec in risk_assessment.recommendations:
                        print(f"  - {rec}")

                # Test position optimization
                total_portfolio_value = 50000.0  # 50k portfolio
                optimized_positions = adapter.optimize_position_sizes(
                    recommended_values,
                    total_portfolio_value
                )

                print(f"\nOptimized Position Sizes:")
                print("=" * 30)
                for symbol, value in optimized_positions.items():
                    print(f"  {symbol}: ${value:,.0f}")

                # Test trading analysis
                analysis = adapter.analyze_trading_opportunities(
                    current_positions,
                    recommended_positions
                )

                print(f"\nTrading Analysis:")
                print("=" * 20)
                print(f"Buy Signals: {len(analysis['trading_signals']['buy'])}")
                print(f"Hold Signals: {len(analysis['trading_signals']['hold'])}")
                print(f"Sell Signals: {len(analysis['trading_signals']['sell'])}")

                if 'risk_assessment' in analysis:
                    ra = analysis['risk_assessment']
                    print(f"Risk Compliance: {ra['approved']}")
                    print(f"Risk Score: {ra['risk_score']:.1f}/100")
                    print(f"VaR 95%: {ra['var_95_percent']:.2f}%")

        else:
            print("Status file not found, using example data")
            # Use example data for testing
            market_data = {
                'AAPL': {'price': 150.0},
                'GOOGL': {'price': 2500.0},
                'TSLA': {'price': 800.0},
                'MSFT': {'price': 300.0},
                'NVDA': {'price': 400.0}
            }

            adapter.update_market_data(market_data)

            # Example positions
            recommended_positions = [
                {'symbol': 'AAPL', 'qty': 67, 'price': 150.0, 'value': 10000.0, 'action': 'buy'},
                {'symbol': 'GOOGL', 'qty': 4, 'price': 2500.0, 'value': 10000.0, 'action': 'buy'},
                {'symbol': 'TSLA', 'qty': 12, 'price': 800.0, 'value': 10000.0, 'action': 'buy'},
                {'symbol': 'MSFT', 'qty': 33, 'price': 300.0, 'value': 10000.0, 'action': 'buy'},
                {'symbol': 'NVDA', 'qty': 25, 'price': 400.0, 'value': 10000.0, 'action': 'buy'}
            ]

            current_values = {}
            recommended_values = {pos['symbol']: pos['value'] for pos in recommended_positions}

            risk_assessment = adapter.assess_portfolio_risk(current_values, recommended_values)

            print("Example Risk Assessment:")
            print("=" * 30)
            print(f"Risk Approved: {risk_assessment.approved}")
            print(f"Risk Score: {risk_assessment.risk_score:.1f}/100")
            print(f"Portfolio VaR (95%): {risk_assessment.var_95:.4f}")

        print("\nRisk Integration Test Completed Successfully!")
        return True

    except Exception as e:
        print(f"Error in risk integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise

    success = test_risk_integration()
    sys.exit(0 if success else 1)