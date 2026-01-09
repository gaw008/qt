#!/usr/bin/env python3
"""
Full System Test Runner - Bypasses Market Time Restrictions
Provides detailed reporting at each stage for functional testing
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add paths for imports
BOT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "bot"))
sys.path.append(BOT_BASE)

from dotenv import load_dotenv
load_dotenv('.env')

# Import necessary modules
try:
    from bot.config import SETTINGS
    from bot.data import DataManager
    from bot.execution_tiger import TigerExecutor
    import bot.stock_selection as stock_selection
    import bot.multi_factor_analysis as multi_factor_analysis
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and paths are correct")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemTester:
    """Full system functionality tester without market time restrictions"""

    def __init__(self):
        self.data_manager = DataManager()
        self.results = {}

    def test_stock_selection(self):
        """Test stock selection phase and report results"""
        print("\n" + "="*60)
        print("STAGE 1: STOCK SELECTION PHASE")
        print("="*60)

        try:
            # Get stock universe
            universe_file = SETTINGS.get('STOCK_UNIVERSE_FILE', 'all_stock_symbols.csv')
            universe_size = int(SETTINGS.get('SELECTION_UNIVERSE_SIZE', 5000))
            result_size = int(SETTINGS.get('SELECTION_RESULT_SIZE', 20))

            print(f"Universe file: {universe_file}")
            print(f"Universe size: {universe_size}")
            print(f"Target selection size: {result_size}")

            # Run stock selection
            print("\nRunning stock selection...")
            start_time = time.time()

            selected_stocks = stock_selection.run_selection(
                universe_size=universe_size,
                result_size=result_size,
                ignore_market_time=True  # Key: bypass market time check
            )

            duration = time.time() - start_time

            if selected_stocks:
                print(f"\n✅ Selection completed in {duration:.2f} seconds")
                print(f"📊 Selected {len(selected_stocks)} stocks from {universe_size} universe")
                print("\nSelected stocks with scores:")
                for i, stock in enumerate(selected_stocks[:10], 1):  # Show top 10
                    symbol = stock.get('symbol', 'Unknown')
                    score = stock.get('score', 0)
                    print(f"  {i:2d}. {symbol:8s} - Score: {score:.2f}")

                if len(selected_stocks) > 10:
                    print(f"  ... and {len(selected_stocks) - 10} more stocks")

                self.results['selection'] = {
                    'success': True,
                    'stocks': selected_stocks,
                    'count': len(selected_stocks),
                    'universe_size': universe_size,
                    'duration': duration
                }
                return selected_stocks
            else:
                print("❌ Stock selection failed - no stocks selected")
                self.results['selection'] = {'success': False, 'error': 'No stocks selected'}
                return []

        except Exception as e:
            print(f"❌ Stock selection failed with error: {e}")
            logger.exception("Stock selection error")
            self.results['selection'] = {'success': False, 'error': str(e)}
            return []

    def test_multi_factor_analysis(self, selected_stocks):
        """Test multi-factor analysis and report scores"""
        print("\n" + "="*60)
        print("STAGE 2: MULTI-FACTOR ANALYSIS PHASE")
        print("="*60)

        if not selected_stocks:
            print("❌ Skipping multi-factor analysis - no stocks to analyze")
            return []

        try:
            print(f"Analyzing {len(selected_stocks)} selected stocks...")
            start_time = time.time()

            # Run multi-factor analysis
            analyzed_stocks = multi_factor_analysis.analyze_stocks(
                selected_stocks,
                ignore_market_time=True  # Key: bypass market time check
            )

            duration = time.time() - start_time

            if analyzed_stocks:
                print(f"\n✅ Multi-factor analysis completed in {duration:.2f} seconds")
                print(f"📊 Analyzed {len(analyzed_stocks)} stocks")
                print("\nStocks with detailed scores:")

                for i, stock in enumerate(analyzed_stocks[:10], 1):  # Show top 10
                    symbol = stock.get('symbol', 'Unknown')
                    total_score = stock.get('total_score', 0)
                    momentum = stock.get('momentum_score', 0)
                    valuation = stock.get('valuation_score', 0)
                    liquidity = stock.get('liquidity_score', 0)
                    print(f"  {i:2d}. {symbol:8s} - Total: {total_score:.2f} (M:{momentum:.1f} V:{valuation:.1f} L:{liquidity:.1f})")

                if len(analyzed_stocks) > 10:
                    print(f"  ... and {len(analyzed_stocks) - 10} more stocks")

                self.results['analysis'] = {
                    'success': True,
                    'stocks': analyzed_stocks,
                    'count': len(analyzed_stocks),
                    'duration': duration
                }
                return analyzed_stocks
            else:
                print("❌ Multi-factor analysis failed - no analysis results")
                self.results['analysis'] = {'success': False, 'error': 'No analysis results'}
                return []

        except Exception as e:
            print(f"❌ Multi-factor analysis failed with error: {e}")
            logger.exception("Multi-factor analysis error")
            self.results['analysis'] = {'success': False, 'error': str(e)}
            return []

    def test_trading_decisions(self, analyzed_stocks):
        """Test trading decision logic and report actions"""
        print("\n" + "="*60)
        print("STAGE 3: TRADING DECISION PHASE")
        print("="*60)

        if not analyzed_stocks:
            print("❌ Skipping trading decisions - no analyzed stocks")
            return

        try:
            print(f"Making trading decisions for {len(analyzed_stocks)} stocks...")

            # Simulate trading decisions based on scores
            buy_threshold = 75.0
            sell_threshold = 25.0

            buy_orders = []
            hold_positions = []
            sell_orders = []

            for stock in analyzed_stocks:
                symbol = stock.get('symbol', 'Unknown')
                score = stock.get('total_score', 0)
                current_price = stock.get('current_price', 0)

                if score >= buy_threshold:
                    # Calculate position size (simplified)
                    position_size = min(100, int(10000 / max(current_price, 1)))  # $10k position max
                    buy_orders.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': position_size,
                        'price': current_price,
                        'score': score
                    })
                elif score <= sell_threshold:
                    # Assume we have positions to sell
                    sell_orders.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': 50,  # Simplified
                        'price': current_price,
                        'score': score
                    })
                else:
                    hold_positions.append({
                        'symbol': symbol,
                        'action': 'HOLD',
                        'score': score
                    })

            print(f"\n📈 Trading Decisions Summary:")
            print(f"  BUY orders:  {len(buy_orders)}")
            print(f"  HOLD positions: {len(hold_positions)}")
            print(f"  SELL orders: {len(sell_orders)}")

            if buy_orders:
                print(f"\n🟢 BUY Orders:")
                for order in buy_orders[:5]:  # Show top 5
                    print(f"  BUY {order['quantity']:3d} {order['symbol']:8s} @ ${order['price']:6.2f} (Score: {order['score']:5.2f})")

            if sell_orders:
                print(f"\n🔴 SELL Orders:")
                for order in sell_orders[:5]:  # Show top 5
                    print(f"  SELL {order['quantity']:3d} {order['symbol']:8s} @ ${order['price']:6.2f} (Score: {order['score']:5.2f})")

            if hold_positions:
                print(f"\n🟡 HOLD Positions:")
                for pos in hold_positions[:5]:  # Show top 5
                    print(f"  HOLD {pos['symbol']:8s} (Score: {pos['score']:5.2f})")

            self.results['trading'] = {
                'success': True,
                'buy_orders': buy_orders,
                'sell_orders': sell_orders,
                'hold_positions': hold_positions,
                'total_actions': len(buy_orders) + len(sell_orders) + len(hold_positions)
            }

        except Exception as e:
            print(f"❌ Trading decisions failed with error: {e}")
            logger.exception("Trading decisions error")
            self.results['trading'] = {'success': False, 'error': str(e)}

    def generate_summary_report(self):
        """Generate comprehensive test results summary"""
        print("\n" + "="*60)
        print("SYSTEM TEST SUMMARY REPORT")
        print("="*60)

        print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Selection results
        if 'selection' in self.results:
            sel = self.results['selection']
            if sel['success']:
                print(f"\n✅ Stock Selection: SUCCESS")
                print(f"   Selected {sel['count']} stocks from {sel.get('universe_size', 'unknown')} universe")
                print(f"   Completed in {sel.get('duration', 0):.2f} seconds")
            else:
                print(f"\n❌ Stock Selection: FAILED - {sel.get('error', 'Unknown error')}")

        # Analysis results
        if 'analysis' in self.results:
            ana = self.results['analysis']
            if ana['success']:
                print(f"\n✅ Multi-Factor Analysis: SUCCESS")
                print(f"   Analyzed {ana['count']} stocks")
                print(f"   Completed in {ana.get('duration', 0):.2f} seconds")
            else:
                print(f"\n❌ Multi-Factor Analysis: FAILED - {ana.get('error', 'Unknown error')}")

        # Trading results
        if 'trading' in self.results:
            trd = self.results['trading']
            if trd['success']:
                print(f"\n✅ Trading Decisions: SUCCESS")
                print(f"   Generated {len(trd['buy_orders'])} BUY orders")
                print(f"   Generated {len(trd['sell_orders'])} SELL orders")
                print(f"   Identified {len(trd['hold_positions'])} HOLD positions")
                print(f"   Total trading actions: {trd['total_actions']}")
            else:
                print(f"\n❌ Trading Decisions: FAILED - {trd.get('error', 'Unknown error')}")

        print("\n" + "="*60)
        print("Full system test completed!")
        print("="*60)

def main():
    """Main test execution function"""
    print("🚀 Starting Full Quantitative Trading System Test")
    print("⚠️  Market time restrictions DISABLED for testing")
    print("🧪 Running complete functionality test with detailed reporting")

    tester = SystemTester()

    # Stage 1: Stock Selection
    selected_stocks = tester.test_stock_selection()

    # Stage 2: Multi-Factor Analysis
    analyzed_stocks = tester.test_multi_factor_analysis(selected_stocks)

    # Stage 3: Trading Decisions
    tester.test_trading_decisions(analyzed_stocks)

    # Final Report
    tester.generate_summary_report()

if __name__ == "__main__":
    main()