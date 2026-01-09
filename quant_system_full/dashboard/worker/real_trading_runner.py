#!/usr/bin/env python3
"""
Real Quantitative Trading Runner
使用真实的市场数据和策略，但在安全的模拟模式下运行
"""
import os, time, sys
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Add project paths
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE)
sys.path.append(os.path.join(BASE, "dashboard", "backend"))

from bot.config import SETTINGS
from bot.data import fetch_history
from bot.alpha_router import get_alpha_signals
from state_manager import write_status, append_log, is_killed

load_dotenv()

# IT和医疗保健行业股票池
STOCK_UNIVERSE = {
    'IT': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AMD', 'CRM', 'ORCL', 'ADBE'],
    'HEALTHCARE': ['UNH', 'JNJ', 'PFE', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY', 'MRK', 'CVS']
}

class RealTradingBot:
    def __init__(self, sectors=['IT', 'HEALTHCARE']):
        self.sectors = sectors
        self.symbols = []
        for sector in sectors:
            if sector in STOCK_UNIVERSE:
                self.symbols.extend(STOCK_UNIVERSE[sector])
        
        self.portfolio = {}
        self.total_pnl = 0.0
        self.positions = []
        self.last_prices = {}
        
        append_log(f"Real Trading Bot initialized for sectors: {sectors}")
        append_log(f"Stock universe: {self.symbols}")
    
    def fetch_market_data(self):
        """获取所有股票的市场数据"""
        data = {}
        for symbol in self.symbols:
            try:
                df = fetch_history(None, symbol, 'day', 100, dry_run=SETTINGS.dry_run)
                if df is not None and not df.empty:
                    data[symbol] = df
                    self.last_prices[symbol] = df['close'].iloc[-1]
                    append_log(f"Data fetched for {symbol}: ${self.last_prices[symbol]:.2f}")
                else:
                    append_log(f"Warning: No data for {symbol}")
            except Exception as e:
                append_log(f"Error fetching data for {symbol}: {e}")
        
        return data
    
    def generate_signals(self, data):
        """生成交易信号"""
        if not data:
            return pd.DataFrame()
        
        try:
            signals_df = get_alpha_signals(data)
            return signals_df
        except Exception as e:
            append_log(f"Error generating signals: {e}")
            return pd.DataFrame()
    
    def execute_trades(self, signals_df):
        """执行交易（模拟）"""
        if signals_df.empty:
            return
        
        for _, row in signals_df.iterrows():
            symbol = row['symbol']
            signal = row['signal']
            
            if symbol not in self.last_prices:
                continue
                
            current_price = self.last_prices[symbol]
            
            # 简单的仓位管理
            if signal == 1:  # BUY信号
                if symbol not in self.portfolio or self.portfolio[symbol]['quantity'] == 0:
                    quantity = 100  # 固定买入100股
                    self.portfolio[symbol] = {
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': datetime.now()
                    }
                    self.positions.append({
                        'symbol': symbol,
                        'qty': quantity,
                        'price': current_price
                    })
                    append_log(f"BUY {quantity} shares of {symbol} @ ${current_price:.2f}")
            
            elif signal == -1:  # SELL信号
                if symbol in self.portfolio and self.portfolio[symbol]['quantity'] > 0:
                    position = self.portfolio[symbol]
                    quantity = position['quantity']
                    entry_price = position['entry_price']
                    
                    # 计算盈亏
                    pnl = (current_price - entry_price) * quantity
                    self.total_pnl += pnl
                    
                    # 清空仓位
                    self.portfolio[symbol] = {'quantity': 0, 'entry_price': 0, 'entry_time': None}
                    self.positions = [pos for pos in self.positions if pos['symbol'] != symbol]
                    
                    append_log(f"SELL {quantity} shares of {symbol} @ ${current_price:.2f} | P&L: ${pnl:.2f}")
    
    def update_status(self):
        """更新状态到Dashboard"""
        current_positions = []
        for symbol, position in self.portfolio.items():
            if position['quantity'] > 0:
                current_price = self.last_prices.get(symbol, position['entry_price'])
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                current_positions.append({
                    'symbol': symbol,
                    'qty': position['quantity'],
                    'price': current_price,
                    'entry_price': position['entry_price'],
                    'unrealized_pnl': unrealized_pnl
                })
        
        total_unrealized = sum(pos['unrealized_pnl'] for pos in current_positions)
        
        status = {
            'bot': 'running',
            'pnl': round(self.total_pnl + total_unrealized, 2),
            'realized_pnl': round(self.total_pnl, 2),
            'unrealized_pnl': round(total_unrealized, 2),
            'positions': current_positions,
            'last_signal': 'ANALYZING',
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(self.symbols),
            'active_positions': len(current_positions)
        }
        
        write_status(status)
    
    def run(self, interval=300):
        """主运行循环"""
        append_log("Real Trading Bot started!")
        
        while True:
            try:
                if is_killed():
                    append_log("Trading bot paused by kill switch.")
                    write_status({'bot': 'paused'})
                    time.sleep(5)
                    continue
                
                append_log("=== Trading Cycle Started ===")
                
                # 1. 获取市场数据
                market_data = self.fetch_market_data()
                
                if not market_data:
                    append_log("No market data available, waiting...")
                    time.sleep(interval)
                    continue
                
                # 2. 生成交易信号
                signals = self.generate_signals(market_data)
                
                if not signals.empty:
                    append_log(f"Generated signals for {len(signals)} symbols")
                    
                    # 3. 执行交易
                    self.execute_trades(signals)
                else:
                    append_log("No trading signals generated")
                
                # 4. 更新状态
                self.update_status()
                
                append_log(f"=== Cycle Complete | Total P&L: ${self.total_pnl:.2f} ===")
                
                # 5. 等待下一个周期
                time.sleep(interval)
                
            except Exception as e:
                append_log(f"Error in trading loop: {e}")
                import traceback
                append_log(traceback.format_exc())
                time.sleep(60)  # 错误后等待1分钟再试

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sectors', default='IT,HEALTHCARE', help='Comma-separated sectors')
    parser.add_argument('--interval', type=int, default=300, help='Trading interval in seconds')
    
    args = parser.parse_args()
    sectors = [s.strip() for s in args.sectors.split(',')]
    
    bot = RealTradingBot(sectors=sectors)
    bot.run(interval=args.interval)

if __name__ == "__main__":
    import argparse
    main()