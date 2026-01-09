#!/usr/bin/env python3
"""
自动交易执行引擎 - 动态资金管理版本
基于AI选股推荐和当前持仓状态，执行买入/卖出决策
实施动态资金管理，移除硬编码限制
"""

import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Tiger API imports
try:
    from tigeropen.tiger_open_config import TigerOpenClientConfig
    from tigeropen.trade.trade_client import TradeClient
    from tigeropen.quote.quote_client import QuoteClient
    from tigeropen.common.util.contract_utils import stock_contract
    from tigeropen.common.util.order_utils import market_order, limit_order
    from tigeropen.common.consts import Market, OrderStatus, OrderType
    TIGER_SDK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Tiger SDK not available: {e}")
    TIGER_SDK_AVAILABLE = False

from state_manager import append_log

class AutoTradingEngine:
    """自动交易执行引擎 - 动态资金管理版本"""

    def __init__(self, dry_run=True, max_position_value=None, max_daily_trades=10):
        """
        初始化交易引擎

        Args:
            dry_run: 是否为模拟模式
            max_position_value: 单个持仓最大价值($) - 如果为None则动态计算
            max_daily_trades: 每日最大交易次数
        """
        self.dry_run = dry_run

        # 动态资金管理配置
        self.trading_safety_factor = float(os.getenv('TRADING_SAFETY_FACTOR', '0.8'))
        self.min_cash_reserve = float(os.getenv('MIN_CASH_RESERVE', '5000'))
        self.max_position_percent = float(os.getenv('MAX_POSITION_PERCENT', '0.20'))
        self.fallback_buying_power = float(os.getenv('FALLBACK_BUYING_POWER', '50000'))
        self.max_daily_outflow_percent = float(os.getenv('MAX_DAILY_OUTFLOW_PERCENT', '0.80'))

        # 如果没有指定max_position_value，将动态计算
        self.max_position_value = max_position_value
        self.max_daily_trades = max_daily_trades
        self.daily_trade_count = 0
        self.trade_history = []

        # 失败股票黑名单：记录连续失败的股票
        self.failed_symbols = {}  # {symbol: failure_count}
        self.max_failures = 1    # 最大连续失败次数（改为1次立即生效）

        # Initialize Tiger API clients
        if TIGER_SDK_AVAILABLE and not dry_run:
            self._init_tiger_clients()
        else:
            self.trade_client = None
            self.quote_client = None

        append_log(f"[AUTO_TRADING] Dynamic Fund Management Engine initialized")
        append_log(f"[AUTO_TRADING] DRY_RUN: {dry_run}")
        append_log(f"[AUTO_TRADING] Safety Factor: {self.trading_safety_factor:.1%}")
        append_log(f"[AUTO_TRADING] Min Cash Reserve: ${self.min_cash_reserve:,.2f}")
        append_log(f"[AUTO_TRADING] Max Position %: {self.max_position_percent:.1%}")
        append_log(f"[AUTO_TRADING] Max Daily Outflow %: {self.max_daily_outflow_percent:.1%}")

    def get_raw_buying_power(self):
        """获取原始账户购买力（未应用安全系数）"""
        if self.dry_run or not self.trade_client:
            # 模拟模式使用fallback值
            raw_buying_power = self.fallback_buying_power
            append_log(f"[AUTO_TRADING] DRY RUN mode: using fallback buying power ${raw_buying_power:,.2f}")
            return raw_buying_power

        try:
            assets_result = self.trade_client.get_assets()

            if hasattr(assets_result, '__iter__') and not isinstance(assets_result, str):
                assets_list = list(assets_result)
                if assets_list:
                    asset = assets_list[0]
                else:
                    append_log(f"[AUTO_TRADING] No assets found, using fallback")
                    return self.fallback_buying_power
            else:
                asset = assets_result

            if hasattr(asset, 'summary'):
                summary = asset.summary
            else:
                summary = asset

            # 优先获取现金余额
            cash_balance = getattr(summary, 'cash', None)
            if cash_balance is not None and cash_balance > 0:
                raw_buying_power = float(cash_balance)
                append_log(f"[AUTO_TRADING] Tiger API cash balance: ${raw_buying_power:,.2f}")
                return raw_buying_power
            else:
                # 备用：获取购买力
                buying_power = getattr(summary, 'buying_power', None)
                if buying_power is not None and buying_power > 0:
                    raw_buying_power = float(buying_power)
                    append_log(f"[AUTO_TRADING] Tiger API buying power: ${raw_buying_power:,.2f}")
                    return raw_buying_power
                else:
                    raw_buying_power = self.fallback_buying_power
                    append_log(f"[AUTO_TRADING] Tiger API failed, using fallback: ${raw_buying_power:,.2f}")
                    return raw_buying_power

        except Exception as e:
            append_log(f"[AUTO_TRADING] Error getting buying power from Tiger API: {e}")
            raw_buying_power = self.fallback_buying_power
            append_log(f"[AUTO_TRADING] Using fallback buying power: ${raw_buying_power:,.2f}")
            return raw_buying_power

    def get_buying_power(self):
        """获取可用于交易的安全购买力"""
        raw_buying_power = self.get_raw_buying_power()

        # 应用安全系数和现金储备
        safe_buying_power = max(0, (raw_buying_power - self.min_cash_reserve) * self.trading_safety_factor)

        append_log(f"[AUTO_TRADING] Raw buying power: ${raw_buying_power:,.2f}")
        append_log(f"[AUTO_TRADING] Safety factor: {self.trading_safety_factor:.1%}")
        append_log(f"[AUTO_TRADING] Min cash reserve: ${self.min_cash_reserve:,.2f}")
        append_log(f"[AUTO_TRADING] Safe buying power: ${safe_buying_power:,.2f}")

        return safe_buying_power

    def get_max_position_value(self, available_buying_power):
        """动态计算单个持仓最大价值"""
        if self.max_position_value is None:
            # 根据可用购买力动态计算
            max_allocation = available_buying_power * self.max_position_percent
        else:
            # 使用固定值和百分比的较小值
            max_allocation = min(self.max_position_value, available_buying_power * self.max_position_percent)

        append_log(f"[AUTO_TRADING] Max position value: ${max_allocation:,.2f}")
        return max_allocation

    def _init_tiger_clients(self):
        """初始化Tiger API客户端"""
        try:
            # Use props configuration
            props_dir = str((Path(__file__).parent.parent.parent / "props").resolve())
            client_config = TigerOpenClientConfig(props_path=props_dir)

            self.trade_client = TradeClient(client_config)
            self.quote_client = QuoteClient(client_config)

            append_log(f"[AUTO_TRADING] Tiger API clients initialized")
        except Exception as e:
            append_log(f"[AUTO_TRADING] Error initializing Tiger clients: {e}")
            self.trade_client = None
            self.quote_client = None

    def analyze_trading_opportunities(self, current_positions: List[Dict],
                                    recommended_positions: List[Dict]) -> Dict[str, List[Dict]]:
        """
        分析交易机会，生成买入/卖出信号

        Args:
            current_positions: 当前真实持仓
            recommended_positions: AI推荐持仓

        Returns:
            交易信号字典 {'buy': [...], 'sell': [...], 'hold': [...]}
        """

        # 获取当前购买力
        available_buying_power = self.get_buying_power()
        append_log(f"[AUTO_TRADING] Available buying power: ${available_buying_power:,.2f}")

        # 当前持仓映射
        current_symbols = {}
        for pos in current_positions:
            symbol = pos.get('symbol')  # Try symbol first
            if not symbol:  # Fall back to other possible keys
                continue
            current_symbols[symbol] = pos

        # AI推荐映射（考虑action="buy"或"strong_buy"的）
        recommended_buys = {}
        for pos in recommended_positions:
            action = pos.get("action", "").lower()
            if action in ["buy", "strong_buy"]:
                symbol = pos.get('symbol')
                if symbol:
                    recommended_buys[symbol] = pos

        trading_signals = {
            'buy': [],
            'sell': [],
            'hold': []
        }

        # 1. 买入信号：AI推荐买入但当前未持仓
        for symbol, rec_pos in recommended_buys.items():
            if symbol not in current_symbols:

                # 检查是否在失败黑名单中
                if symbol in self.failed_symbols and self.failed_symbols[symbol] >= self.max_failures:
                    append_log(f"[AUTO_TRADING] Skipping {symbol} - failed {self.failed_symbols[symbol]} times")
                    continue

                # 动态计算单个持仓最大价值
                max_allocation = self.get_max_position_value(available_buying_power)
                current_price = rec_pos.get('price', 0)

                if current_price > 0 and max_allocation > current_price:
                    qty = int(max_allocation / current_price)
                    estimated_value = qty * current_price

                    # 确保不超过购买力
                    if qty > 0 and estimated_value <= available_buying_power:
                        trading_signals['buy'].append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'qty': qty,
                            'price': current_price,
                            'estimated_value': estimated_value,
                            'reason': f"AI recommends BUY (score: {rec_pos.get('score', 0)}) - using {estimated_value/available_buying_power*100:.1f}% buying power"
                        })
                        append_log(f"[AUTO_TRADING] Buy signal for {symbol}: {qty} shares @ ${current_price:.2f} = ${estimated_value:,.2f}")
                    else:
                        append_log(f"[AUTO_TRADING] Insufficient buying power for {symbol}: need ${estimated_value:,.2f}, have ${available_buying_power:,.2f}")
                else:
                    append_log(f"[AUTO_TRADING] Invalid price or insufficient funds for {symbol}: price=${current_price:.2f}, max_allocation=${max_allocation:,.2f}")

        # 2. 卖出信号：当前持仓但不在AI推荐买入列表中
        for symbol, curr_pos in current_symbols.items():
            if symbol not in recommended_buys:
                trading_signals['sell'].append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'qty': curr_pos.get('quantity', 0),
                    'price': curr_pos.get('market_price', 0),
                    'estimated_value': curr_pos.get('market_value', 0),
                    'reason': "Not in AI recommendations, exit position"
                })
            else:
                # 3. 持有信号：当前持仓且在AI推荐中
                trading_signals['hold'].append({
                    'symbol': symbol,
                    'action': 'HOLD',
                    'qty': curr_pos.get('quantity', 0),
                    'price': curr_pos.get('market_price', 0),
                    'estimated_value': curr_pos.get('market_value', 0),
                    'reason': f"Hold position, AI still recommends (score: {recommended_buys[symbol].get('score', 0)})"
                })

        append_log(f"[AUTO_TRADING] Trading analysis complete:")
        append_log(f"  - Buy signals: {len(trading_signals['buy'])}")
        append_log(f"  - Sell signals: {len(trading_signals['sell'])}")
        append_log(f"  - Hold positions: {len(trading_signals['hold'])}")

        return trading_signals

    def execute_trading_signals(self, trading_signals: Dict[str, List[Dict]]) -> List[Dict]:
        """
        执行交易信号 - 实现分阶段执行以确保流动性管理

        Args:
            trading_signals: 交易信号字典

        Returns:
            执行结果列表
        """

        execution_results = []

        # 风险控制检查
        risk_check = self._perform_risk_checks(trading_signals)
        if not risk_check['passed']:
            append_log(f"[AUTO_TRADING] Risk check failed: {risk_check['reason']}")
            return [{
                'success': False,
                'error': f"Risk check failed: {risk_check['reason']}",
                'timestamp': datetime.now().isoformat()
            }]

        # 检查每日交易限制
        if self.daily_trade_count >= self.max_daily_trades:
            append_log(f"[AUTO_TRADING] Daily trade limit reached ({self.max_daily_trades})")
            return execution_results

        # Phase 1: 执行所有卖出订单
        sell_proceeds = 0.0
        successful_sells = 0
        append_log(f"[AUTO_TRADING] Phase 1: Executing {len(trading_signals.get('sell', []))} sell orders")

        for sell_signal in trading_signals.get('sell', []):
            if self.daily_trade_count >= self.max_daily_trades:
                break

            # 单个订单风险检查
            if self._validate_single_order(sell_signal):
                result = self._execute_order(sell_signal)
                execution_results.append(result)

                if result.get('success'):
                    self.daily_trade_count += 1
                    successful_sells += 1
                    # 累计卖出收益（用于重新计算购买力）
                    sell_proceeds += sell_signal.get('estimated_value', 0)
                    append_log(f"[AUTO_TRADING] Sell executed: {sell_signal['symbol']}, proceeds: ${sell_signal.get('estimated_value', 0):,.2f}")
            else:
                append_log(f"[AUTO_TRADING] Skipped {sell_signal['symbol']} - failed validation")

        # Phase 2: 等待并重新计算购买力（如果有卖出交易成功）
        if successful_sells > 0:
            append_log(f"[AUTO_TRADING] Phase 1 complete: {successful_sells} sells executed, ${sell_proceeds:,.2f} total proceeds")
            append_log(f"[AUTO_TRADING] Waiting for settlement before buy orders...")

            # 在实际生产中，这里可以添加短暂延迟或检查订单状态
            # 但对于现有系统，我们立即重新计算购买力
            time.sleep(1)  # 短暂延迟以允许系统更新

            # 重新获取更新后的购买力
            updated_buying_power = self.get_buying_power()
            append_log(f"[AUTO_TRADING] Updated buying power after sells: ${updated_buying_power:,.2f}")

            # 重新验证买入订单的购买力
            buy_signals = trading_signals.get('buy', [])
            validated_buys = self._revalidate_buy_orders(buy_signals, updated_buying_power)
            append_log(f"[AUTO_TRADING] Revalidated {len(validated_buys)}/{len(buy_signals)} buy orders")
        else:
            validated_buys = trading_signals.get('buy', [])
            append_log(f"[AUTO_TRADING] No sell orders executed, proceeding with original buy orders")

        # Phase 3: 执行验证后的买入订单
        append_log(f"[AUTO_TRADING] Phase 2: Executing {len(validated_buys)} buy orders")

        for buy_signal in validated_buys:
            if self.daily_trade_count >= self.max_daily_trades:
                break

            # 单个订单风险检查
            if self._validate_single_order(buy_signal):
                result = self._execute_order(buy_signal)
                execution_results.append(result)

                if result.get('success'):
                    self.daily_trade_count += 1
                    append_log(f"[AUTO_TRADING] Buy executed: {buy_signal['symbol']}, cost: ${buy_signal.get('estimated_value', 0):,.2f}")
            else:
                append_log(f"[AUTO_TRADING] Skipped {buy_signal['symbol']} - failed validation")

        append_log(f"[AUTO_TRADING] Execution complete: {len(execution_results)} total orders processed")
        return execution_results

    def _perform_risk_checks(self, trading_signals: Dict[str, List[Dict]]) -> Dict:
        """
        执行风险控制检查 - 使用动态限制

        Args:
            trading_signals: 交易信号

        Returns:
            风险检查结果
        """

        # 检查总交易价值
        total_buy_value = sum(signal.get('estimated_value', 0) for signal in trading_signals.get('buy', []))
        total_sell_value = sum(signal.get('estimated_value', 0) for signal in trading_signals.get('sell', []))

        # 检查单日交易量限制 - 动态基于购买力
        available_buying_power = self.get_buying_power()
        max_daily_buy_limit = available_buying_power  # 不超过可用购买力

        if total_buy_value > max_daily_buy_limit:
            return {
                'passed': False,
                'reason': f'Daily buy limit exceeded: ${total_buy_value:,.2f} > ${max_daily_buy_limit:,.2f} (available buying power)'
            }

        # 检查交易数量限制
        total_trades = len(trading_signals.get('buy', [])) + len(trading_signals.get('sell', []))
        if total_trades > self.max_daily_trades:
            return {
                'passed': False,
                'reason': f'Too many trades: {total_trades} > {self.max_daily_trades}'
            }

        # 检查是否有有效的交易信号
        if total_trades == 0:
            return {
                'passed': True,
                'reason': 'No trades to execute'
            }

        # 检查净现金流 - 动态计算基于实际购买力
        net_cash = total_sell_value - total_buy_value
        max_allowed_outflow = available_buying_power * self.max_daily_outflow_percent

        if net_cash < -max_allowed_outflow:
            return {
                'passed': False,
                'reason': f'Insufficient funds: net outflow ${abs(net_cash):,.2f} > ${max_allowed_outflow:,.2f} ({self.max_daily_outflow_percent:.0%} of buying power)'
            }

        return {
            'passed': True,
            'reason': f'Risk checks passed - {total_trades} trades, net cash: ${net_cash:,.2f}, max outflow: ${max_allowed_outflow:,.2f}'
        }

    def _validate_single_order(self, order_signal: Dict) -> bool:
        """
        验证单个订单 - 使用动态限制

        Args:
            order_signal: 订单信号

        Returns:
            是否通过验证
        """

        symbol = order_signal.get('symbol', '')
        qty = order_signal.get('qty', 0)
        price = order_signal.get('price', 0)
        estimated_value = order_signal.get('estimated_value', 0)

        # 基本字段检查
        if not symbol or qty <= 0 or price <= 0:
            append_log(f"[VALIDATION] Invalid order parameters for {symbol}")
            return False

        # 单笔交易价值检查 - 动态计算
        buying_power = self.get_buying_power()
        max_allowed_value = self.get_max_position_value(buying_power) * 1.5  # 允许50%溢价

        if estimated_value > max_allowed_value:
            append_log(f"[VALIDATION] Order value too high: ${estimated_value:,.2f} > ${max_allowed_value:,.2f} for {symbol}")
            return False

        # 股价合理性检查 ($1 - $10,000)
        if price < 1.0 or price > 10000:
            append_log(f"[VALIDATION] Price out of range: ${price:.2f} for {symbol}")
            return False

        # 数量合理性检查 (1 - 10,000股)
        if qty < 1 or qty > 10000:
            append_log(f"[VALIDATION] Quantity out of range: {qty} shares for {symbol}")
            return False

        return True

    def _revalidate_buy_orders(self, buy_signals: List[Dict], updated_buying_power: float) -> List[Dict]:
        """
        重新验证买入订单（在卖出后更新购买力）

        Args:
            buy_signals: 原始买入信号
            updated_buying_power: 更新后的购买力

        Returns:
            验证后的买入信号列表
        """
        validated_buys = []

        for buy_signal in buy_signals:
            estimated_value = buy_signal.get('estimated_value', 0)

            # 检查是否仍在购买力范围内
            if estimated_value <= updated_buying_power:
                validated_buys.append(buy_signal)
            else:
                # 重新计算数量以适应更新的购买力
                symbol = buy_signal.get('symbol', '')
                price = buy_signal.get('price', 0)
                max_allocation = self.get_max_position_value(updated_buying_power)

                if price > 0 and max_allocation > price:
                    new_qty = int(max_allocation / price)
                    new_estimated_value = new_qty * price

                    if new_qty > 0 and new_estimated_value <= updated_buying_power:
                        updated_signal = buy_signal.copy()
                        updated_signal.update({
                            'qty': new_qty,
                            'estimated_value': new_estimated_value,
                            'reason': f"Adjusted for updated buying power: {new_estimated_value/updated_buying_power*100:.1f}%"
                        })
                        validated_buys.append(updated_signal)
                        append_log(f"[AUTO_TRADING] Adjusted {symbol}: {new_qty} shares @ ${price:.2f} = ${new_estimated_value:,.2f}")
                    else:
                        append_log(f"[AUTO_TRADING] Removed {symbol} - insufficient updated buying power")
                else:
                    append_log(f"[AUTO_TRADING] Removed {symbol} - invalid price or allocation")

        return validated_buys

    def _execute_order(self, order_signal: Dict) -> Dict:
        """
        执行单个订单

        Args:
            order_signal: 订单信号

        Returns:
            执行结果
        """

        symbol = order_signal.get('symbol', '')
        action = order_signal.get('action', '')
        qty = order_signal.get('qty', 0)
        price = order_signal.get('price', 0)

        append_log(f"[AUTO_TRADING] Executing {action} order: {qty} shares of {symbol} @ ${price:.2f}")

        if self.dry_run:
            # 模拟执行
            result = {
                'success': True,
                'symbol': symbol,
                'action': action,
                'qty': qty,
                'price': price,
                'order_id': f"SIM_{int(time.time())}",
                'message': "Simulated order execution",
                'timestamp': datetime.now().isoformat()
            }
            append_log(f"[AUTO_TRADING] SIMULATED {action}: {qty} {symbol} @ ${price:.2f}")

        else:
            # 真实执行
            result = self._place_real_order(symbol, action, qty, price)

        # 更新失败计数
        self._update_failure_count(symbol, result.get('success', False))

        # 记录交易历史
        self.trade_history.append(result)

        return result

    def _place_real_order(self, symbol: str, action: str, qty: int, price: float) -> Dict:
        """
        下真实订单

        Args:
            symbol: 股票代码
            action: 交易动作 (BUY/SELL)
            qty: 数量
            price: 价格

        Returns:
            执行结果
        """

        if not self.trade_client:
            return {
                'success': False,
                'symbol': symbol,
                'action': action,
                'qty': qty,
                'price': price,
                'error': "Tiger trade client not available",
                'timestamp': datetime.now().isoformat()
            }

        try:
            # 创建股票合约
            contract = stock_contract(symbol, Market.US)

            # 创建市价订单（更容易成交）
            order = market_order(
                account="41169270",  # Tiger账户ID
                contract=contract,
                action=action,
                quantity=qty
            )

            # 下订单
            order_result = self.trade_client.place_order(order)

            if order_result and hasattr(order_result, 'id'):
                append_log(f"[AUTO_TRADING] SUCCESS: {action} order placed - Order ID: {order_result.id}")
                return {
                    'success': True,
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'price': price,
                    'order_id': order_result.id,
                    'message': f"Order placed successfully",
                    'timestamp': datetime.now().isoformat()
                }
            else:
                append_log(f"[AUTO_TRADING] ERROR: Failed to place {action} order for {symbol}")
                return {
                    'success': False,
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'price': price,
                    'error': "Order placement failed",
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            append_log(f"[AUTO_TRADING] EXCEPTION placing order for {symbol}: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'action': action,
                'qty': qty,
                'price': price,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _update_failure_count(self, symbol: str, success: bool):
        """
        更新股票失败计数

        Args:
            symbol: 股票代码
            success: 是否成功
        """
        if success:
            # 成功则重置失败计数
            if symbol in self.failed_symbols:
                del self.failed_symbols[symbol]
                append_log(f"[AUTO_TRADING] Reset failure count for {symbol}")
        else:
            # 失败则增加计数
            self.failed_symbols[symbol] = self.failed_symbols.get(symbol, 0) + 1
            append_log(f"[AUTO_TRADING] {symbol} failed {self.failed_symbols[symbol]} times")

            # 如果达到最大失败次数，加入黑名单
            if self.failed_symbols[symbol] >= self.max_failures:
                append_log(f"[AUTO_TRADING] {symbol} added to blacklist after {self.failed_symbols[symbol]} failures")

    def reset_blacklist(self):
        """重置黑名单，清除失败的股票记录"""
        self.failed_symbols.clear()
        append_log(f"[AUTO_TRADING] Blacklist reset - all failed symbols cleared")

    def get_trading_summary(self) -> Dict:
        """获取交易摘要"""
        return {
            'daily_trade_count': self.daily_trade_count,
            'max_daily_trades': self.max_daily_trades,
            'total_trades': len(self.trade_history),
            'dry_run': self.dry_run,
            'last_trades': self.trade_history[-5:] if self.trade_history else [],
            'failed_symbols': self.failed_symbols,
            'blacklisted_symbols': [symbol for symbol, count in self.failed_symbols.items() if count >= self.max_failures],
            'dynamic_config': {
                'trading_safety_factor': self.trading_safety_factor,
                'min_cash_reserve': self.min_cash_reserve,
                'max_position_percent': self.max_position_percent,
                'fallback_buying_power': self.fallback_buying_power,
                'max_daily_outflow_percent': self.max_daily_outflow_percent
            }
        }

    def get_fund_management_status(self) -> Dict:
        """获取资金管理状态"""
        raw_buying_power = self.get_raw_buying_power()
        safe_buying_power = self.get_buying_power()
        max_position_value = self.get_max_position_value(safe_buying_power)

        return {
            'raw_buying_power': raw_buying_power,
            'safe_buying_power': safe_buying_power,
            'cash_reserve': self.min_cash_reserve,
            'safety_factor': self.trading_safety_factor,
            'max_position_value': max_position_value,
            'max_position_percent': self.max_position_percent,
            'max_daily_outflow': safe_buying_power * self.max_daily_outflow_percent,
            'utilization_rate': (raw_buying_power - safe_buying_power) / raw_buying_power if raw_buying_power > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }