import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { formatCurrency, formatPercent, getChangeColor } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  DollarSign,
  Plus,
  X,
  AlertTriangle,
  Clock,
  CheckCircle,
  Zap,
  ArrowUpCircle,
  ArrowDownCircle
} from 'lucide-react'

interface RecentTrade {
  symbol: string
  action: string
  qty: number
  price: number
  reason: string
  success: boolean
  order_id: string
  timestamp: string
}

export default function Trading() {
  const [orderForm, setOrderForm] = useState({
    symbol: '',
    side: 'buy' as 'buy' | 'sell',
    type: 'market' as 'market' | 'limit' | 'stop' | 'stop_limit',
    quantity: '',
    price: '',
    stop_price: ''
  })

  const queryClient = useQueryClient()

  // Fetch positions
  const { data: positionsData, isLoading: positionsLoading } = useQuery({
    queryKey: ['positions'],
    queryFn: () => apiClient.getPositions(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Fetch orders
  const { data: ordersData, isLoading: ordersLoading } = useQuery({
    queryKey: ['orders'],
    queryFn: () => apiClient.getOrders({ limit: 20 }),
    refetchInterval: 15000, // Refresh every 15 seconds
  })

  // Fetch portfolio summary
  const { data: portfolioData } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: () => apiClient.getPortfolioSummary(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Fetch system status for recent auto trades
  const { data: systemStatusData, isLoading: systemStatusLoading } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => apiClient.getSystemStatus(),
    refetchInterval: 15000, // Refresh every 15 seconds
  })

  const positions = positionsData?.data || []
  const orders = ordersData?.data || []
  const portfolio = portfolioData?.data
  const recentAutoTrades: RecentTrade[] = systemStatusData?.data?.intraday_recent_trades || []

  // Create order mutation
  const createOrderMutation = useMutation({
    mutationFn: (order: any) => apiClient.createOrder(order),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] })
      queryClient.invalidateQueries({ queryKey: ['positions'] })
      queryClient.invalidateQueries({ queryKey: ['portfolio-summary'] })
      setOrderForm({
        symbol: '',
        side: 'buy',
        type: 'market',
        quantity: '',
        price: '',
        stop_price: ''
      })
    },
  })

  const orderError = createOrderMutation.error instanceof Error
    ? createOrderMutation.error
    : null
  const orderErrorMessage = orderError ? orderError.message : null

  const handleSubmitOrder = (e: React.FormEvent) => {
    e.preventDefault()

    const order = {
      symbol: orderForm.symbol.toUpperCase(),
      side: orderForm.side,
      type: orderForm.type,
      quantity: parseInt(orderForm.quantity),
      ...(orderForm.type !== 'market' && orderForm.price && { price: parseFloat(orderForm.price) }),
      ...(orderForm.type.includes('stop') && orderForm.stop_price && { stop_price: parseFloat(orderForm.stop_price) })
    }

    createOrderMutation.mutate(order)
  }

  const getOrderStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'filled':
        return CheckCircle
      case 'pending':
        return Clock
      case 'cancelled':
      case 'rejected':
        return X
      default:
        return AlertTriangle
    }
  }

  const getOrderStatusVariant = (status: string) => {
    switch (status.toLowerCase()) {
      case 'filled':
        return 'success' as const
      case 'pending':
      case 'partially_filled':
        return 'warning' as const
      case 'cancelled':
      case 'rejected':
        return 'error' as const
      default:
        return 'secondary' as const
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Trading</h1>
        <p className="text-muted-foreground">Manage positions and place orders</p>
      </div>

      {/* Trading Stats */}
      {portfolio && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {formatCurrency(portfolio.buying_power)}
                </div>
                <div className="text-sm text-muted-foreground">Buying Power</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {portfolio.positions_count}
                </div>
                <div className="text-sm text-muted-foreground">Open Positions</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className={`text-2xl font-bold font-mono-numbers ${getChangeColor(portfolio.daily_pnl)}`}>
                  {formatCurrency(portfolio.daily_pnl)}
                </div>
                <div className="text-sm text-muted-foreground">Daily P&L</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {orders.filter(o => o.status === 'pending').length}
                </div>
                <div className="text-sm text-muted-foreground">Pending Orders</div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Order Entry */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Plus className="h-5 w-5" />
              <span>Place Order</span>
            </CardTitle>
            <CardDescription>
              Enter order details to buy or sell securities
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmitOrder} className="space-y-4">
              {/* Symbol */}
              <div>
                <label className="text-sm font-medium">Symbol</label>
                <input
                  type="text"
                  value={orderForm.symbol}
                  onChange={(e) => setOrderForm({ ...orderForm, symbol: e.target.value })}
                  placeholder="AAPL"
                  className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                  required
                />
              </div>

              {/* Side */}
              <div>
                <label className="text-sm font-medium">Side</label>
                <div className="grid grid-cols-2 gap-2 mt-1">
                  <Button
                    type="button"
                    variant={orderForm.side === 'buy' ? 'default' : 'outline'}
                    onClick={() => setOrderForm({ ...orderForm, side: 'buy' })}
                    className="w-full"
                  >
                    Buy
                  </Button>
                  <Button
                    type="button"
                    variant={orderForm.side === 'sell' ? 'default' : 'outline'}
                    onClick={() => setOrderForm({ ...orderForm, side: 'sell' })}
                    className="w-full"
                  >
                    Sell
                  </Button>
                </div>
              </div>

              {/* Order Type */}
              <div>
                <label className="text-sm font-medium">Order Type</label>
                <select
                  value={orderForm.type}
                  onChange={(e) => setOrderForm({ ...orderForm, type: e.target.value as any })}
                  className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                >
                  <option value="market">Market</option>
                  <option value="limit">Limit</option>
                  <option value="stop">Stop</option>
                  <option value="stop_limit">Stop Limit</option>
                </select>
              </div>

              {/* Quantity */}
              <div>
                <label className="text-sm font-medium">Quantity</label>
                <input
                  type="number"
                  value={orderForm.quantity}
                  onChange={(e) => setOrderForm({ ...orderForm, quantity: e.target.value })}
                  placeholder="100"
                  min="1"
                  className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                  required
                />
              </div>

              {/* Price (for limit orders) */}
              {orderForm.type !== 'market' && (
                <div>
                  <label className="text-sm font-medium">Price</label>
                  <input
                    type="number"
                    value={orderForm.price}
                    onChange={(e) => setOrderForm({ ...orderForm, price: e.target.value })}
                    placeholder="150.00"
                    step="0.01"
                    className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                    required
                  />
                </div>
              )}

              {/* Stop Price (for stop orders) */}
              {orderForm.type.includes('stop') && (
                <div>
                  <label className="text-sm font-medium">Stop Price</label>
                  <input
                    type="number"
                    value={orderForm.stop_price}
                    onChange={(e) => setOrderForm({ ...orderForm, stop_price: e.target.value })}
                    placeholder="145.00"
                    step="0.01"
                    className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                    required
                  />
                </div>
              )}

              <Button
                type="submit"
                className="w-full"
                disabled={createOrderMutation.isPending}
              >
                {createOrderMutation.isPending ? 'Placing Order...' : 'Place Order'}
              </Button>

              {orderErrorMessage && (
                <div className="text-sm text-bear">
                  Error: {orderErrorMessage}
                </div>
              )}
            </form>
          </CardContent>
        </Card>

        {/* Current Positions */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <DollarSign className="h-5 w-5" />
              <span>Current Positions</span>
            </CardTitle>
            <CardDescription>
              Active trading positions with real-time P&L
            </CardDescription>
          </CardHeader>
          <CardContent>
            {positionsLoading ? (
              <div className="space-y-3">
                {[1, 2, 3].map(i => (
                  <div key={i} className="h-16 loading-skeleton" />
                ))}
              </div>
            ) : positions.length > 0 ? (
              <div className="space-y-4">
                {positions.map((position) => (
                  <div
                    key={position.symbol}
                    className="flex items-center justify-between p-4 rounded-lg border"
                  >
                    <div className="flex items-center space-x-4">
                      <div>
                        <div className="font-semibold text-lg">{position.symbol}</div>
                        <div className="text-sm text-muted-foreground">
                          {position.quantity} shares @ {formatCurrency(position.avg_price)}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Entry: {new Date(position.entry_time).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-mono-numbers font-semibold text-lg">
                        {formatCurrency(position.market_value)}
                      </div>
                      <div className={`font-mono-numbers ${getChangeColor(position.unrealized_pnl)}`}>
                        {formatCurrency(position.unrealized_pnl)} ({formatPercent(position.unrealized_pnl_percent)})
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Current: {formatCurrency(position.current_price)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                No open positions
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent Orders */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Clock className="h-5 w-5" />
            <span>Recent Orders</span>
          </CardTitle>
          <CardDescription>
            Order history and execution status
          </CardDescription>
        </CardHeader>
        <CardContent>
          {ordersLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="h-16 loading-skeleton" />
              ))}
            </div>
          ) : orders.length > 0 ? (
            <div className="space-y-3">
              {orders.map((order) => {
                const StatusIcon = getOrderStatusIcon(order.status)
                return (
                  <div
                    key={order.id}
                    className="flex items-center justify-between p-3 rounded border"
                  >
                    <div className="flex items-center space-x-3">
                      <StatusIcon className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="font-semibold">
                          {order.side.toUpperCase()} {order.quantity} {order.symbol}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {order.type.toUpperCase()}
                          {order.price && ` @ ${formatCurrency(order.price)}`}
                          {order.avg_fill_price && ` (Filled @ ${formatCurrency(order.avg_fill_price)})`}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {new Date(order.created_at).toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant={getOrderStatusVariant(order.status)}>
                        {order.status.toUpperCase()}
                      </Badge>
                      {order.filled_quantity && order.filled_quantity > 0 && (
                        <div className="text-xs text-muted-foreground mt-1">
                          Filled: {order.filled_quantity}/{order.quantity}
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              No recent orders
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Auto Trades (Intraday) */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="h-5 w-5 text-yellow-500" />
            <span>Recent Auto Trades</span>
          </CardTitle>
          <CardDescription>
            Automated intraday trades with execution reasons
          </CardDescription>
        </CardHeader>
        <CardContent>
          {systemStatusLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="h-16 loading-skeleton" />
              ))}
            </div>
          ) : recentAutoTrades.length > 0 ? (
            <div className="space-y-3">
              {recentAutoTrades.map((trade, index) => {
                const isBuy = trade.action?.toUpperCase() === 'BUY'
                const ActionIcon = isBuy ? ArrowUpCircle : ArrowDownCircle
                const actionColor = isBuy ? 'text-bull' : 'text-bear'
                return (
                  <div
                    key={`${trade.symbol}-${trade.timestamp}-${index}`}
                    className="flex items-center justify-between p-3 rounded border"
                  >
                    <div className="flex items-center space-x-3">
                      <ActionIcon className={`h-5 w-5 ${actionColor}`} />
                      <div>
                        <div className="font-semibold">
                          <span className={actionColor}>{trade.action?.toUpperCase()}</span>
                          {' '}{trade.qty} {trade.symbol}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          @ {formatCurrency(trade.price)}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {new Date(trade.timestamp).toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right max-w-[200px]">
                      <Badge variant={trade.success ? 'success' : 'error'}>
                        {trade.success ? 'SUCCESS' : 'FAILED'}
                      </Badge>
                      <div className="text-xs text-muted-foreground mt-1 break-words">
                        {trade.reason}
                      </div>
                      {trade.order_id && (
                        <div className="text-xs text-muted-foreground mt-1 font-mono">
                          ID: {trade.order_id.slice(0, 12)}...
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              No recent auto trades
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
