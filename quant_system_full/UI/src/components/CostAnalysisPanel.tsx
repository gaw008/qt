import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { formatBasisPoints, formatCurrency } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Calculator,
  AlertCircle,
  CheckCircle
} from 'lucide-react'

export default function CostAnalysisPanel() {
  // Fetch recent orders with cost analysis
  const { data: ordersData, isLoading } = useQuery({
    queryKey: ['orders'],
    queryFn: () => apiClient.getOrders({ limit: 10 }),
    refetchInterval: 10000,
  })

  const orders = ordersData?.data || []
  const ordersWithCost = orders.filter(order => order.cost_analysis)

  // Calculate cost statistics
  const avgCostBps = ordersWithCost.length > 0
    ? ordersWithCost.reduce((sum, order) => sum + (order.cost_analysis?.cost_basis_points || 0), 0) / ordersWithCost.length
    : 0

  const totalCostSaved = ordersWithCost.reduce((sum, order) => {
    const costBps = order.cost_analysis?.cost_basis_points || 0
    if (costBps < 15) { // Saved vs. typical market order
      const orderValue = order.quantity * (order.avg_fill_price || order.price || 0)
      return sum + (orderValue * (15 - costBps) / 10000)
    }
    return sum
  }, 0)

  const getCostStatus = (costBps: number) => {
    if (costBps <= 15) return { status: 'Low', variant: 'success' as const, icon: CheckCircle }
    if (costBps <= 30) return { status: 'Medium', variant: 'warning' as const, icon: AlertCircle }
    return { status: 'High', variant: 'error' as const, icon: AlertCircle }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Calculator className="h-5 w-5" />
          <span>Cost Analysis</span>
        </CardTitle>
        <CardDescription>
          Real-time trading cost optimization
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            <div className="h-6 loading-skeleton" />
            <div className="h-4 loading-skeleton" />
            <div className="h-4 loading-skeleton" />
          </div>
        ) : (
          <div className="space-y-4">
            {/* Cost Summary */}
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-lg font-bold font-mono-numbers">
                  {formatBasisPoints(avgCostBps)}
                </div>
                <div className="text-xs text-muted-foreground">Avg Cost</div>
              </div>
              <div className="text-center p-3 bg-bull/10 rounded-lg">
                <div className="text-lg font-bold font-mono-numbers text-bull">
                  {formatCurrency(totalCostSaved)}
                </div>
                <div className="text-xs text-muted-foreground">Saved Today</div>
              </div>
            </div>

            {/* Recent Orders with Cost Analysis */}
            <div className="space-y-2">
              <div className="text-sm font-medium">Recent Orders</div>
              {ordersWithCost.length > 0 ? (
                ordersWithCost.slice(0, 5).map((order) => {
                  const costBps = order.cost_analysis?.cost_basis_points || 0
                  const costStatus = getCostStatus(costBps)
                  const StatusIcon = costStatus.icon

                  return (
                    <div
                      key={order.id}
                      className="flex items-center justify-between p-2 rounded border"
                    >
                      <div className="flex items-center space-x-2">
                        <StatusIcon className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <div className="text-sm font-medium">{order.symbol}</div>
                          <div className="text-xs text-muted-foreground">
                            {order.side.toUpperCase()} {order.quantity}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant={costStatus.variant} className="text-xs">
                          {formatBasisPoints(costBps)}
                        </Badge>
                        <div className="text-xs text-muted-foreground mt-1">
                          {costStatus.status} Cost
                        </div>
                      </div>
                    </div>
                  )
                })
              ) : (
                <div className="text-center text-sm text-muted-foreground py-4">
                  No recent orders with cost analysis
                </div>
              )}
            </div>

            {/* Cost Optimization Status */}
            <div className="p-3 bg-muted/30 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Optimization Active</span>
                <div className="flex items-center space-x-1">
                  <div className="h-2 w-2 bg-bull rounded-full animate-pulse" />
                  <span className="text-xs text-bull">Live</span>
                </div>
              </div>
              <div className="text-xs text-muted-foreground">
                Automatically optimizing order types and timing to minimize execution costs
              </div>
            </div>

            {/* Cost Breakdown */}
            <div className="space-y-2">
              <div className="text-sm font-medium">Cost Breakdown</div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Spread Cost:</span>
                  <span>~{formatBasisPoints(avgCostBps * 0.6)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Market Impact:</span>
                  <span>~{formatBasisPoints(avgCostBps * 0.3)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Timing Cost:</span>
                  <span>~{formatBasisPoints(avgCostBps * 0.1)}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
