import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { formatCurrency, formatDate, formatPercent, getChangeColor, truncateText } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import AlertsPanel from '@/components/AlertsPanel'
import CostAnalysisPanel from '@/components/CostAnalysisPanel'
import MarketStateIndicator from '@/components/MarketStateIndicator'
import ConsoleLogs from '@/components/ConsoleLogs'
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  PieChart,
  Activity
} from 'lucide-react'

export default function Dashboard() {
  const [botReason, setBotReason] = useState<string>('')
  const [botActionStatus, setBotActionStatus] = useState<{type: 'idle' | 'working' | 'success' | 'error'; message?: string}>({
    type: 'idle',
  })
  const [strategyProfileId, setStrategyProfileId] = useState<string>('')
  const [strategyActionStatus, setStrategyActionStatus] = useState<{type: 'idle' | 'working' | 'success' | 'error'; message?: string}>({
    type: 'idle',
  })

  // Fetch portfolio data
  const { data: portfolioData, isLoading: portfolioLoading } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: () => apiClient.getPortfolioSummary(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Fetch positions data
  const { data: positionsData, isLoading: positionsLoading } = useQuery({
    queryKey: ['positions'],
    queryFn: () => apiClient.getPositions(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Fetch market state
  const { data: marketStateData } = useQuery({
    queryKey: ['market-state'],
    queryFn: () => apiClient.getMarketState(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: selectionData, isLoading: selectionLoading } = useQuery({
    queryKey: ['stock-selection'],
    queryFn: () => apiClient.getStockSelection({ limit: 10 }),
    refetchInterval: 60000, // Refresh every 60 seconds
  })

  const { data: systemStatusData, refetch: refetchSystemStatus } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => apiClient.getSystemStatus(),
    refetchInterval: 10000,
  })

  const { data: strategyProfilesData, isLoading: strategyProfilesLoading, refetch: refetchStrategyProfiles } = useQuery({
    queryKey: ['strategy-profiles'],
    queryFn: () => apiClient.getStrategyProfiles(),
  })

  const portfolio = portfolioData?.data
  const positions = positionsData?.data || []
  const marketState = marketStateData?.data
  const selection = selectionData?.data
  const systemStatus = systemStatusData?.data
  const strategyProfiles = strategyProfilesData?.data?.profiles || {}
  const activeStrategyProfile = strategyProfilesData?.data?.active_profile || ''
  const botPaused = Boolean(systemStatus?.paused || systemStatus?.bot === 'paused')
  const selectionStocks = selection?.stocks || []
  const dailyPnl = portfolio?.daily_pnl
  const dailyPnlIcon = dailyPnl !== undefined && dailyPnl >= 0 ? TrendingUp : TrendingDown

  // Debug logging
  console.log('[Dashboard] Portfolio data:', portfolioData)
  console.log('[Dashboard] Positions data:', positionsData)
  console.log('[Dashboard] Portfolio extracted:', portfolio)
  console.log('[Dashboard] Positions extracted:', positions)

  useEffect(() => {
    if (!strategyProfileId && activeStrategyProfile) {
      setStrategyProfileId(activeStrategyProfile)
    }
  }, [activeStrategyProfile, strategyProfileId])

  useEffect(() => {
    if (strategyProfileId && strategyProfileId !== activeStrategyProfile) {
      setStrategyActionStatus({ type: 'idle' })
    }
  }, [strategyProfileId, activeStrategyProfile])

  const statsCards = [
    {
      title: 'Portfolio Value',
      value: portfolio ? formatCurrency(portfolio.total_value) : '-',
      change: portfolio ? formatPercent(portfolio.total_pnl_percent) : '-',
      changeValue: portfolio ? formatCurrency(portfolio.total_pnl) : '-',
      icon: DollarSign,
      isPositive: portfolio ? portfolio.total_pnl >= 0 : null,
    },
    {
      title: 'Daily P&L',
      value: portfolio ? formatCurrency(portfolio.daily_pnl) : '-',
      change: portfolio ? formatPercent(portfolio.daily_pnl_percent) : '-',
      changeValue: null,
      icon: dailyPnlIcon,
      isPositive: dailyPnl !== undefined ? dailyPnl >= 0 : null,
    },
    {
      title: 'Active Positions',
      value: portfolio ? portfolio.positions_count.toString() : '-',
      change: positions.length > 0 ? `${positions.filter(p => p.unrealized_pnl > 0).length} profitable` : '-',
      changeValue: null,
      icon: PieChart,
      isPositive: null,
    },
    {
      title: 'Buying Power',
      value: portfolio ? formatCurrency(portfolio.buying_power) : '-',
      change: portfolio ? `${((portfolio.buying_power / portfolio.total_value) * 100).toFixed(1)}% available` : '-',
      changeValue: null,
      icon: Activity,
      isPositive: null,
    },
  ]

  const getActionVariant = (action?: string) => {
    if (!action) return 'secondary' as const
    const normalized = action.toLowerCase()
    if (normalized.includes('strong')) return 'success' as const
    if (normalized.includes('sell')) return 'error' as const
    if (normalized.includes('hold')) return 'warning' as const
    if (normalized.includes('buy')) return 'default' as const
    return 'secondary' as const
  }

  const formatActionLabel = (action?: string) => {
    if (!action) return 'PICK'
    return action.replace(/_/g, ' ').toUpperCase()
  }

  const handleStopBot = async () => {
    setBotActionStatus({ type: 'working' })
    try {
      const response = await apiClient.stopBot(botReason)
      setBotActionStatus({
        type: 'success',
        message: response.data?.reason ? `Stopped: ${response.data.reason}` : 'Bot stopped',
      })
      refetchSystemStatus()
    } catch (error) {
      setBotActionStatus({
        type: 'error',
        message: error instanceof Error ? error.message : 'Failed to stop bot',
      })
    }
  }

  const handleResumeBot = async () => {
    setBotActionStatus({ type: 'working' })
    try {
      await apiClient.resumeBot(botReason)
      setBotActionStatus({
        type: 'success',
        message: 'Bot resumed',
      })
      refetchSystemStatus()
    } catch (error) {
      setBotActionStatus({
        type: 'error',
        message: error instanceof Error ? error.message : 'Failed to resume bot',
      })
    }
  }

  const handleApplyStrategy = async () => {
    if (!strategyProfileId) return
    setStrategyActionStatus({ type: 'working' })
    try {
      await apiClient.setActiveStrategy(strategyProfileId)
      await apiClient.restartRunner(`strategy_switch:${strategyProfileId}`)
      setStrategyActionStatus({
        type: 'success',
        message: `Strategy ${strategyProfileId} applied. Runner restart requested.`,
      })
      refetchStrategyProfiles()
      refetchSystemStatus()
    } catch (error) {
      setStrategyActionStatus({
        type: 'error',
        message: error instanceof Error ? error.message : 'Failed to apply strategy',
      })
    }
  }

  return (
    <div className="space-y-6">
      {/* Market State Indicator */}
      {marketState && (
        <MarketStateIndicator marketState={marketState} />
      )}

      {/* Portfolio Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statsCards.map((stat, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {stat.title}
              </CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold font-mono-numbers">
                {portfolioLoading ? (
                  <div className="h-8 w-24 loading-skeleton" />
                ) : (
                  stat.value
                )}
              </div>
              {stat.change && (
                <div className="flex items-center space-x-2 text-sm">
                  <span className={stat.isPositive !== null ? getChangeColor(stat.isPositive ? 1 : -1) : 'text-muted-foreground'}>
                    {stat.change}
                  </span>
                  {stat.changeValue && (
                    <span className="text-muted-foreground">
                      ({stat.changeValue})
                    </span>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Bot Control */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Bot Control</span>
          </CardTitle>
          <CardDescription>
            Pause or resume the trading runner. Changes take effect immediately.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <div className="space-y-1 min-w-[220px]">
                <Label htmlFor="bot_reason_dashboard">Reason (optional)</Label>
                <Input
                  id="bot_reason_dashboard"
                  value={botReason}
                  onChange={(e) => setBotReason(e.target.value)}
                  placeholder="Manual pause"
                />
              </div>
              <div className="flex flex-wrap items-center gap-3 pt-5">
                <Button
                  variant="destructive"
                  onClick={handleStopBot}
                  disabled={botActionStatus.type === 'working' || botPaused}
                >
                  {botActionStatus.type === 'working' ? 'Stopping...' : 'Stop Bot'}
                </Button>
                <Button
                  variant="outline"
                  onClick={handleResumeBot}
                  disabled={botActionStatus.type === 'working' || !botPaused}
                >
                  {botActionStatus.type === 'working' ? 'Starting...' : 'Start Bot'}
                </Button>
                <span className="text-xs text-muted-foreground">
                  Status: {botPaused ? 'Paused' : 'Running'}
                </span>
              </div>
            </div>
            {botActionStatus.type === 'success' && (
              <div className="text-sm text-bull">{botActionStatus.message}</div>
            )}
            {botActionStatus.type === 'error' && (
              <div className="text-sm text-bear">{botActionStatus.message}</div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Strategy Switcher */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Strategy Switcher</span>
          </CardTitle>
          <CardDescription>
            Switch selection/trading profiles and restart the runner to apply changes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1">
                <Label htmlFor="strategy_profile">Active Profile</Label>
                <select
                  id="strategy_profile"
                  className="h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={strategyProfileId}
                  onChange={(e) => setStrategyProfileId(e.target.value)}
                  disabled={strategyProfilesLoading}
                >
                  {!strategyProfileId && (
                    <option value="">Select a profile</option>
                  )}
                  {Object.entries(strategyProfiles).map(([id, profile]) => (
                    <option key={id} value={id}>
                      {id} - {(profile as {name?: string}).name || 'Profile'}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-1">
                <Label>Description</Label>
                <div className="min-h-[40px] rounded-md border border-muted bg-muted/40 px-3 py-2 text-sm">
                  {(strategyProfiles[strategyProfileId] as {description?: string})?.description || '—'}
                </div>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={handleApplyStrategy}
                disabled={strategyActionStatus.type === 'working' || !strategyProfileId}
              >
                {strategyActionStatus.type === 'working' ? 'Applying...' : 'Apply & Restart Bot'}
              </Button>
              <Button variant="outline" onClick={() => refetchStrategyProfiles()}>
                Reload
              </Button>
              <span className="text-xs text-muted-foreground">
                Current: {activeStrategyProfile || 'Unknown'}
              </span>
            </div>
            {strategyActionStatus.type === 'success' && (
              <div className="text-sm text-bull">{strategyActionStatus.message}</div>
            )}
            {strategyActionStatus.type === 'error' && (
              <div className="text-sm text-bear">{strategyActionStatus.message}</div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Positions Overview */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Current Positions</CardTitle>
            <CardDescription>
              Real-time position monitoring with P&L
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
                {positions.slice(0, 8).map((position) => (
                  <div
                    key={position.symbol}
                    className="flex items-center justify-between p-3 rounded-lg border"
                  >
                    <div className="flex items-center space-x-4">
                      <div>
                        <div className="font-semibold">{position.symbol}</div>
                        <div className="text-sm text-muted-foreground">
                          {position.quantity} shares @ {formatCurrency(position.avg_price)}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-mono-numbers font-semibold">
                        {formatCurrency(position.market_value)}
                      </div>
                      <div className={`text-sm font-mono-numbers ${getChangeColor(position.unrealized_pnl)}`}>
                        {formatCurrency(position.unrealized_pnl)} ({formatPercent(position.unrealized_pnl_percent)})
                      </div>
                    </div>
                  </div>
                ))}
                {positions.length > 8 && (
                  <div className="text-center text-sm text-muted-foreground">
                    +{positions.length - 8} more positions
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                No active positions
              </div>
            )}
          </CardContent>
        </Card>

        {/* Cost Analysis Panel */}
        <div className="space-y-6">
          <CostAnalysisPanel />
          <AlertsPanel />
        </div>
      </div>

      {/* Stock Selection Results */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
            <div>
              <CardTitle>Stock Selection</CardTitle>
              <CardDescription>
                Latest stock picks from the selection engine
              </CardDescription>
              {selection?.strategy && selection?.strategy !== 'unknown' && (
                <Badge variant="outline" className="mt-2">
                  {selection.strategy}
                </Badge>
              )}
            </div>
            <div className="text-left sm:text-right">
              <div className="text-xs text-muted-foreground">Selection Date</div>
              <div className="text-sm font-mono-numbers">
                {selection?.timestamp ? formatDate(selection.timestamp) : '-'}
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {selectionLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-14 loading-skeleton" />
              ))}
            </div>
          ) : selectionStocks.length > 0 ? (
            <div className="space-y-3">
              {selectionStocks.map((stock, index) => {
                const rank = stock.rank ?? index + 1
                const reasoning = typeof stock.metrics?.reasoning === 'string'
                  ? stock.metrics.reasoning
                  : ''
                const componentEntries = Object.entries(stock.component_scores || {})
                  .filter(([, value]) => typeof value === 'number')

                return (
                  <div key={`${stock.symbol}-${rank}`} className="rounded-lg border p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-mono-numbers text-muted-foreground">#{rank}</span>
                          <span className="font-semibold">{stock.symbol}</span>
                          {stock.action && (
                            <Badge variant={getActionVariant(stock.action)} className="text-xs">
                              {formatActionLabel(stock.action)}
                            </Badge>
                          )}
                        </div>
                        {reasoning && (
                          <div className="text-xs text-muted-foreground mt-1">
                            {truncateText(reasoning, 140)}
                          </div>
                        )}
                      </div>
                      <div className="text-right">
                        {typeof stock.score === 'number' && (
                          <div className="text-sm font-mono-numbers">
                            Score {stock.score.toFixed(1)}
                          </div>
                        )}
                        {typeof stock.confidence === 'number' && (
                          <div className="text-xs text-muted-foreground">
                            Conf {Math.round(stock.confidence * 100)}%
                          </div>
                        )}
                      </div>
                    </div>
                    {componentEntries.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-3">
                        {componentEntries.map(([key, value]) => (
                          <Badge key={key} variant="outline" className="text-xs capitalize">
                            {key.replace(/_/g, ' ')} {Number(value).toFixed(1)}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">
              No selection results available yet.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Risk Metrics */}
      {portfolio && (
        <Card>
          <CardHeader>
            <CardTitle>Risk Metrics</CardTitle>
            <CardDescription>
              Portfolio risk analysis and performance metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {portfolio.risk_metrics.sharpe_ratio.toFixed(2)}
                </div>
                <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {formatPercent(portfolio.risk_metrics.max_drawdown)}
                </div>
                <div className="text-sm text-muted-foreground">Max Drawdown</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {portfolio.risk_metrics.portfolio_beta.toFixed(2)}
                </div>
                <div className="text-sm text-muted-foreground">Portfolio Beta</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {formatPercent(portfolio.risk_metrics.volatility)}
                </div>
                <div className="text-sm text-muted-foreground">Volatility</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Runner Console Logs */}
      <ConsoleLogs />
    </div>
  )
}
