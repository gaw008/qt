import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { formatPercent, getChangeColor } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Target,
  TrendingUp,
  Activity,
  Settings,
  BarChart3,
  RefreshCw,
  Zap,
  Shield
} from 'lucide-react'

interface Strategy {
  id: string
  name: string
  description: string
  status: 'active' | 'inactive' | 'testing'
  weight: number
  performance: {
    total_return: number
    sharpe_ratio: number
    win_rate: number
    max_drawdown: number
    total_trades: number
    profitable_trades: number
  }
  factors: {
    valuation: number
    momentum: number
    technical: number
    volume: number
    market_sentiment: number
  }
  riskLevel: 'low' | 'medium' | 'high'
}

export default function Strategies() {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const [intradayForm, setIntradayForm] = useState<Record<string, string> | null>(null)
  const [intradaySaveStatus, setIntradaySaveStatus] = useState<{type: 'idle' | 'saving' | 'success' | 'error'; message?: string}>({
    type: 'idle',
  })

  // Fetch market state for factor weights
  const { data: marketStateData, isLoading: marketStateLoading, refetch: refetchMarketState } = useQuery({
    queryKey: ['market-state'],
    queryFn: () => apiClient.getMarketState(),
    refetchInterval: 30000,
  })

  // Fetch system status for strategy info
  const { data: systemStatusData, isLoading: systemStatusLoading } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => apiClient.getSystemStatus(),
    refetchInterval: 10000,
  })

  const { data: intradayConfigData, isLoading: intradayConfigLoading, refetch: refetchIntradayConfig } = useQuery({
    queryKey: ['intraday-config'],
    queryFn: () => apiClient.getIntradayConfig(),
  })

  const marketState = marketStateData?.data
  const systemStatus = systemStatusData?.data
  const intradayConfig = intradayConfigData?.data?.config

  useEffect(() => {
    if (!intradayConfig || intradayForm) return
    const formatted: Record<string, string> = {}
    Object.entries(intradayConfig).forEach(([key, value]) => {
      formatted[key] = value !== undefined && value !== null ? String(value) : ''
    })
    setIntradayForm(formatted)
  }, [intradayConfig, intradayForm])

  // Mock strategies data (in production, this would come from API)
  const strategies: Strategy[] = [
    {
      id: 'value_momentum',
      name: 'Value Momentum',
      description: 'Combines undervaluation metrics with positive momentum signals',
      status: 'active',
      weight: 0.35,
      performance: {
        total_return: 0.247,
        sharpe_ratio: 1.82,
        win_rate: 0.64,
        max_drawdown: -0.125,
        total_trades: 342,
        profitable_trades: 219
      },
      factors: {
        valuation: 0.40,
        momentum: 0.30,
        technical: 0.15,
        volume: 0.10,
        market_sentiment: 0.05
      },
      riskLevel: 'medium'
    },
    {
      id: 'technical_breakout',
      name: 'Technical Breakout',
      description: 'Identifies resistance breaks with volume confirmation',
      status: 'active',
      weight: 0.25,
      performance: {
        total_return: 0.198,
        sharpe_ratio: 1.54,
        win_rate: 0.58,
        max_drawdown: -0.156,
        total_trades: 287,
        profitable_trades: 166
      },
      factors: {
        valuation: 0.10,
        momentum: 0.20,
        technical: 0.45,
        volume: 0.20,
        market_sentiment: 0.05
      },
      riskLevel: 'high'
    },
    {
      id: 'earnings_momentum',
      name: 'Earnings Momentum',
      description: 'Growth stocks with positive earnings surprises',
      status: 'active',
      weight: 0.20,
      performance: {
        total_return: 0.312,
        sharpe_ratio: 2.01,
        win_rate: 0.71,
        max_drawdown: -0.098,
        total_trades: 156,
        profitable_trades: 111
      },
      factors: {
        valuation: 0.25,
        momentum: 0.35,
        technical: 0.15,
        volume: 0.15,
        market_sentiment: 0.10
      },
      riskLevel: 'medium'
    },
    {
      id: 'defensive_quality',
      name: 'Defensive Quality',
      description: 'High quality stocks with low volatility profiles',
      status: 'inactive',
      weight: 0.10,
      performance: {
        total_return: 0.142,
        sharpe_ratio: 1.67,
        win_rate: 0.69,
        max_drawdown: -0.067,
        total_trades: 98,
        profitable_trades: 68
      },
      factors: {
        valuation: 0.30,
        momentum: 0.10,
        technical: 0.10,
        volume: 0.05,
        market_sentiment: 0.45
      },
      riskLevel: 'low'
    },
    {
      id: 'market_neutral',
      name: 'Market Neutral',
      description: 'Long/short strategy minimizing market exposure',
      status: 'testing',
      weight: 0.10,
      performance: {
        total_return: 0.089,
        sharpe_ratio: 1.23,
        win_rate: 0.55,
        max_drawdown: -0.045,
        total_trades: 234,
        profitable_trades: 129
      },
      factors: {
        valuation: 0.20,
        momentum: 0.20,
        technical: 0.20,
        volume: 0.20,
        market_sentiment: 0.20
      },
      riskLevel: 'low'
    }
  ]

  const getStrategyStatusVariant = (status: string) => {
    switch (status) {
      case 'active':
        return 'success' as const
      case 'testing':
        return 'warning' as const
      case 'inactive':
        return 'secondary' as const
      default:
        return 'secondary' as const
    }
  }

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'low':
        return 'text-bull'
      case 'medium':
        return 'text-yellow-600'
      case 'high':
        return 'text-bear'
      default:
        return 'text-muted-foreground'
    }
  }

  const intradayNumberFields = new Set([
    'lookback_bars',
    'fast_ema',
    'slow_ema',
    'atr_period',
    'trail_atr',
    'hard_stop_atr',
    'momentum_lookback',
    'min_volume_ratio',
    'entry_score_threshold',
    'weight_power',
    'max_positions',
    'max_position_percent',
    'min_trade_value',
    'min_data_coverage',
    'cooldown_seconds',
    'buy_price_buffer_pct',
    'commission_per_share',
    'min_commission',
    'fee_per_order',
    'slippage_bps',
    'max_daily_cost_pct',
    'max_daily_loss_pct',
    'open_buffer_minutes',
  ])

  const handleIntradayChange = (key: string, value: string) => {
    setIntradayForm((prev) => ({
      ...(prev || {}),
      [key]: value,
    }))
  }

  const handleSaveIntradayConfig = async () => {
    if (!intradayForm) return
    setIntradaySaveStatus({ type: 'saving' })
    const payload: Record<string, any> = {}
    for (const [key, value] of Object.entries(intradayForm)) {
      if (value === '') continue
      if (intradayNumberFields.has(key)) {
        const numericValue = Number(value)
        if (Number.isNaN(numericValue)) {
          setIntradaySaveStatus({ type: 'error', message: `Invalid number for ${key}` })
          return
        }
        payload[key] = numericValue
      } else {
        payload[key] = value
      }
    }
    try {
      const response = await apiClient.updateIntradayConfig(payload)
      const updatedConfig = response.data?.config
      if (updatedConfig) {
        const formatted: Record<string, string> = {}
        Object.entries(updatedConfig).forEach(([key, value]) => {
          formatted[key] = value !== undefined && value !== null ? String(value) : ''
        })
        setIntradayForm(formatted)
      }
      setIntradaySaveStatus({
        type: 'success',
        message: response.message || 'Config saved. Restart runner to apply changes.',
      })
    } catch (error) {
      setIntradaySaveStatus({
        type: 'error',
        message: error instanceof Error ? error.message : 'Failed to save config',
      })
    }
  }


  const activeStrategies = strategies.filter(s => s.status === 'active')
  const totalReturn = activeStrategies.reduce((sum, s) => sum + (s.performance.total_return * s.weight), 0)
  const avgSharpe = activeStrategies.reduce((sum, s) => sum + (s.performance.sharpe_ratio * s.weight), 0)
  const avgWinRate = activeStrategies.reduce((sum, s) => sum + (s.performance.win_rate * s.weight), 0)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold">Strategies</h1>
          <p className="text-muted-foreground">Trading strategy selection, weights, and performance</p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={() => refetchMarketState()}
          disabled={marketStateLoading}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${marketStateLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Strategy Portfolio Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers text-bull">
                {formatPercent(totalReturn)}
              </div>
              <div className="text-sm text-muted-foreground">Blended Return</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers">
                {avgSharpe.toFixed(2)}
              </div>
              <div className="text-sm text-muted-foreground">Avg Sharpe Ratio</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers">
                {formatPercent(avgWinRate)}
              </div>
              <div className="text-sm text-muted-foreground">Avg Win Rate</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers">
                {activeStrategies.length}/{strategies.length}
              </div>
              <div className="text-sm text-muted-foreground">Active Strategies</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Strategy List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5" />
            <span>Trading Strategies</span>
          </CardTitle>
          <CardDescription>
            Active and available trading strategies with performance metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          {systemStatusLoading ? (
            <div className="space-y-4">
              {[1, 2, 3, 4, 5].map(i => (
                <div key={i} className="h-24 loading-skeleton" />
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {strategies.map((strategy) => (
                <div
                  key={strategy.id}
                  className={`p-4 rounded-lg border transition-all ${
                    selectedStrategy === strategy.id ? 'border-primary bg-accent/50' : 'hover:bg-accent/30'
                  } cursor-pointer`}
                  onClick={() => setSelectedStrategy(selectedStrategy === strategy.id ? null : strategy.id)}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div>
                        <div className="font-semibold text-lg">{strategy.name}</div>
                        <div className="text-sm text-muted-foreground">{strategy.description}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={getStrategyStatusVariant(strategy.status)}>
                        {strategy.status.toUpperCase()}
                      </Badge>
                      <span className={`text-sm font-medium ${getRiskLevelColor(strategy.riskLevel)}`}>
                        {strategy.riskLevel.toUpperCase()}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-3">
                    <div className="text-center">
                      <div className={`text-lg font-bold font-mono-numbers ${getChangeColor(strategy.performance.total_return)}`}>
                        {formatPercent(strategy.performance.total_return)}
                      </div>
                      <div className="text-xs text-muted-foreground">Return</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold font-mono-numbers">
                        {strategy.performance.sharpe_ratio.toFixed(2)}
                      </div>
                      <div className="text-xs text-muted-foreground">Sharpe</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold font-mono-numbers">
                        {formatPercent(strategy.performance.win_rate)}
                      </div>
                      <div className="text-xs text-muted-foreground">Win Rate</div>
                    </div>
                    <div className="text-center">
                      <div className={`text-lg font-bold font-mono-numbers ${getChangeColor(strategy.performance.max_drawdown)}`}>
                        {formatPercent(strategy.performance.max_drawdown)}
                      </div>
                      <div className="text-xs text-muted-foreground">Max DD</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold font-mono-numbers">
                        {strategy.performance.profitable_trades}/{strategy.performance.total_trades}
                      </div>
                      <div className="text-xs text-muted-foreground">Trades</div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">Portfolio Weight</span>
                      <span className="text-sm font-mono-numbers">{formatPercent(strategy.weight)}</span>
                    </div>
                    <Progress value={strategy.weight * 100} className="h-2" />
                  </div>

                  {selectedStrategy === strategy.id && (
                    <div className="mt-4 pt-4 border-t">
                      <div className="text-sm font-medium mb-3">Factor Weights</div>
                      <div className="space-y-3">
                        {Object.entries(strategy.factors).map(([factor, weight]) => (
                          <div key={factor}>
                            <div className="flex justify-between items-center mb-1">
                              <span className="text-xs capitalize">{factor.replace('_', ' ')}</span>
                              <span className="text-xs font-mono-numbers">{formatPercent(weight)}</span>
                            </div>
                            <Progress value={weight * 100} className="h-1.5" />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Intraday Strategy Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <span>Intraday Strategy Settings</span>
          </CardTitle>
          <CardDescription>
            5-minute trend strategy configuration. Changes require runner restart.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {intradayConfigLoading ? (
            <div className="h-24 loading-skeleton" />
          ) : intradayForm ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="space-y-1">
                  <Label htmlFor="signal_period">Signal Period</Label>
                  <Input
                    id="signal_period"
                    value={intradayForm.signal_period || ''}
                    onChange={(e) => handleIntradayChange('signal_period', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="lookback_bars">Lookback Bars</Label>
                  <Input
                    id="lookback_bars"
                    type="number"
                    value={intradayForm.lookback_bars || ''}
                    onChange={(e) => handleIntradayChange('lookback_bars', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="fast_ema">Fast EMA</Label>
                  <Input
                    id="fast_ema"
                    type="number"
                    value={intradayForm.fast_ema || ''}
                    onChange={(e) => handleIntradayChange('fast_ema', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="slow_ema">Slow EMA</Label>
                  <Input
                    id="slow_ema"
                    type="number"
                    value={intradayForm.slow_ema || ''}
                    onChange={(e) => handleIntradayChange('slow_ema', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="atr_period">ATR Period</Label>
                  <Input
                    id="atr_period"
                    type="number"
                    value={intradayForm.atr_period || ''}
                    onChange={(e) => handleIntradayChange('atr_period', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="trail_atr">Trail ATR</Label>
                  <Input
                    id="trail_atr"
                    type="number"
                    value={intradayForm.trail_atr || ''}
                    onChange={(e) => handleIntradayChange('trail_atr', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="hard_stop_atr">Hard Stop ATR</Label>
                  <Input
                    id="hard_stop_atr"
                    type="number"
                    value={intradayForm.hard_stop_atr || ''}
                    onChange={(e) => handleIntradayChange('hard_stop_atr', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="momentum_lookback">Momentum Lookback</Label>
                  <Input
                    id="momentum_lookback"
                    type="number"
                    value={intradayForm.momentum_lookback || ''}
                    onChange={(e) => handleIntradayChange('momentum_lookback', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="min_volume_ratio">Min Volume Ratio</Label>
                  <Input
                    id="min_volume_ratio"
                    type="number"
                    value={intradayForm.min_volume_ratio || ''}
                    onChange={(e) => handleIntradayChange('min_volume_ratio', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="entry_score_threshold">Entry Score</Label>
                  <Input
                    id="entry_score_threshold"
                    type="number"
                    value={intradayForm.entry_score_threshold || ''}
                    onChange={(e) => handleIntradayChange('entry_score_threshold', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="weight_power">Weight Power</Label>
                  <Input
                    id="weight_power"
                    type="number"
                    value={intradayForm.weight_power || ''}
                    onChange={(e) => handleIntradayChange('weight_power', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="max_positions">Max Positions</Label>
                  <Input
                    id="max_positions"
                    type="number"
                    value={intradayForm.max_positions || ''}
                    onChange={(e) => handleIntradayChange('max_positions', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="max_position_percent">Max Position %</Label>
                  <Input
                    id="max_position_percent"
                    type="number"
                    value={intradayForm.max_position_percent || ''}
                    onChange={(e) => handleIntradayChange('max_position_percent', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="min_trade_value">Min Trade Value</Label>
                  <Input
                    id="min_trade_value"
                    type="number"
                    value={intradayForm.min_trade_value || ''}
                    onChange={(e) => handleIntradayChange('min_trade_value', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="min_data_coverage">Min Data Coverage</Label>
                  <Input
                    id="min_data_coverage"
                    type="number"
                    value={intradayForm.min_data_coverage || ''}
                    onChange={(e) => handleIntradayChange('min_data_coverage', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="cooldown_seconds">Cooldown Seconds</Label>
                  <Input
                    id="cooldown_seconds"
                    type="number"
                    value={intradayForm.cooldown_seconds || ''}
                    onChange={(e) => handleIntradayChange('cooldown_seconds', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="buy_price_buffer_pct">Buy Price Buffer %</Label>
                  <Input
                    id="buy_price_buffer_pct"
                    type="number"
                    value={intradayForm.buy_price_buffer_pct || ''}
                    onChange={(e) => handleIntradayChange('buy_price_buffer_pct', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="commission_per_share">Commission Per Share</Label>
                  <Input
                    id="commission_per_share"
                    type="number"
                    value={intradayForm.commission_per_share || ''}
                    onChange={(e) => handleIntradayChange('commission_per_share', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="min_commission">Min Commission</Label>
                  <Input
                    id="min_commission"
                    type="number"
                    value={intradayForm.min_commission || ''}
                    onChange={(e) => handleIntradayChange('min_commission', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="fee_per_order">Fee Per Order</Label>
                  <Input
                    id="fee_per_order"
                    type="number"
                    value={intradayForm.fee_per_order || ''}
                    onChange={(e) => handleIntradayChange('fee_per_order', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="slippage_bps">Slippage (bps)</Label>
                  <Input
                    id="slippage_bps"
                    type="number"
                    value={intradayForm.slippage_bps || ''}
                    onChange={(e) => handleIntradayChange('slippage_bps', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="max_daily_cost_pct">Max Daily Cost %</Label>
                  <Input
                    id="max_daily_cost_pct"
                    type="number"
                    value={intradayForm.max_daily_cost_pct || ''}
                    onChange={(e) => handleIntradayChange('max_daily_cost_pct', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="max_daily_loss_pct">Max Daily Loss %</Label>
                  <Input
                    id="max_daily_loss_pct"
                    type="number"
                    value={intradayForm.max_daily_loss_pct || ''}
                    onChange={(e) => handleIntradayChange('max_daily_loss_pct', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="open_buffer_minutes">Open Buffer Minutes</Label>
                  <Input
                    id="open_buffer_minutes"
                    type="number"
                    value={intradayForm.open_buffer_minutes || ''}
                    onChange={(e) => handleIntradayChange('open_buffer_minutes', e.target.value)}
                  />
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={handleSaveIntradayConfig} disabled={intradaySaveStatus.type === 'saving'}>
                  {intradaySaveStatus.type === 'saving' ? 'Saving...' : 'Save Config'}
                </Button>
                <Button variant="outline" onClick={() => refetchIntradayConfig()}>
                  Reload
                </Button>
                <span className="text-xs text-muted-foreground">
                  Restart runner required after save.
                </span>
              </div>
              {intradaySaveStatus.type === 'success' && (
                <div className="text-sm text-bull">{intradaySaveStatus.message}</div>
              )}
              {intradaySaveStatus.type === 'error' && (
                <div className="text-sm text-bear">{intradaySaveStatus.message}</div>
              )}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">No intraday config found.</div>
          )}
        </CardContent>
      </Card>

      {/* Market Regime Factor Weights */}
      {marketState?.factor_weights && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Settings className="h-5 w-5" />
              <span>Market Regime Factor Weights</span>
            </CardTitle>
            <CardDescription>
              Current factor weights based on {marketState.current_state?.replace('_', ' ') || 'market conditions'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                {Object.entries(marketState.factor_weights).slice(0, 3).map(([factor, weight]) => (
                  <div key={factor}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium capitalize">{factor.replace('_', ' ')}</span>
                      <span className="text-sm font-mono-numbers">{formatPercent(weight as number)}</span>
                    </div>
                    <Progress value={(weight as number) * 100} className="h-2" />
                  </div>
                ))}
              </div>
              <div className="space-y-4">
                {Object.entries(marketState.factor_weights).slice(3).map(([factor, weight]) => (
                  <div key={factor}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium capitalize">{factor.replace('_', ' ')}</span>
                      <span className="text-sm font-mono-numbers">{formatPercent(weight as number)}</span>
                    </div>
                    <Progress value={(weight as number) * 100} className="h-2" />
                  </div>
                ))}
              </div>
            </div>

            {marketState.parameters && (
              <div className="mt-6 pt-6 border-t">
                <div className="text-sm font-medium mb-3">Market Regime Parameters</div>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div className="text-center">
                    <div className="text-lg font-bold font-mono-numbers">
                      {marketState.parameters.position_size_multiplier.toFixed(2)}x
                    </div>
                    <div className="text-xs text-muted-foreground">Position Size</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold font-mono-numbers">
                      {marketState.parameters.risk_threshold_multiplier.toFixed(2)}x
                    </div>
                    <div className="text-xs text-muted-foreground">Risk Threshold</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold font-mono-numbers">
                      {marketState.parameters.stop_loss_multiplier.toFixed(2)}x
                    </div>
                    <div className="text-xs text-muted-foreground">Stop Loss</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold font-mono-numbers">
                      {marketState.parameters.max_positions}
                    </div>
                    <div className="text-xs text-muted-foreground">Max Positions</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold font-mono-numbers">
                      {formatPercent(marketState.parameters.volatility_target)}
                    </div>
                    <div className="text-xs text-muted-foreground">Vol Target</div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Strategy Optimization */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Optimization Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-lg bg-accent/50">
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-primary" />
                  <span className="text-sm font-medium">AI Strategy Optimizer</span>
                </div>
                <Badge variant={systemStatus?.ai_learning_enabled ? 'success' : 'secondary'}>
                  {systemStatus?.ai_learning_enabled ? 'Active' : 'Inactive'}
                </Badge>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-accent/50">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="h-4 w-4 text-primary" />
                  <span className="text-sm font-medium">Performance Tracking</span>
                </div>
                <Badge variant="success">Active</Badge>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-accent/50">
                <div className="flex items-center space-x-2">
                  <Shield className="h-4 w-4 text-primary" />
                  <span className="text-sm font-medium">Risk Management</span>
                </div>
                <Badge variant="success">Active</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>Performance Summary</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm">Total Strategies</span>
                  <span className="text-sm font-mono-numbers font-semibold">{strategies.length}</span>
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm">Active Strategies</span>
                  <span className="text-sm font-mono-numbers font-semibold">{activeStrategies.length}</span>
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm">Best Performer</span>
                  <span className="text-sm font-semibold">
                    {strategies.reduce((best, s) => s.performance.sharpe_ratio > best.performance.sharpe_ratio ? s : best).name}
                  </span>
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm">Total Trades</span>
                  <span className="text-sm font-mono-numbers font-semibold">
                    {strategies.reduce((sum, s) => sum + s.performance.total_trades, 0)}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
