import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { formatCurrency, formatPercent, getChangeColor } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import {
  Shield,
  AlertTriangle,
  TrendingDown,
  Activity,
  Target,
  BarChart3
} from 'lucide-react'

export default function Risk() {
  // Fetch portfolio data
  const { data: portfolioData, isLoading: portfolioLoading } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: () => apiClient.getPortfolioSummary(),
    refetchInterval: 10000,
  })

  // Fetch positions data
  const { data: positionsData, isLoading: positionsLoading } = useQuery({
    queryKey: ['positions'],
    queryFn: () => apiClient.getPositions(),
    refetchInterval: 10000,
  })

  // Fetch risk metrics
  const { data: riskData, isLoading: riskLoading } = useQuery({
    queryKey: ['risk-metrics'],
    queryFn: () => apiClient.getRiskMetrics(),
    refetchInterval: 30000,
  })

  const portfolio = portfolioData?.data
  const positions = positionsData?.data || []
  const riskMetrics = riskData?.data

  const getRiskLevel = (value: number, thresholds: { low: number; medium: number }) => {
    if (value <= thresholds.low) return { level: 'Low', variant: 'success' as const, color: 'text-bull' }
    if (value <= thresholds.medium) return { level: 'Medium', variant: 'warning' as const, color: 'text-yellow-600' }
    return { level: 'High', variant: 'error' as const, color: 'text-bear' }
  }

  const getConcentrationRisk = (positions: any[]) => {
    if (positions.length === 0) return 0
    const totalValue = positions.reduce((sum, p) => sum + p.market_value, 0)
    const maxPosition = Math.max(...positions.map(p => p.market_value))
    return (maxPosition / totalValue) * 100
  }

  const getSectorConcentration = (positions: any[]) => {
    const sectorMap = new Map()
    const totalValue = positions.reduce((sum, p) => sum + p.market_value, 0)

    positions.forEach(position => {
      const sector = position.sector || 'Unknown'
      sectorMap.set(sector, (sectorMap.get(sector) || 0) + position.market_value)
    })

    return Array.from(sectorMap.entries())
      .map(([sector, value]) => ({
        sector,
        value,
        percentage: (value / totalValue) * 100
      }))
      .sort((a, b) => b.percentage - a.percentage)
  }

  const concentrationRisk = getConcentrationRisk(positions)
  const sectorConcentration = getSectorConcentration(positions)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Risk Management</h1>
        <p className="text-muted-foreground">Portfolio risk analysis and monitoring</p>
      </div>

      {/* Risk Overview */}
      {portfolio && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {portfolio.risk_metrics.sharpe_ratio.toFixed(2)}
                </div>
                <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {formatPercent(portfolio.risk_metrics.max_drawdown)}
                </div>
                <div className="text-sm text-muted-foreground">Max Drawdown</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {portfolio.risk_metrics.portfolio_beta.toFixed(2)}
                </div>
                <div className="text-sm text-muted-foreground">Portfolio Beta</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold font-mono-numbers">
                  {formatPercent(portfolio.risk_metrics.volatility)}
                </div>
                <div className="text-sm text-muted-foreground">Volatility</div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="h-5 w-5" />
              <span>Risk Metrics</span>
            </CardTitle>
            <CardDescription>
              Key risk indicators and portfolio health metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            {portfolioLoading || riskLoading ? (
              <div className="space-y-4">
                {[1, 2, 3, 4].map(i => (
                  <div key={i} className="h-16 loading-skeleton" />
                ))}
              </div>
            ) : (
              <div className="space-y-6">
                {/* Value at Risk */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Value at Risk (1-day, 95%)</span>
                    <span className="text-sm font-mono-numbers">
                      {riskMetrics?.var_1d ? formatCurrency(riskMetrics.var_1d) : 'N/A'}
                    </span>
                  </div>
                  <Progress
                    value={riskMetrics?.var_1d ? Math.min((Math.abs(riskMetrics.var_1d) / (portfolio?.total_value || 1)) * 100 * 10, 100) : 0}
                    className="h-2"
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    Maximum expected loss in one day
                  </div>
                </div>

                {/* Portfolio Concentration */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Position Concentration</span>
                    <span className="text-sm font-mono-numbers">
                      {concentrationRisk.toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={concentrationRisk} className="h-2" />
                  <div className="text-xs text-muted-foreground mt-1">
                    Largest position as % of portfolio
                  </div>
                </div>

                {/* Portfolio Beta */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Market Beta</span>
                    <span className="text-sm font-mono-numbers">
                      {portfolio?.risk_metrics.portfolio_beta.toFixed(2) || 'N/A'}
                    </span>
                  </div>
                  <Progress
                    value={portfolio ? Math.min(Math.abs(portfolio.risk_metrics.portfolio_beta) * 50, 100) : 0}
                    className="h-2"
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    Sensitivity to market movements
                  </div>
                </div>

                {/* Volatility */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Volatility (Annualized)</span>
                    <span className="text-sm font-mono-numbers">
                      {portfolio ? formatPercent(portfolio.risk_metrics.volatility) : 'N/A'}
                    </span>
                  </div>
                  <Progress
                    value={portfolio ? Math.min(portfolio.risk_metrics.volatility * 100 * 5, 100) : 0}
                    className="h-2"
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    Price movement volatility
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Risk Alerts & Warnings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5" />
              <span>Risk Alerts</span>
            </CardTitle>
            <CardDescription>
              Active risk warnings and recommendations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Concentration Risk Alert */}
              {concentrationRisk > 25 && (
                <div className="p-3 rounded-lg border border-yellow-200 bg-yellow-50">
                  <div className="flex items-start space-x-2">
                    <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5" />
                    <div>
                      <div className="text-sm font-medium text-yellow-800">
                        High Concentration Risk
                      </div>
                      <div className="text-xs text-yellow-700">
                        Largest position represents {concentrationRisk.toFixed(1)}% of portfolio
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* High Beta Alert */}
              {portfolio && Math.abs(portfolio.risk_metrics.portfolio_beta) > 1.5 && (
                <div className="p-3 rounded-lg border border-orange-200 bg-orange-50">
                  <div className="flex items-start space-x-2">
                    <TrendingDown className="h-4 w-4 text-orange-600 mt-0.5" />
                    <div>
                      <div className="text-sm font-medium text-orange-800">
                        High Market Sensitivity
                      </div>
                      <div className="text-xs text-orange-700">
                        Portfolio beta of {portfolio.risk_metrics.portfolio_beta.toFixed(2)} indicates high market correlation
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* High Volatility Alert */}
              {portfolio && portfolio.risk_metrics.volatility > 0.3 && (
                <div className="p-3 rounded-lg border border-red-200 bg-red-50">
                  <div className="flex items-start space-x-2">
                    <Activity className="h-4 w-4 text-red-600 mt-0.5" />
                    <div>
                      <div className="text-sm font-medium text-red-800">
                        High Volatility
                      </div>
                      <div className="text-xs text-red-700">
                        Portfolio volatility of {formatPercent(portfolio.risk_metrics.volatility)} is above recommended levels
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Low Risk State */}
              {concentrationRisk < 15 && portfolio && Math.abs(portfolio.risk_metrics.portfolio_beta) < 1.2 && portfolio.risk_metrics.volatility < 0.2 && (
                <div className="p-3 rounded-lg border border-green-200 bg-green-50">
                  <div className="flex items-start space-x-2">
                    <Shield className="h-4 w-4 text-green-600 mt-0.5" />
                    <div>
                      <div className="text-sm font-medium text-green-800">
                        Risk Levels Normal
                      </div>
                      <div className="text-xs text-green-700">
                        Portfolio risk metrics are within acceptable ranges
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Sector Concentration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5" />
            <span>Sector Allocation</span>
          </CardTitle>
          <CardDescription>
            Portfolio diversification across sectors
          </CardDescription>
        </CardHeader>
        <CardContent>
          {positionsLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="h-12 loading-skeleton" />
              ))}
            </div>
          ) : sectorConcentration.length > 0 ? (
            <div className="space-y-4">
              {sectorConcentration.slice(0, 8).map((sector) => (
                <div key={sector.sector} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">{sector.sector}</span>
                    <div className="text-right">
                      <span className="text-sm font-mono-numbers">{sector.percentage.toFixed(1)}%</span>
                      <div className="text-xs text-muted-foreground">
                        {formatCurrency(sector.value)}
                      </div>
                    </div>
                  </div>
                  <Progress value={sector.percentage} className="h-2" />
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              No position data available
            </div>
          )}
        </CardContent>
      </Card>

      {/* Position Risk Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5" />
            <span>Position Risk Analysis</span>
          </CardTitle>
          <CardDescription>
            Risk assessment for individual positions
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
              {positions
                .sort((a, b) => b.market_value - a.market_value)
                .slice(0, 10)
                .map((position) => {
                  const portfolioWeight = portfolio ? (position.market_value / portfolio.total_value) * 100 : 0
                  const riskLevel = getRiskLevel(portfolioWeight, { low: 10, medium: 20 })

                  return (
                    <div
                      key={position.symbol}
                      className="flex items-center justify-between p-3 rounded border"
                    >
                      <div className="flex items-center space-x-4">
                        <div>
                          <div className="font-semibold">{position.symbol}</div>
                          <div className="text-sm text-muted-foreground">
                            {position.quantity} shares
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center space-x-2">
                          <Badge variant={riskLevel.variant} className="text-xs">
                            {portfolioWeight.toFixed(1)}%
                          </Badge>
                          <span className={`text-sm ${riskLevel.color}`}>
                            {riskLevel.level}
                          </span>
                        </div>
                        <div className="text-sm font-mono-numbers">
                          {formatCurrency(position.market_value)}
                        </div>
                        <div className={`text-xs font-mono-numbers ${getChangeColor(position.unrealized_pnl)}`}>
                          {formatCurrency(position.unrealized_pnl)}
                        </div>
                      </div>
                    </div>
                  )
                })}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              No positions to analyze
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
