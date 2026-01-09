import { formatPercent, getChangeColor } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  TrendingUp,
  TrendingDown,
  Clock,
  Activity,
  AlertTriangle
} from 'lucide-react'
import { MarketState } from '@/types'

interface MarketStateIndicatorProps {
  marketState: MarketState
}

export default function MarketStateIndicator({ marketState }: MarketStateIndicatorProps) {
  const getMarketStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'open':
        return 'success'
      case 'closed':
        return 'secondary'
      case 'pre-market':
      case 'after-hours':
        return 'warning'
      default:
        return 'secondary'
    }
  }

  const getMarketTrendIcon = (trend: number) => {
    if (trend > 0) return TrendingUp
    if (trend < 0) return TrendingDown
    return Activity
  }

  const MarketTrendIcon = getMarketTrendIcon(marketState.market_trend || 0)

  return (
    <Card className="bg-gradient-to-r from-card/80 to-card border-muted">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-primary" />
            <span>Market State</span>
          </div>
          <Badge variant={getMarketStatusColor(marketState.status)}>
            {marketState.status}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Market Trend */}
          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <MarketTrendIcon className={`h-4 w-4 ${getChangeColor(marketState.market_trend || 0)}`} />
              <span className={`text-sm font-medium ${getChangeColor(marketState.market_trend || 0)}`}>
                Trend
              </span>
            </div>
            <div className={`text-lg font-bold font-mono-numbers ${getChangeColor(marketState.market_trend || 0)}`}>
              {formatPercent(marketState.market_trend || 0)}
            </div>
          </div>

          {/* Volatility */}
          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <Activity className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground">Volatility</span>
            </div>
            <div className="text-lg font-bold font-mono-numbers">
              {formatPercent(marketState.volatility || 0)}
            </div>
          </div>

          {/* Volume */}
          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground">Volume</span>
            </div>
            <div className="text-lg font-bold font-mono-numbers">
              {marketState.volume_ratio ? `${(marketState.volume_ratio * 100).toFixed(0)}%` : 'N/A'}
            </div>
          </div>

          {/* Fear & Greed */}
          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 mb-1">
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground">Sentiment</span>
            </div>
            <div className="text-lg font-bold font-mono-numbers">
              {marketState.fear_greed_index || 'N/A'}
            </div>
          </div>
        </div>

        {/* Market Hours */}
        {marketState.next_open && (
          <div className="mt-4 pt-3 border-t">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center space-x-2 text-muted-foreground">
                <Clock className="h-4 w-4" />
                <span>Next Open:</span>
              </div>
              <span className="font-mono-numbers">
                {new Date(marketState.next_open).toLocaleString()}
              </span>
            </div>
          </div>
        )}

        {/* Market Conditions */}
        <div className="mt-4 grid grid-cols-2 gap-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Regime:</span>
            <Badge variant="outline" className="text-xs">
              {marketState.regime || 'Normal'}
            </Badge>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Risk Level:</span>
            <Badge
              variant={
                (marketState.risk_level || 'medium') === 'high' ? 'error' :
                (marketState.risk_level || 'medium') === 'low' ? 'success' : 'warning'
              }
              className="text-xs"
            >
              {(marketState.risk_level || 'medium').toUpperCase()}
            </Badge>
          </div>
        </div>

        {/* Market Interpretation */}
        {marketState.market_interpretation && (
          <div className="mt-4 pt-3 border-t">
            <div className="flex items-start space-x-2">
              <AlertTriangle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
              <p className="text-sm text-muted-foreground leading-relaxed">
                {marketState.market_interpretation}
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}