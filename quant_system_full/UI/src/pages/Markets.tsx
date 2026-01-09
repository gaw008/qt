import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { formatCurrency, formatPercent, getChangeColor } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Filter,
  Grid,
  List,
  RefreshCw
} from 'lucide-react'

export default function Markets() {
  const [viewMode, setViewMode] = useState<'list' | 'heatmap'>('list')
  const [selectedSector, setSelectedSector] = useState<string>('all')

  // Fetch market data
  const { data: assetsData, isLoading: assetsLoading, refetch: refetchAssets } = useQuery({
    queryKey: ['assets'],
    queryFn: () => apiClient.getAssets({ limit: 100 }),
    refetchInterval: 10000,
  })

  // Fetch heatmap data
  const { data: heatmapData, isLoading: heatmapLoading } = useQuery({
    queryKey: ['heatmap', selectedSector],
    queryFn: () => apiClient.getHeatmapData(selectedSector === 'all' ? undefined : selectedSector),
    refetchInterval: 15000,
  })

  const assets = assetsData?.data || []
  const heatmapItems = heatmapData?.data || []

  const sectors = ['all', 'Technology', 'Healthcare', 'Financial', 'Consumer', 'Energy', 'Industrial', 'Materials', 'Utilities', 'Real Estate']

  const getHeatmapColor = (changePercent: number) => {
    const intensity = Math.min(Math.abs(changePercent) / 5, 1)
    if (changePercent > 0) {
      return `bg-bull opacity-${Math.round(intensity * 100)}`
    } else if (changePercent < 0) {
      return `bg-bear opacity-${Math.round(intensity * 100)}`
    }
    return 'bg-muted'
  }

  const getHeatmapSize = (marketCap: number) => {
    if (marketCap > 100000000000) return 'h-20 w-32' // Large cap
    if (marketCap > 10000000000) return 'h-16 w-24'  // Mid cap
    return 'h-12 w-20' // Small cap
  }

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold">Markets</h1>
          <p className="text-muted-foreground">Live market data and analysis</p>
        </div>

        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetchAssets()}
            disabled={assetsLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${assetsLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>

          <div className="flex border rounded-lg">
            <Button
              variant={viewMode === 'list' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('list')}
              className="rounded-r-none"
            >
              <List className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === 'heatmap' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('heatmap')}
              className="rounded-l-none"
            >
              <Grid className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Sector Filter */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Filter className="h-5 w-5" />
            <span>Sector Filter</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {sectors.map((sector) => (
              <Button
                key={sector}
                variant={selectedSector === sector ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedSector(sector)}
              >
                {sector === 'all' ? 'All Sectors' : sector}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Market Overview Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers text-bull">
                {assets.filter(a => a.change > 0).length}
              </div>
              <div className="text-sm text-muted-foreground">Gainers</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers text-bear">
                {assets.filter(a => a.change < 0).length}
              </div>
              <div className="text-sm text-muted-foreground">Losers</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers">
                {assets.length}
              </div>
              <div className="text-sm text-muted-foreground">Total Assets</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers">
                {assets.length > 0 ? formatPercent(
                  assets.reduce((sum, a) => sum + a.change_percent, 0) / assets.length
                ) : '0%'}
              </div>
              <div className="text-sm text-muted-foreground">Avg Change</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Market Data Display */}
      {viewMode === 'list' ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Live Quotes</span>
            </CardTitle>
            <CardDescription>
              Real-time market data with price and volume information
            </CardDescription>
          </CardHeader>
          <CardContent>
            {assetsLoading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4, 5].map(i => (
                  <div key={i} className="h-16 loading-skeleton" />
                ))}
              </div>
            ) : (
              <div className="space-y-2">
                {assets
                  .filter(asset => selectedSector === 'all' || asset.sector === selectedSector)
                  .slice(0, 50)
                  .map((asset) => (
                    <div
                      key={asset.symbol}
                      className="flex items-center justify-between p-4 rounded-lg border hover:bg-accent/50 transition-colors"
                    >
                      <div className="flex items-center space-x-4">
                        <div>
                          <div className="font-semibold text-lg">{asset.symbol}</div>
                          <div className="text-sm text-muted-foreground">{asset.name}</div>
                          {asset.sector && (
                            <Badge variant="outline" className="text-xs mt-1">
                              {asset.sector}
                            </Badge>
                          )}
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="text-xl font-bold font-mono-numbers">
                          {formatCurrency(asset.price)}
                        </div>
                        <div className={`flex items-center space-x-1 ${getChangeColor(asset.change)}`}>
                          {asset.change > 0 ? (
                            <TrendingUp className="h-4 w-4" />
                          ) : asset.change < 0 ? (
                            <TrendingDown className="h-4 w-4" />
                          ) : (
                            <Activity className="h-4 w-4" />
                          )}
                          <span className="font-mono-numbers">
                            {formatCurrency(asset.change)} ({formatPercent(asset.change_percent)})
                          </span>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Vol: {asset.volume.toLocaleString()}
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Grid className="h-5 w-5" />
              <span>Market Heatmap</span>
            </CardTitle>
            <CardDescription>
              Visual representation of market performance by sector and market cap
            </CardDescription>
          </CardHeader>
          <CardContent>
            {heatmapLoading ? (
              <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-2">
                {[...Array(24)].map((_, i) => (
                  <div key={i} className="h-16 w-20 loading-skeleton" />
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-2">
                {heatmapItems.slice(0, 50).map((item) => (
                  <div
                    key={item.symbol}
                    className={`
                      ${getHeatmapSize(item.market_cap)}
                      ${getHeatmapColor(item.change_percent_1d)}
                      rounded-lg p-2 flex flex-col justify-between text-white text-xs
                      hover:scale-105 transition-all cursor-pointer
                      border border-white/20
                    `}
                    title={`${item.name} (${item.symbol}): ${formatPercent(item.change_percent_1d)}`}
                  >
                    <div className="font-semibold truncate">{item.symbol}</div>
                    <div className="font-mono-numbers">
                      {formatPercent(item.change_percent_1d)}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Heatmap Legend */}
            <div className="mt-6 p-4 bg-muted/50 rounded-lg">
              <div className="text-sm font-medium mb-2">Legend</div>
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-bear rounded"></div>
                  <span>Negative</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-muted rounded"></div>
                  <span>Neutral</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-bull rounded"></div>
                  <span>Positive</span>
                </div>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Size represents market capitalization • Color intensity represents percentage change
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Top Movers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Top Gainers */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-bull">
              <TrendingUp className="h-5 w-5" />
              <span>Top Gainers</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {assets
                .filter(a => a.change > 0)
                .sort((a, b) => b.change_percent - a.change_percent)
                .slice(0, 5)
                .map((asset, index) => (
                  <div key={asset.symbol} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="text-sm font-mono-numbers text-muted-foreground w-4">
                        #{index + 1}
                      </div>
                      <div>
                        <div className="font-semibold">{asset.symbol}</div>
                        <div className="text-sm text-muted-foreground truncate">
                          {asset.name}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-mono-numbers">{formatCurrency(asset.price)}</div>
                      <div className="text-sm font-mono-numbers text-bull">
                        +{formatPercent(asset.change_percent)}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>

        {/* Top Losers */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-bear">
              <TrendingDown className="h-5 w-5" />
              <span>Top Losers</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {assets
                .filter(a => a.change < 0)
                .sort((a, b) => a.change_percent - b.change_percent)
                .slice(0, 5)
                .map((asset, index) => (
                  <div key={asset.symbol} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="text-sm font-mono-numbers text-muted-foreground w-4">
                        #{index + 1}
                      </div>
                      <div>
                        <div className="font-semibold">{asset.symbol}</div>
                        <div className="text-sm text-muted-foreground truncate">
                          {asset.name}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-mono-numbers">{formatCurrency(asset.price)}</div>
                      <div className="text-sm font-mono-numbers text-bear">
                        {formatPercent(asset.change_percent)}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
