import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { formatCurrency, formatPercent, getChangeColor } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { AssetFilter } from '@/types'
import {
  Search,
  Filter,
  TrendingUp,
  BarChart3,
  RefreshCw,
  Download,
  Star,
  CheckCircle
} from 'lucide-react'

export default function Screener() {
  const [filter, setFilter] = useState<AssetFilter>({})
  const [showFilters, setShowFilters] = useState(true)

  // Fetch assets with filter
  const { data: assetsData, isLoading: assetsLoading, refetch: refetchAssets } = useQuery({
    queryKey: ['screener-assets', filter],
    queryFn: () => apiClient.getAssets({ limit: 100 }),
    refetchInterval: 15000,
  })

  const assets = assetsData?.data || []

  // Mock selection results (in production, this would come from selection API)
  const selectionResults = [
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      score: 87.5,
      rank: 1,
      sector: 'Technology',
      price: 175.43,
      change_percent: 2.34,
      factors: {
        valuation: 82,
        momentum: 91,
        technical: 88,
        volume: 85,
        market_sentiment: 89
      },
      signals: ['breakout', 'earnings_momentum', 'volume_surge']
    },
    {
      symbol: 'MSFT',
      name: 'Microsoft Corporation',
      score: 85.2,
      rank: 2,
      sector: 'Technology',
      price: 378.91,
      change_percent: 1.87,
      factors: {
        valuation: 78,
        momentum: 89,
        technical: 86,
        volume: 84,
        market_sentiment: 87
      },
      signals: ['value_momentum', 'technical_strength']
    },
    {
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      score: 83.8,
      rank: 3,
      sector: 'Technology',
      price: 142.56,
      change_percent: 3.12,
      factors: {
        valuation: 85,
        momentum: 84,
        technical: 82,
        volume: 83,
        market_sentiment: 84
      },
      signals: ['undervalued', 'momentum_building']
    },
    {
      symbol: 'NVDA',
      name: 'NVIDIA Corporation',
      score: 82.1,
      rank: 4,
      sector: 'Technology',
      price: 485.23,
      change_percent: 4.56,
      factors: {
        valuation: 75,
        momentum: 95,
        technical: 83,
        volume: 88,
        market_sentiment: 86
      },
      signals: ['breakout', 'volume_surge', 'momentum_leader']
    },
    {
      symbol: 'META',
      name: 'Meta Platforms Inc.',
      score: 80.5,
      rank: 5,
      sector: 'Technology',
      price: 489.12,
      change_percent: 2.89,
      factors: {
        valuation: 81,
        momentum: 82,
        technical: 79,
        volume: 80,
        market_sentiment: 81
      },
      signals: ['value_momentum', 'technical_setup']
    }
  ]

  const sectors = ['All', 'Technology', 'Healthcare', 'Financial', 'Consumer', 'Energy', 'Industrial', 'Materials', 'Utilities', 'Real Estate']

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-bull'
    if (score >= 60) return 'text-yellow-600'
    return 'text-bear'
  }

  const getScoreVariant = (score: number) => {
    if (score >= 80) return 'success' as const
    if (score >= 60) return 'warning' as const
    return 'error' as const
  }

  const getSignalBadgeVariant = (signal: string) => {
    if (signal.includes('breakout') || signal.includes('momentum')) return 'success' as const
    if (signal.includes('value') || signal.includes('undervalued')) return 'default' as const
    return 'secondary' as const
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold">Stock Screener</h1>
          <p className="text-muted-foreground">Multi-factor screening and selection results</p>
        </div>

        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowFilters(!showFilters)}
          >
            <Filter className="h-4 w-4 mr-2" />
            {showFilters ? 'Hide' : 'Show'} Filters
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetchAssets()}
            disabled={assetsLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${assetsLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Screening Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers">
                {selectionResults.length}
              </div>
              <div className="text-sm text-muted-foreground">Top Picks</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers">
                {assets.length}
              </div>
              <div className="text-sm text-muted-foreground">Screened Stocks</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers text-bull">
                {selectionResults.filter(s => s.change_percent > 0).length}
              </div>
              <div className="text-sm text-muted-foreground">Gainers</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-2xl font-bold font-mono-numbers">
                {(selectionResults.reduce((sum, s) => sum + s.score, 0) / selectionResults.length).toFixed(1)}
              </div>
              <div className="text-sm text-muted-foreground">Avg Score</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filter Panel */}
      {showFilters && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Filter className="h-5 w-5" />
              <span>Screening Filters</span>
            </CardTitle>
            <CardDescription>
              Customize screening criteria for stock selection
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Sector Filter */}
              <div>
                <label className="text-sm font-medium mb-2 block">Sectors</label>
                <div className="flex flex-wrap gap-2">
                  {sectors.map((sector) => (
                    <Button
                      key={sector}
                      variant={filter.sectors?.includes(sector) ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => {
                        const sectors = filter.sectors || []
                        if (sector === 'All') {
                          setFilter({ ...filter, sectors: [] })
                        } else {
                          const newSectors = sectors.includes(sector)
                            ? sectors.filter(s => s !== sector)
                            : [...sectors, sector]
                          setFilter({ ...filter, sectors: newSectors })
                        }
                      }}
                    >
                      {sector}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Price Range */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Min Price</label>
                  <input
                    type="number"
                    placeholder="0"
                    className="w-full px-3 py-2 border rounded-md bg-background"
                    onChange={(e) => setFilter({ ...filter, min_price: parseFloat(e.target.value) || undefined })}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Max Price</label>
                  <input
                    type="number"
                    placeholder="1000"
                    className="w-full px-3 py-2 border rounded-md bg-background"
                    onChange={(e) => setFilter({ ...filter, max_price: parseFloat(e.target.value) || undefined })}
                  />
                </div>
              </div>

              {/* Market Cap Range */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Min Market Cap</label>
                  <input
                    type="number"
                    placeholder="0"
                    className="w-full px-3 py-2 border rounded-md bg-background"
                    onChange={(e) => setFilter({ ...filter, min_market_cap: parseFloat(e.target.value) || undefined })}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Max Market Cap</label>
                  <input
                    type="number"
                    placeholder="1000000000000"
                    className="w-full px-3 py-2 border rounded-md bg-background"
                    onChange={(e) => setFilter({ ...filter, max_market_cap: parseFloat(e.target.value) || undefined })}
                  />
                </div>
              </div>

              {/* Volume Range */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Min Volume</label>
                  <input
                    type="number"
                    placeholder="0"
                    className="w-full px-3 py-2 border rounded-md bg-background"
                    onChange={(e) => setFilter({ ...filter, min_volume: parseFloat(e.target.value) || undefined })}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Max Volume</label>
                  <input
                    type="number"
                    placeholder="1000000000"
                    className="w-full px-3 py-2 border rounded-md bg-background"
                    onChange={(e) => setFilter({ ...filter, max_volume: parseFloat(e.target.value) || undefined })}
                  />
                </div>
              </div>

              <div className="flex space-x-2">
                <Button
                  onClick={() => refetchAssets()}
                  className="flex-1"
                >
                  <Search className="h-4 w-4 mr-2" />
                  Apply Filters
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setFilter({})}
                >
                  Reset
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Selection Results */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <Star className="h-5 w-5 text-yellow-500" />
                <span>Top Selection Results</span>
              </CardTitle>
              <CardDescription>
                Highest scoring stocks based on multi-factor analysis
              </CardDescription>
            </div>
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {selectionResults.map((result) => (
              <div
                key={result.symbol}
                className="p-4 rounded-lg border hover:bg-accent/30 transition-all"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`text-2xl font-bold font-mono-numbers ${getScoreColor(result.score)}`}>
                      #{result.rank}
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold text-lg">{result.symbol}</span>
                        <Badge variant="outline" className="text-xs">
                          {result.sector}
                        </Badge>
                      </div>
                      <div className="text-sm text-muted-foreground">{result.name}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center space-x-2">
                      <Badge variant={getScoreVariant(result.score)} className="text-sm">
                        Score: {result.score.toFixed(1)}
                      </Badge>
                    </div>
                    <div className="text-xl font-mono-numbers mt-1">
                      {formatCurrency(result.price)}
                    </div>
                    <div className={`text-sm font-mono-numbers ${getChangeColor(result.change_percent)}`}>
                      {result.change_percent > 0 ? '+' : ''}{formatPercent(result.change_percent / 100)}
                    </div>
                  </div>
                </div>

                {/* Factor Scores */}
                <div className="grid grid-cols-5 gap-3 mb-3">
                  {Object.entries(result.factors).map(([factor, score]) => (
                    <div key={factor} className="text-center">
                      <div className={`text-sm font-bold font-mono-numbers ${getScoreColor(score)}`}>
                        {score}
                      </div>
                      <div className="text-xs text-muted-foreground capitalize">
                        {factor.replace('_', ' ')}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Signals */}
                <div className="flex flex-wrap gap-2">
                  {result.signals.map((signal) => (
                    <Badge key={signal} variant={getSignalBadgeVariant(signal)} className="text-xs">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      {signal.replace('_', ' ').toUpperCase()}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Screening Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>Top Gainers</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {selectionResults
                .filter(s => s.change_percent > 0)
                .sort((a, b) => b.change_percent - a.change_percent)
                .slice(0, 5)
                .map((result) => (
                  <div key={result.symbol} className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold">{result.symbol}</div>
                      <div className="text-xs text-muted-foreground">{result.name}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-mono-numbers">{formatCurrency(result.price)}</div>
                      <div className="text-sm font-mono-numbers text-bull">
                        +{formatPercent(result.change_percent / 100)}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Sector Distribution</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {sectors.slice(1).map((sector) => {
                const count = selectionResults.filter(s => s.sector === sector).length
                const percentage = (count / selectionResults.length) * 100
                return (
                  <div key={sector}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm">{sector}</span>
                      <span className="text-sm font-mono-numbers">{count} stocks</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
