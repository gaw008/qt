import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { formatPercent } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Bell,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Target
} from 'lucide-react'

export default function AlertsPanel() {
  const { data: alertsData, isLoading } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => apiClient.getAlerts({ limit: 8 }),
    refetchInterval: 30000,
  })

  const alerts = alertsData?.data || []
  const activeAlerts = alerts.filter(alert => alert.status === 'active')
  const criticalAlerts = alerts.filter(alert => alert.severity === 'critical')

  const getAlertIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'price_movement':
        return TrendingUp
      case 'volume_spike':
        return Target
      case 'risk_limit':
        return AlertTriangle
      case 'trade_execution':
        return CheckCircle
      default:
        return Bell
    }
  }

  const getAlertVariant = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'error' as const
      case 'high':
        return 'warning' as const
      case 'medium':
        return 'secondary' as const
      case 'low':
        return 'outline' as const
      default:
        return 'secondary' as const
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'text-bear'
      case 'high':
        return 'text-yellow-600'
      case 'medium':
        return 'text-blue-600'
      case 'low':
        return 'text-muted-foreground'
      default:
        return 'text-muted-foreground'
    }
  }

  const formatAlertTime = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`
    return date.toLocaleDateString()
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Bell className="h-5 w-5" />
          <span>Smart Alerts</span>
          {criticalAlerts.length > 0 && (
            <Badge variant="error" className="text-xs">
              {criticalAlerts.length} Critical
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          AI-powered trading alerts and notifications
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
            {/* Alert Summary */}
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-lg font-bold font-mono-numbers">
                  {activeAlerts.length}
                </div>
                <div className="text-xs text-muted-foreground">Active Alerts</div>
              </div>
              <div className="text-center p-3 bg-bear/10 rounded-lg">
                <div className="text-lg font-bold font-mono-numbers text-bear">
                  {criticalAlerts.length}
                </div>
                <div className="text-xs text-muted-foreground">Critical</div>
              </div>
            </div>

            {/* Recent Alerts */}
            <div className="space-y-2">
              <div className="text-sm font-medium">Recent Alerts</div>
              {alerts.length > 0 ? (
                alerts.slice(0, 6).map((alert) => {
                  const AlertIcon = getAlertIcon(alert.type)
                  return (
                    <div
                      key={alert.id}
                      className="flex items-start justify-between p-3 rounded border"
                    >
                      <div className="flex items-start space-x-3">
                        <AlertIcon className={`h-4 w-4 mt-0.5 ${getSeverityColor(alert.severity)}`} />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate">
                            {alert.title}
                          </div>
                          <div className="text-xs text-muted-foreground line-clamp-2">
                            {alert.message}
                          </div>
                          {alert.symbol && (
                            <div className="text-xs font-mono-numbers text-primary mt-1">
                              {alert.symbol}
                              {alert.price_change && (
                                <span className={alert.price_change > 0 ? 'text-bull ml-2' : 'text-bear ml-2'}>
                                  {alert.price_change > 0 ? '+' : ''}{formatPercent(alert.price_change)}
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="flex flex-col items-end space-y-1">
                        <Badge variant={getAlertVariant(alert.severity)} className="text-xs">
                          {alert.severity.toUpperCase()}
                        </Badge>
                        <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          <span>{formatAlertTime(alert.timestamp)}</span>
                        </div>
                      </div>
                    </div>
                  )
                })
              ) : (
                <div className="text-center text-sm text-muted-foreground py-4">
                  No recent alerts
                </div>
              )}
            </div>

            {/* Alert Controls */}
            <div className="p-3 bg-muted/30 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Alert System</span>
                <div className="flex items-center space-x-1">
                  <div className="h-2 w-2 bg-bull rounded-full animate-pulse" />
                  <span className="text-xs text-bull">Active</span>
                </div>
              </div>
              <div className="text-xs text-muted-foreground">
                AI monitoring market conditions and portfolio performance
              </div>
            </div>

            {/* Alert Categories */}
            <div className="space-y-2">
              <div className="text-sm font-medium">Alert Categories</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Price Movement:</span>
                  <span>{alerts.filter(a => a.type === 'price_movement').length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Volume Spikes:</span>
                  <span>{alerts.filter(a => a.type === 'volume_spike').length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Risk Limits:</span>
                  <span>{alerts.filter(a => a.type === 'risk_limit').length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Trade Execution:</span>
                  <span>{alerts.filter(a => a.type === 'trade_execution').length}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
