import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Terminal,
  Pause,
  Play,
  RefreshCw,
  ChevronDown,
  ChevronUp
} from 'lucide-react'

export default function ConsoleLogs() {
  const [isPaused, setIsPaused] = useState(false)
  const [isExpanded, setIsExpanded] = useState(true)

  const { data: logsData, isLoading, refetch, isFetching } = useQuery({
    queryKey: ['runner-logs'],
    queryFn: () => apiClient.getRunnerLogs(100),
    refetchInterval: isPaused ? false : 30000, // 30 seconds refresh
    staleTime: 25000,
  })

  const logs = logsData?.data?.lines || []

  // Get log level color
  const getLogColor = (line: string): string => {
    const upperLine = line.toUpperCase()
    if (upperLine.includes('ERROR') || upperLine.includes('CRITICAL')) {
      return 'text-red-500'
    }
    if (upperLine.includes('WARNING') || upperLine.includes('WARN')) {
      return 'text-yellow-500'
    }
    if (upperLine.includes('SUCCESS') || upperLine.includes('COMPLETED')) {
      return 'text-green-500'
    }
    if (upperLine.includes('INFO')) {
      return 'text-blue-400'
    }
    return 'text-gray-300'
  }

  const handleRefresh = () => {
    refetch()
  }

  const togglePause = () => {
    setIsPaused(!isPaused)
  }

  const toggleExpand = () => {
    setIsExpanded(!isExpanded)
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Terminal className="h-5 w-5" />
            <CardTitle className="text-lg">Runner Console</CardTitle>
            {isFetching && !isLoading && (
              <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefresh}
              disabled={isFetching}
              title="Refresh logs"
            >
              <RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={togglePause}
              title={isPaused ? 'Resume auto-refresh' : 'Pause auto-refresh'}
            >
              {isPaused ? (
                <Play className="h-4 w-4 text-green-500" />
              ) : (
                <Pause className="h-4 w-4 text-yellow-500" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleExpand}
              title={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
        <CardDescription className="flex items-center space-x-4">
          <span>Trading bot console output</span>
          <span className="text-xs">
            {isPaused ? (
              <span className="text-yellow-500">Paused</span>
            ) : (
              <span className="text-green-500">Auto-refresh: 30s</span>
            )}
          </span>
        </CardDescription>
      </CardHeader>

      {isExpanded && (
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {[1, 2, 3, 4, 5].map(i => (
                <div key={i} className="h-4 loading-skeleton" />
              ))}
            </div>
          ) : logs.length > 0 ? (
            <div
              className="bg-gray-900 rounded-lg p-4 font-mono text-sm overflow-auto max-h-80"
              style={{ minHeight: '200px' }}
            >
              {logs.map((line, index) => (
                <div
                  key={index}
                  className={`${getLogColor(line)} whitespace-pre-wrap break-all leading-relaxed`}
                >
                  {line}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8 bg-gray-900 rounded-lg">
              No logs available
            </div>
          )}
        </CardContent>
      )}
    </Card>
  )
}
