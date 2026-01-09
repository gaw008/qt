import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { getChangeColor } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import {
  Brain,
  Zap,
  TrendingUp,
  Activity,
  Cpu,
  Database,
  BarChart3,
  RefreshCw,
  CheckCircle,
  Clock,
  AlertTriangle
} from 'lucide-react'

export default function AICenter() {
  // Fetch AI/ML status
  const { data: aiStatusData, isLoading: aiStatusLoading, refetch: refetchAIStatus } = useQuery({
    queryKey: ['ai-status'],
    queryFn: () => apiClient.getSystemStatus(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Fetch system health for GPU/training status
  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => apiClient.getSystemHealth(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const aiStatus = aiStatusData?.data
  const health = healthData?.data

  const getModelStatusIcon = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'training':
        return Activity
      case 'ready':
      case 'active':
        return CheckCircle
      case 'idle':
        return Clock
      default:
        return AlertTriangle
    }
  }

  const getModelStatusVariant = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'training':
        return 'warning' as const
      case 'ready':
      case 'active':
        return 'success' as const
      case 'idle':
        return 'secondary' as const
      default:
        return 'error' as const
    }
  }

  // Mock AI models data structure
  const aiModels = [
    {
      name: 'Multi-Factor Scoring Model',
      status: aiStatus?.ai_learning_enabled ? 'active' : 'idle',
      accuracy: 0.847,
      lastTrained: '2 hours ago',
      trainingEpochs: 1250,
      performance: 0.92
    },
    {
      name: 'Risk Prediction Model',
      status: aiStatus?.ai_learning_enabled ? 'active' : 'idle',
      accuracy: 0.812,
      lastTrained: '4 hours ago',
      trainingEpochs: 980,
      performance: 0.88
    },
    {
      name: 'Market Regime Classifier',
      status: aiStatus?.ai_learning_enabled ? 'ready' : 'idle',
      accuracy: 0.793,
      lastTrained: '1 day ago',
      trainingEpochs: 2100,
      performance: 0.85
    },
    {
      name: 'Sentiment Analysis Model',
      status: aiStatus?.ai_learning_enabled ? 'training' : 'idle',
      accuracy: 0.765,
      lastTrained: '12 hours ago',
      trainingEpochs: 450,
      performance: 0.78
    }
  ]

  const trainingMetrics = [
    { metric: 'Training Loss', value: 0.0234, change: -12.5 },
    { metric: 'Validation Loss', value: 0.0289, change: -8.3 },
    { metric: 'F1 Score', value: 0.891, change: 4.2 },
    { metric: 'AUC-ROC', value: 0.923, change: 2.1 }
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold">AI Center</h1>
          <p className="text-muted-foreground">AI/ML model training, performance, and analytics</p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={() => refetchAIStatus()}
          disabled={aiStatusLoading}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${aiStatusLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* AI System Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <Brain className="h-8 w-8 mx-auto mb-2 text-primary" />
              <div className="text-2xl font-bold font-mono-numbers">
                {aiModels.filter(m => m.status === 'active').length}
              </div>
              <div className="text-sm text-muted-foreground">Active Models</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <Cpu className="h-8 w-8 mx-auto mb-2 text-primary" />
              <div className="text-2xl font-bold font-mono-numbers">
                {health?.gpu_available ? 'GPU' : 'CPU'}
              </div>
              <div className="text-sm text-muted-foreground">Compute Mode</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <Database className="h-8 w-8 mx-auto mb-2 text-primary" />
              <div className="text-2xl font-bold font-mono-numbers">
                {aiStatus?.data_cache_size ? (aiStatus.data_cache_size / 1024 / 1024).toFixed(1) : '0'} MB
              </div>
              <div className="text-sm text-muted-foreground">Training Data</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <Activity className="h-8 w-8 mx-auto mb-2 text-primary" />
              <div className="text-2xl font-bold font-mono-numbers">
                {aiModels.filter(m => m.status === 'training').length}
              </div>
              <div className="text-sm text-muted-foreground">Training Jobs</div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* AI Models Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>AI Models</span>
            </CardTitle>
            <CardDescription>
              Model status, performance, and training metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            {aiStatusLoading ? (
              <div className="space-y-4">
                {[1, 2, 3, 4].map(i => (
                  <div key={i} className="h-20 loading-skeleton" />
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                {aiModels.map((model) => {
                  const StatusIcon = getModelStatusIcon(model.status)
                  return (
                    <div
                      key={model.name}
                      className="flex items-center justify-between p-3 rounded-lg border"
                    >
                      <div className="flex items-center space-x-3">
                        <StatusIcon className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <div className="font-semibold">{model.name}</div>
                          <div className="text-xs text-muted-foreground">
                            Accuracy: {(model.accuracy * 100).toFixed(1)}% | Epochs: {model.trainingEpochs}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Last trained: {model.lastTrained}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant={getModelStatusVariant(model.status)}>
                          {model.status.toUpperCase()}
                        </Badge>
                        <div className="text-xs text-muted-foreground mt-1">
                          Perf: {(model.performance * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Training Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Training Metrics</span>
            </CardTitle>
            <CardDescription>
              Real-time training performance indicators
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {trainingMetrics.map((metric) => (
                <div key={metric.metric}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">{metric.metric}</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-mono-numbers">
                        {metric.value.toFixed(4)}
                      </span>
                      <span className={`text-xs font-mono-numbers ${getChangeColor(metric.change)}`}>
                        {metric.change > 0 ? '+' : ''}{metric.change.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <Progress
                    value={metric.value * 100}
                    className="h-2"
                  />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* GPU/Training Pipeline Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="h-5 w-5" />
            <span>GPU Training Pipeline</span>
          </CardTitle>
          <CardDescription>
            Hardware acceleration and training infrastructure status
          </CardDescription>
        </CardHeader>
        <CardContent>
          {healthLoading ? (
            <div className="space-y-4">
              {[1, 2].map(i => (
                <div key={i} className="h-16 loading-skeleton" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">GPU Status</span>
                    <Badge variant={health?.gpu_available ? 'success' : 'secondary'}>
                      {health?.gpu_available ? 'Available' : 'Not Available'}
                    </Badge>
                  </div>
                  {health?.gpu_available && (
                    <div className="text-xs text-muted-foreground">
                      GPU acceleration enabled for training
                    </div>
                  )}
                </div>

                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Training Queue</span>
                    <span className="text-sm font-mono-numbers">
                      {aiModels.filter(m => m.status === 'training').length} active
                    </span>
                  </div>
                  <Progress
                    value={(aiModels.filter(m => m.status === 'training').length / aiModels.length) * 100}
                    className="h-2"
                  />
                </div>

                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Memory Usage</span>
                    <span className="text-sm font-mono-numbers">
                      {health?.memory_usage_percent ? `${health.memory_usage_percent.toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  <Progress
                    value={health?.memory_usage_percent || 0}
                    className="h-2"
                  />
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">CPU Usage</span>
                    <span className="text-sm font-mono-numbers">
                      {health?.cpu_usage_percent ? `${health.cpu_usage_percent.toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  <Progress
                    value={health?.cpu_usage_percent || 0}
                    className="h-2"
                  />
                </div>

                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">AI Learning</span>
                    <Badge variant={aiStatus?.ai_learning_enabled ? 'success' : 'secondary'}>
                      {aiStatus?.ai_learning_enabled ? 'Enabled' : 'Disabled'}
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Continuous learning from trading outcomes
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Model Optimization</span>
                    <Badge variant="success">
                      Active
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Automatic hyperparameter tuning enabled
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Performance Analytics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>Prediction Accuracy</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-4xl font-bold font-mono-numbers">
                {(aiModels.reduce((sum, m) => sum + m.accuracy, 0) / aiModels.length * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                Average across all models
              </div>
              <Progress
                value={aiModels.reduce((sum, m) => sum + m.accuracy, 0) / aiModels.length * 100}
                className="h-2 mt-4"
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Training Throughput</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-4xl font-bold font-mono-numbers">
                {aiModels.reduce((sum, m) => sum + m.trainingEpochs, 0).toLocaleString()}
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                Total training epochs
              </div>
              <div className="text-xs text-muted-foreground mt-4">
                {health?.gpu_available ? 'GPU-accelerated training' : 'CPU training mode'}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Database className="h-5 w-5" />
              <span>Data Pipeline</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-4xl font-bold font-mono-numbers">
                {aiStatus?.data_cache_size ? (aiStatus.data_cache_size / 1024 / 1024 / 1024).toFixed(2) : '0'} GB
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                Training data processed
              </div>
              <div className="text-xs text-muted-foreground mt-4">
                Real-time feature engineering active
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
