/**
 * Backtesting Reports Dashboard Component
 * 回测报告仪表板组件
 *
 * Comprehensive backtesting validation interface for institutional-quality reports
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { AlertCircle, Download, FileText } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

// Types
interface BacktestRequest {
  strategy_name: string;
  start_date: string;
  end_date: string;
  universe?: string[];
  rebalance_frequency: string;
  initial_capital: number;
  transaction_costs: number;
  output_formats: string[];
  include_statistical_tests: boolean;
  include_charts: boolean;
  include_crisis_analysis: boolean;
}

interface ReportStatus {
  request_id: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: number;
  message: string;
  started_at: string;
  completed_at?: string;
  error_message?: string;
  output_files?: Record<string, string>;
  summary_metrics?: BacktestSummary;
}

interface BacktestSummary {
  strategy_name: string;
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  phase_results: Record<string, Record<string, number>>;
  volatility: number;
  var_95: number;
  calmar_ratio: number;
}

interface RecentReport {
  request_id: string;
  strategy_name: string;
  status: string;
  started_at: string;
  completed_at?: string;
  available_formats: string[];
}

// API functions
const API_BASE = 'http://localhost:8000/api/backtesting';

const generateReport = async (request: BacktestRequest): Promise<{request_id: string}> => {
  const response = await fetch(`${API_BASE}/generate-report`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer wgyjd0508`
    },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    throw new Error(`Failed to generate report: ${response.statusText}`);
  }

  return response.json();
};

const getReportStatus = async (requestId: string): Promise<ReportStatus> => {
  const response = await fetch(`${API_BASE}/status/${requestId}`, {
    headers: {
      'Authorization': `Bearer wgyjd0508`
    }
  });

  if (!response.ok) {
    throw new Error(`Failed to get report status: ${response.statusText}`);
  }

  return response.json();
};

const getRecentReports = async (): Promise<RecentReport[]> => {
  const response = await fetch(`${API_BASE}/recent-reports`, {
    headers: {
      'Authorization': `Bearer wgyjd0508`
    }
  });

  if (!response.ok) {
    throw new Error(`Failed to get recent reports: ${response.statusText}`);
  }

  return response.json();
};

const generateSampleReport = async (formats: string[]): Promise<{request_id: string}> => {
  const params = new URLSearchParams();
  formats.forEach(format => params.append('output_formats', format));

  const response = await fetch(`${API_BASE}/test/generate-sample?${params}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer wgyjd0508`
    }
  });

  if (!response.ok) {
    throw new Error(`Failed to generate sample report: ${response.statusText}`);
  }

  return response.json();
};

const downloadReport = (requestId: string, format: string) => {
  const url = `${API_BASE}/download/${requestId}/${format}`;
  window.open(url, '_blank');
};

// Status badge component
const StatusBadge: React.FC<{status: string}> = ({ status }) => {
  const variants: Record<string, string> = {
    pending: 'bg-yellow-100 text-yellow-800',
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    error: 'bg-red-100 text-red-800'
  };

  return (
    <Badge className={variants[status] || 'bg-gray-100 text-gray-800'}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  );
};

// Performance metrics display
const PerformanceMetrics: React.FC<{metrics: BacktestSummary}> = ({ metrics }) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card>
        <CardContent className="p-4">
          <div className="text-2xl font-bold text-green-600">
            {(metrics.total_return * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Total Return</div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="text-2xl font-bold text-blue-600">
            {metrics.sharpe_ratio.toFixed(2)}
          </div>
          <div className="text-sm text-gray-600">Sharpe Ratio</div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="text-2xl font-bold text-red-600">
            {(metrics.max_drawdown * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Max Drawdown</div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="text-2xl font-bold text-purple-600">
            {(metrics.win_rate * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Win Rate</div>
        </CardContent>
      </Card>
    </div>
  );
};

// Phase results component
const PhaseResults: React.FC<{phaseResults: Record<string, Record<string, number>>}> = ({ phaseResults }) => {
  return (
    <div className="space-y-4">
      {Object.entries(phaseResults).map(([phase, results]) => (
        <Card key={phase}>
          <CardHeader>
            <CardTitle className="text-lg">{phase}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-lg font-semibold">
                  {(results.total_return * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">Phase Return</div>
              </div>
              <div>
                <div className="text-lg font-semibold">
                  ${results.start_value?.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">Start Value</div>
              </div>
              <div>
                <div className="text-lg font-semibold">
                  ${results.end_value?.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">End Value</div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

// Main component
export const BacktestingReports: React.FC = () => {
  const [activeTab, setActiveTab] = useState('generate');
  const [recentReports, setRecentReports] = useState<RecentReport[]>([]);
  const [reportStatuses, setReportStatuses] = useState<Record<string, ReportStatus>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form state for report generation
  const [formData, setFormData] = useState<BacktestRequest>({
    strategy_name: 'Multi-Factor Strategy',
    start_date: '2006-01-01',
    end_date: '2025-01-01',
    universe: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    rebalance_frequency: 'monthly',
    initial_capital: 1000000,
    transaction_costs: 0.001,
    output_formats: ['html', 'pdf'],
    include_statistical_tests: true,
    include_charts: true,
    include_crisis_analysis: true
  });

  // Load recent reports on component mount
  useEffect(() => {
    loadRecentReports();
  }, []);

  // Poll for status updates
  useEffect(() => {
    const interval = setInterval(() => {
      const activeRequests = Object.keys(reportStatuses).filter(
        id => ['pending', 'running'].includes(reportStatuses[id]?.status)
      );

      if (activeRequests.length > 0) {
        activeRequests.forEach(requestId => {
          getReportStatus(requestId)
            .then(status => {
              setReportStatuses(prev => ({
                ...prev,
                [requestId]: status
              }));
            })
            .catch(console.error);
        });
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [reportStatuses]);

  const loadRecentReports = useCallback(async () => {
    try {
      const reports = await getRecentReports();
      setRecentReports(reports);
    } catch (err) {
      console.error('Failed to load recent reports:', err);
    }
  }, []);

  const handleGenerateReport = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await generateReport(formData);

      // Add to tracking
      setReportStatuses(prev => ({
        ...prev,
        [result.request_id]: {
          request_id: result.request_id,
          status: 'pending',
          progress: 0,
          message: 'Report generation started',
          started_at: new Date().toISOString()
        }
      }));

      // Refresh recent reports
      await loadRecentReports();

      setActiveTab('status');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate report');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateSample = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await generateSampleReport(['html', 'pdf']);

      setReportStatuses(prev => ({
        ...prev,
        [result.request_id]: {
          request_id: result.request_id,
          status: 'pending',
          progress: 0,
          message: 'Sample report generation started',
          started_at: new Date().toISOString()
        }
      }));

      await loadRecentReports();
      setActiveTab('status');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate sample report');
    } finally {
      setLoading(false);
    }
  };

  const handleFormChange = (field: keyof BacktestRequest, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Backtesting Reports</h2>
          <p className="text-muted-foreground">
            Generate comprehensive three-phase validation reports for institutional analysis
          </p>
        </div>
        <div className="flex space-x-2">
          <Button
            onClick={handleGenerateSample}
            variant="outline"
            disabled={loading}
          >
            <FileText className="mr-2 h-4 w-4" />
            Generate Sample
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="generate">Generate Report</TabsTrigger>
          <TabsTrigger value="status">Report Status</TabsTrigger>
          <TabsTrigger value="history">Report History</TabsTrigger>
        </TabsList>

        {/* Generate Report Tab */}
        <TabsContent value="generate" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Backtest Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="strategy_name">Strategy Name</Label>
                  <Input
                    id="strategy_name"
                    value={formData.strategy_name}
                    onChange={(e) => handleFormChange('strategy_name', e.target.value)}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="initial_capital">Initial Capital</Label>
                  <Input
                    id="initial_capital"
                    type="number"
                    value={formData.initial_capital}
                    onChange={(e) => handleFormChange('initial_capital', parseFloat(e.target.value))}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="start_date">Start Date</Label>
                  <Input
                    id="start_date"
                    type="date"
                    value={formData.start_date}
                    onChange={(e) => handleFormChange('start_date', e.target.value)}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="end_date">End Date</Label>
                  <Input
                    id="end_date"
                    type="date"
                    value={formData.end_date}
                    onChange={(e) => handleFormChange('end_date', e.target.value)}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="rebalance_frequency">Rebalance Frequency</Label>
                  <Select
                    value={formData.rebalance_frequency}
                    onValueChange={(value) => handleFormChange('rebalance_frequency', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="weekly">Weekly</SelectItem>
                      <SelectItem value="monthly">Monthly</SelectItem>
                      <SelectItem value="quarterly">Quarterly</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="transaction_costs">Transaction Costs (%)</Label>
                  <Input
                    id="transaction_costs"
                    type="number"
                    step="0.001"
                    value={formData.transaction_costs * 100}
                    onChange={(e) => handleFormChange('transaction_costs', parseFloat(e.target.value) / 100)}
                  />
                </div>
              </div>

              <div className="space-y-4">
                <Label>Output Formats</Label>
                <div className="flex flex-wrap gap-4">
                  {['html', 'pdf', 'excel', 'json'].map(format => (
                    <div key={format} className="flex items-center space-x-2">
                      <Checkbox
                        id={format}
                        checked={formData.output_formats.includes(format)}
                        onCheckedChange={(checked) => {
                          const newFormats = checked
                            ? [...formData.output_formats, format]
                            : formData.output_formats.filter(f => f !== format);
                          handleFormChange('output_formats', newFormats);
                        }}
                      />
                      <Label htmlFor={format}>{format.toUpperCase()}</Label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <Label>Analysis Options</Label>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="statistical_tests"
                      checked={formData.include_statistical_tests}
                      onCheckedChange={(checked) => handleFormChange('include_statistical_tests', checked)}
                    />
                    <Label htmlFor="statistical_tests">Include Statistical Significance Tests</Label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="charts"
                      checked={formData.include_charts}
                      onCheckedChange={(checked) => handleFormChange('include_charts', checked)}
                    />
                    <Label htmlFor="charts">Include Interactive Charts</Label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="crisis_analysis"
                      checked={formData.include_crisis_analysis}
                      onCheckedChange={(checked) => handleFormChange('include_crisis_analysis', checked)}
                    />
                    <Label htmlFor="crisis_analysis">Include Crisis Period Analysis</Label>
                  </div>
                </div>
              </div>

              <Button
                onClick={handleGenerateReport}
                disabled={loading}
                className="w-full"
              >
                {loading ? 'Generating...' : 'Generate Comprehensive Report'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Report Status Tab */}
        <TabsContent value="status" className="space-y-6">
          <div className="space-y-4">
            {Object.values(reportStatuses).length === 0 ? (
              <Card>
                <CardContent className="p-6 text-center">
                  <p className="text-muted-foreground">No active report generation requests</p>
                </CardContent>
              </Card>
            ) : (
              Object.values(reportStatuses).map(status => (
                <Card key={status.request_id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">
                        {status.summary_metrics?.strategy_name || 'Report Generation'}
                      </CardTitle>
                      <StatusBadge status={status.status} />
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>{status.message}</span>
                        <span>{Math.round(status.progress * 100)}%</span>
                      </div>
                      <Progress value={status.progress * 100} />
                    </div>

                    {status.error_message && (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>{status.error_message}</AlertDescription>
                      </Alert>
                    )}

                    {status.status === 'completed' && status.output_files && (
                      <div className="space-y-4">
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(status.output_files).map(([format, _]) => (
                            <Button
                              key={format}
                              variant="outline"
                              size="sm"
                              onClick={() => downloadReport(status.request_id, format)}
                            >
                              <Download className="mr-2 h-4 w-4" />
                              Download {format.toUpperCase()}
                            </Button>
                          ))}
                        </div>

                        {status.summary_metrics && (
                          <div className="space-y-4">
                            <h4 className="font-semibold">Performance Summary</h4>
                            <PerformanceMetrics metrics={status.summary_metrics} />

                            <h4 className="font-semibold">Phase Results</h4>
                            <PhaseResults phaseResults={status.summary_metrics.phase_results} />
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        {/* Report History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Recent Reports</CardTitle>
                <Button variant="outline" onClick={loadRecentReports}>
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentReports.length === 0 ? (
                  <p className="text-muted-foreground text-center py-4">No reports found</p>
                ) : (
                  recentReports.map(report => (
                    <div
                      key={report.request_id}
                      className="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div>
                        <h4 className="font-semibold">{report.strategy_name}</h4>
                        <p className="text-sm text-muted-foreground">
                          Started: {new Date(report.started_at).toLocaleString()}
                          {report.completed_at && (
                            <> • Completed: {new Date(report.completed_at).toLocaleString()}</>
                          )}
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <StatusBadge status={report.status} />
                        {report.status === 'completed' && report.available_formats.map(format => (
                          <Button
                            key={format}
                            variant="outline"
                            size="sm"
                            onClick={() => downloadReport(report.request_id, format)}
                          >
                            {format.toUpperCase()}
                          </Button>
                        ))}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
