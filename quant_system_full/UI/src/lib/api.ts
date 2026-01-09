import { ApiResponse, Asset, Position, Order, PortfolioSummary, MarketState, Alert, HeatmapData, AssetFilter, StockSelectionResponse } from '@/types'

// Build version for cache busting - change this to force new build
const BUILD_VERSION = '2026-01-04-v1'

// API Configuration
// Use proxy in development, full URL in production
// IMPORTANT: Production must use api.wgyjdaiassistant.cc, NOT trade.wgyjdaiassistant.cc
const API_BASE_URL = import.meta.env.DEV ? '' : (import.meta.env.VITE_API_BASE_URL || 'https://api.wgyjdaiassistant.cc')

// Log build info on load
console.log(`[API] Build version: ${BUILD_VERSION}`)
console.log(`[API] Configured API_BASE_URL: ${API_BASE_URL}`)

// API Token for development - should be environment variable in production
const API_TOKEN = 'W1Db8xgTZnCmm0hawKaHnXlH4piXZd3VmK_lTQSxIfM'

// Get session token from localStorage for authenticated requests
const getSessionToken = () => localStorage.getItem('session_token')

// API client class
class ApiClient {
  private baseURL: string
  private token: string

  constructor(baseURL: string, token: string = '') {
    this.baseURL = baseURL
    this.token = token
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`

    // Use session token if available, otherwise fall back to API token
    const sessionToken = getSessionToken()
    const authHeader = sessionToken ? `Session ${sessionToken}` : `Bearer ${this.token}`

    const defaultHeaders = {
      'Content-Type': 'application/json',
      'Authorization': authHeader,
    }

    const config: RequestInit = {
      ...options,
      // Security: Include credentials for cookie-based authentication
      credentials: 'include',
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    }

    try {
      console.log(`[API] Making request to: ${url}`)
      console.log(`[API] Base URL: "${this.baseURL}"`)
      console.log(`[API] DEV mode: ${import.meta.env.DEV}`)
      const response = await fetch(url, config)

      console.log(`[API] Response status: ${response.status}`)

      if (!response.ok) {
        console.error(`[API] Request failed: ${response.status} ${response.statusText}`)

        // Handle authentication errors - clear invalid session and redirect to login
        if (response.status === 403 || response.status === 401) {
          console.warn('[API] Session invalid or expired, redirecting to login')
          localStorage.removeItem('session_token')
          // Only redirect if not already on login page
          if (!window.location.pathname.includes('/login')) {
            window.location.href = '/login'
          }
          throw new Error('Session expired')
        }

        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      console.log(`[API] Response data for ${endpoint}:`, data)

      // Handle API response format
      if (data.success !== undefined) {
        return data as ApiResponse<T>
      } else {
        // Wrap direct data responses
        return {
          success: true,
          data: data as T
        }
      }
    } catch (error) {
      console.error(`[API] Request to ${endpoint} failed:`, error)
      throw error
    }
  }

  // Market Data APIs
  async getAssets(params?: {
    limit?: number
    offset?: number
    asset_type?: string
  }): Promise<ApiResponse<Asset[]>> {
    const searchParams = new URLSearchParams()
    if (params?.limit) searchParams.append('limit', params.limit.toString())
    if (params?.offset) searchParams.append('offset', params.offset.toString())
    if (params?.asset_type) searchParams.append('asset_type', params.asset_type)

    const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
    return this.request<Asset[]>(`/api/markets/assets${query}`)
  }

  async getHeatmapData(sector?: string): Promise<ApiResponse<HeatmapData[]>> {
    const query = sector ? `?sector=${encodeURIComponent(sector)}` : ''
    return this.request<HeatmapData[]>(`/api/markets/heatmap${query}`)
  }

  async filterAssets(filter: AssetFilter): Promise<ApiResponse<Asset[]>> {
    return this.request<Asset[]>('/api/markets/filter', {
      method: 'POST',
      body: JSON.stringify(filter),
    })
  }

  // Portfolio APIs
  async getPositions(): Promise<ApiResponse<Position[]>> {
    return this.request<Position[]>('/api/positions')
  }

  async getPosition(symbol: string): Promise<ApiResponse<Position>> {
    return this.request<Position>(`/api/positions/${symbol}`)
  }

  async getPortfolioSummary(): Promise<ApiResponse<PortfolioSummary>> {
    return this.request<PortfolioSummary>('/api/portfolio/summary')
  }

  // Trading APIs
  async getOrders(params?: {
    status?: string
    symbol?: string
    limit?: number
  }): Promise<ApiResponse<Order[]>> {
    const searchParams = new URLSearchParams()
    if (params?.status) searchParams.append('status', params.status)
    if (params?.symbol) searchParams.append('symbol', params.symbol)
    if (params?.limit) searchParams.append('limit', params.limit.toString())

    const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
    return this.request<Order[]>(`/api/orders${query}`)
  }

  async createOrder(order: {
    symbol: string
    side: 'buy' | 'sell'
    type: 'market' | 'limit' | 'stop' | 'stop_limit'
    quantity: number
    price?: number
    stop_price?: number
  }): Promise<ApiResponse<Order>> {
    return this.request<Order>('/api/orders', {
      method: 'POST',
      body: JSON.stringify(order),
    })
  }

  // Risk & Analytics APIs
  async getRiskMetrics(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/risk/metrics')
  }

  async getMarketState(): Promise<ApiResponse<MarketState>> {
    return this.request<MarketState>('/api/market-state')
  }

  // Alert APIs
  async getAlerts(params?: {
    severity?: string
    alert_type?: string
    limit?: number
  }): Promise<ApiResponse<Alert[]>> {
    const searchParams = new URLSearchParams()
    if (params?.severity) searchParams.append('severity', params.severity)
    if (params?.alert_type) searchParams.append('alert_type', params.alert_type)
    if (params?.limit) searchParams.append('limit', params.limit.toString())

    const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
    return this.request<Alert[]>(`/api/alerts${query}`)
  }

  async acknowledgeAlert(alertId: string): Promise<ApiResponse<boolean>> {
    return this.request<boolean>(`/api/alerts/${alertId}/acknowledge`, {
      method: 'POST',
    })
  }

  async resolveAlert(alertId: string): Promise<ApiResponse<boolean>> {
    return this.request<boolean>(`/api/alerts/${alertId}/resolve`, {
      method: 'POST',
    })
  }

  // System APIs
  async getSystemStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/system/status')
  }

  async stopBot(reason?: string): Promise<ApiResponse<{ok: boolean; killed: boolean; reason?: string}>> {
    return this.request<{ok: boolean; killed: boolean; reason?: string}>('/kill', {
      method: 'POST',
      body: JSON.stringify({ reason: reason || 'manual' }),
    })
  }

  async resumeBot(note?: string): Promise<ApiResponse<{ok: boolean; killed: boolean}>> {
    return this.request<{ok: boolean; killed: boolean}>('/resume', {
      method: 'POST',
      body: JSON.stringify({ note }),
    })
  }

  async getStrategyProfiles(): Promise<ApiResponse<{profiles: Record<string, any>; active_profile?: string}>> {
    return this.request<{profiles: Record<string, any>; active_profile?: string}>('/api/strategy/profiles')
  }

  async setActiveStrategy(profileId: string): Promise<ApiResponse<{active_profile: string; restart_required: boolean}>> {
    return this.request<{active_profile: string; restart_required: boolean}>('/api/strategy/active', {
      method: 'PUT',
      body: JSON.stringify({ profile_id: profileId }),
    })
  }

  async restartRunner(reason?: string): Promise<ApiResponse<{restart_requested: boolean; reason?: string}>> {
    return this.request<{restart_requested: boolean; reason?: string}>('/api/runner/restart', {
      method: 'POST',
      body: JSON.stringify({ reason }),
    })
  }

  async getIntradayConfig(): Promise<ApiResponse<{config: Record<string, any>; restart_required: boolean; config_path?: string}>> {
    return this.request<{config: Record<string, any>; restart_required: boolean; config_path?: string}>('/api/strategy/intraday')
  }

  async updateIntradayConfig(payload: Record<string, any>): Promise<ApiResponse<{config: Record<string, any>; restart_required: boolean; config_path?: string}>> {
    return this.request<{config: Record<string, any>; restart_required: boolean; config_path?: string}>('/api/strategy/intraday', {
      method: 'PUT',
      body: JSON.stringify(payload),
    })
  }

  async getSystemHealth(): Promise<ApiResponse<any>> {
    return this.request<any>('/health')
  }

  // Runner Logs API
  async getRunnerLogs(lines: number = 100): Promise<ApiResponse<{lines: string[], total: number}>> {
    return this.request<{lines: string[], total: number}>(`/api/runner/logs?lines=${lines}`)
  }

  // Stock Selection APIs
  async getStockSelection(params?: {
    strategy?: string
    limit?: number
  }): Promise<ApiResponse<StockSelectionResponse>> {
    const searchParams = new URLSearchParams()
    if (params?.strategy) searchParams.append('strategy', params.strategy)
    if (params?.limit) searchParams.append('limit', params.limit.toString())

    const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
    return this.request<StockSelectionResponse>(`/api/stocks/screener${query}`)
  }
}

// Create and export API client instance
export const apiClient = new ApiClient(API_BASE_URL, API_TOKEN)

// WebSocket connection for real-time updates
export class WebSocketClient {
  private ws: WebSocket | null = null
  private reconnectInterval: number = 5000
  private reconnectAttempts: number = 0
  private maxReconnectAttempts: number = 10
  private listeners: Map<string, ((data: any) => void)[]> = new Map()

  // Security: Use secure WebSocket (wss://) in production
  constructor(private url: string = import.meta.env.DEV
    ? 'ws://localhost:8000/ws'
    : (import.meta.env.VITE_WS_URL || 'wss://api.wgyjdaiassistant.cc/ws')) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log(`Connecting to WebSocket: ${this.url}`) // Debug logging
        this.ws = new WebSocket(this.url)

        this.ws.onopen = () => {
          console.log('WebSocket connected')
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data)
            this.handleMessage(message)
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
          }
        }

        this.ws.onclose = () => {
          console.log('WebSocket disconnected')
          this.reconnect()
        }

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          reject(error)
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  private reconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`Reconnecting WebSocket (attempt ${this.reconnectAttempts})...`)

      setTimeout(() => {
        this.connect().catch(console.error)
      }, this.reconnectInterval)
    }
  }

  private handleMessage(message: { type: string; data: any }): void {
    const listeners = this.listeners.get(message.type) || []
    listeners.forEach(listener => listener(message.data))
  }

  subscribe(type: string, callback: (data: any) => void): () => void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, [])
    }
    this.listeners.get(type)!.push(callback)

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(type) || []
      const index = listeners.indexOf(callback)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
}

// Create and export WebSocket client instance
export const wsClient = new WebSocketClient()
