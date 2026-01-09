import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Format currency values
export function formatCurrency(value: number, currency: string = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

// Format large numbers with K, M, B suffixes
export function formatNumber(value: number): string {
  if (Math.abs(value) >= 1e9) {
    return (value / 1e9).toFixed(1) + 'B'
  } else if (Math.abs(value) >= 1e6) {
    return (value / 1e6).toFixed(1) + 'M'
  } else if (Math.abs(value) >= 1e3) {
    return (value / 1e3).toFixed(1) + 'K'
  }
  return value.toFixed(2)
}

// Format percentage values
export function formatPercent(value: number, decimals: number = 2): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`
}

// Format basis points
export function formatBasisPoints(value: number): string {
  return `${value.toFixed(1)} bps`
}

// Get color class for price changes
export function getChangeColor(change: number): string {
  if (change > 0) return 'text-bull'
  if (change < 0) return 'text-bear'
  return 'text-neutral'
}

// Get background color class for price changes
export function getChangeBgColor(change: number): string {
  if (change > 0) return 'bg-bull/10'
  if (change < 0) return 'bg-bear/10'
  return 'bg-neutral/10'
}

// Format time ago
export function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)

  if (diffInSeconds < 60) {
    return 'Just now'
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60)
    return `${minutes}m ago`
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600)
    return `${hours}h ago`
  } else {
    const days = Math.floor(diffInSeconds / 86400)
    return `${days}d ago`
  }
}

// Format date for display
export function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// Truncate text with ellipsis
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength - 3) + '...'
}

// Generate random ID
export function generateId(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36)
}

// Debounce function
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout>
  return (...args: Parameters<T>) => {
    clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

// Deep merge objects
export function deepMerge<T extends Record<string, any>>(target: T, source: Partial<T>): T {
  const result: Record<string, any> = { ...target }

  for (const key in source) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      result[key] = deepMerge(result[key] || {}, source[key] as any)
    } else if (source[key] !== undefined) {
      result[key] = source[key] as any
    }
  }

  return result as T
}

// Calculate portfolio metrics
export function calculatePortfolioMetrics(positions: any[]) {
  const totalValue = positions.reduce((sum, pos) => sum + pos.market_value, 0)
  const totalPnL = positions.reduce((sum, pos) => sum + pos.unrealized_pnl, 0)
  const totalPnLPercent = totalValue > 0 ? (totalPnL / (totalValue - totalPnL)) * 100 : 0

  return {
    totalValue,
    totalPnL,
    totalPnLPercent,
    positionsCount: positions.length,
  }
}

// Validate API response
export function isValidApiResponse<T>(response: any): response is { success: boolean; data: T } {
  return response && typeof response.success === 'boolean' && response.data !== undefined
}
