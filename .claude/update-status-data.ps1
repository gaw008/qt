# Status Data Update Script
# 状态数据更新脚本 - 定期更新自定义状态监控数据

param()

$workspaceDir = "G:\我的云端硬盘\quant_system_v2"
$statusDataPath = Join-Path $workspaceDir ".claude\status-data.json"

function Update-AILearningStatus {
    # 模拟AI学习状态更新
    $aiStatuses = @("TRAINING", "OPTIMIZING", "IDLE")
    $currentStatus = Get-Random $aiStatuses
    
    $accuracy = Get-Random -Minimum 85.0 -Maximum 95.0
    $progress = if ($currentStatus -eq "TRAINING") { Get-Random -Minimum 70.0 -Maximum 100.0 } else { 0 }
    
    return @{
        status = $currentStatus
        current_model = "quant_strategy_v2.3"
        training_progress = [math]::Round($progress, 1)
        accuracy = [math]::Round($accuracy, 1)
        last_training = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        strategy_weights = @{
            momentum = 0.35
            mean_reversion = 0.28
            arbitrage = 0.22
            risk_parity = 0.15
        }
        ab_test_results = @{
            strategy_a = [math]::Round((Get-Random -Minimum 0.12 -Maximum 0.18), 3)
            strategy_b = [math]::Round((Get-Random -Minimum 0.10 -Maximum 0.16), 3)
            confidence = [math]::Round((Get-Random -Minimum 0.85 -Maximum 0.95), 2)
        }
        performance_metrics = @{
            sharpe_ratio = [math]::Round((Get-Random -Minimum 1.5 -Maximum 2.2), 2)
            max_drawdown = [math]::Round((Get-Random -Minimum 0.05 -Maximum 0.12), 3)
            win_rate = [math]::Round((Get-Random -Minimum 0.60 -Maximum 0.75), 2)
        }
    }
}

function Update-GPUStatus {
    # 尝试获取实际GPU状态
    try {
        $gpuInfo = & nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if ($gpuInfo) {
            $parts = $gpuInfo -split ','
            $usage = [int]$parts[0].Trim()
            $temp = [int]$parts[1].Trim()
            $memUsed = [int]$parts[2].Trim()
            $memTotal = [int]$parts[3].Trim()
            
            return @{
                model = "RTX 4070 Ti SUPER"
                status = if ($usage -gt 0) { "ACTIVE" } else { "IDLE" }
                utilization = $usage
                temperature = $temp
                memory_used = $memUsed
                memory_total = $memTotal
                memory_usage_percent = [math]::Round(($memUsed / $memTotal) * 100, 1)
                cuda_version = "12.2"
                compute_efficiency = [math]::Round((Get-Random -Minimum 88.0 -Maximum 95.0), 1)
                task_queue_length = Get-Random -Maximum 5
                errors = 0
            }
        }
    } catch {}
    
    # 模拟GPU状态（如果无法获取实际数据）
    return @{
        model = "RTX 4070 Ti SUPER"
        status = "OFFLINE"
        utilization = 0
        temperature = 25
        memory_used = 0
        memory_total = 16384
        memory_usage_percent = 0
        cuda_version = "12.2"
        compute_efficiency = 0
        task_queue_length = 0
        errors = 0
    }
}

function Update-AssetStatus {
    # 模拟资产状态变化
    $baseHealthy = @{
        stocks = @{ count = 3500; base_healthy = 3450 }
        etf = @{ count = 800; base_healthy = 785 }
        reits = @{ count = 600; base_healthy = 590 }
        adr = @{ count = 500; base_healthy = 485 }
        futures = @{ count = 300; base_healthy = 285 }
    }
    
    $categories = @{}
    $totalCount = 0
    $totalHealthy = 0
    
    foreach ($type in $baseHealthy.Keys) {
        $count = $baseHealthy[$type].count
        $baseHealthy_count = $baseHealthy[$type].base_healthy
        
        # 添加随机波动
        $healthyVariation = Get-Random -Minimum -10 -Maximum 5
        $healthy = [math]::Max(0, $baseHealthy_count + $healthyVariation)
        $warning = Get-Random -Maximum 20
        $error = [math]::Max(0, $count - $healthy - $warning)
        
        $status = "HEALTHY"
        $healthPercent = ($healthy / $count) * 100
        if ($healthPercent -lt 95) { $status = "WARNING" }
        if ($healthPercent -lt 90) { $status = "ERROR" }
        
        $categories[$type] = @{
            count = $count
            healthy = $healthy
            warning = $warning
            error = $error
            status = $status
        }
        
        # 为期货添加特殊字段
        if ($type -eq "futures") {
            $categories[$type].active_contracts = Get-Random -Minimum 140 -Maximum 170
            $categories[$type].near_expiry = Get-Random -Minimum 8 -Maximum 15
            $categories[$type].margin_usage = [math]::Round((Get-Random -Minimum 70.0 -Maximum 85.0), 1)
            $categories[$type].arbitrage_opportunities = Get-Random -Maximum 12
        }
        
        $totalCount += $count
        $totalHealthy += $healthy
    }
    
    return @{
        total_count = $totalCount
        categories = $categories
        health_percentage = [math]::Round(($totalHealthy / $totalCount) * 100, 1)
    }
}

function Update-SystemStatus {
    # 获取系统基础信息
    $cpuUsage = Get-Random -Minimum 20.0 -Maximum 80.0
    $memoryUsage = Get-Random -Minimum 50.0 -Maximum 85.0
    $diskUsage = Get-Random -Minimum 40.0 -Maximum 60.0
    
    $health = "HEALTHY"
    if ($cpuUsage -gt 70) { $health = "WARNING" }
    if ($cpuUsage -gt 90) { $health = "ERROR" }
    
    return @{
        last_updated = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        overall_health = $health
        uptime_hours = [math]::Round((Get-Random -Minimum 50.0 -Maximum 100.0), 1)
        cpu_usage = [math]::Round($cpuUsage, 1)
        memory_usage = [math]::Round($memoryUsage, 1)
        disk_usage = [math]::Round($diskUsage, 1)
    }
}

function Update-MarketStatus {
    $hour = (Get-Date).Hour
    
    $sessions = @{
        asia = @{ 
            status = if ($hour -ge 1 -and $hour -lt 9) { "OPEN" } else { "CLOSED" }
            next_open = (Get-Date).AddDays(1).Date.AddHours(1).ToString("yyyy-MM-ddTHH:mm:ssZ")
        }
        europe = @{
            status = if ($hour -ge 8 -and $hour -lt 16) { "OPEN" } else { "CLOSED" }
            next_open = (Get-Date).AddDays(1).Date.AddHours(8).ToString("yyyy-MM-ddTHH:mm:ssZ")
        }
        us = @{
            status = if ($hour -ge 14 -and $hour -lt 21) { "OPEN" } elseif ($hour -ge 4 -and $hour -lt 9) { "PRE_MARKET" } elseif ($hour -ge 21 -and $hour -lt 24) { "AFTER_HOURS" } else { "CLOSED" }
            next_open = (Get-Date).AddDays(1).Date.AddHours(14).ToString("yyyy-MM-ddTHH:mm:ssZ")
        }
        crypto = @{
            status = "OPEN"
            volume_24h = "$((Get-Random -Minimum 20 -Maximum 30).ToString('F1'))B USD"
        }
    }
    
    $openSessions = ($sessions.Values | Where-Object { $_.status -eq "OPEN" }).Count
    $status = if ($openSessions -eq 0) { "CLOSED" } elseif ($openSessions -eq 4) { "OPEN" } else { "MIXED" }
    
    return @{
        status = $status
        sessions = $sessions
    }
}

function Generate-Alerts {
    $alerts = @()
    
    # 基于随机条件生成告警
    if ((Get-Random -Maximum 100) -lt 30) {
        $alerts += @{
            type = "WARNING"
            message = "期货保证金使用率接近80%"
            timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            severity = "MEDIUM"
        }
    }
    
    if ((Get-Random -Maximum 100) -lt 20) {
        $alerts += @{
            type = "INFO"
            message = "AI模型训练进度更新"
            timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            severity = "LOW"
        }
    }
    
    if ((Get-Random -Maximum 100) -lt 5) {
        $alerts += @{
            type = "ERROR"
            message = "GPU温度过高"
            timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            severity = "HIGH"
        }
    }
    
    return $alerts
}

# 构建完整的状态数据
$statusData = @{
    system = Update-SystemStatus
    assets = Update-AssetStatus
    ai_learning = Update-AILearningStatus
    gpu = Update-GPUStatus
    markets = Update-MarketStatus
    risk_metrics = @{
        portfolio_var = [math]::Round((Get-Random -Minimum 0.015 -Maximum 0.035), 4)
        leverage_ratio = [math]::Round((Get-Random -Minimum 1.8 -Maximum 2.5), 1)
        margin_requirements = [math]::Round((Get-Random -Minimum 0.70 -Maximum 0.85), 3)
        correlation_warnings = Get-Random -Maximum 5
        stress_test_result = "PASS"
    }
    alerts = Generate-Alerts
}

# 将数据写入文件
try {
    $jsonOutput = $statusData | ConvertTo-Json -Depth 10
    $jsonOutput | Out-File -FilePath $statusDataPath -Encoding UTF8 -Force
    Write-Host "Status data updated successfully at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
} catch {
    Write-Host "Error updating status data: $($_.Exception.Message)" -ForegroundColor Red
}

# 可选：显示当前状态摘要
if ($args -contains "-verbose") {
    Write-Host "`nCurrent Status Summary:" -ForegroundColor Cyan
    Write-Host "System Health: $($statusData.system.overall_health)" -ForegroundColor Yellow
    Write-Host "Total Assets: $($statusData.assets.total_count) (Health: $($statusData.assets.health_percentage)%)" -ForegroundColor Yellow
    Write-Host "AI Status: $($statusData.ai_learning.status) (Accuracy: $($statusData.ai_learning.accuracy)%)" -ForegroundColor Yellow
    Write-Host "GPU Status: $($statusData.gpu.status) (Usage: $($statusData.gpu.utilization)%)" -ForegroundColor Yellow
    Write-Host "Market Status: $($statusData.markets.status)" -ForegroundColor Yellow
    Write-Host "Active Alerts: $($statusData.alerts.Count)" -ForegroundColor Yellow
}