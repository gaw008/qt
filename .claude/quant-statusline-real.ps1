# Real Quant System Status Monitor
# 基于实际系统数据的量化交易系统状态监控

param()

# 读取输入的JSON数据
$input = $input | ConvertFrom-Json -ErrorAction SilentlyContinue

# 系统基础信息
$workspaceDir = if ($input.workspace.project_dir) { $input.workspace.project_dir } else { "G:\我的云端硬盘\quant_system_v2" }
$model = if ($input.model.display_name) { ($input.model.display_name -split " ")[0] } else { "C" }

# 颜色代码
$GREEN = "`e[32m"
$RED = "`e[31m"
$YELLOW = "`e[33m"
$BLUE = "`e[34m"
$CYAN = "`e[36m"
$MAGENTA = "`e[35m"
$RESET = "`e[0m"

# 实际状态文件路径
$realStatusPath = Join-Path $workspaceDir "quant_system_full\dashboard\state\status.json"
$customStatusPath = Join-Path $workspaceDir ".claude\status-data.json"

function Get-RealSystemStatus {
    try {
        if (Test-Path $realStatusPath) {
            $jsonContent = Get-Content $realStatusPath -Raw | ConvertFrom-Json
            return $jsonContent
        } else {
            return $null
        }
    } catch {
        return $null
    }
}

function Get-CustomStatusData {
    try {
        if (Test-Path $customStatusPath) {
            $jsonContent = Get-Content $customStatusPath -Raw | ConvertFrom-Json
            return $jsonContent
        } else {
            return $null
        }
    } catch {
        return $null
    }
}

function Get-SystemHealthStatus {
    param($realStatus)
    
    if ($realStatus) {
        $healthyTasks = $realStatus.task_health.healthy_tasks
        $totalTasks = $realStatus.task_health.total_tasks
        $totalErrors = $realStatus.task_statistics.total_errors
        $uptime = $realStatus.scheduler_uptime
        
        if ($uptime -eq "Running" -and $healthyTasks -eq $totalTasks -and $totalErrors -eq 0) {
            return "${GREEN}系统:正常(${healthyTasks}/${totalTasks})${RESET}"
        } elseif ($uptime -eq "Running" -and $healthyTasks -gt 0) {
            return "${YELLOW}系统:警告(${healthyTasks}/${totalTasks})${RESET}"
        } else {
            return "${RED}系统:错误(${healthyTasks}/${totalTasks})${RESET}"
        }
    } else {
        return "${YELLOW}系统:未知${RESET}"
    }
}

function Get-MarketStatusFromReal {
    param($realStatus)
    
    if ($realStatus -and $realStatus.market_status) {
        $marketPhase = $realStatus.market_status.market_phase
        $isOpen = $realStatus.market_status.is_market_open
        $marketType = $realStatus.market_status.market_type
        
        if ($isOpen) {
            return "${GREEN}市场:${marketType}开市${RESET}"
        } elseif ($marketPhase -eq "pre_market" -or $marketPhase -eq "after_hours") {
            return "${YELLOW}市场:${marketType}盘前后${RESET}"
        } else {
            return "${BLUE}市场:${marketType}休市${RESET}"
        }
    } else {
        return "${YELLOW}市场:未知${RESET}"
    }
}

function Get-TaskPerformanceStatus {
    param($realStatus)
    
    if ($realStatus -and $realStatus.task_statistics) {
        $selectionRuns = $realStatus.task_statistics.selection_runs
        $tradingRuns = $realStatus.task_statistics.trading_runs
        $monitoringRuns = $realStatus.task_statistics.monitoring_runs
        
        $totalRuns = $selectionRuns + $tradingRuns + $monitoringRuns
        
        if ($totalRuns -gt 1000) {
            return "${GREEN}任务:高效(${totalRuns}次)${RESET}"
        } elseif ($totalRuns -gt 100) {
            return "${YELLOW}任务:正常(${totalRuns}次)${RESET}"
        } else {
            return "${BLUE}任务:启动中(${totalRuns}次)${RESET}"
        }
    } else {
        return "${YELLOW}任务:N/A${RESET}"
    }
}

function Get-PnLStatus {
    param($realStatus)
    
    if ($realStatus -and $realStatus.PSObject.Properties.Name -contains "pnl") {
        $pnl = $realStatus.pnl
        
        if ($pnl -gt 0) {
            return "${GREEN}盈亏:+${pnl}${RESET}"
        } elseif ($pnl -lt 0) {
            return "${RED}盈亏:${pnl}${RESET}"
        } else {
            return "${BLUE}盈亏:0.0${RESET}"
        }
    } else {
        return "${BLUE}盈亏:N/A${RESET}"
    }
}

function Get-PositionsStatus {
    param($realStatus)
    
    if ($realStatus -and $realStatus.positions) {
        $positionCount = $realStatus.positions.Count
        
        if ($positionCount -gt 0) {
            return "${GREEN}持仓:${positionCount}${RESET}"
        } else {
            return "${BLUE}持仓:0${RESET}"
        }
    } else {
        return "${BLUE}持仓:N/A${RESET}"
    }
}

function Get-AIStatusFromCustom {
    param($customStatus)
    
    if ($customStatus -and $customStatus.ai_learning) {
        $aiStatus = $customStatus.ai_learning.status
        $accuracy = [math]::Round($customStatus.ai_learning.accuracy, 1)
        
        $statusColor = switch ($aiStatus) {
            "TRAINING" { $CYAN }
            "OPTIMIZING" { $YELLOW }
            "IDLE" { $BLUE }
            "ERROR" { $RED }
            default { $YELLOW }
        }
        
        $statusText = switch ($aiStatus) {
            "TRAINING" { "训练中" }
            "OPTIMIZING" { "优化中" }  
            "IDLE" { "待机" }
            "ERROR" { "错误" }
            default { "未知" }
        }
        
        return "${statusColor}AI:${statusText}(${accuracy}%)${RESET}"
    } else {
        return "${BLUE}AI:待机${RESET}"
    }
}

function Get-GPUStatusFromCustom {
    param($customStatus)
    
    if ($customStatus -and $customStatus.gpu) {
        $gpuStatus = $customStatus.gpu.status
        $usage = $customStatus.gpu.utilization
        $temp = $customStatus.gpu.temperature
        
        $statusColor = switch ($gpuStatus) {
            "ACTIVE" { 
                $color = $GREEN
                if ($usage -gt 90 -or $temp -gt 80) { $color = $YELLOW }
                if ($temp -gt 85) { $color = $RED }
                $color
            }
            "IDLE" { $BLUE }
            "OFFLINE" { $YELLOW }
            default { $YELLOW }
        }
        
        if ($gpuStatus -eq "ACTIVE") {
            return "${statusColor}GPU:${usage}%/${temp}C${RESET}"
        } else {
            return "${statusColor}GPU:离线${RESET}"
        }
    } else {
        # 尝试获取实际GPU状态
        try {
            $gpuInfo = & nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null
            if ($gpuInfo) {
                $parts = $gpuInfo -split ','
                $usage = [int]$parts[0].Trim()
                $temp = [int]$parts[1].Trim()
                
                $color = $GREEN
                if ($usage -gt 90 -or $temp -gt 80) { $color = $YELLOW }
                if ($temp -gt 85) { $color = $RED }
                
                return "${color}GPU:${usage}%/${temp}C${RESET}"
            }
        } catch {}
        
        return "${YELLOW}GPU:N/A${RESET}"
    }
}

function Get-AssetStatusFromCustom {
    param($customStatus)
    
    if ($customStatus -and $customStatus.assets) {
        $totalAssets = $customStatus.assets.total_count
        $healthPercent = [math]::Round($customStatus.assets.health_percentage, 1)
        
        $statusColor = $GREEN
        if ($healthPercent -lt 95) { $statusColor = $YELLOW }
        if ($healthPercent -lt 90) { $statusColor = $RED }
        
        return "${statusColor}资产:${totalAssets}(${healthPercent}%)${RESET}"
    } else {
        return "${BLUE}资产:5700(95.0%)${RESET}"
    }
}

# 获取实际系统状态和自定义状态数据
$realStatus = Get-RealSystemStatus
$customStatus = Get-CustomStatusData

# 构建状态组件
$systemHealth = Get-SystemHealthStatus $realStatus
$marketStatus = Get-MarketStatusFromReal $realStatus
$taskStatus = Get-TaskPerformanceStatus $realStatus
$pnlStatus = Get-PnLStatus $realStatus
$positionsStatus = Get-PositionsStatus $realStatus

# 从自定义状态获取额外信息
$aiStatus = Get-AIStatusFromCustom $customStatus
$gpuStatus = Get-GPUStatusFromCustom $customStatus
$assetStatus = Get-AssetStatusFromCustom $customStatus

# 获取当前时间
$currentTime = Get-Date -Format "HH:mm:ss"

# 构建完整状态栏 - 优化显示，重点信息优先
$statusComponents = @()
$statusComponents += $model
$statusComponents += $systemHealth
$statusComponents += $marketStatus

# 只在有实际数据时显示PnL和持仓
if ($realStatus) {
    if ($realStatus.pnl -ne 0) { $statusComponents += $pnlStatus }
    if ($realStatus.positions.Count -gt 0) { $statusComponents += $positionsStatus }
}

$statusComponents += $taskStatus
$statusComponents += $aiStatus
$statusComponents += $gpuStatus
$statusComponents += "${CYAN}${currentTime}${RESET}"

# 组合状态栏
$statusLine = ($statusComponents | Where-Object { $_ }) -join " | "

# 输出状态栏
Write-Output $statusLine