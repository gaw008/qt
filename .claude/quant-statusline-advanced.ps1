# Advanced Quant System Status Monitor
# 高级多资产量化交易系统状态监控

param()

# 读取输入的JSON数据
$input = $input | ConvertFrom-Json -ErrorAction SilentlyContinue

# 系统基础信息
$workspaceDir = if ($input.workspace.project_dir) { $input.workspace.project_dir } else { "G:\我的云端硬盘\quant_system_v2" }
$model = if ($input.model.display_name) { $input.model.display_name -replace "Claude ", "C" } else { "Claude" }
$outputStyle = if ($input.output_style.name) { $input.output_style.name } else { "Default" }

# 颜色代码
$GREEN = "`e[32m"
$RED = "`e[31m"
$YELLOW = "`e[33m"
$BLUE = "`e[34m"
$CYAN = "`e[36m"
$MAGENTA = "`e[35m"
$RESET = "`e[0m"

# 状态数据文件路径
$statusDataPath = Join-Path $workspaceDir ".claude\status-data.json"

function Get-StatusData {
    try {
        if (Test-Path $statusDataPath) {
            $jsonContent = Get-Content $statusDataPath -Raw | ConvertFrom-Json
            return $jsonContent
        } else {
            # 返回默认状态数据
            return @{
                system = @{ overall_health = "UNKNOWN"; cpu_usage = 0 }
                assets = @{ total_count = 5700; health_percentage = 95.0 }
                ai_learning = @{ status = "IDLE"; training_progress = 0; accuracy = 85.0 }
                gpu = @{ status = "OFFLINE"; utilization = 0; temperature = 0 }
                markets = @{ status = "UNKNOWN" }
            }
        }
    } catch {
        # 发生错误时返回错误状态
        return @{
            system = @{ overall_health = "ERROR"; cpu_usage = 0 }
            assets = @{ total_count = 0; health_percentage = 0 }
            ai_learning = @{ status = "ERROR"; training_progress = 0; accuracy = 0 }
            gpu = @{ status = "ERROR"; utilization = 0; temperature = 0 }
            markets = @{ status = "ERROR" }
        }
    }
}

function Get-SystemHealthDisplay {
    param($statusData)
    
    $health = $statusData.system.overall_health
    $cpuUsage = [math]::Round($statusData.system.cpu_usage, 0)
    
    $statusColor = switch ($health) {
        "HEALTHY" { $GREEN }
        "WARNING" { $YELLOW }
        "ERROR" { $RED }
        default { $YELLOW }
    }
    
    $healthText = switch ($health) {
        "HEALTHY" { "正常" }
        "WARNING" { "警告" }
        "ERROR" { "错误" }
        default { "未知" }
    }
    
    return "${statusColor}系统:${healthText}(${cpuUsage}%)${RESET}"
}

function Get-AssetStatusDisplay {
    param($statusData)
    
    $totalAssets = $statusData.assets.total_count
    $healthPercent = [math]::Round($statusData.assets.health_percentage, 1)
    
    $statusColor = $GREEN
    if ($healthPercent -lt 95) { $statusColor = $YELLOW }
    if ($healthPercent -lt 90) { $statusColor = $RED }
    
    # 获取各类别状态摘要
    $categories = $statusData.assets.categories
    $warnings = 0
    $errors = 0
    
    foreach ($category in $categories.PSObject.Properties) {
        $warnings += $category.Value.warning
        $errors += $category.Value.error
    }
    
    $alertText = ""
    if ($errors -gt 0) { $alertText = "E:$errors" }
    elseif ($warnings -gt 0) { $alertText = "W:$warnings" }
    
    return "${statusColor}资产:${totalAssets}(${healthPercent}%)${alertText}${RESET}"
}

function Get-AIStatusDisplay {
    param($statusData)
    
    $aiStatus = $statusData.ai_learning.status
    $progress = [math]::Round($statusData.ai_learning.training_progress, 1)
    $accuracy = [math]::Round($statusData.ai_learning.accuracy, 1)
    
    $statusColor = $GREEN
    $statusText = switch ($aiStatus) {
        "TRAINING" { "训练中"; $statusColor = $CYAN }
        "OPTIMIZING" { "优化中"; $statusColor = $YELLOW }
        "IDLE" { "待机"; $statusColor = $BLUE }
        "ERROR" { "错误"; $statusColor = $RED }
        default { "未知"; $statusColor = $YELLOW }
    }
    
    if ($accuracy -lt 80) { $statusColor = $YELLOW }
    if ($accuracy -lt 70) { $statusColor = $RED }
    
    $displayText = if ($aiStatus -eq "TRAINING") { "${progress}%" } else { "${accuracy}%" }
    
    return "${statusColor}AI:${statusText}(${displayText})${RESET}"
}

function Get-GPUStatusDisplay {
    param($statusData)
    
    $gpuStatus = $statusData.gpu.status
    $usage = $statusData.gpu.utilization
    $temp = $statusData.gpu.temperature
    $memPercent = [math]::Round($statusData.gpu.memory_usage_percent, 0)
    
    $statusColor = switch ($gpuStatus) {
        "ACTIVE" { 
            $color = $GREEN
            if ($usage -gt 90 -or $temp -gt 80) { $color = $YELLOW }
            if ($usage -eq 100 -or $temp -gt 85) { $color = $RED }
            $color
        }
        "IDLE" { $BLUE }
        "OFFLINE" { $YELLOW }
        "ERROR" { $RED }
        default { $YELLOW }
    }
    
    $statusText = switch ($gpuStatus) {
        "ACTIVE" { "运行中" }
        "IDLE" { "空闲" }
        "OFFLINE" { "离线" }
        "ERROR" { "错误" }
        default { "未知" }
    }
    
    if ($gpuStatus -eq "ACTIVE") {
        return "${statusColor}GPU:${statusText}(${usage}%/${temp}C)${RESET}"
    } else {
        return "${statusColor}GPU:${statusText}${RESET}"
    }
}

function Get-FuturesStatusDisplay {
    param($statusData)
    
    if ($statusData.assets.categories.futures) {
        $futures = $statusData.assets.categories.futures
        $marginUsage = [math]::Round($futures.margin_usage, 1)
        $activeContracts = $futures.active_contracts
        $nearExpiry = $futures.near_expiry
        
        $statusColor = $GREEN
        $statusText = "正常"
        
        if ($marginUsage -gt 80) {
            $statusColor = $YELLOW
            $statusText = "高保证金"
        }
        
        if ($marginUsage -gt 90) {
            $statusColor = $RED
            $statusText = "保证金风险"
        }
        
        return "${statusColor}期货:${statusText}(${activeContracts}/${nearExpiry})${RESET}"
    } else {
        return "${YELLOW}期货:N/A${RESET}"
    }
}

function Get-MarketStatusDisplay {
    param($statusData)
    
    $marketStatus = $statusData.markets.status
    
    $statusColor = switch ($marketStatus) {
        "OPEN" { $GREEN }
        "MIXED" { $YELLOW }
        "CLOSED" { $BLUE }
        "ERROR" { $RED }
        default { $YELLOW }
    }
    
    $statusText = switch ($marketStatus) {
        "OPEN" { "开市" }
        "MIXED" { "部分开市" }
        "CLOSED" { "休市" }
        "ERROR" { "错误" }
        default { "未知" }
    }
    
    # 检查加密货币市场（24/7）
    if ($statusData.markets.sessions.crypto.status -eq "OPEN") {
        $statusText += "+加密"
    }
    
    return "${statusColor}市场:${statusText}${RESET}"
}

function Get-AlertsDisplay {
    param($statusData)
    
    if ($statusData.alerts -and $statusData.alerts.Count -gt 0) {
        $criticalAlerts = @($statusData.alerts | Where-Object { $_.severity -eq "HIGH" })
        $warningAlerts = @($statusData.alerts | Where-Object { $_.severity -eq "MEDIUM" })
        
        if ($criticalAlerts.Count -gt 0) {
            return "${RED}告警:${criticalAlerts.Count}${RESET}"
        } elseif ($warningAlerts.Count -gt 0) {
            return "${YELLOW}警告:${warningAlerts.Count}${RESET}"
        }
    }
    
    return ""
}

# 获取状态数据
$statusData = Get-StatusData

# 构建各个状态组件
$systemHealth = Get-SystemHealthDisplay $statusData
$assetStatus = Get-AssetStatusDisplay $statusData
$aiStatus = Get-AIStatusDisplay $statusData
$gpuStatus = Get-GPUStatusDisplay $statusData
$futuresStatus = Get-FuturesStatusDisplay $statusData
$marketStatus = Get-MarketStatusDisplay $statusData
$alertsStatus = Get-AlertsDisplay $statusData

# 获取当前时间
$currentTime = Get-Date -Format "HH:mm:ss"

# 构建完整状态栏
$statusComponents = @($model, $systemHealth, $assetStatus, $aiStatus, $gpuStatus, $futuresStatus, $marketStatus)

# 添加告警信息（如果有的话）
if ($alertsStatus) {
    $statusComponents += $alertsStatus
}

# 添加时间戳
$statusComponents += "${CYAN}${currentTime}${RESET}"

# 组合状态栏
$statusLine = ($statusComponents | Where-Object { $_ }) -join " | "

# 输出状态栏
Write-Output $statusLine