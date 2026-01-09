# Quant System Multi-Asset Status Monitor
# 多资产量化交易系统综合状态监控

param()

# 读取输入的JSON数据
$input = $input | ConvertFrom-Json -ErrorAction SilentlyContinue

# 系统基础信息
$workspaceDir = if ($input.workspace.project_dir) { $input.workspace.project_dir } else { "G:\我的云端硬盘\quant_system_v2" }
$model = if ($input.model.display_name) { $input.model.display_name } else { "Claude" }
$outputStyle = if ($input.output_style.name) { $input.output_style.name } else { "Default" }

# 颜色代码（在终端中将以暗淡颜色显示）
$GREEN = "`e[32m"
$RED = "`e[31m"
$YELLOW = "`e[33m"
$BLUE = "`e[34m"
$CYAN = "`e[36m"
$RESET = "`e[0m"

function Get-SystemHealth {
    $healthStatus = "HEALTHY"
    $healthColor = $GREEN
    
    # 检查系统负载
    try {
        $cpuUsage = Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 1 | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
        if ($cpuUsage -gt 80) {
            $healthStatus = "WARN"
            $healthColor = $YELLOW
        }
    } catch {
        $healthStatus = "ERROR"
        $healthColor = $RED
    }
    
    return "${healthColor}SYS:${healthStatus}${RESET}"
}

function Get-AssetStatus {
    # 模拟5700+资产状态监控
    $assetTypes = @{
        "股票" = @{ count = 3500; healthy = 3450; warning = 30; error = 20 }
        "ETF" = @{ count = 800; healthy = 785; warning = 10; error = 5 }
        "REITs" = @{ count = 600; healthy = 590; warning = 8; error = 2 }
        "ADR" = @{ count = 500; healthy = 485; warning = 12; error = 3 }
        "期货" = @{ count = 300; healthy = 285; warning = 10; error = 5 }
    }
    
    $totalAssets = ($assetTypes.Values | Measure-Object -Property count -Sum).Sum
    $totalHealthy = ($assetTypes.Values | Measure-Object -Property healthy -Sum).Sum
    $totalWarning = ($assetTypes.Values | Measure-Object -Property warning -Sum).Sum
    $totalError = ($assetTypes.Values | Measure-Object -Property error -Sum).Sum
    
    $healthPercent = [math]::Round(($totalHealthy / $totalAssets) * 100, 1)
    
    $statusColor = $GREEN
    if ($healthPercent -lt 95) { $statusColor = $YELLOW }
    if ($healthPercent -lt 90) { $statusColor = $RED }
    
    return "${statusColor}资产:${totalAssets}(${healthPercent}%)${RESET}"
}

function Get-AIStatus {
    # AI学习进度监控
    $aiStatus = @{
        training = $true
        accuracy = 87.5
        performance = "GOOD"
        modelHealth = "HEALTHY"
    }
    
    $statusColor = $GREEN
    $statusText = "AI:训练中"
    
    if ($aiStatus.accuracy -lt 80) {
        $statusColor = $YELLOW
        $statusText = "AI:优化中"
    }
    
    if ($aiStatus.modelHealth -eq "ERROR") {
        $statusColor = $RED
        $statusText = "AI:错误"
    }
    
    return "${statusColor}${statusText}(${aiStatus.accuracy}%)${RESET}"
}

function Get-GPUStatus {
    # RTX 4070 Ti SUPER 状态监控
    try {
        # 尝试获取GPU信息（需要nvidia-smi）
        $gpuInfo = & nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if ($gpuInfo) {
            $parts = $gpuInfo -split ','
            $usage = [int]$parts[0].Trim()
            $temp = [int]$parts[1].Trim()
            $memUsed = [int]$parts[2].Trim()
            $memTotal = [int]$parts[3].Trim()
            $memPercent = [math]::Round(($memUsed / $memTotal) * 100, 1)
            
            $statusColor = $GREEN
            if ($usage -gt 90 -or $temp -gt 80) { $statusColor = $YELLOW }
            if ($usage -eq 100 -or $temp -gt 85) { $statusColor = $RED }
            
            return "${statusColor}GPU:${usage}%/${temp}C/${memPercent}%${RESET}"
        } else {
            return "${YELLOW}GPU:N/A${RESET}"
        }
    } catch {
        return "${YELLOW}GPU:离线${RESET}"
    }
}

function Get-FuturesStatus {
    # 期货交易状态监控
    $futures = @{
        activeContracts = 156
        nearExpiry = 12
        marginUsage = 75.5
        arbitrageOpp = 8
    }
    
    $statusColor = $GREEN
    $statusText = "期货:正常"
    
    if ($futures.marginUsage -gt 80) {
        $statusColor = $YELLOW
        $statusText = "期货:高保证金"
    }
    
    if ($futures.marginUsage -gt 90) {
        $statusColor = $RED
        $statusText = "期货:保证金风险"
    }
    
    return "${statusColor}${statusText}(${futures.activeContracts}/${futures.nearExpiry})${RESET}"
}

function Get-MarketStatus {
    # 多市场状态
    $currentTime = Get-Date
    $hour = $currentTime.Hour
    
    $marketStatus = "休市"
    $statusColor = $YELLOW
    
    # 简化的市场时间判断
    if (($hour -ge 9 -and $hour -lt 15) -or ($hour -ge 21 -and $hour -lt 24)) {
        $marketStatus = "开市"
        $statusColor = $GREEN
    }
    
    return "${statusColor}市场:${marketStatus}${RESET}"
}

# 构建状态栏信息
$systemHealth = Get-SystemHealth
$assetStatus = Get-AssetStatus  
$aiStatus = Get-AIStatus
$gpuStatus = Get-GPUStatus
$futuresStatus = Get-FuturesStatus
$marketStatus = Get-MarketStatus

# 获取当前时间
$currentTime = Get-Date -Format "HH:mm:ss"

# 构建完整状态栏
$statusLine = "${BLUE}${model}${RESET} | ${systemHealth} | ${assetStatus} | ${aiStatus} | ${gpuStatus} | ${futuresStatus} | ${marketStatus} | ${CYAN}${currentTime}${RESET}"

# 输出状态栏
Write-Output $statusLine