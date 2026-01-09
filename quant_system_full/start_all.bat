@echo off
echo ========================================
echo 启动量化交易系统 - Quantitative Trading System
echo ========================================
echo.

REM 检查Python是否安装
"C:\Users\26729\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python first.
    pause
    exit /b 1
)

REM 检查配置文件
if not exist .env (
    echo [WARNING] .env file not found. Using default configuration.
    echo Please create .env file from config.example.env for production use.
    echo.
)

REM 创建日志目录
if not exist "dashboard\state" (
    mkdir "dashboard\state"
)

echo [INFO] Starting Quantitative Trading System...
echo.

REM 检查现有的runner.py进程
echo [CHECK] Checking for existing runner.py processes...
tasklist /FI "IMAGENAME eq python.exe" | find "runner.py" >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARN] Found existing runner.py processes, terminating them...
    taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq *runner.py*" >nul 2>&1
    timeout /t 2 /nobreak >nul
    echo [OK] Existing processes terminated
) else (
    echo [OK] No existing runner.py processes found
)

REM 清理可能存在的PID文件
if exist "dashboard\worker\runner.pid" (
    echo [INFO] Removing stale PID file...
    del "dashboard\worker\runner.pid"
)

echo.

REM 启动后端API服务器
echo [1/3] Starting Backend API Server (Port 8000)...
start "Backend API" cmd /k "cd /d "%~dp0dashboard\backend" && "C:\Users\26729\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000"

REM 等待后端启动
timeout /t 3 /nobreak >nul
echo [INFO] Backend API starting...

REM 启动Worker进程
echo [2/3] Starting Worker Process (Background Tasks)...
start "Worker Process" cmd /k "cd /d "%~dp0dashboard\worker" && "C:\Users\26729\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe" runner.py"

REM 等待Worker启动
timeout /t 2 /nobreak >nul
echo [INFO] Worker process starting...

REM 启动前端Dashboard
echo [3/3] Starting Frontend Dashboard (Port 8501)...
start "Frontend Dashboard" cmd /k "cd /d "%~dp0dashboard\frontend" && "C:\Users\26729\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe" -m streamlit run streamlit_app.py"

REM 等待前端启动
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo 系统启动完成！System Started Successfully!
echo ========================================
echo.
echo 访问地址 Access URLs:
echo [Dashboard] http://localhost:8501
echo [API Docs]  http://localhost:8000/docs
echo [API Token] wgyjd0508 (Bearer authentication)
echo.
echo 组件状态 Component Status:
echo - Backend API: Running on port 8000
echo - Worker Process: Background tasks scheduler
echo - Frontend Dashboard: Web interface on port 8501
echo.
echo 配置信息 Configuration:
echo - Market: US Stock Market
echo - Mode: DRY_RUN=true (Paper trading)
echo - Data Source: Yahoo Finance API + MCP
echo.
echo 使用说明 Instructions:
echo 1. 打开浏览器访问 http://localhost:8501 查看Dashboard
echo 2. 系统默认运行在模拟模式，不会执行实际交易
echo 3. 可通过Dashboard监控和控制交易策略
echo 4. 按Ctrl+C关闭各个组件
echo.
echo 注意事项 Notes:
echo - 确保网络连接正常以获取市场数据
echo - 生产环境请修改.env中的ADMIN_TOKEN
echo - 实盘交易前请配置Tiger Brokers API密钥
echo.

REM 打开浏览器
timeout /t 3 /nobreak >nul
start http://localhost:8501

echo [INFO] Browser opening Dashboard...
echo [INFO] All components are starting up...
echo.
echo 按任意键关闭此窗口 Press any key to close this window...
pause >nul