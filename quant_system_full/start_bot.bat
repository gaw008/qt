@echo off
chcp 65001 >nul
title 量化交易机器人启动器

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   量化交易机器人启动器                        ║
echo ║              Quantitative Trading Bot Launcher              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🚀 启动量化交易系统...
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)

REM 进入项目目录
cd /d "%~dp0"

REM 启动完整系统
python start_bot.py --sectors IT,HEALTHCARE --interval 300

pause