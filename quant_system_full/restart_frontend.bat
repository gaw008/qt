@echo off
echo Restarting Frontend with Fixed Configuration...
echo.

cd /d "C:\quant_system_v2\quant_system_full\UI"

echo Killing any existing processes on ports 3000-3010...
for /l %%i in (3000,1,3010) do (
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr :%%i') do (
        taskkill /F /PID %%p >nul 2>&1
    )
)

echo.
echo Starting fresh Vite dev server...
echo Configuration changes applied:
echo - Removed VITE_API_BASE_URL from .env.local to use proxy
echo - Added enhanced debugging to API client
echo - Added debugging to Dashboard component
echo.

npm run dev

echo.
echo IMPORTANT: After the server starts:
echo 1. Open browser to http://localhost:3000 (not 3005)
echo 2. Press Ctrl+Shift+R to hard refresh and clear cache
echo 3. Open Developer Tools (F12) and check Console tab
echo 4. You should now see real portfolio data instead of "0" and "-"
echo.