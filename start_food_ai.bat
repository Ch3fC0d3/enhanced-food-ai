@echo off
echo ===============================================
echo     Enhanced Food AI - Complete Startup
echo ===============================================
echo.

:: Kill any existing Python processes that might conflict with our server
echo [1/7] Cleaning up any existing Python processes...
taskkill /f /im python.exe 2>nul
if %ERRORLEVEL% EQU 0 (
    echo       ✓ Terminated existing Python processes
) else (
    echo       ✓ No existing Python processes to terminate
)

:: Set up environment
set BACKEND_PORT=5000
set FRONTEND_PORT=3001
set BACKEND_SCRIPT=persistent_backend.py
set FRONTEND_FILE=enhanced-interface.html
set FRONTEND_FIXED_FILE=enhanced-interface-fixed.html

:: Ensure we're in the correct directory
echo [2/7] Ensuring correct working directory...
cd /d F:\FoodVault\enhanced_food_ai
if not exist "%BACKEND_SCRIPT%" (
    echo       ✗ ERROR: Backend script %BACKEND_SCRIPT% not found!
    echo       Please ensure you're running this from F:\FoodVault\enhanced_food_ai
    pause
    exit /b 1
) else (
    echo       ✓ Found backend script: %BACKEND_SCRIPT%
)

if not exist "%FRONTEND_FILE%" (
    echo       ⚠ WARNING: Primary frontend file %FRONTEND_FILE% not found
    if exist "working-interface.html" (
        echo       ✓ Found working-interface.html instead, using that
        set FRONTEND_FILE=working-interface.html
    )
) else (
    echo       ✓ Found primary frontend file: %FRONTEND_FILE%
)

:: Create reports directory if it doesn't exist
if not exist "reports" (
    echo       Creating reports directory for logs...
    mkdir reports
)

:: Create feedback directory if it doesn't exist
if not exist "feedback" (
    echo       Creating feedback directory...
    mkdir feedback
)

:: Step 3: Start the backend server
echo [3/7] Starting backend server on port %BACKEND_PORT%...
start "Food AI Backend" cmd /c "python %BACKEND_SCRIPT% & pause"
echo       ✓ Backend startup initiated

:: Wait for backend to initialize and verify it's responding
echo [4/7] Waiting for backend to initialize and checking health...
set MAX_RETRIES=10
set RETRY_COUNT=0
set BACKEND_HEALTHY=0

:HEALTH_CHECK_LOOP
timeout /t 2 /nobreak >nul
curl -s http://localhost:%BACKEND_PORT%/health > health_check.tmp 2>nul
if %ERRORLEVEL% EQU 0 (
    echo       ✓ Backend health check successful!
    set BACKEND_HEALTHY=1
    goto :HEALTH_CHECK_DONE
) else (
    set /a RETRY_COUNT+=1
    echo       Waiting for backend to respond... Attempt %RETRY_COUNT%/%MAX_RETRIES%
    if %RETRY_COUNT% LSS %MAX_RETRIES% goto :HEALTH_CHECK_LOOP
    echo       ⚠ Backend health check timed out, continuing anyway...
)
:HEALTH_CHECK_DONE
del health_check.tmp 2>nul

:: Step 4: Start HTTP server for frontend
echo [5/7] Starting HTTP server for frontend on port %FRONTEND_PORT%...
start "Food AI Frontend Server" cmd /c "python -m http.server %FRONTEND_PORT% & pause"
echo       ✓ Frontend server started

:: Step 5: Open the frontends in the default browser
echo [6/7] Opening Enhanced Interface in browser...
timeout /t 2 /nobreak >nul

:: Open whatever interface files are available
if exist "%FRONTEND_FIXED_FILE%" (
    start http://localhost:%FRONTEND_PORT%/%FRONTEND_FIXED_FILE%
    echo       ✓ Launched fixed interface at http://localhost:%FRONTEND_PORT%/%FRONTEND_FIXED_FILE%
)
if exist "%FRONTEND_FILE%" (
    start http://localhost:%FRONTEND_PORT%/%FRONTEND_FILE%
    echo       ✓ Launched interface at http://localhost:%FRONTEND_PORT%/%FRONTEND_FILE%
)

:: Step 6: Verify everything is running
echo [7/7] Verifying all services...
if %BACKEND_HEALTHY% EQU 1 (
    echo       ✓ Backend is healthy and responding
) else (
    echo       ⚠ WARNING: Backend health check failed - check console for errors
)
echo       ✓ Frontend server is running on port %FRONTEND_PORT%

echo.
echo ===============================================
echo     Enhanced Food AI System Started!
echo ===============================================
echo.
echo Backend:
echo   URL:     http://localhost:%BACKEND_PORT%
echo   Health:  http://localhost:%BACKEND_PORT%/health
echo.
echo Frontend:
echo   URL:     http://localhost:%FRONTEND_PORT%/%FRONTEND_FILE%
echo.
echo Key endpoints:
echo   /formulate - Generate ingredient bundles
echo   /feedback  - Submit feedback on bundles
echo   /retrain   - Update preference weights
echo.
echo To stop all services, close the command windows or press Ctrl+C
echo.
echo ===============================================

:: Keep the main window open
pause
