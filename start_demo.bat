@echo off
echo ====================================================
echo Starting HybridCredit-LLM Local Demo Environment...
echo ====================================================
echo.

echo [1/3] Activating Virtual Environment...
call credit-risk-env\Scripts\activate
if %errorlevel% neq 0 (
    echo Error: Failed to activate credit-risk-env.
    pause
    exit /b %errorlevel%
)
echo Success.
echo.

echo [2/3] Changing to flask-app directory...
cd flask-app
if %errorlevel% neq 0 (
    echo Error: Failed to find flask-app directory.
    pause
    exit /b %errorlevel%
)
echo Success.
echo.

echo [3/3] Launching Flask Application...
echo.
echo The application is now running. Do not close this window!
echo Return to your presentation and click the 'Launch Live Demo' button.
echo.
python app.py

pause
