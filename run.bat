@echo off
REM ──────────────────────────────────────────────
REM  Claude Trading Bot — Windows Launcher
REM  Usage: run.bat  [--pairs BTC ETH] [--balance 10000]
REM ──────────────────────────────────────────────

REM ── Load .env if it exists ─────────────────────
if exist ".env" (
    for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
        if not "%%A"=="" if not "%%A:~0,1%"=="#" set %%A=%%B
    )
    echo [OK] Loaded API key from .env
)

REM ── Check for API key ──────────────────────────
if "%ANTHROPIC_API_KEY%"=="" (
    echo.
    echo [ERROR] ANTHROPIC_API_KEY is not set.
    echo.
    echo   Option 1 - set it in this terminal:
    echo      set ANTHROPIC_API_KEY=sk-ant-your-key-here
    echo.
    echo   Option 2 - create a .env file:
    echo      copy .env.example .env
    echo      then open .env and paste your key
    echo.
    echo   Get your key at: https://console.anthropic.com/settings/keys
    echo.
    pause
    exit /b 1
)

REM ── Check Python ───────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)
echo [OK] Python detected

REM ── Install dependencies ───────────────────────
python -c "import rich, anthropic, requests, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install -r requirements.txt --quiet
)
echo [OK] Dependencies ready

REM ── Launch ─────────────────────────────────────
echo.
echo Starting Claude Trading Bot...
echo.
python trading_bot.py %*
pause
