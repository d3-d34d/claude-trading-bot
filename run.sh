#!/bin/bash
# ──────────────────────────────────────────────
#  Claude Trading Bot — Mac/Linux Launcher
#  Usage: ./run.sh  [--pairs BTC ETH] [--balance 10000]
# ──────────────────────────────────────────────

set -e

# ── Load .env if it exists ─────────────────────
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✅  Loaded API key from .env"
fi

# ── Check for API key ──────────────────────────
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "❌  ANTHROPIC_API_KEY is not set."
  echo ""
  echo "    Option 1 — set it now:"
  echo "       export ANTHROPIC_API_KEY='sk-ant-your-key-here'"
  echo ""
  echo "    Option 2 — create a .env file:"
  echo "       cp .env.example .env"
  echo "       # then edit .env and paste your key"
  echo ""
  echo "    Get your key at: https://console.anthropic.com/settings/keys"
  echo ""
  exit 1
fi

# ── Check Python version ───────────────────────
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
  echo "❌  Python not found. Install Python 3.10+ from https://python.org"
  exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅  Python $PY_VERSION detected"

# ── Install dependencies if missing ────────────
if ! $PYTHON -c "import rich, anthropic, requests, pandas, numpy" 2>/dev/null; then
  echo "📦  Installing dependencies…"
  $PYTHON -m pip install -r requirements.txt --quiet
fi
echo "✅  Dependencies ready"

# ── Launch the bot ─────────────────────────────
echo ""
echo "⚡  Starting Claude Trading Bot…"
echo ""
$PYTHON trading_bot.py "$@"
