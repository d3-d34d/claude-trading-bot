# ⚡ Claude-Powered Crypto Trading Bot

> A live terminal trading bot powered by Claude AI — collects real-world crypto market data, analyzes technical indicators, predicts short-term price movements, and paper-trades automatically.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Powered by Claude](https://img.shields.io/badge/Powered%20by-Claude%20AI-orange?logo=anthropic)
![Paper Trading](https://img.shields.io/badge/Mode-Paper%20Trading-green)
![Live Data](https://img.shields.io/badge/Data-Binance%20Live-yellow?logo=binance)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📺 Live Terminal Preview

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  ⚡ CLAUDE TRADING BOT │ PAPER TRADING │  2026-04-07  14:32:07                      ║
║  Portfolio: $10,284.37  │  P&L: +$284.37  (+2.84%)                                 ║
╠═════════════════════════════════════╦════════════════════════════════════════════════╣
║  📈 Live Prices & AI Signals        ║  💼 Portfolio                                 ║
║ ─────────────────────────────────── ║ ────────────────────────────────────────────  ║
║  Pair       Price      24h%  Action ║  Asset    Qty           Value        P&L      ║
║  BTC/USDT   $83,412    +2.1%  BUY  ║  USDT     —            $7,284.37     —        ║
║  ETH/USDT   $3,201     -0.4%  HOLD ║  BTC      0.012140     $1,013.44    +$13.44   ║
║  SOL/USDT   $142.50    +1.8%  SELL ║  ETH      0.295000     $944.30      +$5.60    ║
╠═════════════════════════════════════╬════════════════════════════════════════════════╣
║  📊 Technical Indicators            ║  📋 Trade History                             ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║  🤖 Activity Log — Next analysis in 47s                                             ║
║  14:30:01  ✅ BUY  BTCUSDT  $83,412  (conf=8)                                      ║
║  14:25:00  🔴 SELL SOLUSDT  $143.10  P&L: +$5.20                                 ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
```

---

## 🧂 How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                   BOT ARCHITECTURE                          │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │  Binance     │    │  Technical   │    │  Claude AI   │  │
│  │  Public API  │───▶│  Analysis    │───▶│  Analysis    │  │
│  │  (free, no   │    │  Engine      │    │  Engine      │  │
│  │   key needed)│    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│       │                                                 │
│                                                             │
│                                                ┌────────────▼─────────┐ │
│                                                │  Live Terminal UI    │ │
│                                                │  (Rich dashboard)    │ │
│                                                └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Analysis Cycle (every 5 minutes)

```
Every 5 min:
  1. Fetch 60 hourly candles per coin from Binance
  2. Compute: RSI-14, MACD(12,26,9), Bollinger(20), EMA-20/50, ATR-14
  3. Send all data + portfolio state to Claude
  4. Claude returns JSON signal per coin:
       { "action": "BUY", "confidence": 8, "reasoning": "...", "price_target": 84000 }
  5. If confidence >= 6:  execute paper trade
  6. Update terminal dashboard
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- An Anthropic API key (free to get)
- Internet connection (for live market data)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/d3-d34d/claude-trading-bot.git
cd claude-trading-bot
```

---

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Get Your Anthropic API Key

1. Visit [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
2. Click **"Create Key"** → give it a name like `trading-bot`
3. Copy the key — it starts with: `sk-ant...`

---

### Step 4 — Configure Your API Key

```bash
# Mac / Linux
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Windows Command Prompt
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Or create a .env file:**
```bash
cp .env.example .env
# Open .env and paste your key
```

---

### Step 5 — Run the Bot

```bash
# Mac / Linux
./run.sh

# Windows
run.bat

# Or directly
python trading_bot.py
```

Press **Ctrl+C** to stop and see your session summary.

---

## ♧�️ Disclaimer

This is a **paper trading bot** -- no real money is ever at risk. For educational purposes only. Crypto markets are highly volatile -- always DYOR.
