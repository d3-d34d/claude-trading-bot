#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         CLAUDE-POWERED CRYPTO TRADING BOT  (Paper Mode)         ║
║  Real-time market data + AI analysis + simulated trading         ║
╚══════════════════════════════════════════════════════════════════╝

HOW IT WORKS:
  1. Fetches live OHLCV data from Binance public API (no key needed)
  2. Computes RSI, MACD, Bollinger Bands, EMA indicators
  3. Sends all market context to Claude every 5 minutes for analysis
  4. Claude returns BUY / SELL / HOLD signals with reasoning
  5. Bot executes paper trades and tracks your portfolio P&L
  6. Everything is displayed in a live, auto-refreshing terminal UI

USAGE:
  export ANTHROPIC_API_KEY="your-key-here"
  python trading_bot.py

  Optional flags:
    --pairs   BTC ETH SOL ADA      # Which coins to trade (default: BTC ETH SOL)
    --balance 50000                 # Starting paper balance in USDT (default: 10000)
    --interval 300                  # Claude analysis interval in seconds (default: 300)
    --risk 0.15                     # Fraction of balance per trade (default: 0.10)
"""

import os
import sys
import time
import json
import argparse
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional
from collections import deque, defaultdict

try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.console import Console
    from rich import box
    from rich.rule import Rule
    from rich.align import Align
    from rich.columns import Columns
    from rich.progress_bar import ProgressBar
except ImportError:
    print("Missing dependency: pip install rich")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Missing dependency: pip install anthropic")
    sys.exit(1)


# ─────────────────────────────────────────────
# CONFIG (overridden by CLI args)
# ─────────────────────────────────────────────
DEFAULT_PAIRS     = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_BALANCE   = 10_000.0
DEFAULT_INTERVAL  = 300        # seconds between Claude analyses
DEFAULT_RISK      = 0.10       # 10% of USDT balance per BUY trade
MIN_CONFIDENCE    = 6          # Claude must score >= this to trigger a trade
LOG_LINES         = 10         # number of log messages to keep visible
REFRESH_SECS      = 8          # how often to refresh prices (seconds)
KLINE_LIMIT       = 60         # candles fetched per request (1-hour candles)


# ─────────────────────────────────────────────
# BINANCE PUBLIC DATA FETCHER
# ─────────────────────────────────────────────
class BinanceFetcher:
    BASE = "https://api.binance.com/api/v3"
    HEADERS = {"User-Agent": "claude-trading-bot/2.0"}

    def _get(self, endpoint: str, params: dict):
        r = requests.get(f"{self.BASE}/{endpoint}", params=params,
                         headers=self.HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()

    def price(self, symbol: str) -> float:
        return float(self._get("ticker/price", {"symbol": symbol})["price"])

    def stats_24h(self, symbol: str) -> dict:
        return self._get("ticker/24hr", {"symbol": symbol})

    def klines(self, symbol: str, interval: str = "1h", limit: int = KLINE_LIMIT) -> pd.DataFrame:
        raw = self._get("klines", {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(raw, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "close_ts", "quote_vol", "trades", "tb_base", "tb_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df


# ─────────────────────────────────────────────
# TECHNICAL ANALYSIS
# ─────────────────────────────────────────────
class TA:
    @staticmethod
    def rsi(s: pd.Series, period: int = 14) -> float:
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return float((100 - 100 / (1 + rs)).iloc[-1])

    @staticmethod
    def macd(s: pd.Series):
        e12 = s.ewm(span=12, adjust=False).mean()
        e26 = s.ewm(span=26, adjust=False).mean()
        m = e12 - e26
        sig = m.ewm(span=9, adjust=False).mean()
        return float(m.iloc[-1]), float(sig.iloc[-1]), float((m - sig).iloc[-1])

    @staticmethod
    def bollinger(s: pd.Series, period: int = 20):
        sma = s.rolling(period).mean()
        std = s.rolling(period).std()
        return float((sma + 2 * std).iloc[-1]), float(sma.iloc[-1]), float((sma - 2 * std).iloc[-1])

    @staticmethod
    def ema(s: pd.Series, span: int) -> float:
        return float(s.ewm(span=span, adjust=False).mean().iloc[-1])

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> float:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])


# ─────────────────────────────────────────────
# PAPER TRADING ENGINE  (enhanced with stats)
# ─────────────────────────────────────────────
class PaperEngine:
    def __init__(self, initial_usdt: float):
        self.usdt            = initial_usdt
        self.initial         = initial_usdt
        self.holdings: dict[str, float]  = {}
        self.trades: list[dict]          = []
        self.open_positions: dict[str, float] = {}   # symbol -> entry price
        self.start_time      = datetime.now()
        self.start_date      = date.today()

        # Per-day tracking
        self.daily_pnl: dict[str, float] = defaultdict(float)   # "YYYY-MM-DD" -> pnl
        self.daily_start_value: dict[str, float] = {}           # "YYYY-MM-DD" -> value at start of day

        # Trade statistics
        self.wins            = 0
        self.losses          = 0
        self.best_trade_pnl  = 0.0    # best single SELL pnl
        self.worst_trade_pnl = 0.0    # worst single SELL pnl
        self.total_realized  = 0.0    # sum of all realized P&L

    # ── trade execution ──────────────────────
    def buy(self, symbol: str, price: float, usdt_amount: float, reason: str) -> Optional[dict]:
        usdt_amount = min(usdt_amount, self.usdt * 0.99)
        if usdt_amount < 5 or price <= 0:
            return None
        qty = usdt_amount / price
        self.usdt -= usdt_amount
        self.holdings[symbol] = self.holdings.get(symbol, 0) + qty
        self.open_positions[symbol] = price
        t = dict(
            time=datetime.now().strftime("%H:%M:%S"),
            date=date.today().isoformat(),
            side="BUY", symbol=symbol,
            price=price, qty=qty, usdt=usdt_amount,
            pnl=None, reason=reason[:80]
        )
        self.trades.append(t)
        return t

    def sell(self, symbol: str, price: float, reason: str) -> Optional[dict]:
        qty = self.holdings.get(symbol, 0)
        if qty <= 0 or price <= 0:
            return None
        usdt = qty * price
        self.usdt += usdt
        entry = self.open_positions.pop(symbol, price)
        pnl = (price - entry) * qty
        self.holdings[symbol] = 0

        # Update stats
        self.total_realized += pnl
        today_str = date.today().isoformat()
        self.daily_pnl[today_str] += pnl
        if pnl >= 0:
            self.wins += 1
            self.best_trade_pnl = max(self.best_trade_pnl, pnl)
        else:
            self.losses += 1
            self.worst_trade_pnl = min(self.worst_trade_pnl, pnl)

        t = dict(
            time=datetime.now().strftime("%H:%M:%S"),
            date=today_str,
            side="SELL", symbol=symbol,
            price=price, qty=qty, usdt=usdt,
            pnl=pnl, reason=reason[:80]
        )
        self.trades.append(t)
        return t

    # ── portfolio metrics ─────────────────────
    def total_value(self, prices: dict) -> float:
        return self.usdt + sum(
            qty * prices.get(sym, 0)
            for sym, qty in self.holdings.items()
        )

    def pnl(self, prices: dict) -> float:
        return self.total_value(prices) - self.initial

    def pnl_pct(self, prices: dict) -> float:
        return self.pnl(prices) / self.initial * 100

    def days_running(self) -> int:
        return max(1, (date.today() - self.start_date).days + 1)

    def today_pnl(self) -> float:
        return self.daily_pnl.get(date.today().isoformat(), 0.0)

    def win_rate(self) -> float:
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0

    def avg_trade_pnl(self) -> float:
        total = self.wins + self.losses
        return (self.total_realized / total) if total > 0 else 0.0

    def uptime_str(self) -> str:
        elapsed = datetime.now() - self.start_time
        h = int(elapsed.total_seconds() // 3600)
        m = int((elapsed.total_seconds() % 3600) // 60)
        s = int(elapsed.total_seconds() % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


# ─────────────────────────────────────────────
# CLAUDE AI MARKET ANALYST
# ─────────────────────────────────────────────
class ClaudeAnalyst:
    SYSTEM = """You are an expert algorithmic crypto trading analyst.
You receive structured market data (price, volume, RSI, MACD, Bollinger Bands, EMA)
and must respond with a precise JSON trading signal.
Be data-driven, concise, and decisive. Never hedge or be vague."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model

    def analyze(self, market_data: dict, portfolio: dict) -> dict:
        pairs = list(market_data.keys())
        pair_schema = {p: {
            "action": "BUY | SELL | HOLD",
            "confidence": "1-10 integer",
            "reasoning": "1-2 sentence data-driven justification",
            "price_target": "your predicted price in the next 1-4 hours (number)"
        } for p in pairs}
        pair_schema["market_summary"] = "2-3 sentence overall market read"

        prompt = f"""Analyze this live crypto market snapshot and return a trading signal.

## Portfolio State
{json.dumps(portfolio, indent=2)}

## Market Data
{json.dumps(market_data, indent=2)}

## Signal Rules
- Only recommend BUY if RSI < 65, price is near or below lower Bollinger Band,
  MACD histogram is positive or turning positive, and confidence >= 6.
- Only recommend SELL if RSI > 70, price above upper Bollinger Band,
  or strong downside momentum detected.
- Default to HOLD when signals are mixed or confidence < 6.
- Factor in 24h volume and trend direction.

## Required JSON Response (strict, no extra text):
{json.dumps(pair_schema, indent=2)}"""

        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=1200,
                system=self.SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "market_summary": "Analysis failed – JSON error"}
        except Exception as e:
            return {"error": str(e), "market_summary": f"Analysis failed: {e}"}


# ─────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────
def pnl_color(val: float) -> str:
    if val > 0:   return "bright_green"
    if val < 0:   return "bright_red"
    return "white"

def pnl_str(val: float, prefix: str = "$") -> str:
    return f"{prefix}{val:+,.2f}"

def pnl_text(val: float, prefix: str = "$") -> Text:
    return Text(pnl_str(val, prefix), style=f"bold {pnl_color(val)}")

def colored_pct(val: float) -> Text:
    return Text(f"{val:+.2f}%", style=pnl_color(val))


# ─────────────────────────────────────────────
# MAIN TRADING BOT
# ─────────────────────────────────────────────
class TradingBot:
    def __init__(self, pairs: list[str], balance: float, interval: int,
                 risk: float, api_key: str):
        self.pairs      = pairs
        self.interval   = interval
        self.risk       = risk

        self.fetcher    = BinanceFetcher()
        self.engine     = PaperEngine(balance)
        self.analyst    = ClaudeAnalyst(api_key)
        self.console    = Console()

        self.prices:     dict[str, float] = {}
        self.indicators: dict[str, dict]  = {}
        self.signals:    dict[str, dict]  = {}
        self.market_summary: str          = "Waiting for first Claude analysis…"
        self.status:     str              = "Starting up…"
        self.log:        deque            = deque(maxlen=LOG_LINES)
        self.next_analysis_at: float      = time.time()
        self.analysis_count:  int         = 0
        self.lock        = threading.Lock()
        self._running    = False

    # ── logging ───────────────────────────────
    def _log(self, msg: str, style: str = "white"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.appendleft(f"[dim]{ts}[/dim]  [{style}]{msg}[/{style}]")

    # ── market data ───────────────────────────
    def _refresh_prices(self):
        for sym in self.pairs:
            try:
                df    = self.fetcher.klines(sym)
                stats = self.fetcher.stats_24h(sym)
                price = float(stats["lastPrice"])

                rsi              = TA.rsi(df["close"])
                macd, sig, hist  = TA.macd(df["close"])
                bb_up, bb_mid, bb_lo = TA.bollinger(df["close"])
                ema20            = TA.ema(df["close"], 20)
                ema50            = TA.ema(df["close"], 50)
                atr_val          = TA.atr(df)

                with self.lock:
                    self.prices[sym] = price
                    self.indicators[sym] = dict(
                        price        = price,
                        change_24h   = float(stats["priceChangePercent"]),
                        volume_24h   = float(stats["volume"]),
                        high_24h     = float(stats["highPrice"]),
                        low_24h      = float(stats["lowPrice"]),
                        rsi          = round(rsi, 2),
                        macd         = round(macd, 6),
                        macd_signal  = round(sig, 6),
                        macd_hist    = round(hist, 6),
                        bb_upper     = round(bb_up, 4),
                        bb_mid       = round(bb_mid, 4),
                        bb_lower     = round(bb_lo, 4),
                        ema20        = round(ema20, 4),
                        ema50        = round(ema50, 4),
                        atr          = round(atr_val, 4),
                        vs_ema20_pct = round((price - ema20) / ema20 * 100, 3),
                    )
            except Exception as e:
                self._log(f"⚠ Data fetch error [{sym}]: {e}", "yellow")

    # ── Claude analysis ───────────────────────
    def _run_analysis(self):
        self.status = "🤖  Claude is thinking…"
        self._log("Requesting Claude market analysis…", "cyan")
        try:
            with self.lock:
                mkt_snapshot = dict(self.indicators)
                portfolio_state = dict(
                    usdt_balance   = round(self.engine.usdt, 2),
                    holdings       = {s: round(q, 8) for s, q in self.engine.holdings.items() if q > 0},
                    open_positions = {s: round(p, 4) for s, p in self.engine.open_positions.items()},
                )

            result = self.analyst.analyze(mkt_snapshot, portfolio_state)

            with self.lock:
                if "error" not in result:
                    self.analysis_count += 1
                    self.market_summary = result.get("market_summary", "")
                    for sym in self.pairs:
                        if sym in result:
                            self.signals[sym] = result[sym]

                    for sym in self.pairs:
                        sig        = self.signals.get(sym, {})
                        action     = sig.get("action", "HOLD").upper()
                        confidence = int(sig.get("confidence", 0))
                        reason     = sig.get("reasoning", "")

                        if action == "BUY" and confidence >= MIN_CONFIDENCE:
                            usdt_to_spend = self.engine.usdt * self.risk
                            t = self.engine.buy(sym, self.prices.get(sym, 0), usdt_to_spend, reason)
                            if t:
                                self._log(f"✅ BUY  {sym}  ${t['price']:,.2f}  (conf={confidence})", "bright_green")
                        elif action == "SELL" and confidence >= MIN_CONFIDENCE:
                            t = self.engine.sell(sym, self.prices.get(sym, 0), reason)
                            if t:
                                pnl = t.get("pnl", 0)
                                col = "bright_green" if pnl >= 0 else "bright_red"
                                emoji = "💰" if pnl >= 0 else "📉"
                                self._log(f"{emoji} SELL {sym}  ${t['price']:,.2f}  P&L: ${pnl:+.2f}", col)
                        else:
                            self._log(f"⏸  HOLD {sym}  (conf={confidence})", "dim")

                    self.status = f"Last analysis: {datetime.now().strftime('%H:%M:%S')}  (#{self.analysis_count})"
                else:
                    self._log(f"Claude error: {result.get('error')}", "yellow")
                    self.status = "⚠ Analysis error – retrying next cycle"

        except Exception as e:
            self._log(f"Analysis exception: {e}", "red")
            self.status = f"Error: {e}"

    # ── dashboard ─────────────────────────────
    def _build_layout(self) -> Layout:
        with self.lock:
            prices      = dict(self.prices)
            indicators  = dict(self.indicators)
            signals     = dict(self.signals)
            trades      = list(self.engine.trades)
            holdings    = dict(self.engine.holdings)
            usdt_bal    = self.engine.usdt
            total_val   = self.engine.total_value(prices)
            pnl         = self.engine.pnl(prices)
            pnl_pct     = self.engine.pnl_pct(prices)
            today_pnl_v = self.engine.today_pnl()
            wins        = self.engine.wins
            losses      = self.engine.losses
            win_rate    = self.engine.win_rate()
            best_trade  = self.engine.best_trade_pnl
            worst_trade = self.engine.worst_trade_pnl
            avg_pnl     = self.engine.avg_trade_pnl()
            days        = self.engine.days_running()
            uptime      = self.engine.uptime_str()
            log_lines   = list(self.log)
            summary     = self.market_summary
            status_msg  = self.status
            next_at     = self.next_analysis_at

        secs_left = max(0, int(next_at - time.time()))

        # ── HEADER ────────────────────────────────────────────────
        now_str   = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        pnl_col   = pnl_color(pnl)
        today_col = pnl_color(today_pnl_v)

        hdr = Text(justify="center")
        hdr.append("⚡ CLAUDE TRADING BOT ", style="bold bright_cyan")
        hdr.append("│ PAPER MODE │ ", style="bold yellow")
        hdr.append(f"{now_str}  │  ", style="dim white")
        hdr.append(f"💼 ${total_val:,.2f}  ", style="bold white")
        hdr.append("│  Total P&L: ", style="white")
        hdr.append(f"${pnl:+,.2f} ({pnl_pct:+.2f}%)", style=f"bold {pnl_col}")
        hdr.append("  │  Today: ", style="white")
        hdr.append(f"${today_pnl_v:+,.2f}", style=f"bold {today_col}")

        # ── PRICES & SIGNALS ──────────────────────────────────────
        pt = Table(box=box.SIMPLE_HEAD, header_style="bold bright_cyan", expand=True)
        pt.add_column("Pair",        style="cyan",    width=12)
        pt.add_column("Price",       justify="right", width=14)
        pt.add_column("24h %",       justify="right", width=8)
        pt.add_column("Action",      justify="center",width=7)
        pt.add_column("Conf",        justify="center",width=5)
        pt.add_column("AI Target",   justify="right", width=12)

        for sym in self.pairs:
            ind  = indicators.get(sym, {})
            sig  = signals.get(sym, {})
            p    = prices.get(sym, 0)
            chg  = ind.get("change_24h", 0)
            act  = sig.get("action", "—").upper()
            conf = sig.get("confidence", 0)
            tgt  = sig.get("price_target", "—")

            chg_sty = pnl_color(chg)
            act_sty = {"BUY": "bold bright_green", "SELL": "bold bright_red",
                       "HOLD": "bold yellow"}.get(act, "dim white")
            tgt_col = pnl_color(float(tgt) - p if isinstance(tgt, (int, float)) and p else 0)

            pt.add_row(
                sym.replace("USDT", "/USDT"),
                f"${p:,.4f}" if p < 1 else f"${p:,.2f}",
                Text(f"{chg:+.2f}%", style=chg_sty),
                Text(act, style=act_sty),
                str(conf) if conf else "–",
                Text(f"${float(tgt):,.2f}" if isinstance(tgt, (int,float)) else str(tgt), style=tgt_col),
            )

        # ── INDICATORS ────────────────────────────────────────────
        it = Table(box=box.SIMPLE_HEAD, header_style="bold magenta", expand=True)
        it.add_column("Pair",      style="cyan",   width=12)
        it.add_column("RSI",       justify="right",width=7)
        it.add_column("MACD Hist", justify="right",width=11)
        it.add_column("BB %",      justify="right",width=8)
        it.add_column("vs EMA20",  justify="right",width=9)
        it.add_column("ATR",       justify="right",width=10)

        for sym in self.pairs:
            ind   = indicators.get(sym, {})
            rsi   = ind.get("rsi", 0)
            hist  = ind.get("macd_hist", 0)
            p     = ind.get("price", 0)
            bb_up = ind.get("bb_upper", 0)
            bb_lo = ind.get("bb_lower", 0)
            vs20  = ind.get("vs_ema20_pct", 0)
            atr   = ind.get("atr", 0)

            bb_width = bb_up - bb_lo
            bb_pos   = ((p - bb_lo) / bb_width * 100) if bb_width else 0

            rsi_sty  = ("bright_green" if rsi < 40 else "bright_red" if rsi > 70 else "white")
            hist_sty = pnl_color(hist)
            vs_sty   = pnl_color(vs20)

            it.add_row(
                sym.replace("USDT", "/USDT"),
                Text(f"{rsi:.1f}", style=rsi_sty),
                Text(f"{hist:+.5f}", style=hist_sty),
                f"{bb_pos:.1f}%",
                Text(f"{vs20:+.2f}%", style=vs_sty),
                f"{atr:.4f}",
            )

        # ── PORTFOLIO ─────────────────────────────────────────────
        ptf = Table(box=box.SIMPLE_HEAD, header_style="bold bright_green", expand=True)
        ptf.add_column("Asset",   style="cyan",   width=10)
        ptf.add_column("Qty",     justify="right",width=14)
        ptf.add_column("Value",   justify="right",width=14)
        ptf.add_column("Entry",   justify="right",width=12)
        ptf.add_column("Unreal. P&L", justify="right", width=12)

        ptf.add_row("USDT", "—",
                    Text(f"${usdt_bal:,.2f}", style="bold white"),
                    "—", "—")

        for sym, qty in holdings.items():
            if qty > 0.0000001:
                p     = prices.get(sym, 0)
                val   = qty * p
                entry = self.engine.open_positions.get(sym, 0)
                upnl  = (p - entry) * qty if entry else 0
                ptf.add_row(
                    sym.replace("USDT", ""),
                    f"{qty:.6f}",
                    f"${val:,.2f}",
                    f"${entry:,.2f}",
                    pnl_text(upnl),
                )

        # ── TRADE HISTORY  (with colored P&L) ───────────────────
        tlt = Table(box=box.SIMPLE_HEAD, header_style="bold yellow", expand=True)
        tlt.add_column("Time",   width=8)
        tlt.add_column("Day",    width=5, justify="center")
        tlt.add_column("Side",   width=5)
        tlt.add_column("Sym",    width=5)
        tlt.add_column("Price",  justify="right", width=11)
        tlt.add_column("USDT",   justify="right", width=10)
        tlt.add_column("P&L",    justify="right", width=10)
        tlt.add_column("Result", width=6, justify="center")

        start_d = self.engine.start_date
        for t in reversed(trades[-12:]):
            side_sty = "bold bright_green" if t["side"] == "BUY" else "bold bright_red"
            trade_date = date.fromisoformat(t["date"]) if "date" in t else start_d
            day_num    = (trade_date - start_d).days + 1
            raw_pnl    = t.get("pnl")

            if raw_pnl is not None:
                pnl_cell   = pnl_text(raw_pnl)
                result_txt = Text("WIN ✅" if raw_pnl >= 0 else "LOSS ❌",
                                  style="bright_green" if raw_pnl >= 0 else "bright_red")
            else:
                pnl_cell   = Text("—", style="dim")
                result_txt = Text("OPEN", style="yellow")

            tlt.add_row(
                t["time"],
                f"D{day_num}",
                Text(t["side"], style=side_sty),
                t["symbol"].replace("USDT", ""),
                f"${t['price']:,.2f}",
                f"${t['usdt']:,.2f}",
                pnl_cell,
                result_txt,
            )

        # ── STATS PANEL ───────────────────────────────────────────
        st = Table(box=box.SIMPLE_HEAD, header_style="bold bright_white",
                   expand=True, show_header=False)
        st.add_column("Label", style="dim",   width=18)
        st.add_column("Value", justify="right", width=14)

        total_trades = wins + losses
        st.add_row("🗓  Day",         Text(f"Day {days}  |  {uptime} uptime", style="bold white"))
        st.add_row("💵  Balance",     Text(f"${usdt_bal:,.2f} USDT", style="bold white"))
        st.add_row("📈  Total P&L",   pnl_text(pnl))
        st.add_row("📅  Today P&L",   pnl_text(today_pnl_v))
        st.add_row("🏆  Win Rate",
                   Text(f"{win_rate:.1f}%  ({wins}W / {losses}L)",
                        style="bright_green" if win_rate >= 50 else "bright_red"))
        st.add_row("🔢  Trades",      Text(f"{total_trades} total", style="white"))
        st.add_row("💰  Best Trade",  pnl_text(best_trade))
        st.add_row("📉  Worst Trade", pnl_text(worst_trade))
        st.add_row("📊  Avg Trade",   pnl_text(avg_pnl))
        st.add_row("🤖  Analyses",    Text(f"#{self.analysis_count}", style="cyan"))

        # ── DAILY P&L MINI-CHART ─────────────────────────────────
        day_rows = sorted(self.engine.daily_pnl.items())[-7:]  # last 7 days
        daily_panel_rows = []
        for d_str, d_pnl in day_rows:
            bar_len  = min(20, int(abs(d_pnl) / max(1, abs(pnl)) * 20))
            bar_char = "█" * bar_len
            col      = "bright_green" if d_pnl >= 0 else "bright_red"
            daily_panel_rows.append(
                f"[dim]{d_str}[/dim]  [{col}]{bar_char}[/{col}]  [{col}]${d_pnl:+,.2f}[/{col}]"
            )
        daily_str = "\n".join(daily_panel_rows) if daily_panel_rows else "[dim]No closed trades yet[/dim]"

        # ── ACTIVITY LOG ─────────────────────────────────────────
        ai_log = "\n".join(log_lines) if log_lines else "[dim]No events yet[/dim]"

        # ── ASSEMBLE LAYOUT ───────────────────────────────────────
        layout = Layout()
        layout.split_column(
            Layout(name="header",  size=3),
            Layout(name="body"),
            Layout(name="bottom",  size=17),
            Layout(name="footer",  size=3),
        )
        layout["body"].split_row(
            Layout(name="left",  ratio=3),
            Layout(name="right", ratio=2),
        )
        layout["left"].split_column(
            Layout(name="prices",     ratio=2),
            Layout(name="indicators", ratio=2),
        )
        layout["right"].split_column(
            Layout(name="portfolio",  ratio=2),
            Layout(name="stats",      ratio=3),
        )
        layout["bottom"].split_row(
            Layout(name="tradelog", ratio=3),
            Layout(name="daily",    ratio=2),
        )

        layout["header"].update(Panel(hdr, style="bold"))
        layout["prices"].update(Panel(pt,  title="[bold bright_cyan]📈 Live Prices & AI Signals[/]"))
        layout["indicators"].update(Panel(it, title="[bold magenta]📊 Technical Indicators[/]"))
        layout["portfolio"].update(Panel(ptf, title="[bold bright_green]💼 Portfolio[/]"))
        layout["stats"].update(Panel(st,  title="[bold bright_white]📊 Performance Stats[/]"))
        layout["tradelog"].update(Panel(tlt, title="[bold yellow]📋 Trade History  (WIN=green · LOSS=red · OPEN=yellow)[/]"))
        layout["daily"].update(Panel(daily_str, title="[bold cyan]📅 Daily P&L History[/]"))
        layout["footer"].update(Panel(
            Text.from_markup(
                f"[bold cyan]🤖 Claude:[/bold cyan]  {summary}\n"
                f"[dim]{status_msg}  |  Next analysis in {secs_left}s[/dim]"
            )
        ))
        return layout

    # ── background threads ────────────────────
    def _price_loop(self):
        while self._running:
            self._refresh_prices()
            time.sleep(REFRESH_SECS)

    def _analysis_loop(self):
        while self._running:
            if time.time() >= self.next_analysis_at and self.prices:
                self._run_analysis()
                self.next_analysis_at = time.time() + self.interval
            else:
                time.sleep(1)

    # ── entry point ───────────────────────────
    def run(self):
        self._running = True
        self._log("Bot started. Fetching initial market data…", "cyan")

        t1 = threading.Thread(target=self._price_loop,    daemon=True)
        t2 = threading.Thread(target=self._analysis_loop, daemon=True)
        t1.start(); t2.start()

        self.console.print("[cyan]Loading market data…[/cyan]")
        for _ in range(30):
            if len(self.prices) == len(self.pairs):
                break
            time.sleep(1)

        try:
            with Live(self._build_layout(), refresh_per_second=1,
                      screen=True, redirect_stderr=False) as live:
                while True:
                    live.update(self._build_layout())
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            p = self.engine.pnl(self.prices)
            self.console.print(f"\n[yellow]Bot stopped. Session summary:[/yellow]")
            self.console.print(f"  Days running : {self.engine.days_running()}")
            self.console.print(f"  Uptime       : {self.engine.uptime_str()}")
            self.console.print(f"  Total trades : {self.engine.wins + self.engine.losses}")
            self.console.print(f"  Win rate     : {self.engine.win_rate():.1f}%")
            self.console.print(f"  Final P&L    : [{'bright_green' if p >= 0 else 'bright_red'}]${p:+,.2f}[/]")
            self.console.print(f"  Best trade   : [bright_green]${self.engine.best_trade_pnl:+,.2f}[/bright_green]")
            self.console.print(f"  Worst trade  : [bright_red]${self.engine.worst_trade_pnl:+,.2f}[/bright_red]")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Claude-Powered Crypto Paper Trading Bot")
    parser.add_argument("--pairs",    nargs="+", default=None,
                        help="Coin symbols e.g. BTC ETH SOL (USDT pairs)")
    parser.add_argument("--balance",  type=float, default=DEFAULT_BALANCE)
    parser.add_argument("--interval", type=int,   default=DEFAULT_INTERVAL)
    parser.add_argument("--risk",     type=float, default=DEFAULT_RISK)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n❌  ANTHROPIC_API_KEY is not set.")
        print("    Get your key: https://console.anthropic.com/settings/keys")
        print("    Then run:    export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    if args.pairs:
        pairs = [p.upper() + ("USDT" if not p.upper().endswith("USDT") else "")
                 for p in args.pairs]
    else:
        pairs = DEFAULT_PAIRS

    console = Console()
    console.print(f"\n[bold bright_cyan]⚡ Claude-Powered Crypto Trading Bot v2[/bold bright_cyan]")
    console.print(f"   Pairs    : [cyan]{', '.join(pairs)}[/cyan]")
    console.print(f"   Balance  : [green]${args.balance:,.2f} USDT (paper)[/green]")
    console.print(f"   Interval : [yellow]{args.interval}s between AI analyses[/yellow]")
    console.print(f"   Risk     : [yellow]{args.risk*100:.0f}% per trade[/yellow]")
    console.print(f"\n[dim]Press Ctrl+C to stop and see your session summary.[/dim]\n")
    time.sleep(1.5)

    bot = TradingBot(pairs=pairs, balance=args.balance,
                     interval=args.interval, risk=args.risk, api_key=api_key)
    bot.run()


if __name__ == "__main__":
    main()
