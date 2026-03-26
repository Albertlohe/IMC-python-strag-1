"""
IMC Prosperity Trader v2 — RSI + MACD + Bollinger Bands + EMA Z-score + Fear & Greed
Author: Albertt | IIT Kharagpur — M.Tech Safety Engineering & Analytics

Indicators:
  - RSI (14-period)        : Momentum filter        | <30 BUY | >70 SELL
  - MACD (12/26/9)         : Trend confirmation      | histogram sign → direction
  - Bollinger Bands (20/2σ): Volatility breakout     | lower band BUY | upper band SELL
  - EMA Z-score            : Mean reversion          | Z < -1.5 BUY | Z > +1.5 SELL
  - Fear & Greed proxy     : Position sizing         | wide spread → cautious | tight → aggressive

Signal Confluence: each indicator votes +1 (bullish) / -1 (bearish).
Trade only when 2+ indicators agree — dramatically reduces false signals.

Usage:
  python imc_trader_v2.py            # run backtest + dashboard
  python imc_trader_v2.py --export   # also export compact trader.py for IMC upload
"""

# ── Standard lib / third-party ───────────────────────────────────────────────
import sys
import types
import json
import math
import random
import copy
import argparse
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATAMODEL STUB  (replaces the real `datamodel` module from IMC)
# ═════════════════════════════════════════════════════════════════════════════
dm = types.ModuleType("datamodel")


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __repr__(self):
        side = "BUY " if self.quantity > 0 else "SELL"
        return f"{side} {abs(self.quantity):3d} {self.symbol:<18} @ {self.price}"


class TradingState:
    def __init__(self, order_depths, position, traderData="", timestamp=0):
        self.order_depths = order_depths
        self.position = position
        self.own_trades = {}
        self.traderData = traderData
        self.timestamp = timestamp


dm.OrderDepth = OrderDepth
dm.Order = Order
dm.TradingState = TradingState
sys.modules["datamodel"] = dm

# ═════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Position limits
POSITION_LIMITS = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}

# Market making — Rainforest Resin only
MM_FAIR_VALUE  = {"RAINFOREST_RESIN": 10_000}
MM_BASE_SPREAD = 2
MM_ORDER_SIZE  = 10

# EMA / Z-score
EMA_SHORT = 5
EMA_LONG  = 20
Z_ENTRY   = 1.5

# RSI
RSI_PERIOD  = 14
RSI_OVERBUY = 70
RSI_OVERSELL = 30

# MACD
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD    = 2.0

# Confluence threshold: how many indicators must agree to trade
CONFLUENCE_MIN = 2

# Order sizing
BASE_ORDER_SIZE = 8
FEAR_SPREAD_PCT = 0.003
GREED_SIZE_MULT = 1.5

# ═════════════════════════════════════════════════════════════════════════════
# 3. INDICATOR FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def ema_update(prev_ema: float, new_price: float, window: int) -> float:
    """Single-step EMA update. α = 2 / (window + 1)."""
    alpha = 2.0 / (window + 1)
    return alpha * new_price + (1 - alpha) * prev_ema


def compute_ema_series(prices: list, window: int) -> list:
    """Full EMA series from a price list (used for visualization)."""
    if not prices:
        return []
    ema = [prices[0]]
    alpha = 2.0 / (window + 1)
    for p in prices[1:]:
        ema.append(alpha * p + (1 - alpha) * ema[-1])
    return ema


def compute_rsi(prices: list, period: int = 14) -> float:
    """
    Relative Strength Index (Wilder smoothing).
    Returns 0-100.  >70 overbought (SELL), <30 oversold (BUY).
    Returns 50 (neutral) if not enough data.
    """
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains  = [max(d, 0)       for d in deltas]
    losses = [abs(min(d, 0))  for d in deltas]
    avg_gain = sum(gains[:period])  / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i])  / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rsi_series(prices: list, period: int = 14) -> list:
    """RSI value at every tick (for visualization)."""
    result = [50.0] * len(prices)
    for i in range(period + 1, len(prices) + 1):
        result[i - 1] = compute_rsi(prices[:i], period)
    return result


def compute_macd(prices: list, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD = EMA(fast) - EMA(slow).
    Signal = EMA(MACD, signal_period).
    Returns (macd_val, signal_val, histogram_val).
    Bullish: histogram > 0 | Bearish: histogram < 0.
    """
    if len(prices) < slow:
        return 0.0, 0.0, 0.0
    ema_fast = compute_ema_series(prices, fast)
    ema_slow = compute_ema_series(prices, slow)
    macd_line   = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = compute_ema_series(macd_line, signal)
    histogram   = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line[-1], signal_line[-1], histogram[-1]


def macd_series(prices: list, fast=12, slow=26, signal=9):
    """Full MACD / signal / histogram series (for visualization)."""
    ema_f = compute_ema_series(prices, fast)
    ema_s = compute_ema_series(prices, slow)
    macd  = [f - s for f, s in zip(ema_f, ema_s)]
    sig   = compute_ema_series(macd, signal)
    hist  = [m - s for m, s in zip(macd, sig)]
    return macd, sig, hist


def compute_bollinger(prices: list, period: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands: middle = SMA(period), upper/lower = middle ± num_std * σ.
    Returns (upper, middle, lower).
    Price > upper → overbought (SELL) | Price < lower → oversold (BUY).
    """
    if len(prices) < period:
        mid = prices[-1]
        return mid, mid, mid
    window = prices[-period:]
    mid    = sum(window) / period
    std    = math.sqrt(sum((p - mid) ** 2 for p in window) / period)
    return mid + num_std * std, mid, mid - num_std * std


def bollinger_series(prices: list, period: int = 20, num_std: float = 2.0):
    """Full Bollinger series (for visualization)."""
    upper, middle, lower = [], [], []
    for i in range(1, len(prices) + 1):
        u, m, l = compute_bollinger(prices[:i], period, num_std)
        upper.append(u)
        middle.append(m)
        lower.append(l)
    return upper, middle, lower


def best_bid_ask(order_depth):
    best_bid = max(order_depth.buy_orders)  if order_depth.buy_orders  else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
    return best_bid, best_ask


def clamp_qty(product: str, desired_qty: int, current_pos: int) -> int:
    limit = POSITION_LIMITS.get(product, 20)
    if desired_qty > 0:
        return min(desired_qty,  limit - current_pos)
    else:
        return max(desired_qty, -limit - current_pos)


# ═════════════════════════════════════════════════════════════════════════════
# 4. TRADER CLASS
# ═════════════════════════════════════════════════════════════════════════════

from datamodel import OrderDepth, TradingState, Order  # noqa: E402 (stub already loaded)


class Trader:
    """
    Enhanced IMC Prosperity Trader v2.
    Combines: Market Making + EMA Z-score + RSI + MACD + Bollinger Bands.
    Uses signal confluence: only trades when CONFLUENCE_MIN indicators agree.
    """

    def run(self, state: TradingState) -> tuple:

        # ── Load memory ────────────────────────────────────────────────────
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}

        price_history = mem.get("price_history", {})
        ema_short     = mem.get("ema_short", {})
        ema_long      = mem.get("ema_long", {})
        macd_ema_fast = mem.get("macd_ema_fast", {})
        macd_ema_slow = mem.get("macd_ema_slow", {})
        macd_signal_d = mem.get("macd_signal", {})

        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            orders = []
            current_pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            best_bid, best_ask = best_bid_ask(order_depth)
            mid_price = (best_bid + best_ask) / 2.0

            # ── Update price history ────────────────────────────────────────
            hist = price_history.get(product, [])
            hist.append(mid_price)
            if len(hist) > 50:
                hist = hist[-50:]
            price_history[product] = hist

            # ── Incremental EMA updates ─────────────────────────────────────
            if product not in ema_short:
                ema_short[product] = mid_price
                ema_long[product]  = mid_price
            else:
                ema_short[product] = ema_update(ema_short[product], mid_price, EMA_SHORT)
                ema_long[product]  = ema_update(ema_long[product],  mid_price, EMA_LONG)

            # ── Incremental MACD ────────────────────────────────────────────
            if product not in macd_ema_fast:
                macd_ema_fast[product]  = mid_price
                macd_ema_slow[product]  = mid_price
                macd_signal_d[product]  = 0.0
            else:
                macd_ema_fast[product] = ema_update(macd_ema_fast[product], mid_price, MACD_FAST)
                macd_ema_slow[product] = ema_update(macd_ema_slow[product], mid_price, MACD_SLOW)
            macd_val = macd_ema_fast[product] - macd_ema_slow[product]
            macd_signal_d[product] = ema_update(macd_signal_d[product], macd_val, MACD_SIGNAL)
            macd_hist_val = macd_val - macd_signal_d[product]

            es = ema_short[product]
            el = ema_long[product]

            # ── Fear & Greed proxy ──────────────────────────────────────────
            spread_ratio = (best_ask - best_bid) / mid_price if mid_price > 0 else 0
            fear_mode    = spread_ratio > FEAR_SPREAD_PCT
            size_mult    = 1.0 if fear_mode else GREED_SIZE_MULT

            # ── Rolling std dev ─────────────────────────────────────────────
            std_dev = 0.0
            if len(hist) >= 5:
                w      = hist[-10:]
                mean_p = sum(w) / len(w)
                std_dev = math.sqrt(sum((p - mean_p) ** 2 for p in w) / len(w))

            # ════════════════════════════════════════════════════════════════
            # STRATEGY A — Market Making (Rainforest Resin only)
            # ════════════════════════════════════════════════════════════════
            if product in MM_FAIR_VALUE:
                fair_val  = MM_FAIR_VALUE[product]
                spread    = MM_BASE_SPREAD + (2 if fear_mode else 0)
                bid_price = fair_val - spread
                ask_price = fair_val + spread

                if best_ask <= bid_price:
                    qty = clamp_qty(product, int(MM_ORDER_SIZE * size_mult), current_pos)
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                if best_bid >= ask_price:
                    qty = clamp_qty(product, -int(MM_ORDER_SIZE * size_mult), current_pos)
                    if qty < 0:
                        orders.append(Order(product, best_bid, qty))

                buy_qty  = clamp_qty(product,  MM_ORDER_SIZE, current_pos)
                sell_qty = clamp_qty(product, -MM_ORDER_SIZE, current_pos)
                if buy_qty  > 0: orders.append(Order(product, bid_price,  buy_qty))
                if sell_qty < 0: orders.append(Order(product, ask_price, sell_qty))

            # ════════════════════════════════════════════════════════════════
            # STRATEGY B — Multi-Indicator Confluence (Kelp / Squid Ink)
            # ════════════════════════════════════════════════════════════════
            else:
                if len(hist) < max(RSI_PERIOD, BB_PERIOD, MACD_SLOW) + 2:
                    if orders:
                        result[product] = orders
                    continue

                rsi_val                  = compute_rsi(hist, RSI_PERIOD)
                bb_upper, bb_mid, bb_lower = compute_bollinger(hist, BB_PERIOD, BB_STD)
                z_score = (es - el) / std_dev if std_dev > 0 else 0

                # Each indicator votes: +1 BUY, -1 SELL, 0 neutral
                votes = []
                votes.append(1  if rsi_val < RSI_OVERSELL else -1 if rsi_val > RSI_OVERBUY else 0)
                votes.append(1  if macd_hist_val > 0        else -1 if macd_hist_val < 0     else 0)
                votes.append(1  if mid_price <= bb_lower     else -1 if mid_price >= bb_upper  else 0)
                votes.append(1  if z_score < -Z_ENTRY        else -1 if z_score > Z_ENTRY      else 0)

                bull_count = votes.count(1)
                bear_count = votes.count(-1)
                trade_size = int(BASE_ORDER_SIZE * size_mult)

                if bull_count >= CONFLUENCE_MIN:
                    qty = clamp_qty(product, trade_size, current_pos)
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))

                elif bear_count >= CONFLUENCE_MIN:
                    qty = clamp_qty(product, -trade_size, current_pos)
                    if qty < 0:
                        orders.append(Order(product, best_bid, qty))

                else:
                    # No consensus → passive market making around EMA mid
                    spread_half = max(1, int(std_dev * 0.5))
                    ema_mid  = (es + el) / 2
                    buy_qty  = clamp_qty(product,  BASE_ORDER_SIZE // 2, current_pos)
                    sell_qty = clamp_qty(product, -BASE_ORDER_SIZE // 2, current_pos)
                    if buy_qty  > 0: orders.append(Order(product, int(ema_mid - spread_half),  buy_qty))
                    if sell_qty < 0: orders.append(Order(product, int(ema_mid + spread_half), sell_qty))

            if orders:
                result[product] = orders

        # ── Save memory ────────────────────────────────────────────────────
        trader_data = json.dumps({
            "price_history": price_history,
            "ema_short":     ema_short,
            "ema_long":      ema_long,
            "macd_ema_fast": macd_ema_fast,
            "macd_ema_slow": macd_ema_slow,
            "macd_signal":   macd_signal_d,
        })
        return result, 0, trader_data


# ═════════════════════════════════════════════════════════════════════════════
# 5. BACKTESTER
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(n_ticks: int = 150, seed: int = 42):
    """Simulate the trader over n_ticks random-walk ticks. Returns logs."""
    random.seed(seed)
    PRODUCTS    = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]
    base_prices = {"RAINFOREST_RESIN": 10000, "KELP": 5000, "SQUID_INK": 2000}
    volatility  = {"RAINFOREST_RESIN": 3,     "KELP": 12,   "SQUID_INK": 20}

    trader      = Trader()
    trader_data = ""
    position    = {p: 0 for p in PRODUCTS}
    cash        = 0.0
    prices      = copy.deepcopy(base_prices)

    log         = {p: [] for p in PRODUCTS}
    log_ticks, log_pnl, log_cash = [], [], []
    log_pos     = {p: [] for p in PRODUCTS}
    log_buys    = {p: [] for p in PRODUCTS}
    log_sells   = {p: [] for p in PRODUCTS}
    log_buy_px  = {p: [] for p in PRODUCTS}
    log_sell_px = {p: [] for p in PRODUCTS}

    for tick in range(n_ticks):
        for p in PRODUCTS:
            trend     = random.choice([-1, -1, 0, 0, 0, 1, 1]) * volatility[p] * 0.3
            prices[p] += random.gauss(trend, volatility[p])
            prices[p]  = max(prices[p], 100)
            log[p].append(prices[p])

        order_depths = {}
        for p in PRODUCTS:
            od = OrderDepth()
            sp = random.choice([1, 1, 2, 2, 5])
            od.buy_orders  = {int(prices[p] - sp):  10}
            od.sell_orders = {int(prices[p] + sp): -10}
            order_depths[p] = od

        state = TradingState(
            order_depths=order_depths,
            position=copy.deepcopy(position),
            traderData=trader_data,
            timestamp=tick,
        )
        orders_dict, _, trader_data = trader.run(state)

        for prod, ords in orders_dict.items():
            for o in ords:
                new_pos = position[prod] + o.quantity
                if abs(new_pos) <= POSITION_LIMITS.get(prod, 20):
                    position[prod]  = new_pos
                    cash           -= o.price * o.quantity
                    if o.quantity > 0:
                        log_buys[prod].append(tick)
                        log_buy_px[prod].append(o.price)
                    else:
                        log_sells[prod].append(tick)
                        log_sell_px[prod].append(o.price)

        pv  = sum(position[p] * prices[p] for p in PRODUCTS)
        pnl = cash + pv
        log_ticks.append(tick)
        log_pnl.append(pnl)
        log_cash.append(cash)
        for p in PRODUCTS:
            log_pos[p].append(position[p])

    print(f"\n✅ Backtest complete — {n_ticks} ticks")
    print(f"   Final PnL      : {pnl:,.2f} SeaShells")
    print(f"   Final positions: {position}")
    for p in PRODUCTS:
        print(f"   {p:22s}: {len(log_buys[p])} buys, {len(log_sells[p])} sells")

    return {
        "PRODUCTS":    PRODUCTS,
        "log":         log,
        "log_ticks":   log_ticks,
        "log_pnl":     log_pnl,
        "log_cash":    log_cash,
        "log_pos":     log_pos,
        "log_buys":    log_buys,
        "log_sells":   log_sells,
        "log_buy_px":  log_buy_px,
        "log_sell_px": log_sell_px,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def plot_dashboard(logs: dict, focus: str = "KELP", save_path: str = "indicator_dashboard.png"):
    """Render the 6-panel indicator dashboard and save to disk."""
    PRODUCTS     = logs["PRODUCTS"]
    log          = logs["log"]
    log_ticks    = logs["log_ticks"]
    log_pnl      = logs["log_pnl"]
    log_pos      = logs["log_pos"]
    log_buys     = logs["log_buys"]
    log_sells    = logs["log_sells"]
    log_buy_px   = logs["log_buy_px"]
    log_sell_px  = logs["log_sell_px"]

    prices_arr = log[focus]
    ticks_arr  = list(range(len(prices_arr)))

    rsi_arr                            = rsi_series(prices_arr, RSI_PERIOD)
    macd_arr, sig_arr, hist_arr        = macd_series(prices_arr, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    bb_upper_arr, bb_mid_arr, bb_lower = bollinger_series(prices_arr, BB_PERIOD, BB_STD)
    ema_s_arr = compute_ema_series(prices_arr, EMA_SHORT)
    ema_l_arr = compute_ema_series(prices_arr, EMA_LONG)

    # ── Colour palette ───────────────────────────────────────────────────────
    DARK_BG  = "#0d1117"
    PANEL_BG = "#161b22"
    GRID_CLR = "#30363d"
    TEXT_CLR = "#c9d1d9"
    BULL_CLR = "#3fb950"
    BEAR_CLR = "#f85149"
    PRICE_CLR = "#58a6ff"
    EMA_S_CLR = "#d2a8ff"
    EMA_L_CLR = "#ffa657"
    BB_CLR    = "#79c0ff"
    MACD_CLR  = "#58a6ff"
    SIG_CLR   = "#ff7b72"

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    fig.suptitle(
        f"IMC Prosperity v2 — {focus} Indicator Dashboard",
        fontsize=15, color=TEXT_CLR, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.3,
                           height_ratios=[2.5, 1, 1, 1])

    def style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color=TEXT_CLR, fontsize=10, pad=6)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        ax.grid(color=GRID_CLR, linewidth=0.4, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)

    # ── Panel 1: Price + BB + EMA + Buy/Sell signals ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ticks_arr, prices_arr, color=PRICE_CLR, linewidth=1.4, label="Price",          zorder=3)
    ax1.plot(ticks_arr, ema_s_arr,  color=EMA_S_CLR, linewidth=1,   label=f"EMA {EMA_SHORT}", linestyle="--", alpha=0.8)
    ax1.plot(ticks_arr, ema_l_arr,  color=EMA_L_CLR, linewidth=1,   label=f"EMA {EMA_LONG}",  linestyle="--", alpha=0.8)
    ax1.plot(ticks_arr, bb_upper_arr, color=BB_CLR, linewidth=0.8, linestyle=":", alpha=0.7, label="BB Upper")
    ax1.plot(ticks_arr, bb_mid_arr,   color=BB_CLR, linewidth=0.6, linestyle=":", alpha=0.4, label="BB Mid")
    ax1.plot(ticks_arr, bb_lower,     color=BB_CLR, linewidth=0.8, linestyle=":", alpha=0.7, label="BB Lower")
    ax1.fill_between(ticks_arr, bb_upper_arr, bb_lower, color=BB_CLR, alpha=0.05)

    if log_buys[focus]:
        ax1.scatter(log_buys[focus], log_buy_px[focus], marker="^",
                    color=BULL_CLR, s=80, zorder=5, label="Buy signal")
    if log_sells[focus]:
        ax1.scatter(log_sells[focus], log_sell_px[focus], marker="v",
                    color=BEAR_CLR, s=80, zorder=5, label="Sell signal")

    ax1.legend(fontsize=7, loc="upper left", framealpha=0.3, labelcolor=TEXT_CLR)
    style_ax(ax1, f"{focus} — Price + Bollinger Bands + EMA + Signals")
    ax1.set_ylabel("Price", color=TEXT_CLR, fontsize=8)

    # ── Panel 2: RSI ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ticks_arr, rsi_arr, color="#e3b341", linewidth=1.2, label="RSI")
    ax2.axhline(RSI_OVERBUY,  color=BEAR_CLR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(RSI_OVERSELL, color=BULL_CLR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(50, color=TEXT_CLR, linewidth=0.4, linestyle=":", alpha=0.3)
    ax2.fill_between(ticks_arr, rsi_arr, RSI_OVERBUY,
                     where=[r > RSI_OVERBUY  for r in rsi_arr], alpha=0.2, color=BEAR_CLR)
    ax2.fill_between(ticks_arr, rsi_arr, RSI_OVERSELL,
                     where=[r < RSI_OVERSELL for r in rsi_arr], alpha=0.2, color=BULL_CLR)
    ax2.text(2, RSI_OVERBUY  + 1,  "Overbought (SELL)", color=BEAR_CLR, fontsize=7)
    ax2.text(2, RSI_OVERSELL - 5,  "Oversold (BUY)",    color=BULL_CLR, fontsize=7)
    ax2.set_ylim(0, 100)
    style_ax(ax2, f"RSI ({RSI_PERIOD}-period)")
    ax2.set_ylabel("RSI", color=TEXT_CLR, fontsize=8)

    # ── Panel 3: MACD ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(ticks_arr, macd_arr, color=MACD_CLR, linewidth=1.2, label="MACD")
    ax3.plot(ticks_arr, sig_arr,  color=SIG_CLR,  linewidth=1,   label="Signal", linestyle="--")
    ax3.bar(ticks_arr, hist_arr,
            color=[BULL_CLR if h >= 0 else BEAR_CLR for h in hist_arr],
            alpha=0.5, width=0.8, label="Histogram")
    ax3.axhline(0, color=TEXT_CLR, linewidth=0.4, alpha=0.4)
    ax3.legend(fontsize=7, loc="upper left", framealpha=0.3, labelcolor=TEXT_CLR)
    style_ax(ax3, f"MACD ({MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL})")
    ax3.set_ylabel("MACD", color=TEXT_CLR, fontsize=8)

    # ── Panel 4: Positions (all products) ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    p_colors = [PRICE_CLR, BULL_CLR, BEAR_CLR]
    for p, c in zip(PRODUCTS, p_colors):
        ax4.plot(log_ticks, log_pos[p], label=p.replace("_", " ").title(), color=c, linewidth=1)
    ax4.axhline( 50, color="#ff6b6b", linewidth=0.6, linestyle=":", alpha=0.5)
    ax4.axhline(-50, color="#ff6b6b", linewidth=0.6, linestyle=":", alpha=0.5)
    ax4.axhline(  0, color=TEXT_CLR,  linewidth=0.4, alpha=0.3)
    ax4.legend(fontsize=7, framealpha=0.3, labelcolor=TEXT_CLR)
    style_ax(ax4, "Positions (all products)")
    ax4.set_ylabel("Units", color=TEXT_CLR, fontsize=8)

    # ── Panel 5: Cumulative PnL ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(log_ticks, log_pnl, color="#f0e68c", linewidth=1.5, label="PnL")
    ax5.axhline(0, color=TEXT_CLR, linewidth=0.5, linestyle="--", alpha=0.4)
    ax5.fill_between(log_ticks, log_pnl, 0,
                     where=[p >= 0 for p in log_pnl], alpha=0.2, color=BULL_CLR)
    ax5.fill_between(log_ticks, log_pnl, 0,
                     where=[p  < 0 for p in log_pnl], alpha=0.2, color=BEAR_CLR)
    style_ax(ax5, "Cumulative PnL (SeaShells)")
    ax5.set_ylabel("SeaShells", color=TEXT_CLR, fontsize=8)

    # ── Panel 6: EMA Z-score ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    z_scores = []
    for i in range(len(prices_arr)):
        window = prices_arr[max(0, i - 9): i + 1]
        es_i   = ema_s_arr[i]
        el_i   = ema_l_arr[i]
        if len(window) >= 5:
            mean_w = sum(window) / len(window)
            std_i  = math.sqrt(sum((p - mean_w) ** 2 for p in window) / len(window))
            z_scores.append((es_i - el_i) / std_i if std_i > 0 else 0)
        else:
            z_scores.append(0)

    ax6.plot(ticks_arr, z_scores, color="#c9d1d9", linewidth=1, label="Z-score")
    ax6.axhline( Z_ENTRY, color=BEAR_CLR, linewidth=0.8, linestyle="--", alpha=0.7, label=f"Sell zone (+{Z_ENTRY})")
    ax6.axhline(-Z_ENTRY, color=BULL_CLR, linewidth=0.8, linestyle="--", alpha=0.7, label=f"Buy zone (-{Z_ENTRY})")
    ax6.axhline(0, color=TEXT_CLR, linewidth=0.4, alpha=0.3)
    ax6.fill_between(ticks_arr, z_scores,  Z_ENTRY,  where=[z >  Z_ENTRY for z in z_scores], alpha=0.2, color=BEAR_CLR)
    ax6.fill_between(ticks_arr, z_scores, -Z_ENTRY,  where=[z < -Z_ENTRY for z in z_scores], alpha=0.2, color=BULL_CLR)
    ax6.legend(fontsize=7, framealpha=0.3, labelcolor=TEXT_CLR)
    style_ax(ax6, f"EMA Z-score — Mean Reversion Signal ({focus})")
    ax6.set_ylabel("Z-score", color=TEXT_CLR, fontsize=8)
    ax6.set_xlabel("Tick",    color=TEXT_CLR, fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.show()
    print(f"✅ Dashboard saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. EXPORT COMPACT trader.py FOR IMC UPLOAD
# ═════════════════════════════════════════════════════════════════════════════

COMPACT_TRADER_CODE = '''from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json, math

POSITION_LIMITS = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
MM_FAIR_VALUE   = {"RAINFOREST_RESIN": 10_000}
MM_BASE_SPREAD  = 2;  MM_ORDER_SIZE = 10
EMA_SHORT = 5;  EMA_LONG = 20;  Z_ENTRY = 1.5
RSI_PERIOD = 14;  RSI_OVERBUY = 70;  RSI_OVERSELL = 30
MACD_FAST = 12;  MACD_SLOW = 26;  MACD_SIGNAL = 9
BB_PERIOD = 20;  BB_STD = 2.0
CONFLUENCE_MIN = 2
BASE_ORDER_SIZE = 8
FEAR_SPREAD_PCT = 0.003;  GREED_SIZE_MULT = 1.5

def ema_update(prev, new, w): return (2/(w+1))*new + (1-2/(w+1))*prev

def compute_rsi(prices, period=14):
    if len(prices) < period+1: return 50.0
    d=[prices[i]-prices[i-1] for i in range(1,len(prices))]
    ag=sum(max(x,0) for x in d[:period])/period
    al=sum(abs(min(x,0)) for x in d[:period])/period
    for i in range(period,len(d)):
        ag=(ag*(period-1)+max(d[i],0))/period
        al=(al*(period-1)+abs(min(d[i],0)))/period
    return 100 if al==0 else 100-(100/(1+ag/al))

def compute_bollinger(prices, period=20, ns=2.0):
    if len(prices)<period: m=prices[-1]; return m,m,m
    w=prices[-period:]; m=sum(w)/period
    s=math.sqrt(sum((p-m)**2 for p in w)/period)
    return m+ns*s, m, m-ns*s

def best_bid_ask(od):
    return (max(od.buy_orders) if od.buy_orders else None,
            min(od.sell_orders) if od.sell_orders else None)

def clamp_qty(prod, qty, pos):
    lim=POSITION_LIMITS.get(prod,20)
    return min(qty,lim-pos) if qty>0 else max(qty,-lim-pos)

class Trader:
    def run(self, state: TradingState) -> tuple:
        try: mem=json.loads(state.traderData) if state.traderData else {}
        except: mem={}
        ph=mem.get("ph",{}); es=mem.get("es",{}); el=mem.get("el",{})
        mf=mem.get("mf",{}); ms=mem.get("ms",{}); mg=mem.get("mg",{})
        result={}
        for prod,od in state.order_depths.items():
            orders=[]; pos=state.position.get(prod,0)
            if not od.buy_orders or not od.sell_orders: continue
            bb,ba=best_bid_ask(od); mid=(bb+ba)/2.0
            h=ph.get(prod,[]); h.append(mid)
            if len(h)>50: h=h[-50:]
            ph[prod]=h
            if prod not in es: es[prod]=mid; el[prod]=mid
            else: es[prod]=ema_update(es[prod],mid,EMA_SHORT); el[prod]=ema_update(el[prod],mid,EMA_LONG)
            if prod not in mf: mf[prod]=mid; ms[prod]=mid; mg[prod]=0.0
            else: mf[prod]=ema_update(mf[prod],mid,MACD_FAST); ms[prod]=ema_update(ms[prod],mid,MACD_SLOW)
            mv=mf[prod]-ms[prod]; mg[prod]=ema_update(mg[prod],mv,MACD_SIGNAL); mh=mv-mg[prod]
            fear=(ba-bb)/mid>FEAR_SPREAD_PCT if mid>0 else False
            sz=1.0 if fear else GREED_SIZE_MULT
            w=h[-10:]; std=math.sqrt(sum((p-sum(w)/len(w))**2 for p in w)/len(w)) if len(h)>=5 else 0
            if prod in MM_FAIR_VALUE:
                fv=MM_FAIR_VALUE[prod]; sp=MM_BASE_SPREAD+(2 if fear else 0)
                if ba<=fv-sp:
                    q=clamp_qty(prod,int(MM_ORDER_SIZE*sz),pos)
                    if q>0: orders.append(Order(prod,ba,q))
                if bb>=fv+sp:
                    q=clamp_qty(prod,-int(MM_ORDER_SIZE*sz),pos)
                    if q<0: orders.append(Order(prod,bb,q))
                bq=clamp_qty(prod,MM_ORDER_SIZE,pos); sq=clamp_qty(prod,-MM_ORDER_SIZE,pos)
                if bq>0: orders.append(Order(prod,fv-sp,bq))
                if sq<0: orders.append(Order(prod,fv+sp,sq))
            else:
                if len(h)<max(RSI_PERIOD,BB_PERIOD,MACD_SLOW)+2:
                    if orders: result[prod]=orders
                    continue
                rsi=compute_rsi(h,RSI_PERIOD)
                bu,bm,bl=compute_bollinger(h,BB_PERIOD,BB_STD)
                z=(es[prod]-el[prod])/std if std>0 else 0
                votes=[]
                votes.append(1 if rsi<RSI_OVERSELL else -1 if rsi>RSI_OVERBUY else 0)
                votes.append(1 if mh>0 else -1 if mh<0 else 0)
                votes.append(1 if mid<=bl else -1 if mid>=bu else 0)
                votes.append(1 if z<-Z_ENTRY else -1 if z>Z_ENTRY else 0)
                bc=votes.count(1); sc=votes.count(-1)
                ts=int(BASE_ORDER_SIZE*sz)
                if bc>=CONFLUENCE_MIN:
                    q=clamp_qty(prod,ts,pos)
                    if q>0: orders.append(Order(prod,ba,q))
                elif sc>=CONFLUENCE_MIN:
                    q=clamp_qty(prod,-ts,pos)
                    if q<0: orders.append(Order(prod,bb,q))
                else:
                    sh=max(1,int(std*0.5)); em=(es[prod]+el[prod])/2
                    bq=clamp_qty(prod,BASE_ORDER_SIZE//2,pos); sq2=clamp_qty(prod,-BASE_ORDER_SIZE//2,pos)
                    if bq>0: orders.append(Order(prod,int(em-sh),bq))
                    if sq2<0: orders.append(Order(prod,int(em+sh),sq2))
            if orders: result[prod]=orders
        td=json.dumps({"ph":ph,"es":es,"el":el,"mf":mf,"ms":ms,"mg":mg})
        return result,0,td
'''


def export_trader(path: str = "trader.py"):
    """Write the compact IMC-ready trader.py to disk."""
    import os
    with open(path, "w") as f:
        f.write(COMPACT_TRADER_CODE.strip())
    kb = os.path.getsize(path) / 1024
    print(f"\n✅ trader.py exported!")
    print(f"   Size: {kb:.2f} KB  (limit = 100 KB ✓)")
    print(f"   Path: {os.path.abspath(path)}")
    print("\n📤 Upload to: prosperity.imc.com/game → Upload & Log tab")


# ═════════════════════════════════════════════════════════════════════════════
# 8. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMC Trader v2 — backtest + dashboard")
    parser.add_argument("--ticks",  type=int, default=150,    help="Number of backtest ticks (default 150)")
    parser.add_argument("--seed",   type=int, default=42,     help="Random seed (default 42)")
    parser.add_argument("--focus",  type=str, default="KELP", help="Product for indicator charts (default KELP)")
    parser.add_argument("--export", action="store_true",      help="Also export compact trader.py for IMC upload")
    parser.add_argument("--no-plot", action="store_true",     help="Skip the dashboard (backtest only)")
    args = parser.parse_args()

    print("=" * 60)
    print(" IMC Prosperity Trader v2 — RSI + MACD + BB + EMA Z-score")
    print("=" * 60)
    print(f"  Ticks  : {args.ticks}")
    print(f"  Seed   : {args.seed}")
    print(f"  Focus  : {args.focus}")
    print()

    logs = run_backtest(n_ticks=args.ticks, seed=args.seed)

    if not args.no_plot:
        plot_dashboard(logs, focus=args.focus)

    if args.export:
        export_trader()
