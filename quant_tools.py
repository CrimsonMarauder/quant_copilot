"""
quant_tools.py
--------------
The quant engine. Deeper than a basic demo:

  * analyze_pair      - Engle-Granger cointegration AND an ADF stationarity test on
                        the spread, plus Ornstein-Uhlenbeck half-life and hedge ratio.
  * backtest_pair     - OUT-OF-SAMPLE backtest: the hedge ratio is fit on a training
                        slice, a causal ROLLING z-score generates signals, and we
                        report TRAIN vs TEST metrics (Sharpe, Sortino, drawdown,
                        time-in-market, win rate) plus a buy-&-hold benchmark.
  * screen_pairs      - rank every pair in a basket by cointegration strength.
  * backtest_portfolio- pick the cointegrated pairs in a basket and combine them into
                        one equally-weighted, market-neutral portfolio.

All functions fetch their own data and return small JSON-friendly dicts.
Educational only: no transaction costs, borrow costs, or slippage.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

import config


# --------------------------------------------------------------------- helpers
def _download(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    period = max(int(lookback_days) + 10, 60)
    raw = yf.download(tickers, period=f"{period}d", interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise ValueError("No price data returned. Check the ticker symbols.")
    prices = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].copy()
    if not isinstance(raw.columns, pd.MultiIndex):
        prices.columns = [tickers[0]]
    return prices.dropna(how="all").ffill().dropna()


def _hedge_ratio(p1: pd.Series, p2: pd.Series) -> float:
    X = sm.add_constant(p2.values)
    return float(sm.OLS(p1.values, X).fit().params[1])


def _best_coint(p1: pd.Series, p2: pd.Series) -> float:
    """Engle-Granger is order-dependent, so test both directions and keep the better p-value."""
    _, pa, _ = coint(p1, p2)
    _, pb, _ = coint(p2, p1)
    return float(min(pa, pb))


def _grade(coint_p: float, adf_p: float):
    """Translate the raw p-values into a graded, plain-language verdict."""
    if coint_p < config.COINT_STRONG:
        return "strong", True, "These two move together tightly — a reliable pair to trade."
    if coint_p < config.COINT_PVALUE or adf_p < config.COINT_STRONG:
        return "moderate", True, "These two mostly move together — tradeable, but watch it more closely."
    if coint_p < config.COINT_WEAK:
        return "weak", False, "Only a loose link this window — risky to trade as a pair; try a longer lookback."
    return "none", False, "These two don't reliably move together right now — not a good pair."


def _half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    lag = s.shift(1).dropna()
    s = s.loc[lag.index]
    beta = sm.OLS((s - lag).values, sm.add_constant(lag.values)).fit().params[1]
    return float("inf") if beta >= 0 else float(-np.log(2) / beta)


def _metrics(returns: pd.Series, positions: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0, "time_in_market_pct": 0.0}
    equity = (1 + r).cumprod()
    downside = r[r < 0].std()
    sortino = float(np.sqrt(252) * r.mean() / downside) if downside and downside > 0 else 0.0
    return {
        "sharpe": round(float(np.sqrt(252) * r.mean() / r.std()), 2),
        "sortino": round(sortino, 2),
        "total_return_pct": round(float(equity.iloc[-1] - 1) * 100, 2),
        "max_drawdown_pct": round(float((equity / equity.cummax() - 1).min()) * 100, 2),
        "time_in_market_pct": round(float((positions.reindex(r.index).fillna(0) != 0).mean()) * 100, 1),
    }


def _positions(z: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    pos = pd.Series(0.0, index=z.index)
    cur = 0.0
    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = cur
            continue
        if cur == 0.0:
            cur = -1.0 if zi > entry_z else (1.0 if zi < -entry_z else 0.0)
        elif abs(zi) < exit_z:
            cur = 0.0
        pos.iloc[i] = cur
    return pos


def _spread_returns(p1, p2, hedge):
    r1, r2 = p1.pct_change(), p2.pct_change()
    return (r1 - hedge * r2) / (1 + abs(hedge)), r1, r2


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder-style RSI of a series (here applied to the spread)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _leg_actions(spread_dir: int, hedge: float, t1: str, t2: str):
    """Translate a spread direction (+1 long spread, -1 short spread) into leg trades.

    spread = p1 - hedge*p2, so 'long the spread' means long p1 and short hedge*p2.
    The sign of the hedge ratio decides leg 2's direction.
    """
    if spread_dir == 0:
        return "NEUTRAL - no trade", None, None, None
    a1 = "LONG" if spread_dir > 0 else "SHORT"
    if hedge >= 0:
        a2 = "SHORT" if spread_dir > 0 else "LONG"
    else:
        a2 = "LONG" if spread_dir > 0 else "SHORT"
    long_leg = t1 if a1 == "LONG" else t2
    short_leg = t2 if a1 == "LONG" else t1
    return f"{a1} {t1} / {a2} {t2}", long_leg, short_leg, (a1, a2)


# --------------------------------------------------------------------- TOOL 1
def analyze_pair(ticker1: str, ticker2: str, lookback_days: int = config.LOOKBACK_DAYS) -> dict:
    """Run the statistical tests that decide if two stocks are tradeable as a pair.

    Computes the Engle-Granger cointegration p-value, an Augmented Dickey-Fuller
    (ADF) stationarity p-value on the spread, the mean-reversion half-life in days,
    and the hedge ratio. Call this first to judge whether a pair is worth backtesting.

    Args:
        ticker1: First ticker, e.g. "KO".
        ticker2: Second ticker, e.g. "PEP".
        lookback_days: Days of history to analyse.

    Returns:
        Dict with coint_pvalue, adf_pvalue, both pass/fail flags, half_life_days, hedge_ratio.
    """
    try:
        t1, t2 = ticker1.upper().strip(), ticker2.upper().strip()
        prices = _download([t1, t2], lookback_days)
        if t1 not in prices or t2 not in prices or len(prices) < 40:
            return {"error": f"Not enough overlapping data for {t1} and {t2}."}
        p1, p2 = prices[t1], prices[t2]
        hedge = _hedge_ratio(p1, p2)
        spread = p1 - hedge * p2
        coint_p = _best_coint(p1, p2)
        adf_p = float(adfuller(spread.dropna(), maxlag=1, autolag=None)[1])
        hl = _half_life(spread)
        corr = float(p1.pct_change().corr(p2.pct_change()))
        verdict, tradeable, summary = _grade(coint_p, adf_p)
        return {
            "pair": f"{t1}/{t2}",
            "verdict": verdict,                 # strong | moderate | weak | none
            "tradeable_as_pair": bool(tradeable),
            "plain_summary": summary,
            "coint_pvalue": round(float(coint_p), 4),
            "adf_pvalue": round(adf_p, 4),
            "cointegrated": bool(tradeable),
            "spread_stationary": bool(adf_p < config.COINT_STRONG),
            "return_correlation": round(corr, 3),
            "half_life_days": round(hl, 1) if np.isfinite(hl) else "not mean-reverting",
            "hedge_ratio": round(hedge, 3),
            "days_analysed": int(len(prices)),
            "tip": "If the verdict is weak/none for two similar companies, try lookback_days=1095 (3 years).",
        }
    except Exception as e:
        return {"error": f"analyze_pair failed for {ticker1}/{ticker2}: {e}"}


# --------------------------------------------------------------------- TOOL 2
def backtest_pair(ticker1: str, ticker2: str, lookback_days: int = config.LOOKBACK_DAYS,
                  entry_z: float = config.ENTRY_Z, exit_z: float = config.EXIT_Z) -> dict:
    """Out-of-sample backtest of a z-score pairs strategy on two stocks.

    The hedge ratio is fit on the first 70% of history (train); signals use a causal
    rolling z-score; performance is reported separately for the train and the unseen
    test slice, alongside an equal-weight buy-and-hold benchmark over the test slice.
    A big gap between train and test Sharpe is a sign of overfitting.

    Args:
        ticker1: First ticker, e.g. "V".
        ticker2: Second ticker, e.g. "MA".
        lookback_days: Days of history to test on.
        entry_z: Z-score to open a trade.
        exit_z: Z-score to close a trade.

    Returns:
        Dict with train and test metrics (sharpe, sortino, returns, drawdown,
        time-in-market), trade stats, and the benchmark return.
    """
    try:
        t1, t2 = ticker1.upper().strip(), ticker2.upper().strip()
        prices = _download([t1, t2], lookback_days)
        if t1 not in prices or t2 not in prices or len(prices) < 60:
            return {"error": f"Not enough overlapping data for {t1} and {t2}."}
        p1, p2 = prices[t1], prices[t2]

        split = int(len(prices) * config.TRAIN_FRAC)
        hedge = _hedge_ratio(p1.iloc[:split], p2.iloc[:split])      # fit on TRAIN only
        spread = p1 - hedge * p2
        mu = spread.rolling(config.ROLL_WINDOW).mean()
        sd = spread.rolling(config.ROLL_WINDOW).std()
        z = (spread - mu) / sd                                      # causal rolling z

        pos = _positions(z, entry_z, exit_z)
        spread_ret, r1, r2 = _spread_returns(p1, p2, hedge)
        strat = (pos.shift(1) * spread_ret).dropna()

        train_idx = strat.index[strat.index <= prices.index[split]]
        test_idx = strat.index[strat.index > prices.index[split]]
        train_m = _metrics(strat.loc[train_idx], pos)
        test_m = _metrics(strat.loc[test_idx], pos)

        # trade + benchmark stats on the test slice
        test_pos = pos.reindex(test_idx).fillna(0)
        trades = int((test_pos.diff().fillna(0) != 0).sum())
        block = (test_pos != test_pos.shift()).cumsum()
        pnl = strat.loc[test_idx].groupby(block).sum()
        active = pnl[test_pos.groupby(block).first() != 0]
        win_rate = round(float((active > 0).mean()) * 100, 1) if len(active) else 0.0
        bench = (0.5 * r1 + 0.5 * r2).loc[test_idx].dropna()
        bench_ret = round(float(((1 + bench).cumprod().iloc[-1] - 1)) * 100, 2) if len(bench) else 0.0

        return {
            "pair": f"{t1}/{t2}", "hedge_ratio": round(hedge, 3),
            "train": train_m, "test": test_m,
            "test_num_trades": trades, "test_win_rate_pct": win_rate,
            "test_benchmark_buyhold_pct": bench_ret,
            "params": {"entry_z": entry_z, "exit_z": exit_z, "roll_window": config.ROLL_WINDOW,
                       "train_frac": config.TRAIN_FRAC},
        }
    except Exception as e:
        return {"error": f"backtest_pair failed for {ticker1}/{ticker2}: {e}"}


# --------------------------------------------------------------------- TOOL 3
def screen_pairs(tickers: str, lookback_days: int = config.LOOKBACK_DAYS) -> dict:
    """Rank every pair in a basket of stocks by cointegration strength.

    Args:
        tickers: Comma-separated tickers, e.g. "KO,PEP,XOM,CVX,JPM,BAC".
        lookback_days: Days of history to analyse.

    Returns:
        Dict with the top pairs ranked by cointegration p-value, each with half-life.
    """
    try:
        syms = list(dict.fromkeys(s.strip().upper() for s in tickers.split(",") if s.strip()))
        if len(syms) < 2:
            return {"error": "Provide at least two comma-separated tickers."}
        syms = syms[:8]
        prices = _download(syms, lookback_days)
        syms = [s for s in syms if s in prices.columns]
        res = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a, b = syms[i], syms[j]
                try:
                    pv = _best_coint(prices[a], prices[b])
                    hl = _half_life(prices[a] - _hedge_ratio(prices[a], prices[b]) * prices[b])
                    res.append({"pair": f"{a}/{b}", "coint_pvalue": round(float(pv), 4),
                                "half_life_days": round(hl, 1) if np.isfinite(hl) else "n/a"})
                except Exception:
                    continue
        res.sort(key=lambda r: r["coint_pvalue"])
        return {"tickers_analysed": syms, "top_pairs": res[:5],
                "note": "Lower p-value = stronger cointegration; < 0.05 is the usual cutoff."}
    except Exception as e:
        return {"error": f"screen_pairs failed: {e}"}


# --------------------------------------------------------------------- TOOL 4
def backtest_portfolio(tickers: str, lookback_days: int = config.LOOKBACK_DAYS) -> dict:
    """Build and backtest a market-neutral PORTFOLIO from the cointegrated pairs in a basket.

    Screens all pairs, keeps those that are cointegrated (p < 0.05), backtests each
    out-of-sample, and combines them with equal weight into one portfolio. Use this
    when the user wants a diversified multi-pair strategy rather than a single pair.

    Args:
        tickers: Comma-separated tickers, e.g. "KO,PEP,XOM,CVX,JPM,BAC,F,GM".
        lookback_days: Days of history.

    Returns:
        Dict with the selected pairs, each pair's test Sharpe/return, and the combined
        portfolio's test Sharpe, return, and max drawdown.
    """
    try:
        syms = list(dict.fromkeys(s.strip().upper() for s in tickers.split(",") if s.strip()))[:8]
        if len(syms) < 2:
            return {"error": "Provide at least two comma-separated tickers."}
        prices = _download(syms, lookback_days)
        syms = [s for s in syms if s in prices.columns]
        split = int(len(prices) * config.TRAIN_FRAC)

        leg_curves, chosen = [], []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a, b = syms[i], syms[j]
                try:
                    pv = _best_coint(prices[a], prices[b])
                    if pv >= config.COINT_PVALUE:
                        continue
                    p1, p2 = prices[a], prices[b]
                    hedge = _hedge_ratio(p1.iloc[:split], p2.iloc[:split])
                    spread = p1 - hedge * p2
                    z = (spread - spread.rolling(config.ROLL_WINDOW).mean()) / spread.rolling(config.ROLL_WINDOW).std()
                    pos = _positions(z, config.ENTRY_Z, config.EXIT_Z)
                    sret, _, _ = _spread_returns(p1, p2, hedge)
                    strat = (pos.shift(1) * sret).dropna()
                    test = strat[strat.index > prices.index[split]]
                    if len(test) < 10 or test.std() == 0:
                        continue
                    m = _metrics(test, pos)
                    chosen.append({"pair": f"{a}/{b}", "coint_pvalue": round(float(pv), 4),
                                   "test_sharpe": m["sharpe"], "test_return_pct": m["total_return_pct"]})
                    leg_curves.append(test)
                except Exception:
                    continue

        if not leg_curves:
            return {"selected_pairs": [], "note": "No cointegrated pairs (p < 0.05) found in this basket."}

        port = pd.concat(leg_curves, axis=1).fillna(0).mean(axis=1)  # equal-weight combine
        pm = _metrics(port, pd.Series(1.0, index=port.index))
        return {
            "selected_pairs": sorted(chosen, key=lambda c: c["coint_pvalue"]),
            "num_pairs": len(chosen),
            "portfolio_test_sharpe": pm["sharpe"],
            "portfolio_test_return_pct": pm["total_return_pct"],
            "portfolio_test_max_drawdown_pct": pm["max_drawdown_pct"],
            "note": "Equal-weighted combination of the cointegrated pairs, evaluated out-of-sample.",
        }
    except Exception as e:
        return {"error": f"backtest_portfolio failed: {e}"}


# --------------------------------------------------------------------- TOOL 5
def current_signal(ticker1: str, ticker2: str, lookback_days: int = config.LOOKBACK_DAYS,
                   entry_z: float = config.ENTRY_Z, exit_z: float = config.EXIT_Z) -> dict:
    """Give a live, actionable pairs-trading signal: which leg to LONG and which to SHORT right now.

    For a (preferably cointegrated) pair this computes where the spread sits today versus
    its recent history and translates that into a concrete recommendation. It reports the
    current z-score, Bollinger bands on the spread, the spread's RSI, its historical
    percentile, whether it is reverting or still diverging, the recent return correlation
    (relationship health), the expected days to revert (half-life), and a signal strength.

    Use this when the user asks which stock is cheap/rich, which to buy or short, or what
    the trade is now. Always pair it with analyze_pair so you know if the pair is cointegrated.

    Args:
        ticker1: First ticker, e.g. "KO".
        ticker2: Second ticker, e.g. "PEP".
        lookback_days: Days of history to analyse.
        entry_z: Z-score threshold that defines the Bollinger entry bands.
        exit_z: Z-score inside which the pair is considered fairly valued (no trade).

    Returns:
        Dict with the recommendation (which leg to long/short), the rich/cheap assessment,
        and all supporting indicators.
    """
    try:
        t1, t2 = ticker1.upper().strip(), ticker2.upper().strip()
        prices = _download([t1, t2], lookback_days)
        if t1 not in prices or t2 not in prices or len(prices) < 40:
            return {"error": f"Not enough overlapping data for {t1} and {t2}."}
        p1, p2 = prices[t1], prices[t2]

        hedge = _hedge_ratio(p1, p2)
        spread = p1 - hedge * p2
        mu = spread.rolling(config.ROLL_WINDOW).mean()
        sd = spread.rolling(config.ROLL_WINDOW).std()
        z = (spread - mu) / sd

        cur_spread = float(spread.iloc[-1])
        cur_mu = float(mu.iloc[-1]); cur_sd = float(sd.iloc[-1])
        cur_z = float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else 0.0
        upper_band = cur_mu + entry_z * cur_sd
        lower_band = cur_mu - entry_z * cur_sd

        # supporting indicators
        percentile = float((spread < cur_spread).mean() * 100)            # where today's spread sits historically
        rsi_series = _rsi(spread)
        spread_rsi = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None
        z_prev = float(z.iloc[-6]) if len(z) > 6 and pd.notna(z.iloc[-6]) else cur_z
        if abs(cur_z) < abs(z_prev) - 0.1:
            z_trend = "reverting toward mean"
        elif abs(cur_z) > abs(z_prev) + 0.1:
            z_trend = "still diverging"
        else:
            z_trend = "flat"
        r1, r2 = p1.pct_change(), p2.pct_change()
        recent_corr = float(r1.tail(60).corr(r2.tail(60)))
        coint_p = _best_coint(p1, p2)
        hl = _half_life(spread)

        # decide direction from the current z-score
        if cur_z >= entry_z:
            spread_dir = -1                       # spread rich -> short it
        elif cur_z <= -entry_z:
            spread_dir = 1                        # spread cheap -> long it
        else:
            spread_dir = 0                        # inside bands -> no trade
        signal, long_leg, short_leg, _ = _leg_actions(spread_dir, hedge, t1, t2)

        # which name is rich vs cheap (relative-value read)
        rich_leg = t1 if (cur_z > 0) == (hedge >= 0) else t2
        cheap_leg = t2 if rich_leg == t1 else t1
        assessment = (f"{rich_leg} looks rich and {cheap_leg} looks cheap "
                      f"(spread at {percentile:.0f}th percentile, z = {cur_z:+.2f})")

        # strength + cautions
        cointegrated = coint_p < config.COINT_PVALUE
        if cointegrated and abs(cur_z) >= 2:
            strength = "strong"
        elif abs(cur_z) >= 1.5:
            strength = "moderate"
        else:
            strength = "weak"
        cautions = []
        if not cointegrated:
            cautions.append(f"pair is NOT cointegrated (p={coint_p:.3f}); mean-reversion is unreliable")
        if z_trend == "still diverging" and spread_dir != 0:
            cautions.append("spread is still widening - reversion may not have started")
        if recent_corr < 0.3:
            cautions.append(f"recent 60-day return correlation is low ({recent_corr:.2f}); relationship may be weakening")
        if not np.isfinite(hl) or hl > 60:
            cautions.append("half-life is long/unstable; reversion may be slow")

        # one plain-English line a non-technical user can act on
        if spread_dir == 0:
            plain = (f"No trade right now: {t1} and {t2} are priced about normally relative to "
                     f"each other (gap is near its average), so there's no clear edge yet.")
        else:
            days_txt = f"about {hl:.0f} days" if np.isfinite(hl) else "an unclear time"
            plain = (f"Buy (long) {long_leg} and short {short_leg}. {rich_leg} looks expensive and "
                     f"{cheap_leg} looks cheap versus their usual relationship, and that gap historically "
                     f"closes over {days_txt}. You profit if the gap narrows back to normal.")

        return {
            "pair": f"{t1}/{t2}",
            "signal": signal,
            "plain_recommendation": plain,
            "long_leg": long_leg, "short_leg": short_leg,
            "assessment": assessment,
            "strength": strength if spread_dir != 0 else "no-trade",
            "current_zscore": round(cur_z, 2),
            "spread_percentile": round(percentile, 1),
            "spread_rsi": round(spread_rsi, 1) if spread_rsi is not None else None,
            "zscore_trend": z_trend,
            "bollinger_bands": {"lower": round(lower_band, 3), "mid": round(cur_mu, 3),
                                "upper": round(upper_band, 3), "current_spread": round(cur_spread, 3)},
            "recent_correlation_60d": round(recent_corr, 3),
            "cointegrated": bool(cointegrated), "coint_pvalue": round(float(coint_p), 4),
            "hedge_ratio": round(hedge, 3),
            "expected_reversion_days": round(hl, 1) if np.isfinite(hl) else "n/a",
            "cautions": cautions,
            "rule": f"Trade when |z| >= {entry_z}; exit near |z| < {exit_z}. Educational only.",
        }
    except Exception as e:
        return {"error": f"current_signal failed for {ticker1}/{ticker2}: {e}"}
