#!/usr/bin/env python3
"""xstocks-movers-forecast-01 entry: forecast the next 5 session days' moves
of the Nasdaq xStocks universe (order + return bins per day).

Model (see MODEL.md in the work dir for the backtests behind every choice):
  - Each name's daily log return is Student-t with df=5 (fat tails beat the
    normal on a year of Nasdaq daily closes: +0.20/day over 8 names) and zero
    drift, except day 1 (below).
  - Daily vol: 30% the trailing realized vol from the hourly pool bars
    (sqrt(24 * mean squared hourly log return)) + 70% a prior. The prior is
    85% the name's own typical daily vol (1y of Nasdaq closes: MSTR 5.1%,
    TSLA 2.9%, NVDA 2.4%, META 2.6%, MSFT/AMZN/GOOGL ~2.1%, AAPL 1.6%,
    SPCXX 2.3% — no listed proxy, so the cross-name median) + 15% the
    cross-name median of the raw hourly vols (a regime hedge for the week).
    Shrinking toward the per-name prior beat shrinking toward the cross-name
    median by ~0.27/day in backtest. Floor 0.005.
  - Day 1's return runs from the previous session day's 16:00 NY close, and
    the history already holds the first hours of it (it ends 2h after that
    close in the final). The observed part is already inside day 1's return,
    so day 1's mean is shifted by the observed partial log move and its sd
    is scaled by sqrt(remaining hours / 24).
  - Order probabilities come from 20,000 Monte Carlo draws of a JOINT
    Student-t(5) with correlation estimated from the hourly log returns,
    shrunk 50/50 toward an equicorrelation matrix (validated best on the
    proxy backtest; raw pairwise correlations from ~21 hourly bars are far
    too noisy to use unshrunk). No momentum tilt: trailing 1-20d returns
    showed zero cross-sectional rank prediction (mean Spearman -0.04..+0.00).
  - Bin probabilities are computed in log space (simple return = exp(log
    return) - 1, so bin edges are log-transformed) with the t CDF from the
    regularized incomplete beta function.
  - Deterministic: mulberry32 + polar Box-Muller, fixed seed; the same
    forecast is written for days 2-5 (identical distributions), day 1 gets
    its own (partial-move) forecast.

Stdlib only. Reads $MAKEFUN_INPUT_DIR else /input, writes
$MAKEFUN_OUTPUT_DIR else /output. Window, universe, places and bin edges
are read from task.json, never hardcoded.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- model constants (see MODEL.md) ---
DF = 5                      # Student-t degrees of freedom
DRAWS = 100_000             # Monte Carlo draws per order distribution (raised
                            # from 20k: smoother place probabilities, runtime
                            # still ~10 s vs the 300 s limit)
SEED = 20_261_107           # fixed seed (same value the house baseline uses)
SMOOTHING = 0.5             # half a draw on every (name, place)
MICRO = 1_000_000           # probabilities are floored to 6 decimals
MIN_VOL = 0.005             # lowest daily vol, one bin's width
SESSION_HOURS = 24          # hours from one session day's close to the next
RAW_WEIGHT = 0.3            # weight on the trailing realized hourly vol
PRIOR_NAME_WEIGHT = 0.85    # weight on the name's own prior inside the prior
CORR_SHRINK = 0.5           # weight on the equicorrelation matrix
HOUR_MS = 3_600_000

# Typical daily vol (RMS of daily log returns, 1y of Nasdaq closes to
# 2026-11-13, Yahoo Finance adj closes). SPCXX (SpaceX, private) has no
# listed proxy: the cross-name median of the other eight.
NAME_PRIOR = {
    "AAPLX": 0.0158,
    "AMZNX": 0.0210,
    "GOOGLX": 0.0212,
    "METAX": 0.0261,
    "MSFTX": 0.0211,
    "MSTRX": 0.0510,
    "NVDAX": 0.0244,
    "TSLAX": 0.0289,
    "SPCXX": 0.0228,
}
MEDIAN_PRIOR = 0.0228


# --- small numeric helpers (pure stdlib, deterministic) ---
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    maxit, eps, fpmin = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(x, df):
    """CDF of the standard Student-t with df degrees of freedom."""
    xb = df / (df + x * x)
    ib = _betainc(df / 2.0, 0.5, xb)
    return 1.0 - 0.5 * ib if x > 0 else 0.5 * ib


def _imul(a, b):
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF


class Rng:
    """mulberry32 + polar Box-Muller, fixed seed: deterministic on any Python."""

    def __init__(self, seed):
        self.a = seed & 0xFFFFFFFF
        self.spare = None

    def uniform(self):
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = _imul(self.a ^ (self.a >> 15), 1 | self.a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    def gauss(self):
        if self.spare is not None:
            v = self.spare
            self.spare = None
            return v
        while True:
            u = 2.0 * self.uniform() - 1.0
            v = 2.0 * self.uniform() - 1.0
            s = u * u + v * v
            if s < 1.0 and s != 0.0:
                break
        f = math.sqrt((-2.0 * math.log(s)) / s)
        self.spare = v * f
        return u * f


def cholesky(mat):
    """Lower Cholesky factor; None if the matrix is not positive definite."""
    n = len(mat)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = mat[i][i] - s
                if v <= 0.0:
                    return None
                L[i][j] = math.sqrt(v)
            else:
                if L[j][j] == 0.0:
                    return None
                L[i][j] = (mat[i][j] - s) / L[j][j]
    return L


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if den == 0.0:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# --- input ---
def load_bars(path):
    try:
        with open(path) as f:
            bars = [json.loads(line) for line in f if line.strip()]
    except OSError:
        return []
    bars.sort(key=lambda b: b["t"])
    return bars


def hourly_log_returns(bars):
    """log returns of close c between bars exactly one hour apart, keyed by end time."""
    out = {}
    for i in range(1, len(bars)):
        if bars[i]["t"] - bars[i - 1]["t"] == HOUR_MS:
            out[bars[i]["t"]] = math.log(int(bars[i]["c"]) / int(bars[i - 1]["c"]))
    return out


def main():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json")))
    symbols = [u["symbol"] for u in task["universe"]]
    n = len(symbols)
    places = task["places"]
    edges = task["bin_edges"]
    n_bins = len(edges) + 1
    days = task["window"]["days"]

    bars = {u["symbol"]: load_bars(os.path.join(INPUT_DIR, u["file"]))
            for u in task["universe"]}
    rets = {s: hourly_log_returns(bars[s]) for s in symbols}

    # --- daily vols: realized hourly vol shrunk toward the per-name prior ---
    raw_vol = {}
    for s in symbols:
        rs = list(rets[s].values())
        if len(rs) >= 2:
            raw_vol[s] = max(MIN_VOL, math.sqrt(SESSION_HOURS * sum(r * r for r in rs) / len(rs)))
    med_raw = median(list(raw_vol.values()))
    vol = {}
    for s in symbols:
        name_prior = NAME_PRIOR.get(s, MEDIAN_PRIOR)
        prior = PRIOR_NAME_WEIGHT * name_prior + (1.0 - PRIOR_NAME_WEIGHT) * med_raw \
            if med_raw is not None else name_prior
        vol[s] = max(MIN_VOL, RAW_WEIGHT * raw_vol[s] + (1.0 - RAW_WEIGHT) * prior) \
            if s in raw_vol else max(MIN_VOL, prior)

    # --- correlation of hourly log returns, shrunk toward equicorrelation ---
    corr = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            common = sorted(set(rets[symbols[i]]) & set(rets[symbols[j]]))
            c = pearson([rets[symbols[i]][t] for t in common],
                        [rets[symbols[j]][t] for t in common]) if len(common) >= 3 else 0.0
            corr[i][j] = corr[j][i] = c
    off = [corr[i][j] for i in range(n) for j in range(n) if i != j]
    rho = sum(off) / len(off) if off else 0.0
    shrunk = [[(1.0 - CORR_SHRINK) * corr[i][j] + CORR_SHRINK * (1.0 if i == j else rho)
               for j in range(n)] for i in range(n)]
    for i in range(n):
        shrunk[i][i] += 1e-9
    L = cholesky(shrunk)
    if L is None:  # fall back to equicorrelation, then to independence
        equi = [[1.0 if i == j else rho for j in range(n)] for i in range(n)]
        for i in range(n):
            equi[i][i] += 1e-9
        L = cholesky(equi)
    if L is None:
        L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # --- day 1: the history already holds the first hours of its return ---
    # Previous session day's close, 16:00 New York (task.json window day 1's
    # "previous" date + "close" time, America/New_York).
    first = days[0]
    try:
        from zoneinfo import ZoneInfo
        ny = ZoneInfo(task.get("twap", {}).get("time_zone", "America/New_York"))
    except Exception:
        ny = timezone(timedelta(hours=-5))  # November is EST (UTC-5)
    hh, mm = (int(x) for x in first.get("close", "16:00").split(":"))
    prev_close = datetime.fromisoformat(first["previous"]).replace(
        hour=hh, minute=mm, tzinfo=ny).astimezone(timezone.utc)
    prev_close_ms = int(prev_close.timestamp() * 1000)

    partial = {}
    rem_frac = 1.0
    last_t = None
    for s in symbols:
        b = bars[s]
        if not b:
            partial[s] = 0.0
            continue
        last_t = b[-1]["t"] if last_t is None else max(last_t, b[-1]["t"])
        anchor = None
        for bar in b:
            if bar["t"] <= prev_close_ms:
                anchor = bar
            else:
                break
        if anchor is None:
            anchor = b[0]
        if b[-1]["t"] > prev_close_ms:
            partial[s] = math.log(int(b[-1]["c"]) / int(anchor["c"]))
        else:
            partial[s] = 0.0
    if last_t is not None and last_t > prev_close_ms:
        observed_h = (last_t - prev_close_ms) / HOUR_MS
        rem_frac = max(0.0, min(1.0, 1.0 - observed_h / SESSION_HOURS))

    # --- sizes: Student-t(DF) bin probabilities, in log space ---
    def sizes_for(mu, sd):
        rows = []
        scale = {s: sd[s] * math.sqrt((DF - 2) / DF) for s in symbols}
        log_edges = [math.log1p(e) for e in edges]
        for s in symbols:
            sc = scale[s]
            m = mu[s]
            for b in range(n_bins):
                lo = -math.inf if b == 0 else log_edges[b - 1]
                hi = math.inf if b == n_bins - 1 else log_edges[b]
                plo = t_cdf((lo - m) / sc, DF) if lo != -math.inf else 0.0
                phi = t_cdf((hi - m) / sc, DF) if hi != math.inf else 1.0
                u = math.floor(max(phi - plo, 0.0) * MICRO)
                if u > 0:
                    rows.append({"symbol": s, "bin": b, "p": u / MICRO})
        return rows

    mu_rest = {s: 0.0 for s in symbols}
    sd_rest = dict(vol)
    mu_day1 = dict(partial)
    sd_day1 = {s: vol[s] * math.sqrt(rem_frac) for s in symbols}
    sizes_rest = sizes_for(mu_rest, sd_rest)
    sizes_day1 = sizes_for(mu_day1, sd_day1) if rem_frac < 1.0 or any(partial.values()) else sizes_rest

    # --- order: Monte Carlo from the joint Student-t(DF) ---
    def order_for(mu, sd):
        rng = Rng(SEED)
        scale = [sd[symbols[i]] * math.sqrt((DF - 2) / DF) for i in range(n)]
        counts = [[0] * n for _ in range(places)]
        draw = [0.0] * n
        idx = list(range(n))
        for _ in range(DRAWS):
            z = [rng.gauss() for _ in range(n)]
            chi = sum(rng.gauss() ** 2 for _ in range(DF))
            w = math.sqrt(DF / chi) if chi > 0 else 1.0
            for i in range(n):
                draw[i] = mu[symbols[i]] + scale[i] * w * sum(L[i][k] * z[k] for k in range(i + 1))
            idx.sort(key=lambda a: (-draw[a], a))
            for k in range(places):
                counts[k][idx[k]] += 1
        units = [[math.floor((c + SMOOTHING) / (DRAWS + SMOOTHING * n) * MICRO)
                  for c in row] for row in counts]
        probs = [{"symbol": symbols[i], "place": k + 1, "p": u / MICRO}
                 for k, row in enumerate(units) for i, u in enumerate(row)]
        order = []
        for k in range(places):
            best = -1
            for i in range(n):
                if symbols[i] in order:
                    continue
                if best < 0 or units[k][i] > units[k][best] or (
                        units[k][i] == units[k][best] and symbols[i] < symbols[best]):
                    best = i
            order.append(symbols[best])
        return probs, order

    probs_rest, order_rest = order_for(mu_rest, sd_rest)
    if rem_frac < 1.0 or any(partial.values()):
        probs_day1, order_day1 = order_for(mu_day1, sd_day1)
    else:
        probs_day1, order_day1 = probs_rest, order_rest

    out_days = []
    for d in days:
        if d["date"] == first["date"]:
            out_days.append({"date": d["date"], "probabilities": probs_day1,
                             "order": order_day1, "sizes": sizes_day1})
        else:
            out_days.append({"date": d["date"], "probabilities": probs_rest,
                             "order": order_rest, "sizes": sizes_rest})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(json.dumps({"days": out_days}) + "\n")


if __name__ == "__main__":
    main()
