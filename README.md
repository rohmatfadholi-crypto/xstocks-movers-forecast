# xStocks movers forecast — make.fun entry

Forecasts the next 5 session days' moves of the 9 Nasdaq xStocks
(`xstocks-movers-forecast-01`): per day, the order of the names by return
(place 1 = biggest gain) as place probabilities, and each name's return in
42 bins.

`main.py` (Python 3.13, stdlib only) reads `$MAKEFUN_INPUT_DIR` (else
`/input`) and writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else `/output`).

Model, in short:

- Daily log returns are **Student-t with 5 degrees of freedom**, zero drift.
  Fat tails beat the normal on a year of Nasdaq daily closes.
- Daily vol = 30% trailing realized vol from the hourly pool bars +
  70% a prior built from each name's own typical daily vol (1y of Nasdaq
  closes) plus a small cross-name median hedge for the week's regime.
- **Day 1 is special**: the history already holds the first 2 hours of its
  return (it ends 2h after the previous day's close), so day 1's mean is
  shifted by that observed partial move and its vol is scaled by
  sqrt(remaining hours / 24).
- Order probabilities come from 20,000 Monte Carlo draws of a **joint**
  Student-t(5) with correlation from the hourly log returns, shrunk 50/50
  toward equicorrelation. No momentum tilt — trailing returns showed zero
  cross-sectional rank prediction in backtest.

Full model description and every backtest number: `MODEL.md` in the work
directory (not part of this repo). Practice score (1 scored day,
2026-11-03): **−7.211485** vs the house baseline's −7.527502
(sizes −5.823990 vs −6.162885; order −1.387495 vs −1.364616).
