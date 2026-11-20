[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cervus-snoopy/aspen-elm-finch-ibxd/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cervus-snoopy/aspen-elm-finch-ibxd/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecast, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of the players Exophase tracks who will have earned it at
2026-11-26T00:00:00Z — as five quantiles per trophy — from the house's
recorded reads of Exophase's trophy page up to the case checkpoint.

## The model: counts + dilution + saturating fits

Exophase does not refresh its counts continuously; it refreshes in batches.
On launch night the page sat unchanged (638 players tracked, every rate
identical) for hours, then refreshed to 3,868 players with every rate but
the 100% one lower. A refresh *dilutes* rates: the denominator jumps while
earner counts grow slowly. Predicting that rates simply stay put is why
persistence scores ~1.51 on the practice cases. This entry works in counts:

1. **Counts, not rates.** For each read and trophy, earners
   `E = round(rate * players_tracked / 100)`.
2. **Time** is `u = ln(1 + hours since launch)` (2026-11-19T00:00:00Z), so
   growth saturates instead of exploding over a 4.5-day horizon.
3. **Players tracked** `P(t)`: when P varies across reads, a least-squares
   line of `ln(P)` on `u`, read at the target and capped at 12x the last
   read. When the history is flat (no refresh yet), a dilution prior:
   `P_target = P_last * 2.0` for a 90-minute horizon, `1 + horizon_days *
   1.5` for longer horizons (capped at 12x).
4. **Earners** `E_i(t)`: when they vary, a least-squares line of
   `ln(max(E_i, 1))` on `u`, read at the target, capped at `P_target` and
   at 12x the last count, and floored at the last count (earners never
   un-earn a trophy). When they are flat, they stay flat — the rate then
   falls purely because P grows.
5. **Median rate** `= E_i(target) / P(target) * 100`. On the flat practice
   histories this halves the median (e.g. the platinum goes 0.63% →
   0.31%), which is the dilution the refresh actually produces.
6. **Spread** on the logit scale combines a random-walk term
   `0.5 * sqrt(horizon_days)` with the residual scatter of the earner
   fit; flat histories get a wide sigma of 1.0, because the size of the
   unseen refresh is uncertain and an honest wide band beats a narrow
   wrong one. Sigma is clamped to [0.15, 1.5].
7. **The 100% trophy** is never diluted: it stayed at 100% through the
   first refresh, so its median is held at 99.99 with quantiles near 100.

Quantiles are `inv_logit(logit(median) + z_tau * sigma)` for
`z = (-1.645, -0.674, 0, 0.674, 1.645)`, rounded to 4 decimals and sorted
so they are non-decreasing.

## How to run

```sh
MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every
`*.in` file in `$MAKEFUN_INPUT_DIR` (default `/input/`), in name order,
and writes one `/output/<id>.out` per case to `$MAKEFUN_OUTPUT_DIR`
(default `/output/`). It is deterministic — same input bytes, same
output bytes — and runs in well under a second per case.

## Input / output contract

- Input: UTF-8 JSON, schema `makefun/gta6-trophies-case/v1` — `id`, `at`,
  `horizon_minutes`, `target`, `levels`, `game`, `trophies` (in id order),
  `history` (every valid read before `at`, oldest first, each with
  `players_tracked` and per-trophy `rates`, null where a read lacked the
  trophy).
- Output: UTF-8 JSON, at most 65,536 bytes, exactly one key `trophies`:
  one row per trophy, each exactly `{"id": n, "q": [q05, q25, q50, q75,
  q95]}` with five finite numbers in [0, 100], non-decreasing.

## Local test results

- Practice `p00` / `p01`: both outputs valid (55 rows, ids 1–55 once each,
  5 finite non-decreasing quantiles each, 2,789 bytes), runtime ~0.05 s
  for both cases. Platinum median 0.63% → 0.3135% (dilution working);
  the 100% trophy forecasts [99.9482, 99.9804, 99.99, 99.9949, 99.9981].
- Against a proxy answer (earners frozen, players refreshing 638 →
  3,868 as the real first refresh did), pinball loss ≈ 0.59 vs ≈ 1.69
  for persistence on the same proxy (house persistence is 1.51 on the
  real practice answer); 90% coverage 0.98 vs 0.
- Synthetic rich-history case (357 reads over 2 days, players
  638 → 25,546, horizon 6,510 min): valid output, deterministic across
  two runs (byte-identical), ~0.07 s.

## License

MIT — see `LICENSE`.
