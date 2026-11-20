#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a target instant.

Model: counts + dilution + saturating fits (see README.md).

  * Work in earner COUNTS, not rates: E_i(t) = round(R_i(t) * P(t) / 100).
  * Time is u = ln(1 + hours since launch, 2026-11-19T00:00:00Z).
  * Players tracked P(t): least-squares line of ln(P) on u when P varies;
    a dilution prior (P grows by a factor G) when the history is flat.
  * Earners E_i(t): least-squares line of ln(max(E_i, 1)) on u when E_i
    varies; constant earners when the history is flat (the dilution then
    comes entirely from P growing, so rates fall by ~G).
  * Median rate = E_i(target) / P(target) * 100.
  * Spread on the logit scale combines a random-walk term that grows
    with the horizon and the residual scatter of the earner fit; flat
    histories get a wide sigma of 1.0 because the refresh size is unknown.

Standard library only. Deterministic: fixed iteration order, no
randomness, no wall-clock use. Reads every *.in file in
$MAKEFUN_INPUT_DIR (else /input/) and writes one /output/<id>.out per
case to $MAKEFUN_OUTPUT_DIR (else /output/).
"""

import json
import math
import os
from datetime import datetime, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Standard-normal quantiles for the five requested levels.
Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)

SIGMA_MIN = 0.15
SIGMA_MAX = 1.5
P_GROWTH_CAP = 12.0
EARNER_GROWTH_CAP = 12.0


def parse_time(text):
    """Parse an ISO-8601 timestamp (with a trailing Z) as aware UTC."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since_launch(dt):
    """Hours from launch (negative for pre-launch reads)."""
    return (dt - LAUNCH).total_seconds() / 3600.0


def u_of(dt):
    """Saturating time coordinate u = ln(1 + hours since launch)."""
    hours = hours_since_launch(dt)
    # ln(1 + h) needs 1 + h > 0; every real read is within hours of
    # launch, but guard anyway so a stray timestamp cannot crash us.
    return math.log(max(1.0 + hours, 1e-6))


def least_squares(xs, ys):
    """Return (intercept, slope) of the least-squares line y ~ a + b*x.

    Returns None when there are fewer than two distinct x values.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (mean_y - slope * mean_x, slope)


def residual_std(xs, ys, line):
    """Population std of the residuals of y around the fitted line."""
    if line is None or len(xs) < 3:
        return 0.0
    intercept, slope = line
    sq = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(sq / len(xs))


def logit(pct):
    """logit of a percent, clamped to [0.01, 99.99] first."""
    p = min(max(pct, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(eta):
    """Inverse logit, as a percent in [0.01, 100]."""
    if eta >= 0.0:
        p = 1.0 / (1.0 + math.exp(-eta))
    else:
        e = math.exp(eta)
        p = e / (1.0 + e)
    return min(max(p * 100.0, 0.01), 100.0)


def forecast_case(case):
    """Forecast one case dict; return the output dict for it."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = float(case.get("horizon_minutes") or 0)
    horizon_days = max(horizon_minutes, 0.0) / 1440.0

    target_dt = parse_time(case["target"]) if case.get("target") else None
    if target_dt is None and case.get("at"):
        target_dt = parse_time(case["at"])
    u_target = u_of(target_dt) if target_dt is not None else 0.0

    n_trophies = len(trophies)

    # ---- Collect the reads: (u, players, [rates]) oldest first. ----
    reads = []
    for entry in history:
        try:
            at = parse_time(entry["at"])
        except (KeyError, TypeError, ValueError):
            continue
        players = entry.get("players_tracked")
        rates = entry.get("rates") or []
        if players is None or players <= 0:
            continue
        reads.append((u_of(at), float(players), rates))

    if not reads:
        # No history at all: fall back to a near-zero rate everywhere.
        return {
            "trophies": [
                {"id": t.get("id"), "q": [0.01, 0.01, 0.01, 0.01, 0.01]}
                for t in trophies
            ]
        }

    p_last = reads[-1][1]
    # Last non-null rate per trophy (for the flat / fallback paths).
    last_rate = []
    for i in range(n_trophies):
        rate = None
        for _, _, rates in reversed(reads):
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        last_rate.append(rate if rate is not None else 0.01)

    # ---- Players tracked at the target. ----
    p_values = [p for _, p, _ in reads]
    if len(set(p_values)) > 1:
        line = least_squares([u for u, _, _ in reads],
                             [math.log(p) for _, p, _ in reads])
        if line is not None:
            p_target = math.exp(line[0] + line[1] * u_target)
        else:
            p_target = p_last
        # Never extrapolate wildly, and never forecast fewer players
        # than the last read saw (the tracked set only grows).
        p_target = min(max(p_target, p_last), p_last * P_GROWTH_CAP)
    else:
        # Flat history: Exophase has not refreshed yet. The next
        # refresh dilutes rates, so apply the dilution prior.
        if horizon_minutes <= 90.0:
            growth = 2.0
        else:
            growth = 1.0 + horizon_days * 1.5
        growth = min(growth, P_GROWTH_CAP)
        p_target = p_last * growth

    # ---- Per-trophy earner forecast. ----
    rows = []
    for i, trophy in enumerate(trophies):
        # Earner counts for the reads where this trophy has a rate.
        xs, ys = [], []
        for u, players, rates in reads:
            if i < len(rates) and rates[i] is not None:
                earners = int(round(float(rates[i]) * players / 100.0))
                xs.append(u)
                ys.append(math.log(max(earners, 1)))
        e_last = None
        if xs:
            e_last = math.exp(ys[-1])

        flat_earners = len(set(ys)) <= 1 if ys else True
        if last_rate[i] >= 99.99:
            # The 100% trophy: every tracked player holds it, and the
            # first refresh left it at 100% while every other rate fell
            # (new players earn this one immediately). Do NOT dilute it:
            # its median stays at 99.99 and its band sits near 100.
            median = 99.99
            sigma = 1.0 if flat_earners else min(max(
                math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2
                          + residual_std(xs, ys, least_squares(xs, ys)) ** 2),
                SIGMA_MIN), SIGMA_MAX)
        elif not xs:
            # Trophy never observed: near-zero median, wide band.
            median = 0.01
            sigma = 1.0
        elif flat_earners:
            # Earners do not jump while the page is unrefreshed; the
            # rate falls purely because P grows.
            median = (math.exp(ys[0]) / p_target) * 100.0 if ys else 0.01
            sigma = 1.0
        else:
            line = least_squares(xs, ys)
            if line is None:
                e_target = math.exp(ys[-1])
                resid = 0.0
            else:
                e_target = math.exp(line[0] + line[1] * u_target)
                resid = residual_std(xs, ys, line)
            # Earners cannot exceed the tracked players, and a fit
            # must not extrapolate wildly from the last observed count.
            # Earner counts are also monotone in reality (nobody
            # un-earns a trophy), so a fitted fall below the last
            # observed count is rounding noise: floor at E_last.
            if e_last is not None:
                e_target = min(e_target, e_last * EARNER_GROWTH_CAP)
                e_target = max(e_target, e_last)
            e_target = min(max(e_target, 0.0), p_target)
            median = (e_target / p_target) * 100.0 if p_target > 0 else 0.01
            rw = 0.5 * math.sqrt(horizon_days)
            sigma = math.sqrt(rw * rw + resid * resid)
            sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)

        median = min(max(median, 0.01), 99.99)
        centre = logit(median)
        quantiles = []
        for z in Z_VALUES:
            quantiles.append(
                round(inv_logit_pct(centre + z * sigma), 4)
            )
        # Safety net: quantiles must be non-decreasing.
        quantiles.sort()
        rows.append({"id": trophy.get("id"), "q": quantiles})

    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(output_dir, exist_ok=True)

    names = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
