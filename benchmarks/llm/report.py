"""Aggregate benchmark result files into summary.json and summary.md.

Usage: python -m benchmarks.llm.report [--results /var/tmp/optunai/results]
           [--out /var/tmp/optunai]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def q(values: list[float | None], p: float) -> float | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(vals, p)) if vals else None


def curve_stats(runs: list[dict[str, Any]], key: str) -> dict[str, list[float | None]]:
    """Per-trial median and IQR of a monotone per-trial series across seeds."""
    n = max(len(r["trials"]) for r in runs)
    rows = []
    for r in runs:
        series: list[float | None] = []
        last: float | None = None
        for t in r["trials"]:
            v = t.get(key)
            if v is not None:
                last = v
            series.append(last)
        series += [last] * (n - len(series))
        rows.append(series)
    med, lo, hi = [], [], []
    for i in range(n):
        col = [row[i] for row in rows if row[i] is not None]
        med.append(q(col, 50))
        lo.append(q(col, 25))
        hi.append(q(col, 75))
    return {"median": med, "q25": lo, "q75": hi}


def arm_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    bests = [r["best_value"] for r in runs]
    usd = [r["usd"] for r in runs]
    calls = [c for r in runs for c in r["calls"] if c.get("purpose") in ("propose", "repair")]
    latencies = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
    cost_per_call = [c["usd"] for c in calls if c.get("usd") is not None]
    errors = sum(1 for c in calls if c.get("error"))
    n_attempted = sum(sum(t["source"] not in (None, "startup") for t in r["trials"]) for r in runs)
    n_fallback = sum(r["n_fallback"] for r in runs)
    n_violation = sum(r["n_violation_trials"] for r in runs)
    n_repair = sum(r["n_repair"] for r in runs)
    out = {
        "n_seeds": len(runs),
        "n_trials": runs[0]["n_trials"],
        "best_median": q(bests, 50),
        "best_q25": q(bests, 25),
        "best_q75": q(bests, 75),
        "usd_per_study_median": q(usd, 50),
        "usd_total": float(sum(usd)),
        "wall_s_median": q([r["wall_s"] for r in runs], 50),
        "n_llm_attempted": n_attempted,
        "fallback_rate": (n_fallback / n_attempted) if n_attempted else None,
        "violation_rate": (n_violation / n_attempted) if n_attempted else None,
        "repair_rate": (n_repair / n_attempted) if n_attempted else None,
        "call_errors": errors,
        "latency_s_median": q(latencies, 50),
        "latency_s_q75": q(latencies, 75),
        "usd_per_call_median": q(cost_per_call, 50),
        "curve": curve_stats(runs, "best"),
        "cum_usd": curve_stats(runs, "cum_usd"),
        "model": next((c["model"] for c in calls if c.get("model")), None),
    }
    if "heldout_mean" in runs[0]:
        out["heldout_median"] = q([r["heldout_mean"] for r in runs], 50)
        out["heldout_q25"] = q([r["heldout_mean"] for r in runs], 25)
        out["heldout_q75"] = q([r["heldout_mean"] for r in runs], 75)
        out["seen_minus_heldout_median"] = q(
            [r["best_value"] - r["heldout_mean"] for r in runs if r["best_value"] is not None], 50
        )
    if "total_steps" in runs[0]:
        out["total_steps_median"] = q([r["total_steps"] for r in runs], 50)
        out["n_pruned_median"] = q([r["n_pruned"] for r in runs], 50)
        out["false_prunes_total"] = int(sum(r["false_prunes"] for r in runs))
        out["n_pruned_total"] = int(sum(r["n_pruned"] for r in runs))
        out["false_prune_rate"] = (
            out["false_prunes_total"] / out["n_pruned_total"] if out["n_pruned_total"] else 0.0
        )
        out["judge_usd_total"] = float(sum(r.get("judge_usd", 0.0) for r in runs))
        out["judge_calls_total"] = int(sum(r.get("judge_calls", 0) for r in runs))
    new_params = []
    for r in runs:
        best_so_far: float | None = None
        sign = 1.0 if r["direction"] == "maximize" else -1.0
        for t in r["trials"]:
            if t.get("new_params"):
                improved = (
                    t["value"] is not None
                    and best_so_far is not None
                    and sign * t["value"] > sign * best_so_far
                )
                new_params.append(
                    {
                        "seed": r["seed"],
                        "trial": t["number"],
                        "new_params": t["new_params"],
                        "value": t["value"],
                        "best_before": best_so_far,
                        "improved_best": improved,
                        "hypothesis": t.get("hypothesis"),
                    }
                )
            if t["value"] is not None and t["state"] == "COMPLETE":
                best_so_far = (
                    t["value"]
                    if best_so_far is None or sign * t["value"] > sign * best_so_far
                    else best_so_far
                )
    if new_params:
        out["new_params_events"] = new_params
    return out


def build(results: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(results.glob("*/*.json")):
        r = json.loads(path.read_text())
        grouped[(r["bench"], r["problem"], r["arm"])].append(r)
    summary: dict[str, Any] = {}
    for (bench, problem, arm), runs in grouped.items():
        summary.setdefault(bench, {}).setdefault(problem, {})[arm] = arm_summary(runs)
    return summary


def fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def markdown(summary: dict[str, Any]) -> str:
    lines = []
    for bench, problems in summary.items():
        for problem, arms in problems.items():
            direction = "lower is better" if bench in ("synthetic",) else "higher is better"
            lines.append(f"\n### {bench} / {problem} ({direction})\n")
            if bench == "pruner":
                lines.append(
                    "| pruner | final best median [IQR] | steps median | pruned | false-prune rate | judge calls | judge USD |"
                )
                lines.append("|---|---|---|---|---|---|---|")
                for arm, s in arms.items():
                    lines.append(
                        f"| {arm} | {fmt(s['best_median'], 4)} [{fmt(s['best_q25'], 4)}, {fmt(s['best_q75'], 4)}] "
                        f"| {fmt(s.get('total_steps_median'), 0)} | {s.get('n_pruned_total', 0)} "
                        f"| {fmt(s.get('false_prune_rate'), 2)} | {s.get('judge_calls_total', 0)} "
                        f"| {fmt(s.get('judge_usd_total'), 4)} |"
                    )
                continue
            lines.append(
                "| arm | model | best median [IQR] | held-out | USD/study | USD/proposal | latency s | fallback | violations |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for arm, s in arms.items():
                held = (
                    f"{fmt(s['heldout_median'], 4)} [{fmt(s['heldout_q25'], 4)}, {fmt(s['heldout_q75'], 4)}]"
                    if "heldout_median" in s
                    else "-"
                )
                lines.append(
                    f"| {arm} | {s.get('model') or '-'} | {fmt(s['best_median'], 4)} "
                    f"[{fmt(s['best_q25'], 4)}, {fmt(s['best_q75'], 4)}] | {held} "
                    f"| {fmt(s['usd_per_study_median'], 3)} | {fmt(s['usd_per_call_median'], 4)} "
                    f"| {fmt(s['latency_s_median'], 1)} | {fmt(s['fallback_rate'], 2)} "
                    f"| {fmt(s['violation_rate'], 2)} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/var/tmp/optunai/results")
    ap.add_argument("--out", default="/var/tmp/optunai")
    args = ap.parse_args()
    summary = build(Path(args.results))
    out = Path(args.out)
    (out / "summary.json").write_text(json.dumps(summary, default=str))
    (out / "summary.md").write_text(markdown(summary))
    print(markdown(summary))


if __name__ == "__main__":
    main()
