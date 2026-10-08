"""Summarise the scaling-ladder benchmarks (rank-flip, few-shot ladder, noise-floor ablation)
into JSON + Markdown, and render the HTML artefact.

Usage: python -m benchmarks.llm.ladder_report --results /var/tmp/optunai-ladder/results \
           --out-dir benchmarks/llm/results/ladder --html /var/tmp/optunai-ladder/optunai-scaling-ladder-20261008.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from benchmarks.llm import artefact as _artefact
from benchmarks.llm.artefact import bar_chart
from benchmarks.llm.artefact import fmt
from benchmarks.llm.artefact import SLOTS
from benchmarks.llm.artefact import table
from optuna.ladder._decision import decide


ARM_LABEL = {
    "tpe": "TPE (40 full trials)",
    "llm": "LLMSampler alone (no threshold in prompt)",
    "llm-noise": "LLMSampler + register_noise (threshold in prompt)",
    "llm-ladder": "LLMSampler + Ladder (rungs, gate, threshold)",
    "full": "full budget, no pruning",
    "ladder": "Ladder (extrapolation)",
    "hb-2": "Hyperband η=2",
    "hb-3": "Hyperband η=3",
    "hb-4": "Hyperband η=4",
    "sha-2": "ASHA η=2",
    "sha-3": "ASHA η=3",
    "sha-4": "ASHA η=4",
    "median": "MedianPruner",
}
ARM_ORDER = ["tpe", "llm", "llm-noise", "llm-ladder"]
RANK_ORDER = ["full", "ladder", "hb-2", "hb-3", "hb-4", "median", "sha-2", "sha-3", "sha-4"]
SLOT = {
    "full": 3,
    "ladder": 0,
    "hb-2": 1,
    "hb-3": 1,
    "hb-4": 1,
    "median": 4,
    "sha-2": 2,
    "sha-3": 2,
    "sha-4": 2,
    "tpe": 0,
    "llm": 1,
    "llm-noise": 2,
    "llm-ladder": 7,
}


_artefact.ARM_SLOT.update(SLOT)


def css(arm: str) -> str:
    return f"var(--c{SLOT.get(arm, 7)})"


def count(v: float | None) -> str:
    return "–" if v is None else f"{v:.0f}"


def _cal(v: dict[str, Any]) -> str:
    cal = v.get("calibration") or []
    if not cal:
        return "–"
    return f"{sum(c['n_inside'] for c in cal)} / {sum(c['n_frozen'] for c in cal)}"


def med(xs: list[float | None]) -> float | None:
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else None


def iqr(xs: list[float | None]) -> str:
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "–"
    return f"[{fmt(float(np.percentile(v, 25)), 4)}, {fmt(float(np.percentile(v, 75)), 4)}]"


# ---- few-shot ladder --------------------------------------------------------------------------


def load_fewshot(results: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text())
        for p in sorted((results / "fewshot_ladder").glob("*.json"))
        if "summary" not in p.name
    ]


def ablation(rec: dict[str, Any], seed_sd: float, threshold: float) -> dict[str, Any]:
    """Walk the trial sequence: every new best is a 'win' without a threshold; classify each with
    the threshold rule (single runs on both sides, so the difference has SD √2·seed_SD)."""
    best: float | None = None
    wins: list[dict[str, Any]] = []
    se = math.sqrt(2.0) * seed_sd
    for t in rec["trials"]:
        if t["state"] != "COMPLETE" or t["value"] is None:
            continue
        v = float(t["value"])
        if best is None:
            best = v
            continue
        if v > best:
            diff = v - best
            d = decide(diff, diff - 2 * se, diff + 2 * se, threshold, can_run_more=False)
            wins.append(
                {
                    "trial": t["number"],
                    "difference": diff,
                    "decision": d.value,
                    "expected_effect": t.get("expected_effect"),
                }
            )
            best = v
    counts = {
        k: sum(w["decision"] == k for w in wins)
        for k in ("select", "insufficient_evidence", "defer")
    }
    sub = sum(w["difference"] < threshold for w in wins)
    return {
        "declared_wins": len(wins),
        "below_threshold": sub,
        "decisions": counts,
        "wins": wins,
        "seed_sd": seed_sd,
        "threshold": threshold,
    }


def fewshot_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_arm.setdefault(r["arm"], []).append(r)
    # Seed SD / threshold of the objective: pooled from every study that registered noise.
    sds = [r["noise"]["seed_sd"] for r in records if r.get("noise")]
    thresholds = [r["noise"]["threshold"] for r in records if r.get("noise")]
    seed_sd = float(np.median(sds)) if sds else 0.0
    threshold = float(np.median(thresholds)) if thresholds else 0.0
    out: dict[str, Any] = {"seed_sd": seed_sd, "threshold": threshold, "arms": {}, "studies": []}
    for arm in ARM_ORDER:
        rs = by_arm.get(arm)
        if not rs:
            continue
        acc = [r["acceptance"] for r in rs]
        abl = [ablation(r, seed_sd, threshold) for r in rs]
        out["arms"][arm] = {
            "n_seeds": len(rs),
            "best_of_n_median": med([r["best_value"] for r in rs]),
            "best_of_n_iqr": iqr([r["best_value"] for r in rs]),
            "reevaluated_median": med([a["reevaluated_value"] for a in acc]),
            "reevaluated_iqr": iqr([a["reevaluated_value"] for a in acc]),
            "selection_bias_gap_median": med([a["selection_bias_gap"] for a in acc]),
            "selection_bias_gaps": [a["selection_bias_gap"] for a in acc],
            "holdout_median": med([a["holdout_value"] for a in acc]),
            "holdout_iqr": iqr([a["holdout_value"] for a in acc]),
            "incumbent_holdout_median": med([a["incumbent_holdout_value"] for a in acc]),
            "holdout_difference_median": med([a["difference"] for a in acc]),
            "decisions": [a["decision"] for a in acc],
            "equivalent_compute_multipliers": [
                a.get("equivalent_compute_multiplier") for a in acc
            ],
            "near_optimal": [r["near_optimal"]["near_optimal"] for r in rs],
            "on_boundary": [r["near_optimal"]["on_boundary"] for r in rs],
            "max_abs_delta_median": med([r["near_optimal"]["max_abs_delta"] for r in rs]),
            "usd_median": med([r["usd"] for r in rs]),
            "usd_total": float(sum(r["usd"] for r in rs)),
            "n_trials_median": med([r["n_trials"] for r in rs]),
            "n_pruned_median": med([r["n_pruned"] for r in rs]),
            "n_deferred_total": int(sum(r.get("n_deferred", 0) for r in rs)),
            "compute_units_median": med([r["compute_units"] for r in rs]),
            "wall_s_median": med([r["wall_s"] for r in rs]),
            "status_counts": {
                k: int(sum(r["status_counts"].get(k, 0) for r in rs))
                for k in rs[0]["status_counts"]
            },
            "ablation": {
                "declared_wins": int(sum(a["declared_wins"] for a in abl)),
                "below_threshold": int(sum(a["below_threshold"] for a in abl)),
                "select": int(sum(a["decisions"]["select"] for a in abl)),
                "insufficient_evidence": int(
                    sum(a["decisions"]["insufficient_evidence"] for a in abl)
                ),
                "defer": int(sum(a["decisions"]["defer"] for a in abl)),
            },
            "calibration": [r.get("calibration") for r in rs if r.get("calibration")],
            "ladder_fit": [r.get("ladder_fit") for r in rs if r.get("ladder_fit")],
            "latency_s_median": med(
                [
                    float(np.median([c["latency_s"] for c in r.get("ledger_calls", [])]))
                    if r.get("ledger_calls")
                    else None
                    for r in rs
                ]
            ),
        }
        for r, a, b in zip(rs, acc, abl):
            out["studies"].append(
                {
                    "arm": arm,
                    "seed": r["seed"],
                    "best_of_n": r["best_value"],
                    "reevaluated": a["reevaluated_value"],
                    "gap": a["selection_bias_gap"],
                    "holdout": a["holdout_value"],
                    "incumbent_holdout": a["incumbent_holdout_value"],
                    "difference": a["difference"],
                    "interval": a["interval"],
                    "threshold": a["threshold"],
                    "decision": a["decision"],
                    "multiplier": a.get("equivalent_compute_multiplier"),
                    "near_optimal": r["near_optimal"]["near_optimal"],
                    "max_abs_delta": r["near_optimal"]["max_abs_delta"],
                    "on_boundary": r["near_optimal"]["on_boundary"],
                    "usd": r["usd"],
                    "n_trials": r["n_trials"],
                    "n_pruned": r["n_pruned"],
                    "n_deferred": r.get("n_deferred", 0),
                    "compute_units": r["compute_units"],
                    "wall_s": r["wall_s"],
                    "declared_wins": b["declared_wins"],
                    "wins_not_select": b["declared_wins"] - b["decisions"]["select"],
                    "new_params": sorted(
                        {k for t in r["trials"] for k in (t.get("new_params") or {})}
                    ),
                }
            )
    return out


# ---- markdown --------------------------------------------------------------------------------


def markdown(rank: dict[str, Any], few: dict[str, Any]) -> str:
    lines = ["# Scaling-ladder benchmarks", ""]
    lines += [
        "## Rank-flip (synthetic, 24 candidates per pool, rungs 1/2/4/8/16, noise SD 0.01)",
        "",
    ]
    n = next((v["n_seeds"] for k, v in rank.items() if isinstance(v, dict) and "n_seeds" in v), 0)
    lines.append(
        f"{n} seeds (pools). True best is at median rank {rank['pool']['true_best_rank_at_first_rung_median']:.0f} of 24 at the first rung; {rank['pool']['pair_flip_rate_median']:.0%} of candidate pairs flip order between the first rung and the target."
    )
    lines += [
        "",
        "| arm | P(select true best) | P(true best reaches target) | regret median [IQR] | compute (fraction of full) | candidates completed |",
        "|---|---|---|---|---|---|",
    ]
    for arm in RANK_ORDER:
        v = rank.get(arm)
        if not v:
            continue
        lines.append(
            f"| {ARM_LABEL[arm]} | {v['p_select_true_best']:.2f} | {v['p_true_best_reached_target']:.2f} | {fmt(v['regret_median'], 4)} {v['regret_iqr'] and '[' + ', '.join(fmt(x, 4) for x in v['regret_iqr']) + ']'} | {v['compute_median']:.0f} ({v['compute_fraction_of_full']:.2f}) | {v['n_completed_median']:.0f} |"
        )
    ec = rank.get("equal_compute_comparison")
    if ec:
        a = ec["nearest_rank_arm"]
        lines.append(
            f"\nEqual compute: Ladder {ec['ladder']['p_select_true_best']:.2f} at {ec['ladder']['compute_median']:.0f} vs {ARM_LABEL[a]} {ec[a]['p_select_true_best']:.2f} at {ec[a]['compute_median']:.0f}."
        )
    if rank.get("ladder", {}).get("interval_coverage") is not None:
        lines.append(
            f"Frozen 90% prediction intervals covered the final value in {rank['ladder']['interval_coverage']:.0%} of {rank['ladder']['n_frozen']} promotions."
        )
    lines += ["", "## Few-shot MLP head with the ladder (higher is better)", ""]
    lines.append(
        f"Seed SD of the incumbent (3 fresh seeds per study, pooled median): {few['seed_sd']:.4f}; frozen threshold = 2·SD = {few['threshold']:.4f}."
    )
    lines += [
        "",
        "| arm | seeds | best-of-N median [IQR] | fresh-seed re-evaluation | selection-bias gap | holdout (fresh seed) | incumbent holdout | decision per study | near-optimal | USD/study | trials | pruned | deferred | frozen intervals inside / n |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, v in few["arms"].items():
        lines.append(
            f"| {ARM_LABEL[arm]} | {v['n_seeds']} | {fmt(v['best_of_n_median'], 4)} {v['best_of_n_iqr']} | {fmt(v['reevaluated_median'], 4)} {v['reevaluated_iqr']} | {fmt(v['selection_bias_gap_median'], 4)} | {fmt(v['holdout_median'], 4)} {v['holdout_iqr']} | {fmt(v['incumbent_holdout_median'], 4)} | {', '.join(v['decisions'])} | {sum(v['near_optimal'])}/{len(v['near_optimal'])} | {fmt(v['usd_median'], 2)} | {count(v['n_trials_median'])} | {count(v['n_pruned_median'])} | {v['n_deferred_total']} | {_cal(v)} |"
        )
    lines += [
        "",
        "## Noise-floor ablation",
        "",
        "A 'declared win' is every new best-so-far in the study. The thresholded rule tests each against the frozen threshold with a √2·seed_SD interval.",
        "",
        "| arm | declared wins | below threshold | select | insufficient_evidence | defer |",
        "|---|---|---|---|---|---|",
    ]
    for arm, v in few["arms"].items():
        a = v["ablation"]
        lines.append(
            f"| {ARM_LABEL[arm]} | {a['declared_wins']} | {a['below_threshold']} | {a['select']} | {a['insufficient_evidence']} | {a['defer']} |"
        )
    return "\n".join(lines) + "\n"


# ---- html ------------------------------------------------------------------------------------


def mapping_rows(doc: Path) -> list[list[str]]:
    rows = []
    in_table = False
    for line in doc.read_text().splitlines():
        if line.startswith("| Post section |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 3:
                rows.append([f"<span>{inline_md(c)}</span>" for c in cells])
        elif in_table and not line.strip():
            break
    return rows


def inline_md_keep(text: str) -> str:
    """Code spans and emphasis on text that is already HTML-safe (from md_to_html)."""
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


def inline_md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def scatter(rank: dict[str, Any]) -> str:
    W, H = 560, 320
    L, R, T, B = 56, 16, 16, 44
    pw, ph = W - L - R, H - T - B
    pts = [
        (arm, rank[arm]["compute_fraction_of_full"], rank[arm]["p_select_true_best"])
        for arm in RANK_ORDER
        if arm in rank
    ]
    parts = [
        f"<svg viewBox='0 0 {W} {H}' class='chart' role='img' aria-label='P(select true best) against compute'>"
    ]
    for i in range(5):
        gy = T + i * ph / 4
        parts.append(
            f"<line x1='{L}' x2='{W - R}' y1='{gy:.1f}' y2='{gy:.1f}' class='grid'/><text x='{L - 6}' y='{gy + 4:.1f}' class='tick' text-anchor='end'>{1 - i * 0.25:.2f}</text>"
        )
    for i in range(5):
        gx = L + i * pw / 4
        parts.append(
            f"<text x='{gx:.1f}' y='{H - B + 16}' class='tick' text-anchor='middle'>{i * 0.25:.2f}</text>"
        )
    parts.append(f"<line x1='{L}' x2='{W - R}' y1='{H - B}' y2='{H - B}' class='axis'/>")
    parts.append(
        f"<text x='{L + pw / 2:.1f}' y='{H - 6}' class='label' text-anchor='middle'>compute spent, fraction of evaluating every candidate at full budget</text>"
    )
    parts.append(
        f"<text transform='translate(12 {T + ph / 2:.1f}) rotate(-90)' class='label' text-anchor='middle'>P(select the true target-scale best)</text>"
    )
    tags = {
        "full": "full",
        "ladder": "Ladder",
        "hb-2": "H2",
        "hb-3": "H3",
        "hb-4": "H4",
        "median": "M",
        "sha-2": "A2",
        "sha-3": "A3",
        "sha-4": "A4",
    }
    offsets = {
        "full": (-28, -12),
        "ladder": (0, -12),
        "hb-2": (0, -12),
        "hb-3": (22, 4),
        "hb-4": (22, 4),
        "median": (-22, 4),
        "sha-2": (-22, 4),
        "sha-3": (0, 20),
        "sha-4": (0, 20),
    }
    for arm, x, y in pts:
        cx, cy = L + x * pw, T + (1 - y) * ph
        parts.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='{css(arm)}' stroke='var(--surface)' stroke-width='2'>"
            f"<title>{html.escape(ARM_LABEL[arm])}: P={y:.2f}, compute {x:.2f}</title></circle>"
        )
        dx, dy = offsets.get(arm, (0, 18))
        parts.append(
            f"<text x='{cx + dx:.1f}' y='{cy + dy:.1f}' class='tick' text-anchor='middle'>{tags[arm]}</text>"
        )
    parts.append("</svg>")
    return f"<figure><figcaption>Rank-flip benchmark: selection accuracy against compute, median over seeds. H = Hyperband, A = ASHA, M = MedianPruner, with the reduction factor; colours as in the table.</figcaption>{''.join(parts)}</figure>"


def render_html(
    rank: dict[str, Any],
    few: dict[str, Any],
    mapping: list[list[str]],
    notes: dict[str, str],
    cost: dict[str, Any],
) -> str:
    sections = []
    sections.append(
        f"<section><h2>What was ported</h2><p>{notes['intro']}</p>{table(['Post section', 'What the post says', 'optunai mechanism'], mapping)}<p class='muted'>{notes['not_ported']}</p></section>"
    )
    # rank-flip
    n = next((v["n_seeds"] for k, v in rank.items() if isinstance(v, dict) and "n_seeds" in v), 0)
    rows = []
    for arm in RANK_ORDER:
        v = rank.get(arm)
        if not v:
            continue
        rows.append(
            [
                f"<span class='key'><i style='background:{css(arm)}'></i>{html.escape(ARM_LABEL[arm])}</span>",
                f"{v['p_select_true_best']:.2f}",
                f"{v['p_true_best_reached_target']:.2f}",
                f"{fmt(v['regret_median'], 4)} [{', '.join(fmt(x, 4) for x in v['regret_iqr'])}]",
                f"{v['compute_median']:.0f} ({v['compute_fraction_of_full']:.2f})",
                f"{v['n_completed_median']:.0f}",
            ]
        )
    ec = rank["equal_compute_comparison"]
    a = ec["nearest_rank_arm"]
    sections.append(
        f"<section><h2>Rank-flip benchmark: extrapolation against rank, {n} seeds</h2><p>{notes['rankflip']}</p>"
        f"<div class='grid'>{scatter(rank)}<div>{table(['arm', 'P(select true best)', 'P(true best reaches target)', 'true regret median [IQR]', 'compute median (fraction of full)', 'candidates completed'], rows)}"
        f"<p>Equal compute: Ladder <b>{ec['ladder']['p_select_true_best']:.2f}</b> at {ec['ladder']['compute_median']:.0f} units against {html.escape(ARM_LABEL[a])} <b>{ec[a]['p_select_true_best']:.2f}</b> at {ec[a]['compute_median']:.0f}. "
        f"Frozen 90 % prediction intervals covered the final value in {rank['ladder']['interval_coverage']:.0%} of {rank['ladder']['n_frozen']} promotions. "
        f"The true best sits at median rank {rank['pool']['true_best_rank_at_first_rung_median']:.0f} of 24 at the first rung; {rank['pool']['pair_flip_rate_median']:.0%} of candidate pairs flip order between the first rung and the target.</p></div></div></section>"
    )
    # few-shot
    rows = []
    for arm, v in few["arms"].items():
        rows.append(
            [
                f"<span class='key'><i style='background:{css(arm)}'></i>{html.escape(ARM_LABEL[arm])}</span>",
                v["n_seeds"],
                f"{fmt(v['best_of_n_median'], 4)} {v['best_of_n_iqr']}",
                f"{fmt(v['reevaluated_median'], 4)} {v['reevaluated_iqr']}",
                fmt(v["selection_bias_gap_median"], 4),
                f"{fmt(v['holdout_median'], 4)} {v['holdout_iqr']}",
                fmt(v["incumbent_holdout_median"], 4),
                fmt(v["holdout_difference_median"], 4),
                ", ".join(v["decisions"]),
                f"{sum(v['near_optimal'])}/{len(v['near_optimal'])}",
                fmt(v["usd_median"], 2),
                count(v["n_trials_median"]),
                count(v["n_pruned_median"]),
                v["n_deferred_total"],
                _cal(v),
            ]
        )
    study_rows = []
    for s in few["studies"]:
        study_rows.append(
            [
                html.escape(ARM_LABEL[s["arm"]]),
                s["seed"],
                fmt(s["best_of_n"], 4),
                fmt(s["reevaluated"], 4),
                fmt(s["gap"], 4),
                fmt(s["holdout"], 4),
                fmt(s["incumbent_holdout"], 4),
                f"{fmt(s['difference'], 4)} [{fmt(s['interval'][0], 4)}, {fmt(s['interval'][1], 4)}]",
                fmt(s["threshold"], 4),
                s["decision"],
                fmt(s["multiplier"], 2) if s["multiplier"] else "–",
                f"{'yes' if s['near_optimal'] else 'no'} (max |Δ| {fmt(s['max_abs_delta'], 4)})",
                ", ".join(s["on_boundary"]) or "none",
                ", ".join(s["new_params"]) or "–",
                s["n_trials"],
                s["n_pruned"],
                s["n_deferred"],
                fmt(s["compute_units"], 1),
                fmt(s["usd"], 2),
            ]
        )
    bars = (
        bar_chart(
            [
                (
                    ARM_LABEL[a].split(" (")[0].replace("LLMSampler", "LLM"),
                    v["selection_bias_gap_median"],
                    a,
                )
                for a, v in few["arms"].items()
            ],
            title="Selection-bias gap: best-of-N minus its fresh-seed re-evaluation, median over seeds (how much the best-of-N overstated)",
            y_label="gap",
            digits=4,
        )
        if few["arms"]
        else ""
    )
    sections.append(
        f"<section><h2>Few-shot MLP head with the ladder</h2><p>{notes['fewshot']}</p>"
        f"<p>Seed SD of the incumbent (3 fresh seeds per study, pooled median) <b>{few['seed_sd']:.4f}</b>; frozen threshold = 2·SD = <b>{few['threshold']:.4f}</b>, set before any holdout result was read.</p>"
        + table(
            [
                "arm",
                "seeds",
                "best-of-N median [IQR]",
                "fresh-seed re-evaluation",
                "selection-bias gap",
                "holdout (fresh seed)",
                "incumbent holdout",
                "holdout difference",
                "decision per study",
                "near-optimal",
                "USD/study",
                "trials",
                "pruned",
                "deferred",
                "frozen intervals: inside / n",
            ],
            rows,
        )
        + f"<div style='max-width:620px'>{bars}</div><h3>Per study</h3>"
        + table(
            [
                "arm",
                "seed",
                "best-of-N",
                "re-evaluated",
                "gap",
                "holdout",
                "incumbent holdout",
                "difference [interval]",
                "threshold",
                "decision",
                "×compute",
                "near-optimal",
                "on boundary",
                "new params used",
                "trials",
                "pruned",
                "deferred",
                "compute units",
                "USD",
            ],
            study_rows,
        )
        + "</section>"
    )
    # ablation
    rows = []
    for arm, v in few["arms"].items():
        ab = v["ablation"]
        rows.append(
            [
                f"<span class='key'><i style='background:{css(arm)}'></i>{html.escape(ARM_LABEL[arm])}</span>",
                ab["declared_wins"],
                ab["below_threshold"],
                ab["select"],
                ab["insufficient_evidence"],
                ab["defer"],
            ]
        )
    sections.append(
        f"<section><h2>Noise-floor ablation</h2><p>{notes['ablation']}</p>{table(['arm', 'declared wins (new best-so-far)', 'wins below the threshold', 'select', 'insufficient_evidence', 'defer'], rows)}</section>"
    )
    # cost
    rows = [
        [html.escape(k), fmt(v, 2) if isinstance(v, (int, float)) else html.escape(str(v))]
        for k, v in cost.items()
    ]
    sections.append(
        f"<section><h2>Cost and latency</h2>{table(['item', 'value'], rows)}<p>{notes['cost']}</p></section>"
    )
    sections.append(
        f"<section><h2>Verdict: which of the post's ideas changed decisions</h2>{notes['verdict']}</section>"
    )
    sections.append(f"<section><h2>Method</h2>{notes['method']}</section>")
    body = "".join(sections)
    c = {f"c{i}": SLOTS[i] for i in range(8)}
    light = " ".join(f"--{k}:{v[0]};" for k, v in c.items())
    dark = " ".join(f"--{k}:{v[1]};" for k, v in c.items())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>optunai scaling ladder</title>
<meta name="description" content="Scaling-ladder discipline inside an LLM-driven Optuna: rung extrapolation, pre-registered thresholds, acceptance, run statuses.">
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.1); {light} }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.1); {dark} }} }}
:root[data-theme="dark"] {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.1); {dark} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:20px; margin:36px 0 8px; }} h3 {{ font-size:16px; margin:20px 0 6px; }}
.sub {{ color:var(--ink2); margin:0 0 8px; }} section > p {{ max-width:900px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; margin:12px 0; }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px; }}
figcaption {{ font-size:13px; color:var(--ink2); margin-bottom:6px; }} .chart {{ width:100%; height:auto; display:block; }}
line.grid {{ stroke:var(--grid); stroke-width:1; }} line.axis {{ stroke:var(--axis); stroke-width:1; }}
.tick {{ fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }} .label {{ fill:var(--ink2); font-size:12px; }}
.key i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:8px 0 16px; font-variant-numeric:tabular-nums; display:block; overflow-x:auto; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); vertical-align:top; }} th {{ color:var(--ink2); font-weight:600; }}
.muted {{ color:var(--muted); }} code {{ background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:0 4px; font-size:13px; }}
ul {{ max-width:900px; }}
</style></head>
<body><main>
<h1>optunai: scaling-ladder discipline inside an LLM-driven Optuna</h1>
<p class="sub">{inline_md_keep(notes["subtitle"])}</p>
{body}
</main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/var/tmp/optunai-ladder/results")
    ap.add_argument("--out-dir", default="benchmarks/llm/results/ladder")
    ap.add_argument("--html", default=None)
    ap.add_argument(
        "--notes", default=None, help="markdown with '## key' sections for the artefact prose"
    )
    ap.add_argument("--mapping-doc", default="docs/scaling-ladder.md")
    args = ap.parse_args()
    results = Path(args.results)
    rank = json.loads((results / "rankflip" / "summary.json").read_text())
    few = fewshot_summary(load_fewshot(results))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"rankflip": rank, "fewshot_ladder": few}, indent=1, default=str)
    )
    (out_dir / "summary.md").write_text(markdown(rank, few))
    print(markdown(rank, few))
    if args.html:
        from benchmarks.llm.artefact import md_to_html

        notes = md_to_html(Path(args.notes).read_text()) if args.notes else {}
        for k in (
            "subtitle",
            "intro",
            "not_ported",
            "rankflip",
            "fewshot",
            "ablation",
            "cost",
            "verdict",
            "method",
        ):
            notes.setdefault(k, "")
        cost = {
            "LLM spend, few-shot ladder arms (USD, all studies)": sum(
                v["usd_total"] for v in few["arms"].values()
            ),
            **{
                f"USD per study, {ARM_LABEL[a]}": v["usd_median"]
                for a, v in few["arms"].items()
                if v["usd_median"]
            },
            "rank-flip benchmark": "no model calls",
        }
        Path(args.html).write_text(
            render_html(rank, few, mapping_rows(Path(args.mapping_doc)), notes, cost)
        )
        print("wrote", args.html)


if __name__ == "__main__":
    main()
