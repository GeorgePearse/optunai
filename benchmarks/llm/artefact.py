"""Render the benchmark summary into one self-contained HTML page (inline SVG, CSS variables,
light and dark via prefers-color-scheme).

Usage: python -m benchmarks.llm.artefact --summary /var/tmp/optunai/summary.json \
           --notes /var/tmp/optunai/notes.md --out /var/tmp/optunai/optunai-llm-sampler-20261007.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


# Categorical slots from the validated reference palette (light, dark), fixed per arm.
SLOTS = [
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"),
    ("#008300", "#008300"),
    ("#4a3aa7", "#9085e9"),
    ("#e34948", "#e66767"),
]
ARM_SLOT = {
    "tpe": 0,
    "llm:gemini": 1,
    "llm-noctx:gemini": 2,
    "random": 3,
    "cmaes": 4,
    "llm:claude": 5,
    "llm:qwen": 6,
    "llm-new:gemini": 7,
    "llm-noctx:claude": 7,
    "llm-new:claude": 7,
    "tpe+none": 0,
    "tpe+median": 1,
    "tpe+llm": 2,
    "tpe+llm-aggressive": 7,
}
ARM_LABEL = {
    "tpe": "TPE",
    "random": "Random",
    "cmaes": "CMA-ES",
    "llm:claude": "LLM, context (Claude Sonnet 5.5)",
    "llm-noctx:claude": "LLM, no context (Claude Sonnet 5.5)",
    "llm:gemini": "LLM, context (Gemini 3.5 Flash)",
    "llm-noctx:gemini": "LLM, no context (Gemini 3.5 Flash)",
    "llm-new:gemini": "LLM, context + new params (Gemini 3.5 Flash)",
    "llm:qwen": "LLM, context (Qwen 3.7 Plus)",
    "llm-new:claude": "LLM, context + new params (Claude)",
    "tpe+none": "no pruning",
    "tpe+median": "MedianPruner",
    "tpe+llm": "LLMPruner (Jev, median floor)",
    "tpe+llm-aggressive": "LLMPruner aggressive",
}
W, H, PAD = 560, 300, dict(l=56, r=16, t=16, b=40)


def slot(arm: str) -> int:
    return ARM_SLOT.get(arm, 7)


def css_var(arm: str) -> str:
    return f"var(--c{slot(arm)})"


def fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, (int, float)):
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return html.escape(str(v))


def line_chart(
    series: list[dict[str, Any]],
    *,
    x_label: str,
    y_label: str,
    log_y: bool = False,
    title: str,
) -> str:
    """series: [{arm, x: [...], median: [...], q25: [...], q75: [...]}]"""
    xs_all = [x for s in series for x in s["x"]]
    ys_all = [
        y
        for s in series
        for key in ("median", "q25", "q75")
        for y in s[key]
        if y is not None and math.isfinite(y)
    ]
    if not xs_all or not ys_all:
        return f"<p class='muted'>{html.escape(title)}: no data yet</p>"
    floor = max(min(ys_all), 1e-3) if log_y else None
    tf = (lambda y: math.log10(max(y, floor))) if log_y else (lambda y: y)
    x0, x1 = min(xs_all), max(xs_all)
    y0, y1 = tf(min(ys_all)), tf(max(ys_all))
    if y1 == y0:
        y1 = y0 + 1
    if x1 == x0:
        x1 = x0 + 1
    pw, ph = W - PAD["l"] - PAD["r"], H - PAD["t"] - PAD["b"]

    def sx(x: float) -> float:
        return PAD["l"] + (x - x0) / (x1 - x0) * pw

    def sy(y: float) -> float:
        return PAD["t"] + (1 - (tf(y) - y0) / (y1 - y0)) * ph

    parts = [
        f"<svg viewBox='0 0 {W} {H}' class='chart' role='img' aria-label='{html.escape(title)}'>"
    ]
    # grid + axes
    for i in range(5):
        gy = PAD["t"] + i * ph / 4
        val = y1 - i * (y1 - y0) / 4
        label = f"{10**val:.3g}" if log_y else f"{val:.3g}"
        parts.append(
            f"<line x1='{PAD['l']}' x2='{W - PAD['r']}' y1='{gy:.1f}' y2='{gy:.1f}' class='grid'/>"
            f"<text x='{PAD['l'] - 6}' y='{gy + 4:.1f}' class='tick' text-anchor='end'>{label}</text>"
        )
    for i in range(5):
        gx = PAD["l"] + i * pw / 4
        val = x0 + i * (x1 - x0) / 4
        parts.append(
            f"<text x='{gx:.1f}' y='{H - PAD['b'] + 16}' class='tick' text-anchor='middle'>{val:.3g}</text>"
        )
    parts.append(
        f"<line x1='{PAD['l']}' x2='{W - PAD['r']}' y1='{H - PAD['b']}' y2='{H - PAD['b']}' class='axis'/>"
    )
    parts.append(
        f"<text x='{PAD['l'] + pw / 2:.1f}' y='{H - 6}' class='label' text-anchor='middle'>{html.escape(x_label)}</text>"
    )
    parts.append(
        f"<text transform='translate(12 {PAD['t'] + ph / 2:.1f}) rotate(-90)' class='label' text-anchor='middle'>{html.escape(y_label)}</text>"
    )
    for s in series:
        color = css_var(s["arm"])
        pts = [
            (sx(x), sy(m), sy(lo) if lo is not None else None, sy(hi) if hi is not None else None)
            for x, m, lo, hi in zip(s["x"], s["median"], s["q25"], s["q75"])
            if m is not None
        ]
        if not pts:
            continue
        band = [p for p in pts if p[2] is not None and p[3] is not None]
        if band:
            top = " ".join(f"{x:.1f},{hi:.1f}" for x, _, _, hi in band)
            bottom = " ".join(f"{x:.1f},{lo:.1f}" for x, _, lo, _ in reversed(band))
            parts.append(f"<polygon points='{top} {bottom}' fill='{color}' opacity='0.18'/>")
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts)
        parts.append(
            f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'>"
            f"<title>{html.escape(ARM_LABEL.get(s['arm'], s['arm']))}: median best-so-far, band = IQR over seeds</title></polyline>"
        )
        x_last, y_last, _, _ = pts[-1]
        parts.append(
            f"<circle cx='{x_last:.1f}' cy='{y_last:.1f}' r='4' fill='{color}' stroke='var(--surface)' stroke-width='2'/>"
        )
    parts.append("</svg>")
    legend = "".join(
        f"<span class='key'><i style='background:{css_var(s['arm'])}'></i>{html.escape(ARM_LABEL.get(s['arm'], s['arm']))}</span>"
        for s in series
    )
    return f"<figure><figcaption>{html.escape(title)}</figcaption>{''.join(parts)}<div class='legend'>{legend}</div></figure>"


def bar_chart(
    items: list[tuple[str, float | None, str]], *, title: str, y_label: str, digits: int = 2
) -> str:
    """items: [(label, value, arm)]"""
    vals = [v for _, v, _ in items if v is not None]
    if not vals:
        return f"<p class='muted'>{html.escape(title)}: no data yet</p>"
    vmax = max(vals) or 1.0
    pw, ph = W - PAD["l"] - PAD["r"], H - PAD["t"] - PAD["b"]
    n = len(items)
    bw = pw / n * 0.6
    parts = [
        f"<svg viewBox='0 0 {W} {H}' class='chart' role='img' aria-label='{html.escape(title)}'>"
    ]
    for i in range(5):
        gy = PAD["t"] + i * ph / 4
        val = vmax - i * vmax / 4
        parts.append(
            f"<line x1='{PAD['l']}' x2='{W - PAD['r']}' y1='{gy:.1f}' y2='{gy:.1f}' class='grid'/>"
            f"<text x='{PAD['l'] - 6}' y='{gy + 4:.1f}' class='tick' text-anchor='end'>{val:.3g}</text>"
        )
    for i, (label, v, arm) in enumerate(items):
        cx = PAD["l"] + (i + 0.5) * pw / n
        if v is not None:
            hgt = v / vmax * ph
            parts.append(
                f"<rect x='{cx - bw / 2:.1f}' y='{H - PAD['b'] - hgt:.1f}' width='{bw:.1f}' height='{hgt:.1f}' "
                f"rx='4' fill='{css_var(arm)}'><title>{html.escape(label)}: {v:.{digits}f}</title></rect>"
                f"<text x='{cx:.1f}' y='{H - PAD['b'] - hgt - 6:.1f}' class='tick' text-anchor='middle'>{v:.{digits}f}</text>"
            )
        parts.append(
            f"<text x='{cx:.1f}' y='{H - PAD['b'] + 16}' class='tick' text-anchor='middle'>{html.escape(label)}</text>"
        )
    parts.append(
        f"<line x1='{PAD['l']}' x2='{W - PAD['r']}' y1='{H - PAD['b']}' y2='{H - PAD['b']}' class='axis'/>"
        f"<text transform='translate(12 {PAD['t'] + ph / 2:.1f}) rotate(-90)' class='label' text-anchor='middle'>{html.escape(y_label)}</text>"
        "</svg>"
    )
    return f"<figure><figcaption>{html.escape(title)}</figcaption>{''.join(parts)}</figure>"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{c if isinstance(c, str) and c.startswith('<') else fmt(c)}</td>" for c in row
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def arm_rows(arms: dict[str, Any], *, heldout: bool = False) -> str:
    headers = (
        ["arm", "best (median)", "IQR"]
        + (["held-out (median)"] if heldout else [])
        + [
            "USD / study",
            "USD / proposal",
            "latency s",
            "fallback",
            "violations",
            "repairs",
        ]
    )
    rows = []
    for arm, s in arms.items():
        row: list[Any] = [
            f"<span class='key'><i style='background:{css_var(arm)}'></i>{html.escape(ARM_LABEL.get(arm, arm))}</span>",
            s["best_median"],
            f"[{fmt(s['best_q25'])}, {fmt(s['best_q75'])}]",
        ]
        if heldout:
            row.append(s.get("heldout_median"))
        row += [
            s["usd_per_study_median"],
            s["usd_per_call_median"],
            s["latency_s_median"],
            s["fallback_rate"],
            s["violation_rate"],
            s["repair_rate"],
        ]
        rows.append(row)
    return table(headers, rows)


def regret_series(
    arms: dict[str, Any], key: str = "curve", x_key: str | None = None
) -> list[dict[str, Any]]:
    out = []
    for arm, s in arms.items():
        c = s[key]
        n = len(c["median"])
        if x_key == "usd":
            if not any(v for v in s["cum_usd"]["median"] if v):
                continue
            x = [v or 0.0 for v in s["cum_usd"]["median"]]
        else:
            x = list(range(1, n + 1))
        out.append({"arm": arm, "x": x, "median": c["median"], "q25": c["q25"], "q75": c["q75"]})
    return out


def render(summary: dict[str, Any], notes: dict[str, str]) -> str:
    sections = []
    sections.append(f"<section><h2>Design</h2>{notes.get('design', '')}</section>")
    # synthetic
    syn = summary.get("synthetic", {})
    if syn:
        figs = "".join(
            line_chart(
                regret_series(arms),
                x_label="trial",
                y_label="best value so far (log)",
                log_y=True,
                title=f"{problem}: best-so-far vs trials, median and IQR over 5 seeds",
            )
            for problem, arms in syn.items()
        )
        tables = "".join(f"<h3>{html.escape(p)}</h3>{arm_rows(a)}" for p, a in syn.items())
        sections.append(
            f"<section><h2>Synthetic functions (minimise, 30 trials, 5 seeds)</h2>{notes.get('synthetic', '')}"
            f"<div class='grid'>{figs}</div>{tables}</section>"
        )
    skl = summary.get("sklearn", {})
    if skl:
        figs = "".join(
            line_chart(
                regret_series(arms),
                x_label="trial",
                y_label="best macro F1 so far",
                title=f"{problem}: best-so-far vs trials, median and IQR over 5 seeds",
            )
            + line_chart(
                regret_series(arms, x_key="usd"),
                x_label="cumulative USD",
                y_label="best macro F1 so far",
                title=f"{problem}: best-so-far vs USD (LLM arms)",
            )
            for problem, arms in skl.items()
        )
        tables = "".join(f"<h3>{html.escape(p)}</h3>{arm_rows(a)}" for p, a in skl.items())
        sections.append(
            f"<section><h2>Gradient boosting on sklearn datasets (maximise macro F1, 30 trials, 5 seeds)</h2>"
            f"{notes.get('sklearn', '')}<div class='grid'>{figs}</div>{tables}</section>"
        )
    few = summary.get("fewshot", {})
    if few:
        arms = few["mlp_head"]
        figs = line_chart(
            regret_series(arms),
            x_label="trial",
            y_label="mean macro AP, seen cases",
            title="few-shot MLP head: best-so-far vs trials, median and IQR over seeds",
        ) + line_chart(
            regret_series(arms, x_key="usd"),
            x_label="cumulative USD",
            y_label="mean macro AP, seen cases",
            title="few-shot MLP head: best-so-far vs USD (LLM arms)",
        )
        held = bar_chart(
            [
                (ARM_LABEL.get(a, a).split(" (")[0], s.get("heldout_median"), a)
                for a, s in arms.items()
            ],
            title="held-out cases (world seed 2): score of each arm's final pick, median over seeds",
            y_label="mean macro AP, held-out",
            digits=3,
        )
        events = []
        for a, s in arms.items():
            for e in s.get("new_params_events", []):
                events.append(
                    [
                        html.escape(ARM_LABEL.get(a, a)),
                        e["seed"],
                        e["trial"],
                        html.escape(
                            json.dumps({k: v["value"] for k, v in e["new_params"].items()})
                        ),
                        html.escape(next(iter(e["new_params"].values()))["rationale"][:220]),
                        e["value"],
                        e["best_before"],
                        "yes" if e["improved_best"] else "no",
                    ]
                )
        ev_table = (
            table(
                [
                    "arm",
                    "seed",
                    "trial",
                    "new parameter",
                    "rationale",
                    "value",
                    "best before",
                    "new best?",
                ],
                events,
            )
            if events
            else "<p class='muted'>no new parameters were proposed</p>"
        )
        sections.append(
            f"<section><h2>Few-shot MLP head over cached embeddings (in-house target)</h2>{notes.get('fewshot', '')}"
            f"<div class='grid'>{figs}{held}</div>{arm_rows(arms, heldout=True)}"
            f"<h3>Open-vocabulary proposals</h3>{ev_table}</section>"
        )
    pr = summary.get("pruner", {})
    if pr:
        figs = ""
        rows = []
        for problem, arms in pr.items():
            figs += bar_chart(
                [
                    (ARM_LABEL.get(a, a).split(" (")[0], s.get("total_steps_median"), a)
                    for a, s in arms.items()
                ],
                title=f"{problem}: boosting steps spent per study (median)",
                y_label="steps",
                digits=0,
            )
            figs += bar_chart(
                [
                    (ARM_LABEL.get(a, a).split(" (")[0], s.get("best_median"), a)
                    for a, s in arms.items()
                ],
                title=f"{problem}: final best macro F1 (median)",
                y_label="macro F1",
                digits=4,
            )
            for a, s in arms.items():
                rows.append(
                    [
                        html.escape(problem),
                        f"<span class='key'><i style='background:{css_var(a)}'></i>{html.escape(ARM_LABEL.get(a, a))}</span>",
                        s["best_median"],
                        f"[{fmt(s['best_q25'])}, {fmt(s['best_q75'])}]",
                        s.get("total_steps_median"),
                        s.get("n_pruned_total"),
                        s.get("false_prune_rate"),
                        s.get("judge_calls_total"),
                        s.get("judge_usd_total"),
                    ]
                )
        sections.append(
            f"<section><h2>Pruners (TPE sampler, 30 trials, 5 seeds)</h2>{notes.get('pruner', '')}"
            f"<div class='grid'>{figs}</div>"
            + table(
                [
                    "dataset",
                    "pruner",
                    "final best",
                    "IQR",
                    "steps",
                    "pruned",
                    "false-prune rate",
                    "judge calls",
                    "judge USD",
                ],
                rows,
            )
            + "</section>"
        )
    sections.append(f"<section><h2>Verdict</h2>{notes.get('verdict', '')}</section>")
    sections.append(f"<section><h2>Method and cost</h2>{notes.get('method', '')}</section>")
    body = "".join(sections)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>optunai LLM sampler vs TPE</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.1);
  --c0:{SLOTS[0][0]}; --c1:{SLOTS[1][0]}; --c2:{SLOTS[2][0]}; --c3:{SLOTS[3][0]}; --c4:{SLOTS[4][0]}; --c5:{SLOTS[5][0]}; --c6:{SLOTS[6][0]}; --c7:{SLOTS[7][0]}; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.1);
  --c0:{SLOTS[0][1]}; --c1:{SLOTS[1][1]}; --c2:{SLOTS[2][1]}; --c3:{SLOTS[3][1]}; --c4:{SLOTS[4][1]}; --c5:{SLOTS[5][1]}; --c6:{SLOTS[6][1]}; --c7:{SLOTS[7][1]}; }} }}
:root[data-theme="dark"] {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.1);
  --c0:{SLOTS[0][1]}; --c1:{SLOTS[1][1]}; --c2:{SLOTS[2][1]}; --c3:{SLOTS[3][1]}; --c4:{SLOTS[4][1]}; --c5:{SLOTS[5][1]}; --c6:{SLOTS[6][1]}; --c7:{SLOTS[7][1]}; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:20px; margin:36px 0 8px; }} h3 {{ font-size:16px; margin:20px 0 6px; }}
.sub {{ color:var(--ink2); margin:0 0 8px; }}
section > p, .notes p {{ max-width:860px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; margin:12px 0; }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px; }}
figcaption {{ font-size:13px; color:var(--ink2); margin-bottom:6px; }}
.chart {{ width:100%; height:auto; display:block; }}
.grid line.grid, line.grid {{ stroke:var(--grid); stroke-width:1; }} line.axis {{ stroke:var(--axis); stroke-width:1; }}
.tick {{ fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }} .label {{ fill:var(--ink2); font-size:12px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:8px; font-size:12px; color:var(--ink2); }}
.key i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:8px 0 16px; font-variant-numeric:tabular-nums; display:block; overflow-x:auto; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); vertical-align:top; }} th {{ color:var(--ink2); font-weight:600; }}
.muted {{ color:var(--muted); }} code {{ background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:0 4px; font-size:13px; }}
pre {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px; overflow-x:auto; font-size:13px; }}
</style></head>
<body><main>
<h1>optunai: an LLM sampler and pruner inside a forked Optuna, vs TPE</h1>
<p class="sub">{html.escape(notes.get("subtitle", ""))}</p>
{body}
</main></body></html>"""


def md_to_html(text: str) -> dict[str, str]:
    """Very small markdown: '## key' headings split sections; paragraphs, bullet lists, code spans."""
    import re

    out: dict[str, str] = {}
    key = "subtitle"
    buf: list[str] = []

    def flush() -> None:
        chunks = "\n".join(buf).strip().split("\n\n")
        parts = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            if chunk.lstrip().startswith("- "):
                items = "".join(
                    f"<li>{inline(line.strip()[2:])}</li>"
                    for line in chunk.splitlines()
                    if line.strip()
                )
                parts.append(f"<ul>{items}</ul>")
            elif chunk.startswith("```"):
                parts.append(f"<pre>{html.escape(chunk.strip('`').strip())}</pre>")
            else:
                parts.append(f"<p>{inline(' '.join(chunk.splitlines()))}</p>")
        out[key] = (
            "".join(parts)
            if key != "subtitle"
            else html.unescape(re.sub("<[^>]+>", "", " ".join(chunks)))
        )
        buf.clear()

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", s)
        return s

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            key = line[3:].strip()
        else:
            buf.append(line)
    flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="/var/tmp/optunai/summary.json")
    ap.add_argument("--notes", default="/var/tmp/optunai/notes.md")
    ap.add_argument("--out", default="/var/tmp/optunai/optunai-llm-sampler-20261007.html")
    args = ap.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    notes = md_to_html(Path(args.notes).read_text()) if Path(args.notes).exists() else {}
    Path(args.out).write_text(render(summary, notes))
    print("wrote", args.out, Path(args.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
