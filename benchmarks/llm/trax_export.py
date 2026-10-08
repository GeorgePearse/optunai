"""Export the Experiments and Beliefs one study wrote to trackinizer, for the artefact.

Usage: python -m benchmarks.llm.trax_export --study "fewshot/mlp_head/llm:gemini/seed0" \
           --out /var/tmp/optunai/trax.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request


def get(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    experiments = []
    offset = 0
    while True:
        page = get(f"{args.url}/api/inquiries?kind=Experiment&limit=200&offset={offset}")
        if not page:
            break
        experiments += [e for e in page if e.get("title", "").startswith(args.study + " trial ")]
        offset += len(page)
        if len(page) < 200:
            break
    beliefs = {}
    for e in experiments:
        for edge in e.get("proves") or []:
            bid = edge["id"]
            if bid not in beliefs:
                b = get(f"{args.url}/api/inquiries/{bid}")
                beliefs[bid] = {
                    "id": bid,
                    "title": b.get("title"),
                    "description": b.get("description"),
                    "proved_by": [p.get("id") for p in b.get("proved_by") or []],
                }
    rows = []
    for e in sorted(experiments, key=lambda e: int(e["title"].rsplit(" ", 1)[-1])):
        edge = (e.get("proves") or [{}])[0]
        rows.append(
            {
                "trial": int(e["title"].rsplit(" ", 1)[-1]),
                "experiment_id": e["id"],
                "belief_id": edge.get("id"),
                "belief_title": beliefs.get(edge.get("id"), {}).get("title"),
                "valence": edge.get("valence"),
                "outcome": e.get("outcome"),
                "config": (e.get("config") or {}).get("params"),
                "status": e.get("status"),
            }
        )
    Path(args.out).write_text(
        json.dumps(
            {"study": args.study, "url": args.url, "experiments": rows, "beliefs": beliefs},
            indent=1,
        )
    )
    print(len(rows), "experiments,", len(beliefs), "beliefs ->", args.out)


if __name__ == "__main__":
    main()
