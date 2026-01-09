#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export selection_results.json to CSV with symbol, score, weight, rank.

Usage:
  python scripts/export_selection_to_csv.py \
      --state-file dashboard/state/selection_results.json \
      --out dashboard/state/top20_selection.csv
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Export selection JSON to CSV")
    p.add_argument("--state-file", default=str(Path("dashboard") / "state" / "selection_results.json"))
    p.add_argument("--out", default=str(Path("dashboard") / "state" / "top20_selection.csv"))
    args = p.parse_args()

    state_path = Path(args.state_file)
    out_path = Path(args.out)

    if not state_path.exists():
        raise SystemExit(f"State file not found: {state_path}")

    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    ts = data.get("timestamp", "")
    # Try both possible field names for stock lists
    stocks = data.get("selected_stocks", []) or data.get("stocks", [])

    for i, item in enumerate(stocks):
        rows.append({
            "timestamp": ts,
            "symbol": item.get("symbol"),
            "rank": item.get("rank", i + 1),  # Use index if rank not provided
            "score": item.get("score", 0.0) if item.get("score") is not None else 0.0,
            "weight": item.get("metrics", {}).get("weight", 1.0 / len(stocks)),  # Equal weight if not provided
            "action": item.get("action", "buy"),  # Default action
            "confidence": item.get("confidence", item.get("score", 0.5)),  # Use score as confidence if not provided
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "symbol", "rank", "score", "weight", "action", "confidence"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[export] Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

