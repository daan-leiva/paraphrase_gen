#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

COLS = [
    "epoch",
    "train_loss",
    "eval_loss",
    "eval_bleu",
    "eval_rouge1",
    "eval_rouge2",
    "eval_rougeL",
    "eval_rougeLsum",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer_state", required=True, help="Path to trainer_state.json")
    ap.add_argument("--out_csv", default="epoch_metrics.csv", help="Output CSV path")
    args = ap.parse_args()

    with open(args.trainer_state, "r", encoding="utf-8") as f:
        state = json.load(f)

    rows = []
    last_train_loss = None

    # iterate in file order
    for rec in state.get("log_history", []):
        # update last seen train loss
        if "loss" in rec and "eval_loss" not in rec:
            last_train_loss = rec.get("loss")
            continue

        # on eval records one row per epoch
        if "eval_loss" in rec:
            rows.append(
                {
                    "epoch": rec.get("epoch"),
                    "train_loss": last_train_loss,
                    "eval_loss": rec.get("eval_loss"),
                    "eval_bleu": rec.get("eval_bleu"),
                    "eval_rouge1": rec.get("eval_rouge1"),
                    "eval_rouge2": rec.get("eval_rouge2"),
                    "eval_rougeL": rec.get("eval_rougeL"),
                    "eval_rougeLsum": rec.get("eval_rougeLsum"),
                }
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_path} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
